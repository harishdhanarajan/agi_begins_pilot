"""DreamerV3-architecture agent: orchestration layer.

The neural pieces live in ``model.py``, ``world_model.py``, and
``actor_critic.py``. This file is the glue that:

  * keeps a running posterior state ``(h, z)`` across an episode so the agent
    can ``act`` one step at a time;
  * pushes finished episodes into the sequence replay;
  * runs one gradient step on the world model and one on the actor-critic
    per ``train_step`` call;
  * exposes ``discover(env)`` and ``explain(report)`` so the CLI in
    ``main.py`` stays one-line on top.

Everything is env-agnostic: the only thing the agent reads from ``env`` is
``reset()``, ``step(action)``, and ``num_actions``. Image shape is inferred
from the first observation; nothing is hardcoded.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from .actor_critic import ActorCritic
from .nets import frames_to_tensor, one_hot
from .probes import (
    probe_action_geometry,
    probe_convergence,
    probe_manifold_dimension,
    probe_state_count,
    probe_termination,
)
from .replay import SequenceReplay
from .world_model import WorldModel, WorldModelConfig


# ---- viz (observer only; does not affect agent behavior) -------------------

def _detect_tile_size(frame: np.ndarray) -> int:
    row = frame[frame.shape[0] // 2]
    min_run = frame.shape[1]
    run = 1
    for c in range(1, len(row)):
        if (row[c] == row[c - 1]).all():
            run += 1
        else:
            if run < min_run:
                min_run = run
            run = 1
    return max(1, min(min_run, frame.shape[1]))


def _frame_to_ascii(frame: np.ndarray) -> str:
    tile = _detect_tile_size(frame)
    h, w, _ = frame.shape
    unique = sorted({tuple(c.tolist()) for c in frame.reshape(-1, 3)}, key=sum)
    palette = " .oO@#*+"
    color_char = {c: palette[min(i, len(palette) - 1)] for i, c in enumerate(unique)}
    lines = []
    for r in range(tile // 2, h, tile):
        row_chars = "".join(color_char[tuple(frame[r, c].tolist())] for c in range(tile // 2, w, tile))
        lines.append(row_chars)
    return "\n".join(lines)


def _render_step(frame: np.ndarray, header: str) -> None:
    print("\033[2J\033[H", end="")
    print(header)
    print(_frame_to_ascii(frame))


# ---- env interface ---------------------------------------------------------

class AgentEnv(Protocol):
    @property
    def num_actions(self) -> int: ...
    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> tuple[np.ndarray, bool, float]: ...


# ---- agent -----------------------------------------------------------------

class DreamerAgent:
    """Drives the world model and actor-critic from outside.

    Holds three optimizers — one for the world model parameters, one for the
    actor, one for the critic. Each ``train_step`` does:
      1. world-model loss on a sampled batch; gradient step.
      2. imagine ``horizon`` steps from the posterior states the world model
         just produced; actor-critic loss on the imagined trajectory; gradient
         steps.
      3. EMA update of the slow target critic.
    """

    def __init__(
        self,
        num_actions: int,
        image_shape: tuple[int, int, int],
        seq_len: int = 16,
        imag_horizon: int = 8,
        batch_size: int = 16,
        wm_lr: float = 1e-4,
        actor_lr: float = 3e-5,
        critic_lr: float = 3e-5,
        clip_grad: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.image_shape = image_shape
        self.seq_len = seq_len
        self.imag_horizon = imag_horizon
        self.batch_size = batch_size
        self.clip_grad = clip_grad

        cfg = WorldModelConfig(num_actions=num_actions, image_shape=image_shape)
        self.cfg = cfg
        self.world_model = WorldModel(cfg).to(self.device)
        self.world_model.twohot.to(self.device)

        stoch_flat = cfg.stoch_groups * cfg.stoch_classes
        self.ac = ActorCritic(
            num_actions=num_actions,
            deter_dim=cfg.deter_dim,
            stoch_flat=stoch_flat,
            twohot=self.world_model.twohot,
            hidden=cfg.hidden,
        ).to(self.device)

        self.wm_opt = torch.optim.AdamW(self.world_model.parameters(), lr=wm_lr, eps=1e-8, weight_decay=0.0)
        self.actor_opt = torch.optim.AdamW(self.ac.actor.parameters(), lr=actor_lr, eps=1e-8, weight_decay=0.0)
        self.critic_opt = torch.optim.AdamW(self.ac.critic.parameters(), lr=critic_lr, eps=1e-8, weight_decay=0.0)

        # running state used by ``act`` to maintain (h, z) across an episode
        self._h: torch.Tensor | None = None
        self._z: torch.Tensor | None = None
        self._prev_action_onehot: torch.Tensor | None = None

        # diagnostics accumulators (for the report; do not influence behavior)
        self._wm_loss_history: list[float] = []
        self._actor_info_last: dict[str, float] = {}
        self._critic_info_last: dict[str, float] = {}
        self._imagined_returns: list[float] = []
        self._train_steps_taken = 0

    # ---- per-episode acting ----

    def reset_state(self) -> None:
        h, z = self.world_model.rssm.initial(1, self.device)
        self._h = h
        self._z = z
        self._prev_action_onehot = torch.zeros(1, self.num_actions, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, rng: np.random.Generator, greedy: bool = False) -> int:
        """One-step posterior + action sample.

        Maintains ``self._h, self._z`` across calls within an episode. Always
        call ``reset_state()`` at the start of each episode.
        """
        assert self._h is not None and self._z is not None and self._prev_action_onehot is not None
        x = frames_to_tensor([obs]).to(self.device)
        x_emb = self.world_model.encoder(x)
        h, _ = self.world_model.rssm.img_step(self._h, self._z, self._prev_action_onehot)
        _, z = self.world_model.rssm.obs_step(h, x_emb)
        self._h = h
        self._z = z

        z_flat = z.reshape(1, -1)
        actor_logits = self.ac.actor(h, z_flat)
        if greedy:
            a_idx = int(actor_logits.argmax(dim=-1).item())
        else:
            probs = F.softmax(actor_logits, dim=-1)
            # numpy categorical sample — keeps RNG explicit + reproducible
            p = probs.detach().cpu().numpy().reshape(-1)
            p = p / p.sum()
            a_idx = int(rng.choice(self.num_actions, p=p))
        self._prev_action_onehot = one_hot([a_idx], self.num_actions).to(self.device)
        return a_idx

    # ---- training ----

    def train_step(self, replay: SequenceReplay, rng: np.random.Generator) -> dict[str, float]:
        if not replay.ready(self.batch_size):
            return {}

        np_batch = replay.sample(self.batch_size, self.seq_len, rng)
        batch = {
            # frames_to_tensor handles leading (B, T) dims and moves the
            # channel axis from -1 to -3, so we get (B, T, 3, H, W) directly.
            "obs": frames_to_tensor(np_batch["obs"]).to(self.device),
            "action": torch.from_numpy(np_batch["action"]).to(self.device),
            "reward": torch.from_numpy(np_batch["reward"]).to(self.device),
            "cont": torch.from_numpy(np_batch["cont"]).to(self.device),
            "first": torch.from_numpy(np_batch["first"]).to(self.device),
        }

        # ---- world model update ----
        wm_loss, wm_info, states = self.world_model.loss(batch)
        self.wm_opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.clip_grad)
        self.wm_opt.step()
        self._wm_loss_history.append(wm_info["wm/total"])

        # ---- imagine + actor-critic update ----
        h = states["h"].detach()
        z = states["z"].detach()
        B, T = h.shape[:2]
        start_h = h.reshape(B * T, -1)
        start_z = z.reshape(B * T, *z.shape[2:])

        traj = self.world_model.imagine(start_h, start_z, self.ac.actor, self.imag_horizon)

        actor_loss, actor_info, returns = self.ac.actor_loss(traj)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), self.clip_grad)
        self.actor_opt.step()

        critic_loss, critic_info = self.ac.critic_loss(traj, returns)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), self.clip_grad)
        self.critic_opt.step()

        self.ac.update_target()

        self._actor_info_last = actor_info
        self._critic_info_last = critic_info
        self._imagined_returns.append(float(returns.mean().item()))
        self._train_steps_taken += 1

        out: dict[str, float] = {}
        out.update(wm_info)
        out.update(actor_info)
        out.update(critic_info)
        return out


# ---- top-level loop --------------------------------------------------------

def collect_episode(
    env: AgentEnv,
    agent: DreamerAgent,
    rng: np.random.Generator,
    replay: SequenceReplay,
    watch: bool = False,
    watch_label: str = "",
    greedy: bool = False,
) -> tuple[list[tuple[np.ndarray, int, float, bool]], list[tuple[float, bool]]]:
    """Run one episode and push it into the replay.

    Returns the raw episode tuples and a (reward, done) trace used by the
    termination probe.
    """
    obs = env.reset()
    agent.reset_state()
    episode: list[tuple[np.ndarray, int, float, bool]] = []
    outcome_trace: list[tuple[float, bool]] = []
    step_idx = 0
    if watch:
        _render_step(obs, f"{watch_label}  step 0  (reset)")
        time.sleep(0.06)
    while True:
        a = agent.act(obs, rng, greedy=greedy)
        next_obs, done, reward = env.step(a)
        episode.append((obs, a, float(reward), bool(done)))
        outcome_trace.append((float(reward), bool(done)))
        step_idx += 1
        if watch:
            tag = " WIN" if reward > 0 else (" timeout" if done else "")
            _render_step(next_obs, f"{watch_label}  step {step_idx}  action={a}  reward={reward}{tag}")
            time.sleep(0.06)
        obs = next_obs
        if done:
            break
    replay.add(episode)
    return episode, outcome_trace


def discover(
    env: AgentEnv,
    num_episodes: int = 300,
    train_steps: int = 5000,
    batch_size: int = 16,
    seq_len: int = 16,
    imag_horizon: int = 8,
    train_ratio: int = 1,
    seed: int = 0,
    watch: bool = False,
    log_every: int = 0,
) -> dict:
    """Closed-loop discovery: collect an episode, train a few steps, repeat.

    Training cap is the smaller of (train_steps, num_episodes * train_ratio *
    avg_episode_length). The exact knob is ``train_ratio`` — train_ratio
    gradient steps per env step is the DreamerV3 setting.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # peek a frame to learn image shape
    sample_obs = env.reset()
    image_shape = (sample_obs.shape[2], sample_obs.shape[0], sample_obs.shape[1])  # (C, H, W)
    env.reset()  # rewind; env.reset is idempotent

    agent = DreamerAgent(
        num_actions=env.num_actions,
        image_shape=image_shape,
        seq_len=seq_len,
        imag_horizon=imag_horizon,
        batch_size=batch_size,
    )
    replay = SequenceReplay(capacity=max(2 * num_episodes, 200))

    per_episode_outcomes: list[list[tuple[float, bool]]] = []
    train_steps_done = 0

    chunk = max(1, num_episodes // 10)
    for ep in range(num_episodes):
        _, trace = collect_episode(
            env, agent, rng, replay,
            watch=watch,
            watch_label=f"ep {ep + 1}/{num_episodes}",
        )
        per_episode_outcomes.append(trace)

        # training: aim for train_ratio gradient steps per env step in this episode
        env_steps_this_ep = len(trace)
        steps_target = env_steps_this_ep * train_ratio
        steps_remaining_in_budget = max(0, train_steps - train_steps_done)
        steps_this_ep = min(steps_target, steps_remaining_in_budget)
        for _ in range(steps_this_ep):
            info = agent.train_step(replay, rng)
            train_steps_done += 1
            if info and log_every and train_steps_done % log_every == 0:
                bits = ", ".join(f"{k}={v:.4f}" for k, v in info.items())
                print(f"  step {train_steps_done:>5}  {bits}")

        if (ep + 1) % chunk == 0 or ep == num_episodes - 1:
            success_so_far = sum(1 for t in per_episode_outcomes if any(r > 0 for r, _ in t))
            print(
                f"[learner] episodes {ep + 1}/{num_episodes}  "
                f"successes={success_so_far}  train_steps={train_steps_done}  "
                f"wm_loss={agent._wm_loss_history[-1] if agent._wm_loss_history else float('nan'):.4f}",
                flush=True,
            )

    # ---- post-training diagnostics ----
    transitions = replay.iter_transitions()
    frames = replay.iter_frames()

    # dedup frames by raw pixels so the probes see one row per distinct state
    seen: dict[bytes, np.ndarray] = {}
    for f in frames:
        seen.setdefault(f.tobytes(), f)
    unique_frames = list(seen.values())

    encoder_fn = _encoder_fn(agent)
    latents = encoder_fn(unique_frames)

    report = {
        "transitions_recorded": len(transitions),
        "unique_frames_observed": len(unique_frames),
        "loss_history": list(agent._wm_loss_history),
        "convergence": probe_convergence(agent._wm_loss_history),
        "state_count": probe_state_count(latents),
        "manifold": probe_manifold_dimension(latents),
        "action_geometry": probe_action_geometry(
            transitions,
            encoder_fn=encoder_fn,
            num_actions=env.num_actions,
            latent_dim=latents.shape[1],
        ),
        "termination": probe_termination(per_episode_outcomes),
        "actor_info": dict(agent._actor_info_last),
        "critic_info": dict(agent._critic_info_last),
        "avg_imagined_return": (
            float(np.mean(agent._imagined_returns)) if agent._imagined_returns else 0.0
        ),
        "train_steps_done": train_steps_done,
    }
    return report


def _encoder_fn(agent: DreamerAgent):
    """Returns a callable ``frames -> (N, D) tensor`` that runs frames through
    the trained encoder. Used by the diagnostic probes.
    """
    @torch.no_grad()
    def fn(frames: list[np.ndarray]) -> torch.Tensor:
        agent.world_model.encoder.eval()
        out = agent.world_model.encoder(frames_to_tensor(frames).to(agent.device))
        agent.world_model.encoder.train()
        return out.cpu()
    return fn


# ---- explain ---------------------------------------------------------------

def explain(report: dict) -> str:
    lines: list[str] = []

    t = report["termination"]
    lines.append(
        f"outcome: {t['num_successful']}/{t['num_episodes']} episodes successful "
        f"(success_rate={t['success_rate']:.1%})"
    )
    if t["num_successful"] > 0:
        lines.append(f"  avg steps to success: {t['avg_steps_to_success']:.1f}")
    if t["num_successful"] < t["num_episodes"]:
        lines.append(f"  avg steps to failure: {t['avg_steps_to_failure']:.1f}")

    ai = report.get("actor_info") or {}
    ci = report.get("critic_info") or {}
    if ai:
        lines.append(
            f"  actor: policy={ai.get('actor/policy', 0.0):.4f}  "
            f"entropy={ai.get('actor/entropy', 0.0):.4f}  "
            f"adv_scale={ai.get('actor/adv_scale', 0.0):.4f}"
        )
    if ci:
        lines.append(
            f"  critic: main={ci.get('critic/main', 0.0):.4f}  "
            f"slow={ci.get('critic/slow', 0.0):.4f}"
        )
    lines.append(
        f"  avg imagined return over training: {report.get('avg_imagined_return', 0.0):.4f}  "
        f"(train steps: {report.get('train_steps_done', 0)})"
    )

    conv = report["convergence"]
    lines.append(
        f"convergence: final wm-loss {conv['final_loss']:.5f}  "
        f"(min {conv['min_loss']:.5f}, max {conv['max_loss']:.5f})"
    )

    sc = report["state_count"]
    lines.append(
        f"distinct latent classes: {sc['distinct']}  (gap threshold {sc['threshold']:.4f})"
    )
    lines.append(f"unique pixel frames observed: {report['unique_frames_observed']}")

    ag = report["action_geometry"]
    norms = ag["mean_delta_norms"]
    lines.append(
        f"action rank (independent move directions): {ag['action_rank']}  "
        f"-- singular values {['%.3f' % s for s in ag['singular_values']]}"
    )
    if ag.get("raw_action_rank") != ag.get("action_rank"):
        threshold = ag.get("axis_similarity_threshold")
        threshold_text = f", angle-gap threshold {threshold:.2f}" if threshold is not None else ""
        lines.append(
            f"  raw SVD rank {ag['raw_action_rank']} compressed to "
            f"{ag['signed_axis_count']} signed movement axes{threshold_text}"
        )
    for i, j in ag["inverse_pairs"]:
        c = ag["cosine_matrix"][i][j]
        lines.append(f"  actions {i} and {j} are inverses (cos={c:+.2f})")
    for a, norm in enumerate(norms):
        lines.append(
            f"  action {a}: |delta_z|={norm:.3f}, "
            f"effective={ag['effective_counts'][a]} transitions"
        )

    m = report["manifold"]
    lines.append(
        f"latent storage: participation ratio {m['participation_ratio']:.2f} "
        f"(how spread out the encoder packs states; not the world dim)"
    )

    d = ag["action_rank"]
    k = report["unique_frames_observed"]
    if d > 0 and k > 1:
        side = k ** (1.0 / d)
        side_int = round(side)
        if side_int >= 2 and abs(side - side_int) < 0.05 and side_int ** d == k:
            shape = " x ".join([str(side_int)] * d)
            lines.append(
                f"==> structural hypothesis: regular {d}D lattice of side {side_int} ({shape})"
            )
        else:
            lines.append(f"==> {d}D structure with {k} states; not a regular lattice")
    elif d == 0:
        lines.append("==> no independent action directions detected")
    return "\n".join(lines)
