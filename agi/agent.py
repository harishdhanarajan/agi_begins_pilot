"""Learner-only agent.

The agent is given only:
  * env.num_actions  — count of opaque action handles
  * env.reset() / env.step(a) — the two primitives of life

It is never told the world is a grid, that frames are 2D, or what an action
means. Discovery happens via *learning*:

  1. Act randomly. Record (frame_t, action, frame_{t+1}) transitions.
  2. Train a JEPA-style world model:
        z_t   = encoder(frame_t)
        z_t+1 = target_encoder(frame_{t+1})              [no gradient]
        z_hat = predictor(z_t, action_one_hot)
        loss  = ||z_hat - z_t+1||^2
     target_encoder is an exponential moving average of encoder. This
     prevents the trivial collapse of mapping everything to a constant.
  3. Probe the trained model for emergent structure. Probes are
     world-agnostic: they only ask geometric questions of the latent space.
       * convergence — did the model fit the dynamics?
       * state count — how many latent equivalence classes did the encoder
         carve out (gap-based clustering, no priors on count or shape).
       * manifold dimension — PCA on observed latents; how many components
         carry 95% of the variance.
       * action geometry — mean shift per action in latent space; cosine
         similarity reveals inverse-vector pairs.

The probes never name a topology. They report primitives; interpretation
('this is an 8x8 grid') is downstream cognition, not part of the learner.
This keeps the agent the same code for *any* world it is dropped into.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Encoder, Predictor


# ---- 0. interfaces ----------------------------------------------------------

class AgentEnv(Protocol):
    @property
    def num_actions(self) -> int: ...
    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> tuple[np.ndarray, bool]: ...


# ---- 1. data collection -----------------------------------------------------

def collect_transitions(
    env: AgentEnv,
    num_episodes: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, int, np.ndarray]]:
    transitions = []
    for _ in range(num_episodes):
        s = env.reset()
        while True:
            a = int(rng.integers(0, env.num_actions))
            sp, done = env.step(a)
            transitions.append((s, a, sp))
            s = sp
            if done:
                break
    return transitions


def _frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    arr = np.stack(frames).astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


def _one_hot(actions: list[int], num_actions: int) -> torch.Tensor:
    out = torch.zeros(len(actions), num_actions)
    for i, a in enumerate(actions):
        out[i, a] = 1.0
    return out


# ---- 2. learner -------------------------------------------------------------

class JEPALearner:
    def __init__(
        self,
        num_actions: int,
        latent_dim: int = 32,
        lr: float = 1e-3,
        ema_decay: float = 0.99,
    ) -> None:
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.predictor = Predictor(latent_dim=latent_dim, num_actions=num_actions)
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=lr,
        )
        self.ema_decay = ema_decay

    def train_step(
        self,
        frames_t: torch.Tensor,
        actions: list[int],
        frames_tp1: torch.Tensor,
    ) -> float:
        z_t = self.encoder(frames_t)
        with torch.no_grad():
            z_target = self.target_encoder(frames_tp1)
        a_oh = _one_hot(actions, self.num_actions)
        z_pred = self.predictor(z_t, a_oh)
        loss = F.mse_loss(z_pred, z_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                tp.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

        return float(loss.item())

    @torch.no_grad()
    def encode(self, frames: list[np.ndarray]) -> torch.Tensor:
        self.encoder.eval()
        out = self.encoder(_frames_to_tensor(frames))
        self.encoder.train()
        return out


def train(
    learner: JEPALearner,
    transitions: list[tuple[np.ndarray, int, np.ndarray]],
    num_steps: int = 3000,
    batch_size: int = 64,
    rng: np.random.Generator | None = None,
    log_every: int = 500,
) -> list[float]:
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(transitions)
    history: list[float] = []
    for step in range(num_steps):
        idx = rng.integers(0, n, size=batch_size)
        batch = [transitions[i] for i in idx]
        frames_t = _frames_to_tensor([t[0] for t in batch])
        actions = [t[1] for t in batch]
        frames_tp1 = _frames_to_tensor([t[2] for t in batch])
        loss = learner.train_step(frames_t, actions, frames_tp1)
        history.append(loss)
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1:>5}/{num_steps}  loss={loss:.5f}")
    return history


# ---- 3. probes --------------------------------------------------------------
#
# Each probe asks a single geometric question of the trained latent space.
# None of them know what the world is.


def probe_convergence(history: list[float], window: int = 200) -> dict:
    if len(history) < window:
        window = max(1, len(history) // 4)
    start = float(np.mean(history[:window]))
    end = float(np.mean(history[-window:]))
    ratio = end / max(start, 1e-12)
    return {"initial_loss": start, "final_loss": end, "shrink_ratio": ratio}


def probe_state_count(latents: torch.Tensor) -> dict:
    """Count distinct latent equivalence classes via a gap criterion on the
    sorted pairwise distance distribution.

    Intuition: if the encoder learned to separate the world's states,
    intra-class distances are near zero and inter-class distances are
    bounded away from zero. The largest *relative* gap in the sorted
    positive distance list separates 'same' from 'different'. Union-find
    over distances under that threshold yields the cluster count. No
    built-in expected count or shape.
    """
    n = latents.shape[0]
    if n <= 1:
        return {"distinct": n, "threshold": 0.0}
    d = torch.cdist(latents, latents)
    iu = torch.triu_indices(n, n, offset=1)
    pairwise = d[iu[0], iu[1]].cpu().numpy()

    sorted_pos = np.sort(pairwise[pairwise > 1e-6])
    if len(sorted_pos) < 2:
        return {"distinct": 1, "threshold": 0.0}
    ratios = sorted_pos[1:] / np.maximum(sorted_pos[:-1], 1e-12)
    cut_idx = int(np.argmax(ratios))
    threshold = float((sorted_pos[cut_idx] + sorted_pos[cut_idx + 1]) / 2)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    d_np = d.cpu().numpy()
    for i in range(n):
        for j in range(i + 1, n):
            if d_np[i, j] < threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    distinct = len({find(i) for i in range(n)})
    return {"distinct": distinct, "threshold": threshold}


def probe_manifold_dimension(latents: torch.Tensor, variance_target: float = 0.95) -> dict:
    z = latents - latents.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(z)
    var = sv.pow(2)
    cumulative = torch.cumsum(var, dim=0) / var.sum().clamp_min(1e-12)
    dim = int((cumulative < variance_target).sum().item()) + 1
    participation = (var.sum() ** 2) / (var.pow(2).sum().clamp_min(1e-12))
    return {
        "dim_for_variance": dim,
        "participation_ratio": float(participation.item()),
        "variance_target": variance_target,
        "spectrum": var.cpu().numpy().tolist(),
    }


def probe_action_geometry(
    learner: JEPALearner,
    transitions: list[tuple[np.ndarray, int, np.ndarray]],
) -> dict:
    """For each action, mean latent displacement over transitions where the
    pixel observation actually changed. Cosine similarity between action
    vectors flags inverse pairs (near -1), duplicates (near +1), and
    independent directions (near 0)."""
    by_action: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {a: [] for a in range(learner.num_actions)}
    for s, a, sp in transitions:
        if s.shape == sp.shape and not np.array_equal(s, sp):
            by_action[a].append((s, sp))

    mean_delta = torch.zeros(learner.num_actions, learner.latent_dim)
    effective_counts: dict[int, int] = {}
    for a, samples in by_action.items():
        effective_counts[a] = len(samples)
        if not samples:
            continue
        frames_t = _frames_to_tensor([x[0] for x in samples])
        frames_tp1 = _frames_to_tensor([x[1] for x in samples])
        with torch.no_grad():
            z_t = learner.encoder(frames_t)
            z_tp1 = learner.encoder(frames_tp1)
        mean_delta[a] = (z_tp1 - z_t).mean(dim=0)

    norms = mean_delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normed = mean_delta / norms
    cos = (normed @ normed.t()).cpu().numpy().tolist()
    return {
        "effective_counts": effective_counts,
        "cosine_matrix": cos,
        "mean_delta_norms": mean_delta.norm(dim=1).cpu().numpy().tolist(),
    }


# ---- 4. orchestration ------------------------------------------------------


def discover(
    env: AgentEnv,
    num_episodes: int = 300,
    train_steps: int = 3000,
    batch_size: int = 64,
    latent_dim: int = 32,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"[learner] collecting transitions over {num_episodes} episodes")
    transitions = collect_transitions(env, num_episodes, rng)
    print(f"[learner] recorded {len(transitions)} transitions")

    learner = JEPALearner(num_actions=env.num_actions, latent_dim=latent_dim)
    print(f"[learner] training JEPA world model for {train_steps} steps")
    history = train(learner, transitions, num_steps=train_steps, batch_size=batch_size, rng=rng)

    seen: dict[bytes, np.ndarray] = {}
    for s, a, sp in transitions:
        seen[s.tobytes()] = s
        seen[sp.tobytes()] = sp
    unique_frames = list(seen.values())
    latents = learner.encode(unique_frames)

    return {
        "transitions_recorded": len(transitions),
        "unique_frames_observed": len(unique_frames),
        "loss_history": history,
        "convergence": probe_convergence(history),
        "state_count": probe_state_count(latents),
        "manifold": probe_manifold_dimension(latents),
        "action_geometry": probe_action_geometry(learner, transitions),
    }


def explain(report: dict) -> str:
    lines: list[str] = []
    conv = report["convergence"]
    lines.append(
        f"convergence: loss {conv['initial_loss']:.4f} -> {conv['final_loss']:.4f} "
        f"(shrunk to {conv['shrink_ratio'] * 100:.1f}% of start)"
    )
    sc = report["state_count"]
    lines.append(
        f"distinct latent classes: {sc['distinct']}  "
        f"(gap threshold {sc['threshold']:.4f})"
    )
    lines.append(f"unique pixel frames observed: {report['unique_frames_observed']}")
    m = report["manifold"]
    lines.append(
        f"latent manifold: {m['dim_for_variance']}D for {int(m['variance_target'] * 100)}% variance, "
        f"participation ratio {m['participation_ratio']:.2f}"
    )

    ag = report["action_geometry"]
    norms = ag["mean_delta_norms"]
    lines.append("action vectors in latent space:")
    for a, norm in enumerate(norms):
        lines.append(
            f"  action {a}: |delta_z|={norm:.3f}, "
            f"effective={ag['effective_counts'][a]} transitions"
        )
    cos = ag["cosine_matrix"]
    inverse_threshold = -0.85
    duplicate_threshold = 0.85
    n = len(cos)
    for i in range(n):
        for j in range(i + 1, n):
            c = cos[i][j]
            if c < inverse_threshold:
                lines.append(f"  actions {i} and {j} look like inverses (cos={c:+.2f})")
            elif c > duplicate_threshold:
                lines.append(f"  actions {i} and {j} look like duplicates (cos={c:+.2f})")
    return "\n".join(lines)
