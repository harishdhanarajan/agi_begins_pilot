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

import time
from copy import deepcopy
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Encoder, Predictor, ValueHead


# ---- VIZ (brick-2 helper; remove this block + the `watch` plumbing to retire) ----
#
# Renders a pixel frame as an ASCII grid for a human watching the agent play.
# The agent never sees this output; it's purely for the observer. The renderer
# is env-agnostic: it detects tile size from the frame itself and assigns one
# character per unique color, sorted by brightness.

def _detect_tile_size(frame: np.ndarray) -> int:
    """Smallest run of identical pixels in the middle row — works for any env
    that paints uniform-colored tiles, regardless of tile_pixels setting."""
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
    # one char per unique color, ordered dark -> bright
    unique = sorted({tuple(c.tolist()) for c in frame.reshape(-1, 3)}, key=sum)
    palette = " .oO@#*+"
    color_char = {c: palette[min(i, len(palette) - 1)] for i, c in enumerate(unique)}
    lines = []
    for r in range(tile // 2, h, tile):
        row = "".join(color_char[tuple(frame[r, c].tolist())] for c in range(tile // 2, w, tile))
        lines.append(row)
    return "\n".join(lines)


def _render_step(frame: np.ndarray, header: str) -> None:
    # clear screen + home cursor (ANSI; works in modern Windows Terminal / PS).
    print("\033[2J\033[H", end="")
    print(header)
    print(_frame_to_ascii(frame))


# ---- END VIZ ----


# ---- 0. interfaces ----------------------------------------------------------

class AgentEnv(Protocol):
    @property
    def num_actions(self) -> int: ...
    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> tuple[np.ndarray, bool, float]: ...


Transition = tuple[np.ndarray, int, np.ndarray, float, bool]


# ---- 1. data collection -----------------------------------------------------

def collect_transitions(
    env: AgentEnv,
    num_episodes: int,
    rng: np.random.Generator,
    learner: "JEPALearner | None" = None,
    watch: bool = False,
    watch_label: str = "",
) -> list[Transition]:
    """Collect transitions for `num_episodes` episodes.

    If `learner` is provided, actions are chosen by `learner.select_action`
    (novelty-biased). Otherwise actions are uniform random — that path is
    the bootstrap used in the very first cycle of `discover`, before the
    learner has any useful predictor.

    Every observed frame is registered with the learner's buffer so the next
    cycle's action selection has more to compare against.
    """
    transitions: list[Transition] = []
    for ep in range(num_episodes):
        s = env.reset()
        if learner is not None:
            learner.observe(s)
        if watch:
            _render_step(s, f"{watch_label}  ep {ep + 1}/{num_episodes}  step 0  (reset)")
            time.sleep(0.08)
        ep_start = len(transitions)
        step_idx = 0
        while True:
            if learner is not None:
                a = learner.select_action(s, rng)
            else:
                a = int(rng.integers(0, env.num_actions))
            sp, done, r = env.step(a)
            transitions.append((s, a, sp, r, done))
            if learner is not None:
                learner.observe(sp)
            step_idx += 1
            if watch:
                tag = " WIN" if r > 0 else (" timeout" if done else "")
                _render_step(
                    sp,
                    f"{watch_label}  ep {ep + 1}/{num_episodes}  step {step_idx}  "
                    f"action={a}  reward={r}{tag}",
                )
                time.sleep(0.08)
            s = sp
            if done:
                break
        if learner is not None:
            learner.record_episode(transitions[ep_start:])
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


def _scale_invariant(x: np.ndarray) -> np.ndarray:
    """Center and divide by std. Returns zeros when all entries are equal —
    the signal carries no information about which action is better, so its
    softmax contribution should be flat."""
    s = float(x.std())
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / s


# ---- 2. learner -------------------------------------------------------------

class JEPALearner:
    def __init__(
        self,
        num_actions: int,
        latent_dim: int = 32,
        lr: float = 1e-3,
        ema_decay: float = 0.99,
        max_success_trajectories: int = 64,
    ) -> None:
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.predictor = Predictor(latent_dim=latent_dim, num_actions=num_actions)
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )
        self.ema_decay = ema_decay

        # value head: separate module, separate optimizer, so its gradient
        # never touches the encoder or predictor params. The encoder's only
        # job stays JEPA latent prediction; value bootstrapping happens on
        # detached latents.
        self.value_head = ValueHead(latent_dim=latent_dim)
        self.value_opt = torch.optim.AdamW(
            self.value_head.parameters(), lr=lr, weight_decay=1e-4
        )

        # novelty buffer: deduped seen frames (by raw pixels) and a cached
        # tensor of their target-encoder embeddings. select_action compares
        # the predictor's ẑ_next against this buffer; the action whose
        # predicted next-latent is farthest from anything already seen wins.
        self._seen_pixels: set[bytes] = set()
        self._seen_frames: list[np.ndarray] = []
        self._buffer_z: torch.Tensor | None = None
        # encoder's own learned "same vs different state" distance, derived
        # from the buffer's pairwise spectrum (same gap-clustering algorithm
        # as probe_state_count). Used by the success-memory match to decide
        # whether the current frame counts as the same state as a stored one.
        self._buffer_match_threshold: float = 0.0

        # running mean of predicted value during training. Surfaced in
        # explain() so we can tell at a glance whether the value head is
        # learning anything (stays near zero on unrewarded envs, rises on
        # rewarded ones once Bellman backpropagation kicks in).
        self._v_pred_sum: float = 0.0
        self._v_pred_n: int = 0

        # episodic success memory (this-run-only; resets when a new learner
        # is constructed, which happens once per program invocation). Stores
        # prefixes of episodes that hit reward > 0, so future visits to the
        # same state can bias toward the action that worked. Curated to the
        # K shortest trajectories — a "best/shortest known paths" prior.
        self.max_success_trajectories = max_success_trajectories
        self._success_memory: list[list[tuple[np.ndarray, int]]] = []
        # flattened, re-encoded view of the memory for fast NN lookup.
        # Rebuilt each cycle so it lives in the current latent space.
        self._success_states_z: torch.Tensor | None = None
        self._success_actions: list[int] = []
        self._success_remaining: list[int] = []
        self._best_success_len: int | None = None
        # diagnostics: how often did the match condition fire?
        self._mem_hits: int = 0
        self._mem_queries: int = 0

    def train_step(
        self,
        frames_t: torch.Tensor,
        actions: list[int],
        frames_tp1: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        # --- JEPA latent loss (unchanged from brick 2) ---
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

        # --- TD(0) value bootstrap (separate path, encoder shielded) ---
        # Both latents are computed inside no_grad, so the value loss can
        # only flow into value_head params — never into encoder or predictor.
        # Target = reward + (1 - done) * V(s'); undiscounted because episodes
        # terminate, so no magic gamma.
        with torch.no_grad():
            z_now = self.encoder(frames_t)
            z_next = self.encoder(frames_tp1)
            v_next = self.value_head(z_next)
            target_v = rewards + (1.0 - dones) * v_next
        v_pred = self.value_head(z_now)
        value_loss = F.mse_loss(v_pred, target_v)
        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        with torch.no_grad():
            self._v_pred_sum += float(v_pred.mean().item())
            self._v_pred_n += 1

        return float(loss.item())

    @torch.no_grad()
    def encode(self, frames: list[np.ndarray]) -> torch.Tensor:
        self.encoder.eval()
        out = self.encoder(_frames_to_tensor(frames))
        self.encoder.train()
        return out

    # ---- novelty buffer ----

    def observe(self, frame: np.ndarray) -> None:
        """Register a frame as 'seen'. Dedup is by raw pixels, so a finite
        world has a finite buffer no matter how many transitions we collect."""
        h = frame.tobytes()
        if h in self._seen_pixels:
            return
        self._seen_pixels.add(h)
        self._seen_frames.append(frame)

    @torch.no_grad()
    def refresh_buffer(self) -> None:
        """Re-encode the seen frames with the current target encoder. Call this
        between training chunks so novelty distances reflect the latest model.

        Also recomputes `_buffer_match_threshold` — the encoder's own learned
        "same vs different state" distance, found by the gap-clustering
        algorithm probe_state_count uses. Cached here so select_action can
        decide success-memory matches without recomputing pairwise distances
        on every step.
        """
        if not self._seen_frames:
            self._buffer_z = None
            self._buffer_match_threshold = 0.0
            return
        self.target_encoder.eval()
        self._buffer_z = self.target_encoder(_frames_to_tensor(self._seen_frames))

        n = self._buffer_z.shape[0]
        if n < 2:
            self._buffer_match_threshold = 0.0
            return
        d = torch.cdist(self._buffer_z, self._buffer_z)
        iu = torch.triu_indices(n, n, offset=1)
        pairwise = d[iu[0], iu[1]].cpu().numpy()
        sorted_pos = np.sort(pairwise[pairwise > 1e-6])
        if len(sorted_pos) < 2:
            self._buffer_match_threshold = 0.0
            return
        ratios = sorted_pos[1:] / np.maximum(sorted_pos[:-1], 1e-12)
        cut_idx = int(np.argmax(ratios))
        self._buffer_match_threshold = float(
            (sorted_pos[cut_idx] + sorted_pos[cut_idx + 1]) / 2
        )

    # ---- success memory ----

    def record_episode(self, episode: list[Transition]) -> None:
        """If the episode hit reward > 0, store the prefix (states + actions)
        leading up to and including the rewarding step. Curated to the K
        shortest trajectories — longer paths are dropped when full, so the
        memory always reflects the best known ways to win.

        Only the first reward in the episode is anchored, so trajectories
        terminate at the first positive outcome rather than running on past
        it. Works on any env: when no episode ever sees reward > 0 (e.g.
        CycleEnv, RaggedGridEnv), nothing is stored and memory stays empty.
        """
        for i, (_s, _a, _sp, r, _done) in enumerate(episode):
            if r > 0:
                trajectory = [(step[0], step[1]) for step in episode[: i + 1]]
                traj_len = len(trajectory)
                if self._best_success_len is None or traj_len < self._best_success_len:
                    self._best_success_len = traj_len
                self._success_memory.append(trajectory)
                if len(self._success_memory) > self.max_success_trajectories:
                    self._success_memory.sort(key=len)
                    self._success_memory = self._success_memory[
                        : self.max_success_trajectories
                    ]
                return

    @torch.no_grad()
    def refresh_success_index(self) -> None:
        """Re-encode the success memory under the current target encoder so
        the nearest-neighbor lookup in select_action lives in the latest
        latent space. Maintains parallel arrays so a single NN query returns
        both the recommended action and the steps-remaining-to-goal at that
        point in its parent trajectory (used as a tiebreaker among matches:
        prefer the shorter known path forward).
        """
        if not self._success_memory:
            self._success_states_z = None
            self._success_actions = []
            self._success_remaining = []
            return
        flat_frames: list[np.ndarray] = []
        self._success_actions = []
        self._success_remaining = []
        for traj in self._success_memory:
            L = len(traj)
            for i, (frame, action) in enumerate(traj):
                flat_frames.append(frame)
                self._success_actions.append(action)
                self._success_remaining.append(L - i)
        self.target_encoder.eval()
        self._success_states_z = self.target_encoder(_frames_to_tensor(flat_frames))

    @torch.no_grad()
    def select_action(
        self,
        frame: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Pick an action biased toward predicted-novel AND predicted-valuable
        next states, with an extra bias toward replaying actions that worked
        from this state in past successful episodes.

        For each candidate action a:
            z_hat = predictor(z, a)                  # imagined next latent
            novelty[a] = min distance from z_hat to the seen-frames buffer
            value[a]   = value_head(z_hat)           # expected future return

        Memory bias: encode the current frame with the target encoder, look
        it up against the flattened success memory; if the nearest stored
        state is within the encoder's own learned "same state" threshold
        (`_buffer_match_threshold`), put a 1.0 on the action that was taken
        there. Ties broken by smallest remaining-steps-to-goal — the
        "shortest known path forward" prior. When no match exists, no
        successes have been recorded yet, or the env has no reward at all,
        memory_bias stays zero and behavior reduces to novelty + value.

        All three signals are standardized (center, divide by std) so none
        dominates by absolute scale, then summed and softmaxed — exploration
        stays alive because the bias is a one-hot, not a hard override.

        Falls back to uniform random when the buffer has fewer entries than
        the action count — not enough data to compare actions yet.
        """
        if self._buffer_z is None or self._buffer_z.shape[0] < self.num_actions:
            return int(rng.integers(0, self.num_actions))

        self.encoder.eval()
        z = self.encoder(_frames_to_tensor([frame]))
        self.encoder.train()

        novelty = np.zeros(self.num_actions, dtype=np.float64)
        value = np.zeros(self.num_actions, dtype=np.float64)
        for a in range(self.num_actions):
            a_oh = _one_hot([a], self.num_actions)
            z_hat = self.predictor(z, a_oh)
            novelty[a] = float(torch.cdist(z_hat, self._buffer_z).min().item())
            value[a] = float(self.value_head(z_hat).item())

        memory_bias = np.zeros(self.num_actions, dtype=np.float64)
        self._mem_queries += 1
        if (
            self._success_states_z is not None
            and self._buffer_match_threshold > 0.0
        ):
            self.target_encoder.eval()
            z_target = self.target_encoder(_frames_to_tensor([frame]))
            dists = torch.cdist(z_target, self._success_states_z)[0]
            within = (dists < self._buffer_match_threshold).nonzero().flatten()
            if within.numel() > 0:
                remaining = np.array(
                    [self._success_remaining[int(i)] for i in within]
                )
                best = int(within[int(np.argmin(remaining))].item())
                memory_bias[self._success_actions[best]] = 1.0
                self._mem_hits += 1

        score = (
            _scale_invariant(novelty)
            + _scale_invariant(value)
            + _scale_invariant(memory_bias)
        )
        if not np.any(score):
            return int(rng.integers(0, self.num_actions))
        e = np.exp(score - score.max())
        probs = e / e.sum()
        return int(rng.choice(self.num_actions, p=probs))


def train(
    learner: JEPALearner,
    transitions: list[Transition],
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
        rewards = torch.tensor([float(t[3]) for t in batch], dtype=torch.float32)
        dones = torch.tensor([float(t[4]) for t in batch], dtype=torch.float32)
        loss = learner.train_step(frames_t, actions, frames_tp1, rewards, dones)
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
    end = float(np.mean(history[-window:]))
    minimum = float(min(history))
    maximum = float(max(history))
    return {
        "final_loss": end,
        "min_loss": minimum,
        "max_loss": maximum,
        "stability": end / max(minimum, 1e-12),
    }


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
    transitions: list[Transition],
) -> dict:
    """For each action, mean latent displacement over transitions where the
    pixel observation actually changed. Cosine similarity between action
    vectors flags inverse pairs (near -1), duplicates (near +1), and
    independent directions (near 0)."""
    by_action: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {a: [] for a in range(learner.num_actions)}
    for s, a, sp, _r, _done in transitions:
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

    # rank of the action-vector matrix = number of independent directions
    # the agent can move in latent space = world dimensionality under
    # translation. uses SVD with a relative tolerance against the largest
    # singular value, which is robust to overall scaling.
    sv = torch.linalg.svdvals(mean_delta)
    rel_tol = 0.05
    action_rank = int((sv > sv.max() * rel_tol).sum().item()) if sv.numel() else 0

    # count pairs whose cosine is very close to -1.
    inverse_pairs: list[tuple[int, int]] = []
    n = mean_delta.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if cos[i][j] < -0.85:
                inverse_pairs.append((i, j))

    return {
        "effective_counts": effective_counts,
        "cosine_matrix": cos,
        "mean_delta_norms": mean_delta.norm(dim=1).cpu().numpy().tolist(),
        "action_rank": action_rank,
        "inverse_pairs": inverse_pairs,
        "singular_values": sv.cpu().numpy().tolist(),
    }


def probe_termination(transitions: list[Transition]) -> dict:
    """Per-episode outcome statistics.

    Splits the flat transition list back into episodes on the `done` flag,
    then asks: how often did random exploration stumble into a positive-
    reward terminal state, and how long did it take when it did? This is
    our first concrete handle on 'is the agent winning?' Random play is
    the baseline; brick 2 will replace it with novelty-weighted action
    selection and we expect this number to go up.
    """
    episodes: list[list[tuple[float, bool]]] = []
    current: list[tuple[float, bool]] = []
    for _s, _a, _sp, r, done in transitions:
        current.append((r, done))
        if done:
            episodes.append(current)
            current = []

    num_ep = len(episodes)
    successes = [ep for ep in episodes if any(r > 0 for r, _ in ep)]
    failures = [ep for ep in episodes if not any(r > 0 for r, _ in ep)]

    success_rate = len(successes) / num_ep if num_ep else 0.0
    avg_len_success = float(np.mean([len(ep) for ep in successes])) if successes else 0.0
    avg_len_failure = float(np.mean([len(ep) for ep in failures])) if failures else 0.0

    return {
        "num_episodes": num_ep,
        "num_successful": len(successes),
        "success_rate": success_rate,
        "avg_steps_to_success": avg_len_success,
        "avg_steps_to_failure": avg_len_failure,
    }


# ---- 4. orchestration ------------------------------------------------------


def discover(
    env: AgentEnv,
    num_episodes: int = 300,
    train_steps: int = 5000,
    batch_size: int = 64,
    latent_dim: int = 32,
    num_cycles: int = 10,
    seed: int = 0,
    watch: bool = False,
    log_every: int = 0,
) -> dict:
    """Closed-loop discovery: alternate collection and training.

    The loop is `num_cycles` rounds of (collect a chunk, train a chunk). The
    first chunk is uniform random (no learner yet); every subsequent chunk
    uses the partially-trained learner to bias action choice toward
    predicted-novel next states. Total work matches the old single-shot
    version — only the *interleaving* is new, plus the novelty-biased action
    selection that the interleaving makes possible.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    ep_per_cycle = max(1, num_episodes // num_cycles)
    steps_per_cycle = max(1, train_steps // num_cycles)

    learner = JEPALearner(num_actions=env.num_actions, latent_dim=latent_dim)
    transitions: list[Transition] = []
    history: list[float] = []
    per_cycle_outcomes: list[dict] = []

    for cycle in range(num_cycles):
        # cycle 0 uses random actions to seed the buffer; from cycle 1 onward
        # the learner's predictor drives action selection.
        actor = learner if cycle > 0 else None
        mode = "model-biased (novelty + value)" if actor else "uniform random — bootstrap"
        print(
            f"[learner] cycle {cycle + 1}/{num_cycles}  collecting {ep_per_cycle} episodes "
            f"({mode})",
            flush=True,
        )
        chunk = collect_transitions(
            env,
            ep_per_cycle,
            rng,
            learner=actor,
            watch=watch,
            watch_label=f"cycle {cycle + 1}/{num_cycles} ({mode})",
        )
        per_cycle_outcomes.append(probe_termination(chunk))
        transitions.extend(chunk)

        print(
            f"[learner] cycle {cycle + 1}/{num_cycles}  training {steps_per_cycle} steps "
            f"on {len(transitions)} accumulated transitions",
            flush=True,
        )
        h = train(
            learner,
            transitions,
            num_steps=steps_per_cycle,
            batch_size=batch_size,
            rng=rng,
            log_every=log_every,
        )
        history.extend(h)
        learner.refresh_buffer()
        learner.refresh_success_index()

    seen: dict[bytes, np.ndarray] = {}
    for s, _a, sp, _r, _done in transitions:
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
        "termination": probe_termination(transitions),
        "per_cycle_outcomes": per_cycle_outcomes,
        "avg_predicted_value": (
            learner._v_pred_sum / learner._v_pred_n if learner._v_pred_n else 0.0
        ),
        "success_memory": {
            "trajectories_kept": len(learner._success_memory),
            "best_path_length": learner._best_success_len,
            "mem_bias_hit_rate": (
                learner._mem_hits / learner._mem_queries
                if learner._mem_queries
                else 0.0
            ),
        },
    }


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

    if "per_cycle_outcomes" in report and report["per_cycle_outcomes"]:
        per_cycle = report["per_cycle_outcomes"]
        rate_vals = [c["success_rate"] for c in per_cycle]
        rates = [f"{r:.0%}" for r in rate_vals]
        lines.append(f"  per-cycle success rate: {' -> '.join(rates)}")
        if len(rate_vals) >= 2:
            peak = max(rate_vals)
            peak_idx = rate_vals.index(peak)
            final = rate_vals[-1]
            mid = len(rate_vals) // 2
            early = sum(rate_vals[:mid]) / max(mid, 1)
            late = sum(rate_vals[mid:]) / max(len(rate_vals) - mid, 1)
            lines.append(
                f"  trend: peak {peak:.0%} (cycle {peak_idx + 1}) -> final {final:.0%}  "
                f"|  early-half {early:.0%} vs late-half {late:.0%}"
            )

    if "avg_predicted_value" in report:
        lines.append(
            f"  avg predicted value over training: {report['avg_predicted_value']:.4f}"
        )

    if "success_memory" in report:
        sm = report["success_memory"]
        if sm["best_path_length"] is not None:
            lines.append(
                f"  best successful path length: {sm['best_path_length']} steps"
            )
        lines.append(
            f"  success-memory bias hit rate: {sm['mem_bias_hit_rate']:.1%} "
            f"({sm['trajectories_kept']} trajectories kept)"
        )

    conv = report["convergence"]
    lines.append(
        f"convergence: final loss {conv['final_loss']:.5f}  "
        f"(min {conv['min_loss']:.5f}, max {conv['max_loss']:.5f})"
    )

    sc = report["state_count"]
    lines.append(
        f"distinct latent classes: {sc['distinct']}  "
        f"(gap threshold {sc['threshold']:.4f})"
    )
    lines.append(f"unique pixel frames observed: {report['unique_frames_observed']}")

    ag = report["action_geometry"]
    norms = ag["mean_delta_norms"]
    lines.append(
        f"action rank (independent move directions): {ag['action_rank']}  "
        f"-- singular values {['%.3f' % s for s in ag['singular_values']]}"
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

    # generic lattice hypothesis. given an N-D world with K distinct states,
    # the simplest hypothesis is a regular N-D lattice of side K**(1/N). Use
    # the directly observed pixel-frame count as the state-count input here;
    # the encoder cluster count above serves as an audit that the encoder
    # learned to keep those frames apart.
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
            lines.append(
                f"==> {d}D structure with {k} states; not a regular lattice"
            )
    elif d == 0:
        lines.append("==> no independent action directions detected")
    return "\n".join(lines)
