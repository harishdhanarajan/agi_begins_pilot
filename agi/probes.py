"""Diagnostic probes.

Each probe asks a single geometric question of either the training history or
the encoder's latent space. None of them know what the world is — they only
read off generic numerical properties. They are pure spectators: nothing here
influences the agent's behavior. That's what lets them stay alive across the
DreamerV3 rewrite — they just point at the new world model's encoder instead
of the old JEPA encoder.

  * probe_convergence       — is training stable / has loss settled?
  * probe_state_count       — gap-clustering in latent space
  * probe_manifold_dimension— PCA participation ratio (audit)
  * probe_action_geometry   — action displacement matrix; rank + inverse pairs
  * probe_termination       — per-episode outcome statistics
"""

from __future__ import annotations

import numpy as np
import torch


def probe_convergence(history: list[float], window: int = 200) -> dict:
    if not history:
        return {"final_loss": 0.0, "min_loss": 0.0, "max_loss": 0.0, "stability": 1.0}
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
    transitions: list[tuple[np.ndarray, int, np.ndarray, float, bool]],
    encoder_fn,
    num_actions: int,
    latent_dim: int,
) -> dict:
    """For each action, mean latent displacement over transitions where the
    pixel observation actually changed. Cosine similarity between action
    vectors flags inverse pairs (near -1), duplicates (near +1), and
    independent directions (near 0).

    ``encoder_fn`` is called as ``encoder_fn(frames)`` where ``frames`` is a
    list of numpy frames; it returns a (N, D) tensor of latents. This lets
    the same probe run on any world model — JEPA, DreamerV3, or future
    architectures.
    """
    by_action: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {a: [] for a in range(num_actions)}
    for s, a, sp, _r, _done in transitions:
        if s.shape == sp.shape and not np.array_equal(s, sp):
            by_action[a].append((s, sp))

    mean_delta = torch.zeros(num_actions, latent_dim)
    effective_counts: dict[int, int] = {}
    for a, samples in by_action.items():
        effective_counts[a] = len(samples)
        if not samples:
            continue
        z_t = encoder_fn([x[0] for x in samples])
        z_tp1 = encoder_fn([x[1] for x in samples])
        mean_delta[a] = (z_tp1 - z_t).mean(dim=0)

    norms = mean_delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normed = mean_delta / norms
    cos = (normed @ normed.t()).cpu().numpy().tolist()

    sv = torch.linalg.svdvals(mean_delta)
    rel_tol = 0.05
    raw_action_rank = int((sv > sv.max() * rel_tol).sum().item()) if sv.numel() else 0

    n = mean_delta.shape[0]
    abs_cos_values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            abs_cos_values.append(abs(float(cos[i][j])))
    axis_similarity_threshold: float | None = None
    if len(abs_cos_values) >= 2:
        sorted_abs = np.sort(np.array(abs_cos_values, dtype=np.float64))
        gaps = sorted_abs[1:] - sorted_abs[:-1]
        gap_idx = int(np.argmax(gaps))
        if gaps[gap_idx] > 1e-9:
            axis_similarity_threshold = float(
                (sorted_abs[gap_idx] + sorted_abs[gap_idx + 1]) / 2
            )

    nonzero_actions = [
        a for a in range(n) if float(mean_delta[a].norm().item()) >= 1e-12
    ]
    inverse_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if axis_similarity_threshold is not None:
                if cos[i][j] <= -axis_similarity_threshold:
                    inverse_pairs.append((i, j))
            elif (
                len(nonzero_actions) == 2
                and raw_action_rank == 1
                and i in nonzero_actions
                and j in nonzero_actions
                and cos[i][j] < 0
            ):
                inverse_pairs.append((i, j))

    axis_reps: list[torch.Tensor] = []
    for a in nonzero_actions:
        norm = mean_delta[a].norm()
        direction = mean_delta[a] / norm
        if axis_similarity_threshold is not None and any(
            abs(float(direction @ rep)) >= axis_similarity_threshold
            for rep in axis_reps
        ):
            continue
        axis_reps.append(direction)
    signed_axis_count = len(axis_reps)
    action_rank = min(raw_action_rank, signed_axis_count) if signed_axis_count else raw_action_rank

    return {
        "effective_counts": effective_counts,
        "cosine_matrix": cos,
        "mean_delta_norms": mean_delta.norm(dim=1).cpu().numpy().tolist(),
        "action_rank": action_rank,
        "raw_action_rank": raw_action_rank,
        "signed_axis_count": signed_axis_count,
        "axis_similarity_threshold": axis_similarity_threshold,
        "inverse_pairs": inverse_pairs,
        "singular_values": sv.cpu().numpy().tolist(),
    }


def probe_termination(
    episodes: list[list[tuple[float, bool]]],
) -> dict:
    """Per-episode outcome statistics.

    ``episodes`` is a list of episodes, each a list of (reward, done) per
    step. Splitting from the buffer happens at the caller because that's
    where the episode boundaries live.
    """
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
