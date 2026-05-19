"""Generic neural-net utilities.

Nothing here is env-specific. These are the small mathematical building blocks
the rest of the agent leans on:

  * symlog / symexp     — wide-range scalar squashing
  * twohot encode/loss  — symlog-bucketed regression target for rewards & values
  * OneHotST            — straight-through one-hot categorical sample
  * MLP                 — LayerNorm + GELU stack used by every head and the RSSM
  * frames_to_tensor    — uint8 (B,H,W,3) numpy → float32 (B,3,H,W) torch
  * one_hot             — integer action list → one-hot tensor

Keeping these in a single module means every other file can pull the same
primitives, and there is exactly one place to look if a numerical detail
needs to change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- scalar squashing ------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


# ---- twohot encoding -------------------------------------------------------
#
# A scalar y is represented as a probability vector over B symlog-spaced bins.
# Encoding: place the mass on the two neighboring bins so the bin-center mean
# (under symexp) exactly reconstructs y. This gives a differentiable
# regression target that handles wide value ranges without the gradient
# pathologies of plain MSE on raw rewards/returns.

class TwoHot:
    def __init__(self, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> None:
        self.num_bins = num_bins
        # bin edges are evenly spaced in symlog space so they're dense near zero
        # and sparse far from zero — matching the typical scale of returns.
        edges = torch.linspace(low, high, num_bins)
        self.register_centers(edges)

    def register_centers(self, edges: torch.Tensor) -> None:
        self._centers_symlog = edges
        self._centers_value = symexp(edges)

    def to(self, device: torch.device) -> "TwoHot":
        self._centers_symlog = self._centers_symlog.to(device)
        self._centers_value = self._centers_value.to(device)
        return self

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """Encode scalar(s) y -> twohot probability vector over bins."""
        y_sl = symlog(y).unsqueeze(-1)  # (..., 1)
        centers = self._centers_symlog  # (B,)
        # find the upper bin index whose center is >= y_sl
        below = (centers <= y_sl).sum(dim=-1) - 1  # (...,)
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1
        cb = centers[below]
        ca = centers[above]
        # linear interpolation weights so weighted-sum of centers == y_sl
        denom = (ca - cb).clamp_min(1e-8)
        w_above = ((y_sl.squeeze(-1) - cb) / denom).clamp(0.0, 1.0)
        w_below = 1.0 - w_above
        out = torch.zeros(*y.shape, self.num_bins, device=y.device, dtype=torch.float32)
        out.scatter_(-1, below.unsqueeze(-1), w_below.unsqueeze(-1))
        out.scatter_add_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
        return out

    def decode(self, probs: torch.Tensor) -> torch.Tensor:
        """probs over bins -> scalar value (symexp of the mean symlog center)."""
        return (probs * self._centers_value).sum(dim=-1)

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        return self.decode(F.softmax(logits, dim=-1))


def twohot_loss(logits: torch.Tensor, target: torch.Tensor, twohot: TwoHot) -> torch.Tensor:
    """Cross-entropy of softmax(logits) against the twohot encoding of target."""
    target_probs = twohot.encode(target).detach()
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1)


# ---- straight-through categorical ------------------------------------------

def one_hot_st(logits: torch.Tensor) -> torch.Tensor:
    """Straight-through one-hot categorical sample.

    Forward: draws a discrete one-hot sample from Categorical(softmax(logits)).
    Backward: gradients flow through the softmax probabilities, as if the
    one-hot were the probabilities themselves. Standard ``y = probs +
    (sample - probs).detach()`` trick.

    Used for the RSSM's stochastic latent z_t so the world-model loss can
    flow back into the prior/posterior logits.
    """
    probs = F.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, num_samples=1).squeeze(-1)
    sample = torch.zeros_like(flat)
    sample.scatter_(-1, idx.unsqueeze(-1), 1.0)
    sample = sample.reshape(probs.shape)
    return probs + (sample - probs).detach()


# ---- categorical KL --------------------------------------------------------

def categorical_kl(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
) -> torch.Tensor:
    """KL(p || q) for two categorical distributions parameterized by logits.

    Both inputs have shape (..., C). Returns shape (...).
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


# ---- MLP block -------------------------------------------------------------

class MLP(nn.Module):
    """LayerNorm + GELU MLP. Stays small and standard so the heads stay
    interchangeable and there's one place to tune the activation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        layers: int = 2,
    ) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---- tensor adapters -------------------------------------------------------

def frames_to_tensor(frames: list[np.ndarray] | np.ndarray) -> torch.Tensor:
    """uint8 (..., H, W, 3) -> float32 (..., 3, H, W) in [0, 1]."""
    if isinstance(frames, list):
        arr = np.stack(frames)
    else:
        arr = frames
    arr = arr.astype(np.float32) / 255.0
    # move channels to the position before H, W
    return torch.from_numpy(np.moveaxis(arr, -1, -3)).contiguous()


def one_hot(actions: list[int] | torch.Tensor, num_actions: int) -> torch.Tensor:
    if isinstance(actions, list):
        idx = torch.tensor(actions, dtype=torch.long)
    else:
        idx = actions.long()
    return F.one_hot(idx, num_classes=num_actions).float()
