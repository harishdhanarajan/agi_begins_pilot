"""Neural modules for the world model.

Nothing here is world-specific. Same Encoder works on any image size; same
Predictor works for any action count. The agent does not know it is looking
at a grid, a cycle, or anything else — it only sees pixels and opaque action
indices, and it must learn whatever representation lets it predict the future.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Pixels -> latent vector.

    Two conv blocks for local feature extraction, then adaptive pooling to a
    fixed 4x4 grid so the head is size-agnostic, then an MLP head. The 4x4
    pool preserves coarse positional information that pure global-average
    pooling would destroy.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """(latent, action_one_hot) -> next_latent.

    Pure MLP. Predicts in latent space, never in pixel space — this is JEPA's
    central bet, and it makes the loss meaningful even when much of the pixel
    detail is irrelevant. Single-output by design: reward awareness is the
    value head's job, on the encoder side, so the predictor's trunk is not
    asked to balance conflicting objectives.
    """

    def __init__(self, latent_dim: int = 32, num_actions: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


class ValueHead(nn.Module):
    """latent -> scalar V.

    Predicts expected (undiscounted) future return from a given latent state.
    Trained by TD(0) bootstrap: V(s) <- r + (1 - done) * V(s'). The input
    latent is fed in detached, so value learning does not perturb the encoder
    that the JEPA loss is responsible for shaping. Without this shielding,
    encoder and value heads compete for the same representation and JEPA's
    latent loss fails to converge (the brick-3 lesson).
    """

    def __init__(self, latent_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
