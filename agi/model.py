"""Neural modules for the DreamerV3-architecture world model + actor-critic.

Pure ``nn.Module`` definitions, no training logic, no env-specific constants.
Each module takes its shapes as constructor arguments so the same code runs
on any world the agent is dropped into.

Modules
-------
* Encoder         pixels  -> embedding vector
* Decoder         (h, z)  -> pixels  (reconstructs the input shape)
* RSSMCell        GRU + categorical prior + categorical posterior
* RewardHead      (h, z)  -> twohot logits
* ContinueHead    (h, z)  -> binary logit (1 - done)
* Actor           (h, z)  -> categorical logits over actions
* Critic          (h, z)  -> twohot logits over value bins
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import MLP, one_hot_st


# ---- vision ----------------------------------------------------------------

class Encoder(nn.Module):
    """Pixels -> embedding vector.

    Two conv blocks for local features, AdaptiveAvgPool to a 4x4 spatial map
    so the head is size-agnostic, then a linear projection to ``embed_dim``.
    Works on any (H, W, 3) input shape because of the adaptive pool.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """(h, z) -> pixels reconstruction.

    The output shape is set at construction from a sample observation, so the
    decoder reproduces whatever (H, W) the env happens to render. After the
    transposed conv stack lifts a 4x4 feature map to 16x16, a final bilinear
    resize maps to the exact target (H, W). This sidesteps the need for
    perfectly matched conv arithmetic across arbitrary input shapes.
    """

    def __init__(
        self,
        deter_dim: int,
        stoch_flat_dim: int,
        out_shape: tuple[int, int, int],
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self._out_shape = out_shape  # (C, H, W) where C=3
        self.input_proj = nn.Sequential(
            nn.Linear(deter_dim + stoch_flat_dim, hidden_channels * 4 * 4),
            nn.LayerNorm(hidden_channels * 4 * 4),
            nn.GELU(),
        )
        self._hidden_channels = hidden_channels
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_shape[0], kernel_size=3, padding=1),
        )

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(torch.cat([h, z_flat], dim=-1))
        x = x.view(-1, self._hidden_channels, 4, 4)
        x = self.deconv(x)
        x = F.interpolate(x, size=self._out_shape[1:], mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


# ---- recurrent state-space model -------------------------------------------

class RSSMCell(nn.Module):
    """Deterministic GRU + categorical stochastic latent.

    State has two parts at every step:
      h_t   deterministic recurrent state (``deter_dim``)
      z_t   stochastic categorical latent (``K`` groups of ``C`` classes)

    Two transitions:

    img_step(h, z, a) -> (h_next, prior_logits)
        - inp = MLP([z_flat, a_onehot])
        - h_next = GRUCell(inp, h)
        - prior_logits = MLP(h_next).reshape(K, C)

    obs_step(h, x_emb) -> (post_logits, z)
        - post_logits = MLP([h, x_emb]).reshape(K, C)
        - z = OneHotST(post_logits)   # straight-through categorical sample

    The split is what lets the model train its dynamics from observations
    (posterior is informed by ``x_emb``) and still imagine forward without
    observations (prior alone gives the next-state distribution).
    """

    def __init__(
        self,
        num_actions: int,
        embed_dim: int,
        deter_dim: int = 128,
        stoch_groups: int = 16,
        stoch_classes: int = 16,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_groups = stoch_groups
        self.stoch_classes = stoch_classes
        self.stoch_flat = stoch_groups * stoch_classes

        # input to the GRU: [z_flat, a_onehot] -> hidden
        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_flat + num_actions, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.gru = nn.GRUCell(hidden, deter_dim)

        # prior logits from h alone (used for imagination)
        self.prior_net = MLP(deter_dim, hidden, self.stoch_flat, layers=1)

        # posterior logits from (h, encoder_embedding)
        self.post_net = MLP(deter_dim + embed_dim, hidden, self.stoch_flat, layers=1)

    def initial(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_groups, self.stoch_classes, device=device)
        return h, z

    def img_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_flat = z.reshape(z.shape[0], -1)
        inp = self.img_in(torch.cat([z_flat, a_onehot], dim=-1))
        h_next = self.gru(inp, h)
        prior_logits = self.prior_net(h_next).view(-1, self.stoch_groups, self.stoch_classes)
        return h_next, prior_logits

    def obs_step(
        self,
        h: torch.Tensor,
        x_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        post_logits = self.post_net(torch.cat([h, x_emb], dim=-1)).view(
            -1, self.stoch_groups, self.stoch_classes
        )
        z = one_hot_st(post_logits)
        return post_logits, z

    def prior_sample(self, prior_logits: torch.Tensor) -> torch.Tensor:
        return one_hot_st(prior_logits)


# ---- heads -----------------------------------------------------------------

class _StateHead(nn.Module):
    """Common pattern: MLP over [h, z_flat] -> out_dim logits."""

    def __init__(
        self,
        deter_dim: int,
        stoch_flat: int,
        out_dim: int,
        hidden: int = 128,
        zero_init_last: bool = False,
    ) -> None:
        super().__init__()
        self.net = MLP(deter_dim + stoch_flat, hidden, out_dim, layers=2)
        if zero_init_last:
            # Zero the final linear layer so initial logits are flat. For
            # twohot heads this yields a near-zero expected scalar (bins are
            # symmetric in symexp around 0); for the actor it gives a uniform
            # initial policy, which is what we want for exploration.
            last = list(self.net.net)[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z_flat], dim=-1))


class RewardHead(_StateHead):
    """(h, z) -> twohot logits over value bins."""

    def __init__(self, deter_dim: int, stoch_flat: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__(deter_dim, stoch_flat, out_dim, hidden, zero_init_last=True)


class ContinueHead(_StateHead):
    """(h, z) -> single scalar logit; sigmoid gives P(continue) = 1 - P(done)."""


class Actor(_StateHead):
    """(h, z) -> categorical logits over actions."""

    def __init__(self, deter_dim: int, stoch_flat: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__(deter_dim, stoch_flat, out_dim, hidden, zero_init_last=True)


class Critic(_StateHead):
    """(h, z) -> twohot logits over value bins."""

    def __init__(self, deter_dim: int, stoch_flat: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__(deter_dim, stoch_flat, out_dim, hidden, zero_init_last=True)
