"""World model.

Owns the encoder, RSSM cell, decoder, reward head, and continue head, and the
two operations that the actor-critic loop needs from them:

  * ``observe``  given a sequence of (obs, action), return the posterior and
                 prior states at every step. Used during world-model training.
  * ``imagine``  given start states, an actor, and a horizon, roll the prior
                 forward step-by-step. Used to generate the trajectories the
                 actor-critic trains on.

Plus ``loss`` which computes the full world-model objective (reconstruction +
reward + continue + KL prior + KL representation) on a sampled batch.

All shapes carry a leading ``(B, T)`` time dimension; modules in ``model.py``
operate on flat ``(B,)`` tensors, so this file is responsible for the
T-loop and the batch flattening before each head call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (
    ContinueHead,
    Decoder,
    Encoder,
    RewardHead,
    RSSMCell,
)
from .nets import TwoHot, categorical_kl, one_hot, twohot_loss


@dataclass
class WorldModelConfig:
    num_actions: int
    image_shape: tuple[int, int, int]   # (C, H, W)
    embed_dim: int = 64
    deter_dim: int = 128
    stoch_groups: int = 16
    stoch_classes: int = 16
    hidden: int = 128
    value_bins: int = 255
    value_range: tuple[float, float] = (-20.0, 20.0)
    free_nats: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(in_channels=cfg.image_shape[0], embed_dim=cfg.embed_dim)
        self.rssm = RSSMCell(
            num_actions=cfg.num_actions,
            embed_dim=cfg.embed_dim,
            deter_dim=cfg.deter_dim,
            stoch_groups=cfg.stoch_groups,
            stoch_classes=cfg.stoch_classes,
            hidden=cfg.hidden,
        )
        stoch_flat = cfg.stoch_groups * cfg.stoch_classes
        self.decoder = Decoder(
            deter_dim=cfg.deter_dim,
            stoch_flat_dim=stoch_flat,
            out_shape=cfg.image_shape,
        )
        self.reward_head = RewardHead(cfg.deter_dim, stoch_flat, cfg.value_bins, hidden=cfg.hidden)
        self.continue_head = ContinueHead(cfg.deter_dim, stoch_flat, 1, hidden=cfg.hidden)
        self.twohot = TwoHot(num_bins=cfg.value_bins, low=cfg.value_range[0], high=cfg.value_range[1])

    # ---- helpers -----------------------------------------------------------

    def _flatten_bt(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

    def _unflatten_bt(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        return x.reshape(B, T, *x.shape[1:])

    def _reset_where(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Zero out rows of x where mask == 1. mask is (B,)."""
        keep = (1.0 - mask).view(-1, *([1] * (x.dim() - 1)))
        return x * keep

    # ---- observe -----------------------------------------------------------

    def observe(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        first: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the RSSM forward across a (B, T) batch.

        Args
        ----
        obs     (B, T, C, H, W) float in [0, 1]
        action  (B, T) long; action taken AT step t
        first   (B, T) float; 1 on the first step of the window

        Convention: the action that produced obs[t+1] is action[t]. At step 0
        there is no prior action; we feed a zero one-hot, which the RSSM's
        input MLP will turn into a "no info" embedding.
        """
        B, T = obs.shape[:2]
        device = obs.device
        h, z = self.rssm.initial(B, device)

        # encode all observations in one batched pass for speed
        x_emb_all = self.encoder(self._flatten_bt(obs))
        x_emb_all = self._unflatten_bt(x_emb_all, B, T)

        a_onehot_all = one_hot(action, self.cfg.num_actions).to(device)

        prior_logits_list: list[torch.Tensor] = []
        post_logits_list: list[torch.Tensor] = []
        h_list: list[torch.Tensor] = []
        z_list: list[torch.Tensor] = []

        prev_action = torch.zeros(B, self.cfg.num_actions, device=device)

        for t in range(T):
            # reset h, z, prev_action wherever a new window begins
            m = first[:, t]
            h = self._reset_where(h, m)
            z = self._reset_where(z, m)
            prev_action = self._reset_where(prev_action, m)

            h, prior_logits = self.rssm.img_step(h, z, prev_action)
            post_logits, z = self.rssm.obs_step(h, x_emb_all[:, t])

            prior_logits_list.append(prior_logits)
            post_logits_list.append(post_logits)
            h_list.append(h)
            z_list.append(z)

            prev_action = a_onehot_all[:, t]

        return {
            "h": torch.stack(h_list, dim=1),                       # (B, T, deter)
            "z": torch.stack(z_list, dim=1),                       # (B, T, K, C)
            "prior_logits": torch.stack(prior_logits_list, dim=1),
            "post_logits": torch.stack(post_logits_list, dim=1),
            "x_emb": x_emb_all,
        }

    # ---- imagine -----------------------------------------------------------

    def imagine(
        self,
        start_h: torch.Tensor,
        start_z: torch.Tensor,
        actor: nn.Module,
        horizon: int,
    ) -> dict[str, torch.Tensor]:
        """Roll the prior forward H steps starting from (start_h, start_z).

        Each step samples an action from ``actor((h, z))`` (straight-through
        one-hot so gradients can flow back into the actor), advances the RSSM
        prior, and reads the reward/continue heads. Returns tensors of shape
        (B, H, ...).
        """
        B = start_h.shape[0]
        K = self.cfg.stoch_groups
        C = self.cfg.stoch_classes

        h = start_h.detach()
        z = start_z.detach()

        h_list, z_list, action_list, log_prob_list, entropy_list = [], [], [], [], []
        reward_list, cont_list = [], []
        actor_logits_list = []

        for _ in range(horizon):
            z_flat = z.reshape(B, K * C)
            actor_logits = actor(h, z_flat)
            probs = F.softmax(actor_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            a_idx = dist.sample()
            a_onehot = F.one_hot(a_idx, num_classes=self.cfg.num_actions).float()
            log_prob = dist.log_prob(a_idx)
            entropy = dist.entropy()

            h, prior_logits = self.rssm.img_step(h, z, a_onehot)
            z = self.rssm.prior_sample(prior_logits)
            # Detach h, z from the RSSM/world-model graph: imagined rollouts
            # train the actor and critic only; the world model is updated by
            # its own loss on real data. Without this, critic_loss.backward()
            # would needlessly populate RSSM .grad buffers.
            h = h.detach()
            z = z.detach()
            z_flat = z.reshape(B, K * C)

            r_logits = self.reward_head(h, z_flat)
            r = self.twohot.expectation(r_logits)
            c = torch.sigmoid(self.continue_head(h, z_flat).squeeze(-1))

            h_list.append(h)
            z_list.append(z)
            action_list.append(a_idx)
            log_prob_list.append(log_prob)
            entropy_list.append(entropy)
            reward_list.append(r)
            cont_list.append(c)
            actor_logits_list.append(actor_logits)

        return {
            "h": torch.stack(h_list, dim=1),                    # (B, H, deter)
            "z": torch.stack(z_list, dim=1),                    # (B, H, K, C)
            "action": torch.stack(action_list, dim=1),          # (B, H)
            "log_prob": torch.stack(log_prob_list, dim=1),      # (B, H)
            "entropy": torch.stack(entropy_list, dim=1),        # (B, H)
            "reward": torch.stack(reward_list, dim=1),          # (B, H)
            "cont": torch.stack(cont_list, dim=1),              # (B, H)
            "actor_logits": torch.stack(actor_logits_list, dim=1),
        }

    # ---- loss --------------------------------------------------------------

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        """Compute the world-model loss on a batch.

        batch keys: obs (B, T, C, H, W) float in [0,1], action (B, T) long,
                    reward (B, T), cont (B, T), first (B, T).
        """
        B, T = batch["obs"].shape[:2]
        out = self.observe(batch["obs"], batch["action"], batch["first"])

        h = out["h"]
        z = out["z"]
        z_flat = z.reshape(B, T, -1)

        # decode pixels
        recon = self.decoder(
            self._flatten_bt(h),
            self._flatten_bt(z_flat),
        )
        recon = self._unflatten_bt(recon, B, T)
        target = batch["obs"]
        L_recon = F.mse_loss(recon, target, reduction="none").sum(dim=(-3, -2, -1)).mean()

        # reward + continue
        r_logits = self.reward_head(self._flatten_bt(h), self._flatten_bt(z_flat))
        r_logits = self._unflatten_bt(r_logits, B, T)
        L_reward = twohot_loss(r_logits, batch["reward"], self.twohot).mean()

        c_logits = self.continue_head(self._flatten_bt(h), self._flatten_bt(z_flat)).squeeze(-1)
        c_logits = c_logits.view(B, T)
        L_cont = F.binary_cross_entropy_with_logits(c_logits, batch["cont"])

        # KL with balancing + free nats
        prior_logits = out["prior_logits"]
        post_logits = out["post_logits"]
        # per-group KL summed over groups, averaged over (B, T)
        kl_dyn = categorical_kl(post_logits.detach(), prior_logits).sum(dim=-1)        # prior -> sg(post)
        kl_rep = categorical_kl(post_logits, prior_logits.detach()).sum(dim=-1)        # post  -> sg(prior)
        free = self.cfg.free_nats
        L_dyn = kl_dyn.clamp_min(free).mean()
        L_rep = kl_rep.clamp_min(free).mean()

        total = L_recon + L_reward + L_cont + self.cfg.beta_dyn * L_dyn + self.cfg.beta_rep * L_rep

        breakdown = {
            "wm/total": float(total.item()),
            "wm/recon": float(L_recon.item()),
            "wm/reward": float(L_reward.item()),
            "wm/cont": float(L_cont.item()),
            "wm/kl_dyn": float(L_dyn.item()),
            "wm/kl_rep": float(L_rep.item()),
        }
        states = {"h": h, "z": z}
        return total, breakdown, states
