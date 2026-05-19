"""Actor-critic trained on imagined trajectories.

Given a world model that can ``imagine`` H steps forward from a start state,
the actor-critic does three things on every update:

  1. Imagine a trajectory ``(h, z, action, reward, cont, log_prob, entropy)``
     of length H starting from a batch of posterior states drawn from the
     replay-conditioned world model.
  2. Compute lambda returns along the trajectory using the slow target critic.
  3. Take a REINFORCE step on the actor with the return-minus-value advantage,
     normalized by an EMA of its percentile range, plus an entropy bonus.
  4. Take a regression step on the critic toward the lambda return, with a
     slow-critic regularizer that pulls the critic toward its own EMA copy.

Pure tensor ops; no env-specific logic.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Actor, Critic
from .nets import TwoHot, twohot_loss


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_actions: int,
        deter_dim: int,
        stoch_flat: int,
        twohot: TwoHot,
        hidden: int = 128,
        lam: float = 0.95,
        entropy_coef: float = 3e-4,
        target_decay: float = 0.98,
        slow_reg: float = 1.0,
    ) -> None:
        super().__init__()
        self.actor = Actor(deter_dim, stoch_flat, num_actions, hidden=hidden)
        self.critic = Critic(deter_dim, stoch_flat, twohot.num_bins, hidden=hidden)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.twohot = twohot
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.target_decay = target_decay
        self.slow_reg = slow_reg

        # EMA of the advantage percentile range, used to normalize advantages
        # so the actor's effective learning rate is invariant to reward scale.
        self.register_buffer("_adv_scale", torch.ones(1))
        self._adv_initialized = False

    @torch.no_grad()
    def update_target(self) -> None:
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(self.target_decay).add_(p.data, alpha=1.0 - self.target_decay)

    @staticmethod
    def lambda_return(
        reward: torch.Tensor,
        value: torch.Tensor,
        cont: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Truncated lambda-return.

        reward, value, cont, all shape (B, H). cont[t] = 1 means episode
        continues past step t. Return shape (B, H).
        """
        B, H = reward.shape
        out = torch.zeros_like(reward)
        # bootstrap from the final value
        next_g = value[:, -1]
        for t in reversed(range(H)):
            v_next = value[:, t + 1] if t + 1 < H else next_g
            target = reward[:, t] + cont[:, t] * ((1.0 - lam) * v_next + lam * next_g)
            out[:, t] = target
            next_g = target
        return out

    def _critic_values(self, h: torch.Tensor, z_flat: torch.Tensor, slow: bool) -> torch.Tensor:
        head = self.target_critic if slow else self.critic
        logits = head(h, z_flat)
        return self.twohot.expectation(logits)

    def actor_loss(
        self,
        traj: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        h = traj["h"]
        z = traj["z"]
        B, H = h.shape[:2]
        z_flat = z.reshape(B, H, -1)

        with torch.no_grad():
            value = self._critic_values(
                h.reshape(B * H, -1),
                z_flat.reshape(B * H, -1),
                slow=True,
            ).view(B, H)
            ret = self.lambda_return(traj["reward"], value, traj["cont"], self.lam)
            adv = ret - value

            # EMA-scaled advantage: divide by the 95-5 percentile range,
            # smoothed across updates, so the actor sees a roughly unit-scale
            # advantage regardless of the env's reward magnitude.
            flat_adv = adv.abs().reshape(-1)
            if flat_adv.numel() > 1:
                lo = torch.quantile(flat_adv, 0.05)
                hi = torch.quantile(flat_adv, 0.95)
                cur_scale = (hi - lo).clamp_min(1e-3)
            else:
                cur_scale = torch.tensor(1.0, device=adv.device)
            if not self._adv_initialized:
                self._adv_scale.copy_(cur_scale.detach().view(1))
                self._adv_initialized = True
            else:
                self._adv_scale.mul_(0.99).add_(cur_scale.detach().view(1) * 0.01)
            scale = self._adv_scale.clamp_min(1.0)
            adv_norm = adv / scale

        log_prob = traj["log_prob"]
        entropy = traj["entropy"]
        policy_term = -(log_prob * adv_norm.detach()).mean()
        entropy_term = -self.entropy_coef * entropy.mean()
        loss = policy_term + entropy_term
        info = {
            "actor/loss": float(loss.item()),
            "actor/policy": float(policy_term.item()),
            "actor/entropy": float(entropy.mean().item()),
            "actor/adv_scale": float(self._adv_scale.item()),
            "actor/return_mean": float(ret.mean().item()),
        }
        return loss, info, ret.detach()

    def critic_loss(
        self,
        traj: dict[str, torch.Tensor],
        lambda_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        h = traj["h"]
        z = traj["z"]
        B, H = h.shape[:2]
        z_flat = z.reshape(B, H, -1)

        logits = self.critic(h.reshape(B * H, -1), z_flat.reshape(B * H, -1)).view(B, H, -1)
        # main regression to the lambda return
        L_main = twohot_loss(logits, lambda_returns.detach(), self.twohot).mean()

        # slow critic regularizer: pull the critic toward the EMA copy's
        # current scalar prediction, smoothing the moving target.
        with torch.no_grad():
            slow_v = self._critic_values(
                h.reshape(B * H, -1),
                z_flat.reshape(B * H, -1),
                slow=True,
            ).view(B, H)
        L_slow = twohot_loss(logits, slow_v.detach(), self.twohot).mean()

        loss = L_main + self.slow_reg * L_slow
        info = {
            "critic/loss": float(loss.item()),
            "critic/main": float(L_main.item()),
            "critic/slow": float(L_slow.item()),
        }
        return loss, info
