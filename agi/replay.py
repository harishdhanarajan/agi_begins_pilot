"""Sequence replay buffer.

The RSSM trains on sequences, not iid transitions, so the buffer stores whole
episodes and serves up fixed-length windows. Sampling is uniform over
(episode, start-offset) pairs; if a chosen window runs off the end of an
episode the tail is padded by repeating the last frame, and a ``first`` flag
marks the very first step of each window so the world model knows to reset
its recurrent state.

The buffer is env-agnostic — it never looks at the observation contents.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


class SequenceReplay:
    def __init__(self, capacity: int = 1000) -> None:
        self._episodes: deque[dict[str, np.ndarray]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def total_steps(self) -> int:
        return sum(int(ep["action"].shape[0]) for ep in self._episodes)

    def add(self, episode: Iterable[tuple[np.ndarray, int, float, bool]]) -> None:
        """Each tuple is (obs_t, action_t, reward_{t+1}, done_{t+1}) — i.e.
        the observation that was acted on, the action taken, and the resulting
        reward/done. The terminal observation is appended last with a dummy
        action so the buffer has T+1 obs for T actions.
        """
        obs_list: list[np.ndarray] = []
        action_list: list[int] = []
        reward_list: list[float] = []
        cont_list: list[float] = []
        for obs, a, r, done in episode:
            obs_list.append(obs)
            action_list.append(int(a))
            reward_list.append(float(r))
            cont_list.append(0.0 if done else 1.0)
        if not obs_list:
            return
        self._episodes.append({
            "obs": np.stack(obs_list),
            "action": np.asarray(action_list, dtype=np.int64),
            "reward": np.asarray(reward_list, dtype=np.float32),
            "cont": np.asarray(cont_list, dtype=np.float32),
        })

    def ready(self, batch_size: int) -> bool:
        return len(self._episodes) >= max(1, batch_size // 4)

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Return a batch of length-T windows.

        Keys:
          obs     (B, T, H, W, 3)  uint8
          action  (B, T)           int64
          reward  (B, T)           float32   reward attributed to the t-th step
          cont    (B, T)           float32   0 on the step that ended the episode
          first   (B, T)           float32   1 on the first step of the window
        """
        n_eps = len(self._episodes)
        if n_eps == 0:
            raise RuntimeError("replay is empty; cannot sample")
        ep_indices = rng.integers(0, n_eps, size=batch_size)
        obs_batch, act_batch, rew_batch, cont_batch, first_batch = [], [], [], [], []
        for ei in ep_indices:
            ep = self._episodes[int(ei)]
            ep_len = int(ep["action"].shape[0])
            start = int(rng.integers(0, max(1, ep_len)))
            obs_w, act_w, rew_w, cont_w, first_w = self._window(ep, start, seq_len)
            obs_batch.append(obs_w)
            act_batch.append(act_w)
            rew_batch.append(rew_w)
            cont_batch.append(cont_w)
            first_batch.append(first_w)
        return {
            "obs": np.stack(obs_batch),
            "action": np.stack(act_batch),
            "reward": np.stack(rew_batch),
            "cont": np.stack(cont_batch),
            "first": np.stack(first_batch),
        }

    @staticmethod
    def _window(
        ep: dict[str, np.ndarray],
        start: int,
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ep_len = int(ep["action"].shape[0])
        end = start + seq_len
        if end <= ep_len:
            obs = ep["obs"][start:end]
            act = ep["action"][start:end]
            rew = ep["reward"][start:end]
            cont = ep["cont"][start:end]
        else:
            # pad by repeating the terminal step (cont=0 on padded steps ensures
            # the world model treats them as already-terminated and lambda
            # returns past the end carry no weight)
            real = ep_len - start
            pad = seq_len - real
            obs = np.concatenate([ep["obs"][start:ep_len], np.repeat(ep["obs"][ep_len - 1:ep_len], pad, axis=0)])
            act = np.concatenate([ep["action"][start:ep_len], np.zeros(pad, dtype=ep["action"].dtype)])
            rew = np.concatenate([ep["reward"][start:ep_len], np.zeros(pad, dtype=ep["reward"].dtype)])
            cont = np.concatenate([ep["cont"][start:ep_len], np.zeros(pad, dtype=ep["cont"].dtype)])
        first = np.zeros(seq_len, dtype=np.float32)
        first[0] = 1.0
        return obs, act, rew, cont, first

    def iter_frames(self) -> list[np.ndarray]:
        """All observations across all episodes, flattened. Used by the
        diagnostic probes (state count, action geometry) to get a sample of
        the env's frame distribution after training is done.
        """
        out: list[np.ndarray] = []
        for ep in self._episodes:
            for i in range(ep["obs"].shape[0]):
                out.append(ep["obs"][i])
        return out

    def iter_transitions(self) -> list[tuple[np.ndarray, int, np.ndarray, float, bool]]:
        """Flat list of (s, a, s', r, done) tuples. Same shape as the old IID
        view, used by the action-geometry probe. The current obs is paired
        with the *next* obs in the same episode; the terminal step is dropped.
        """
        out: list[tuple[np.ndarray, int, np.ndarray, float, bool]] = []
        for ep in self._episodes:
            n = int(ep["action"].shape[0])
            for i in range(n - 1):
                done = ep["cont"][i + 1] == 0.0
                out.append((ep["obs"][i], int(ep["action"][i]), ep["obs"][i + 1], float(ep["reward"][i]), bool(done)))
        return out
