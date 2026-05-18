"""Escape-grid environment.

This file is the world the agent inhabits. It deliberately exposes the
minimum possible interface:

    obs = env.reset()
    obs, done, reward = env.step(action)
    env.num_actions  # how many opaque action handles exist

Everything else the agent must discover. In particular:

* observations are pixels, not symbolic state.
* action handles are integers 0..N-1 with no semantic name.
  the env never tells the agent which one is "up".
* rewards are sparse: a positive number on a "good" terminal state, zero
  otherwise. the agent must still discover *what makes* a state good;
  the reward signal is the only "outcome" hint, no shaping.

The default rules mirror the JavaScript game in this repo, but constructor
parameters can change the world size and move budget without changing agent
code.
"""

from __future__ import annotations

import numpy as np


class EscapeGridEnv:
    def __init__(
        self,
        size: int = 7,
        max_moves: int = 25,
        tile_pixels: int = 4,
        seed: int | None = None,
    ) -> None:
        self._size = size
        self._max_moves = max_moves
        self._tile_pixels = tile_pixels
        self._rng = np.random.default_rng(seed)

        # action handles are opaque ints. internally they map to (drow, dcol).
        # the agent never sees this mapping.
        self._action_deltas = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]
        # shuffle so action 0 isn't always "up" across runs — discourages
        # the human (me) from accidentally encoding action semantics anywhere.
        self._rng.shuffle(self._action_deltas)

        self._exit = (size - 1, size - 1)
        self._player = (0, 0)
        self._moves_left = max_moves
        self._terminated = True  # forces reset() before step()

    # ---- public surface ---------------------------------------------------

    @property
    def num_actions(self) -> int:
        return len(self._action_deltas)

    def reset(self) -> np.ndarray:
        while True:
            r = int(self._rng.integers(0, self._size))
            c = int(self._rng.integers(0, self._size))
            if (r, c) != self._exit:
                break
        self._player = (r, c)
        self._moves_left = self._max_moves
        self._terminated = False
        return self._render()

    def step(self, action: int) -> tuple[np.ndarray, bool, float]:
        if self._terminated:
            raise RuntimeError("episode is over; call reset() first")
        if not 0 <= action < self.num_actions:
            raise ValueError(f"unknown action {action!r}")

        dr, dc = self._action_deltas[action]
        nr = self._player[0] + dr
        nc = self._player[1] + dc

        # bumping into a wall is a legal no-op; the move still counts.
        if 0 <= nr < self._size and 0 <= nc < self._size:
            self._player = (nr, nc)

        self._moves_left -= 1
        reached_exit = self._player == self._exit
        if reached_exit or self._moves_left <= 0:
            self._terminated = True
        reward = 1.0 if reached_exit else 0.0
        return self._render(), self._terminated, reward

    # ---- rendering --------------------------------------------------------

    # three distinct colors. the agent does not know which is which.
    _BG = np.array([24, 24, 32], dtype=np.uint8)
    _PLAYER = np.array([235, 200, 60], dtype=np.uint8)
    _EXIT = np.array([60, 200, 120], dtype=np.uint8)

    def _render(self) -> np.ndarray:
        s = self._size
        tp = self._tile_pixels
        img = np.broadcast_to(self._BG, (s * tp, s * tp, 3)).copy()

        er, ec = self._exit
        img[er * tp:(er + 1) * tp, ec * tp:(ec + 1) * tp] = self._EXIT

        pr, pc = self._player
        img[pr * tp:(pr + 1) * tp, pc * tp:(pc + 1) * tp] = self._PLAYER
        return img
