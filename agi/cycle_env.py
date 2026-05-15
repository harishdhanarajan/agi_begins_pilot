"""A one-dimensional cyclic world.

The agent sees only pixels and opaque actions, just like in EscapeGridEnv.
There are eight states arranged in a ring. One action moves clockwise, the
other counter-clockwise, but the action handles are shuffled so the agent is
not told which is which.
"""

from __future__ import annotations

import numpy as np


class CycleEnv:
    def __init__(
        self,
        num_states: int = 8,
        max_moves: int = 25,
        tile_pixels: int = 4,
        seed: int | None = None,
    ) -> None:
        self._num_states = num_states
        self._max_moves = max_moves
        self._tile_pixels = tile_pixels
        self._rng = np.random.default_rng(seed)

        self._action_deltas = [-1, 1]
        self._rng.shuffle(self._action_deltas)

        self._state = 0
        self._moves_left = max_moves
        self._terminated = True

    @property
    def num_actions(self) -> int:
        return len(self._action_deltas)

    def reset(self) -> np.ndarray:
        self._state = int(self._rng.integers(0, self._num_states))
        self._moves_left = self._max_moves
        self._terminated = False
        return self._render()

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        if self._terminated:
            raise RuntimeError("episode is over; call reset() first")
        if not 0 <= action < self.num_actions:
            raise ValueError(f"unknown action {action!r}")

        self._state = (self._state + self._action_deltas[action]) % self._num_states
        self._moves_left -= 1
        if self._moves_left <= 0:
            self._terminated = True
        return self._render(), self._terminated

    _BG = np.array([24, 24, 32], dtype=np.uint8)
    _PLAYER = np.array([235, 200, 60], dtype=np.uint8)

    def _render(self) -> np.ndarray:
        tp = self._tile_pixels
        img = np.broadcast_to(self._BG, (tp, self._num_states * tp, 3)).copy()
        start = self._state * tp
        img[:, start:start + tp] = self._PLAYER
        return img
