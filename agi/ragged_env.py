"""A ragged two-dimensional world.

The first row has ten cells. The remaining rows have seven cells each.
The agent sees only pixels and opaque actions; it is not told the row lengths
or that the world is ragged.
"""

from __future__ import annotations

import numpy as np


class RaggedGridEnv:
    def __init__(
        self,
        row_lengths: tuple[int, ...] = (10, 7, 7, 7, 7, 7, 7),
        max_moves: int = 30,
        tile_pixels: int = 4,
        seed: int | None = None,
    ) -> None:
        self._row_lengths = row_lengths
        self._max_cols = max(row_lengths)
        self._max_moves = max_moves
        self._tile_pixels = tile_pixels
        self._rng = np.random.default_rng(seed)
        self._cells = [
            (row, col)
            for row, row_length in enumerate(row_lengths)
            for col in range(row_length)
        ]
        self._cell_set = set(self._cells)

        self._action_deltas = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]
        self._rng.shuffle(self._action_deltas)

        self._player = self._cells[0]
        self._moves_left = max_moves
        self._terminated = True

    @property
    def num_actions(self) -> int:
        return len(self._action_deltas)

    def reset(self) -> np.ndarray:
        index = int(self._rng.integers(0, len(self._cells)))
        self._player = self._cells[index]
        self._moves_left = self._max_moves
        self._terminated = False
        return self._render()

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        if self._terminated:
            raise RuntimeError("episode is over; call reset() first")
        if not 0 <= action < self.num_actions:
            raise ValueError(f"unknown action {action!r}")

        dr, dc = self._action_deltas[action]
        next_cell = (self._player[0] + dr, self._player[1] + dc)
        if next_cell in self._cell_set:
            self._player = next_cell

        self._moves_left -= 1
        if self._moves_left <= 0:
            self._terminated = True
        return self._render(), self._terminated

    _VOID = np.array([8, 8, 12], dtype=np.uint8)
    _BG = np.array([24, 24, 32], dtype=np.uint8)
    _PLAYER = np.array([235, 200, 60], dtype=np.uint8)

    def _render(self) -> np.ndarray:
        tp = self._tile_pixels
        rows = len(self._row_lengths)
        img = np.broadcast_to(
            self._VOID,
            (rows * tp, self._max_cols * tp, 3),
        ).copy()

        for row, col in self._cells:
            img[row * tp:(row + 1) * tp, col * tp:(col + 1) * tp] = self._BG

        row, col = self._player
        img[row * tp:(row + 1) * tp, col * tp:(col + 1) * tp] = self._PLAYER
        return img
