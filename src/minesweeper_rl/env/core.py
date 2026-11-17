"""Pure Minesweeper game logic (NumPy only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RevealResult:
    newly_revealed: int = 0
    hit_mine: bool = False
    won: bool = False
    first_click: bool = False
    valid: bool = True


class GameOverError(RuntimeError):
    """Raised when revealing a cell after the game has ended."""


class Board:
    """Minesweeper board with lazy mine placement (first click is always safe)."""

    def __init__(self, height: int, width: int, n_mines: int, rng: np.random.Generator):
        if height < 1 or width < 1:
            raise ValueError("height and width must be >= 1")
        if not 1 <= n_mines <= height * width - 1:
            raise ValueError(f"n_mines must be in [1, {height * width - 1}], got {n_mines}")
        self.height = int(height)
        self.width = int(width)
        self.n_mines = int(n_mines)
        self.rng = rng
        self.mines = np.zeros((height, width), dtype=bool)
        self.revealed = np.zeros((height, width), dtype=bool)
        self.counts = np.zeros((height, width), dtype=np.int8)
        self.mines_placed = False
        self._hit_mine = False
        self._exploded: tuple[int, int] | None = None
        self._n_revealed = 0

    # ---- properties -------------------------------------------------
    @property
    def n_safe_total(self) -> int:
        return self.height * self.width - self.n_mines

    @property
    def n_safe_remaining(self) -> int:
        revealed_safe = self._n_revealed - (1 if self._hit_mine else 0)
        return self.n_safe_total - revealed_safe

    @property
    def won(self) -> bool:
        return (not self._hit_mine) and self.mines_placed and self.n_safe_remaining == 0

    @property
    def done(self) -> bool:
        return self._hit_mine or self.won

    # ---- mine placement ---------------------------------------------
    def place_mines(self, mask: np.ndarray) -> None:
        """Place mines from a boolean mask (test hook; also used internally)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.height, self.width):
            raise ValueError("mask shape mismatch")
        if int(mask.sum()) != self.n_mines:
            raise ValueError("mask must contain exactly n_mines mines")
        self.mines = mask.copy()
        padded = np.pad(self.mines.astype(np.int8), 1)
        h, w = self.height, self.width
        counts = np.zeros((h, w), dtype=np.int8)
        for dr in range(3):
            for dc in range(3):
                if dr == 1 and dc == 1:
                    continue
                counts += padded[dr : dr + h, dc : dc + w]
        self.counts = counts
        self.mines_placed = True

    def _place_random(self, safe_r: int, safe_c: int) -> None:
        n = self.height * self.width
        safe_idx = safe_r * self.width + safe_c
        candidates = np.delete(np.arange(n), safe_idx)
        chosen = self.rng.choice(candidates, size=self.n_mines, replace=False)
        flat = np.zeros(n, dtype=bool)
        flat[chosen] = True
        self.place_mines(flat.reshape(self.height, self.width))

    # ---- actions ----------------------------------------------------
    def reveal(self, row: int, col: int) -> RevealResult:
        if self.done:
            raise GameOverError("game is over")
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(f"cell ({row}, {col}) out of bounds")
        if self.revealed[row, col]:
            return RevealResult(valid=False)

        first = not self.mines_placed
        if first:
            self._place_random(row, col)

        if self.mines[row, col]:
            self.revealed[row, col] = True
            self._n_revealed += 1
            self._hit_mine = True
            self._exploded = (row, col)
            return RevealResult(newly_revealed=1, hit_mine=True, won=False, first_click=first)

        newly = 1
        self.revealed[row, col] = True
        stack = [(row, col)]
        h, w = self.height, self.width
        while stack:
            r, c = stack.pop()
            if self.counts[r, c] != 0:
                continue
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    if not self.revealed[nr, nc]:
                        # neighbours of a zero cell are never mines
                        self.revealed[nr, nc] = True
                        newly += 1
                        stack.append((nr, nc))
        self._n_revealed += newly
        return RevealResult(newly_revealed=newly, hit_mine=False, won=self.won, first_click=first)

    # ---- views ------------------------------------------------------
    def observation(self) -> np.ndarray:
        obs = np.zeros((10, self.height, self.width), dtype=np.float32)
        obs[0] = ~self.revealed
        rr, cc = np.nonzero(self.revealed)
        # The exploded mine's cell uses its adjacent count (always 0..8, keeps one-hot valid).
        obs[1 + self.counts[rr, cc].astype(np.int64), rr, cc] = 1.0
        return obs

    def action_mask(self) -> np.ndarray:
        return (~self.revealed).reshape(-1)

    def render_text(self, show_mines: bool = False) -> str:
        h, w = self.height, self.width
        cw = max(2, len(str(w - 1)) + 1)
        rw = len(str(h - 1))
        lines = [" " * rw + " " + "".join(str(c).rjust(cw) for c in range(w))]
        for r in range(h):
            cells = []
            for c in range(w):
                if self._exploded == (r, c):
                    ch = "X"
                elif self.revealed[r, c]:
                    ch = "." if self.counts[r, c] == 0 else str(int(self.counts[r, c]))
                elif show_mines and self.mines[r, c]:
                    ch = "*"
                else:
                    ch = "#"
                cells.append(ch.rjust(cw))
            lines.append(str(r).rjust(rw) + " " + "".join(cells))
        return "\n".join(lines)
