"""Single-board Gymnasium Minesweeper environment (reference implementation)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from minesweeper_rl.env.core import Board

R_WIN = 1.0
R_LOSS = -1.0
R_SAFE = 0.3
R_FIRST = 0.0


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, height: int = 9, width: int = 9, n_mines: int = 10, render_mode: str | None = None):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        # Validate parameters early.
        Board(height, width, n_mines, np.random.default_rng(0))
        self.height, self.width, self.n_mines = height, width, n_mines
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(height * width)
        self.observation_space = spaces.Box(0.0, 1.0, (10, height, width), dtype=np.float32)
        self.board: Board | None = None
        self._steps = 0
        self._max_steps = height * width

    def _info(self, invalid: bool = False) -> dict:
        b = self.board
        return {
            "action_mask": b.action_mask(),
            "won": b.won,
            "revealed": int(b.revealed.sum()),
            "invalid_action": invalid,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = Board(self.height, self.width, self.n_mines, self.np_random)
        self._steps = 0
        return self.board.observation(), self._info()

    def step(self, action):
        if self.board is None:
            raise RuntimeError("call reset() before step()")
        if self.board.done:
            raise RuntimeError("episode finished; call reset()")
        self._steps += 1
        r, c = divmod(int(action), self.width)
        res = self.board.reveal(r, c)
        if not res.valid:
            reward, terminated = 0.0, False
        elif res.hit_mine:
            reward, terminated = R_LOSS, True
        elif res.won:
            reward, terminated = R_WIN, True
        elif res.first_click:
            reward, terminated = R_FIRST, False
        else:
            reward, terminated = R_SAFE, False
        truncated = (not terminated) and self._steps >= self._max_steps
        return self.board.observation(), reward, terminated, truncated, self._info(not res.valid)

    def render(self):
        if self.render_mode == "ansi" and self.board is not None:
            return self.board.render_text(show_mines=self.board.done)
        return None
