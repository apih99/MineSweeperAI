"""Baseline agents: uniform random and a simple rule-based solver."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate

from minesweeper_rl.agents.base import Agent

_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)


def _neigh_sum(a: np.ndarray) -> np.ndarray:
    """Sum of the 8 neighbours of every cell (off-board counts as 0)."""
    return correlate(a.astype(np.int32), _KERNEL, mode="constant", cval=0)


class RandomAgent(Agent):
    name = "random"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))


class RuleAgent(Agent):
    """Two basic deduction rules, iterated to a fixpoint; random guess when stuck.

    Stateless across steps: everything is recomputed from the observation.
    """

    name = "rule"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    @staticmethod
    def deduce(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (safe_mask, mine_mask): boolean (H, W) arrays over hidden cells."""
        hidden = obs[0] > 0.5
        revealed = ~hidden
        num = np.argmax(obs[1:10], axis=0).astype(np.int32)
        mine = np.zeros_like(hidden)
        safe = np.zeros_like(hidden)
        while True:
            unknown = hidden & ~mine & ~safe
            mc = _neigh_sum(mine)
            uc = _neigh_sum(unknown)
            # Rule 1: number already met by known mines -> other hidden neighbours are safe.
            r1 = revealed & (num == mc)
            # Rule 2: number needs every remaining unknown neighbour to be a mine.
            r2 = revealed & (uc > 0) & (num == mc + uc)
            new_safe = unknown & (_neigh_sum(r1) > 0)
            new_mine = unknown & (_neigh_sum(r2) > 0)
            if not new_safe.any() and not new_mine.any():
                break
            safe |= new_safe
            mine |= new_mine
        return safe, mine

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        safe, mine = self.deduce(obs)
        flat_safe = safe.reshape(-1) & mask
        if flat_safe.any():
            return int(np.argmax(flat_safe))  # lowest index
        cand = np.flatnonzero(mask & ~mine.reshape(-1))
        if cand.size == 0:
            cand = np.flatnonzero(mask)
        return int(self.rng.choice(cand))
