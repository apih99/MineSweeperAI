"""Vectorized batch of N independent Minesweeper boards (same rules as ``MinesweeperEnv``).

RNG policy (deliberate; see the M2 lesson about accidentally shared streams):
  * Board ``i`` owns exactly one ``np.random.Generator`` (``self._rngs[i]``), created by
    ``gymnasium.utils.seeding.np_random(seeds[i])`` -- the same generator a single
    ``MinesweeperEnv.reset(seed=seeds[i])`` would use.
  * That generator is used ONLY for board ``i``'s mine placement, one ``choice`` call per
    episode at the first click, and it persists across that board's auto-resets (like a single
    env that is seeded once and then reset without a seed).
  * No other component draws from these generators; there is no shared/global stream.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from gymnasium.utils import seeding
from scipy import ndimage

from minesweeper_rl.env.gym_env import R_FIRST, R_LOSS, R_SAFE, R_WIN

_LABEL_STRUCT = np.zeros((3, 3, 3), dtype=bool)
_LABEL_STRUCT[1] = True  # 8-connectivity within a board, never across the batch axis


def _box3(x: np.ndarray) -> np.ndarray:
    """3x3 box sum (ints) / dilation (bool) over the last two axes of (k,H,W), zero padded."""
    k, h, w = x.shape
    p = np.zeros((k, h + 2, w + 2), dtype=x.dtype)
    p[:, 1:-1, 1:-1] = x
    if x.dtype == bool:
        r = p[:, :-2] | p[:, 1:-1] | p[:, 2:]
        return r[:, :, :-2] | r[:, :, 1:-1] | r[:, :, 2:]
    r = p[:, :-2] + p[:, 1:-1] + p[:, 2:]
    return r[:, :, :-2] + r[:, :, 1:-1] + r[:, :, 2:]


class BatchedMinesweeperEnv:
    def __init__(self, n_envs: int, height: int = 9, width: int = 9, n_mines: int = 10):
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if height < 1 or width < 1:
            raise ValueError("height and width must be >= 1")
        if not 1 <= n_mines <= height * width - 1:
            raise ValueError(f"n_mines must be in [1, {height * width - 1}], got {n_mines}")
        self.n_envs, self.height, self.width, self.n_mines = n_envs, height, width, n_mines
        self.n_cells = height * width
        self.max_steps = self.n_cells
        N, H, W = n_envs, height, width
        self.mines = np.zeros((N, H, W), dtype=bool)
        self.revealed = np.zeros((N, H, W), dtype=bool)
        self.counts = np.zeros((N, H, W), dtype=np.int8)
        self.zero_labels = np.zeros((N, H, W), dtype=np.int32)
        self.mines_placed = np.zeros(N, dtype=bool)
        self.steps = np.zeros(N, dtype=np.int64)
        self._rngs: list[np.random.Generator] | None = None
        # _candidates[s] == np.delete(np.arange(n_cells), s), precomputed (identical arrays, so
        # rng.choice consumes the generator exactly like Board._place_random).
        base = np.arange(self.n_cells)
        self._candidates = np.stack([np.delete(base, s) for s in range(self.n_cells)])
        self._arange_envs = np.arange(N)
        self._obs_state = self._initial_obs(N).reshape(N, 10, self.n_cells)

    # ---- helpers ----------------------------------------------------
    def _initial_obs(self, k: int) -> np.ndarray:
        obs = np.zeros((k, 10, self.height, self.width), dtype=np.float32)
        obs[:, 0] = 1.0  # channel 0 = hidden, channel 1+c = revealed with c adjacent mines
        return obs

    def _obs_add(self, newly: np.ndarray) -> None:
        """Update the persistent one-hot observation for freshly revealed cells ``newly`` (N,H,W)."""
        n_idx, cell = np.nonzero(newly.reshape(self.n_envs, self.n_cells))
        counts = self.counts.reshape(self.n_envs, self.n_cells)[n_idx, cell]
        self._obs_state[n_idx, 0, cell] = 0.0
        self._obs_state[n_idx, 1 + counts.astype(np.intp), cell] = 1.0

    def _mask(self) -> np.ndarray:
        return ~self.revealed.reshape(self.n_envs, self.n_cells)

    def _reset_boards(self, sel: np.ndarray) -> None:
        self.mines[sel] = False
        self.revealed[sel] = False
        self.counts[sel] = 0
        self.zero_labels[sel] = 0
        self.mines_placed[sel] = False
        self.steps[sel] = 0
        self._obs_state[sel] = self._initial_obs(len(sel)).reshape(len(sel), 10, self.n_cells)

    def _place(self, idx: np.ndarray, flat_actions: np.ndarray) -> None:
        """Place mines on boards ``idx`` (first click); consumes each board's own RNG exactly
        like ``Board._place_random``."""
        rngs, cand, n_mines = self._rngs, self._candidates, self.n_mines
        chosen = np.empty((len(idx), n_mines), dtype=np.int64)
        for row, (i, act) in enumerate(zip(idx.tolist(), flat_actions[idx].tolist())):
            chosen[row] = rngs[i].choice(cand[act], size=n_mines, replace=False)
        self.mines.reshape(self.n_envs, self.n_cells)[idx[:, None], chosen] = True
        m = self.mines[idx].astype(np.int8)
        counts = _box3(m) - m
        self.counts[idx] = counts
        labels, _ = ndimage.label(counts == 0, structure=_LABEL_STRUCT)
        self.zero_labels[idx] = labels
        self.mines_placed[idx] = True

    def _flood(self, idx: np.ndarray, flat_actions: np.ndarray) -> None:
        """Reveal zero-count component of the clicked cell plus its numbered border."""
        zl = self.zero_labels[idx]
        lab = zl.reshape(len(idx), self.n_cells)[np.arange(len(idx)), flat_actions[idx]]
        region = zl == lab[:, None, None]
        self.revealed[idx] |= _box3(region)

    # ---- full-state snapshot (for exact checkpoint/resume) ----------------
    _STATE_ARRAYS = ("mines", "revealed", "counts", "zero_labels", "mines_placed", "steps", "_obs_state")

    def current(self):
        """(obs, action_mask) of the boards as they stand now (same as the last step's output)."""
        N = self.n_envs
        return self._obs_state.reshape(N, 10, self.height, self.width).copy(), self._mask()

    def get_state(self) -> dict:
        """Everything needed to continue bit-exactly: all arrays + every board's Generator state."""
        if self._rngs is None:
            raise RuntimeError("call reset() before get_state()")
        st = {k: getattr(self, k).copy() for k in self._STATE_ARRAYS}
        st["rng_states"] = [r.bit_generator.state for r in self._rngs]
        return st

    def set_state(self, st: dict) -> None:
        for k in self._STATE_ARRAYS:
            arr = getattr(self, k)
            new = np.asarray(st[k])
            if new.shape != arr.shape or new.dtype != arr.dtype:
                raise ValueError(f"env state '{k}' mismatch: {new.shape}/{new.dtype} vs {arr.shape}/{arr.dtype}")
            arr[...] = new
        if len(st["rng_states"]) != self.n_envs:
            raise ValueError("env state has the wrong number of board generators")
        rngs = []
        for s in st["rng_states"]:
            g = np.random.Generator(np.random.PCG64())
            g.bit_generator.state = s
            rngs.append(g)
        self._rngs = rngs

    # ---- gym-like API -----------------------------------------------
    def reset(self, seeds: Sequence[int] | int | None = None):
        """Reset all boards. ``seeds``: int s -> s, s+1, ...; sequence -> one per board;
        None -> keep existing generators (fresh random ones if never seeded)."""
        N = self.n_envs
        if seeds is None:
            if self._rngs is None:
                self._rngs = [seeding.np_random(None)[0] for _ in range(N)]
        else:
            if isinstance(seeds, (int, np.integer)):
                seeds = [int(seeds) + i for i in range(N)]
            seeds = list(seeds)
            if len(seeds) != N:
                raise ValueError(f"expected {N} seeds, got {len(seeds)}")
            self._rngs = [seeding.np_random(int(s))[0] for s in seeds]
        self._reset_boards(np.arange(N))
        info = {
            "action_mask": self._mask(),
            "won": np.zeros(N, dtype=bool),
            "invalid_action": np.zeros(N, dtype=bool),
            "episode_done": np.zeros(N, dtype=bool),
        }
        return self._obs_state.reshape(N, 10, self.height, self.width).copy(), info

    def step(self, actions):
        if self._rngs is None:
            raise RuntimeError("call reset() before step()")
        N, HW = self.n_envs, self.n_cells
        a = np.asarray(actions)
        if a.shape != (N,) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"actions must be an integer array of shape ({N},)")
        a = a.astype(np.int64, copy=False)
        if (a < 0).any() or (a >= HW).any():
            raise ValueError(f"actions must be in [0, {HW})")
        ar = self._arange_envs
        rev_flat = self.revealed.reshape(N, HW)  # views onto the state arrays
        valid = ~rev_flat[ar, a]
        prev_revealed = self.revealed.copy()
        self.steps += 1

        first = valid & ~self.mines_placed
        if first.any():
            self._place(np.flatnonzero(first), a)
        hit = valid & self.mines.reshape(N, HW)[ar, a]
        safe = valid & ~hit
        v = np.flatnonzero(valid)
        rev_flat[v, a[v]] = True
        zero = safe & (self.counts.reshape(N, HW)[ar, a] == 0)
        if zero.any():
            self._flood(np.flatnonzero(zero), a)

        won = safe & (rev_flat.sum(axis=1) == HW - self.n_mines)
        terminated = hit | won
        truncated = ~terminated & (self.steps >= self.max_steps)

        rewards = np.full(N, R_SAFE, dtype=np.float32)
        rewards[first] = R_FIRST
        rewards[won] = R_WIN
        rewards[hit] = R_LOSS
        rewards[~valid] = 0.0

        done = terminated | truncated
        self._obs_add(self.revealed & ~prev_revealed)
        shape = (N, 10, self.height, self.width)
        state = self._obs_state.reshape(shape)
        # final_obs is only meaningful on finished boards; other rows are left zero (cheap).
        final_obs = np.zeros(shape, dtype=np.float32)
        final_mask = self._mask()
        info = {
            "won": won,
            "invalid_action": ~valid,
            "episode_done": done,
            "final_obs": final_obs,
            "final_mask": final_mask,
        }
        if done.any():
            sel = np.flatnonzero(done)
            final_obs[sel] = state[sel]
            self._reset_boards(sel)
            info["action_mask"] = self._mask()
        else:
            info["action_mask"] = final_mask
        return state.copy(), rewards, terminated, truncated, info
