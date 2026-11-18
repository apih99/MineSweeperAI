import numpy as np
import pytest

from minesweeper_rl.env.batched import BatchedMinesweeperEnv
from minesweeper_rl.env.gym_env import MinesweeperEnv

CONFIGS = [(5, 5, 3, 8, 250), (9, 9, 10, 6, 300), (16, 16, 40, 4, 400), (3, 3, 8, 6, 60)]


@pytest.mark.parametrize("h,w,m,n,steps", CONFIGS)
def test_exact_equivalence(h, w, m, n, steps):
    seeds = [100 + 7 * i for i in range(n)]
    be = BatchedMinesweeperEnv(n, h, w, m)
    obs, info = be.reset(seeds)
    singles = [MinesweeperEnv(h, w, m) for _ in range(n)]
    for i, e in enumerate(singles):
        o, inf = e.reset(seed=seeds[i])
        np.testing.assert_array_equal(o, obs[i])
        np.testing.assert_array_equal(inf["action_mask"], info["action_mask"][i])
    trng = np.random.default_rng(12345)  # test-only stream, independent of env streams
    episodes = 0
    for _ in range(steps):
        mask = info["action_mask"]
        actions = np.where(mask, trng.random(mask.shape), -1.0).argmax(axis=1)
        for i in range(n):
            if trng.random() < 0.07 and (~mask[i]).any():
                actions[i] = trng.choice(np.flatnonzero(~mask[i]))
        obs, rew, term, trunc, info = be.step(actions)
        for i, e in enumerate(singles):
            o, r, t, tr, inf = e.step(int(actions[i]))
            assert r == rew[i] and t == term[i] and tr == trunc[i]
            assert inf["won"] == info["won"][i]
            assert inf["invalid_action"] == info["invalid_action"][i]
            assert info["episode_done"][i] == (t or tr)
            if t or tr:
                episodes += 1
                np.testing.assert_array_equal(o, info["final_obs"][i])
                np.testing.assert_array_equal(inf["action_mask"], info["final_mask"][i])
                o, inf = e.reset()  # no seed: continues the RNG stream
            np.testing.assert_array_equal(o, obs[i])
            np.testing.assert_array_equal(inf["action_mask"], info["action_mask"][i])
            if e.board.mines_placed:
                np.testing.assert_array_equal(e.board.mines, be.mines[i])
                np.testing.assert_array_equal(e.board.counts, be.counts[i])
    assert episodes > n  # multiple episodes per board on average


def test_seed_behaviour():
    def run(seeds):
        be = BatchedMinesweeperEnv(4, 9, 9, 10)
        be.reset(seeds)
        be.step(np.array([0, 10, 40, 80]))
        return be.mines.copy()

    a = run(5)
    np.testing.assert_array_equal(a, run(5))
    np.testing.assert_array_equal(a, run([5, 6, 7, 8]))
    assert not np.array_equal(a, run(500))
    assert not np.array_equal(a[0], a[1])  # different seeds within a batch -> different boards


def test_shapes_dtypes_onehot():
    be = BatchedMinesweeperEnv(5, 6, 7, 8)
    obs, info = be.reset(0)
    assert obs.shape == (5, 10, 6, 7) and obs.dtype == np.float32
    assert info["action_mask"].shape == (5, 42) and info["action_mask"].dtype == bool
    rng = np.random.default_rng(0)
    for _ in range(40):
        a = np.where(info["action_mask"], rng.random((5, 42)), -1).argmax(1)
        obs, r, te, tr, info = be.step(a)
        assert r.dtype == np.float32 and te.dtype == bool and tr.dtype == bool
        assert info["final_obs"].shape == obs.shape
        np.testing.assert_array_equal(obs.sum(axis=1), np.ones((5, 6, 7)))
        np.testing.assert_array_equal(obs[:, 0].reshape(5, -1).astype(bool), info["action_mask"])


def test_first_click_win_autoresets():
    be = BatchedMinesweeperEnv(3, 3, 3, 8)
    be.reset(1)
    obs, r, te, tr, info = be.step(np.array([0, 4, 8]))
    assert te.all() and not tr.any() and info["won"].all() and (r == 1.0).all()
    assert (info["final_obs"][:, 0].sum(axis=(1, 2)) == 8).all()  # 8 hidden mines
    assert (obs[:, 0] == 1).all() and (obs[:, 1:] == 0).all()  # fresh episodes
    assert info["action_mask"].all()
    assert (be.steps == 0).all() and not be.mines_placed.any()
    obs, r, te, tr, info = be.step(np.array([1, 1, 1]))  # works again
    assert te.all() and (r == 1.0).all()


def test_validation():
    with pytest.raises(ValueError):
        BatchedMinesweeperEnv(2, 3, 3, 9)
    with pytest.raises(ValueError):
        BatchedMinesweeperEnv(0)
    be = BatchedMinesweeperEnv(2, 3, 3, 1)
    with pytest.raises(RuntimeError):
        be.step(np.array([0, 0]))
    be.reset([1, 2])
    with pytest.raises(ValueError):
        be.step(np.array([0, 9]))
    with pytest.raises(ValueError):
        be.reset([1])
