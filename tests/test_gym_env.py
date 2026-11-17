import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import minesweeper_rl.env  # noqa: F401  (registers env)
from minesweeper_rl.env import MinesweeperEnv


def test_check_env():
    check_env(MinesweeperEnv(5, 5, 3))
    check_env(MinesweeperEnv(9, 9, 10))


def test_make():
    env = gymnasium.make("MinesweeperRL-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (10, 9, 9)
    assert info["action_mask"].all()


def test_registration_idempotent():
    import importlib

    importlib.reload(minesweeper_rl.env)


def test_spaces():
    env = MinesweeperEnv(4, 6, 3)
    assert env.action_space.n == 24
    assert env.observation_space.shape == (10, 4, 6)


def test_first_click_reward_zero_then_safe():
    env = MinesweeperEnv(9, 9, 10)
    env.reset(seed=0)
    _, r, term, trunc, info = env.step(40)
    assert r == 0.0 and not term and not info["invalid_action"]
    # Find a hidden safe cell.
    b = env.unwrapped.board
    rr, cc = np.argwhere(~b.revealed & ~b.mines)[0]
    _, r, term, _, info = env.step(int(rr * 9 + cc))
    assert r == pytest.approx(0.3) or (term and r == 1.0)


def test_mine_reward():
    env = MinesweeperEnv(5, 5, 3)
    env.reset(seed=0)
    env.step(12)
    b = env.unwrapped.board
    rr, cc = np.argwhere(b.mines)[0]
    _, r, term, _, info = env.step(int(rr * 5 + cc))
    assert r == -1.0 and term and not info["won"]


def test_win_reward_and_first_click_win():
    env = MinesweeperEnv(3, 3, 8)
    env.reset(seed=0)
    _, r, term, _, info = env.step(4)
    assert r == 1.0 and term and info["won"]

    env = MinesweeperEnv(5, 5, 3)
    env.reset(seed=1)
    env.step(12)
    b = env.unwrapped.board
    last = None
    for idx in np.argwhere(~b.revealed & ~b.mines):
        if b.revealed[idx[0], idx[1]]:
            continue
        last = env.step(int(idx[0] * 5 + idx[1]))
    obs, r, term, _, info = last
    assert term and info["won"] and r == 1.0


def test_invalid_action():
    env = MinesweeperEnv(5, 5, 3)
    env.reset(seed=0)
    obs1, *_ = env.step(12)
    obs2, r, term, trunc, info = env.step(12)
    assert r == 0.0 and not term and info["invalid_action"]
    assert (obs1 == obs2).all()


def test_truncation():
    env = MinesweeperEnv(5, 5, 3)
    env.reset(seed=0)
    env.step(12)
    trunc = False
    for _ in range(24):
        _, _, term, trunc, _ = env.step(12)
        if trunc:
            break
    assert trunc and not term


def test_determinism():
    def run(seed):
        env = MinesweeperEnv(9, 9, 10)
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for a in [40, 0, 80, 8, 72, 20, 55]:
            o, r, t, tr, _ = env.step(a)
            out.append((o, r, t))
            if t or tr:
                break
        return out

    a, b = run(5), run(5)
    for x, y in zip(a[1:], b[1:]):
        assert (x[0] == y[0]).all() and x[1:] == y[1:]
    boards = []
    for s in (0, 1, 2):
        env = MinesweeperEnv(9, 9, 10)
        env.reset(seed=s)
        env.step(40)
        boards.append(env.unwrapped.board.mines.copy())
    assert any((boards[0] != x).any() for x in boards[1:])


def test_render_ansi():
    env = MinesweeperEnv(5, 5, 3, render_mode="ansi")
    env.reset(seed=0)
    assert "#" in env.render()


def test_random_agent_info_keys():
    env = MinesweeperEnv(5, 5, 3)
    obs, info = env.reset(seed=0)
    for k in ("action_mask", "won", "revealed", "invalid_action"):
        assert k in info
    for _ in range(100):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            break


def test_invalid_params():
    with pytest.raises(ValueError):
        MinesweeperEnv(5, 5, 0)
    with pytest.raises(ValueError):
        MinesweeperEnv(5, 5, 25)
