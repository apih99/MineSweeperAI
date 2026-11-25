import numpy as np
import pytest
from PIL import Image

from minesweeper_rl.agents.dqn import DQNAgent
from minesweeper_rl.agents.network import QNetwork
from minesweeper_rl.env.gym_env import MinesweeperEnv
from minesweeper_rl.render import DPI, FIG_H, FIG_W, play_frames, render_frame, write_gif


@pytest.fixture(scope="module")
def agent():
    import torch

    torch.manual_seed(0)
    return DQNAgent(net=QNetwork(channels=8, n_layers=2).eval())


def test_q_values_masked_and_consistent_with_act(agent):
    env = MinesweeperEnv(5, 5, 3)
    obs, info = env.reset(seed=1)
    obs, _, _, _, info = env.step(12)
    q = agent.q_values(obs, info["action_mask"])
    assert q.shape == (5, 5)
    assert np.isneginf(q[env.board.revealed]).all()
    assert np.isfinite(q[~env.board.revealed]).all()
    assert int(np.argmax(q)) == agent.act(obs, info["action_mask"])


def test_render_frame_size_and_content(agent):
    env = MinesweeperEnv(5, 5, 3)
    obs, info = env.reset(seed=2)
    q = agent.q_values(obs, info["action_mask"])
    img = render_frame(env.board, q, (2, 2), move=1)
    assert isinstance(img, Image.Image)
    assert img.size == (int(FIG_W * DPI), int(FIG_H * DPI))
    assert np.asarray(img).std() > 5  # not blank


def _find_games(agent, n_seeds=300):
    found = {}
    for seed in range(n_seeds):
        frames, won, n = play_frames(agent, 4, 4, 2, seed)
        found.setdefault(won, (seed, frames, n))
        if len(found) == 2:
            break
    return found


def test_play_frames_and_gif_win_and_loss(agent, tmp_path):
    found = _find_games(agent)
    assert set(found) == {True, False}, "expected both a win and a loss within 300 tiny games"
    for won, (seed, frames, n) in found.items():
        assert len(frames) == n + 1  # one frame per move + the final-state frame
        out = write_gif(frames, tmp_path / f"g_{won}.gif", fps=4, hold_s=0.5)
        assert out.stat().st_size > 0
        im = Image.open(out)
        assert im.n_frames == len(frames)
