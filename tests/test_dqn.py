import numpy as np
import pytest
import torch

from minesweeper_rl.agents.dqn import (
    Batch,
    DQNAgent,
    ReplayBuffer,
    codes_to_onehot,
    double_dqn_targets,
    dqn_loss,
    epsilon_greedy_actions,
    make_checkpoint,
    make_streams,
    obs_to_codes,
)
from minesweeper_rl.agents.network import QNetwork
from minesweeper_rl.env.batched import BatchedMinesweeperEnv
from minesweeper_rl.evaluate import evaluate


class Stub(torch.nn.Module):
    """Returns a fixed (B, HW) Q row regardless of input."""

    def __init__(self, row):
        super().__init__()
        self.row = torch.tensor(row, dtype=torch.float32)
        self.w = torch.nn.Parameter(torch.zeros(1))  # so it has a parameter

    def forward(self, x):
        return self.row[None].repeat(x.shape[0], 1) + self.w


def _batch(done, next_mask, reward=0.5, action=0):
    n = len(done)
    return Batch(
        torch.zeros(n, 10, 2, 2),
        torch.full((n,), action, dtype=torch.long),
        torch.full((n,), reward),
        torch.zeros(n, 10, 2, 2),
        torch.tensor(next_mask),
        torch.tensor(done),
    )


def test_double_dqn_online_selects_target_evaluates():
    online = Stub([1.0, 5.0, 3.0, 9.0])  # best legal (cells 0..2): index 1
    target = Stub([100.0, 7.0, 200.0, 300.0])
    b = _batch([False], [[True, True, True, False]])
    y = double_dqn_targets(online, target, b, gamma=0.5)
    assert y.item() == pytest.approx(0.5 + 0.5 * 7.0)  # not 200 (target's own max) nor 9


def test_done_target_is_reward_and_allfalse_mask_finite():
    online = Stub([1.0, 5.0, 3.0, 9.0])
    target = Stub([100.0, 7.0, 200.0, 300.0])
    b = _batch([True, True], [[False] * 4, [True] * 4], reward=-1.0)
    y = double_dqn_targets(online, target, b, gamma=0.9)
    assert torch.allclose(y, torch.tensor([-1.0, -1.0]))
    # full loss + backward stays finite with an all-False next mask on a done transition
    net = QNetwork(channels=4, n_layers=1)
    b2 = Batch(torch.zeros(2, 10, 2, 2), torch.tensor([0, 1]), torch.tensor([1.0, -1.0]),
               torch.zeros(2, 10, 2, 2), torch.zeros(2, 4, dtype=torch.bool), torch.tensor([True, True]))
    loss, _ = dqn_loss(net, QNetwork(channels=4, n_layers=1), b2, 0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_replay_roundtrip_and_circular():
    env = BatchedMinesweeperEnv(4, 5, 5, 3)
    obs, info = env.reset(seeds=7)
    rng = np.random.default_rng(0)
    for _ in range(6):
        codes = obs_to_codes(obs)
        back = codes_to_onehot(torch.from_numpy(codes)).numpy()
        assert np.array_equal(back, obs)
        a = np.argmax(np.where(info["action_mask"], rng.random((4, 25)), -1), axis=1)
        obs, _, _, _, info = env.step(a)

    buf = ReplayBuffer(10, 5, 5, np.random.default_rng(0))
    for k in range(7):  # 28 transitions into capacity 10
        buf.add_batch(np.full((4, 5, 5), k % 10, np.uint8), np.full(4, k), np.zeros(4),
                      np.zeros((4, 5, 5), np.uint8), np.ones((4, 25), bool), np.zeros(4, bool))
    assert len(buf) == 10 and buf.pos == 28 % 10
    assert set(buf.action.tolist()) <= {4, 5, 6}  # oldest overwritten
    s = buf.sample(6)
    assert s.state.shape == (6, 10, 5, 5) and s.next_mask.shape == (6, 25)
    assert s.action.dtype == torch.long and s.done.dtype == torch.bool


def test_epsilon_greedy_always_legal():
    net = QNetwork(channels=4, n_layers=1)
    rng = np.random.default_rng(1)
    for eps in (0.0, 0.5, 1.0):
        for _ in range(50):
            mask = rng.random((32, 25)) < 0.3
            mask[:, 3] = True
            obs = np.zeros((32, 10, 5, 5), np.float32)
            a = epsilon_greedy_actions(net, obs, mask, eps, rng)
            assert mask[np.arange(32), a].all()


def test_streams_independent_and_deterministic():
    a1, b1, t1 = make_streams(3)
    a2, b2, t2 = make_streams(3)
    assert a1.random() == a2.random() and b1.random() == b2.random() and t1 == t2
    a3, b3, _ = make_streams(3)
    assert a3.random() != b3.random()


def test_agent_roundtrip_any_board_size(tmp_path):
    net = QNetwork(channels=8, n_layers=2)
    path = tmp_path / "m.pt"
    torch.save(make_checkpoint(net, {"channels": 8, "n_layers": 2}), path)
    agent = DQNAgent.load(str(path))
    for h, w, m in [(5, 5, 3), (9, 9, 10)]:
        res = evaluate(agent, h, w, m, 5)
        assert res["n_games"] == 5
