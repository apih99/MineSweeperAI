import numpy as np
import pytest
import torch

from minesweeper_rl.agents.dqn import DQNAgent
from minesweeper_rl.agents.network import QNetwork
from minesweeper_rl.env.batched import BatchedMinesweeperEnv
from minesweeper_rl.evaluate import evaluate, evaluate_batched


def _agent(seed, channels=16, n_layers=3):
    torch.manual_seed(seed)
    return DQNAgent(QNetwork(channels=channels, n_layers=n_layers))


@pytest.mark.parametrize(
    "h,w,m,n,batch,seed_base",
    [(5, 5, 3, 300, 64, 1_000_000), (9, 9, 10, 60, 16, 1_000_000), (5, 5, 3, 50, 256, 123)],
)
def test_batched_matches_evaluate(h, w, m, n, batch, seed_base):
    agent = _agent(1)
    a = evaluate(agent, h, w, m, n, seed_base=seed_base)
    b = evaluate_batched(agent, h, w, m, n, seed_base=seed_base, batch=batch)
    assert a["wins"] == b["wins"]
    assert a["mean_steps"] == b["mean_steps"]
    assert a["mean_safe_revealed_frac"] == pytest.approx(b["mean_safe_revealed_frac"], abs=1e-12)
    assert a["win_rate_ci95"] == b["win_rate_ci95"]
    assert b["n_games"] == n


def test_batched_accepts_bare_net_and_is_deterministic():
    net = _agent(2).net
    r1 = evaluate_batched(net, 5, 5, 3, 40, batch=16)
    r2 = evaluate_batched(net, 5, 5, 3, 40, batch=7)  # chunking must not matter
    assert (r1["wins"], r1["mean_steps"]) == (r2["wins"], r2["mean_steps"])


def test_env_state_roundtrip_continues_exactly():
    rng = np.random.default_rng(0)

    def run(env, obs, mask, n):
        outs = []
        for _ in range(n):
            a = np.argmax(np.where(mask, rng.random(mask.shape), -1), axis=1)
            obs, r, te, tr, info = env.step(a)
            mask = info["action_mask"]
            outs.append((obs, r, te, tr))
        return outs

    e1 = BatchedMinesweeperEnv(8, 5, 5, 3)
    obs, info = e1.reset(seeds=5)
    run(e1, obs, info["action_mask"], 7)
    snap = e1.get_state()
    e2 = BatchedMinesweeperEnv(8, 5, 5, 3)
    e2.reset(seeds=999)
    e2.set_state(snap)
    o1, m1 = e1.current()
    o2, m2 = e2.current()
    assert np.array_equal(o1, o2) and np.array_equal(m1, m2)
    state = rng.bit_generator.state
    out1 = run(e1, o1, m1, 40)
    rng.bit_generator.state = state
    out2 = run(e2, o2, m2, 40)
    for x, y in zip(out1, out2):
        for u, v in zip(x, y):
            assert np.array_equal(u, v)
