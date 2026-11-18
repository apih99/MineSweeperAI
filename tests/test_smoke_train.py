import pytest

from minesweeper_rl.agents.dqn import DQNAgent
from minesweeper_rl.evaluate import evaluate
from minesweeper_rl.train import TrainConfig, train

RANDOM_WIN_RATE = 0.088


@pytest.mark.slow
def test_smoke_train_beats_random(tmp_path):
    cfg = TrainConfig(
        name="smoke", out_dir=str(tmp_path), seed=0, channels=32, n_layers=4,
        total_steps=SMOKE_STEPS, eval_every=SMOKE_STEPS, eval_games=50,
        learning_starts=2_000, target_update_every=200, updates_per_step=2,
        eps_decay_frac=0.3, verbose=False,
    )
    train(cfg)
    agent = DQNAgent.load(str(tmp_path / "smoke" / "final.pt"))
    res = evaluate(agent, 5, 5, 3, 300)
    assert res["win_rate"] > 0.20, res["win_rate"]
    assert res["win_rate"] > 2 * RANDOM_WIN_RATE


SMOKE_STEPS = 30_000
