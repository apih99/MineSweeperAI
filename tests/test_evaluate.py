from minesweeper_rl.agents.baselines import RandomAgent, RuleAgent
from gymnasium.utils import seeding

from minesweeper_rl.evaluate import EVAL_SEED_BASE, agent_rng, evaluate, wilson_interval


def _strip(d):
    return {k: v for k, v in d.items() if k != "seconds"}


def test_deterministic():
    for cls in (RandomAgent, RuleAgent):
        a = evaluate(cls(), 5, 5, 3, 50)
        b = evaluate(cls(), 5, 5, 3, 50)
        assert _strip(a) == _strip(b)


def test_rule_beats_random():
    r = evaluate(RandomAgent(), 5, 5, 3, 300)
    s = evaluate(RuleAgent(), 5, 5, 3, 300)
    assert s["win_rate"] > r["win_rate"] + 0.2
    assert s["mean_safe_revealed_frac"] > r["mean_safe_revealed_frac"]


def test_result_fields():
    r = evaluate(RandomAgent(), 5, 5, 3, 20)
    lo, hi = r["win_rate_ci95"]
    assert lo <= r["win_rate"] <= hi
    assert r["board"] == "5x5" and r["n_games"] == 20 and r["n_mines"] == 3
    assert EVAL_SEED_BASE == 1_000_000
    assert 0 <= r["mean_safe_revealed_frac"] <= 1


def test_wilson():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and 0 < hi < 0.05


def test_agent_rng_independent_of_env_rng():
    for seed in (0, 1, EVAL_SEED_BASE, EVAL_SEED_BASE + 7):
        env_rng, _ = seeding.np_random(seed)
        assert not (agent_rng(seed).random(8) == env_rng.random(8)).all()


def test_rule_agent_win_rate_5x5_regression():
    # With the correlated-RNG bug this was ~0.37; with independent streams ~0.69.
    r = evaluate(RuleAgent(), 5, 5, 3, 1000)
    assert r["win_rate"] > 0.55
