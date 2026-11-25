"""How do the DQN's 9x9 losses happen? Classifies the final (fatal) move of lost games.

For every lost game on unseen seeds, checks the board state before the fatal click with the rule solver's
two deduction rules: if the solver could identify at least one certainly-safe hidden cell, the agent had a
risk-free move available and missed it ("avoidable by simple deduction"); otherwise the fatal click was a
guess by the rule solver's standards. Deeper logic (subset reasoning, global mine counting) is NOT checked,
so "guess" is an upper bound on the truly forced guesses.

    python scripts/analyze_losses.py [--games 10000] -> results/dqn/loss_analysis_9x9_10.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minesweeper_rl.agents.baselines import RuleAgent
from minesweeper_rl.agents.dqn import DQNAgent
from minesweeper_rl.env.gym_env import MinesweeperEnv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models" / "dqn_9x9_10.pt"))
    ap.add_argument("--games", type=int, default=10_000)
    ap.add_argument("--seed-base", type=int, default=5_000_000)
    ap.add_argument("--out", default=str(ROOT / "results" / "dqn" / "loss_analysis_9x9_10.json"))
    a = ap.parse_args()
    agent = DQNAgent(checkpoint=a.checkpoint)
    env = MinesweeperEnv(9, 9, 10)
    losses = avoidable = 0
    for i in range(a.games):
        obs, info = env.reset(seed=a.seed_base + i)
        while True:
            safe, _ = RuleAgent.deduce(obs)
            had_safe = bool((safe.reshape(-1) & info["action_mask"]).any())
            obs, _, term, trunc, info = env.step(agent.act(obs, info["action_mask"]))
            if term or trunc:
                break
        if not env.board.won:
            losses += 1
            avoidable += had_safe
    res = {"checkpoint": Path(a.checkpoint).name, "board": "9x9", "n_mines": 10, "n_games": a.games,
           "seed_base": a.seed_base, "losses": losses, "loss_rate": losses / a.games,
           "losses_with_rule_safe_move_available": avoidable,
           "losses_without_rule_safe_move": losses - avoidable,
           "share_without_rule_safe_move": (losses - avoidable) / max(losses, 1)}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
