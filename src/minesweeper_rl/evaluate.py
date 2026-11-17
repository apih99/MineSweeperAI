"""Fixed-seed evaluation harness.

CLI: python -m minesweeper_rl.evaluate --agent rule --height 9 --width 9 --mines 10 --games 10000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from minesweeper_rl.agents.base import Agent
from minesweeper_rl.agents.baselines import RandomAgent, RuleAgent
from minesweeper_rl.env.gym_env import MinesweeperEnv

# Reserved for evaluation: game i uses env seed EVAL_SEED_BASE + i. Training
# must use different (non-overlapping) seeds.
EVAL_SEED_BASE = 1_000_000

# Registry of agent factories; a checkpoint-based DQN agent can be added here later.
AGENTS: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "rule": RuleAgent,
}


def wilson_interval(wins: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def evaluate(
    agent: Agent,
    height: int,
    width: int,
    n_mines: int,
    n_games: int,
    seed_base: int = EVAL_SEED_BASE,
) -> dict:
    env = MinesweeperEnv(height, width, n_mines)
    n_safe = height * width - n_mines
    wins = 0
    frac_sum = 0.0
    steps_sum = 0
    t0 = time.perf_counter()
    for i in range(n_games):
        seed = seed_base + i
        obs, info = env.reset(seed=seed)
        agent.reset(np.random.default_rng(seed))
        steps = 0
        while True:
            action = agent.act(obs, info["action_mask"])
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        b = env.board
        won = bool(b.won)
        wins += won
        # On a loss, the exploded mine cell is revealed but is not a safe cell.
        safe_revealed = int(b.revealed.sum()) - (0 if won else 1)
        frac_sum += safe_revealed / n_safe
        steps_sum += steps
    seconds = time.perf_counter() - t0
    return {
        "agent": agent.name,
        "board": f"{height}x{width}",
        "n_mines": n_mines,
        "n_games": n_games,
        "wins": wins,
        "win_rate": wins / n_games,
        "win_rate_ci95": wilson_interval(wins, n_games),
        "mean_safe_revealed_frac": frac_sum / n_games,
        "mean_steps": steps_sum / n_games,
        "seconds": seconds,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", choices=sorted(AGENTS), required=True)
    p.add_argument("--height", type=int, default=9)
    p.add_argument("--width", type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--games", type=int, default=10_000)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args(argv)
    res = evaluate(AGENTS[a.agent](), a.height, a.width, a.mines, a.games)
    lo, hi = res["win_rate_ci95"]
    print(f"agent={res['agent']} board={res['board']} mines={res['n_mines']} games={res['n_games']}")
    print(f"  win rate      : {100 * res['win_rate']:.2f}%  (95% CI {100 * lo:.2f}-{100 * hi:.2f}%)")
    print(f"  safe revealed : {100 * res['mean_safe_revealed_frac']:.2f}%")
    print(f"  mean steps    : {res['mean_steps']:.2f}")
    print(f"  time          : {res['seconds']:.1f}s")
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
