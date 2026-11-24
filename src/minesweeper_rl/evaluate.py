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
import torch

from minesweeper_rl.agents.base import Agent
from minesweeper_rl.agents.baselines import RandomAgent, RuleAgent
from minesweeper_rl.agents.dqn import DQNAgent
from minesweeper_rl.agents.network import masked_argmax
from minesweeper_rl.env.batched import BatchedMinesweeperEnv
from minesweeper_rl.env.gym_env import MinesweeperEnv

# Reserved for evaluation: game i uses env seed EVAL_SEED_BASE + i. Training
# must use different (non-overlapping) seeds.
EVAL_SEED_BASE = 1_000_000

# Agent RNG stream id. Gymnasium seeds its env RNG as Generator(PCG64(SeedSequence(seed))),
# which is bit-identical to np.random.default_rng(seed). Seeding the agent with the plain
# seed would hand it the same draws the env uses for mine placement, correlating the
# agent's random clicks with mine locations. Mixing in this constant gives an independent stream.
AGENT_STREAM = 0xA9E7


def agent_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, AGENT_STREAM])


# Registry of agent factories. "dqn" additionally needs ``checkpoint=<path>``.
AGENTS: dict[str, Callable[..., Agent]] = {
    "random": RandomAgent,
    "rule": RuleAgent,
    "dqn": DQNAgent,
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
        agent.reset(agent_rng(seed))
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


def evaluate_batched(
    net_or_agent,
    height: int,
    width: int,
    n_mines: int,
    n_games: int,
    seed_base: int = EVAL_SEED_BASE,
    batch: int = 256,
) -> dict:
    """Greedy, masked evaluation of a Q-network on many boards at once.

    Same result as ``evaluate(DQNAgent(net), ...)``: game ``i`` is played on the board that
    ``MinesweeperEnv.reset(seed=seed_base + i)`` would produce (each board's generator is
    seeded with ``seed_base + i`` and its mines are placed on the first click). Games run in
    chunks of ``batch``; a board's first finish is recorded and its later auto-reset games
    (unseeded continuation) are ignored. The policy uses no randomness.
    """
    net = getattr(net_or_agent, "net", net_or_agent)
    dev = next(net.parameters()).device
    n_safe = height * width - n_mines
    hw = height * width
    env = BatchedMinesweeperEnv(max(1, min(batch, n_games)), height, width, n_mines)
    B = env.n_envs
    wins = steps_sum = 0
    frac_sum = 0.0
    t0 = time.perf_counter()
    for c0 in range(0, n_games, B):
        k = min(B, n_games - c0)
        # padding boards (j >= k) get unused seeds past the end of the range; their results are ignored
        obs, info = env.reset(seeds=[seed_base + c0 + j for j in range(B)])
        mask = info["action_mask"]
        alive = np.arange(B) < k
        nsteps = np.zeros(B, dtype=np.int64)
        while alive.any():
            with torch.no_grad():
                q = net(torch.from_numpy(obs).to(dev))
                a = masked_argmax(q, torch.from_numpy(mask).to(dev)).cpu().numpy()
            obs, _, _, _, info = env.step(a)
            mask = info["action_mask"]
            nsteps += alive
            fin = np.flatnonzero(info["episode_done"] & alive)
            if len(fin):
                won = info["won"][fin]
                revealed = hw - info["final_obs"][fin, 0].reshape(len(fin), hw).sum(axis=1)  # channel 0 = hidden
                safe_revealed = revealed.astype(np.int64) - np.where(won, 0, 1)
                wins += int(won.sum())
                frac_sum += float((safe_revealed / n_safe).sum())
                steps_sum += int(nsteps[fin].sum())
                alive[fin] = False
    seconds = time.perf_counter() - t0
    return {
        "agent": getattr(net_or_agent, "name", "dqn"),
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
    p.add_argument("--checkpoint", type=str, default=None, help="model .pt (required for --agent dqn)")
    a = p.parse_args(argv)
    if a.agent == "dqn":
        if not a.checkpoint:
            p.error("--agent dqn requires --checkpoint")
        agent = AGENTS["dqn"](checkpoint=a.checkpoint)
        res = evaluate_batched(agent, a.height, a.width, a.mines, a.games)  # same numbers as evaluate(), much faster
    else:
        agent = AGENTS[a.agent]()
        res = evaluate(agent, a.height, a.width, a.mines, a.games)
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
