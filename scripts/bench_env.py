"""Throughput benchmark: single MinesweeperEnv loop vs BatchedMinesweeperEnv (9x9 / 10 mines).

Agent compute is excluded. Two action policies, both picked with masked random scores + argmax:
  * "random": random legal (unrevealed) cell. Episodes are very short (a random agent hits a mine
    after ~3-4 clicks), so mine placement (one Generator.choice per episode) is a large share.
  * "safe-peek": random unrevealed NON-MINE cell (peeks at the mines; benchmark only). Gives long
    episodes like a competent agent would, so it shows steady-state stepping throughput.
Timing uses process CPU time (robust to other programs competing for the laptop); each number is
the best of --repeats runs.
Usage: python scripts/bench_env.py [--seconds 3] [--repeats 3] [--profile]
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import numpy as np

from minesweeper_rl.env.batched import BatchedMinesweeperEnv
from minesweeper_rl.env.gym_env import MinesweeperEnv

H, W, M = 9, 9, 10


def bench_single(seconds: float, peek: bool) -> float:
    env = MinesweeperEnv(H, W, M)
    rng = np.random.default_rng(0)
    _, info = env.reset(seed=0)
    steps, t0 = 0, time.process_time()
    while time.process_time() - t0 < seconds:
        for _ in range(200):
            ok = info["action_mask"]
            if peek and env.board.mines_placed:
                ok = ok & ~env.board.mines.reshape(-1)
            a = int(np.where(ok, rng.random(H * W), -1.0).argmax())
            _, _, te, tr, info = env.step(a)
            steps += 1
            if te or tr:
                _, info = env.reset()
    return steps / (time.process_time() - t0)


def bench_batched(n: int, seconds: float, peek: bool) -> float:
    env = BatchedMinesweeperEnv(n, H, W, M)
    rng = np.random.default_rng(0)
    _, info = env.reset(0)
    steps, t0 = 0, time.process_time()
    while time.process_time() - t0 < seconds:
        for _ in range(20):
            ok = info["action_mask"]
            if peek:
                ok = ok & ~(env.mines_placed[:, None] & env.mines.reshape(n, H * W))
            a = np.where(ok, rng.random((n, H * W)), -1.0).argmax(axis=1)
            _, _, _, _, info = env.step(a)
            steps += n
    return steps / (time.process_time() - t0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", action="store_true", help="cProfile the N=256 batched run")
    args = p.parse_args()
    print(f"{H}x{W}/{M} mines, best of {args.repeats} x {args.seconds:.0f}s per config")
    for peek in (False, True):
        print(f"\npolicy: {'safe-peek' if peek else 'random'}")
        print(f"{'config':<14}{'steps/sec':>12}{'speedup':>10}")
        base = max(bench_single(args.seconds, peek) for _ in range(args.repeats))
        print(f"{'single':<14}{base:>12,.0f}{1.0:>9.1f}x")
        for n in (64, 128, 256):
            sps = max(bench_batched(n, args.seconds, peek) for _ in range(args.repeats))
            print(f"{'batched N=' + str(n):<14}{sps:>12,.0f}{sps / base:>9.1f}x")
    if args.profile:
        pr = cProfile.Profile()
        pr.runcall(bench_batched, 256, 3.0, False)
        pstats.Stats(pr).sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
