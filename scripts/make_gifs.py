"""Regenerate the README GIFs in docs/assets/ (first win / first loss on unseen seeds >= 5,000,000).

    python scripts/make_gifs.py            # scans seeds upward from 5,000,000, prints the seeds used
"""
from __future__ import annotations

import argparse
from pathlib import Path

from minesweeper_rl.agents.dqn import DQNAgent
from minesweeper_rl.env.gym_env import MinesweeperEnv
from minesweeper_rl.render import play_frames, write_gif

ROOT = Path(__file__).resolve().parents[1]
UNSEEN = 5_000_000


def outcome(agent: DQNAgent, h: int, w: int, m: int, seed: int) -> bool:
    env = MinesweeperEnv(h, w, m)
    obs, info = env.reset(seed=seed)
    while True:
        obs, _, term, trunc, info = env.step(agent.act(obs, info["action_mask"]))
        if term or trunc:
            return bool(env.board.won)


def first_seed(agent, h, w, m, want_win: bool, start: int = UNSEEN) -> int:
    seed = start
    while outcome(agent, h, w, m, seed) != want_win:
        seed += 1
    return seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models" / "dqn_9x9_10.pt"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets"))
    a = ap.parse_args()
    agent = DQNAgent(checkpoint=a.checkpoint)
    out = Path(a.out)
    jobs = [  # name, (h, w, mines), want_win, fps
        ("win_9x9", (9, 9, 10), True, 2.0),
        ("loss_9x9", (9, 9, 10), False, 2.0),
        ("zero_shot_win_16x16", (16, 16, 40), True, 4.0),
    ]
    for name, (h, w, m), want_win, fps in jobs:
        seed = first_seed(agent, h, w, m, want_win)
        frames, won, n = play_frames(agent, h, w, m, seed)
        assert won == want_win
        path = write_gif(frames, out / f"{name}.gif", fps=fps, hold_s=3.0)
        print(f"{name}: seed {seed}, {'WON' if won else 'LOST'} in {n} moves, {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
