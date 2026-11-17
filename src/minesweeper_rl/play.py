"""Play Minesweeper in the terminal: python -m minesweeper_rl.play"""

from __future__ import annotations

import argparse

import numpy as np

from minesweeper_rl.env.core import Board
from minesweeper_rl.env.gym_env import R_FIRST, R_LOSS, R_SAFE, R_WIN


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Play Minesweeper in the terminal.")
    p.add_argument("--height", type=int, default=9)
    p.add_argument("--width", type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args(argv)
    try:
        board = Board(a.height, a.width, a.mines, np.random.default_rng(a.seed))
    except ValueError as e:
        p.error(str(e))

    print("Enter 'row col' to reveal a cell, 'q' to quit.")
    total = 0.0
    while not board.done:
        print(board.render_text())
        try:
            line = input("> ").strip()
        except EOFError:
            print("\nBye.")
            return 0
        if line.lower() in ("q", "quit", "exit"):
            print("Bye.")
            return 0
        parts = line.replace(",", " ").split()
        try:
            if len(parts) != 2:
                raise ValueError
            r, c = int(parts[0]), int(parts[1])
        except ValueError:
            print("Please enter two integers: row col")
            continue
        if not (0 <= r < board.height and 0 <= c < board.width):
            print(f"Out of range: row 0-{board.height - 1}, col 0-{board.width - 1}")
            continue
        res = board.reveal(r, c)
        if not res.valid:
            print("Cell already revealed.")
            continue
        if res.hit_mine:
            reward = R_LOSS
        elif res.won:
            reward = R_WIN
        elif res.first_click:
            reward = R_FIRST
        else:
            reward = R_SAFE
        total += reward
        print(f"revealed {res.newly_revealed} cell(s), reward {reward:+.1f}, total {total:+.1f}")

    print(board.render_text(show_mines=True))
    print("You won!" if board.won else "Boom! You hit a mine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
