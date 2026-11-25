"""Evaluate the best.pt / final.pt of one or more run directories on a block of seeds.

    python scripts/eval_run.py RUN_DIR [RUN_DIR ...] --out results/experiments/x [--games 10000]
        [--seed-base 5000000] [--device cpu|cuda] [--which best final]

Writes <out>/<run name>_<which>_<tag>.json (tag = "unseen" for seed base 5,000,000, else "seeds<base>").
The board comes from the run's config.yaml. Also usable on a bare checkpoint with --checkpoint/--board.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from minesweeper_rl.agents.dqn import DQNAgent
from minesweeper_rl.evaluate import evaluate_batched

UNSEEN_SEED_BASE = 5_000_000


def tag_for(seed_base: int) -> str:
    return "unseen" if seed_base == UNSEEN_SEED_BASE else f"seeds{seed_base}"


def eval_checkpoint(path: Path, h: int, w: int, m: int, games: int, seed_base: int, device: str) -> dict:
    agent = DQNAgent(checkpoint=str(path))
    agent.net.to(device).eval()
    res = evaluate_batched(agent, h, w, m, games, seed_base=seed_base)
    res.update(checkpoint=path.name, seed_base=seed_base)
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="*")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=10_000)
    p.add_argument("--seed-base", type=int, default=UNSEEN_SEED_BASE)
    p.add_argument("--device", default="cpu")
    p.add_argument("--which", nargs="+", default=["best", "final"])
    p.add_argument("--checkpoint", help="evaluate this bare .pt instead of run dirs (needs --board, --label)")
    p.add_argument("--board", help="HxW/M for --checkpoint, e.g. 16x16/40")
    p.add_argument("--label")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.checkpoint:
        hw, m = a.board.split("/")
        h, w = (int(x) for x in hw.split("x"))
        jobs.append((a.label, Path(a.checkpoint), h, w, int(m)))
    for rd in a.run_dirs:
        rd = Path(rd)
        cfg = yaml.safe_load((rd / "config.yaml").read_text())
        for which in a.which:
            ck = rd / f"{which}.pt"
            if ck.exists():
                jobs.append((f"{rd.name}_{which}", ck, cfg["height"], cfg["width"], cfg["n_mines"]))
            else:
                print(f"{ck} missing -> skipped", flush=True)
    for label, ck, h, w, m in jobs:
        res = eval_checkpoint(ck, h, w, m, a.games, a.seed_base, a.device)
        (out / f"{label}_{tag_for(a.seed_base)}.json").write_text(json.dumps(res, indent=2))
        lo, hi = res["win_rate_ci95"]
        print(f"{label:40s} {h}x{w}/{m} seeds {a.seed_base}+: win {100 * res['win_rate']:.2f}% "
              f"(95% CI {100 * lo:.2f}-{100 * hi:.2f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
