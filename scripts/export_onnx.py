"""Export the trained Q-network to ONNX for the in-browser demo, plus golden fixtures for the JS tests.

    python scripts/export_onnx.py [--checkpoint models/dqn_9x9_10.pt] [--out web/model/dqn_9x9_10.onnx]

ONNX interface
    input  "obs": float32 (1, 10, H, W)   one-hot observation, exactly Board.observation()
    output "q":   float32 (1, H*W)        RAW Q-value per cell, row-major (index = row*W + col)
H and W are dynamic axes (the net is fully convolutional): the same file runs on 5x5, 9x9, 16x16.
Action masking is NOT in the graph; the browser applies it (revealed cell -> -inf, same as mask_q).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minesweeper_rl.agents.network import QNetwork
from minesweeper_rl.env.core import Board

ROOT = Path(__file__).resolve().parent.parent


def load_net(checkpoint: str | Path) -> QNetwork:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net = QNetwork(**ck["net"])
    net.load_state_dict(ck["state_dict"])
    return net.eval()


def export(net: QNetwork, out: Path, opset: int = 17) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 10, 9, 9)
    torch.onnx.export(
        net, (dummy,), str(out), input_names=["obs"], output_names=["q"], opset_version=opset,
        dynamic_axes={"obs": {2: "height", 3: "width"}, "q": {1: "cells"}},
        dynamo=False,
    )


# (height, width, n_mines, mine cells (row, col), clicks (row, col)): hand-set boards for the JS tests.
FIXTURE_BOARDS = [
    ("open 5x5, one click floods", 5, 5, [(0, 0), (4, 4), (2, 4)], [(2, 1), (0, 0)]),
    ("5x5 lose on second click", 5, 5, [(1, 1), (3, 3), (0, 4)], [(4, 0), (1, 1)]),
    ("3x3 win by clearing all safe cells", 3, 3, [(1, 1)], [(0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]),
    ("9x9 corner mines, big flood", 9, 9, [(0, 0), (0, 8), (8, 0), (8, 8), (4, 4), (4, 5), (5, 4), (5, 5), (2, 6), (6, 2)], [(0, 4), (3, 0), (8, 4)]),
    ("9x9 lose after partial reveal", 9, 9, [(0, 0), (1, 2), (3, 3), (4, 4), (5, 1), (6, 6), (7, 3), (8, 8), (2, 7), (8, 0)], [(4, 0), (7, 7), (3, 3)]),
    ("rectangular 4x6", 4, 6, [(0, 1), (2, 3), (3, 5)], [(3, 0), (0, 5), (1, 3), (0, 0)]),
]


def make_fixtures() -> list[dict]:
    out = []
    for name, h, w, mines, clicks in FIXTURE_BOARDS:
        b = Board(h, w, len(mines), np.random.default_rng(0))
        mask = np.zeros((h, w), bool)
        for r, c in mines:
            mask[r, c] = True
        # Mines are placed on the first click (as in the game); here we pre-place the hand-set layout,
        # which is what the JS test does too (placeMines).
        b.place_mines(mask)
        steps = []
        for r, c in clicks:
            if b.done:
                break
            res = b.reveal(r, c)
            steps.append({
                "click": [r, c],
                "newly_revealed": res.newly_revealed, "hit_mine": res.hit_mine, "won": res.won,
                "valid": res.valid,
                "obs": b.observation().argmax(axis=0).reshape(-1).tolist(),  # channel index per cell
                "mask": b.action_mask().astype(int).tolist(),
                "done": bool(b.done),
            })
        out.append({"name": name, "height": h, "width": w, "mines": [list(m) for m in mines],
                    "counts": b.counts.reshape(-1).tolist(), "steps": steps})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "models/dqn_9x9_10.pt"))
    p.add_argument("--out", default=str(ROOT / "web/model/dqn_9x9_10.onnx"))
    p.add_argument("--fixtures", default=str(ROOT / "web/tests/fixtures.json"))
    a = p.parse_args()
    net = load_net(a.checkpoint)
    out = Path(a.out)
    export(net, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")
    fx = Path(a.fixtures)
    fx.parent.mkdir(parents=True, exist_ok=True)
    fx.write_text(json.dumps({"boards": make_fixtures()}, separators=(",", ":")))
    print(f"wrote {fx}")


if __name__ == "__main__":
    main()
