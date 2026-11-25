"""Collect M8 experiment outputs into committed files and a summary.

    python scripts/experiments_collect.py DOWNLOAD_DIR [DOWNLOAD_DIR ...] [--out results/experiments]

DOWNLOAD_DIR is a ``kaggle kernels output`` directory (runs/<name>/metrics.csv, results/<name>/*.json).
For every run found it copies metrics.csv -> <out>/metrics/<name>.csv and the unseen-seed result JSONs
-> <out>/<name>_<best|final>_unseen.json, then writes <out>/summary_<prefix>.json with, per run:
best/final unseen win rate (+95% CI) and the late-training stability (mean/std of the 1,000-game eval
win rate over the last 25% of evals). The std is across evals of ONE run (it mixes real checkpoint-to-
checkpoint swings with the ~1.4-point binomial noise of a 1,000-game eval).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


def late_stats(csv_path: Path, frac: float = 0.25) -> dict:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    wr = np.array([float(r["eval_win_rate"]) for r in rows])
    k = max(2, int(round(len(wr) * frac)))
    tail = wr[-k:]
    return {"n_evals": len(wr), "late_k": k, "late_mean": float(tail.mean()), "late_std": float(tail.std(ddof=1)),
            "late_min": float(tail.min()), "late_max": float(tail.max()), "last_eval": float(wr[-1]),
            "max_eval": float(wr.max()),
            "steps_per_s_median": float(np.median([float(r["board_steps_per_s"]) for r in rows])),
            "wall_s": float(rows[-1]["wall_s"]), "board_steps": int(float(rows[-1]["board_steps"]))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", default="results/experiments")
    p.add_argument("--prefix", default="exp_9x9_", help="only runs whose name starts with this")
    p.add_argument("--summary", default="summary_9x9.json")
    a = p.parse_args()
    out = Path(a.out)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    summary = {}
    for d in map(Path, a.dirs):
        for run in sorted((d / "runs").glob("*")) if (d / "runs").exists() else []:
            if not run.name.startswith(a.prefix) or not (run / "metrics.csv").exists():
                continue
            name = run.name
            shutil.copy(run / "metrics.csv", out / "metrics" / f"{name}.csv")
            ent = {"late": late_stats(run / "metrics.csv"), "finished": (run / "final.pt").exists()}
            for which in ("best", "final"):
                src = d / "results" / name / f"{name}_{which}_unseen.json"
                if src.exists():
                    shutil.copy(src, out / f"{name}_{which}_unseen.json")
                    r = json.loads(src.read_text())
                    ent[which] = {"win_rate": r["win_rate"], "ci95": r["win_rate_ci95"], "wins": r["wins"],
                                  "n_games": r["n_games"]}
            summary[name] = ent
    (out / a.summary).write_text(json.dumps(summary, indent=2))
    for n, e in summary.items():
        b, f_, lt = e.get("best"), e.get("final"), e["late"]
        print(f"{n:28s} best {b['win_rate']:.4f}  final {f_['win_rate']:.4f}  late {lt['late_mean']:.3f}+-{lt['late_std']:.3f}"
              if b and f_ else f"{n}: incomplete")


if __name__ == "__main__":
    main()
