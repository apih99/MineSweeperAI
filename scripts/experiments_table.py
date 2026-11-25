"""Print the markdown tables for the README's Experiments section from results/experiments/.

    python scripts/experiments_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1] / "results" / "experiments"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(r: dict) -> str:
    lo, hi = r["ci95"] if "ci95" in r else r["win_rate_ci95"]
    return f"{pct(r['win_rate'])} ({pct(lo)}-{pct(hi)})"


def table(summary: dict, variants: list[str], label: str) -> None:
    print(f"\n#### {label}: per run\n")
    print("| Run | best.pt, 10k unseen | final.pt, 10k unseen | late evals (last 25%): mean +- std | min-max |")
    print("|---|---|---|---|---|")
    for v in variants:
        for n in sorted(k for k in summary if k.rsplit("_s", 1)[0].endswith("_" + v) or k.rsplit("_s", 1)[0] == v):
            e = summary[n]
            late = e["late"]
            print(f"| {n.split('exp_')[1]} | {ci(e['best'])} | {ci(e['final'])} | "
                  f"{pct(late['late_mean'])} +- {pct(late['late_std'])} | {pct(late['late_min'])}-{pct(late['late_max'])} |")
    print(f"\n#### {label}: per variant (mean of seeds; seed values in brackets)\n")
    print("| Variant | best.pt | final.pt | late-eval mean | late-eval std (mean of seeds) |")
    print("|---|---|---|---|---|")
    for v in variants:
        runs = [e for n, e in summary.items() if n.rsplit("_s", 1)[0].endswith("_" + v)]
        if not runs:
            continue
        b = [r["best"]["win_rate"] for r in runs]
        f = [r["final"]["win_rate"] for r in runs]
        lm = [r["late"]["late_mean"] for r in runs]
        ls = [r["late"]["late_std"] for r in runs]
        fmt = lambda xs: f"{pct(np.mean(xs))} ({', '.join(pct(x) for x in xs)})"  # noqa: E731
        print(f"| {v} | {fmt(b)} | {fmt(f)} | {pct(np.mean(lm))} | {pct(np.mean(ls))} |")


def main() -> None:
    a = EXP / "summary_9x9.json"
    if a.exists():
        table(json.loads(a.read_text()), ["baseline", "lr_decay", "polyak", "both"], "Experiment A (9x9)")
    b = EXP / "summary_16x16.json"
    if b.exists():
        table(json.loads(b.read_text()), ["finetune", "scratch"], "Experiment B (16x16)")


if __name__ == "__main__":
    main()
