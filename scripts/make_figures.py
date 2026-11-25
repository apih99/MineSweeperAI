"""Regenerate every README chart from the committed files in results/.

    python scripts/make_figures.py        # writes docs/assets/learning_curves.png and results_comparison.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "assets"

# One colour per agent in every chart (Okabe-Ito, colour-blind safe).
C_RANDOM, C_RULE, C_DQN = "#8c8c8c", "#0072B2", "#E69F00"
C_DQN_TEXT = "#B26B00"  # darker orange for text on white
INK = "#263238"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#90a4ae", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def pct(ax) -> None:
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def learning_curves() -> None:
    boards = [
        ("5x5 / 3 mines", RES / "dqn/dqn_5x5_3_metrics.csv", "5x5_3", 1e3, "thousand board-steps"),
        ("9x9 / 10 mines", RES / "dqn/dqn_9x9_10_metrics.csv", "9x9_10", 1e6, "million board-steps"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (title, csv_path, tag, unit, xlabel) in zip(axes, boards):
        d = load_csv(csv_path)
        x = d["board_steps"] / unit
        ax.plot(x, d["eval_win_rate"], color=C_DQN, lw=2.2, marker="o", ms=4, zorder=3)
        if "eval_ci_low" in d:
            ax.fill_between(x, d["eval_ci_low"], d["eval_ci_high"], color=C_DQN, alpha=0.18, lw=0)
        rule = load_json(RES / f"baselines/rule_{tag}.json")["win_rate"]
        rand = load_json(RES / f"baselines/random_{tag}.json")["win_rate"]
        ax.axhline(rule, color=C_RULE, lw=2, ls="--", zorder=2)
        ax.axhline(rand, color=C_RANDOM, lw=2, ls="--", zorder=2)
        xmax = x.max()
        ax.set_xlim(0, xmax * 1.0)
        ax.text(xmax, rule - 0.012, f"rule solver {rule:.0%}", color=C_RULE, ha="right", va="top", fontweight="bold")
        ax.text(xmax, rand + 0.012, f"random {rand:.0%}", color="#5f5f5f", ha="right", va="bottom", fontweight="bold")
        ax.text(xmax, 0.99, "DQN (greedy eval)", color=C_DQN_TEXT, ha="right", va="top", fontweight="bold")
        floor = x[np.argmax(d["epsilon"] <= d["epsilon"].min() + 5e-4)]
        ax.axvline(floor, color="#90a4ae", lw=1.2, ls=":", zorder=1)
        ax.text(floor + xmax * 0.01, 0.30 if tag == "5x5_3" else 0.28, "exploration reaches\nits floor (epsilon = 0.01)",
                color="#546e7a", fontsize=10.5, va="center")
        ax.set_ylim(0, 1.0)
        pct(ax)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(f"Training progress ({xlabel})")
        ax.grid(axis="y", color="#eceff1", lw=1)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Win rate (%)")
    fig.suptitle("Learning curves: win rate of the greedy agent during training", x=0.01, ha="left", fontsize=14,
                 fontweight="bold")
    fig.text(0.01, 0.005, "Eval games per point: 500 (5x5), 1,000 (9x9); shaded band = 95% CI where recorded. "
             "Reference lines: 10,000-game baselines.", fontsize=9.5, color="#546e7a")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "learning_curves.png", dpi=100)
    plt.close(fig)


def results_comparison() -> None:
    groups = [
        ("5x5 / 3 mines", "baselines/random_5x5_3.json", "baselines/rule_5x5_3.json", "dqn/dqn_5x5_3.json"),
        ("9x9 / 10 mines", "baselines/random_9x9_10.json", "baselines/rule_9x9_10.json", "dqn/dqn_9x9_10.json"),
        ("16x16 / 40 mines\n(DQN trained on 9x9 only)", "baselines/random_16x16_40.json",
         "baselines/rule_16x16_40.json", "dqn/dqn_9x9_10_on_16x16_40.json"),
    ]
    series = [("random", C_RANDOM, "#5f5f5f"), ("rule solver", C_RULE, C_RULE), ("DQN", C_DQN, C_DQN_TEXT)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bw = 0.26
    for j, (name, color, tcolor) in enumerate(series):
        for i, g in enumerate(groups):
            r = load_json(RES / g[1 + j])
            wr, (lo, hi) = r["win_rate"], r["win_rate_ci95"]
            xpos = i + (j - 1) * (bw + 0.02)
            ax.bar(xpos, wr, bw, color=color, zorder=3, label=name if i == 0 else None)
            ax.errorbar(xpos, wr, yerr=[[wr - lo], [hi - wr]], color=INK, capsize=3, lw=1.3, zorder=4)
            ax.text(xpos, hi + 0.015, f"{wr:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=tcolor)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0, 1.05)
    pct(ax)
    ax.set_ylabel("Win rate over 10,000 games (%)")
    ax.grid(axis="y", color="#eceff1", lw=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=12)
    ax.set_title("Win rate by board and agent (error bars: 95% CI)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "results_comparison.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    learning_curves()
    results_comparison()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
