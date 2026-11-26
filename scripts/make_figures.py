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
        ("16x16 / 40 mines\n(DQN fine-tuned from the 9x9 model)", "baselines/random_16x16_40.json",
         "baselines/rule_16x16_40.json", "dqn/dqn_16x16_40_finetune.json"),
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


EXP = RES / "experiments"
VARIANT_COLORS = {"baseline": "#8c8c8c", "lr_decay": "#0072B2", "polyak": "#009E73", "both": "#D55E00",
                  "finetune": "#D55E00", "scratch": "#0072B2"}


def _runs(prefix: str, variants: list[str]) -> dict[str, list[str]]:
    out = {}
    for v in variants:
        names = sorted(p.stem for p in (EXP / "metrics").glob(f"{prefix}{v}_s*.csv"))
        if names:
            out[v] = names
    return out


def _bars(ax, groups: dict[str, list[str]], title: str, ref_lines: list[tuple[str, float, str]] = ()) -> None:
    """Per variant: bars = mean over seeds of final.pt / best.pt unseen win rate; dots + CI = single runs."""
    xs, labels = [], []
    for i, (v, names) in enumerate(groups.items()):
        for j, which in enumerate(("final", "best")):
            vals = []
            for n in names:
                f = EXP / f"{n}_{which}_unseen.json"
                if f.exists():
                    vals.append(load_json(f))
            if not vals:
                continue
            x = i + (j - 0.5) * 0.36
            m = np.mean([r["win_rate"] for r in vals])
            ax.bar(x, m, 0.34, color=VARIANT_COLORS[v], alpha=1.0 if which == "best" else 0.55, zorder=3,
                   label=None)
            for k, r in enumerate(vals):
                lo, hi = r["win_rate_ci95"]
                xx = x + (k - (len(vals) - 1) / 2) * 0.08
                ax.errorbar(xx, r["win_rate"], yerr=[[r["win_rate"] - lo], [hi - r["win_rate"]]], color=INK, lw=1,
                            capsize=2, marker="o", ms=3.5, zorder=4)
            ax.text(x, max(r["win_rate_ci95"][1] for r in vals) + 0.012, f"{m:.1%}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")
        xs.append(i)
        labels.append(v.replace("_", " "))
    for text, y, color in ref_lines:
        ax.axhline(y, color=color, lw=1.8, ls="--", zorder=2)
        ax.text(-0.48, y, text, color=color, ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    pct(ax)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12.5)
    ax.grid(axis="y", color="#eceff1", lw=1)
    ax.set_axisbelow(True)


def _curves(ax, groups: dict[str, list[str]], unit: float, xlabel: str, title: str) -> None:
    for v, names in groups.items():
        ds = [load_csv(EXP / "metrics" / f"{n}.csv") for n in names]
        for d in ds:
            ax.plot(d["board_steps"] / unit, d["eval_win_rate"], color=VARIANT_COLORS[v], lw=1, alpha=0.45)
        n = min(len(d["board_steps"]) for d in ds)
        ax.plot(ds[0]["board_steps"][:n] / unit, np.mean([d["eval_win_rate"][:n] for d in ds], axis=0),
                color=VARIANT_COLORS[v], lw=2.4, label=v.replace("_", " "))
    ax.set_xlabel(xlabel)
    pct(ax)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12.5)
    ax.grid(axis="y", color="#eceff1", lw=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=10)


def experiments() -> None:
    a = _runs("exp_9x9_", ["baseline", "lr_decay", "polyak", "both"])
    b = _runs("exp_16x16_", ["scratch", "finetune"])
    if not a:
        return
    rows = 2 if b else 1
    fig, axes = plt.subplots(rows, 2, figsize=(12.5, 4.9 * rows), squeeze=False)
    _curves(axes[0, 0], a, 1e6, "million board-steps",
            "A: 9x9 eval win rate (1,000 games) during training")
    axes[0, 0].set_ylim(0.5, 0.95)
    axes[0, 0].set_ylabel("Win rate (%)")
    _bars(axes[0, 1], a, "A: 9x9, 10,000 unseen games (bar = mean of 2 seeds)",
          [("rule solver", load_json(RES / "baselines/rule_9x9_10_unseen.json")["win_rate"], C_RULE)])
    axes[0, 1].set_ylim(0.6, 0.97)
    if b:
        _curves(axes[1, 0], b, 1e6, "million board-steps",
                "B: 16x16 eval win rate (1,000 games) during training")
        zs = load_json(EXP / "zeroshot_9x9_on_16x16_40_unseen.json")["win_rate"]
        axes[1, 0].axhline(zs, color=C_DQN, lw=1.8, ls="--", label=f"9x9 model zero-shot ({zs:.1%})")
        axes[1, 0].legend(frameon=False, loc="lower right", fontsize=10)
        axes[1, 0].set_ylim(0, 0.8)
        axes[1, 0].set_ylabel("Win rate (%)")
        rule = load_json(EXP / "rule_16x16_40_unseen.json")["win_rate"]
        _bars(axes[1, 1], b, "B: 16x16/40, 10,000 unseen games",
              [("rule solver", rule, C_RULE), ("zero-shot", zs, C_DQN_TEXT)])
        axes[1, 1].set_ylim(0, 0.85)
    for ax in axes[:, 1]:
        ax.set_ylabel("Win rate (%)")
    fig.text(0.01, 0.004, "Right panels: pale bar = final.pt, solid bar = best.pt (mean of 2 seeds); dots = single runs "
             "with 95% CI. Left panels: thin lines = single runs, thick = mean of 2 seeds.", fontsize=9.5,
             color="#546e7a")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "experiments.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    learning_curves()
    results_comparison()
    experiments()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
