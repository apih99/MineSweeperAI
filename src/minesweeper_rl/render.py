"""Frames and GIFs of the agent playing, with a Q-value heatmap next to the board.

Each frame has two panels. Left: the board as a player sees it (hidden cells, revealed numbers in the
classic colours). Right: the network's Q-value for every still-hidden cell (revealed cells are greyed).
The cell the agent is about to reveal is outlined on both panels. The last frame of a game shows the
true mines (and the exploded one on a loss) with the Q-values of the final decision.

CLI::

    python -m minesweeper_rl.render --checkpoint models/dqn_9x9_10.pt --height 9 --width 9 \
        --mines 10 --seed 5000000 --out docs/assets/x.gif [--fps 2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from minesweeper_rl.agents.dqn import DQNAgent  # noqa: E402
from minesweeper_rl.env.core import Board  # noqa: E402
from minesweeper_rl.env.gym_env import MinesweeperEnv  # noqa: E402

FIG_W, FIG_H, DPI = 9.6, 4.9, 80  # -> 768 x 392 px frames

# Classic Minesweeper number colours (1..8), readable on the light revealed-cell background.
NUMBER_COLORS = {1: "#1565c0", 2: "#2e7d32", 3: "#c62828", 4: "#1a237e", 5: "#8e2424", 6: "#00838f", 7: "#212121", 8: "#757575"}
HIDDEN = "#aab4c0"
REVEALED = "#f1f3f5"
GRID = "#ffffff"
MINE_EXPLODED = "#e53935"
OUTLINE = "#ff6f00"  # chosen cell
WIN_COLOR = "#2e7d32"
LOSS_COLOR = "#c62828"
BLANK_Q = "#e3e6ea"
CMAP = "viridis"
Q_LABEL = "Q-value (higher = agent prefers /\nthinks safer)"


def _draw_board(ax, board: Board, show_mines: bool, chosen, fontsize: float) -> None:
    h, w = board.height, board.width
    colors = np.empty((h, w, 3))
    hidden_rgb = matplotlib.colors.to_rgb(HIDDEN)
    revealed_rgb = matplotlib.colors.to_rgb(REVEALED)
    for r in range(h):
        for c in range(w):
            colors[r, c] = revealed_rgb if board.revealed[r, c] else hidden_rgb
    if board._exploded is not None:
        colors[board._exploded] = matplotlib.colors.to_rgb(MINE_EXPLODED)
    ax.pcolormesh(np.arange(w + 1), np.arange(h + 1), colors, edgecolors=GRID, linewidth=1.2)
    for r in range(h):
        for c in range(w):
            if board.revealed[r, c] and not board.mines[r, c] and board.counts[r, c] > 0:
                n = int(board.counts[r, c])
                ax.text(c + 0.5, r + 0.5, str(n), ha="center", va="center", color=NUMBER_COLORS[n],
                        fontsize=fontsize, fontweight="bold")
    if show_mines:
        rr, cc = np.nonzero(board.mines)
        ax.plot(cc + 0.5, rr + 0.5, "o", color="#212121", markersize=fontsize * 0.85, markeredgecolor="white",
                markeredgewidth=0.6)
    _finish_axes(ax, h, w, chosen)


def _finish_axes(ax, h: int, w: int, chosen) -> None:
    if chosen is not None:
        r, c = chosen
        ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=OUTLINE, linewidth=3, zorder=5))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def render_frame(
    board: Board,
    q: np.ndarray | None,
    chosen: tuple[int, int] | None,
    move: int,
    status: str = "playing",
    q_note: str = "Q-values of the hidden cells",
) -> Image.Image:
    """One two-panel frame as an RGB PIL image of size (FIG_W*DPI, FIG_H*DPI).

    ``q``: (H, W) Q-values, -inf (or NaN) where a cell is not a legal choice; None draws a blank heatmap.
    ``chosen``: (row, col) to outline on both panels. ``status``: 'playing' | 'won' | 'lost'.
    """
    h, w = board.height, board.width
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    cell_px = min(0.38 * FIG_W * DPI / w, 0.74 * FIG_H * DPI / h)
    fontsize = max(6.0, 0.5 * cell_px * 72 / DPI)
    ax_b = fig.add_axes([0.02, 0.05, 0.42, 0.78])
    ax_q = fig.add_axes([0.47, 0.05, 0.36, 0.78])
    ax_c = fig.add_axes([0.855, 0.10, 0.016, 0.66])

    _draw_board(ax_b, board, show_mines=status != "playing", chosen=chosen if status == "playing" else None,
                fontsize=fontsize)
    ax_b.set_title("Board (what the player sees)" if status == "playing" else "Board (mines revealed)",
                   fontsize=11, color="#37474f", pad=6)

    vals = np.full((h, w), np.nan) if q is None else np.where(np.isfinite(q), q, np.nan)
    finite = vals[np.isfinite(vals)]
    cmap = matplotlib.colormaps[CMAP].with_extremes(bad=BLANK_Q)
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmax - vmin < 1e-6:
            vmin, vmax = vmin - 0.05, vmax + 0.05
    else:
        vmin, vmax = 0.0, 1.0
    mesh = ax_q.pcolormesh(np.arange(w + 1), np.arange(h + 1), np.ma.masked_invalid(vals), cmap=cmap,
                           vmin=vmin, vmax=vmax, edgecolors=GRID, linewidth=1.2)
    _finish_axes(ax_q, h, w, chosen)
    ax_q.set_title(q_note, fontsize=11, color="#37474f", pad=6)
    cb = fig.colorbar(mesh, cax=ax_c)
    cb.set_label(Q_LABEL, fontsize=9, color="#37474f")
    cb.ax.tick_params(labelsize=8, colors="#37474f")
    cb.outline.set_visible(False)

    if status == "won":
        head, color = f"Move {move}  |  WON", WIN_COLOR
    elif status == "lost":
        head, color = f"Move {move}  |  LOST (hit a mine)", LOSS_COLOR
    else:
        head, color = f"Move {move}  |  in progress", "#263238"
    fig.text(0.5, 0.945, head, ha="center", va="center", fontsize=15, fontweight="bold", color=color)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return img


def play_frames(agent: DQNAgent, height: int, width: int, n_mines: int, seed: int) -> tuple[list[Image.Image], bool, int]:
    """Play one greedy game (the board ``MinesweeperEnv.reset(seed=seed)`` makes) and render every move.

    Returns (frames, won, n_moves). Frame k (k < n_moves) is the state before move k+1 with that move
    outlined; the last frame is the final state with the Q-values of the last decision.
    """
    env = MinesweeperEnv(height, width, n_mines)
    obs, info = env.reset(seed=seed)
    frames: list[Image.Image] = []
    move = 0
    while True:
        mask = info["action_mask"]
        q = agent.q_values(obs, mask)
        action = agent.act(obs, mask)
        chosen = divmod(action, width)
        move += 1
        frames.append(render_frame(env.board, q, chosen, move, "playing"))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    won = bool(env.board.won)
    frames.append(render_frame(env.board, q, chosen, move, "won" if won else "lost",
                               q_note="Q-values at the final move"))
    return frames, won, move


def write_gif(frames: list[Image.Image], out: str | Path, fps: float = 2.0, hold_s: float = 3.0) -> Path:
    """Save frames as a looping GIF; the last frame is held for ``hold_s`` seconds."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    step_ms = int(round(1000 / fps))
    durations = [step_ms] * (len(frames) - 1) + [int(hold_s * 1000)]
    pal = [f.quantize(colors=200, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--height", type=int, default=9)
    p.add_argument("--width", type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True, help=".gif path (or .png to save only the first frame)")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--hold", type=float, default=3.0, help="seconds to hold the last frame")
    a = p.parse_args(argv)
    agent = DQNAgent(checkpoint=a.checkpoint)
    frames, won, n = play_frames(agent, a.height, a.width, a.mines, a.seed)
    if a.out.lower().endswith(".png"):
        frames[-1].save(a.out, optimize=True)
    else:
        write_gif(frames, a.out, a.fps, a.hold)
    print(f"seed {a.seed}: {'WON' if won else 'LOST'} in {n} moves -> {a.out} ({Path(a.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
