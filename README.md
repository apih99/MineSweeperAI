# MineSweeperAI

A reinforcement-learning agent (Double DQN, fully convolutional) that learns to play Minesweeper.

> Work in progress. See [PLAN.md](PLAN.md) for the design and milestones.

## Setup (Windows PowerShell)

```powershell
py -V:3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

## Play in the terminal

```powershell
python -m minesweeper_rl.play --height 9 --width 9 --mines 10 --seed 0
```

Enter `row col` to reveal a cell (the first click is always safe); `q` quits.
Legend: `#` hidden, `.` empty, digits = adjacent mines, `*` mine, `X` the mine you hit.

## Baselines

Win rate over 10,000 fixed-seed games (seeds 1,000,000-1,009,999), 95% Wilson CI.
Reproduce: `python -m minesweeper_rl.evaluate --agent rule --height 9 --width 9 --mines 10 --games 10000`.
Raw JSON in [results/baselines/](results/baselines/).

| Agent | Board | Win rate (95% CI) | Mean % safe revealed |
|-------|-------|-------------------|----------------------|
| random | 5x5 / 3 mines | 8.82% (8.28-9.39%) | 71.4% |
| rule | 5x5 / 3 mines | 68.96% (68.05-69.86%) | 81.9% |
| random | 9x9 / 10 mines | 0.00% (0.00-0.04%) | 56.1% |
| rule | 9x9 / 10 mines | 68.11% (67.19-69.02%) | 79.3% |

## Batched environment

`minesweeper_rl.env.batched.BatchedMinesweeperEnv(n_envs, h, w, n_mines)` steps N independent
boards with NumPy array ops (same rewards/termination/truncation as `MinesweeperEnv`; finished
boards auto-reset in the same step, terminal data in `info["final_obs"]` / `info["final_mask"]`).
Board `i` owns its own `Generator` (`seeding.np_random(seeds[i])`), used only for its mine
placement, so a batched run is bit-for-bit identical to N single envs seeded the same way
(`tests/test_batched.py`). Flood fill labels zero-count regions once at mine placement
(`scipy.ndimage.label`, 8-connectivity) and reveals component + border by dilation.

Benchmark (`python scripts/bench_env.py`, 9x9/10 mines, i7-6600U laptop with other programs
running, so numbers are noisy; best of 10 runs, process CPU time, env steps/sec only):

| Policy | single | N=64 | N=128 | N=256 |
|--------|--------|------|-------|-------|
| random legal cell | 10.0k | 83.6k (8.4x) | 60.2k (6.0x) | 81.9k (8.2x) |
| safe-peek (long episodes) | 15.7k | 116.9k (7.4x) | 157.2k (10.0x) | 85.3k (5.4x) |

Speedup varies roughly 6-10x (random policy) and 5-30x (safe-peek) between runs on this machine.
Remaining bottleneck: one `Generator.choice` per episode (~20 us, must stay bit-identical to the
single env), which dominates when episodes are very short (random agent: ~3-4 clicks).
