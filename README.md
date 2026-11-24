# MineSweeperAI

A reinforcement-learning agent (Double DQN, fully convolutional) that learns to play Minesweeper.

> Work in progress. 9x9/10-mine agent: 86.6% win rate vs. 68.1% for the rule solver (see [9x9 results](#9x9-on-a-kaggle-gpu-m6)). See [PLAN.md](PLAN.md) for the design and milestones.

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

## DQN (M4)

Fully convolutional 5x(Conv3x3, 64 ch)+Conv1x1 Q-network, Double DQN with action masking in both
acting and the target argmax. Train: `python -m minesweeper_rl.train --config configs/5x5.yaml` (the M4 hyperparameters; see [Training](#training-resume-and-tensorboard-m5));
evaluate: `python -m minesweeper_rl.evaluate --agent dqn --checkpoint runs/dqn_5x5_3/best.pt --height 5 --width 5 --mines 3`
(batched evaluation: 10,000 greedy 5x5 games in ~22 s on the laptop CPU vs ~45 s one board at a time, with identical results; the CPU gain is only ~2x because the network forward pass dominates, on a GPU it is far larger).
5x5/3 run: 400k board-steps (~23 min on a 2-core laptop CPU), `best.pt` picked by 500-game training evals
(selection on seeds that overlap the first 500 of the 10k eval, so the 10k number is very slightly optimistic;
`final.pt` scores 78.25%). Curve in [results/dqn/dqn_5x5_3_metrics.csv](results/dqn/dqn_5x5_3_metrics.csv).

| Agent | Board | Win rate (95% CI) | Mean % safe revealed |
|-------|-------|-------------------|----------------------|
| random | 5x5 / 3 mines | 8.82% (8.28-9.39%) | 71.4% |
| rule | 5x5 / 3 mines | 68.96% (68.05-69.86%) | 81.9% |
| **DQN (best.pt)** | 5x5 / 3 mines | **79.93% (79.13-80.70%)** | 89.4% |

## Training, resume and TensorBoard (M5)

```powershell
python -m minesweeper_rl.train --config configs/5x5.yaml                    # -> runs/dqn_5x5_3/
python -m minesweeper_rl.train --config configs/9x9.yaml --run-dir runs/a --set seed=1 --set lr=1e-4
python -m minesweeper_rl.train --resume runs/a                              # continue from runs/a/latest.pt
python -m minesweeper_rl.train --resume runs/a --set total_steps=40000000   # ...and extend the run
tensorboard --logdir runs
```

* Configs in [configs/](configs/) (`5x5`, `9x9`, `16x16`) list only what differs from the `TrainConfig`
  defaults (`5x5.yaml` lists everything: it is the M4 run). Unknown keys, bad types and typos in
  `--set` are errors. `device: auto|cpu|cuda` (auto = cuda if available), `torch_threads`.
* Each run directory holds `config.yaml` (resolved), `meta.json` (git commit + dirty flag, seed,
  start time, torch/numpy versions, device), `metrics.csv`, TensorBoard events, `best.pt`
  (best eval win rate), `final.pt`, and the resume state `latest.pt` + `replay-<steps>.npz`.
* Logged every `eval_every` board-steps (CSV row and TensorBoard): `train/loss` and `train/mean_q`
  (averaged over the interval), `train/epsilon`, `train/episode_return`, `train/episode_length`,
  `train/win_rate` (training episodes, with exploration), `eval/win_rate` (+ Wilson CI low/high),
  `perf/board_steps_per_s`, `perf/updates_per_s`.
* `latest.pt` is written every `checkpoint_every_minutes` (default 10), at the end, and on
  Ctrl-C/SIGTERM. It holds everything needed to continue **bit-for-bit**: both nets, Adam state,
  counters, all RNG states (exploration, replay, torch CPU/CUDA, every training board's generator),
  the full batched-env state and the logging accumulators; the replay buffer is the sibling
  `replay-<board_steps>.npz` that `latest.pt` names. Files are written to a temp file and
  `os.replace`d, replay first, so a kill at any moment (even mid-save) leaves the previous
  consistent pair; `metrics.csv` and the TensorBoard curve (`purge_step`) are rewound to the
  checkpoint, so resumed runs have no duplicate or missing rows. All `.pt` files load with
  `torch.load(..., weights_only=True)`.
* On `--resume` only `total_steps`, `eval_every`, `eval_games`, `checkpoint_every_minutes`,
  `verbose`, `torch_threads`, `device` may change (board size, net shape, seeds, ... are rejected).
  Note `total_steps` also stretches the epsilon schedule (decay is `eps_decay_frac * total_steps`).
* Exact-resume is tested on CPU (`tests/test_train_resume.py`: train 2k steps straight vs. 1k +
  fresh-process resume to 2k -> identical weights, optimizer state, replay, RNG states, CSV rows).
  Resuming on a different device or torch version is not guaranteed bit-identical.
* Checkpoint sizes: 5x5 (100k buffer) `latest.pt` ~2.6 MB + replay ~6.7 MB; 9x9 config
  `latest.pt` ~14 MB + replay ~93 MB; 16x16 config ~23 MB + ~223 MB (replay is saved
  uncompressed, ~0.4-0.9 s; `np.savez_compressed` would be ~8x smaller but ~5-10 s).

## 9x9 on a Kaggle GPU (M6)

`configs/9x9.yaml` (same 5x64 network as 5x5, 256 boards per batched step, 20M board-steps) was trained
on a free Kaggle Tesla T4 with [kaggle/run_kaggle.py](kaggle/run_kaggle.py), driven locally by
[scripts/kaggle_run.py](scripts/kaggle_run.py):

```powershell
git push origin main                                    # the kernel clones GitHub at a pinned commit
python scripts/kaggle_run.py push --accelerator NvidiaTeslaT4            # fresh run of configs/9x9.yaml
python scripts/kaggle_run.py push --resume --accelerator NvidiaTeslaT4   # next session: resume latest.pt
python scripts/kaggle_run.py wait                       # poll status
python scripts/kaggle_run.py output --version v1        # -> runs/kaggle/v1/ (gitignored)
```

The kernel prints `nvidia-smi`, refuses to run without CUDA, clones the repo at the pinned SHA, installs the
package with `--no-deps` (Kaggle's CUDA torch is kept; Python 3.13 needs `--ignore-requires-python`), runs
`pytest -m "not slow"`, trains with `max_wall_minutes=690` (clean save before the 12 h session limit), then
evaluates `best.pt`/`final.pt` on 10k standard and 10k unseen seeds. `--resume` attaches the kernel's own
previous version as an input (`kernel_sources`); the run directory found there is copied and resumed.

Throughput on the T4 (4 CPU cores): ~9.1k board-steps/s and ~36 updates/s, i.e. the 20M-step run took
37 minutes. The NumPy environment on the CPU, not the GPU, is the bottleneck.

Win rate over 10,000 games (95% Wilson CI). "Standard" = seeds 1,000,000-1,009,999; "unseen" = seeds
5,000,000-5,009,999 (never used for training, model selection or the learning-curve evals).
`best.pt` is the checkpoint with the best 1,000-game training-time eval (selected on seeds that overlap
the standard range, so only the unseen column is free of selection bias). Raw JSON in
[results/](results/), curve in [results/dqn/dqn_9x9_10_metrics.csv](results/dqn/dqn_9x9_10_metrics.csv),
weights in [models/dqn_9x9_10.pt](models/dqn_9x9_10.pt).

| Agent | Board | Standard seeds | Unseen seeds |
|-------|-------|----------------|--------------|
| random | 5x5 / 3 mines | 8.82% (8.28-9.39%) | - |
| rule | 5x5 / 3 mines | 68.96% (68.05-69.86%) | - |
| **DQN (best.pt)** | 5x5 / 3 mines | **79.93% (79.13-80.70%)** | - |
| random | 9x9 / 10 mines | 0.00% (0.00-0.04%) | 0.01% (0.00-0.06%) |
| rule | 9x9 / 10 mines | 68.11% (67.19-69.02%) | 68.06% (67.14-68.97%) |
| **DQN (best.pt)** | 9x9 / 10 mines | **86.55% (85.87-87.20%)** | **87.57% (86.91-88.20%)** |
| DQN (final.pt, 20M steps) | 9x9 / 10 mines | 79.50% (78.70-80.28%) | 80.05% (79.26-80.82%) |

Caveat: the 1,000-game training evals of consecutive checkpoints swing between ~74% and ~87% late in the
run (constant learning rate, no averaging), so `final.pt` is a noisier draw than `best.pt`; the unseen-seed
number for `best.pt` is the honest estimate of that checkpoint's strength, but picking the best of 40
noisy evals still favours lucky checkpoints a little.
