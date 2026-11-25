# MineSweeperAI — Design Plan

Portfolio/fun project: a reinforcement-learning agent that learns Minesweeper by trial and error.
Decisions below were settled in a design review on 2026-11-17. Implementer: follow them; raise concerns before deviating.

## Goals
- Debug on **5×5 / 3 mines**; primary target **Beginner 9×9 / 10 mines**; stretch **Intermediate 16×16 / 40 mines**.
- Same model must run on any board size (fully convolutional).
- Realistic targets: 9×9 ≈ 70–90% win rate; 16×16 ≈ 30–60%.

## Stack
- Python **3.11 or 3.12** (venv; not 3.14), PyTorch, Gymnasium, NumPy, PyYAML, TensorBoard, pytest, matplotlib/PIL.
- Pip-installable package (`pip install -e .`).

## Game rules (environment)
- **First click always safe** — mines placed after the first click.
- **Reveal-only** action space (no flags). Action = cell index (H·W actions).
- Revealing a 0 flood-fills its neighbours.
- Win = all safe cells revealed; loss = mine revealed.
- Max-steps cap per episode as a safety net.

## Observation
- **One-hot, 10 channels × H × W**: channel 0 = hidden, channels 1–9 = revealed number 0–8.
- Zero-padding in convs means off-board reads as all-zero (distinct from any real cell).

## Action masking
- Revealed cells' Q-values set to −∞ **both** when acting **and** in the Double-DQN target's argmax.
- Random exploration samples from hidden cells only.

## Reward
- Win **+1**, loss **−1**, safe reveal **+0.3**.
- **No reward for the first click** (guaranteed safe).
- Guess penalty: NOT in baseline; later experiment (M8).

## Algorithm
- **Double DQN**: replay buffer + target network.
- **γ = 0.1** initially; γ sweep (incl. 0) is an M8 experiment.
- **ε-greedy**: linear decay 1.0 → 0.01 over first 20–30% of training steps, then 0.01. Evaluation uses ε = 0.

## Network
- Fully convolutional: **5 × [Conv 3×3, 64 ch, padding=1, ReLU]** → **Conv 1×1 → 1 channel** = Q per cell (H×W), masked + flattened.
- No BatchNorm, no fully-connected layers. ~150k params.
- Upgrade option if 16×16 plateaus: residual blocks.

## Environments
- `gym_env.py`: single-board Gymnasium env — reference implementation, used for tests, rendering, SB3 compatibility.
- `batched.py`: N boards as `(N,H,W)` arrays (N ≈ 64–256), auto-reset, used for training. Flood fill per-board (Python or `scipy.ndimage.label`), everything else vectorized.
- Equivalence test: identical trajectories vs. single env on fixed seeds.

## Evaluation
- Headline: **win rate over 10,000 fixed-seed games** (ε = 0).
- Secondary: mean % safe cells revealed, mean episode length.
- Learning curve: quick 1,000-game eval every N training steps.
- Baselines: **random clicker**, **simple rule-based solver** (two basic deduction rules + random guess when stuck).

## Tracking & reproducibility
- TensorBoard live (loss, mean Q, ε, episode reward, eval win rate); final charts via matplotlib from saved CSVs.
- One YAML config per run in `configs/`; each run dir saves resolved config, git commit, seed.
- Checkpoints: `latest.pt` (periodic, enables resume) + `best.pt` (by eval win rate).
- Seed Python/NumPy/PyTorch (no bit-exact GPU promise).

## Compute
- Local laptop (2-core CPU, no GPU): development, tests, full 5×5 runs.
- **Kaggle Notebooks (free GPU)** for 9×9 and 16×16; must resume across 12h session limits.

## Presentation
- README: GIFs of agent play with **Q-value heatmap overlay**, learning curves, results table vs. baselines.
- Final: **in-browser demo** — ONNX export + onnxruntime-web on GitHub Pages (watch agent / play with hints).

## Repo layout
```
MineSweeperAI/
  pyproject.toml
  configs/                  # 5x5.yaml, 9x9.yaml, 16x16.yaml
  src/minesweeper_rl/
    env/core.py             # pure game logic
    env/gym_env.py          # single Gymnasium env
    env/batched.py          # vectorized env
    agents/network.py       # 5-layer FCN
    agents/dqn.py           # Double DQN, replay, masking
    agents/baselines.py     # random + rule-based
    train.py  evaluate.py  render.py
  tests/
  notebooks/kaggle_train.ipynb
  web/                      # ONNX browser demo (M9)
```

## Required tests
- Game logic: first-click safety, mine count, flood fill, win/loss detection, seed determinism.
- Batched env ≡ single env on fixed seeds.
- Masking: never selects a revealed cell (acting and target).
- Smoke test: ~2 min 5×5 training beats random baseline.
- GitHub Actions runs pytest on every push.

## Milestones (strictly in order)
| # | Milestone | Done when |
|---|---|---|
| M0 | Setup: venv, pyproject, git, GitHub, CI | pytest green in CI |
| M1 | Game core + single env | Logic tests pass; playable in terminal |
| M2 | Baselines + eval script | Random & rule-solver win rates on 5×5 and 9×9 (10k games) |
| M3 | Batched env | Equivalent to single env; ≥10× faster |
| M4 | Network + Double DQN + training loop | 5×5 agent clearly beats random; smoke test in CI |
| M5 | Logging, configs, checkpoint/resume | Survives kill + resume |
| M6 | 9×9 on Kaggle | Final 10k-game win rate; ideally beats rule solver |
| M7 | GIFs, heatmap, charts, README | README tells the full story |
| M8 | Experiments (scope narrowed): (A) 9×9 stability: lr decay / Polyak target, 2 seeds each; (B) 16×16 fine-tune from the 9×9 model vs scratch. γ sweep, guess penalty, bigger nets dropped | Results tables + figure in README |
| M9 | Browser demo | Public playable link |

## Workflow
- Opus = advisor/reviewer; Sonnet = implementer. Each milestone: Sonnet implements → Opus reviews against this plan and "done when" criteria → fix → next milestone.
