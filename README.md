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
| random | 5x5 / 3 mines | 4.78% (4.38-5.22%) | 40.2% |
| rule | 5x5 / 3 mines | 35.88% (34.95-36.83%) | 45.4% |
| random | 9x9 / 10 mines | 0.00% (0.00-0.04%) | 45.8% |
| rule | 9x9 / 10 mines | 55.48% (54.50-56.45%) | 65.6% |
