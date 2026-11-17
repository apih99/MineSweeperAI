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
