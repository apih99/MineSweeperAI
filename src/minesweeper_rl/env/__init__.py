from gymnasium.envs.registration import register, registry

from minesweeper_rl.env.core import Board, RevealResult
from minesweeper_rl.env.gym_env import MinesweeperEnv

ENV_ID = "MinesweeperRL-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="minesweeper_rl.env.gym_env:MinesweeperEnv")

__all__ = ["Board", "RevealResult", "MinesweeperEnv", "ENV_ID"]
