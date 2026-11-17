"""Agent interface shared by baselines and (later) the DQN agent."""

from __future__ import annotations

import numpy as np


class Agent:
    """An agent sees only the observation and the legal-action mask.

    Any randomness must come from the generator handed to :meth:`reset`, so
    evaluation is reproducible.
    """

    name: str = "agent"

    def reset(self, rng: np.random.Generator) -> None:
        """Called at the start of every game."""
        self.rng = rng

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """obs: (10, H, W) float32; mask: (H*W,) bool -> flat action index."""
        raise NotImplementedError
