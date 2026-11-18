"""Fully convolutional Q-network: works on any board size with the same weights."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """n_layers x [Conv3x3(pad=1) + ReLU] -> Conv1x1 -> 1 channel. Output: (B, H*W) Q per cell.

    No BatchNorm, no Linear layers, so any (H, W) is accepted.
    """

    def __init__(self, in_channels: int = 10, channels: int = 64, n_layers: int = 5):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(n_layers):
            layers += [nn.Conv2d(c, channels, 3, padding=1), nn.ReLU()]
            c = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).flatten(1)


def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """argmax over cells of q restricted to mask (bool, same shape). Never returns a masked
    cell unless the whole row is masked (then index 0, finite and harmless)."""
    return torch.where(mask, q, torch.full_like(q, float("-inf"))).argmax(dim=1)
