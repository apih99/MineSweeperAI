"""Double DQN: replay buffer, masked target, epsilon-greedy, and a checkpoint-backed Agent.

RNG stream layout (see ``make_streams``): one master seed -> ``SeedSequence(seed).spawn(3)``:
  [0] exploration  (epsilon coin flips + random legal-cell scores)   -> Generator
  [1] replay       (minibatch indices)                               -> Generator
  [2] torch        (network init; seeds torch's global generator)    -> int
The training envs are NOT drawn from this tree: board i is seeded ``train_seed_base + i``
(gymnasium seeding), disjoint from the eval seed range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from minesweeper_rl.agents.base import Agent
from minesweeper_rl.agents.network import QNetwork, mask_q, masked_argmax


def make_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator, int]:
    ss = np.random.SeedSequence(seed).spawn(3)
    return (
        np.random.default_rng(ss[0]),
        np.random.default_rng(ss[1]),
        int(ss[2].generate_state(1)[0] % (2**31 - 1)),
    )


def obs_to_codes(obs: np.ndarray) -> np.ndarray:
    """(N,10,H,W) one-hot -> (N,H,W) uint8 code (0 hidden, 1..9 = number 0..8)."""
    return obs.argmax(axis=1).astype(np.uint8)


def codes_to_onehot(codes: torch.Tensor) -> torch.Tensor:
    """(B,H,W) integer codes -> (B,10,H,W) float32."""
    return F.one_hot(codes.long(), 10).permute(0, 3, 1, 2).float().contiguous()


@dataclass
class Batch:
    state: torch.Tensor  # (B,10,H,W) float
    action: torch.Tensor  # (B,) long
    reward: torch.Tensor  # (B,)
    next_state: torch.Tensor
    next_mask: torch.Tensor  # (B,HW) bool
    done: torch.Tensor  # (B,) bool


class ReplayBuffer:
    def __init__(self, capacity: int, height: int, width: int, rng: np.random.Generator):
        self.capacity, self.rng = capacity, rng
        hw = height * width
        self.state = np.zeros((capacity, height, width), np.uint8)
        self.next_state = np.zeros((capacity, height, width), np.uint8)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.next_mask = np.zeros((capacity, hw), bool)
        self.done = np.zeros(capacity, bool)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, state_codes, action, reward, next_codes, next_mask, done) -> None:
        n = len(action)
        idx = (self.pos + np.arange(n)) % self.capacity
        self.state[idx] = state_codes
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_state[idx] = next_codes
        self.next_mask[idx] = next_mask
        self.done[idx] = done
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Contents as plain arrays (only the filled rows; the bool mask is bit-packed)."""
        n = self.size
        return {
            "state": self.state[:n], "next_state": self.next_state[:n], "action": self.action[:n],
            "reward": self.reward[:n], "done": self.done[:n],
            "next_mask": np.packbits(self.next_mask[:n], axis=1),
            "pos": np.array(self.pos, np.int64), "size": np.array(n, np.int64),
            "capacity": np.array(self.capacity, np.int64),
        }

    def load_arrays(self, d) -> None:
        n, cap = int(d["size"]), int(d["capacity"])
        if cap != self.capacity or self.state.shape[1:] != d["state"].shape[1:]:
            raise ValueError("replay file does not match this buffer's capacity/board size")
        self.state[:n], self.next_state[:n] = d["state"], d["next_state"]
        self.action[:n], self.reward[:n], self.done[:n] = d["action"], d["reward"], d["done"]
        self.next_mask[:n] = np.unpackbits(d["next_mask"], axis=1, count=self.next_mask.shape[1]).astype(bool)
        self.pos, self.size = int(d["pos"]), n

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> Batch:
        """Uniform minibatch (sampling is NumPy; tensors are moved to ``device`` per batch)."""
        i = self.rng.integers(0, self.size, size=batch_size)
        t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
        return Batch(
            codes_to_onehot(t(self.state[i])),
            t(self.action[i]),
            t(self.reward[i]),
            codes_to_onehot(t(self.next_state[i])),
            t(self.next_mask[i]),
            t(self.done[i]),
        )


def double_dqn_targets(online, target, batch: Batch, gamma: float) -> torch.Tensor:
    """y = r + gamma * (1-done) * Q_target(s', argmax_a over legal a of Q_online(s', a)).

    Action selection by the online net (masked); value from the target net. Done transitions
    may have an all-False next mask (won board); they are handled with torch.where so no
    NaN/inf can leak in.
    """
    with torch.no_grad():
        a_star = masked_argmax(online(batch.next_state), batch.next_mask)
        q_next = target(batch.next_state).gather(1, a_star[:, None]).squeeze(1)
        q_next = torch.where(batch.done, torch.zeros_like(q_next), q_next)
        y = batch.reward + gamma * q_next
    assert torch.isfinite(y).all(), "non-finite Double-DQN target"
    return y


def dqn_loss(online, target, batch: Batch, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    y = double_dqn_targets(online, target, batch, gamma)
    q = online(batch.state)
    q_sa = q.gather(1, batch.action[:, None]).squeeze(1)
    return F.smooth_l1_loss(q_sa, y), q_sa.detach()


def epsilon_greedy_actions(
    q_net, obs: np.ndarray, mask: np.ndarray, eps: float, rng: np.random.Generator
) -> np.ndarray:
    """Batch epsilon-greedy over legal cells. obs (N,10,H,W), mask (N,HW) bool."""
    n = mask.shape[0]
    dev = next(q_net.parameters()).device
    with torch.no_grad():
        greedy = masked_argmax(q_net(torch.from_numpy(obs).to(dev)), torch.from_numpy(mask).to(dev))
        greedy = greedy.cpu().numpy()
    scores = np.where(mask, rng.random(mask.shape), -1.0)
    random_a = scores.argmax(axis=1)
    explore = rng.random(n) < eps
    return np.where(explore, random_a, greedy).astype(np.int64)


def make_checkpoint(net: QNetwork, net_cfg: dict, extra: dict | None = None) -> dict:
    ck = {"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "net": dict(net_cfg)}
    ck.update(extra or {})
    return ck


class DQNAgent(Agent):
    """Greedy (eps=0) masked agent; any board size."""

    name = "dqn"

    def __init__(self, net: QNetwork | None = None, checkpoint: str | None = None):
        if net is None:
            if checkpoint is None:
                raise ValueError("DQNAgent needs a net or a checkpoint path")
            # weights_only=True: checkpoints hold only tensors and plain Python types. This also
            # reads M4 best.pt/final.pt (state_dict + net + plain-dict config) and latest.pt.
            ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
            net = QNetwork(**ck["net"])
            net.load_state_dict(ck["state_dict"])
        self.net = net
        self.rng = np.random.default_rng(0)

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        return cls(checkpoint=path)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        dev = next(self.net.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))[None].to(dev)
            m = torch.from_numpy(np.asarray(mask, bool))[None].to(dev)
            return int(masked_argmax(self.net(x), m)[0])

    def q_values(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Masked Q-values as an (H, W) float array; illegal (already revealed) cells are -inf."""
        dev = next(self.net.parameters()).device
        h, w = obs.shape[-2:]
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))[None].to(dev)
            m = torch.from_numpy(np.asarray(mask, bool))[None].to(dev)
            return mask_q(self.net(x), m)[0].cpu().numpy().reshape(h, w)
