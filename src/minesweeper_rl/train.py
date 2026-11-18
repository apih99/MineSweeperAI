"""DQN training loop. CLI: python -m minesweeper_rl.train --name x --height 5 --width 5 --n-mines 3"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from minesweeper_rl.agents.dqn import (
    DQNAgent,
    ReplayBuffer,
    dqn_loss,
    epsilon_greedy_actions,
    make_checkpoint,
    make_streams,
    obs_to_codes,
)
from minesweeper_rl.agents.network import QNetwork
from minesweeper_rl.env.batched import BatchedMinesweeperEnv
from minesweeper_rl.evaluate import EVAL_SEED_BASE, evaluate


@dataclass
class TrainConfig:
    name: str = "dqn_5x5_3"
    height: int = 5
    width: int = 5
    n_mines: int = 3
    seed: int = 0
    train_seed_base: int = 0
    n_envs: int = 64
    total_steps: int = 1_000_000  # board-steps (N per batched step)
    channels: int = 64
    n_layers: int = 5
    buffer_size: int = 100_000
    batch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.1
    grad_clip: float = 10.0
    train_every: int = 1  # batched env steps between update rounds
    updates_per_step: int = 1
    learning_starts: int = 5_000  # transitions
    target_update_every: int = 1_000  # gradient updates
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_frac: float = 0.25
    eval_every: int = 50_000  # board-steps
    eval_games: int = 500
    num_threads: int = 2
    out_dir: str = "runs"
    verbose: bool = True


def epsilon_at(cfg: TrainConfig, board_steps: int) -> float:
    frac = min(1.0, board_steps / max(1.0, cfg.eps_decay_frac * cfg.total_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def train(cfg: TrainConfig) -> dict:
    assert cfg.train_seed_base + cfg.n_envs < EVAL_SEED_BASE, "training seeds overlap eval range"
    torch.set_num_threads(cfg.num_threads)
    explore_rng, buffer_rng, torch_seed = make_streams(cfg.seed)
    torch.manual_seed(torch_seed)

    net_cfg = {"channels": cfg.channels, "n_layers": cfg.n_layers}
    online = QNetwork(**net_cfg)
    target = copy.deepcopy(online)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer_size, cfg.height, cfg.width, buffer_rng)
    env = BatchedMinesweeperEnv(cfg.n_envs, cfg.height, cfg.width, cfg.n_mines)
    obs, info = env.reset(seeds=cfg.train_seed_base)
    mask = info["action_mask"]

    run_dir = Path(cfg.out_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    f = open(run_dir / "metrics.csv", "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["board_steps", "epsilon", "loss", "mean_q", "eval_win_rate", "wall_s"])

    def ckpt() -> dict:
        return make_checkpoint(online, net_cfg, {"config": dataclasses.asdict(cfg)})

    board_steps = n_updates = 0
    next_eval = cfg.eval_every
    loss_val = mean_q = float("nan")
    best = -1.0
    t0 = time.perf_counter()
    history = []
    while board_steps < cfg.total_steps:
        eps = epsilon_at(cfg, board_steps)
        a = epsilon_greedy_actions(online, obs, mask, eps, explore_rng)
        nobs, rew, term, trunc, info = env.step(a)
        done = info["episode_done"]
        # Next state of a finished board is its final_obs/final_mask (env already auto-reset).
        # Truncation is NOT treated as done (bootstrap); only termination is.
        s_next = np.where(done[:, None, None, None], info["final_obs"], nobs)
        m_next = np.where(done[:, None], info["final_mask"], info["action_mask"])
        buf.add_batch(obs_to_codes(obs), a, rew, obs_to_codes(s_next), m_next, term)
        obs, mask = nobs, info["action_mask"]
        board_steps += cfg.n_envs
        it = board_steps // cfg.n_envs

        if len(buf) >= cfg.learning_starts and it % cfg.train_every == 0:
            for _ in range(cfg.updates_per_step):
                loss, q_sa = dqn_loss(online, target, buf.sample(cfg.batch_size), cfg.gamma)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
                opt.step()
                n_updates += 1
                loss_val, mean_q = loss.item(), q_sa.mean().item()
                if n_updates % cfg.target_update_every == 0:
                    target.load_state_dict(online.state_dict())

        if board_steps >= next_eval or board_steps >= cfg.total_steps:
            next_eval += cfg.eval_every
            res = evaluate(DQNAgent(online), cfg.height, cfg.width, cfg.n_mines, cfg.eval_games)
            wr, wall = res["win_rate"], time.perf_counter() - t0
            writer.writerow([board_steps, f"{eps:.4f}", f"{loss_val:.5f}", f"{mean_q:.4f}", f"{wr:.4f}", f"{wall:.1f}"])
            f.flush()
            history.append((board_steps, wr))
            if cfg.verbose:
                print(f"steps={board_steps:>9} eps={eps:.3f} loss={loss_val:.4f} q={mean_q:.3f} "
                      f"eval_win={wr:.3f} t={wall:.0f}s", flush=True)
            if wr > best:
                best = wr
                torch.save(ckpt(), run_dir / "best.pt")
    torch.save(ckpt(), run_dir / "final.pt")
    f.close()
    return {"run_dir": str(run_dir), "best_win_rate": best, "history": history, "online": online}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for fld in dataclasses.fields(TrainConfig):
        if fld.name == "verbose":
            continue
        p.add_argument("--" + fld.name.replace("_", "-"), type=type(fld.default), default=fld.default)
    a = vars(p.parse_args(argv))
    train(TrainConfig(**a))


if __name__ == "__main__":
    main()
