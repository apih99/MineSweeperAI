"""DQN training with YAML configs, TensorBoard/CSV logging and exact checkpoint/resume.

    python -m minesweeper_rl.train --config configs/5x5.yaml [--run-dir runs/x] [--set key=value ...]
    python -m minesweeper_rl.train --resume runs/x [--set total_steps=2000000]

Run directory layout::

    config.yaml   fully resolved TrainConfig        meta.json    git commit, seed, versions, device
    metrics.csv   one averaged row per eval          events.*     TensorBoard
    best.pt       best eval win rate (weights)       final.pt     weights at the end
    latest.pt     full training state (resume)       replay-<board_steps>.npz  its replay buffer

``latest.pt`` + the replay file it names are written atomically (temp file + ``os.replace``;
replay first, then latest.pt), so a hard kill at any moment leaves the previous consistent pair.
Resuming from ``latest.pt`` continues bit-for-bit as if the run had never stopped.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import datetime
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from minesweeper_rl.agents.dqn import (
    ReplayBuffer,
    dqn_loss,
    epsilon_greedy_actions,
    make_checkpoint,
    make_streams,
    obs_to_codes,
)
from minesweeper_rl.agents.network import QNetwork
from minesweeper_rl.checkpoint import (
    arrays_to_tensors,
    atomic_savez,
    atomic_torch_save,
    atomic_write_text,
    get_generator_state,
    load_torch,
    prune_files,
    set_generator_state,
    tensors_to_arrays,
)
from minesweeper_rl.env.batched import BatchedMinesweeperEnv
from minesweeper_rl.evaluate import EVAL_SEED_BASE, evaluate_batched

CKPT_FORMAT = 2


@dataclass
class TrainConfig:
    """Single source of defaults; YAML files only list what differs."""

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
    eval_every: int = 50_000  # board-steps; also the logging interval
    eval_games: int = 500
    device: str = "auto"  # auto | cpu | cuda
    torch_threads: int = 2
    checkpoint_every_minutes: float = 10.0  # latest.pt cadence; 0 = only at end/interrupt
    max_wall_minutes: float | None = None  # stop cleanly (save latest.pt) after this much wall time in THIS process
    out_dir: str = "runs"
    verbose: bool = True


# Keys that may be changed when resuming (everything else would change the run's meaning,
# board size or net shape). NOTE: total_steps also moves the epsilon schedule (it decays over
# eps_decay_frac * total_steps), so extending a run changes epsilon from the resume point on.
SAFE_RESUME_KEYS = frozenset(
    {"total_steps", "eval_every", "eval_games", "checkpoint_every_minutes", "verbose", "torch_threads", "device",
     "max_wall_minutes"}
)

CSV_COLUMNS = [
    "board_steps", "epsilon", "loss", "mean_q", "train_win_rate", "episode_return", "episode_length",
    "eval_win_rate", "eval_ci_low", "eval_ci_high", "n_updates", "n_episodes",
    "wall_s", "board_steps_per_s", "updates_per_s",
]
# Columns that are a pure function of the seed (the rest are wall-clock measurements).
DETERMINISTIC_COLUMNS = CSV_COLUMNS[:-3]


# --------------------------------------------------------------------------- config

def epsilon_at(cfg: TrainConfig, board_steps: int) -> float:
    frac = min(1.0, board_steps / max(1.0, cfg.eps_decay_frac * cfg.total_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def config_from_dict(d: dict | None) -> TrainConfig:
    """Build a TrainConfig; unknown keys and wrongly-typed values raise (no silent typos)."""
    d = dict(d or {})
    fields = {f.name: f for f in dataclasses.fields(TrainConfig)}
    unknown = sorted(set(d) - set(fields))
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}; valid keys: {', '.join(fields)}")
    default = TrainConfig()
    for k, v in d.items():
        if k == "max_wall_minutes":  # Optional[float]: None (no limit) or a positive number
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
                raise ValueError(f"config key 'max_wall_minutes' must be null or a positive number, got {v!r}")
            d[k] = None if v is None else float(v)
            continue
        want = type(getattr(default, k))
        if want in (int, float) and isinstance(v, str):  # YAML 1.1 reads "1e-3" as a string
            try:
                v = d[k] = want(v)
            except ValueError:
                pass
        ok = (
            (want is bool and isinstance(v, bool))
            or (want is int and isinstance(v, int) and not isinstance(v, bool))
            or (want is float and isinstance(v, (int, float)) and not isinstance(v, bool))
            or (want is str and isinstance(v, str))
        )
        if not ok:
            raise ValueError(f"config key '{k}' must be {want.__name__}, got {v!r} ({type(v).__name__})")
        if want is float:
            d[k] = float(v)
    cfg = TrainConfig(**d)
    if cfg.device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto|cpu|cuda, got {cfg.device!r}")
    return cfg


def parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def load_config(path: str | Path | None, overrides: dict | list[str] | None = None) -> TrainConfig:
    d = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        d = loaded or {}
    ov = parse_overrides(overrides) if not isinstance(overrides, dict) else overrides
    return config_from_dict({**d, **ov})


def resume_config(run_dir: str | Path, overrides: dict | list[str] | None = None) -> TrainConfig:
    """Saved config of ``run_dir`` with overrides applied; only SAFE_RESUME_KEYS may change."""
    ov = parse_overrides(overrides) if not isinstance(overrides, dict) else dict(overrides)
    bad = sorted(set(ov) - SAFE_RESUME_KEYS)
    if bad:
        raise ValueError(f"cannot change {', '.join(bad)} when resuming; allowed: {', '.join(sorted(SAFE_RESUME_KEYS))}")
    ck = load_torch(Path(run_dir) / "latest.pt")
    return config_from_dict({**ck["config"], **ov})


# --------------------------------------------------------------------------- helpers

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(name)


def _git_info() -> dict:
    here = Path(__file__).resolve().parent
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
        if h.returncode != 0:
            return {"git_commit": None, "git_dirty": None}
        d = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, timeout=10)
        return {"git_commit": h.stdout.strip(), "git_dirty": bool(d.stdout.strip()) if d.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _fmt(x: float) -> str:
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.6g}"


def _truncate_csv(path: Path, max_step: int) -> None:
    """Drop rows logged after the checkpoint (they will be regenerated identically)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    keep = [r for r in rows[1:] if r and int(r[0]) <= max_step]
    if len(keep) != len(rows) - 1:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(rows[0])
            w.writerows(keep)
        os.replace(tmp, path)


class _StopFlag:
    """SIGINT/SIGTERM set a flag checked between iterations so latest.pt is saved in a consistent state."""

    def __init__(self):
        self.stop = False
        self._old = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._old[sig] = signal.signal(sig, self._handle)
            except ValueError:  # not in the main thread
                pass
        return self

    def _handle(self, signum, frame):
        self.stop = True

    def __exit__(self, *exc):
        for sig, h in self._old.items():
            signal.signal(sig, h)


# --------------------------------------------------------------------------- training

def train(cfg: TrainConfig, run_dir: str | Path | None = None, resume: bool = False,
          stop_at: int | None = None) -> dict:
    """Train (or resume) a run. ``stop_at``: stop early at this many board-steps, saving
    latest.pt (used by tests; the run can be resumed later)."""
    assert cfg.train_seed_base + cfg.n_envs < EVAL_SEED_BASE, "training seeds overlap eval range"
    run_dir = Path(run_dir) if run_dir is not None else Path(cfg.out_dir) / cfg.name
    device = resolve_device(cfg.device)
    torch.set_num_threads(cfg.torch_threads)
    N, HW = cfg.n_envs, cfg.height * cfg.width

    # ---- build everything (fresh), then overwrite from the checkpoint when resuming
    explore_rng, buffer_rng, torch_seed = make_streams(cfg.seed)
    torch.manual_seed(torch_seed)
    net_cfg = {"channels": cfg.channels, "n_layers": cfg.n_layers}
    online = QNetwork(**net_cfg).to(device)
    target = copy.deepcopy(online)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer_size, cfg.height, cfg.width, buffer_rng)
    env = BatchedMinesweeperEnv(N, cfg.height, cfg.width, cfg.n_mines)

    board_steps = n_updates = n_episodes = 0
    next_eval = cfg.eval_every
    best = -1.0
    history: list[list[float]] = []
    wall_base = 0.0  # wall seconds accumulated by earlier sessions
    ep_ret = np.zeros(N, np.float64)  # running per-board episode return / length
    ep_len = np.zeros(N, np.int64)
    acc = torch.zeros(2, dtype=torch.float64, device=device)  # interval sums: loss, mean_q
    ia = {"n_upd": 0, "n_ep": 0, "ret": 0.0, "len": 0.0, "wins": 0.0,  # interval accumulators
          "steps0": 0, "updates0": 0, "wall0": 0.0, "excl": 0.0}

    if resume:
        ck = load_torch(run_dir / "latest.pt")
        if ck.get("format") != CKPT_FORMAT:
            raise ValueError(f"unsupported latest.pt format {ck.get('format')!r}")
        for k in dataclasses.fields(TrainConfig):
            if k.name not in SAFE_RESUME_KEYS and ck["config"].get(k.name) != getattr(cfg, k.name):
                raise ValueError(f"config '{k.name}' differs from the checkpoint; cannot change it on resume")
        with np.load(run_dir / ck["replay_file"], allow_pickle=False) as rz:
            rd = {k: rz[k] for k in rz.files}
        if int(rd["board_steps"]) != ck["board_steps"] or int(rd["n_updates"]) != ck["n_updates"]:
            raise ValueError("replay file does not match latest.pt (step counters differ)")
        buf.load_arrays(rd)
        online.load_state_dict(ck["state_dict"])
        target.load_state_dict(ck["target_state_dict"])
        opt.load_state_dict(ck["optimizer"])
        board_steps, n_updates, n_episodes = ck["board_steps"], ck["n_updates"], ck["n_episodes"]
        next_eval, best, history, wall_base = ck["next_eval"], ck["best"], ck["history"], ck["wall_s"]
        ep_ret, ep_len = ck["ep_ret"].numpy().copy(), ck["ep_len"].numpy().copy()
        acc = torch.tensor(ck["acc_loss_q"], dtype=torch.float64, device=device)
        ia = dict(ck["interval"])
        set_generator_state(explore_rng, ck["rng"]["explore"])
        set_generator_state(buffer_rng, ck["rng"]["buffer"])
        torch.set_rng_state(ck["rng"]["torch_cpu"])
        if ck["rng"]["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng"]["torch_cuda"])
        env.reset(seeds=cfg.train_seed_base)  # allocates generators; overwritten just below
        env.set_state(tensors_to_arrays(ck["env"]))
        obs, mask = env.current()
        _truncate_csv(run_dir / "metrics.csv", board_steps)
    else:
        if (run_dir / "latest.pt").exists() or (run_dir / "metrics.csv").exists():
            raise FileExistsError(f"{run_dir} already holds a run; use --resume or another --run-dir")
        run_dir.mkdir(parents=True, exist_ok=True)
        obs, info = env.reset(seeds=cfg.train_seed_base)
        mask = info["action_mask"]
        (run_dir / "metrics.csv").write_text(",".join(CSV_COLUMNS) + "\n")

    # ---- run metadata
    atomic_write_text(run_dir / "config.yaml", yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    meta_path = run_dir / "meta.json"
    dev_desc = str(device) + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    git = _git_info()
    if resume and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta.setdefault("resumes", []).append({"time": _now(), "board_steps": board_steps, "device": dev_desc, **git})
    else:
        meta = {"start_time": _now(), "seed": cfg.seed, **git, "torch": torch.__version__,
                "numpy": np.__version__, "python": sys.version.split()[0], "platform": platform.platform(),
                "device": dev_desc, "resumes": []}
    atomic_write_text(meta_path, json.dumps(meta, indent=2))

    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(str(run_dir), purge_step=(board_steps + 1) if resume else None)
    csv_f = open(run_dir / "metrics.csv", "a", newline="")
    writer = csv.writer(csv_f)

    t_start = time.perf_counter()
    wall = lambda: wall_base + (time.perf_counter() - t_start)  # noqa: E731
    if not resume:
        ia["wall0"] = 0.0

    def save_latest() -> None:
        t = time.perf_counter()
        prune_files(run_dir, "*.tmp")
        replay_name = f"replay-{board_steps}.npz"
        arrays = buf.to_arrays()
        arrays["board_steps"], arrays["n_updates"] = np.array(board_steps), np.array(n_updates)
        atomic_savez(run_dir / replay_name, arrays)
        delay = float(os.environ.get("MSRL_TEST_CKPT_DELAY_S", "0") or 0)  # test hook: widen the kill window
        if delay:
            time.sleep(delay)
        ck = {
            "format": CKPT_FORMAT,
            "state_dict": _to_cpu(online.state_dict()),  # so DQNAgent.load works on latest.pt too
            "net": dict(net_cfg),
            "target_state_dict": _to_cpu(target.state_dict()),
            "optimizer": _to_cpu(opt.state_dict()),
            "board_steps": board_steps, "n_updates": n_updates, "n_episodes": n_episodes,
            "next_eval": next_eval, "best": best, "history": history, "wall_s": wall(),
            "rng": {
                "explore": get_generator_state(explore_rng),
                "buffer": get_generator_state(buffer_rng),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
            "env": arrays_to_tensors(env.get_state()),
            "ep_ret": torch.from_numpy(ep_ret.copy()), "ep_len": torch.from_numpy(ep_len.copy()),
            "acc_loss_q": acc.tolist(), "interval": dict(ia),
            "replay_file": replay_name,
            "config": dataclasses.asdict(cfg),
        }
        atomic_torch_save(ck, run_dir / "latest.pt")
        prune_files(run_dir, "replay*.npz", keep=replay_name)
        ia["excl"] += time.perf_counter() - t

    def log_and_eval(eps: float) -> None:
        nonlocal best, acc
        t = time.perf_counter()
        res = evaluate_batched(online, cfg.height, cfg.width, cfg.n_mines, cfg.eval_games)
        ia["excl"] += time.perf_counter() - t
        wr, (lo, hi) = res["win_rate"], res["win_rate_ci95"]
        nan = float("nan")
        n_upd, n_ep = ia["n_upd"], ia["n_ep"]
        loss_avg, q_avg = (acc[0].item() / n_upd, acc[1].item() / n_upd) if n_upd else (nan, nan)
        tw, er, el = ((ia["wins"] / n_ep, ia["ret"] / n_ep, ia["len"] / n_ep) if n_ep else (nan, nan, nan))
        now = wall()
        train_s = max(now - ia["wall0"] - ia["excl"], 1e-9)
        sps = (board_steps - ia["steps0"]) / train_s
        ups = (n_updates - ia["updates0"]) / train_s
        row = [board_steps, _fmt(eps), _fmt(loss_avg), _fmt(q_avg), _fmt(tw), _fmt(er), _fmt(el),
               _fmt(wr), _fmt(lo), _fmt(hi), n_updates, n_episodes, f"{now:.1f}", f"{sps:.1f}", f"{ups:.1f}"]
        writer.writerow(row)
        csv_f.flush()
        for tag, v in [("train/epsilon", eps), ("train/loss", loss_avg), ("train/mean_q", q_avg),
                       ("train/episode_return", er), ("train/episode_length", el), ("train/win_rate", tw),
                       ("eval/win_rate", wr), ("eval/win_rate_ci_low", lo), ("eval/win_rate_ci_high", hi),
                       ("perf/board_steps_per_s", sps), ("perf/updates_per_s", ups)]:
            if not (isinstance(v, float) and math.isnan(v)):
                tb.add_scalar(tag, v, board_steps)
        tb.flush()
        history.append([board_steps, wr])
        if cfg.verbose:
            print(f"steps={board_steps:>9} eps={eps:.3f} loss={loss_avg:.4f} q={q_avg:.3f} train_win={tw:.3f} "
                  f"eval_win={wr:.3f} sps={sps:.0f} t={now:.0f}s", flush=True)
        if wr > best:
            best = wr
            atomic_torch_save(make_checkpoint(online, net_cfg, {"config": dataclasses.asdict(cfg)}),
                              run_dir / "best.pt")
        # reset interval accumulators
        acc = torch.zeros(2, dtype=torch.float64, device=device)
        ia.update(n_upd=0, n_ep=0, ret=0.0, len=0.0, wins=0.0, steps0=board_steps, updates0=n_updates,
                  wall0=wall(), excl=0.0)

    stop = cfg.total_steps if stop_at is None else min(stop_at, cfg.total_steps)
    ckpt_s = cfg.checkpoint_every_minutes * 60.0
    last_ckpt = time.perf_counter()
    interrupted = False
    wall_limit_s = None if cfg.max_wall_minutes is None else cfg.max_wall_minutes * 60.0
    t_proc = time.perf_counter()  # wall limit counts this process only (a resumed run gets a fresh budget)
    wall_stopped = False
    try:
        with _StopFlag() as flag:
            while board_steps < stop:
                if flag.stop:
                    interrupted = True
                    break
                if wall_limit_s is not None and time.perf_counter() - t_proc >= wall_limit_s:
                    interrupted = wall_stopped = True
                    break
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
                board_steps += N
                it = board_steps // N

                ep_ret += rew
                ep_len += 1
                if done.any():
                    d = np.flatnonzero(done)
                    n_episodes += len(d)
                    ia["n_ep"] += len(d)
                    ia["ret"] += float(ep_ret[d].sum())
                    ia["len"] += float(ep_len[d].sum())
                    ia["wins"] += float(info["won"][d].sum())
                    ep_ret[d] = 0.0
                    ep_len[d] = 0

                if len(buf) >= cfg.learning_starts and it % cfg.train_every == 0:
                    for _ in range(cfg.updates_per_step):
                        loss, q_sa = dqn_loss(online, target, buf.sample(cfg.batch_size, device), cfg.gamma)
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
                        opt.step()
                        n_updates += 1
                        ia["n_upd"] += 1
                        acc[0] += loss.detach()
                        acc[1] += q_sa.mean()
                        if n_updates % cfg.target_update_every == 0:
                            target.load_state_dict(online.state_dict())

                if board_steps >= next_eval or board_steps >= cfg.total_steps:
                    next_eval += cfg.eval_every
                    log_and_eval(eps)
                if ckpt_s > 0 and time.perf_counter() - last_ckpt >= ckpt_s:
                    save_latest()
                    last_ckpt = time.perf_counter()
            interrupted = interrupted or flag.stop
    except KeyboardInterrupt:  # only if the signal handler could not be installed (non-main thread)
        interrupted = True
    finished = board_steps >= cfg.total_steps and not interrupted
    if finished:
        atomic_torch_save(make_checkpoint(online, net_cfg, {"config": dataclasses.asdict(cfg)}), run_dir / "final.pt")
    save_latest()
    csv_f.close()
    tb.close()
    if cfg.verbose and wall_stopped:
        print(f"max_wall_minutes={cfg.max_wall_minutes} reached at {board_steps} board-steps; saved "
              f"{run_dir / 'latest.pt'} (resume with --resume {run_dir})", flush=True)
    elif cfg.verbose and interrupted:
        print(f"interrupted at {board_steps} board-steps; saved {run_dir / 'latest.pt'}", flush=True)
    return {"run_dir": str(run_dir), "best_win_rate": best, "history": history, "online": online,
            "board_steps": board_steps, "interrupted": interrupted, "finished": finished}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML file with TrainConfig keys (missing keys use dataclass defaults)")
    p.add_argument("--run-dir", help="output directory (default: <out_dir>/<name>)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key (repeatable)")
    p.add_argument("--resume", metavar="RUN_DIR", help="continue RUN_DIR from its latest.pt")
    a = p.parse_args(argv)
    try:
        if a.resume:
            if a.config or a.run_dir:
                p.error("--resume uses the run's saved config; do not combine with --config/--run-dir")
            cfg = resume_config(a.resume, a.set)
            res = train(cfg, run_dir=a.resume, resume=True)
        else:
            cfg = load_config(a.config, a.set)
            res = train(cfg, run_dir=a.run_dir)
    except (ValueError, FileExistsError) as e:
        p.error(str(e))
    print(f"done: run_dir={res['run_dir']} board_steps={res['board_steps']} best_eval_win={res['best_win_rate']:.4f}")


if __name__ == "__main__":
    main()
