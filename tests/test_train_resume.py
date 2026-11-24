import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import minesweeper_rl.train as train_mod
from minesweeper_rl.agents.dqn import DQNAgent, make_checkpoint
from minesweeper_rl.agents.network import QNetwork
from minesweeper_rl.checkpoint import atomic_write, load_torch
from minesweeper_rl.train import (
    DETERMINISTIC_COLUMNS,
    TrainConfig,
    config_from_dict,
    load_config,
    resume_config,
    train,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# The M4 run (runs/dqn_5x5_3/best.pt "config", with num_threads renamed torch_threads).
M4_CONFIG = {
    "name": "dqn_5x5_3", "height": 5, "width": 5, "n_mines": 3, "seed": 0, "train_seed_base": 0,
    "n_envs": 64, "total_steps": 400000, "channels": 64, "n_layers": 5, "buffer_size": 100000,
    "batch_size": 64, "lr": 0.0003, "gamma": 0.1, "grad_clip": 10.0, "train_every": 1,
    "updates_per_step": 1, "learning_starts": 5000, "target_update_every": 1000, "eps_start": 1.0,
    "eps_end": 0.01, "eps_decay_frac": 0.25, "eval_every": 20000, "eval_games": 500,
    "torch_threads": 2, "out_dir": "runs", "verbose": True,
}


# ------------------------------------------------------------------ config
def test_5x5_yaml_is_the_m4_run():
    cfg = load_config(CONFIGS / "5x5.yaml")
    d = dataclasses.asdict(cfg)
    assert {k: d[k] for k in M4_CONFIG} == M4_CONFIG
    assert d["device"] == "auto" and d["checkpoint_every_minutes"] == 10
    # and the dataclass defaults + the two M4 CLI flags give the same thing
    assert cfg == TrainConfig(total_steps=400000, eval_every=20000)


@pytest.mark.parametrize("f", ["9x9.yaml", "16x16.yaml"])
def test_big_configs_load(f):
    cfg = load_config(CONFIGS / f)
    assert cfg.n_envs == 256 and cfg.eval_games == 1000


def test_unknown_key_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown config key"):
        config_from_dict({"totl_steps": 5})
    p = tmp_path / "c.yaml"
    p.write_text("n_env: 4\n")
    with pytest.raises(ValueError, match="n_env"):
        load_config(p)
    with pytest.raises(ValueError, match="lerning_starts"):
        load_config(None, ["lerning_starts=3"])


def test_overrides_and_types():
    cfg = load_config(CONFIGS / "5x5.yaml", ["total_steps=1000", "lr=1e-3", "device=cpu", "verbose=false"])
    assert (cfg.total_steps, cfg.lr, cfg.device, cfg.verbose) == (1000, 1e-3, "cpu", False)
    assert isinstance(load_config(None, ["gamma=1"]).gamma, float)  # int accepted for float fields
    with pytest.raises(ValueError, match="must be int"):
        load_config(None, ["n_envs=2.5"])
    with pytest.raises(ValueError, match="key=value"):
        load_config(None, ["n_envs"])
    with pytest.raises(ValueError, match="device"):
        load_config(None, ["device=tpu"])


# ------------------------------------------------------------------ atomic writes
def test_atomic_write_keeps_old_file_on_failure(tmp_path):
    p = tmp_path / "f.bin"
    atomic_write(p, lambda f: f.write(b"old"))

    def boom(f):
        f.write(b"half")
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError):
        atomic_write(p, boom)
    assert p.read_bytes() == b"old"
    atomic_write(p, lambda f: f.write(b"new"))
    assert p.read_bytes() == b"new" and not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------------------ exact resume
def tiny_cfg(tmp_path, **kw):
    base = dict(
        name="r", out_dir=str(tmp_path), seed=3, n_envs=8, total_steps=2000, channels=8, n_layers=2,
        buffer_size=600, batch_size=16, learning_starts=100, target_update_every=20,
        eval_every=400, eval_games=20, device="cpu", torch_threads=1, checkpoint_every_minutes=0,
        verbose=False,
    )
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def read_rows(run_dir):
    lines = (run_dir / "metrics.csv").read_text().strip().splitlines()
    header = lines[0].split(",")
    idx = [header.index(c) for c in DETERMINISTIC_COLUMNS]
    return [[r.split(",")[i] for i in idx] for r in lines[1:]]


def replay_arrays(run_dir):
    ck = load_torch(run_dir / "latest.pt")
    with np.load(run_dir / ck["replay_file"]) as z:
        return {k: z[k] for k in z.files}


def assert_same_run(a: Path, b: Path):
    ca, cb = load_torch(a / "latest.pt"), load_torch(b / "latest.pt")
    for k in ca["state_dict"]:
        assert torch.equal(ca["state_dict"][k], cb["state_dict"][k]), f"online {k}"
        assert torch.equal(ca["target_state_dict"][k], cb["target_state_dict"][k]), f"target {k}"
    assert ca["optimizer"]["param_groups"] == cb["optimizer"]["param_groups"]
    assert ca["optimizer"]["state"].keys() == cb["optimizer"]["state"].keys()
    for i, st in ca["optimizer"]["state"].items():
        for k, v in st.items():
            assert torch.equal(v, cb["optimizer"]["state"][i][k]), f"opt {i}.{k}"
    for k in ("board_steps", "n_updates", "n_episodes", "next_eval", "best", "history", "acc_loss_q"):
        assert ca[k] == cb[k], k
    assert ca["rng"]["explore"] == cb["rng"]["explore"] and ca["rng"]["buffer"] == cb["rng"]["buffer"]
    assert ca["env"]["rng_states"] == cb["env"]["rng_states"]
    for k in ("mines", "revealed", "counts", "steps", "_obs_state"):
        assert torch.equal(ca["env"][k], cb["env"][k]), f"env {k}"
    ra, rb = replay_arrays(a), replay_arrays(b)
    assert ra.keys() == rb.keys()
    for k in ra:
        assert np.array_equal(ra[k], rb[k]), f"replay {k}"
    assert read_rows(a) == read_rows(b)


def run_interrupted(tmp_path, name, cfg, stop=1000):
    d = tmp_path / name
    train(cfg, run_dir=d, stop_at=stop)
    assert (d / "latest.pt").exists() and not (d / "final.pt").exists()
    # fresh objects: nothing but the files in `d` carries over
    resumed = train(resume_config(d, {}), run_dir=d, resume=True)
    return d, resumed


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    torch.use_deterministic_algorithms(True)
    tmp = tmp_path_factory.mktemp("ref")
    d = tmp / "A"
    train(tiny_cfg(tmp), run_dir=d)
    torch.use_deterministic_algorithms(False)
    return d


def test_exact_resume(tmp_path, reference):
    b, res = run_interrupted(tmp_path, "B", tiny_cfg(tmp_path))
    assert res["finished"] and res["board_steps"] == 2000
    assert_same_run(reference, b)
    rows = read_rows(b)
    assert [r[0] for r in rows] == ["400", "800", "1200", "1600", "2000"]  # no duplicate/missing rows
    assert load_torch(b / "latest.pt")["rng"]["torch_cuda"] is None


def test_resume_survives_csv_rows_past_the_checkpoint(tmp_path, reference):
    """Killed after logging an eval row but before the next checkpoint: the stale row is dropped."""
    cfg = tiny_cfg(tmp_path)
    d = tmp_path / "B"
    train(cfg, run_dir=d, stop_at=1000)
    with open(d / "metrics.csv", "a") as f:
        f.write("1200,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0\n")  # as if written after the checkpoint
    train(resume_config(d, {}), run_dir=d, resume=True)
    assert_same_run(reference, d)


@pytest.mark.parametrize("what", ["explore", "buffer", "env"])
def test_exact_resume_detects_unrestored_rng(tmp_path, reference, monkeypatch, what):
    """Negative control: skipping any single RNG restore must make the comparison fail."""
    if what in ("explore", "buffer"):
        real, calls = train_mod.set_generator_state, []

        def patched(g, state):
            calls.append(1)
            if len(calls) == (1 if what == "explore" else 2):  # restore order: explore, buffer
                return
            real(g, state)

        monkeypatch.setattr(train_mod, "set_generator_state", patched)
    else:
        from gymnasium.utils import seeding

        from minesweeper_rl.env.batched import BatchedMinesweeperEnv

        real_set = BatchedMinesweeperEnv.set_state

        def patched_set(self, st):
            real_set(self, st)
            self._rngs = [seeding.np_random(777 + i)[0] for i in range(self.n_envs)]

        monkeypatch.setattr(BatchedMinesweeperEnv, "set_state", patched_set)
    b, _ = run_interrupted(tmp_path, "B", tiny_cfg(tmp_path))
    with pytest.raises(AssertionError):
        assert_same_run(reference, b)


def test_crash_between_replay_and_latest_write_is_harmless(tmp_path, reference, monkeypatch):
    cfg = tiny_cfg(tmp_path)
    d = tmp_path / "B"
    train(cfg, run_dir=d, stop_at=400)
    real = train_mod.atomic_torch_save

    def die_on_latest(obj, path):
        if Path(path).name == "latest.pt":
            raise RuntimeError("killed after the replay file was written")
        real(obj, path)

    monkeypatch.setattr(train_mod, "atomic_torch_save", die_on_latest)
    with pytest.raises(RuntimeError):
        train(resume_config(d, {}), run_dir=d, resume=True, stop_at=800)
    monkeypatch.setattr(train_mod, "atomic_torch_save", real)
    assert load_torch(d / "latest.pt")["board_steps"] == 400  # old pair intact
    assert (d / "replay-800.npz").exists()  # orphan from the failed save
    train(resume_config(d, {}), run_dir=d, resume=True)
    assert_same_run(reference, d)
    assert [p.name for p in d.glob("replay*")] == ["replay-2000.npz"]


def test_resume_guards(tmp_path):
    cfg = tiny_cfg(tmp_path, total_steps=800)
    d = tmp_path / "B"
    train(cfg, run_dir=d)
    with pytest.raises(FileExistsError):
        train(cfg, run_dir=d)  # would clobber an existing run
    with pytest.raises(ValueError, match="cannot change"):
        resume_config(d, {"height": 6})
    with pytest.raises(ValueError, match="cannot change"):
        resume_config(d, ["channels=16"])
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        train(dataclasses.replace(cfg, n_layers=3), run_dir=d, resume=True)
    # extending a finished run is allowed
    res = train(resume_config(d, ["total_steps=1200"]), run_dir=d, resume=True)
    assert res["board_steps"] == 1200


def test_run_dir_artifacts_and_meta(tmp_path):
    d = tmp_path / "A"
    train(tiny_cfg(tmp_path, total_steps=800), run_dir=d)
    assert yaml.safe_load((d / "config.yaml").read_text())["seed"] == 3
    meta = json.loads((d / "meta.json").read_text())
    for k in ("git_commit", "git_dirty", "seed", "start_time", "torch", "numpy", "device"):
        assert k in meta
    assert meta["seed"] == 3 and meta["device"] == "cpu"
    assert list(d.glob("events.out.tfevents.*"))
    assert (d / "best.pt").exists() and (d / "final.pt").exists()
    header = (d / "metrics.csv").read_text().splitlines()[0].split(",")
    assert {"loss", "mean_q", "train_win_rate", "eval_ci_low", "board_steps_per_s"} <= set(header)


# ------------------------------------------------------------------ weights_only
def test_checkpoints_load_weights_only(tmp_path):
    d = tmp_path / "A"
    train(tiny_cfg(tmp_path, total_steps=800), run_dir=d)
    for name in ("latest.pt", "best.pt", "final.pt"):
        ck = torch.load(d / name, weights_only=True)
        assert "state_dict" in ck and ck["net"] == {"channels": 8, "n_layers": 2}
    agent = DQNAgent.load(str(d / "latest.pt"))  # latest.pt doubles as a model checkpoint
    assert agent.act(np.zeros((10, 5, 5), np.float32), np.ones(25, bool)) in range(25)


def test_m4_format_checkpoint_still_loads(tmp_path):
    net = QNetwork(channels=8, n_layers=2)
    ck = make_checkpoint(net, {"channels": 8, "n_layers": 2}, {"config": dataclasses.asdict(TrainConfig())})
    torch.save(ck, tmp_path / "m4.pt")
    DQNAgent.load(str(tmp_path / "m4.pt"))


def test_weights_only_rejects_arbitrary_objects(tmp_path):
    torch.save({"x": np.zeros(3)}, tmp_path / "bad.pt")  # numpy array = pickled object
    with pytest.raises(Exception):
        torch.load(tmp_path / "bad.pt", weights_only=True)


# ------------------------------------------------------------------ real process (CLI)
@pytest.mark.slow
def test_cli_train_and_resume(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(train_mod.__file__).parents[1])}
    d = tmp_path / "cli"
    common = ["--set", "n_envs=8", "--set", "channels=8", "--set", "n_layers=2", "--set", "buffer_size=600",
              "--set", "batch_size=16", "--set", "learning_starts=100", "--set", "eval_every=400",
              "--set", "eval_games=20", "--set", "device=cpu", "--set", "verbose=false"]
    r = subprocess.run([sys.executable, "-m", "minesweeper_rl.train", "--config", str(CONFIGS / "5x5.yaml"),
                        "--run-dir", str(d), "--set", "total_steps=800", *common],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "minesweeper_rl.train", "--resume", str(d),
                        "--set", "total_steps=1200"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert load_torch(d / "latest.pt")["board_steps"] == 1200
    r = subprocess.run([sys.executable, "-m", "minesweeper_rl.train", "--config", str(CONFIGS / "5x5.yaml"),
                        "--set", "bogus=1"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "unknown config key" in r.stderr
