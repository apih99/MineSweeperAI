"""Kaggle GPU kernel: train (or resume) the 9x9/10-mine DQN, then evaluate it.

Runs ON Kaggle. Pushed by ``scripts/kaggle_run.py``, which rewrites the constants below
(PINNED_SHA, RUN_NAME, SET_OVERRIDES, RESUME_OVERRIDES) in a temporary copy -- the committed file
keeps the placeholders. Everything is cloned from GitHub at PINNED_SHA so a run is reproducible.

Resume across sessions: attach this kernel's previous version as ``kernel_sources`` (the helper's
``--resume`` flag does this); its output appears under /kaggle/input/... and, if it holds
runs/<RUN_NAME>/latest.pt, that run directory is copied to /kaggle/working/runs/ and resumed.
"""
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------- constants
PINNED_SHA = "__PINNED_SHA__"  # filled in by scripts/kaggle_run.py
REPO_URL = "https://github.com/apih99/MineSweeperAI.git"
CONFIG = "configs/9x9.yaml"
RUN_NAME = "dqn_9x9_10"
MAX_WALL_MINUTES = 690  # Kaggle GPU sessions are limited to 12 h; leave time for save + eval
SET_OVERRIDES = {}      # extra --set key=value for a FRESH run, e.g. {"total_steps": 500000}
RESUME_OVERRIDES = {}   # --set for a resumed run (only resume-safe keys: total_steps, eval_every, ...)
EVAL_GAMES = 10_000
UNSEEN_SEED_BASE = 5_000_000

WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")  # outside /kaggle/working so it is not part of the kernel output
RUNS = WORK / "runs"
RESULTS = WORK / "results"


def sh(cmd, check=True, cwd=None):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if check and r.returncode != 0:
        sys.exit(f"FAILED (exit {r.returncode}): {' '.join(cmd)}")
    return r.returncode


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


# ----------------------------------------------------------------------------- 1. environment
banner("1. environment")
sh(["nvidia-smi"], check=False)
print("python", sys.version.replace("\n", " "))
import torch  # noqa: E402  (Kaggle's CUDA build; never reinstalled below)

print("torch", torch.__version__, "cuda build", torch.version.cuda, "cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("FATAL: CUDA is not available in this kernel (GPU not enabled?). Refusing to train on CPU.")
print("gpu:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
print("cpu cores:", os.cpu_count(), platform.platform())
if PINNED_SHA.startswith("__"):
    sys.exit("FATAL: PINNED_SHA was not filled in; push with scripts/kaggle_run.py")

# ----------------------------------------------------------------------------- 2. clone + 3. install
banner(f"2. clone {REPO_URL} @ {PINNED_SHA}")
if REPO.exists():
    shutil.rmtree(REPO)
sh(["git", "clone", REPO_URL, REPO])
sh(["git", "checkout", "--detach", PINNED_SHA], cwd=REPO)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
assert head.startswith(PINNED_SHA), (head, PINNED_SHA)
print("checked out", head)

banner("3. install (keeping Kaggle's CUDA torch)")
# Only missing packages get installed (no --upgrade); torch is deliberately not in this list.
sh([sys.executable, "-m", "pip", "install", "-q", "gymnasium", "pyyaml", "tensorboard", "matplotlib",
    "pillow", "scipy", "numpy", "pytest"])
flags = ["--no-deps"]
if sys.version_info >= (3, 13):  # pyproject says requires-python <3.13
    print("NOTE: Python >= 3.13 -> installing with --ignore-requires-python (not covered by the project's CI)")
    flags.append("--ignore-requires-python")
sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".", *flags], cwd=REPO)
out = subprocess.check_output([sys.executable, "-c", "import torch;print(torch.__version__, torch.cuda.is_available())"],
                              text=True).strip()
print("torch after install:", out)
assert out == f"{torch.__version__} True", "torch was replaced or lost CUDA"

# ----------------------------------------------------------------------------- 4. sanity tests
banner("4. pytest -m 'not slow'")
sh([sys.executable, "-m", "pytest", "-m", "not slow", "-q", "-x", "-p", "no:cacheprovider"], cwd=REPO)

# ----------------------------------------------------------------------------- 5. train / resume
banner("5. train")
RUNS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
run_dir = RUNS / RUN_NAME
print("/kaggle/input contents:")
sh(["find", "/kaggle/input", "-maxdepth", "5"], check=False)
prev = sorted(glob.glob(f"/kaggle/input/**/runs/{RUN_NAME}/latest.pt", recursive=True))
print("previous latest.pt candidates:", prev)


def set_args(d):
    out = []
    for k, v in d.items():
        out += ["--set", f"{k}={v if isinstance(v, str) else json.dumps(v)}"]
    return out


py = [sys.executable, "-m", "minesweeper_rl.train"]
t0 = time.time()
if prev:
    src = Path(prev[-1]).parent
    print(f"RESUMING from {src}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(src, run_dir)
    ov = {"max_wall_minutes": MAX_WALL_MINUTES, "device": "cuda", **RESUME_OVERRIDES}
    sh(py + ["--resume", run_dir, *set_args(ov)], cwd=REPO)
else:
    print("starting a FRESH run")
    ov = {"max_wall_minutes": MAX_WALL_MINUTES, "device": "cuda", **SET_OVERRIDES}
    sh(py + ["--config", CONFIG, "--run-dir", run_dir, *set_args(ov)], cwd=REPO)
print(f"training process took {(time.time() - t0) / 60:.1f} min")

# ----------------------------------------------------------------------------- 7. trim output
# keep latest.pt + the replay file it names (needed to resume); drop any other replay file
ck = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
for f in run_dir.glob("replay*.npz"):
    if f.name != ck["replay_file"]:
        print("deleting stale", f.name)
        f.unlink()
for f in run_dir.glob("*.tmp"):
    f.unlink()
finished = (run_dir / "final.pt").exists()
print("finished:", finished, "| board_steps:", ck["board_steps"], "| files:", sorted(p.name for p in run_dir.iterdir()))

# ----------------------------------------------------------------------------- 6. evaluate
banner("6. evaluate (10k standard seeds + 10k unseen seeds)")
import yaml  # noqa: E402
from minesweeper_rl.agents.dqn import DQNAgent  # noqa: E402
from minesweeper_rl.evaluate import EVAL_SEED_BASE, evaluate_batched  # noqa: E402

summary = {"pinned_sha": PINNED_SHA, "run": RUN_NAME, "finished": finished, "board_steps": int(ck["board_steps"]),
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": sys.version.split()[0],
           "evals": {}}
cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
H, W, M = cfg["height"], cfg["width"], cfg["n_mines"]
for which in ("best", "final"):
    path = run_dir / f"{which}.pt"
    if not path.exists():
        print(f"{which}.pt missing (run not finished?) -> skipped")
        continue
    agent = DQNAgent(checkpoint=str(path))
    agent.net.to("cuda").eval()
    for tag, base in (("standard", EVAL_SEED_BASE), ("unseen", UNSEEN_SEED_BASE)):
        res = evaluate_batched(agent, H, W, M, EVAL_GAMES, seed_base=base)
        res.update(checkpoint=f"{which}.pt", seed_base=base, pinned_sha=PINNED_SHA)
        (RESULTS / f"{which}_{tag}.json").write_text(json.dumps(res, indent=2))
        lo, hi = res["win_rate_ci95"]
        print(f"{which:5s} {tag:8s} seeds {base}+: win {100 * res['win_rate']:.2f}% "
              f"(95% CI {100 * lo:.2f}-{100 * hi:.2f}) safe-revealed {100 * res['mean_safe_revealed_frac']:.1f}% "
              f"steps {res['mean_steps']:.1f}", flush=True)
        summary["evals"][f"{which}_{tag}"] = {"win_rate": res["win_rate"], "ci95": res["win_rate_ci95"]}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

banner("summary")
print(json.dumps(summary, indent=2))
sh(["du", "-sh", WORK], check=False)
print("DONE")
