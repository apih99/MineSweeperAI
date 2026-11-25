"""Kaggle GPU kernel: train (or resume) one DQN run, or a LIST of runs (PARALLEL at a time, one per GPU).

Runs ON Kaggle. Pushed by ``scripts/kaggle_run.py``, which rewrites the constants below
(PINNED_SHA, RUN_NAME, SET_OVERRIDES, RESUME_OVERRIDES) in a temporary copy -- the committed file
keeps the placeholders. Everything is cloned from GitHub at PINNED_SHA so a run is reproducible.

Resume across sessions: attach this kernel's previous version as ``kernel_sources`` (the helper's
``--resume`` flag does this); its output appears under /kaggle/input/... and, if it holds
runs/<RUN_NAME>/latest.pt, that run directory is copied to /kaggle/working/runs/ and resumed.

Two modes. RUN_LIST empty (M6): the single run (CONFIG, RUN_NAME, SET_OVERRIDES) and the original
results/ layout. RUN_LIST non-empty (M8): each entry {"name", "config", "set"} gets its own run
directory; PARALLEL run at a time (worker slot i pins CUDA_VISIBLE_DEVICES=i; the NumPy env is
CPU-bound, so this only pays off when there are spare cores). MAX_WALL_MINUTES is then the training
budget for ALL runs together: a run started later gets what is left, stops cleanly (latest.pt) and is
resumed next session. Finished runs lose their replay file (only needed to resume) and are evaluated
on unseen seeds into results/<name>/.
"""
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
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
RUN_LIST = []           # [] = single-run mode; else [{"name": ..., "config": "configs/...yaml", "set": {...}}, ...]
PARALLEL = 2            # concurrent runs in multi-run mode (capped by the number of GPUs)

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
N_GPU = torch.cuda.device_count()
print("gpu:", torch.cuda.get_device_name(0), "| count:", N_GPU)
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
sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".", *flags], cwd=REPO)
out = subprocess.check_output([sys.executable, "-c", "import torch;print(torch.__version__, torch.cuda.is_available())"],
                              text=True).strip()
print("torch after install:", out)
assert out == f"{torch.__version__} True", "torch was replaced or lost CUDA"
sys.path.insert(0, str(REPO / "src"))  # the editable install happened after this interpreter started

# ----------------------------------------------------------------------------- 4. sanity tests
banner("4. pytest -m 'not slow'")
sh([sys.executable, "-m", "pytest", "-m", "not slow", "-q", "-x", "-p", "no:cacheprovider"], cwd=REPO)

# ----------------------------------------------------------------------------- 5. train / resume
banner("5. train")
RUNS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
MULTI = bool(RUN_LIST)
JOBS = RUN_LIST if MULTI else [{"name": RUN_NAME, "config": CONFIG, "set": SET_OVERRIDES}]
print("/kaggle/input contents:")
sh(["find", "/kaggle/input", "-maxdepth", "5"], check=False)
T0 = time.time()
DEADLINE = T0 + MAX_WALL_MINUTES * 60
PY = [sys.executable, "-m", "minesweeper_rl.train"]
print_lock = threading.Lock()
job_results = {}


def set_args(d):
    out = []
    for k, v in d.items():
        out += ["--set", f"{k}={v if isinstance(v, str) else json.dumps(v)}"]
    return out


def run_streamed(cmd, name, env=None, cwd=None):
    """Run a subprocess, echoing its output line by line with a [name] prefix (safe across threads)."""
    cmd = [str(c) for c in cmd]
    with print_lock:
        print(f"\n[{name}] $ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        with print_lock:
            print(f"[{name}] {line.rstrip()}", flush=True)
    return p.wait()


def trim_run_dir(run_dir):
    """Finished run: drop every replay file (needed only to resume). Unfinished: keep only the one latest.pt names."""
    for f in run_dir.glob("*.tmp"):
        f.unlink()
    if (run_dir / "final.pt").exists():
        for f in run_dir.glob("replay*.npz"):
            f.unlink()
        return
    ck = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
    for f in run_dir.glob("replay*.npz"):
        if f.name != ck["replay_file"]:
            f.unlink()


def do_job(job, slot):
    name = job["name"]
    run_dir = RUNS / name
    res_dir = RESULTS / name if MULTI else RESULTS
    res_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slot)}
    prev = sorted(glob.glob(f"/kaggle/input/**/runs/{name}/latest.pt", recursive=True))
    info = {"name": name, "slot": slot, "t_start_min": (time.time() - T0) / 60}
    t0 = time.time()
    if prev:
        src = Path(prev[-1]).parent
        if run_dir.exists():
            shutil.rmtree(run_dir)
        shutil.copytree(src, run_dir)
        if MULTI:
            for old in glob.glob(f"/kaggle/input/**/results/{name}", recursive=True):
                shutil.copytree(old, res_dir, dirs_exist_ok=True)
        if (run_dir / "final.pt").exists():
            print(f"[{name}] already finished in a previous session; training skipped", flush=True)
            info["resumed"] = "finished"
        else:
            budget = (DEADLINE - time.time()) / 60
            if budget < 3:
                print(f"[{name}] only {budget:.1f} min of budget left; leaving it for the next session", flush=True)
                trim_run_dir(run_dir)
                job_results[name] = {**info, "finished": False, "skipped": "no budget"}
                return
            print(f"[{name}] RESUMING from {src} with {budget:.0f} min", flush=True)
            ov = {"max_wall_minutes": round(budget, 2), "device": "cuda", **RESUME_OVERRIDES}
            run_streamed(PY + ["--resume", run_dir, *set_args(ov)], name, env=env, cwd=REPO)
            info["resumed"] = True
    else:
        budget = (DEADLINE - time.time()) / 60
        if budget < 3:
            print(f"[{name}] only {budget:.1f} min of budget left; not started this session", flush=True)
            job_results[name] = {**info, "finished": False, "skipped": "no budget"}
            return
        print(f"[{name}] starting a FRESH run with {budget:.0f} min", flush=True)
        ov = {"max_wall_minutes": round(budget, 2), "device": "cuda", **job["set"]}
        rc = run_streamed(PY + ["--config", job["config"], "--run-dir", run_dir, *set_args(ov)], name, env=env,
                          cwd=REPO)
        if rc != 0 and not (run_dir / "latest.pt").exists():
            job_results[name] = {**info, "finished": False, "error": f"train exit {rc}"}
            return
    info["train_min"] = (time.time() - t0) / 60
    if not (run_dir / "latest.pt").exists():
        job_results[name] = {**info, "finished": False, "error": "no latest.pt"}
        return
    trim_run_dir(run_dir)
    finished = (run_dir / "final.pt").exists()
    ck = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
    info.update(finished=finished, board_steps=int(ck["board_steps"]))
    print(f"[{name}] finished={finished} board_steps={ck['board_steps']} "
          f"files={sorted(p.name for p in run_dir.iterdir())}", flush=True)
    if finished and MULTI and not all((res_dir / f"{w}_unseen.json").exists() for w in ("best", "final")):
        cmd = [sys.executable, "scripts/eval_run.py", run_dir, "--out", res_dir, "--games", EVAL_GAMES,
               "--seed-base", UNSEEN_SEED_BASE, "--device", "cuda"]
        run_streamed(cmd, name, env=env, cwd=REPO)
    job_results[name] = info


def worker(slot, queue, lock):
    while True:
        with lock:
            if not queue:
                return
            job = queue.pop(0)
        try:
            do_job(job, slot)
        except Exception as e:  # keep the other runs going
            import traceback

            traceback.print_exc()
            job_results[job["name"]] = {"name": job["name"], "finished": False, "error": repr(e)}


n_slots = max(1, min(PARALLEL if MULTI else 1, N_GPU))
print(f"{len(JOBS)} job(s), {n_slots} parallel slot(s), {N_GPU} GPU(s), budget {MAX_WALL_MINUTES} min")
queue, qlock = list(JOBS), threading.Lock()
threads = [threading.Thread(target=worker, args=(i, queue, qlock)) for i in range(n_slots)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"all training done after {(time.time() - T0) / 60:.1f} min")

# ----------------------------------------------------------------------------- 6. single-run evaluation (M6 layout)
import yaml  # noqa: E402
from minesweeper_rl.agents.dqn import DQNAgent  # noqa: E402
from minesweeper_rl.evaluate import EVAL_SEED_BASE, evaluate_batched  # noqa: E402

summary = {"pinned_sha": PINNED_SHA, "multi": MULTI, "gpu": torch.cuda.get_device_name(0), "n_gpu": N_GPU,
           "torch": torch.__version__, "python": sys.version.split()[0], "session_min": (time.time() - T0) / 60,
           "jobs": job_results, "evals": {}}
if not MULTI:
    banner("6. evaluate (10k standard seeds + 10k unseen seeds)")
    run_dir = RUNS / RUN_NAME
    summary.update(run=RUN_NAME, finished=(run_dir / "final.pt").exists())
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
