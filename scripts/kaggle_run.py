"""Local helper for the Kaggle GPU kernel (kaggle/run_kaggle.py).

    python scripts/kaggle_run.py push [--sha SHA] [--run-name N] [--set k=v ...] [--resume]
                                      [--resume-set k=v ...] [--max-wall-minutes M] [--accelerator ACC]
    python scripts/kaggle_run.py status
    python scripts/kaggle_run.py wait  [--every-minutes 10]
    python scripts/kaggle_run.py output --version LABEL      # -> runs/kaggle/LABEL/

``push`` copies kaggle/ to runs/kaggle/_push/, fills in the pinned commit SHA and the overrides in the
copy of run_kaggle.py (the committed file keeps its placeholders), and runs ``kaggle kernels push``.
The SHA must already be on GitHub (verified with ``git ls-remote``). ``--resume`` attaches this
kernel's own previous version as an input (kernel_sources) so the new session resumes its latest.pt.
Never prints or reads the Kaggle token (the CLI reads ~/.kaggle itself).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
BUILD = ROOT / "runs" / "kaggle" / "_push"
KERNEL = json.loads((KAGGLE / "kernel-metadata.json").read_text())["id"]
CLI = ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "kaggle"


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def kaggle(*a: str, check: bool = True) -> int:
    cmd = [str(CLI), *a]
    print("$ kaggle " + " ".join(a), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if check and r.returncode:
        sys.exit(r.returncode)
    return r.returncode


def parse_kv(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def verify_pushed(sha: str) -> None:
    remote = git("ls-remote", "origin")
    heads = [ln.split()[0] for ln in remote.splitlines()]
    if sha in heads:
        return
    git("fetch", "origin", "main")
    r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"commit {sha} is not on GitHub origin/main; `git push origin main` first (git ls-remote check failed)")


def set_const(src: str, name: str, value) -> str:
    pat = re.compile(rf"^{name} = .*$", re.M)
    if not pat.search(src):
        sys.exit(f"constant {name} not found in run_kaggle.py")
    return pat.sub(lambda _m: f"{name} = {value!r}", src, count=1)


def cmd_push(a) -> None:
    sha = a.sha or git("rev-parse", "HEAD")
    sha = git("rev-parse", sha)
    verify_pushed(sha)
    print("pinned commit:", sha)
    if git("status", "--porcelain", "--", "src", "configs", "tests", "pyproject.toml"):
        print("WARNING: uncommitted changes in src/configs/tests; the kernel only sees the pinned commit")
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    src = (KAGGLE / "run_kaggle.py").read_text(encoding="utf-8")
    src = set_const(src, "PINNED_SHA", sha)
    if a.run_name:
        src = set_const(src, "RUN_NAME", a.run_name)
    if a.max_wall_minutes is not None:
        src = set_const(src, "MAX_WALL_MINUTES", a.max_wall_minutes)
    if a.set:
        src = set_const(src, "SET_OVERRIDES", parse_kv(a.set))
    if a.resume_set:
        src = set_const(src, "RESUME_OVERRIDES", parse_kv(a.resume_set))
    if a.config:
        src = set_const(src, "CONFIG", a.config)
    (BUILD / "run_kaggle.py").write_text(src, encoding="utf-8", newline="\n")
    meta = json.loads((KAGGLE / "kernel-metadata.json").read_text())
    if a.resume:
        meta["kernel_sources"] = [KERNEL]
    (BUILD / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    args = ["kernels", "push", "-p", str(BUILD)]
    if a.accelerator:
        args += ["--accelerator", a.accelerator]
    if a.timeout:
        args += ["-t", str(a.timeout)]
    kaggle(*args)


def cmd_status(_a=None) -> int:
    return kaggle("kernels", "status", KERNEL, check=False)


def cmd_wait(a) -> None:
    while True:
        r = subprocess.run([str(CLI), "kernels", "status", KERNEL], cwd=ROOT, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        print(time.strftime("%H:%M:%S"), out, flush=True)
        low = out.lower()
        if any(s in low for s in ("complete", "error", "cancel", "failed")) and "running" not in low:
            return
        time.sleep(a.every_minutes * 60)


def cmd_output(a) -> None:
    dest = ROOT / "runs" / "kaggle" / a.version
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", KERNEL, "-p", str(dest))
    print("downloaded to", dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("push")
    q.add_argument("--sha")
    q.add_argument("--run-name")
    q.add_argument("--config")
    q.add_argument("--set", action="append", default=[], metavar="K=V", help="TrainConfig override for a fresh run")
    q.add_argument("--resume-set", action="append", default=[], metavar="K=V", help="override when resuming")
    q.add_argument("--max-wall-minutes", type=float)
    q.add_argument("--resume", action="store_true", help="attach the previous version's output and resume it")
    q.add_argument("--accelerator", help="e.g. NvidiaTeslaT4 (see kaggle kernels push --help)")
    q.add_argument("--timeout", type=int, help="kernel run-time limit in seconds")
    sub.add_parser("status")
    w = sub.add_parser("wait")
    w.add_argument("--every-minutes", type=float, default=10)
    o = sub.add_parser("output")
    o.add_argument("--version", required=True)
    a = p.parse_args()
    {"push": cmd_push, "status": cmd_status, "wait": cmd_wait, "output": cmd_output}[a.cmd](a)


if __name__ == "__main__":
    main()
