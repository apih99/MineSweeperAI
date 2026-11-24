"""Crash-safe checkpoint I/O helpers.

* Every file is written to ``<name>.tmp`` in the same directory, fsynced, then ``os.replace``d,
  so a kill at any instant leaves either the old file or the new one, never a torn file.
* ``.pt`` files hold only tensors and plain Python types and load with ``weights_only=True``
  (numpy RNG states are plain dicts of ints; numpy arrays are converted to tensors).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def atomic_write(path: str | Path, write_fn) -> None:
    """``write_fn(binary_file)`` writes to a temp file which then atomically replaces ``path``."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(obj, path: str | Path) -> None:
    atomic_write(path, lambda f: torch.save(obj, f))


def atomic_savez(path: str | Path, arrays: dict[str, np.ndarray], compressed: bool = False) -> None:
    save = np.savez_compressed if compressed else np.savez
    atomic_write(path, lambda f: save(f, **arrays))


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write(path, lambda f: f.write(text.encode("utf-8")))


def load_torch(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def get_generator_state(g: np.random.Generator) -> dict:
    return g.bit_generator.state  # plain dict of str/int (property returns a fresh copy)


def set_generator_state(g: np.random.Generator, state: dict) -> None:
    g.bit_generator.state = state


def arrays_to_tensors(d: dict) -> dict:
    """numpy arrays -> tensors (weights_only-safe); other values pass through."""
    return {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def tensors_to_arrays(d: dict) -> dict:
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def prune_files(directory: str | Path, pattern: str, keep: str | Path | None = None) -> None:
    for p in Path(directory).glob(pattern):
        if keep is None or p.name != Path(keep).name:
            try:
                p.unlink()
            except OSError:
                pass
