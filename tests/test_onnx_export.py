"""ONNX export parity: onnxruntime vs PyTorch on real game states (skipped without onnxruntime)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import onnxruntime as ort  # noqa: E402

from minesweeper_rl.agents.network import QNetwork, mask_q  # noqa: E402
from minesweeper_rl.env.core import Board  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("export_onnx", ROOT / "scripts" / "export_onnx.py")
export_onnx = importlib.util.module_from_spec(spec)
sys.modules["export_onnx"] = export_onnx
spec.loader.exec_module(export_onnx)

CKPT = ROOT / "models" / "dqn_9x9_10.pt"
SIZES = [(5, 5, 3), (9, 9, 10), (16, 16, 40)]


@pytest.fixture(scope="module")
def net_and_session(tmp_path_factory):
    if CKPT.exists():
        net = export_onnx.load_net(CKPT)
    else:  # fall back to random weights of the same architecture
        torch.manual_seed(0)
        net = QNetwork(10, 64, 5).eval()
    path = tmp_path_factory.mktemp("onnx") / "m.onnx"
    export_onnx.export(net, path)
    return net, ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _states(h, w, m, n_games, seed):
    """Yield (obs, mask) along random-legal-move playouts of real games."""
    rng = np.random.default_rng(seed)
    for _ in range(n_games):
        b = Board(h, w, m, rng)
        while not b.done:
            mask = b.action_mask()
            yield b.observation(), mask
            a = int(rng.choice(np.flatnonzero(mask)))
            b.reveal(a // w, a % w)


@pytest.mark.parametrize("h,w,m", SIZES)
def test_q_values_match(net_and_session, h, w, m):
    net, sess = net_and_session
    n = 0
    for obs, _ in _states(h, w, m, 15, seed=h):
        with torch.no_grad():
            ref = net(torch.from_numpy(obs)[None]).numpy()
        got = sess.run(["q"], {"obs": obs[None]})[0]
        assert got.shape == (1, h * w)
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5)
        n += 1
    assert n > 30


@pytest.mark.parametrize("h,w,m", SIZES)
def test_greedy_masked_actions_identical(net_and_session, h, w, m):
    net, sess = net_and_session
    n = 0
    for obs, mask in _states(h, w, m, 150, seed=100 + h):
        with torch.no_grad():
            ref = mask_q(net(torch.from_numpy(obs)[None]), torch.from_numpy(mask)[None]).argmax(1).item()
        q = torch.from_numpy(sess.run(["q"], {"obs": obs[None]})[0])
        got = mask_q(q, torch.from_numpy(mask)[None]).argmax(1).item()
        assert got == ref
        assert mask[got]
        n += 1
    assert n >= 200


def test_fixtures_are_consistent():
    fx = export_onnx.make_fixtures()
    assert len(fx) >= 5
    assert any(s["hit_mine"] for b in fx for s in b["steps"])
    assert any(s["won"] for b in fx for s in b["steps"])
