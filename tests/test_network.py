import torch

from minesweeper_rl.agents.network import QNetwork, masked_argmax


def test_any_board_size_same_weights():
    net = QNetwork(channels=8, n_layers=3)
    for h, w in [(5, 5), (9, 9), (16, 16), (4, 7)]:
        out = net(torch.zeros(3, 10, h, w))
        assert out.shape == (3, h * w)


def test_no_linear_or_batchnorm():
    net = QNetwork()
    assert not any(isinstance(m, (torch.nn.Linear, torch.nn.BatchNorm2d)) for m in net.modules())


def test_masked_argmax_never_picks_masked():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        q = -torch.rand(8, 25, generator=g) * 100 - 1  # all negative
        mask = torch.rand(8, 25, generator=g) < 0.3
        mask[:, 0] |= ~mask.any(dim=1)  # at least one legal
        a = masked_argmax(q, mask)
        assert mask[torch.arange(8), a].all()


def test_masked_argmax_ties_and_best_masked():
    q = torch.zeros(1, 6)
    q[0, 2] = 10.0  # best cell is masked out
    mask = torch.tensor([[False, True, False, True, True, False]])
    a = masked_argmax(q, mask)
    assert mask[0, a[0]]
    assert masked_argmax(torch.zeros(1, 4), torch.tensor([[False, False, True, True]]))[0] in (2, 3)
