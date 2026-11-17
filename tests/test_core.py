import numpy as np
import pytest

from minesweeper_rl.env.core import Board, GameOverError


def make(h=5, w=5, m=3, seed=0):
    return Board(h, w, m, np.random.default_rng(seed))


def mask_of(h, w, cells):
    m = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        m[r, c] = True
    return m


def brute_counts(mines):
    h, w = mines.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w:
                        out[r, c] += mines[r + dr, c + dc]
    return out


@pytest.mark.parametrize("cfg", [(5, 5, 3), (9, 9, 10), (3, 3, 8)])
def test_first_click_never_mine(cfg):
    h, w, m = cfg
    clicks = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (h // 2, w // 2)]
    for seed in range(40):
        for r, c in clicks:
            b = make(h, w, m, seed)
            res = b.reveal(r, c)
            assert not res.hit_mine and res.first_click and res.valid
            assert not b.mines[r, c]


def test_dense_board_first_click_wins():
    b = make(3, 3, 8, seed=1)
    res = b.reveal(1, 1)
    assert res.won and b.won and b.done


def test_mine_count():
    b = make(9, 9, 10)
    assert b.mines.sum() == 0 and not b.mines_placed
    b.reveal(4, 4)
    assert b.mines.sum() == 10


def test_counts_bruteforce():
    for seed in range(10):
        b = make(9, 9, 10, seed)
        b.reveal(0, 0)
        assert (b.counts == brute_counts(b.mines)).all()


def test_flood_fill_exact():
    # 4x4, mine at (0,0): zero region is everything except (0,0) neighbours' no-zero cells.
    b = make(4, 4, 1)
    b.place_mines(mask_of(4, 4, [(0, 0)]))
    res = b.reveal(3, 3)
    expected = np.ones((4, 4), dtype=bool)
    expected[0, 0] = False
    assert (b.revealed == expected).all()
    assert res.newly_revealed == 15 and res.won


def test_flood_fill_stops_at_numbers():
    b = make(5, 5, 1)
    b.place_mines(mask_of(5, 5, [(0, 4)]))
    # Mine at (0,4): numbered cells (0,3),(1,3),(1,4). Click corner (4,0) -> floods all but mine.
    b.reveal(4, 0)
    assert b.revealed.sum() == 24
    # Column of mines splits board: mines at col 2.
    b = make(3, 5, 3)
    b.place_mines(mask_of(3, 5, [(0, 2), (1, 2), (2, 2)]))
    res = b.reveal(0, 0)
    expected = np.zeros((3, 5), dtype=bool)
    expected[:, :2] = True  # col 0 is zero; col 1 numbered border
    assert (b.revealed == expected).all() and res.newly_revealed == 6


def test_numbered_cell_reveals_only_itself():
    b = make(4, 4, 1)
    b.place_mines(mask_of(4, 4, [(0, 0)]))
    res = b.reveal(1, 1)
    assert res.newly_revealed == 1 and b.revealed.sum() == 1


def test_loss_and_after_game_over():
    b = make(4, 4, 1)
    b.place_mines(mask_of(4, 4, [(0, 0)]))
    res = b.reveal(0, 0)
    assert res.hit_mine and b.done and not b.won and b.revealed[0, 0]
    with pytest.raises(GameOverError):
        b.reveal(3, 3)
    assert "X" in b.render_text()


def test_win_then_reveal_raises():
    b = make(2, 2, 1)
    b.place_mines(mask_of(2, 2, [(0, 0)]))
    b.reveal(0, 1)
    b.reveal(1, 0)
    assert not b.won and b.n_safe_remaining == 1
    res = b.reveal(1, 1)
    assert res.won and b.won and b.n_safe_remaining == 0
    with pytest.raises(GameOverError):
        b.reveal(0, 1)


def test_already_revealed_invalid():
    b = make(4, 4, 1)
    b.place_mines(mask_of(4, 4, [(0, 0)]))
    b.reveal(1, 1)
    before = b.revealed.copy()
    res = b.reveal(1, 1)
    assert not res.valid and (b.revealed == before).all() and not b.done


def test_determinism():
    def run(seed):
        b = make(9, 9, 10, seed)
        b.reveal(4, 4)
        return b.mines.copy()

    assert (run(3) == run(3)).all()
    assert any((run(0) != run(s)).any() for s in range(1, 5))


def test_observation():
    b = make(9, 9, 10)
    obs = b.observation()
    assert obs.shape == (10, 9, 9) and obs.dtype == np.float32
    assert (obs[0] == 1).all() and obs[1:].sum() == 0
    b.reveal(4, 4)
    obs = b.observation()
    assert (obs.sum(axis=0) == 1).all()
    assert ((obs[0] == 1) == ~b.revealed).all()
    assert (obs[1:, ~b.revealed] == 0).all()
    r, c = np.argwhere(b.revealed)[0]
    assert obs[1 + b.counts[r, c], r, c] == 1


def test_action_mask():
    b = make(5, 5, 3)
    assert b.action_mask().all() and b.action_mask().shape == (25,)
    b.reveal(2, 2)
    assert (b.action_mask() == (~b.revealed).reshape(-1)).all()


def test_render_text():
    b = make(3, 3, 1)
    assert "#" in b.render_text()
    b.place_mines(mask_of(3, 3, [(0, 0)]))
    assert "*" in b.render_text(show_mines=True)
    assert "*" not in b.render_text()


@pytest.mark.parametrize("m", [0, -1, 25])
def test_invalid_mines(m):
    with pytest.raises(ValueError):
        Board(5, 5, m, np.random.default_rng(0))


def test_large_board_no_recursion_error():
    b = Board(16, 16, 1, np.random.default_rng(0))
    b.reveal(8, 8)
