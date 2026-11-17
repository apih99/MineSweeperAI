import numpy as np

from minesweeper_rl.agents.baselines import RandomAgent, RuleAgent
from minesweeper_rl.env.core import Board
from minesweeper_rl.env.gym_env import MinesweeperEnv


def make_board(h, w, mine_cells, reveals):
    b = Board(h, w, len(mine_cells), np.random.default_rng(0))
    m = np.zeros((h, w), dtype=bool)
    for r, c in mine_cells:
        m[r, c] = True
    b.place_mines(m)
    for r, c in reveals:
        b.reveal(r, c)
    return b


def cells(mask):
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(mask))}


def test_random_agent_only_legal():
    rng = np.random.default_rng(1)
    a = RandomAgent()
    a.reset(rng)
    obs = np.zeros((10, 5, 5), np.float32)
    for _ in range(200):
        mask = rng.random(25) < 0.4
        if mask.any():
            assert mask[a.act(obs, mask)]


# Position used by several tests (3x3, mines at (0,0) and (2,2)):
#   (0,2) flood reveals (0,1)=1, (1,1)=2, (1,2)=1; then (1,0)=1 is revealed.
#   (0,1)=1 has the single hidden neighbour (0,0)  -> Rule 2: (0,0) is a mine.
#   (1,0)=1 is then satisfied                     -> Rule 1: (2,0), (2,1) are safe.
#   (1,2)=1 with (2,1) safe                       -> Rule 2: (2,2) is a mine.
def chain_board():
    return make_board(3, 3, [(0, 0), (2, 2)], [(0, 2), (1, 0)])


def test_rule2_identifies_mines_and_rule1_chains_to_safe_cells():
    b = chain_board()
    obs, mask = b.observation(), b.action_mask()
    safe, mine = RuleAgent.deduce(obs)
    assert cells(mine) == {(0, 0), (2, 2)}
    assert cells(safe) == {(2, 0), (2, 1)}


def test_agent_picks_deducible_safe_cell():
    b = chain_board()
    obs, mask = b.observation(), b.action_mask()
    for seed in range(10):
        agent = RuleAgent()
        agent.reset(np.random.default_rng(seed))
        r, c = divmod(agent.act(obs, mask), 3)
        assert not b.mines[r, c]
        assert (r, c) in {(2, 0), (2, 1)}


def test_rule2_only_position_has_mines_but_no_safe():
    # 1x5, mines at 0 and 2; reveal (0,4) floods to (0,3)=1.
    b = make_board(1, 5, [(0, 0), (0, 2)], [(0, 4)])
    safe, mine = RuleAgent.deduce(b.observation())
    assert cells(mine) == {(0, 2)}
    assert not safe.any()


def test_no_deductions_never_picks_deduced_mine():
    # 2x4 with mine at (0,0): hidden (0,0),(1,0) are indistinguishable.
    b = make_board(2, 4, [(0, 0)], [(1, 3)])
    obs, mask = b.observation(), b.action_mask()
    safe, mine = RuleAgent.deduce(obs)
    assert not safe.any() and not mine.any()
    # Sparse 5x5 position, whatever is deduced must never be clicked.
    b2 = make_board(5, 5, [(0, 0), (2, 2), (4, 4)], [(0, 4), (1, 1)])
    obs2, mask2 = b2.observation(), b2.action_mask()
    _, mine2 = RuleAgent.deduce(obs2)
    for s in range(100):
        agent = RuleAgent()
        agent.reset(np.random.default_rng(s))
        assert mask[agent.act(obs, mask)]
        a = agent.act(obs2, mask2)
        assert mask2[a] and not mine2.reshape(-1)[a]


def test_first_move_is_a_hidden_cell():
    b = Board(9, 9, 10, np.random.default_rng(0))
    agent = RuleAgent()
    agent.reset(np.random.default_rng(0))
    assert b.action_mask()[agent.act(b.observation(), b.action_mask())]


def _play_checked(seed, h=9, w=9, n=10):
    env = MinesweeperEnv(h, w, n)
    obs, info = env.reset(seed=seed)
    agent = RuleAgent()
    agent.reset(np.random.default_rng(seed))
    safe_clicks = 0
    while True:
        mask = info["action_mask"]
        safe, mine = agent.deduce(obs)
        a = agent.act(obs, mask)
        mines = env.board.mines
        if env.board.mines_placed:
            assert not (mine & ~mines).any(), "deduced mine is actually safe"
            assert not (safe & mines).any(), "deduced safe cell is actually a mine"
            assert not mine.reshape(-1)[a], "clicked a deduced mine"
            if (safe.reshape(-1) & mask).any():
                safe_clicks += 1
                assert safe.reshape(-1)[a]
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            return safe_clicks


def test_deductions_are_sound_over_games():
    total = sum(_play_checked(s) for s in range(200))
    assert total > 200  # the solver really did make many deduced clicks
