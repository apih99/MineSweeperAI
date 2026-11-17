def test_package_imports():
    import minesweeper_rl

    assert hasattr(minesweeper_rl, "__version__")


def test_dependencies_import():
    import gymnasium
    import numpy
    import torch

    assert torch.__version__
    assert numpy.__version__
    assert gymnasium.__version__
