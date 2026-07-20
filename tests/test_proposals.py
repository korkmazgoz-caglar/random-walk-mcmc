"""
Tests for rwmcmc.proposals.
...
Usage:
    python -m pytest tests/ -v
"""
import numpy as np

from rwmcmc import gaussian_random_walk


def test_shape_preserved_scalar_and_vector():
    rng = np.random.default_rng(0)
    x1 = gaussian_random_walk(np.zeros(3), step_size=1.0, rng=rng)
    x2 = gaussian_random_walk(np.zeros(3), step_size=[1.0, 2.0, 3.0], rng=rng)
    assert x1.shape == (3,)
    assert x2.shape == (3,)


def test_zero_step_size_returns_current_state():
    rng = np.random.default_rng(0)
    x0 = np.array([1.5, -2.0])
    assert np.array_equal(gaussian_random_walk(x0, step_size=0.0, rng=rng), x0)


def test_step_statistics_match_step_size():
    # over many draws, epsilon = x' - x must have mean around 0 and std around step_size
    rng = np.random.default_rng(1)
    x0 = np.full(200000, 10.0) # Gaussian NOT around zero, explicitly
    eps = gaussian_random_walk(x0, step_size=2.0, rng=rng) - x0
    assert abs(eps.mean()) < 0.02
    assert abs(eps.std() - 2.0) < 0.02


def test_reproducible_with_seed():
    a = gaussian_random_walk(np.zeros(5), 1.0, np.random.default_rng(7))
    b = gaussian_random_walk(np.zeros(5), 1.0, np.random.default_rng(7))
    assert np.array_equal(a, b)