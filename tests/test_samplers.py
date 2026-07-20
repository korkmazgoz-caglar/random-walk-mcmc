"""
Tests for rwmcmc.samplers: shapes, reproducibility, and statistical correctness.
...
Usage:
    python -m pytest tests/ -v
"""

import numpy as np
import pytest

from rwmcmc import (
    random_walk_metropolis_hastings,
    random_walk_metropolis_hastings_1d,
    gaussian_1d_log_pdf,
    banana_log_pdf,
)


def test_1d_output_shape_and_type():
    rng = np.random.default_rng(0)
    samples, accepted = random_walk_metropolis_hastings_1d(
        gaussian_1d_log_pdf, x0=0, n_samples=500, step_size=1.0, rng=rng
    )
    assert samples.shape[0] == 500
    assert accepted.shape == (500,)
    assert accepted.dtype == bool


def test_nd_output_shapes():
    rng = np.random.default_rng(0)
    samples, accepted = random_walk_metropolis_hastings(
        banana_log_pdf, x0=[0.0, 0.0], n_samples=500, step_size=[1.0, 1.0], rng=rng
    )
    assert samples.shape == (500, 2)
    assert accepted.shape == (500,)


def test_same_seed_gives_identical_chain():
    # reproducibility: the core promise for scientific use
    s1, a1 = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf, x0=0.0, n_samples=500, step_size=2.0,
        rng=np.random.default_rng(42),
    )
    s2, a2 = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf, x0=0.0, n_samples=500, step_size=2.0,
        rng=np.random.default_rng(42),
    )
    assert np.array_equal(s1, s2)
    assert np.array_equal(a1, a2)


def test_different_seed_gives_different_chain():
    s1, _ = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf, x0=0, n_samples=500, step_size=1.0, rng=np.random.default_rng(1)
    )
    s2, _ = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf, x0=0, n_samples=500, step_size=1.0, rng=np.random.default_rng(2)
    )

    assert not np.array_equal(s1, s2)

def test_gaussian_target_moments():
    # chain's mean should eventually converge to 0 for N(0,1) target distribution
    step_size = 0.1
    samples, accepted = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf, x0=0.0, n_samples=200000, step_size=step_size,
        rng=np.random.default_rng(3),
    )
    step_size = 0.1
    x = samples[2000:]  # discard burn-in
    assert abs(x.mean()) < 0.1
    assert abs(x.std() - 1.0) < 0.1, f"Standard deviation is higher than 0.1 for step_size = {step_size}"


def test_acceptance_rate_reasonable_for_tuned_step():
    _, accepted = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf, x0=0.0, n_samples=10000, step_size=2.4,
        rng=np.random.default_rng(4),
    )
    accepted_mean = accepted.mean()
    assert 0.3 < accepted_mean < 0.6, f"Acceptance rate is not in the optimal range 0.3 !< {accepted_mean} !< 0.6. Tuning the step_size is suggested."


def test_rejected_steps_repeat_previous_sample():
    # by construction of MH: where accepted is False, the chain must not move
    samples, accepted = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf, x0=0.0, n_samples=2000, step_size=5.0,
        rng=np.random.default_rng(5),
    )
    rejected = ~accepted[1:] # accepted contains "is accepted? true/false". first element is the x0, it is excluded. then it is used for masking
    assert np.all(samples[1:][rejected] == samples[:-1][rejected]) 