"""
Tests for rwmcmc.samplers: shapes, reproducibility, and statistical correctness.
...
Usage:
    python -m pytest tests/ -v
"""

import inspect

import numpy as np
import pytest

from rwmcmc import (
    RandomWalkMetropolisHastings,
    banana_log_pdf,
    gaussian_1d_log_pdf,
    random_walk_metropolis_hastings,
    random_walk_metropolis_hastings_1d,
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
        gaussian_1d_log_pdf,
        x0=0.0,
        n_samples=500,
        step_size=2.0,
        rng=np.random.default_rng(42),
    )
    s2, a2 = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf,
        x0=0.0,
        n_samples=500,
        step_size=2.0,
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
        gaussian_1d_log_pdf,
        x0=0.0,
        n_samples=200000,
        step_size=step_size,
        rng=np.random.default_rng(3),
    )
    step_size = 0.1
    x = samples[2000:]  # discard burn-in
    assert abs(x.mean()) < 0.1
    assert abs(x.std() - 1.0) < 0.1, (
        f"Standard deviation is higher than 0.1 for step_size = {step_size}"
    )


def test_acceptance_rate_reasonable_for_tuned_step():
    _, accepted = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf,
        x0=0.0,
        n_samples=10000,
        step_size=2.4,
        rng=np.random.default_rng(4),
    )
    accepted_mean = accepted.mean()
    assert 0.3 < accepted_mean < 0.6, (
        f"Acceptance rate {accepted_mean:.2f} outside optimal 0.3-0.6 range; tune step_size."
    )


def test_rejected_steps_repeat_previous_sample():
    # by construction of MH: where accepted is False, the chain must not move
    samples, accepted = random_walk_metropolis_hastings(
        gaussian_1d_log_pdf,
        x0=0.0,
        n_samples=2000,
        step_size=5.0,
        rng=np.random.default_rng(5),
    )
    # accepted[0] refers to the starting state, not a proposal, so it is excluded.
    rejected = ~accepted[1:]
    assert np.all(samples[1:][rejected] == samples[:-1][rejected])


class _TargetBreakingAfterTheFirstCall:
    """Finite at x0, invalid from the first proposal onwards."""

    def __init__(self, bad_value):
        self.bad_value = bad_value
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return 0.0 if self.calls == 1 else self.bad_value


def test_class_and_function_agree_for_the_same_seed():
    sampler = RandomWalkMetropolisHastings(
        banana_log_pdf, step_size=1.0, rng=np.random.default_rng(11)
    )
    s_class, a_class = sampler.sample(x0=[0.0, 0.0], n_samples=1000)
    s_func, a_func = random_walk_metropolis_hastings(
        banana_log_pdf,
        x0=[0.0, 0.0],
        n_samples=1000,
        step_size=1.0,
        rng=np.random.default_rng(11),
    )
    assert np.array_equal(s_class, s_func)
    assert np.array_equal(a_class, a_func)


def test_public_function_signature_is_unchanged():
    # downstream code calls this by keyword, so the contract must not drift
    sig = inspect.signature(random_walk_metropolis_hastings)
    assert list(sig.parameters) == ["target_log_pdf", "x0", "n_samples", "step_size", "rng"]
    assert sig.parameters["step_size"].default == 1.0
    assert sig.parameters["rng"].default is None


def test_scalar_and_multidimensional_starting_points_both_work():
    one_d = RandomWalkMetropolisHastings(gaussian_1d_log_pdf, rng=np.random.default_rng(1))
    assert one_d.sample(x0=0.0, n_samples=50)[0].shape == (50, 1)
    two_d = RandomWalkMetropolisHastings(banana_log_pdf, rng=np.random.default_rng(1))
    assert two_d.sample(x0=[0.0, 0.0], n_samples=50)[0].shape == (50, 2)


def test_target_log_pdf_must_be_callable():
    with pytest.raises(TypeError):
        RandomWalkMetropolisHastings("not a function")


def test_rng_must_be_a_generator():
    with pytest.raises(TypeError):
        RandomWalkMetropolisHastings(gaussian_1d_log_pdf, rng=42)


@pytest.mark.parametrize("step_size", [-1.0, 0.0, np.inf])
def test_step_size_must_be_positive_and_finite(step_size):
    with pytest.raises(ValueError):
        RandomWalkMetropolisHastings(gaussian_1d_log_pdf, step_size=step_size)


def test_step_size_must_match_the_dimension_of_x0():
    sampler = RandomWalkMetropolisHastings(banana_log_pdf, step_size=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        sampler.sample(x0=[0.0, 0.0], n_samples=10)


@pytest.mark.parametrize("n_samples", [1.5, "10", True, None])
def test_n_samples_must_be_an_integer(n_samples):
    sampler = RandomWalkMetropolisHastings(gaussian_1d_log_pdf)
    with pytest.raises(TypeError):
        sampler.sample(x0=0.0, n_samples=n_samples)


def test_n_samples_must_be_at_least_one():
    sampler = RandomWalkMetropolisHastings(gaussian_1d_log_pdf)
    with pytest.raises(ValueError):
        sampler.sample(x0=0.0, n_samples=0)


@pytest.mark.parametrize("x0", [[], [[0.0, 1.0], [2.0, 3.0]], [0.0, np.nan]])
def test_x0_must_be_a_finite_one_dimensional_point(x0):
    sampler = RandomWalkMetropolisHastings(gaussian_1d_log_pdf)
    with pytest.raises(ValueError):
        sampler.sample(x0=x0, n_samples=10)


def test_x0_must_be_numeric():
    sampler = RandomWalkMetropolisHastings(gaussian_1d_log_pdf)
    with pytest.raises(TypeError):
        sampler.sample(x0="origin", n_samples=10)


def test_target_log_pdf_must_return_one_number():
    sampler = RandomWalkMetropolisHastings(lambda x: np.zeros(3))
    with pytest.raises(ValueError):
        sampler.sample(x0=0.0, n_samples=10)


def test_target_log_pdf_must_be_finite_at_the_starting_point():
    sampler = RandomWalkMetropolisHastings(lambda x: -np.inf)
    with pytest.raises(ValueError):
        sampler.sample(x0=0.0, n_samples=10)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_target_log_pdf_must_stay_valid_during_sampling(bad_value):
    sampler = RandomWalkMetropolisHastings(_TargetBreakingAfterTheFirstCall(bad_value))
    with pytest.raises(ValueError):
        sampler.sample(x0=0.0, n_samples=10)


def test_minus_infinity_outside_the_support_is_allowed():
    # a target may rule out part of the space; those proposals are just rejected
    def bounded(x):
        return 0.0 if np.all(np.abs(x) < 1.0) else -np.inf

    sampler = RandomWalkMetropolisHastings(bounded, step_size=0.5, rng=np.random.default_rng(12))
    samples, _ = sampler.sample(x0=0.0, n_samples=500)
    assert np.all(np.abs(samples) < 1.0)


def test_one_dimensional_wrapper_rejects_a_multidimensional_start():
    with pytest.raises(ValueError):
        random_walk_metropolis_hastings_1d(banana_log_pdf, x0=[0.0, 0.0], n_samples=10)
