"""
Tests for rwmcmc.diagnostics against known analytical results.
...
Usage:
    python -m pytest tests/ -v
"""
import numpy as np

from rwmcmc import (
    acceptance_rate,
    autocorrelation,
    effective_sample_size,
    running_mean,
    summary,
)


def test_acceptance_rate_with_burn_in():
    accepted = np.array([False] * 50 + [True] * 50)
    assert acceptance_rate(accepted) == 0.5
    assert acceptance_rate(accepted, burn_in=50) == 1.0


def test_running_mean_last_value_equals_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    rm = running_mean(x)
    assert np.isclose(rm[-1, 0], x.mean())


def test_autocorrelation():
    rng = np.random.default_rng(1)
    rho = autocorrelation(rng.normal(size=20000), max_lag=50)
    assert np.isclose(rho[0, 0], 1.0)
    assert np.abs(rho[1:, 0]).max() < 0.05


def test_summary_keys_and_dims():
    rng = np.random.default_rng(5)
    samples = rng.normal(size=(1000, 2))
    accepted = rng.random(1000) < 0.4
    st = summary(samples, accepted, burn_in=100)
    assert st["n_samples"] == 900
    assert st["n_dim"] == 2
    assert st["mean"].shape == (2,)
    assert st["ess"].shape == (2,)