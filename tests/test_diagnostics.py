"""
Tests for rwmcmc.diagnostics against known analytical results.
...
Usage:
    python -m pytest tests/ -v
"""
import numpy as np
import pytest

from rwmcmc import (
    acceptance_rate,
    autocorrelation,
    effective_sample_size,
    running_mean,
    summary,
    corner,
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


# I will check them.

def test_ess_iid_close_to_n():
    # independent samples are worth almost their own count
    rng = np.random.default_rng(3)
    n = 20000
    ess = effective_sample_size(rng.normal(size=n))
    assert 0.9 * n < ess[0] <= 1.1 * n


def test_ess_ar1_matches_theory():
    # AR(1) is known ESS ratio (1-phi)/(1+phi); generated loop-free with lfilter
    from scipy.signal import lfilter
    rng = np.random.default_rng(4)
    phi, n = 0.9, 50000
    x = lfilter([1.0], [1.0, -phi], rng.normal(size=n))
    ess = effective_sample_size(x)[0]
    theory = n * (1 - phi) / (1 + phi)
    assert 0.5 * theory < ess < 1.5 * theory


def test_corner_returns_dxd_grid():
    rng = np.random.default_rng(6)
    fig = corner(rng.normal(size=(500, 3)))
    assert len(fig.axes) == 9  # 3x3 grid


def test_corner_rejects_1d():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        corner(rng.normal(size=500))


def test_all_matches_public_names():
    # __all__ must list exactly the public names importable from the package, I always forget...
    import types
    import rwmcmc
    public = {
        name for name in dir(rwmcmc)
        if not name.startswith("_")
        and not isinstance(getattr(rwmcmc, name), types.ModuleType)
    }
    assert public == set(rwmcmc.__all__)
