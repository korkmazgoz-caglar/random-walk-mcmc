"""
Tests for rwmcmc.diagnostics against known analytical results.
...
Usage:
    python -m pytest tests/ -v
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rwmcmc import (
    acceptance_rate,
    autocorrelation,
    corner,
    dashboard,
    effective_sample_size,
    running_mean,
    summary,
)


def test_acceptance_rate_with_burn_in():
    accepted = np.array([False] * 50 + [True] * 50)
    assert acceptance_rate(accepted) == pytest.approx(50 / 99)
    assert acceptance_rate(accepted, burn_in=50) == 1.0


def test_acceptance_rate_does_not_count_the_starting_state_as_a_rejection():
    accepted = np.array([False, True])
    assert acceptance_rate(accepted) == 1.0


@pytest.mark.parametrize(
    "samples",
    [[], np.empty((3, 0)), np.ones((2, 2, 2)), [0.0, np.nan]],
)
def test_diagnostics_reject_invalid_sample_arrays(samples):
    with pytest.raises(ValueError):
        running_mean(samples)


def test_diagnostics_reject_non_numeric_samples():
    with pytest.raises(TypeError, match="samples must be numeric"):
        running_mean(["not-a-number"])


@pytest.mark.parametrize(
    "accepted, exception",
    [
        ([False], ValueError),
        ([[False, True]], ValueError),
        ([0, 1], TypeError),
    ],
)
def test_acceptance_rate_rejects_invalid_decision_arrays(accepted, exception):
    with pytest.raises(exception):
        acceptance_rate(accepted)


@pytest.mark.parametrize("burn_in", [-1, 3, 1.5, True])
def test_summary_rejects_invalid_burn_in(burn_in):
    samples = np.array([0.0, 1.0, 2.0, 3.0])
    accepted = np.array([False, True, True, True])
    with pytest.raises((TypeError, ValueError), match="burn_in"):
        summary(samples, accepted, burn_in=burn_in)


def test_summary_requires_matching_sample_and_acceptance_lengths():
    with pytest.raises(ValueError, match="same length"):
        summary(np.arange(4.0), np.array([False, True, True]))


@pytest.mark.parametrize("max_lag", [-1, 4, 1.5, True])
def test_autocorrelation_rejects_invalid_max_lag(max_lag):
    with pytest.raises((TypeError, ValueError), match="max_lag"):
        autocorrelation(np.arange(4.0), max_lag=max_lag)


@pytest.mark.parametrize("param_names", [["only-one"], ["x", 2]])
def test_dashboard_rejects_invalid_parameter_names(param_names):
    samples = np.arange(20.0).reshape(10, 2)
    accepted = np.array([False] + [True] * 9)
    with pytest.raises((TypeError, ValueError), match="param_names"):
        dashboard(samples, accepted, param_names=param_names)


def test_autocorrelation_does_not_modify_the_input_chain():
    samples = np.array([1.0, 2.0, 4.0, 8.0])
    original = samples.copy()

    autocorrelation(samples)

    np.testing.assert_array_equal(samples, original)


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


@pytest.mark.parametrize("phi", [-0.5, 0.9])
def test_ess_ar1_matches_theory(phi):
    # AR(1) has the known ESS ratio (1-phi)/(1+phi).
    from scipy.signal import lfilter

    rng = np.random.default_rng(4)
    n = 50000
    x = lfilter([1.0], [1.0, -phi], rng.normal(size=n))
    ess = effective_sample_size(x)[0]
    theory = n * (1 - phi) / (1 + phi)
    assert 0.5 * theory < ess < 1.5 * theory
    if phi < 0:
        assert ess > n


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
        name
        for name in dir(rwmcmc)
        if not name.startswith("_") and not isinstance(getattr(rwmcmc, name), types.ModuleType)
    }
    assert public == set(rwmcmc.__all__)


def test_dashboard_panel_grid():
    # one row per dimension, four panels per row
    rng = np.random.default_rng(8)
    samples = rng.normal(size=(500, 2))
    accepted = rng.random(500) < 0.4
    fig = dashboard(samples, accepted, burn_in=50)
    assert len(fig.axes) == 8
    plt.close(fig)


def test_dashboard_summary_line_reports_post_burn_in_size():
    # burn_in must reach the statistics, not only the plots
    rng = np.random.default_rng(9)
    samples = rng.normal(size=400)
    accepted = rng.random(400) < 0.5
    fig = dashboard(samples, accepted, burn_in=100)
    title = fig.get_suptitle()
    assert len(fig.axes) == 4
    assert "n = 300" in title
    assert "dim = 1" in title
    plt.close(fig)


def test_dashboard_uses_given_param_names():
    rng = np.random.default_rng(10)
    samples = rng.normal(size=(300, 2))
    accepted = rng.random(300) < 0.4
    fig = dashboard(samples, accepted, param_names=["alpha", "beta"])
    titles = [ax.get_title() for ax in fig.axes]
    assert "trace: alpha" in titles
    assert "trace: beta" in titles
    plt.close(fig)
