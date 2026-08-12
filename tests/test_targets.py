"""Tests for the example target distributions."""

import numpy as np
import pytest

from rwmcmc import banana_log_pdf, gaussian_1d_log_pdf


def test_gaussian_default_parameters():
    points = np.array([-1.0, 0.0, 1.0])

    np.testing.assert_allclose(
        gaussian_1d_log_pdf(points),
        [-0.5, 0.0, -0.5],
    )


def test_gaussian_respects_mu_and_sigma():
    points = np.array([1.0, 3.0, 5.0])

    np.testing.assert_allclose(
        gaussian_1d_log_pdf(points, mu=3.0, sigma=2.0),
        [-0.5, 0.0, -0.5],
    )


def test_gaussian_supports_batched_input():
    points = np.array(
        [
            [1.0, 3.0],
            [5.0, 7.0],
        ]
    )

    result = gaussian_1d_log_pdf(points, mu=3.0, sigma=2.0)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(
        result,
        [
            [-0.5, 0.0],
            [-0.5, -2.0],
        ],
    )


def test_banana_default_mode_has_zero_log_density():
    assert banana_log_pdf([0.0, 3.0]) == pytest.approx(0.0)


def test_banana_density_follows_the_curved_ridge():
    b = 0.03
    x1 = np.array([-10.0, 0.0, 10.0])
    x2 = 100.0 * b - b * x1**2
    points = np.column_stack([x1, x2])

    np.testing.assert_allclose(
        banana_log_pdf(points, b=b),
        -(x1**2) / 200.0,
    )


def test_banana_decreases_away_from_the_ridge():
    assert banana_log_pdf([0.0, 4.0]) == pytest.approx(-0.5)
    assert banana_log_pdf([0.0, 2.0]) == pytest.approx(-0.5)


def test_banana_supports_batched_input():
    points = np.array(
        [
            [[0.0, 3.0], [10.0, 0.0]],
            [[-10.0, 0.0], [0.0, 4.0]],
        ]
    )

    result = banana_log_pdf(points)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, [[0.0, -0.5], [-0.5, -0.5]])


@pytest.mark.parametrize(
    "x",
    [
        0.0,
        [0.0],
        [0.0, 1.0, 2.0],
        [[0.0], [1.0]],
    ],
)
def test_banana_rejects_wrong_input_shape(x):
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 2\)"):
        banana_log_pdf(x)
