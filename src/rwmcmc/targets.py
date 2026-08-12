"""Target distributions for MCMC sampling.

Functions return log of UNNORMALIZED density: x-independent terms (including
normalization constants) are dropped since they cancel in MH ratios.
"""

import numpy as np
from numpy.typing import ArrayLike


def gaussian_1d_log_pdf(x: ArrayLike, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Log of a 1D Gaussian probability density function, unnormalized.

    sigma must be a finite, strictly positive real scalar.
    """
    if isinstance(sigma, (bool, np.bool_)):
        raise TypeError("sigma must be a real scalar")

    try:
        sigma_array = np.asarray(sigma, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("sigma must be a real scalar") from exc

    if sigma_array.ndim != 0:
        raise TypeError("sigma must be a real scalar")

    sigma_value = float(sigma_array)
    if not np.isfinite(sigma_value):
        raise ValueError("sigma must be finite")
    if sigma_value <= 0.0:
        raise ValueError("sigma must be strictly positive")

    x_array = np.asarray(x, dtype=float)
    return -0.5 * ((x_array - mu) / sigma_value) ** 2


def banana_log_pdf(x: ArrayLike, b: float = 0.03) -> np.ndarray:
    """Log of the two-dimensional Haario banana distribution, unnormalized.

    This target is obtained by applying a nonlinear transformation to a Gaussian.
    Its first coordinate has standard deviation 10, while the conditional
    distribution of the second coordinate is

        x2 | x1 ~ N(100 * b - b * x1**2, 1).

    Therefore, for x = [x1, x2],

        log p(x) = -x1**2 / 200 - 0.5 * (x2 + b * x1**2 - 100 * b)**2.

    The function accepts one point with shape (2,) or a batch with shape
    (..., 2).
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 0 or x_arr.shape[-1] != 2:
        raise ValueError("banana_log_pdf requires input with shape (..., 2)")

    x1 = x_arr[..., 0]
    x2 = x_arr[..., 1]
    return -(x1**2) / 200.0 - 0.5 * (x2 + b * x1**2 - 100.0 * b) ** 2
