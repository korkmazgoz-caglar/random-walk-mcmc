"""Target distributions for MCMC sampling.

Functions return log of UNNORMALIZED density: x-independent terms (including
normalization constants) are dropped since they cancel in MH ratios.
"""
import numpy as np
from numpy.typing import ArrayLike

def gaussian_1d_log_pdf(x: ArrayLike, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Log of 1D Gaussian probability density function, unnormalized."""
    x = np.asarray(x, dtype=float)
    return -0.5 * ((x - mu) / sigma) ** 2

def banana_log_pdf(x: ArrayLike, b: float = .1) -> np.ndarray:
    """Log of a 2D banana-shaped target distribution, unnormalized.

    The banana distribution is a common example in MCMC because it is curved and
    non-Gaussian. It is defined for a 2D input x = [x1, x2] as:

        log p(x) = -0.5 * x1**2 - 0.5 * (x2 - b * x1**2)**2

    This function accepts a length-2 vector-like input and returns the unnormalized
    log probability.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 0 or x_arr.shape[-1] != 2:
        raise ValueError("banana_log_pdf requires input with shape (..., 2)")

    x1 = x_arr[..., 0]
    x2 = x_arr[..., 1]
    return -0.5 * x1**2 - 0.5 * (x2 - b * x1**2) ** 2