"""Target distribution for MCMC sampling.
any x-independent functions are ignored, as they do not affect the MCMC sampling.
"""
import numpy as np
from numpy.typing import ArrayLike

def gaussian_1d_log_pdf(x: ArrayLike, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Log of 1D Gaussian probability density function, unnormalized."""
    return (- (0.5 * ((x - mu) / sigma) ** 2))