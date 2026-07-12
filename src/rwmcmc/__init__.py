"""rwmcmc: lightweight Random-Walk Metropolis-Hastings sampling, diagnostics, and visualization."""
from .targets import gaussian_1d_log_pdf, banana_log_pdf
from .proposals import gaussian_random_walk
from .samplers import (
    random_walk_metropolis_hastings,
    random_walk_metropolis_hastings_1d,
)

__version__ = "0.1.0"

__all__ = [
    "gaussian_1d_log_pdf",
    "banana_log_pdf",
    "gaussian_random_walk",
    "random_walk_metropolis_hastings",
    "random_walk_metropolis_hastings_1d",
]