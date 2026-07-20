"""rwmcmc: lightweight Random-Walk Metropolis-Hastings sampling, diagnostics, and visualization."""
from .targets import gaussian_1d_log_pdf, banana_log_pdf
from .proposals import gaussian_random_walk
from .samplers import (
    random_walk_metropolis_hastings,
    random_walk_metropolis_hastings_1d,
)
from .diagnostics import (
    acceptance_rate,
    autocorrelation,
    dashboard,
    effective_sample_size,
    running_mean,
    summary,
    corner,
)
__version__ = "0.1.0"

__all__ = [
    "gaussian_1d_log_pdf",
    "banana_log_pdf",
    "gaussian_random_walk",
    "random_walk_metropolis_hastings",
    "random_walk_metropolis_hastings_1d",
    "acceptance_rate",
    "autocorrelation",
    "dashboard",
    "effective_sample_size",
    "running_mean",
    "summary",

]