"""rwmcmc: lightweight Random-Walk Metropolis-Hastings sampling, diagnostics, and visualization."""

from importlib.metadata import version as _version

from .diagnostics import (
    acceptance_rate,
    autocorrelation,
    corner,
    dashboard,
    effective_sample_size,
    running_mean,
    summary,
)
from .proposals import gaussian_random_walk
from .samplers import (
    RandomWalkMetropolisHastings,
    random_walk_metropolis_hastings,
    random_walk_metropolis_hastings_1d,
)
from .targets import banana_log_pdf, gaussian_1d_log_pdf

__version__ = _version("rwmcmc")

__all__ = [
    "gaussian_1d_log_pdf",
    "banana_log_pdf",
    "gaussian_random_walk",
    "RandomWalkMetropolisHastings",
    "random_walk_metropolis_hastings",
    "random_walk_metropolis_hastings_1d",
    "acceptance_rate",
    "autocorrelation",
    "dashboard",
    "effective_sample_size",
    "running_mean",
    "summary",
    "corner",
]
