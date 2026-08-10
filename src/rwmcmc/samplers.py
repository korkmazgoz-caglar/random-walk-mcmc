"""Random-walk Metropolis-Hastings samplers.

The algorithm builds a Markov chain whose stationary distribution is the target.
At each step a candidate is drawn from a symmetric Gaussian proposal centred on
the current state. Because the proposal is symmetric, its density cancels in the
acceptance ratio, which therefore reduces to the ratio of target densities.
The comparison is done in log space to avoid numerical underflow.

Both samplers take an explicit rng argument so that runs are reproducible.
"""

import numpy as np
from numpy.typing import ArrayLike

from .proposals import gaussian_random_walk


def random_walk_metropolis_hastings(
    target_log_pdf,
    x0: ArrayLike,
    n_samples: int,
    step_size: float | ArrayLike = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from a target distribution in one or more dimensions.

    Parameters
    ----------
    target_log_pdf : callable
        Returns the log of the target density, which may be unnormalized.
    x0 : array_like
        Starting state of the chain. A scalar is treated as one dimension.
    n_samples : int
        Length of the chain, including the starting state.
    step_size : float or array_like, default=1.0
        Standard deviation of the Gaussian proposal. An array gives a separate
        scale per dimension.
    rng : numpy.random.Generator or None, default=None
        Random number generator. When omitted a fresh one is created, which
        makes the run non-reproducible.

    Returns
    -------
    samples : numpy.ndarray
        Chain of shape (n_samples, d).
    accepted : numpy.ndarray
        Boolean array of shape (n_samples,). accepted[i] is True when the
        proposal at step i was accepted. accepted[0] is False because the
        starting state is not a proposal.
    """
    if rng is None:
        rng = np.random.default_rng()

    x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
    d = x0_arr.size

    samples = np.zeros((n_samples, d))
    samples[0] = x0_arr
    accepted = np.zeros(n_samples, dtype=bool)

    log_p_current = np.asarray(target_log_pdf(x0_arr)).item()

    for i in range(1, n_samples):
        current_x = samples[i - 1]
        proposed_x = gaussian_random_walk(current_x=current_x, step_size=step_size, rng=rng)
        log_p_proposed = np.asarray(target_log_pdf(proposed_x)).item()

        # Symmetric proposal, so the acceptance ratio is just the density ratio.
        log_alpha = log_p_proposed - log_p_current
        if log_alpha >= 0 or rng.uniform() < np.exp(log_alpha):
            samples[i] = proposed_x
            accepted[i] = True
            log_p_current = log_p_proposed
        else:
            samples[i] = current_x

    return samples, accepted


def random_walk_metropolis_hastings_1d(
    target_log_pdf,
    x0: float,
    n_samples: int,
    step_size: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from a one-dimensional target distribution.

    A thin wrapper around random_walk_metropolis_hastings that returns a
    flat chain instead of a column. Given the same seed, both functions draw the
    identical sequence of random numbers.

    Parameters
    ----------
    target_log_pdf : callable
        Returns the log of the target density, which may be unnormalized.
    x0 : float
        Starting state of the chain.
    n_samples : int
        Length of the chain, including the starting state.
    step_size : float, default=1.0
        Standard deviation of the Gaussian proposal.
    rng : numpy.random.Generator or None, default=None
        Random number generator. When omitted a fresh one is created, which
        makes the run non-reproducible.

    Returns
    -------
    samples : numpy.ndarray
        Chain of shape (n_samples,).
    accepted : numpy.ndarray
        Boolean array of shape (n_samples,).
    """
    samples, accepted = random_walk_metropolis_hastings(
        target_log_pdf, x0=x0, n_samples=n_samples, step_size=step_size, rng=rng
    )
    return samples[:, 0], accepted
