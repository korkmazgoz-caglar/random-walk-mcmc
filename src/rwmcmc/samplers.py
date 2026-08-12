"""Random-walk Metropolis-Hastings sampling.

The algorithm builds a Markov chain whose stationary distribution is the target.
At each step a candidate is drawn from a symmetric Gaussian proposal centred on
the current state. Because the proposal is symmetric, its density cancels in the
acceptance ratio, which therefore reduces to the ratio of target densities.
The comparison is done in log space to avoid numerical underflow.

RandomWalkMetropolisHastings holds the algorithm. The module level functions are
thin wrappers over it, kept so that existing code keeps working unchanged.
"""

import numbers

import numpy as np
from numpy.typing import ArrayLike

from .proposals import gaussian_random_walk


def _as_1d_float(value, name: str) -> np.ndarray:
    """Return value as a non-empty, finite, one-dimensional float array."""
    try:
        arr = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array, got {arr.ndim} dimensions")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_positive_int(value, name: str) -> int:
    """Return value as an integer of at least 1, rejecting bool and non-integers."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return int(value)


def _as_log_density(value, where: str, allow_negative_infinity: bool = False) -> float:
    """Return a single log density.

    Nan and positive infinity are always rejected: they would silently corrupt
    the acceptance ratio and freeze the chain. Minus infinity is allowed only
    where it has a meaning, namely a proposal outside the support of the target.
    """
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"target_log_pdf must return one number, got shape {arr.shape} {where}")
    density = float(arr.reshape(()))
    if np.isnan(density):
        raise ValueError(f"target_log_pdf returned nan {where}")
    if density == np.inf:
        raise ValueError(f"target_log_pdf returned positive infinity {where}")
    if density == -np.inf and not allow_negative_infinity:
        raise ValueError(f"target_log_pdf returned minus infinity {where}")
    return density


class RandomWalkMetropolisHastings:
    """Random-walk Metropolis-Hastings sampler.

    The target, the proposal width and the random number generator belong to the
    sampler, so they are given once and a chain is drawn with sample().

    Parameters
    ----------
    target_log_pdf : callable
        Returns the log of the target density, which may be unnormalized. It may
        return minus infinity outside the support of the target.
    step_size : float or array_like, default=1.0
        Standard deviation of the Gaussian proposal. An array gives a separate
        scale per dimension and must then match the dimension of x0.
    rng : numpy.random.Generator or None, default=None
        Random number generator. When omitted a fresh one is created, which makes
        the run non-reproducible.

    Raises
    ------
    TypeError
        If target_log_pdf is not callable, step_size is not numeric, or rng is
        neither None nor a numpy.random.Generator.
    ValueError
        If step_size is empty, not finite, or not strictly positive.

    Examples
    --------
    >>> import numpy as np
    >>> from rwmcmc import banana_log_pdf
    >>> sampler = RandomWalkMetropolisHastings(
    ...     banana_log_pdf, step_size=[4.0, 2.0], rng=np.random.default_rng(42)
    ... )
    >>> samples, accepted = sampler.sample(x0=[0.0, 3.0], n_samples=1000)
    >>> samples.shape
    (1000, 2)
    """

    def __init__(
        self,
        target_log_pdf,
        step_size: float | ArrayLike = 1.0,
        rng: np.random.Generator | None = None,
    ):
        if not callable(target_log_pdf):
            raise TypeError(f"target_log_pdf must be callable, got {type(target_log_pdf).__name__}")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError(
                f"rng must be a numpy.random.Generator or None, got {type(rng).__name__}"
            )

        # Keep the caller's own distinction between a scalar and a sequence, so
        # the proposal receives the step size in the shape it was written in.
        step_was_scalar = np.asarray(step_size).ndim == 0
        validated_step = _as_1d_float(step_size, "step_size")
        if np.any(validated_step <= 0.0):
            raise ValueError("step_size must be strictly positive")

        self.target_log_pdf = target_log_pdf
        # The copy keeps a later change to the caller's array out of the sampler.
        self.step_size = float(validated_step[0]) if step_was_scalar else validated_step.copy()
        self.rng = np.random.default_rng() if rng is None else rng

    def sample(self, x0: ArrayLike, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw a chain of n_samples states, starting from x0.

        Parameters
        ----------
        x0 : array_like
            Starting state. A scalar is treated as one dimension.
        n_samples : int
            Length of the chain, including the starting state.

        Returns
        -------
        samples : numpy.ndarray
            Chain of shape (n_samples, d).
        accepted : numpy.ndarray
            Boolean array of shape (n_samples,). accepted[i] is True when the
            proposal at step i was accepted. accepted[0] is False because the
            starting state is not a proposal.

        Raises
        ------
        TypeError
            If x0 is not numeric, or n_samples is not an integer.
        ValueError
            If x0 is empty, not one-dimensional or not finite; if n_samples is
            below 1; if step_size does not match the dimension of x0; or if
            target_log_pdf does not return one finite number at x0. During
            sampling, nan and positive infinity from target_log_pdf are
            rejected, while minus infinity represents a rejected proposal.
        """
        start = _as_1d_float(x0, "x0")
        n_samples = _as_positive_int(n_samples, "n_samples")
        d = start.size

        step_values = np.atleast_1d(np.asarray(self.step_size, dtype=float))
        if step_values.size not in (1, d):
            raise ValueError(
                f"step_size must be a scalar or have {d} values to match x0, got {step_values.size}"
            )

        log_p_current = _as_log_density(self.target_log_pdf(start), "at x0")

        samples = np.zeros((n_samples, d))
        samples[0] = start
        accepted = np.zeros(n_samples, dtype=bool)

        for i in range(1, n_samples):
            current = samples[i - 1]
            proposed = gaussian_random_walk(
                current_x=current, step_size=self.step_size, rng=self.rng
            )
            log_p_proposed = _as_log_density(
                self.target_log_pdf(proposed),
                "during sampling",
                allow_negative_infinity=True,
            )

            # Symmetric proposal, so the acceptance ratio is just the density ratio.
            log_alpha = log_p_proposed - log_p_current
            if log_alpha >= 0 or self.rng.uniform() < np.exp(log_alpha):
                samples[i] = proposed
                accepted[i] = True
                log_p_current = log_p_proposed
            else:
                samples[i] = current

        return samples, accepted


def random_walk_metropolis_hastings(
    target_log_pdf,
    x0: ArrayLike,
    n_samples: int,
    step_size: float | ArrayLike = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from a target distribution in one or more dimensions.

    Builds a RandomWalkMetropolisHastings with the given settings and draws one
    chain from it.

    Parameters
    ----------
    target_log_pdf : callable
        Returns the log of the target density, which may be unnormalized. It may
        return minus infinity outside the support of the target.
    x0 : array_like
        Starting state of the chain. A scalar is treated as one dimension.
    n_samples : int
        Length of the chain, including the starting state.
    step_size : float or array_like, default=1.0
        Standard deviation of the Gaussian proposal. An array gives a separate
        scale per dimension and must then match the dimension of x0.
    rng : numpy.random.Generator or None, default=None
        Random number generator. When omitted a fresh one is created, which makes
        the run non-reproducible.

    Returns
    -------
    samples : numpy.ndarray
        Chain of shape (n_samples, d).
    accepted : numpy.ndarray
        Boolean array of shape (n_samples,). accepted[i] is True when the
        proposal at step i was accepted. accepted[0] is False because the
        starting state is not a proposal.

    Raises
    ------
    TypeError
        If target_log_pdf is not callable, x0 or step_size is not numeric,
        n_samples is not an integer, or rng is neither None nor a
        numpy.random.Generator.
    ValueError
        If x0 or step_size is empty, not one-dimensional or not finite; if
        step_size is not strictly positive or does not match the dimension of
        x0; if n_samples is below 1; or if target_log_pdf does not return one
        finite number at x0. During sampling, nan and positive infinity from
        target_log_pdf are rejected, while minus infinity represents a rejected
        proposal.
    """
    sampler = RandomWalkMetropolisHastings(
        target_log_pdf=target_log_pdf, step_size=step_size, rng=rng
    )
    return sampler.sample(x0=x0, n_samples=n_samples)


def random_walk_metropolis_hastings_1d(
    target_log_pdf,
    x0: float,
    n_samples: int,
    step_size: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from a one-dimensional target distribution.

    Same as random_walk_metropolis_hastings, but the chain is returned flat
    instead of as a single column. Given the same seed both draw the identical
    sequence of random numbers.

    Parameters
    ----------
    target_log_pdf : callable
        Returns the log of the target density, which may be unnormalized. It may
        return minus infinity outside the support of the target.
    x0 : float
        Starting state of the chain.
    n_samples : int
        Length of the chain, including the starting state.
    step_size : float, default=1.0
        Standard deviation of the Gaussian proposal.
    rng : numpy.random.Generator or None, default=None
        Random number generator. When omitted a fresh one is created, which makes
        the run non-reproducible.

    Returns
    -------
    samples : numpy.ndarray
        Chain of shape (n_samples,).
    accepted : numpy.ndarray
        Boolean array of shape (n_samples,).

    Raises
    ------
    TypeError
        Same cases as random_walk_metropolis_hastings.
    ValueError
        Same cases as random_walk_metropolis_hastings, and additionally if x0
        holds more than one value, since this function returns a flat chain and
        would otherwise drop the remaining dimensions.
    """
    start = _as_1d_float(x0, "x0")
    if start.size != 1:
        raise ValueError(
            f"random_walk_metropolis_hastings_1d needs a one-dimensional starting point, "
            f"got {start.size} values; use random_walk_metropolis_hastings instead"
        )
    samples, accepted = random_walk_metropolis_hastings(
        target_log_pdf, x0=x0, n_samples=n_samples, step_size=step_size, rng=rng
    )
    return samples[:, 0], accepted
