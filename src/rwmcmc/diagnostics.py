"""Convergence diagnostics and plots for MCMC chains.

Statistics are computed with NumPy, figures with matplotlib. Most functions
here take the same three arguments:

samples : (n_samples,) or (n_samples, d) array from the samplers.
accepted : (n_samples,) boolean array from the samplers.
burn_in : number of leading samples to discard.
"""

import numbers

import matplotlib.pyplot as plt
import numpy as np


def _validated_samples(samples: np.ndarray, min_samples: int = 1) -> np.ndarray:
    """Return a finite chain with shape (n_samples, d)."""
    try:
        x = np.asarray(samples, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("samples must be numeric") from exc

    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(f"samples must be a 1D or 2D array, got {x.ndim} dimensions")

    if x.shape[0] < min_samples:
        raise ValueError(f"samples must contain at least {min_samples} rows, got {x.shape[0]}")
    if x.shape[1] == 0:
        raise ValueError("samples must contain at least one dimension")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must contain only finite values")
    return x


def _validated_burn_in(burn_in: int, n_samples: int, min_remaining: int) -> int:
    """Return a non-negative burn-in that leaves enough samples."""
    if isinstance(burn_in, bool) or not isinstance(burn_in, numbers.Integral):
        raise TypeError(f"burn_in must be an integer, got {type(burn_in).__name__}")
    burn_in = int(burn_in)
    if burn_in < 0:
        raise ValueError(f"burn_in must not be negative, got {burn_in}")
    remaining = n_samples - burn_in
    if remaining < min_remaining:
        raise ValueError(f"burn_in must leave at least {min_remaining} samples, got {remaining}")
    return burn_in


def _validated_accepted(accepted: np.ndarray, expected_length: int | None = None) -> np.ndarray:
    """Return the sampler's one-dimensional boolean proposal decisions."""
    values = np.asarray(accepted)
    if values.ndim != 1:
        raise ValueError(f"accepted must be a 1D array, got {values.ndim} dimensions")
    if values.size < 2:
        raise ValueError("accepted must contain the starting state and at least one proposal")
    if values.dtype.kind != "b":
        raise TypeError("accepted must contain boolean values")
    if expected_length is not None and values.size != expected_length:
        raise ValueError(
            f"accepted must have the same length as samples ({expected_length}), got {values.size}"
        )
    return values


def _validated_param_names(param_names: list[str] | None, dimension: int) -> list[str] | None:
    """Return labels that match the chain dimension."""
    if param_names is None:
        return None
    if not isinstance(param_names, (list, tuple)) or not all(
        isinstance(name, str) for name in param_names
    ):
        raise TypeError("param_names must be a list of strings")
    if len(param_names) != dimension:
        raise ValueError(
            f"param_names must have one name per dimension ({dimension}), got {len(param_names)}"
        )
    return list(param_names)


def convert_2d(samples: np.ndarray) -> np.ndarray:
    """Return a finite chain as a 2D array of shape (n_samples, d).

    A 1D chain becomes a single column, so the rest of the module can treat
    every chain the same way.
    """
    return _validated_samples(samples)


def acceptance_rate(accepted: np.ndarray, burn_in: int = 0) -> float:
    """Fraction of proposals accepted, ignoring the first burn_in samples.

    accepted[0] represents the starting state rather than a proposal, so it is
    excluded even when burn_in is zero. A low rate means the proposal steps are
    too large, a high rate means they are too small. The optimal value is about
    0.44 in one dimension and drops towards 0.234 in high dimensions.
    """
    values = _validated_accepted(accepted)
    burn_in = _validated_burn_in(burn_in, values.size, min_remaining=1)
    first_proposal = max(1, burn_in)
    return float(values[first_proposal:].mean())


def running_mean(samples: np.ndarray) -> np.ndarray:
    """Cumulative mean of the chain, one row per step.

    Returns an array of shape (n_samples, d). Once the curve flattens, the
    chain has settled.
    """
    x = convert_2d(samples)
    n = np.arange(1, x.shape[0] + 1)[:, None]
    return np.cumsum(x, axis=0) / n


def autocorrelation(samples: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalized autocorrelation up to max_lag, computed with an FFT.

    rho[0] is 1 by construction. The faster the values decay, the less
    correlated the samples are. Returns an array of shape (max_lag + 1, d).

    Raises
    ------
    TypeError
        If max_lag is not an integer or None.
    ValueError
        If the chain is invalid, has fewer than two rows or has zero variance,
        or if max_lag is outside the available range.
    """
    x = _validated_samples(samples, min_samples=2)
    n, d = x.shape
    if max_lag is None:
        max_lag = min(n - 1, 200)
    else:
        if isinstance(max_lag, bool) or not isinstance(max_lag, numbers.Integral):
            raise TypeError(f"max_lag must be an integer or None, got {type(max_lag).__name__}")
        max_lag = int(max_lag)
        if not 0 <= max_lag < n:
            raise ValueError(f"max_lag must be between 0 and {n - 1}, got {max_lag}")

    centered = x - x.mean(axis=0)

    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(centered, n=nfft, axis=0)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft, axis=0)[: max_lag + 1]
    if np.any(np.isclose(acov[0], 0.0)):
        raise ValueError("Autocorrelation is undefined because the chain has zero variance.")
    return acov / acov[0]


def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """Effective sample size per dimension.

    Uses Geyer's initial positive sequence: consecutive autocorrelation pairs
    (rho_2k + rho_2k+1), starting with rho_0 + rho_1, are accumulated while
    their sum stays positive. Negatively correlated chains can have an
    effective sample size larger than the number of draws.
    """
    x = _validated_samples(samples, min_samples=2)
    n, d = x.shape
    rho = autocorrelation(x, max_lag=min(n - 1, 1000))
    paired_length = rho.shape[0] - rho.shape[0] % 2
    ess = np.empty(d)
    for j in range(d):
        pair_sums = rho[:paired_length, j].reshape(-1, 2).sum(axis=1)
        tau = -1.0
        for pair_sum in pair_sums:
            if pair_sum <= 0:
                break
            tau += 2.0 * pair_sum
        ess[j] = np.inf if tau <= 0 else n / tau
    return ess


def summary(samples: np.ndarray, accepted: np.ndarray, burn_in: int = 0) -> dict:
    """Collect the diagnostics of a chain into one dictionary.

    Returns
    -------
    dict
        Keys n_samples and n_dim (int), acceptance_rate (float), and mean, std
        and ess (arrays of length d).
    """
    x = _validated_samples(samples, min_samples=2)
    values = _validated_accepted(accepted, expected_length=x.shape[0])
    burn_in = _validated_burn_in(burn_in, x.shape[0], min_remaining=2)
    retained = x[burn_in:]
    first_proposal = max(1, burn_in)
    return {
        "n_samples": int(retained.shape[0]),
        "n_dim": int(retained.shape[1]),
        "acceptance_rate": float(values[first_proposal:].mean()),
        "mean": retained.mean(axis=0),
        "std": retained.std(axis=0),
        "ess": effective_sample_size(retained),
    }


def dashboard(
    samples: np.ndarray,
    accepted: np.ndarray,
    burn_in: int = 0,
    param_names: list[str] | None = None,
    max_lag: int | None = None,
    figsize_per_row: tuple = (13.0, 3.0),
):
    """Plot trace, histogram, running mean and autocorrelation in one figure.

    One row per dimension, four panels per row, with a summary line on top.

    Parameters
    ----------
    param_names : list of str or None, default=None
        Axis labels. Defaults to x0, x1, ... when d > 1.
    max_lag : int or None, default=None
        Largest lag shown in the autocorrelation panel.
    figsize_per_row : tuple, default=(13.0, 3.0)
        Figure width and per-row height, in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = _validated_samples(samples, min_samples=2)
    n, d = x.shape
    burn_in = _validated_burn_in(burn_in, n, min_remaining=2)
    param_names = _validated_param_names(param_names, d)
    if param_names is None:
        param_names = [f"x{j}" for j in range(d)] if d > 1 else ["x"]

    stats = summary(x, accepted, burn_in)
    xb = x[burn_in:]
    rho = autocorrelation(xb, max_lag=max_lag)

    fig, axes = plt.subplots(
        d, 4, figsize=(figsize_per_row[0], figsize_per_row[1] * d), squeeze=False
    )
    for j in range(d):
        name = param_names[j]

        ax = axes[j, 0]  # trace
        ax.plot(x[:, j], lw=0.5)
        if burn_in > 0:
            ax.axvspan(0, burn_in, color="gray", alpha=0.25, label="burn-in")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"trace: {name}", fontsize=9)

        ax = axes[j, 1]  # histogram (post burn-in)
        ax.hist(xb[:, j], bins=40, density=True, alpha=0.8)
        ax.set_title(f"histogram: {name}", fontsize=9)

        ax = axes[j, 2]  # running mean (post burn-in)
        ax.plot(running_mean(xb[:, j]), lw=0.8)
        ax.axhline(stats["mean"][j], color="k", ls="--", lw=0.8)
        ax.set_title(f"running mean: {name}", fontsize=9)

        ax = axes[j, 3]  # autocorrelation (post burn-in)
        ax.stem(rho[:, j], basefmt=" ", markerfmt=" ")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_title(f"autocorrelation: {name}", fontsize=9)

    ess_txt = ", ".join(f"{e:.0f}" for e in stats["ess"])
    txt = (
        f"n = {stats['n_samples']} (post burn-in) | dim = {d} | "
        f"acceptance = {stats['acceptance_rate']:.2%} | ESS = [{ess_txt}]"
    )
    fig.suptitle(txt, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def corner(
    samples: np.ndarray,
    burn_in: int = 0,
    param_names: list[str] | None = None,
    bins: int = 40,
    figsize_per_var: float = 2.2,
):
    """Plot each pair of parameters in a triangular grid.

    The diagonal shows the marginal histogram of each parameter, the lower
    triangle the joint histogram of each pair, so correlations are visible.

    Parameters
    ----------
    param_names : list of str or None, default=None
        Axis labels. Defaults to x0, x1, ...
    bins : int, default=40
        Number of bins in the marginal and joint histograms.
    figsize_per_var : float, default=2.2
        Size in inches given to each parameter, in both directions.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the chain has fewer than two dimensions.
    """
    x = _validated_samples(samples)
    burn_in = _validated_burn_in(burn_in, x.shape[0], min_remaining=1)
    x = x[burn_in:]
    n, d = x.shape
    if d < 2:
        raise ValueError("corner plot needs at least 2 dimensions")
    param_names = _validated_param_names(param_names, d)
    if param_names is None:
        param_names = [f"x{j}" for j in range(d)]

    fig, axes = plt.subplots(
        d, d, figsize=(figsize_per_var * d, figsize_per_var * d), squeeze=False
    )
    for i in range(d):  # row
        for j in range(d):  # column
            ax = axes[i, j]
            if j > i:
                ax.axis("off")  # upper triangle: empty
            elif i == j:
                ax.hist(x[:, i], bins=bins, density=True, alpha=0.8)
            else:
                ax.hist2d(x[:, j], x[:, i], bins=bins, cmap="viridis")
            if i == d - 1 and j <= i:
                ax.set_xlabel(param_names[j], fontsize=9)
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[i], fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"joint distributions (n = {n} post burn-in)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
