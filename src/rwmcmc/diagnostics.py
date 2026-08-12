"""Convergence diagnostics and plots for MCMC chains.

Statistics are computed with NumPy, figures with matplotlib. Most functions
here take the same three arguments:

samples : (n_samples,) or (n_samples, d) array from the samplers.
accepted : (n_samples,) boolean array from the samplers.
burn_in : number of leading samples to discard.
"""

import matplotlib.pyplot as plt
import numpy as np


def convert_2d(samples: np.ndarray) -> np.ndarray:
    """Return the chain as a 2D array of shape (n_samples, d).

    A 1D chain becomes a single column, so the rest of the module can treat
    every chain the same way.
    """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return samples[:, None]
    return samples


def acceptance_rate(accepted: np.ndarray, burn_in: int = 0) -> float:
    """Fraction of proposals accepted, ignoring the first burn_in samples.

    accepted[0] represents the starting state rather than a proposal, so it is
    excluded even when burn_in is zero. A low rate means the proposal steps are
    too large, a high rate means they are too small. The optimal value is about
    0.44 in one dimension and drops towards 0.234 in high dimensions.
    """
    accepted = np.asarray(accepted)
    first_proposal = max(1, burn_in)
    return float(accepted[first_proposal:].mean())


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
    ValueError
        If the chain has zero variance, so the correlation is undefined.
    """
    x = convert_2d(samples).astype(float)
    n, d = x.shape
    if max_lag is None:
        max_lag = min(n - 1, 200)

    x -= x.mean(axis=0)  # Autocovariance is defined on deviations from the mean.

    # Any length > 2*n avoids circular wrap-around; powers of two make the FFT fastest.
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=nfft, axis=0)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft, axis=0)[: max_lag + 1]
    if np.any(np.isclose(acov[0], 0.0)):
        raise ValueError("Autocorrelation is undefined because the chain has zero variance.")
    norm_acov = acov / acov[0]
    return norm_acov


def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """Effective sample size per dimension.

    Uses Geyer's initial positive sequence: consecutive autocorrelation pairs
    (rho_2k + rho_2k+1) are accumulated while their sum stays positive. The
    result says how many independent samples the correlated chain is worth.
    """
    x = convert_2d(samples=samples)
    n, d = x.shape
    rho = autocorrelation(x, max_lag=min(n - 1, 1000))
    ess = np.empty(d)
    for j in range(d):
        pair_sums = rho[1:-1:2, j] + rho[2::2, j]
        # pairs: rho1+rho2, rho3+rho4, ...
        tau = 1.0
        for p in pair_sums:
            if p < 0:
                break
            tau += 2.0 * p
        ess[j] = n / tau
    return ess


def summary(samples: np.ndarray, accepted: np.ndarray, burn_in: int = 0) -> dict:
    """Collect the diagnostics of a chain into one dictionary.

    Returns
    -------
    dict
        Keys n_samples and n_dim (int), acceptance_rate (float), and mean, std
        and ess (arrays of length d).
    """
    x = convert_2d(samples=samples)[burn_in:]
    return {
        "n_samples": int(x.shape[0]),
        "n_dim": int(x.shape[1]),
        "acceptance_rate": acceptance_rate(accepted, burn_in),
        "mean": x.mean(axis=0),
        "std": x.std(axis=0),
        "ess": effective_sample_size(x),
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
    x = convert_2d(samples)
    n, d = x.shape

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
    x = convert_2d(samples)[burn_in:]
    n, d = x.shape
    if d < 2:
        raise ValueError("corner plot needs at least 2 dimensions")
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
