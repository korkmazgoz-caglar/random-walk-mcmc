"""
Diagnostics and Visual Dashboard
...
Numpy based function. Plotting uses matplotlib. 'arviz' not required but can be installed with "pip install rwmcmc[diagnostics]".
'to_arviz' convertz data to iData for advanced diagnostics (ESS...)

samples : (n_samples,) or (n_samples, d) array from the samplers in this package.
accepted : (n_samples,) boolean array from the samplers.
burn_in : leading samples to discard before computing statistics.
"""

import numpy as np
import matplotlib.pyplot as plt

def convert_2d(samples: np.ndarray) -> np.ndarray:
    """This function returns samples with shape (n, d) as numpy array, d is the dimension."""
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return samples[:, None]
    return samples

def acceptance_rate(accepted: np.ndarray, burn_in: int=0) -> float:
    """Accepted proposals divided by total number of proposals."""

    return float(np.asarray(accepted)[burn_in:].mean())

def running_mean(samples: np.ndarray)->np.ndarray:
    """Cumulative mean of the chain to check the behavior of the chain."""
    x = convert_2d(samples)
    n = np.arange(1, x.shape[0]+1)[:, None]
    return np.cumsum(x, axis=0)/n

def autocorrelation(samples: np.ndarray, max_lag: int|None = None) -> np.ndarray:
    """rho[0] = 1 by construction. Fast decay means better mixing!"""
    x = convert_2d(samples).astype(float)
    n,d = x.shape
    if max_lag is None:
        max_lag = min(n-1,200)

    x-=x.mean(axis=0) # Get ready for the Fourier Transform...

    nfft = int(2**np.ceil(np.log2(2*n))) # we can just select it as > 2*n but FFT works faster with power of 2..
    f = np.fft.rfft(x, n=nfft, axis=0)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft, axis=0)[:max_lag+1]
    if acov[0].any == 0:
        print("Division by zero, all data are the same. Here, I need to add pytest!\n")
        return 0
    norm_acov = acov / acov[0]
    return norm_acov


def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """Effective sample size per dimension
    
    Geyer's initial positive sequence: sums of autocorrelation pairs (rho_2k + rho_(2k+1) are accumulated as long as they the sum is positive. ESS: How many independent samples the correlated chain is worth.)"""
    x = convert_2d(samples=samples)
    n,d = x.shape
    rho = autocorrelation(x, max_lag=min(n-1, 1000))
    ess = np.empty(d)
    for j in range(d):
        pair_sums = rho[1:-1:2, j]+rho[2::2, j]
        # rho1+rho2, rho3+rho4 vs
        tau = 1.0
        for p in pair_sums:
            if p < 0:
                break
            tau += 2.0*p
        ess[j] = n/tau
    return ess


def summary(samples: np.ndarray, accepted: np.ndarray, burn_in: int=0) -> dict:
    """Summary statistics."""
    x = convert_2d(samples=samples)[burn_in:]
    return{
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
    figsize_per_row: tuple = (13., 3.),
):
    """Visual Dashboard: Trace, histogram, running mean, autocorrelation.

    One row per dimension, four panels per row, plus a summary line on top.
    Returns the matplotlib Figure; call plt.show() or fig.savefig().
    """
    x = convert_2d(samples)
    n, d = x.shape
  
    if param_names is None:
        if d > 1:
            param_names = []
            for j in range(d): 
                param_names.append(f"x{j}")
        else:
            param_names = ["x"]

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

    # ess_txt = ", ".join(f"{e:.0f}" for e in stats["ess"])
    values = []

    for e in stats["ess"]:
        values.append(f"{e:.0f}")

    ess_txt = ", ".join(values)
    txt = (
        f"n = {stats['n_samples']} (post burn-in) | dim = {d} | "
        f"acceptance = {stats['acceptance_rate']:.2%} | ESS = [{ess_txt}]"
    )
    fig.suptitle(txt, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig