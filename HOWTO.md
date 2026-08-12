# HOWTO

A task-oriented guide to `rwmcmc`. Each section answers one concrete question: *where do you do what?* For background and motivation, see the [README](README.md).

## 1. How do I install the package?

As a user (directly from GitHub):

```bash
pip install git+https://github.com/korkmazgoz-caglar/random-walk-mcmc.git
```

As a developer:

```bash
git clone https://github.com/korkmazgoz-caglar/random-walk-mcmc.git
cd random-walk-mcmc
pip install -e ".[dev,notebooks]"
```

Verify the installation:

```bash
python -c "import rwmcmc; print(rwmcmc.__version__)"
```

## 2. How do I run my first sampling from the command line?

Never edit `examples/config.yaml` in place, copy it and make it yours:

```bash
cp examples/config.yaml my_run.yaml
rwmcmc-run my_run.yaml
```

You should see something like:

```
run complete: seed=42 acceptance=38.00%
outputs written to /path/to/results
```

## 3. What do the config entries mean?

```yaml
target: banana          # which log-density to sample: gaussian_1d | banana
x0: [0.0, 3.0]          # where the chain starts; dimension must match the target
n_samples: 8000         # chain length
step_size: [4.0, 2.0]   # proposal std; scalar or one value per dimension
burn_in: 500            # leading samples excluded from statistics
seed: 42                # integer -> deterministic run; null -> fresh random seed
output_dir: results     # where output files are written
save_dashboard: true    # also write dashboard.png
save_corner: true       # also write corner.png (pairwise joint plot, needs >= 2 dims)
param_names: [x1, x2]   # labels used in the dashboard (optional)
```

The documented settings are validated before the sampling loop starts: `burn_in` must be smaller than `n_samples`, `seed` must be a non-negative integer or `null`, `save_corner` requires an `x0` with at least two dimensions, and `param_names` must have one name per dimension. A setting the CLI does not know is refused as well, so a misspelled `burnin` cannot quietly fall back to the default. A file that breaks one of these rules is reported as a short command line error rather than a traceback:

```
usage: rwmcmc-run [-h] config
rwmcmc-run: error: burn_in must be smaller than n_samples (8000), got 9000
```

The command then exits with status 2 and writes nothing: no output directory, no samples, no metadata.

## 4. Where are my results and what is in them?

Inside `output_dir` (default `results/`):

- `samples.npy` — the chain as a NumPy array, shape `(n_samples, d)`. Load with `np.load("results/samples.npy")`.
- `accepted.npy` — boolean array; `True` where the proposal was accepted.
- `dashboard.png` — four diagnostic panels per dimension (see section 6).
- `corner.png` — pairwise joint distributions between dimensions (only written when `save_corner: true` and the target has at least 2 dimensions).
- `run_metadata.yaml` — the complete, human-readable record of the run, in a small schema of its own (`schema_version: 1`): the requested configuration and the effective one with every default filled in, the seed actually used and the NumPy bit generator, the package version and the git revision it came from (when available), the Python and library versions and the platform, the summary statistics, and SHA-256 digests of `samples.npy` and `accepted.npy`.

## 5. How do I reproduce a run, including a "random" one?

Every run records the seed it actually used in `run_metadata.yaml`, even when the config said `seed: null`. To repeat any run exactly:

1. Open `results/run_metadata.yaml` and read the seed under `random_number_generator`.
2. Put that number into your config: `seed: <the recorded seed>`.
3. Run `rwmcmc-run` again.

Repeated with the same source revision and the same library versions that the metadata records, the replayed chain is **bit-for-bit identical** to the original, and you can check it without keeping the old files: compare the `sha256` entries under `outputs` in the two `run_metadata.yaml` files. The digest is taken from the file on disk, so it tells you the two files are byte for byte the same; it does not claim that a different machine would compute the same numbers. The test suite enforces the first.

## 6. How do I read the dashboard?

Each parameter gets one row with four panels:

- **trace** — the raw chain over iterations. It should look like stationary "fuzzy noise". Long flat stretches mean many rejections (step size too large); slow drifts mean tiny steps (step size too small). The grey band marks the burn-in you configured.
- **histogram** — the sampled distribution after burn-in; this is your estimate of the target/posterior.
- **running mean** — the cumulative mean. It should flatten out; if it still moves up and down at the end, the chain is too short.
- **autocorrelation** — how similar samples are at increasing lags. Bars should decay to zero quickly; slow decay means strongly correlated samples and a low ESS.

The line above the panels reports the post-burn-in sample count, acceptance rate, and ESS per dimension. Rules of thumb: acceptance around 44% is optimal for 1D targets, around 23% in higher dimensions; ESS far below `n_samples` signals poor mixing.

The **corner plot** (`corner.png`) complements the dashboard: while the dashboard looks at each dimension separately, the corner plot shows the *joint* structure between dimensions. Correlations and non-linear shapes, like the curved banana density, are only visible here; the per-dimension histograms of the dashboard cannot show them.

## 7. How do I use the package from Python / a notebook?

```python
import numpy as np
from rwmcmc import (
    random_walk_metropolis_hastings,   # sampler
    gaussian_1d_log_pdf, banana_log_pdf,  # example targets
    summary, dashboard, corner,        # diagnostics
)

rng = np.random.default_rng(42)
samples, accepted = random_walk_metropolis_hastings(gaussian_1d_log_pdf, x0=0.0, n_samples=10000, step_size=2.4, rng=rng)
stats = summary(samples, accepted, burn_in=1000)
fig = dashboard(samples, accepted, burn_in=1000)
```

If you run several chains with the same settings, the class API keeps the target, the proposal scale, and the generator in one place:

```python
import numpy as np
from rwmcmc import RandomWalkMetropolisHastings, gaussian_1d_log_pdf

sampler = RandomWalkMetropolisHastings(
    gaussian_1d_log_pdf,
    step_size=2.4,
    rng=np.random.default_rng(42),
)
samples, accepted = sampler.sample(x0=0.0, n_samples=10000)
```

The module-level functions shown above are thin wrappers around this class, so both produce the identical chain when given separate generators initialized with the same seed.

For multi-dimensional targets, the pairwise joint structure is one call away:

```python
samples, accepted = random_walk_metropolis_hastings(
    banana_log_pdf, x0=[0.0, 3.0], n_samples=10000, step_size=[4.0, 2.0], rng=rng
)
fig = corner(samples, burn_in=1000, param_names=["x1", "x2"])
```

To sample your **own** target, pass any function that returns the log-density (up to an additive constant):

```python
def my_log_pdf(x):
    return -0.5 * np.sum((x / 2.0) ** 2)   # N(0, 2^2)

samples, accepted = random_walk_metropolis_hastings(my_log_pdf, x0=[0.0], n_samples=10000, step_size=[2.0], rng=rng)
```

## 8. How do I run the tests?

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

All tests must pass. They check output shapes, seed reproducibility, sampler correctness on a known target, and the diagnostics against analytical results.

## 9. How do I use rwmcmc inside another project?

Add the GitHub URL to your dependency list, e.g. in a conda `environment.yml`:

```yaml
dependencies:
  - pip
  - pip:
      - git+https://github.com/korkmazgoz-caglar/random-walk-mcmc.git
```

then import it like any other package. The [mcmc-bench](https://github.com/thealanjason/mcmc-bench) pipeline uses exactly this mechanism to benchmark `rwmcmc` against `emcee`, `dynesty`, and other samplers.

## 10. Common problems

- **`ImportError: attempted relative import with no known parent package`** — you ran a module file directly (`python src/rwmcmc/samplers.py`). Don't; import the package instead (section 7).
- **`rwmcmc-run: command not found`** — the console script is created at install time; re-run `pip install -e .` in the environment you are using.
- **Notebook can't import rwmcmc** — the Jupyter kernel is a different Python than the one you installed into. Compare `import sys; print(sys.executable)` in the notebook with `which python` in the shell.
- **`ValueError: ... zero variance`** from the diagnostics — your chain never moved (step size far too large, or a degenerate target). Inspect the trace and reduce the step size.
- **`rwmcmc-run: error: ...` with no traceback** — the configuration was refused before sampling started, and the message names the setting to fix. The command exits with status 2 and writes no output.
- **`TypeError: n_samples must be an integer`, `ValueError: step_size must be strictly positive`, and similar** — the sampler checks its arguments before it starts, so a wrong value fails right away instead of producing a chain that looks fine but is meaningless.