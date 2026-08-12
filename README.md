# rwmcmc

[![tests](https://github.com/korkmazgoz-caglar/random-walk-mcmc/actions/workflows/tests.yml/badge.svg)](https://github.com/korkmazgoz-caglar/random-walk-mcmc/actions/workflows/tests.yml)

A lightweight, pip-installable Python package for **Random-Walk Metropolis-Hastings (RWMH)** sampling, with built-in convergence diagnostics, a one-figure visual dashboard, and a fully reproducible, config-driven command line interface.

Developed as a research project for the *Sustainable Computational Engineering* course at RWTH Aachen University. The package is also used as one of the samplers in the collaborative benchmarking pipeline [mcmc-bench](https://github.com/thealanjason/mcmc-bench).

## Motivation: why Bayesian calibration?

Scientific and engineering simulation models almost always contain parameters that cannot be measured directly; friction coefficients, material constants, etc. What we *do* have are noisy observations of the model output. **Bayesian calibration** turns this around: instead of picking one "best" parameter value, it computes the *posterior distribution* density, which parameter values are plausible given the data, and how uncertain we are. This matters because a single fitted number hides risk: two parameter sets can fit the data equally well while predicting very different behaviour outside the observed range (ill-posedness). The posterior makes that uncertainty visible and quantifiable.

The catch: for realistic models the posterior has no closed form meaning it is only known up to a normalizing constant. **Markov Chain Monte Carlo (MCMC)** solves exactly this problem: it draws samples from a distribution using only *unnormalized* (log-)density evaluations. The **random-walk Metropolis-Hastings** algorithm is the simplest robust member of this family, which makes it both a good teaching algorithm and a baseline in benchmarks against more advanced samplers (as done in mcmc-bench, it is still pretty good against SMC, dynesty, emcee, slice).

Because MCMC samples are *correlated* by construction, raw chains must never be trusted blindly. This package therefore has diagnostics:
* acceptance rate
* autocorrelation
* effective sample size (ESS)
* visual dashboard

## Features

- 1D and n-dimensional random-walk Metropolis-Hastings samplers (`samplers.py`)
- Reusable `RandomWalkMetropolisHastings` class holding the target, the proposal scale, and the generator; the functions above are thin wrappers over it
- Input validation in the sampler and in the CLI: a wrong argument or a wrong config entry is refused with a clear message before any sampling starts
- Symmetric Gaussian proposal, separated into its own module (`proposals.py`)
- Example target distributions: standard Gaussian and a two-dimensional banana-shaped density (`targets.py`)
- Diagnostics: acceptance rate, running mean, FFT-based autocorrelation, effective sample size following Geyer's initial positive sequence (`diagnostics.py`)
- One-figure visual dashboard: trace, histogram, running mean, and autocorrelation panels per dimension, and a summary line
- Corner plot for multi-dimensional cases
- Config-driven CLI (`rwmcmc-run`) that records complete run metadata for reproducibility

## Installation

Python 3.10 or newer and Git are required. An isolated virtual environment keeps this
package and its dependencies separate from other projects. On Linux, macOS, or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/korkmazgoz-caglar/random-walk-mcmc.git@v0.2.1"
```

On Windows PowerShell, create and activate the environment with `py -m venv .venv` and
`.\.venv\Scripts\Activate.ps1`, then use the same `python -m pip` installation command.
Pinning `v0.2.1` makes a later installation select the same released source.

For development (editable install, from a local clone):

```bash
git clone https://github.com/korkmazgoz-caglar/random-walk-mcmc.git
cd random-walk-mcmc
python -m pip install -e ".[dev,notebooks]"
```

Verify that the package, version, and command-line entry point are visible from the active
Python environment:

```bash
python -c "import rwmcmc; print(rwmcmc.__version__)"
rwmcmc-run --help
```

Optional extras:

| Extra | Installs | Purpose |
|---|---|---|
| `[notebooks]` | jupyter, seaborn, scipy, etc | running the example notebooks |
| `[dev]` | pytest, ruff, scipy | running the test suite, formatting and linting |

## Quickstart

### Python API

```python
import numpy as np
from rwmcmc import random_walk_metropolis_hastings, banana_log_pdf, dashboard, summary

rng = np.random.default_rng(42)  # seed -> reproducible chain
samples, accepted = random_walk_metropolis_hastings(
    banana_log_pdf, x0=[0.0, 3.0], n_samples=50000, step_size=[4.0, 2.0], rng=rng
)

print(summary(samples, accepted, burn_in=5000))
fig = dashboard(samples, accepted, burn_in=5000, param_names=["x1", "x2"])
fig.savefig("dashboard.png")
```

### Reusable sampler class

Use the class API when the target, proposal scale, and random number generator
should be reused across runs:

```python
import numpy as np
from rwmcmc import RandomWalkMetropolisHastings, banana_log_pdf

sampler = RandomWalkMetropolisHastings(
    banana_log_pdf,
    step_size=[4.0, 2.0],
    rng=np.random.default_rng(42),
)
samples, accepted = sampler.sample(
    x0=[0.0, 3.0],
    n_samples=50000,
)
```

### Your own target

A target can be any callable that accepts the current position and returns its
unnormalized log-density. Additive normalization constants are unnecessary because they
cancel in the Metropolis-Hastings ratio:

```python
import numpy as np
from rwmcmc import random_walk_metropolis_hastings

def my_log_pdf(x):
    x = np.asarray(x)
    return -0.5 * np.sum((x / 2.0) ** 2)

samples, accepted = random_walk_metropolis_hastings(
    my_log_pdf,
    x0=[0.0],
    n_samples=10000,
    step_size=[2.0],
    rng=np.random.default_rng(42),
)
```

### Command line

If you cloned the repository, copy the tracked example and run it:

```bash
cp examples/config.yaml my_run.yaml
rwmcmc-run my_run.yaml
```

For a pip-only installation, the repository's `examples/` directory is not created in
your working directory. The command below downloads only the released example
configuration into the current directory; run it afterwards:

```bash
curl -L https://raw.githubusercontent.com/korkmazgoz-caglar/random-walk-mcmc/v0.2.1/examples/config.yaml -o my_run.yaml
rwmcmc-run my_run.yaml
```

Alternatively, create `my_run.yaml` manually:

```yaml
target: banana          # gaussian_1d | banana
x0: [0.0, 3.0]          # starting point; dimension must match the target
n_samples: 50000        # complete chain length, including burn-in
step_size: [4.0, 2.0]   # proposal std; scalar or one value per dimension
burn_in: 5000           # leading samples excluded from diagnostics
seed: 42                # integer -> deterministic; null -> draw and record a seed
output_dir: results     # output directory
save_dashboard: true    # write dashboard.png
save_corner: true       # write corner.png; requires at least two dimensions
param_names: [x1, x2]   # optional labels
```

All entries are validated before sampling starts. In particular, `n_samples` must be at
least 2, `burn_in` must leave at least two samples for diagnostics, proposal scales must be
finite and strictly positive, `seed` must be a non-negative integer or `null`, and
`param_names` must match the target dimension. Unknown keys are rejected, so a typo
such as `burnin` cannot silently fall back to a default.

Invalid configurations produce a short command-line error and exit status 2 without
creating the output directory:

```text
usage: rwmcmc-run [-h] config
rwmcmc-run: error: burn_in must be smaller than n_samples (50000), got 60000
```

A successful run prints the resolved seed, acceptance rate, and output location. It writes
the following files into the configured output directory:

| File | Content |
|---|---|
| `samples.npy` | the chain, shape `(n_samples, d)` |
| `accepted.npy` | proposal decisions; element 0 marks the starting state, not a proposal |
| `dashboard.png` | visual diagnostics, written when `save_dashboard: true` |
| `corner.png` | pairwise diagnostics, written when `save_corner: true` |
| `run_metadata.yaml` | full record of the run (see below) |


## Reproducibility

Reproducibility of this package:

- **The seed is never hard-coded.** Every sampler takes an explicit `rng` argument; the CLI takes a `seed` entry in the config file.
- **The user chooses.** Set `seed: 42` for a deterministic run, or `seed: null` to let the package draw a fresh seed.
- **A typo is not accepted silently.** The CLI refuses any setting it does not know, so `burnin: 500` fails with a message instead of running with the default and recording a value that was never applied.
- **Nothing is lost.** Every run writes a `run_metadata.yaml` holding the requested configuration next to the effective one with all defaults filled in, the seed that was actually used together with the NumPy bit generator, the package version and the source revision it came from (when available), the Python, NumPy, Matplotlib, PyYAML, operating system and architecture it ran on, the summary statistics, and SHA-256 digests of the `.npy` outputs. Any run — including a "random" one — can therefore be replayed exactly by feeding the recorded seed back into the config, in the same source revision and with the same library versions the metadata records. The metadata makes a difference in that environment visible; it does not rebuild the environment for you.

The digests let you check a replay without keeping the old files: compare the `sha256` entries under `outputs` in the two metadata files. They are taken from the files on disk, so they certify that the two runs wrote byte for byte the same file. That is a narrower claim than saying two different machines would compute the same numbers. The test suite checks the first: the same seed produces identical files, and a drawn seed replayed from its own metadata reproduces the run exactly.

## Diagnostics in one paragraph

MCMC chains are correlated: each sample is a small step from the previous one, so 8000 samples carry less information than 8000 independent draws. The **autocorrelation function** measures how long this "memory" lasts, and the **effective sample size** (ESS) converts it into a single number: how many independent samples the chain is worth. The dashboard shows both, next to the trace and histogram, so that step-size tuning problems are visible at a glance. The ESS estimator follows Geyer's initial positive sequence method (see references) via FFT.

Each parameter gets one dashboard row with four panels:

- **Trace:** the chain over time. Long flat stretches indicate many rejections; slow drift
  suggests steps that are too small or a chain that is too short.
- **Histogram:** the sampled marginal distribution after burn-in.
- **Running mean:** a cumulative estimate that should stabilize as the chain progresses.
- **Autocorrelation:** dependence at increasing lags; faster decay generally means better
  mixing and a larger ESS.

The summary line reports the post-burn-in sample count, proposal acceptance rate, and ESS.
An acceptance rate near 44% is a useful reference in one dimension; the asymptotic
high-dimensional reference is about 23.4%. These are tuning guides, not correctness tests.
A very small ESS relative to the retained chain length indicates poor mixing.

For multi-dimensional runs, `corner.png` complements the per-parameter dashboard by showing
pairwise joint structure, including correlations and nonlinear shapes such as the banana
target's curved ridge.

## Use in mcmc-bench

This package is consumed by the Nextflow benchmarking pipeline [mcmc-bench](https://github.com/thealanjason/mcmc-bench), where it is installed via pip from this repository and compared against `emcee`, `dynesty`, `Slice`, and `SMC` on a surrogate-model calibration task. This demonstrates the package working as an installable dependency in an independent project.

Another project can install a specific release through its dependency file using, for
example:

```text
git+https://github.com/korkmazgoz-caglar/random-walk-mcmc.git@v0.2.1
```

Pinning a tag or commit prevents an environment rebuild from silently selecting a newer
revision.

## Troubleshooting

- **`rwmcmc-run: command not found`:** reactivate the virtual environment and confirm
  the package is installed there with `python -m pip show rwmcmc`.
- **A notebook cannot import `rwmcmc`:** its kernel may use a different Python environment.
  Compare `import sys; print(sys.executable)` inside the notebook with
  `python -c "import sys; print(sys.executable)"` in the terminal.
- **`ImportError: attempted relative import with no known parent package`:** do not execute
  files such as `src/rwmcmc/samplers.py` directly; install and import the package.
- **A zero-variance diagnostic error:** the chain did not move. Inspect the trace and reduce
  an excessively large proposal scale, or check whether the target is degenerate.
- **`rwmcmc-run: error: ...`:** the CLI rejected the configuration before sampling. The
  message identifies the entry to correct; no output is written for an invalid config.
- **Sampler `TypeError` or `ValueError`:** invalid sampler arguments are rejected before the
  chain starts rather than producing a misleading result.

## Development

```bash
python -m pip install -e ".[dev]"
ruff format .           # apply the formatting
ruff check .            # lint
python -m pytest -v     # run the test suite
```

Formatting and linting use [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`: line length 100, rule sets `E` (pycodestyle), `F` (pyflakes) and `I` (import order), with `tutorials/` and `*.md` excluded. GitHub Actions checks formatting, linting, and tests on pushes to `main` and on every pull request, using Python 3.10 and 3.13.

The suite covers output shapes, seed reproducibility, statistical correctness of the sampler against a known target, the diagnostics against analytical results (i.i.d. limits and the AR(1) process with known ESS ratio), and the errors raised for invalid input on both the Python API and command line side.

## Project structure

```
src/rwmcmc/
  __init__.py      public API
  targets.py       example log-densities (gaussian_1d, banana)
  proposals.py     symmetric Gaussian random-walk proposal
  samplers.py      1D and n-dimensional RWMH samplers
  diagnostics.py   statistics + visual dashboard
  cli.py           rwmcmc-run entry point
tests/             pytest suite
tutorials/         example notebooks (01-06) and input data
examples/          example run configuration
.github/workflows/ continuous integration: lint and tests
CHANGELOG.md       release history and unreleased changes
CITATION.cff       citation metadata
```

## References

- H. Haario, E. Saksman, and J. Tamminen (1999), *Adaptive proposal distribution for random walk Metropolis algorithm*, Computational Statistics 14(3), 375–395. https://link.springer.com/article/10.1007/s001800050022
- C. J. Geyer (1992), *Practical Markov Chain Monte Carlo*, Statistical Science 7(4). https://projecteuclid.org/journals/statistical-science/volume-7/issue-4/Practical-Markov-Chain-Monte-Carlo/10.1214/ss/1177011137.full
- A. Gelman et al., *Bayesian Data Analysis*, 3rd ed., Ch. 11. https://sites.stat.columbia.edu/gelman/book/
- Stan Reference Manual, *Effective Sample Size*. https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html

## Citation

See [CITATION.cff](CITATION.cff). Licensed under the [MIT License](LICENSE).