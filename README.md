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

From GitHub:

```bash
pip install git+https://github.com/korkmazgoz-caglar/random-walk-mcmc.git
```

For development (editable install, from a local clone):

```bash
git clone https://github.com/korkmazgoz-caglar/random-walk-mcmc.git
cd random-walk-mcmc
pip install -e ".[dev,notebooks]"
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
    banana_log_pdf, x0=[0.0, 3.0], n_samples=8000, step_size=[4.0, 2.0], rng=rng
)

print(summary(samples, accepted, burn_in=500))
fig = dashboard(samples, accepted, burn_in=500, param_names=["x1", "x2"])
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
    n_samples=8000,
)
```

### Command line

Copy the example configuration and adapt it:

```bash
cp examples/config.yaml my_run.yaml
rwmcmc-run my_run.yaml
```

This writes the following files into the configured output directory:

| File | Content |
|---|---|
| `samples.npy` | the chain, shape `(n_samples, d)` |
| `accepted.npy` | boolean acceptance array |
| `dashboard.png` | visual diagnostics |
| `corner.png` | visual diagnostics |
| `run_metadata.yaml` | full record of the run (see below) |

See [HOWTO.md](HOWTO.md) for a step-by-step guide.

## Reproducibility

Reproducibility of this package:

- **The seed is never hard-coded.** Every sampler takes an explicit `rng` argument; the CLI takes a `seed` entry in the config file.
- **The user chooses.** Set `seed: 42` for a deterministic run, or `seed: null` to let the package draw a fresh seed.
- **A typo is not accepted silently.** The CLI refuses any setting it does not know, so `burnin: 500` fails with a message instead of running with the default and recording a value that was never applied.
- **Nothing is lost.** Every run writes a `run_metadata.yaml` holding the requested configuration next to the effective one with all defaults filled in, the seed that was actually used together with the NumPy bit generator, the package version and the source revision it came from (when available), the Python, NumPy, Matplotlib, PyYAML, operating system and architecture it ran on, the summary statistics, and SHA-256 digests of the `.npy` outputs. Any run — including a "random" one — can therefore be replayed exactly by feeding the recorded seed back into the config, in the same source revision and with the same library versions the metadata records. The metadata makes a difference in that environment visible; it does not rebuild the environment for you.

The digests let you check a replay without keeping the old files: compare the `sha256` entries under `outputs` in the two metadata files. They are taken from the files on disk, so they certify that the two runs wrote byte for byte the same file. That is a narrower claim than saying two different machines would compute the same numbers. The test suite checks the first: the same seed produces identical files, and a drawn seed replayed from its own metadata reproduces the run exactly.

## Diagnostics in one paragraph

MCMC chains are correlated: each sample is a small step from the previous one, so 8000 samples carry less information than 8000 independent draws. The **autocorrelation function** measures how long this "memory" lasts, and the **effective sample size** (ESS) converts it into a single number: how many independent samples the chain is worth. The dashboard shows both, next to the trace and histogram, so that step-size tuning problems are visible at a glance. The ESS estimator follows Geyer's initial positive sequence method (see references) via FFT.

## Use in mcmc-bench

This package is consumed by the Nextflow benchmarking pipeline [mcmc-bench](https://github.com/thealanjason/mcmc-bench), where it is installed via pip from this repository and compared against `emcee`, `dynesty`, `Slice`, and `SMC` on a surrogate-model calibration task. This demonstrates the package working as an installable dependency in an independent project.

## Development

```bash
pip install -e ".[dev]"
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
HOWTO.md           step-by-step guide
CITATION.cff       citation metadata
```

## References

- C. J. Geyer (1992), *Practical Markov Chain Monte Carlo*, Statistical Science 7(4). https://projecteuclid.org/journals/statistical-science/volume-7/issue-4/Practical-Markov-Chain-Monte-Carlo/10.1214/ss/1177011137.full
- A. Gelman et al., *Bayesian Data Analysis*, 3rd ed., Ch. 11. https://sites.stat.columbia.edu/gelman/book/
- Stan Reference Manual, *Effective Sample Size*. https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html

## Citation

See [CITATION.cff](CITATION.cff). Licensed under the [MIT License](LICENSE).