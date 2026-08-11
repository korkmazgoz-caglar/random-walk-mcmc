"""Command line interface: run a sampler from a YAML config file.

Usage:
    rwmcmc-run path/to/config.yaml
Config is currently at:
    examples/config.yaml

Writes into the configured output directory:
    samples.npy        the chain, shape (n_samples, d)
    accepted.npy       boolean acceptance array, accepted[i] = True if accepted
    dashboard.png      visual diagnostics (optional)
    run_metadata.yaml  full record of the run: config used, resolved seed,
                       package versions, timestamp -> makes the run reproducible
"""

import argparse
import datetime
import numbers
from pathlib import Path

import numpy as np
import yaml

import rwmcmc

from .diagnostics import dashboard, summary
from .samplers import _as_1d_float, _as_positive_int, random_walk_metropolis_hastings
from .targets import banana_log_pdf, gaussian_1d_log_pdf

TARGETS = {
    "gaussian_1d": gaussian_1d_log_pdf,
    "banana": banana_log_pdf,
}

REQUIRED_SETTINGS = ("target", "x0", "n_samples", "step_size")


def load_config(path: str | Path) -> dict:
    """Read the configuration file and check it before anything else runs.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        The parsed settings after CLI-specific validation.

    Raises
    ------
    OSError
        If the file cannot be read.
    TypeError
        If a setting has the wrong type.
    ValueError
        If the file is not valid YAML, does not hold a mapping, a required
        setting is missing, the target is unknown, or a setting has an
        impossible value.
    """
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping of settings, got {type(config).__name__}")

    missing = [key for key in REQUIRED_SETTINGS if key not in config]
    if missing:
        raise ValueError(f"missing required setting(s): {', '.join(missing)}")

    target = config["target"]
    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; available are: {sorted(TARGETS)}")

    validate_run_settings(config)
    return config


def validate_run_settings(config: dict) -> None:
    """Check the settings the CLI itself acts on, before any sampling starts.

    x0 and n_samples are checked with the sampler's own rules, so that the two
    places cannot drift apart. step_size is left to the sampler entirely.

    Raises
    ------
    TypeError
        If a setting has the wrong type.
    ValueError
        If a setting has an impossible value.
    """
    n_samples = _as_positive_int(config["n_samples"], "n_samples")
    dimension = _as_1d_float(config["x0"], "x0").size

    seed = config.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError(f"seed must be an integer or null, got {type(seed).__name__}")
        if seed < 0:
            raise ValueError(f"seed must not be negative, got {seed}")

    burn_in = config.get("burn_in", 0)
    if isinstance(burn_in, bool) or not isinstance(burn_in, numbers.Integral):
        raise TypeError(f"burn_in must be an integer, got {type(burn_in).__name__}")
    if burn_in < 0:
        raise ValueError(f"burn_in must not be negative, got {burn_in}")
    if burn_in >= n_samples:
        raise ValueError(f"burn_in must be smaller than n_samples ({n_samples}), got {burn_in}")

    for flag in ("save_dashboard", "save_corner"):
        if flag in config and not isinstance(config[flag], bool):
            raise TypeError(f"{flag} must be true or false, got {type(config[flag]).__name__}")

    if config.get("save_corner", False) and dimension < 2:
        raise ValueError("save_corner requires an x0 with at least two dimensions")

    param_names = config.get("param_names")
    if param_names is not None:
        if not isinstance(param_names, list) or not all(
            isinstance(name, str) for name in param_names
        ):
            raise TypeError("param_names must be a list of strings")
        if len(param_names) != dimension:
            raise ValueError(
                f"param_names must have one name per dimension of x0 ({dimension}), "
                f"got {len(param_names)}"
            )

    output_dir = config.get("output_dir", "results")
    if not isinstance(output_dir, (str, Path)):
        raise TypeError(
            f"output_dir must be a string or pathlib.Path, got {type(output_dir).__name__}"
        )
    if not str(output_dir).strip():
        raise ValueError("output_dir must not be empty")


def resolve_seed(seed) -> int:
    """Return a usable integer seed; draw (and report) one if seed is None."""
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**32)
    return int(seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run random-walk MCMC from a YAML config.")
    parser.add_argument("config", help="path to the YAML configuration file")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    seed = resolve_seed(config.get("seed"))
    rng = np.random.default_rng(seed)

    try:
        samples, accepted = random_walk_metropolis_hastings(
            TARGETS[config["target"]],
            x0=config["x0"],
            n_samples=config["n_samples"],
            step_size=config["step_size"],
            rng=rng,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    burn_in = config.get("burn_in", 0)
    stats = summary(samples, accepted, burn_in=burn_in)

    out = Path(config.get("output_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "samples.npy", samples)
    np.save(out / "accepted.npy", accepted)

    if config.get("save_dashboard", True):
        fig = dashboard(
            samples,
            accepted,
            burn_in=burn_in,
            param_names=config.get("param_names"),
        )
        fig.savefig(out / "dashboard.png", dpi=120)

    if config.get("save_corner", False) and samples.ndim == 2 and samples.shape[1] >= 2:
        from .diagnostics import corner

        fig = corner(samples, burn_in=burn_in, param_names=config.get("param_names"))
        fig.savefig(out / "corner.png", dpi=120)

    metadata = {
        "config": config,
        "resolved_seed": seed,  # actual seed used, even when config had null
        "results": {
            "acceptance_rate": float(stats["acceptance_rate"]),
            "mean": [float(v) for v in stats["mean"]],
            "std": [float(v) for v in stats["std"]],
            "ess": [float(v) for v in stats["ess"]],
        },
        "environment": {
            "rwmcmc_version": rwmcmc.__version__,
            "numpy_version": np.__version__,
        },
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
    }
    with open(out / "run_metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    print(f"run complete: seed={seed} acceptance={stats['acceptance_rate']:.2%}")
    print(f"outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
