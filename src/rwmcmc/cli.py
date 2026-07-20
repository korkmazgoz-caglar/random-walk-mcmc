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
from pathlib import Path

import numpy as np
import yaml

import rwmcmc
from .samplers import random_walk_metropolis_hastings
from .targets import gaussian_1d_log_pdf, banana_log_pdf
from .diagnostics import summary, dashboard

TARGETS = {
    "gaussian_1d": gaussian_1d_log_pdf,
    "banana": banana_log_pdf,
}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if config.get("target") not in TARGETS:
        raise ValueError(
            f"Unknown target {config.get('target')!r}; available are: {sorted(TARGETS)}"
        )
    return config


def resolve_seed(seed) -> int:
    """Return a usable integer seed; draw (and report) one if seed is None."""
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**32)
    return int(seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run random-walk MCMC from a YAML config.")
    parser.add_argument("config", help="path to the YAML configuration file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = resolve_seed(config.get("seed"))
    rng = np.random.default_rng(seed)

    samples, accepted = random_walk_metropolis_hastings(
        TARGETS[config["target"]],
        x0=config["x0"],
        n_samples=int(config["n_samples"]),
        step_size=config["step_size"],
        rng=rng,
    )

    burn_in = int(config.get("burn_in", 0))
    stats = summary(samples, accepted, burn_in=burn_in)

    out = Path(config.get("output_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "samples.npy", samples)
    np.save(out / "accepted.npy", accepted)

    if config.get("save_dashboard", True):
        fig = dashboard(
            samples, accepted, burn_in=burn_in,
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