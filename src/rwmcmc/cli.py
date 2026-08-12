"""Command line interface: run a sampler from a YAML config file.

Usage:
    rwmcmc-run path/to/config.yaml
Config is currently at:
    examples/config.yaml

Writes into the configured output directory:
    samples.npy        the chain, shape (n_samples, d)
    accepted.npy       boolean acceptance array, accepted[i] = True if accepted
    dashboard.png      visual diagnostics (optional)
    run_metadata.yaml  full record of the run: requested and effective config,
                       seed and bit generator, source revision, environment, and
                       SHA-256 digests of the .npy outputs
"""

import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import numbers
import platform
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import yaml

import rwmcmc

from .diagnostics import corner, dashboard, summary
from .samplers import _as_1d_float, _as_positive_int, random_walk_metropolis_hastings
from .targets import banana_log_pdf, gaussian_1d_log_pdf

TARGETS = {
    "gaussian_1d": gaussian_1d_log_pdf,
    "banana": banana_log_pdf,
}

REQUIRED_SETTINGS = ("target", "x0", "n_samples", "step_size")

KNOWN_SETTINGS = REQUIRED_SETTINGS + (
    "burn_in",
    "seed",
    "output_dir",
    "save_dashboard",
    "save_corner",
    "param_names",
)


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

    non_string = sorted(repr(key) for key in config if not isinstance(key, str))
    if non_string:
        raise TypeError(f"configuration keys must be strings, got: {', '.join(non_string)}")

    # Checked before the required settings, so that a misspelled name is reported
    # as the typo it is rather than as a missing setting. A name that slipped
    # through would fall back to its default, and the metadata would then record
    # a setting that was never applied.
    unknown = sorted(key for key in config if key not in KNOWN_SETTINGS)
    if unknown:
        raise ValueError(f"unknown configuration setting(s): {', '.join(unknown)}")

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


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_info(path: Path, array: np.ndarray) -> dict:
    """Describe a written .npy file well enough to check a replay against it.

    The digest is taken from the file on disk, so it certifies that the two
    files are byte for byte the same. That is a stronger claim than saying the
    numbers agree, and a weaker one than saying two platforms would compute the
    same numbers.
    """
    return {
        "sha256": _sha256(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size_bytes": path.stat().st_size,
    }


def _installed_commit() -> str | None:
    """Return the revision pip recorded when the package came from a repository.

    Installing from a git URL writes a direct_url.json next to the package
    metadata, and it carries the commit that was checked out. An install from a
    plain directory has no revision to report.
    """
    try:
        raw = importlib.metadata.distribution("rwmcmc").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None

    try:
        direct_url = json.loads(raw)
    except json.JSONDecodeError:
        return None

    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None

    commit = vcs_info.get("commit_id")
    return commit if isinstance(commit, str) and commit else None


def _source_info() -> dict:
    """Identify the source that produced the run.

    A VCS installation is identified first from the direct_url.json written by
    pip. Otherwise Git information is used only when the imported package is
    actually the src/rwmcmc directory of that checkout. This prevents a regular
    installation inside another project's virtual environment from recording
    the surrounding project's revision as the rwmcmc revision.
    """
    info = {"rwmcmc_version": rwmcmc.__version__, "git_commit": None, "git_dirty": None}

    installed_commit = _installed_commit()
    if installed_commit is not None:
        info["git_commit"] = installed_commit
        return info

    package_dir = Path(rwmcmc.__file__).resolve().parent
    try:
        root = subprocess.run(
            ["git", "-C", str(package_dir), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        checkout_root = Path(root).resolve()

        if package_dir != (checkout_root / "src" / "rwmcmc").resolve():
            return info

        commit = subprocess.run(
            ["git", "-C", str(checkout_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(checkout_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return info

    info["git_commit"] = commit
    info["git_dirty"] = bool(status)
    return info


def _environment_info() -> dict:
    """Describe the software and the platform.

    Nothing that identifies the machine or the user is recorded: it would not
    help a replay and does not belong in a file meant to be shared.
    """
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "pyyaml_version": yaml.__version__,
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
    }


def build_effective_config(config: dict, resolved_seed: int) -> dict:
    """Return the settings actually used, with every default written out.

    The requested configuration only holds what the file said. Writing the
    effective one next to it keeps a run replayable even if a default changes in
    a later version of the package.
    """
    effective = {
        "target": config["target"],
        "x0": config["x0"],
        "n_samples": config["n_samples"],
        "step_size": config["step_size"],
        "burn_in": config.get("burn_in", 0),
        "seed": resolved_seed,
        "output_dir": str(config.get("output_dir", "results")),
        "save_dashboard": config.get("save_dashboard", True),
        "save_corner": config.get("save_corner", False),
        "param_names": config.get("param_names"),
    }
    # An independent copy, so that the effective settings are a snapshot rather
    # than a reference into the requested ones. It also stops PyYAML from writing
    # the shared lists as anchors and aliases, which is hard to read.
    return copy.deepcopy(effective)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run random-walk MCMC from a YAML config.")
    parser.add_argument("config", help="path to the YAML configuration file")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    # Both are taken before anything is written. An output directory inside the
    # repository would otherwise make the working tree look dirty because of the
    # files this very run produced, and the configuration file could still be
    # edited while the sampling is going on.
    config_sha256 = _sha256(Path(args.config))
    source = _source_info()

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

    # A run directory may be reused. Remove only optional outputs that the
    # current configuration disables, so an image from an earlier run cannot
    # be mistaken for an output of the current one.
    optional_outputs = {
        "dashboard.png": config.get("save_dashboard", True),
        "corner.png": config.get("save_corner", False),
    }
    for name, enabled in optional_outputs.items():
        if not enabled:
            (out / name).unlink(missing_ok=True)

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
        fig = corner(samples, burn_in=burn_in, param_names=config.get("param_names"))
        fig.savefig(out / "corner.png", dpi=120)

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input": {"config_sha256": config_sha256},
        "source": source,
        "config": {
            "requested": config,
            "effective": build_effective_config(config, seed),
        },
        "random_number_generator": {
            "seed": seed,
            "bit_generator": type(rng.bit_generator).__name__,
        },
        "results": {
            "acceptance_rate": float(stats["acceptance_rate"]),
            "mean": [float(v) for v in stats["mean"]],
            "std": [float(v) for v in stats["std"]],
            "ess": [float(v) for v in stats["ess"]],
        },
        "environment": _environment_info(),
        "outputs": {
            "samples.npy": _output_info(out / "samples.npy", samples),
            "accepted.npy": _output_info(out / "accepted.npy", accepted),
        },
    }
    with open(out / "run_metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    print(f"run complete: seed={seed} acceptance={stats['acceptance_rate']:.2%}")
    print(f"outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
