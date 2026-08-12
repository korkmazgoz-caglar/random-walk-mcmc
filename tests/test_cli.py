"""Tests for the command line interface.

The point is not to cover every combination of settings, but to keep the promise
the CLI makes: bad input is refused clearly, before any sampling happens.
"""

import hashlib
import json

import numpy as np
import pytest
import yaml

from rwmcmc import cli
from rwmcmc.cli import load_config, main

VALID = {
    "target": "banana",
    "x0": [0.0, 0.0],
    "n_samples": 200,
    "step_size": [1.0, 1.0],
    "burn_in": 50,
    "seed": 42,
    "output_dir": "results",
    "save_dashboard": False,
    "save_corner": False,
    "param_names": ["x1", "x2"],
}


def write_config(tmp_path, **changes):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**VALID, **changes}))
    return path


def test_a_valid_configuration_is_accepted(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config["target"] == "banana"
    assert config["n_samples"] == 200


def test_the_file_must_hold_a_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_invalid_yaml_is_reported_clearly(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("target: [")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_config(path)


def test_a_missing_required_setting_is_reported(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in VALID.items() if k != "step_size"}))
    with pytest.raises(ValueError, match="step_size"):
        load_config(path)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"burn_in": 200}, "burn_in"),
        ({"burn_in": -5}, "burn_in"),
        ({"param_names": ["only_one"]}, "param_names"),
        ({"n_samples": 1.5}, "n_samples"),
        ({"save_dashboard": "yes"}, "save_dashboard"),
        ({"output_dir": "   "}, "output_dir"),
        ({"target": ["banana"]}, "target"),
        ({"burnin": 500}, "unknown configuration setting"),
        (
            {
                "target": "gaussian_1d",
                "x0": [0.0],
                "step_size": 1.0,
                "param_names": ["x"],
                "save_corner": True,
            },
            "save_corner",
        ),
    ],
)
def test_impossible_settings_are_refused(tmp_path, changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        load_config(write_config(tmp_path, **changes))


def test_a_bad_configuration_stops_before_sampling(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fail_if_called(*args, **kwargs):
        pytest.fail("sampler must not run for an invalid configuration")

    monkeypatch.setattr("rwmcmc.cli.random_walk_metropolis_hastings", fail_if_called)
    path = write_config(tmp_path, burn_in=500)
    with pytest.raises(SystemExit) as exit_info:
        main([str(path)])
    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "burn_in" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "results").exists()


def test_a_one_sample_run_is_refused_before_sampling(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fail_if_called(*args, **kwargs):
        pytest.fail("sampler must not run when the CLI cannot compute diagnostics")

    monkeypatch.setattr("rwmcmc.cli.random_walk_metropolis_hastings", fail_if_called)
    path = write_config(tmp_path, n_samples=1, burn_in=0)

    with pytest.raises(SystemExit) as exit_info:
        main([str(path)])

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "n_samples must be at least 2 for CLI diagnostics" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "results").exists()


def test_burn_in_must_leave_two_samples_before_sampling(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fail_if_called(*args, **kwargs):
        pytest.fail("sampler must not run when too few samples remain after burn-in")

    monkeypatch.setattr("rwmcmc.cli.random_walk_metropolis_hastings", fail_if_called)
    path = write_config(tmp_path, n_samples=2, burn_in=1)

    with pytest.raises(SystemExit) as exit_info:
        main([str(path)])

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "burn_in must leave at least 2 samples for CLI diagnostics" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "results").exists()


def test_a_diagnostic_error_is_reported_as_a_cli_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fail_diagnostics(*args, **kwargs):
        raise ValueError("diagnostics are undefined for this chain")

    monkeypatch.setattr("rwmcmc.cli.summary", fail_diagnostics)
    path = write_config(tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main([str(path)])

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "diagnostics are undefined for this chain" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "results").exists()


def test_a_sampler_validation_error_is_reported_as_a_cli_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = write_config(tmp_path, step_size=0.0)
    with pytest.raises(SystemExit) as exit_info:
        main([str(path)])
    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "step_size" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "results").exists()


def write_minimal_config(tmp_path):
    """Only the required settings and a seed, so the defaults can be observed."""
    path = tmp_path / "minimal.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "target": VALID["target"],
                "x0": VALID["x0"],
                "n_samples": VALID["n_samples"],
                "step_size": VALID["step_size"],
                "seed": VALID["seed"],
            }
        )
    )
    return path


def read_metadata(run_dir):
    return yaml.safe_load((run_dir / "run_metadata.yaml").read_text())


def digests(run_dir):
    outputs = read_metadata(run_dir)["outputs"]
    return {name: info["sha256"] for name, info in outputs.items()}


def test_unknown_settings_are_listed_alphabetically(tmp_path):
    with pytest.raises(ValueError, match="burnin, stepsize"):
        load_config(write_config(tmp_path, stepsize=2, burnin=1))


def test_configuration_keys_must_be_strings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("1: oops\n" + yaml.safe_dump(VALID))
    with pytest.raises(TypeError, match="must be strings"):
        load_config(path)


def test_an_unknown_setting_stops_before_sampling(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fail_if_called(*args, **kwargs):
        pytest.fail("sampler must not run for an unknown setting")

    monkeypatch.setattr("rwmcmc.cli.random_walk_metropolis_hastings", fail_if_called)
    path = write_config(tmp_path, burnin=500)
    with pytest.raises(SystemExit) as exit_info:
        main([str(path)])
    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "burnin" in captured.err
    assert not (tmp_path / "results").exists()


def test_disabled_plots_remove_stale_outputs_without_touching_other_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    main([str(write_config(tmp_path, save_dashboard=True, save_corner=True))])

    results = tmp_path / "results"
    dashboard_path = results / "dashboard.png"
    corner_path = results / "corner.png"
    unrelated_path = results / "keep-me.txt"

    assert dashboard_path.exists()
    assert corner_path.exists()
    unrelated_path.write_text("not managed by rwmcmc")

    main([str(write_config(tmp_path, save_dashboard=False, save_corner=False))])

    assert not dashboard_path.exists()
    assert not corner_path.exists()
    assert unrelated_path.read_text() == "not managed by rwmcmc"


def test_the_metadata_describes_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main([str(write_config(tmp_path, save_dashboard=False, save_corner=False))])
    metadata = read_metadata(tmp_path / "results")

    assert metadata["schema_version"] == 1
    assert metadata["created_at_utc"].endswith("+00:00")
    assert metadata["input"]["config_sha256"]
    assert metadata["source"]["rwmcmc_version"]

    assert metadata["config"]["requested"]["target"] == "banana"
    assert set(metadata["config"]["effective"]) == {
        "target",
        "x0",
        "n_samples",
        "step_size",
        "burn_in",
        "seed",
        "output_dir",
        "save_dashboard",
        "save_corner",
        "param_names",
    }
    assert metadata["config"]["effective"]["seed"] == 42

    expected = type(np.random.default_rng(0).bit_generator).__name__
    assert metadata["random_number_generator"]["seed"] == 42
    assert metadata["random_number_generator"]["bit_generator"] == expected

    for key in (
        "python_version",
        "python_implementation",
        "numpy_version",
        "matplotlib_version",
        "pyyaml_version",
        "operating_system",
        "os_release",
        "architecture",
    ):
        assert metadata["environment"][key]


def test_the_metadata_fingerprints_the_output_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main([str(write_config(tmp_path, save_dashboard=False, save_corner=False))])
    results = tmp_path / "results"
    outputs = read_metadata(results)["outputs"]

    assert set(outputs) == {"samples.npy", "accepted.npy"}
    assert outputs["samples.npy"]["shape"] == [200, 2]
    assert outputs["samples.npy"]["dtype"] == "float64"
    assert outputs["accepted.npy"]["shape"] == [200]
    assert outputs["accepted.npy"]["dtype"] == "bool"

    for name, info in outputs.items():
        written = results / name
        assert info["sha256"] == hashlib.sha256(written.read_bytes()).hexdigest()
        assert info["size_bytes"] == written.stat().st_size


def test_the_same_seed_gives_the_same_files_in_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("run_a", "run_b"):
        config = write_config(tmp_path, output_dir=name, save_dashboard=False, save_corner=False)
        main([str(config)])
    assert digests(tmp_path / "run_a") == digests(tmp_path / "run_b")


def test_a_drawn_seed_can_be_replayed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(
        [
            str(
                write_config(
                    tmp_path, seed=None, output_dir="drawn", save_dashboard=False, save_corner=False
                )
            )
        ]
    )
    drawn = read_metadata(tmp_path / "drawn")
    resolved = drawn["random_number_generator"]["seed"]

    assert drawn["config"]["requested"]["seed"] is None
    assert drawn["config"]["effective"]["seed"] == resolved

    main(
        [
            str(
                write_config(
                    tmp_path,
                    seed=resolved,
                    output_dir="replay",
                    save_dashboard=False,
                    save_corner=False,
                )
            )
        ]
    )
    assert digests(tmp_path / "replay") == digests(tmp_path / "drawn")


def test_the_metadata_is_written_even_without_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def no_git(*args, **kwargs):
        raise OSError("git is not installed here")

    monkeypatch.setattr("rwmcmc.cli.subprocess.run", no_git)
    main([str(write_config(tmp_path, save_dashboard=False, save_corner=False))])
    source = read_metadata(tmp_path / "results")["source"]

    assert source["rwmcmc_version"]
    assert source["git_commit"] is None
    assert source["git_dirty"] is None


def test_the_effective_configuration_spells_out_the_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main([str(write_minimal_config(tmp_path))])
    metadata = read_metadata(tmp_path / "results")

    assert "burn_in" not in metadata["config"]["requested"]

    effective = metadata["config"]["effective"]
    assert effective["burn_in"] == 0
    assert effective["output_dir"] == "results"
    assert effective["save_dashboard"] is True
    assert effective["save_corner"] is False
    assert effective["param_names"] is None


def test_the_metadata_records_the_digest_of_the_configuration_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = write_config(tmp_path, save_dashboard=False, save_corner=False)
    main([str(config)])
    recorded = read_metadata(tmp_path / "results")["input"]["config_sha256"]

    assert recorded == hashlib.sha256(config.read_bytes()).hexdigest()


def test_a_pip_installed_revision_is_preferred_to_a_surrounding_checkout(monkeypatch):
    # A virtual environment can live inside an unrelated Git repository. The
    # commit recorded by pip identifies rwmcmc; the surrounding repository does not.
    def fail_if_git_is_called(*args, **kwargs):
        pytest.fail("Git must not be consulted when direct_url.json records the commit")

    class FakeDistribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/korkmazgoz-caglar/random-walk-mcmc",
                    "vcs_info": {"vcs": "git", "commit_id": "0123456789abcdef"},
                }
            )

    monkeypatch.setattr(cli.subprocess, "run", fail_if_git_is_called)
    monkeypatch.setattr(cli.importlib.metadata, "distribution", lambda name: FakeDistribution())

    source = cli._source_info()

    assert source["rwmcmc_version"]
    assert source["git_commit"] == "0123456789abcdef"
    assert source["git_dirty"] is None


def test_an_unrelated_parent_checkout_is_not_reported_as_rwmcmc_source(tmp_path, monkeypatch):
    checkout_root = tmp_path / "unrelated-project"
    package_dir = checkout_root / ".venv" / "lib" / "python" / "site-packages" / "rwmcmc"
    package_dir.mkdir(parents=True)
    package_file = package_dir / "__init__.py"
    package_file.write_text("")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type("Completed", (), {"stdout": f"{checkout_root}\n"})()

    monkeypatch.setattr(cli, "_installed_commit", lambda: None)
    monkeypatch.setattr(cli.rwmcmc, "__file__", str(package_file))
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    source = cli._source_info()

    assert source["git_commit"] is None
    assert source["git_dirty"] is None
    assert len(calls) == 1
    assert calls[0][-2:] == ["rev-parse", "--show-toplevel"]


def test_a_real_source_checkout_records_its_revision_and_dirty_state(tmp_path, monkeypatch):
    checkout_root = tmp_path / "random-walk-mcmc"
    package_dir = checkout_root / "src" / "rwmcmc"
    package_dir.mkdir(parents=True)
    package_file = package_dir / "__init__.py"
    package_file.write_text("")

    outputs = {
        ("rev-parse", "--show-toplevel"): f"{checkout_root}\n",
        ("rev-parse", "HEAD"): "fedcba9876543210\n",
        ("status", "--porcelain"): " M src/rwmcmc/cli.py\n",
    }

    def fake_run(command, **kwargs):
        key = tuple(command[3:])
        return type("Completed", (), {"stdout": outputs[key]})()

    monkeypatch.setattr(cli, "_installed_commit", lambda: None)
    monkeypatch.setattr(cli.rwmcmc, "__file__", str(package_file))
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    source = cli._source_info()

    assert source["git_commit"] == "fedcba9876543210"
    assert source["git_dirty"] is True


def test_the_provenance_is_taken_before_any_output_exists(tmp_path, monkeypatch):
    # the working tree must not be reported as dirty because of the files this
    # very run is about to write
    monkeypatch.chdir(tmp_path)
    seen = {}
    real_source_info = cli._source_info

    def recording_source_info():
        seen["results_existed"] = (tmp_path / "results").exists()
        return real_source_info()

    monkeypatch.setattr(cli, "_source_info", recording_source_info)
    main([str(write_config(tmp_path, save_dashboard=False, save_corner=False))])

    assert seen["results_existed"] is False
    assert (tmp_path / "results").exists()
