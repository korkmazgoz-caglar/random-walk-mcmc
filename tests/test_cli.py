"""Tests for the command line interface.

The point is not to cover every combination of settings, but to keep the promise
the CLI makes: bad input is refused clearly, before any sampling happens.
"""

import pytest
import yaml

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
