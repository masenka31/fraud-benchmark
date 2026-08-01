import dataclasses

import pytest

from fraud_benchmark.config import Config, ConfigError, load_config


def test_load_default_config():
    config = load_config()
    assert config.split_ratios == (0.8, 0.1, 0.1)
    assert config.raw_dir.name == "raw"
    assert config.processed_dir.name == "processed"


def test_default_config_has_paysim_start_date():
    config = load_config()
    assert config.for_dataset("paysim")["start_date"] == "2023-01-01"


def test_for_dataset_returns_empty_dict_for_unknown_dataset():
    assert load_config().for_dataset("nonexistent") == {}


def test_user_config_overrides_defaults(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "split:\n"
        "  ratios: [0.6, 0.2, 0.2]\n"
        "datasets:\n"
        "  paysim:\n"
        "    start_date: '2020-05-01'\n"
    )
    config = load_config(path)
    assert config.split_ratios == (0.6, 0.2, 0.2)
    assert config.for_dataset("paysim")["start_date"] == "2020-05-01"
    # Unspecified keys still come from the defaults.
    assert config.raw_dir.name == "raw"


def test_ratios_must_sum_to_one(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("split:\n  ratios: [0.5, 0.2, 0.2]\n")
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(path)


def test_ratios_must_have_three_entries(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("split:\n  ratios: [0.8, 0.2]\n")
    with pytest.raises(ConfigError, match="three"):
        load_config(path)


def test_config_is_frozen():
    config = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.split_ratios = (0.5, 0.25, 0.25)


def test_missing_config_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("paths: {raw: data/raw\nsplit: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_null_paths_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("paths:\n")
    with pytest.raises(ConfigError, match="paths"):
        load_config(path)


def test_missing_path_key_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("paths:\n  raw: null\n")
    with pytest.raises(ConfigError, match="paths.raw"):
        load_config(path)
