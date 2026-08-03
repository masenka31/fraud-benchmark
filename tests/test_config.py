import dataclasses

import pandas as pd
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


def test_default_config_has_delay_settings():
    config = load_config()
    assert config.delay.median_days == 7.0
    assert config.delay.sigma == 1.0
    assert config.delay.seed == 0


def test_default_campaign_gap_is_one_day():
    assert load_config().campaign_gap_for("paysim") == pd.Timedelta(days=1)


def test_amaretto_overrides_the_campaign_gap_to_one_hour():
    """Measured: Amaretto's anomalies have a 0.7-minute median inter-arrival.

    At the 1-day default it yields 490 episodes with a 3,870-row maximum; at
    1 hour, 1,852 episodes with a median of 8.
    """
    assert load_config().campaign_gap_for("amaretto") == pd.Timedelta(hours=1)


def test_delay_settings_can_be_overridden(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "delay:\n"
        "  median_days: 30.0\n"
        "  sigma: 1.5\n"
        "  seed: 99\n"
        "  max_delay_days: 180.0\n"
        "campaign:\n"
        "  gap: 2D\n"
    )
    config = load_config(path)
    assert config.delay.median_days == 30.0
    assert config.delay.max_delay_days == 180.0
    assert config.campaign_gap_for("paysim") == pd.Timedelta(days=2)


def test_a_per_dataset_gap_override_wins(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "campaign:\n"
        "  gap: 2D\n"
        "datasets:\n"
        "  banksim:\n"
        "    campaign_gap: 30min\n"
    )
    config = load_config(path)
    assert config.campaign_gap_for("banksim") == pd.Timedelta(minutes=30)
    assert config.campaign_gap_for("paysim") == pd.Timedelta(days=2)


def test_invalid_delay_settings_raise_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("delay:\n  median_days: -1.0\n")
    with pytest.raises(ConfigError, match="median_days"):
        load_config(path)


def test_an_unparseable_gap_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("campaign:\n  gap: not-a-duration\n")
    with pytest.raises(ConfigError, match="gap"):
        load_config(path)


def test_a_negative_gap_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("campaign:\n  gap: -3D\n")
    with pytest.raises(ConfigError, match="negative"):
        load_config(path)


def test_a_dataset_without_an_override_gets_the_global_delay():
    config = load_config()
    assert config.delay_for("banksim") == config.delay


def test_a_delay_override_merges_over_the_global_block(tmp_path):
    """Only the named keys move; the rest inherit."""
    path = tmp_path / "custom.yaml"
    path.write_text(
        "delay:\n"
        "  median_days: 7.0\n"
        "  sigma: 1.0\n"
        "  seed: 5\n"
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      median_days: 2.0\n"
    )
    config = load_config(path)
    delay = config.delay_for("banksim")
    assert delay.median_days == 2.0
    assert delay.sigma == 1.0
    assert delay.seed == 5


def test_a_delay_override_does_not_leak_to_other_datasets(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      median_days: 2.0\n"
    )
    config = load_config(path)
    assert config.delay_for("banksim").median_days == 2.0
    assert config.delay_for("sparkov").median_days == config.delay.median_days


def test_a_delay_override_can_set_max_delay_days(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      max_delay_days: 365\n"
    )
    assert load_config(path).delay_for("banksim").max_delay_days == 365.0


def test_an_unknown_delay_key_is_rejected(tmp_path):
    """A typo must fail loudly, not silently inherit the global value."""
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      median_day: 2.0\n"
    )
    with pytest.raises(ConfigError, match="median_day"):
        load_config(path)


def test_an_invalid_delay_override_is_rejected(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      median_days: -1.0\n"
    )
    with pytest.raises(ConfigError, match="median_days"):
        load_config(path)


def test_a_non_mapping_delay_override_is_rejected(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay: 7\n"
    )
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_delay_overrides_are_validated_at_load_time(tmp_path):
    """Errors must surface on load_config, not on the later delay_for call.

    A bad override that only raises when a dataset is prepared would let
    `prepare --all` fail halfway through, after writing other datasets.
    """
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      sigma: 0\n"
    )
    with pytest.raises(ConfigError, match="sigma"):
        load_config(path)


def test_paysim_overrides_the_delay_to_one_day():
    """PaySim's whole span is 30 simulated days; a 7-day median censors 53% of
    its train labels. sigma and seed still inherit."""
    config = load_config()
    delay = config.delay_for("paysim")
    assert delay.median_days == 1.0
    assert delay.sigma == config.delay.sigma
    assert delay.seed == config.delay.seed


def test_saml_d_overrides_the_delay_to_the_aml_regime():
    """AML reporting runs weeks behind the transaction, not days: detection lags,
    and the SAR clock starts only at detection. sigma and seed still inherit."""
    config = load_config()
    delay = config.delay_for("saml_d")
    assert delay.median_days == 30.0
    assert delay.sigma == config.delay.sigma
    assert delay.seed == config.delay.seed


def test_amaretto_keeps_the_card_fraud_delay_despite_being_aml():
    """Its 83-day span cannot carry a realistic 30-day AML median — that censors
    53% of train frauds. The mismatch is documented, not repaired."""
    config = load_config()
    assert config.delay_for("amaretto").median_days == config.delay.median_days


def test_saml_d_overrides_the_delay_to_the_aml_regime():
    """AML reporting runs weeks behind the transaction, not days: detection lags,
    and the SAR clock starts only at detection. sigma and seed still inherit."""
    config = load_config()
    delay = config.delay_for("saml_d")
    assert delay.median_days == 30.0
    assert delay.sigma == config.delay.sigma
    assert delay.seed == config.delay.seed


def test_amaretto_keeps_the_card_fraud_delay_despite_being_aml():
    """Its 83-day span cannot carry a realistic 30-day AML median — that censors
    53% of train frauds. The mismatch is documented, not repaired."""
    config = load_config()
    assert config.delay_for("amaretto").median_days == config.delay.median_days






def test_sparkov_slow_overrides_the_delay_to_a_harsher_regime():
    """At the card-fraud default Sparkov's 487-day train window censors only
    2.1% of train labels, so the delay barely registers. sigma and seed inherit."""
    config = load_config()
    delay = config.delay_for("sparkov_slow")
    assert delay.median_days == 15.0
    assert delay.sigma == 1.665
    assert delay.max_delay_days == 365
    assert delay.seed == config.delay.seed


def test_sparkov_slow_inherits_sparkovs_val_fraction():
    """It must split identically to sparkov, or the rows would not correspond."""
    config = load_config()
    assert (
        config.for_dataset("sparkov_slow")["val_fraction"]
        == config.for_dataset("sparkov")["val_fraction"]
    )


def test_an_unknown_dataset_option_is_rejected(tmp_path):
    """Same argument as the delay block, one level up: a typo must not leave the
    dataset silently on its default while the config claims otherwise."""
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  sparkov:\n"
        "    val_fractio: 0.2\n"
    )
    with pytest.raises(ConfigError, match="val_fractio"):
        load_config(path)


def test_the_error_lists_the_valid_dataset_options(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("datasets:\n  ibm_ccf:\n    entity_ke: card\n")
    with pytest.raises(ConfigError, match="entity_key"):
        load_config(path)


def test_a_non_mapping_dataset_block_is_rejected(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("datasets:\n  banksim: 2023-01-01\n")
    with pytest.raises(ConfigError, match="mapping of options"):
        load_config(path)


def test_the_shipped_config_uses_only_known_options():
    """The tripwire is worthless if the default config cannot pass it."""
    from fraud_benchmark.config import DATASET_OPTIONS

    for name, options in load_config().datasets.items():
        assert not set(options) - DATASET_OPTIONS, name


def test_a_bad_campaign_gap_override_is_caught_at_load_time(tmp_path):
    """Not when its dataset is prepared -- `prepare --all` would die halfway."""
    path = tmp_path / "custom.yaml"
    path.write_text("datasets:\n  banksim:\n    campaign_gap: 'not a duration'\n")
    with pytest.raises(ConfigError, match="duration"):
        load_config(path)
