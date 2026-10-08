"""The config loader refuses what it cannot honour (review fix E8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.config import (  # noqa: E402
    EngineConfig,
    OptimizerSpec,
    load_config,
    save_config,
)
from optimization_engine.optimizers import ConfigurationError  # noqa: E402
from optimization_engine.stress import Shock, StressError  # noqa: E402


def test_a_misspelt_key_is_refused_rather_than_ignored():
    # `max_tracking_eror` loaded cleanly and simply did not constrain anything.
    with pytest.raises(ConfigurationError, match="max_tracking_eror"):
        EngineConfig.from_dict({"max_tracking_eror": 0.03, "optimizer": "min_variance"})


def test_an_unknown_optimizer_key_is_a_configuration_error_not_a_type_error():
    with pytest.raises(ConfigurationError, match="Unknown optimizer key"):
        EngineConfig.from_dict({"optimizer": {"name": "cvar", "cvar_alph": 0.05}})


def test_a_config_round_trips_through_its_own_dict():
    config = load_config(ROOT / "config" / "example_multi_asset.yaml")
    again = EngineConfig.from_dict(config.to_dict())
    assert again.to_dict() == config.to_dict()


def test_the_shipped_indices_config_declares_its_currencies():
    # It used to spell the block `asset_currency`, which nothing read, so the
    # conversion its own comment promised never happened.
    config = load_config(ROOT / "config" / "indices.yaml")
    assert config.currencies["DAX"] == "EUR"
    assert config.currencies["NIKKEI225"] == "JPY"


def test_strict_mandate_round_trips_through_a_saved_config(tmp_path):
    """A new field is three edits, and the third one is the easy one to forget.

    ``strict_mandate`` has to reach the dataclass, ``to_dict``, ``from_dict``
    *and* ``_CONFIG_KEYS``. Without the last, writing a config out and reading
    it back raises ``ConfigurationError`` on the key the writer just produced —
    which is the specific failure this pins.
    """
    assert EngineConfig().strict_mandate is False

    config = EngineConfig(
        expected_returns={"A": 0.05, "B": 0.04},
        strict_mandate=True,
        optimizer=OptimizerSpec(name="min_variance"),
    )
    path = tmp_path / "mandate.yaml"
    save_config(config, path)
    assert "strict_mandate: true" in path.read_text(encoding="utf-8")

    again = load_config(path)
    assert again.strict_mandate is True
    assert again.to_dict() == config.to_dict()


def test_strict_mandate_is_a_key_the_loader_knows():
    # It would otherwise land in the "unknown config key" refusal, which is a
    # much more confusing message than a field that quietly did nothing.
    assert EngineConfig.from_dict({"strict_mandate": True}).strict_mandate is True
    assert EngineConfig.from_dict({}).strict_mandate is False


# ---------------------------------------------------------------------------
# Stress scenarios on the config
# ---------------------------------------------------------------------------


def test_stress_scenarios_round_trip_through_a_saved_config(tmp_path):
    """The same three edits as ``strict_mandate``, plus one this field has alone.

    ``to_dict`` has to emit *plain mappings*, not ``Shock`` objects, because
    ``save_config`` runs the result through ``yaml.safe_dump`` — as do
    ``config_signature`` (through ``json.dumps``) and the app's config panel.
    A dataclass reaching any of those raises a representer error, which is why
    this test writes the file rather than only comparing dicts.
    """
    assert EngineConfig().stress == ()

    config = EngineConfig(
        expected_returns={"A": 0.05, "B": 0.04},
        optimizer=OptimizerSpec(name="min_variance"),
        stress=[
            {
                "name": "risk_off",
                "returns": {"A": -0.20, "B": -0.05},
                "covariance_scale": 2.0,
                "notes": "a 2008-shaped day",
            }
        ],
    )
    # __post_init__ normalizes the raw mappings, so a config built in memory
    # and one loaded from a file behave identically.
    assert isinstance(config.stress, tuple)
    assert isinstance(config.stress[0], Shock)
    assert config.stress[0].name == "risk_off"

    path = tmp_path / "stress.yaml"
    save_config(config, path)
    text = path.read_text(encoding="utf-8")
    assert "risk_off" in text
    assert "!!python" not in text, "to_dict must stay YAML-serializable"

    again = load_config(path)
    assert [s.name for s in again.stress] == ["risk_off"]
    assert again.stress[0].returns == {"A": -0.20, "B": -0.05}
    assert again.stress[0].covariance_scale == 2.0
    assert again.to_dict() == config.to_dict()


def test_a_configs_signature_survives_carrying_shocks():
    """``config_signature`` is ``json.dumps`` over ``to_dict``, and the app
    caches on it. A non-serializable field there breaks every cached run."""
    from optimization_engine.presets import config_signature

    config = EngineConfig(
        stress=[{"name": "gap", "returns": {"A": -0.1}}],
    )
    assert "gap" in config_signature(config)


def test_stress_is_a_key_the_loader_knows_and_validates():
    assert EngineConfig.from_dict({}).stress == ()
    loaded = EngineConfig.from_dict(
        {"stress": [{"name": "gap", "returns": {"A": -0.1}}]}
    )
    assert loaded.stress[0].name == "gap"
    # And a malformed scenario is refused where it was written, not at the
    # solve — the same posture as an unknown config key.
    with pytest.raises(StressError, match="retruns"):
        EngineConfig.from_dict({"stress": [{"name": "gap", "retruns": {"A": -0.1}}]})
    with pytest.raises(StressError, match="Duplicate"):
        EngineConfig(
            stress=[
                {"name": "gap", "returns": {"A": -0.1}},
                {"name": "gap", "returns": {"A": -0.2}},
            ]
        )


def test_an_unsupported_extension_is_refused_before_the_file_is_read(
    tmp_path, monkeypatch
):
    """The extension is checked first, so a key file is never read into memory.

    ``load_config`` read the whole file and only then looked at the suffix:
    any file reachable by path was read in full — a large one at the cost of
    memory — before being refused.
    """
    key = tmp_path / "id_rsa"
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    reads = []
    original = Path.read_text

    def spy(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    with pytest.raises(ValueError, match="Unsupported config extension"):
        load_config(key)
    assert reads == []
