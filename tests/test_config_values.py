"""The value half of review item E8: a config value that cannot mean anything is refused.

The key half — a misspelt key, an unknown optimizer field — is pinned in
``test_config.py``. This module pins the values: a number that makes the
covariance vanish, a decay outside the open unit interval, a boolean that
arrived as text, a benchmark that arrived as a mapping, and a mandate under
``strict_mandate`` that names an asset the panel does not hold. Each of these
used to load, and most of them used to solve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.benchmark import BenchmarkSpec  # noqa: E402
from optimization_engine.config import EngineConfig, OptimizerSpec  # noqa: E402
from optimization_engine.data.loader import prices_to_returns, sample_dataset  # noqa: E402
from optimization_engine.engine import run_engine  # noqa: E402
from optimization_engine.optimizers import ConfigurationError  # noqa: E402


@pytest.fixture(scope="module")
def returns():
    return prices_to_returns(sample_dataset())


# ---------------------------------------------------------------------------
# Numbers that cannot mean anything
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", [0, -252])
def test_a_non_positive_periods_per_year_is_refused(periods):
    # It annualizes the covariance. Zero made Σ the zero matrix, and the solve
    # then reported an optimal, equal-weighted book with a volatility of 0.
    with pytest.raises(ConfigurationError, match="periods_per_year"):
        EngineConfig(periods_per_year=periods)
    with pytest.raises(ConfigurationError, match="periods_per_year"):
        EngineConfig.from_dict({"periods_per_year": periods})


@pytest.mark.parametrize("decay", [0.0, 1.0, 1.5, -0.2])
def test_an_ewma_decay_outside_the_open_unit_interval_is_refused(decay):
    # 1.5 used to surface as a LinAlgError from deep inside the estimator.
    with pytest.raises(ConfigurationError, match="ewma_lambda"):
        EngineConfig(covariance_method="ewma", ewma_lambda=decay)


# ---------------------------------------------------------------------------
# Booleans that arrived as text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["long_only", "fully_invested", "strict_mandate", "denoise"])
def test_a_quoted_false_in_json_reads_as_false(key):
    # bool("false") is True, so a JSON file that quoted its booleans turned
    # every one of these switches on — long_only most dangerously, since it
    # quietly forbade the shorts the author had asked for.
    config = EngineConfig.from_dict(json.loads(json.dumps({key: "false"})))
    assert getattr(config, key) is False
    config = EngineConfig.from_dict({key: "True"})
    assert getattr(config, key) is True


@pytest.mark.parametrize("value", ["no", "off", 2, None, "maybe"])
def test_a_boolean_that_is_not_true_or_false_is_refused(value):
    with pytest.raises(ConfigurationError, match="long_only"):
        EngineConfig.from_dict({"long_only": value})


def test_direct_construction_parses_booleans_the_same_way():
    config = EngineConfig(long_only="false", fully_invested="false")  # type: ignore[arg-type]
    assert config.long_only is False
    assert config.fully_invested is False


# ---------------------------------------------------------------------------
# A benchmark that arrived as a mapping
# ---------------------------------------------------------------------------


def test_a_benchmark_mapping_is_coerced_on_direct_construction():
    # The constraint layers and the stress scenarios were coerced; the
    # benchmark was not, so the first use of it raised AttributeError.
    config = EngineConfig(
        expected_returns={"A": 0.05, "B": 0.07},
        benchmark={"kind": "equal_weight"},  # type: ignore[arg-type]
    )
    assert isinstance(config.benchmark, BenchmarkSpec)
    assert config.benchmark_weight_map() == {"A": 0.5, "B": 0.5}


# ---------------------------------------------------------------------------
# strict_mandate and a bound on an asset the panel does not hold
# ---------------------------------------------------------------------------


def test_a_bound_on_a_misspelt_asset_is_refused_under_strict_mandate(returns):
    config = EngineConfig(
        optimizer=OptimizerSpec(name="min_variance"),
        bounds={"US_Equityy": [0.3, 0.4]},
        strict_mandate=True,
    )
    with pytest.raises(ConfigurationError, match="US_Equityy"):
        run_engine(returns, config)


def test_the_same_bound_without_strict_mandate_is_still_only_a_warning(returns):
    config = EngineConfig(
        optimizer=OptimizerSpec(name="min_variance"),
        bounds={"US_Equityy": [0.3, 0.4]},
    )
    run = run_engine(returns, config)
    assert any("US_Equityy" in w for w in run.warnings)

