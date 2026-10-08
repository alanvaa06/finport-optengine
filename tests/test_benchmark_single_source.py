"""One benchmark per run: the solve, the report and the active analytics agree.

Review item E7, and the explicit-vector case next to it.

``benchmark_weights`` drove the benchmark-relative *constraints* while
``resolve_benchmark(config.benchmark)`` drove the *report*, and the two
``EngineRun`` helpers that read weights disagreed about which came first. With
``benchmark=equal_weight`` and a 60/40 vector, the solve held tracking error to
3.00% against 60/40 while ``performance()``, the relative stream and active
share used 1/N (5.64%), and ``active_risk_decomposition`` used 60/40.

The explicit vector also skipped the validation the spec applies: a 60/40 whose
60 named an asset outside the universe became a benchmark summing to 0.4, and
the tracking-error budget was imposed against that, where
``BenchmarkSpec.weight_vector`` raises in the same case.

``benchmark_weights`` is now shorthand for a ``custom_weights`` spec — validated
and normalized like one — and every use resolves through
``EngineConfig.effective_benchmark()``. Setting both to different things is
refused rather than resolved by a precedence rule nobody can see.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.benchmark import BenchmarkError, BenchmarkSpec  # noqa: E402
from optimization_engine.cli import _apply_benchmark_flags  # noqa: E402
from optimization_engine.config import EngineConfig, OptimizerSpec  # noqa: E402
from optimization_engine.data.loader import prices_to_returns, sample_dataset  # noqa: E402
from optimization_engine.engine import run_engine  # noqa: E402

SIXTY_FORTY = {"US_Equity": 0.6, "US_Treasuries": 0.4}


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset())


def _tracking_error(weights: pd.Series, bench: pd.Series, cov: pd.DataFrame) -> float:
    active = weights - bench.reindex(weights.index).fillna(0.0)
    return float(np.sqrt(active @ cov.loc[weights.index, weights.index] @ active))


# ---------------------------------------------------------------------------
# The explicit vector is validated and normalized like a custom_weights spec
# ---------------------------------------------------------------------------


def test_an_explicit_vector_naming_an_asset_outside_the_universe_is_refused(returns):
    """The repro: SPX_Index is not in the panel, so the solve's benchmark summed to 0.4."""
    config = EngineConfig(
        optimizer=OptimizerSpec(name="min_variance"),
        benchmark_weights={"SPX_Index": 0.6, "US_Treasuries": 0.4},
        max_tracking_error=0.02,
    )
    with pytest.raises(BenchmarkError, match="SPX_Index"):
        run_engine(returns, config)


def test_an_explicit_vector_is_normalized_over_the_universe():
    config = EngineConfig(
        expected_returns={"A": 0.05, "B": 0.06, "C": 0.07},
        benchmark_weights={"A": 0.3, "B": 0.3},
    )
    assert config.benchmark_weight_map() == {"A": 0.5, "B": 0.5, "C": 0.0}


# ---------------------------------------------------------------------------
# One source for the solve, the stream, the report and the active analytics
# ---------------------------------------------------------------------------


def test_two_different_benchmarks_on_one_config_are_refused():
    """The E7 repro, refused where it is written."""
    with pytest.raises(BenchmarkError, match="benchmark_weights"):
        EngineConfig(
            benchmark=BenchmarkSpec(kind="equal_weight"), benchmark_weights=SIXTY_FORTY
        )


def test_the_conflict_is_refused_at_the_solve_when_it_was_made_afterwards(returns):
    config = EngineConfig(
        optimizer=OptimizerSpec(name="min_variance"), benchmark_weights=SIXTY_FORTY
    )
    config.benchmark = BenchmarkSpec(kind="equal_weight")
    with pytest.raises(BenchmarkError, match="benchmark_weights"):
        run_engine(returns, config)


def test_the_same_vector_written_both_ways_is_one_benchmark():
    config = EngineConfig(
        benchmark=BenchmarkSpec(kind="custom_weights", weights=SIXTY_FORTY, label="60/40"),
        benchmark_weights=SIXTY_FORTY,
    )
    assert config.effective_benchmark().display_label == "60/40"


def test_every_view_of_a_run_measures_against_the_benchmark_the_solve_used(returns):
    config = EngineConfig(
        optimizer=OptimizerSpec(name="min_variance"),
        benchmark_weights=SIXTY_FORTY,
        max_tracking_error=0.03,
    )
    run = run_engine(returns, config)
    assets = list(returns.columns)
    expected = pd.Series(SIXTY_FORTY).reindex(assets).fillna(0.0)
    weights = run.result.weights

    # The solve.
    assert _tracking_error(weights, expected, run.cov_matrix) <= 0.03 + 1e-4
    # The resolved benchmark: its weights, and the stream built from them.
    assert run.benchmark is not None
    pd.testing.assert_series_equal(
        run.benchmark.weights.reindex(assets), expected, check_names=False
    )
    stream = (returns * expected).sum(axis=1)
    np.testing.assert_allclose(run.benchmark.returns.to_numpy(), stream.to_numpy())
    # The active analytics.
    pd.testing.assert_series_equal(
        run._benchmark_weights().reindex(assets), expected, check_names=False
    )
    # The report's active share.
    report = run.performance(frequency=None)
    assert report.active_share == pytest.approx(
        0.5 * float((weights - expected).abs().sum())
    )


def test_the_cli_benchmark_flag_replaces_the_configs_vector():
    """``--benchmark`` overrides the config's block — the vector included.

    Before, ``--benchmark equal_weight`` on a config carrying a vector left the
    vector in place, so the solve used it and the report used 1/N.
    """
    config = EngineConfig(expected_returns={"A": 0.05, "B": 0.06}, benchmark_weights={"A": 1.0})
    _apply_benchmark_flags(config, argparse.Namespace(benchmark="equal_weight"), ["A", "B"])
    assert config.benchmark_weights is None
    assert config.benchmark_weight_map() == {"A": 0.5, "B": 0.5}
