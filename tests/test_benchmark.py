"""Benchmark selection, and optimizing against one rather than against cash."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.benchmark import (
    BenchmarkError,
    BenchmarkSpec,
    portfolio_returns_from_weights,
    resolve_benchmark,
)
from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.optimizers import ConfigurationError
from optimization_engine.optimizers.factory import constraints_from_config
from optimization_engine.reporting.exporters import run_sheets
from optimization_engine.ui_state import derive_widget_state


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=31))


def _config(returns: pd.DataFrame, **kwargs) -> EngineConfig:
    spec = kwargs.pop("optimizer", OptimizerSpec(name="mean_variance", risk_aversion=3.0))
    return EngineConfig(optimizer=spec, **kwargs)


def _tracking_error(run) -> float:
    weights = run.result.weights
    bench = run.benchmark.weights.reindex(weights.index).fillna(0.0)
    active = (weights - bench).values
    return float(np.sqrt(max(active @ run.cov_matrix.values @ active, 0.0)))


def _active_share(run) -> float:
    weights = run.result.weights
    bench = run.benchmark.weights.reindex(weights.index).fillna(0.0)
    return float((weights - bench).abs().sum() / 2.0)


# ---------------------------------------------------------------------------
# Specification and resolution
# ---------------------------------------------------------------------------


def test_equal_weight_resolves_to_one_over_n(returns):
    resolved = resolve_benchmark(BenchmarkSpec(kind="equal_weight"), returns)
    assert resolved is not None
    assert np.allclose(resolved.weights.values, 1.0 / returns.shape[1])
    expected = returns.mean(axis=1)
    pd.testing.assert_series_equal(
        resolved.returns, expected, check_names=False, atol=1e-12
    )


def test_single_asset_benchmark_is_that_asset(returns):
    asset = returns.columns[2]
    resolved = resolve_benchmark(
        BenchmarkSpec(kind="single_asset", asset=asset), returns
    )
    pd.testing.assert_series_equal(
        resolved.returns, returns[asset], check_names=False
    )
    assert resolved.label == asset


def test_custom_weights_are_normalized(returns):
    a, b = returns.columns[0], returns.columns[1]
    resolved = resolve_benchmark(
        BenchmarkSpec(kind="custom_weights", weights={a: 3.0, b: 1.0}), returns
    )
    assert resolved.weights[a] == pytest.approx(0.75)
    assert resolved.weights[b] == pytest.approx(0.25)
    assert resolved.weights.sum() == pytest.approx(1.0)


def test_no_benchmark_resolves_to_none(returns):
    assert resolve_benchmark(BenchmarkSpec(), returns) is None
    assert resolve_benchmark(None, returns) is None


def test_unknown_kind_is_rejected():
    with pytest.raises(BenchmarkError, match="Unknown benchmark kind"):
        BenchmarkSpec(kind="whatever")


def test_asset_outside_the_universe_is_rejected(returns):
    with pytest.raises(BenchmarkError, match="not in the universe"):
        resolve_benchmark(
            BenchmarkSpec(kind="single_asset", asset="NOT_AN_ASSET"), returns
        )


def test_custom_weights_naming_unknown_assets_are_rejected(returns):
    with pytest.raises(BenchmarkError, match="outside the universe"):
        resolve_benchmark(
            BenchmarkSpec(kind="custom_weights", weights={"NOPE": 1.0}), returns
        )


def test_zero_sum_custom_weights_are_rejected(returns):
    a, b = returns.columns[0], returns.columns[1]
    with pytest.raises(BenchmarkError, match="sum to zero"):
        resolve_benchmark(
            BenchmarkSpec(kind="custom_weights", weights={a: 1.0, b: -1.0}), returns
        )


def test_buy_and_hold_differs_from_periodic_rebalancing(returns):
    weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    periodic = portfolio_returns_from_weights(returns, weights, "periodic")
    held = portfolio_returns_from_weights(returns, weights, "buy_and_hold")
    # The first period is identical by construction; the paths separate after.
    assert periodic.iloc[0] == pytest.approx(held.iloc[0])
    assert not np.allclose(periodic.values, held.values)
    # A drifting book still compounds to a sensible number.
    assert np.isfinite(float((1 + held).prod()))


def test_buy_and_hold_of_a_single_asset_is_that_asset(returns):
    asset = returns.columns[0]
    weights = pd.Series(0.0, index=returns.columns)
    weights[asset] = 1.0
    held = portfolio_returns_from_weights(returns, weights, "buy_and_hold")
    assert np.allclose(held.values, returns[asset].values, atol=1e-12)


def test_external_series_has_returns_but_no_weights(returns):
    external = pd.DataFrame({"INDEX": returns.mean(axis=1) * 0.9})
    resolved = resolve_benchmark(
        BenchmarkSpec(kind="external", series_name="INDEX"), returns, external
    )
    assert resolved.weights is None
    assert not resolved.has_weights
    assert resolved.weights_frame() is None
    assert resolved.label == "INDEX"


def test_external_series_without_data_is_rejected(returns):
    with pytest.raises(BenchmarkError, match="no external return series"):
        resolve_benchmark(BenchmarkSpec(kind="external", series_name="X"), returns)


def test_external_series_with_no_overlap_is_rejected(returns):
    shifted = pd.Series(
        returns.mean(axis=1).values,
        index=returns.index + pd.DateOffset(years=25),
        name="INDEX",
    )
    with pytest.raises(BenchmarkError, match="shares no dates"):
        resolve_benchmark(BenchmarkSpec(kind="external"), returns, shifted)


def test_spec_round_trips_through_a_dict():
    spec = BenchmarkSpec(
        kind="custom_weights",
        weights={"A": 0.6, "B": 0.4},
        label="Policy",
        rebalance="buy_and_hold",
    )
    restored = BenchmarkSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert BenchmarkSpec.from_dict("equal_weight").kind == "equal_weight"


def test_summary_describes_the_benchmark(returns):
    resolved = resolve_benchmark(BenchmarkSpec(kind="equal_weight"), returns)
    row = resolved.summary().iloc[0]
    assert row["kind"] == "equal_weight"
    assert bool(row["position_based"]) is True
    assert int(row["holdings"]) == returns.shape[1]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_round_trips_the_benchmark_block():
    cfg = EngineConfig(
        expected_returns={"A": 0.05, "B": 0.07},
        benchmark=BenchmarkSpec(kind="single_asset", asset="A"),
        max_tracking_error=0.03,
        max_active_share=0.4,
    )
    restored = EngineConfig.from_dict(cfg.to_dict())
    assert restored.benchmark.kind == "single_asset"
    assert restored.benchmark.asset == "A"
    assert restored.max_tracking_error == pytest.approx(0.03)
    assert restored.max_active_share == pytest.approx(0.4)


def test_one_over_n_depends_on_the_universe_it_is_asked_about():
    cfg = EngineConfig(
        expected_returns={"A": 0.05, "B": 0.07},
        benchmark=BenchmarkSpec(kind="equal_weight"),
    )
    assert cfg.benchmark_weight_map() == {"A": 0.5, "B": 0.5}
    three = cfg.benchmark_weight_map(["A", "B", "C"])
    assert set(three) == {"A", "B", "C"}
    assert three["A"] == pytest.approx(1 / 3)


def test_explicit_weights_are_the_benchmark_when_no_spec_says_otherwise():
    # They used to *win over* a spec that said something else — in the
    # constraints only, while the report kept the spec (review E7). Two
    # different benchmarks are now refused; see test_benchmark_single_source.
    cfg = EngineConfig(
        expected_returns={"A": 0.05, "B": 0.07},
        benchmark_weights={"A": 1.0},
    )
    assert cfg.benchmark_weight_map() == {"A": 1.0, "B": 0.0}


def test_external_benchmark_has_no_weight_map():
    cfg = EngineConfig(
        expected_returns={"A": 0.05},
        benchmark=BenchmarkSpec(kind="external", series_name="INDEX"),
    )
    assert cfg.benchmark_weight_map() is None


def test_constraints_carry_the_benchmark(returns):
    cfg = _config(
        returns,
        benchmark=BenchmarkSpec(kind="equal_weight"),
        max_tracking_error=0.05,
        max_active_share=0.6,
    )
    constraints = constraints_from_config(cfg, list(returns.columns))
    assert constraints.is_benchmark_relative
    assert constraints.max_tracking_error == pytest.approx(0.05)
    vector = constraints.benchmark_vector(list(returns.columns))
    assert vector is not None and vector.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Optimizing against the benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method", ["mean_variance", "min_variance", "max_sharpe", "max_diversification"]
)
def test_tracking_error_budget_binds(returns, method):
    cfg = _config(
        returns,
        optimizer=OptimizerSpec(name=method, risk_aversion=3.0),
        benchmark=BenchmarkSpec(kind="equal_weight"),
        max_tracking_error=0.02,
    )
    run = run_engine(returns, cfg)
    assert _tracking_error(run) <= 0.02 + 1e-4
    assert run.result.violations == []


def test_active_share_budget_binds(returns):
    cfg = _config(
        returns,
        benchmark=BenchmarkSpec(kind="equal_weight"),
        max_active_share=0.15,
    )
    run = run_engine(returns, cfg)
    assert _active_share(run) <= 0.15 + 1e-4
    assert run.result.violations == []


def test_a_tighter_budget_produces_a_portfolio_closer_to_the_index(returns):
    def solve(te: float) -> float:
        cfg = _config(
            returns,
            benchmark=BenchmarkSpec(kind="equal_weight"),
            max_tracking_error=te,
        )
        return _active_share(run_engine(returns, cfg))

    assert solve(0.01) < solve(0.06)


def test_active_mean_variance_reduces_to_mean_variance_at_zero_benchmark(returns):
    absolute = run_engine(
        returns, _config(returns, optimizer=OptimizerSpec(
            name="mean_variance", risk_aversion=4.0))
    ).result.weights
    active = run_engine(
        returns,
        _config(
            returns,
            optimizer=OptimizerSpec(name="active_mean_variance", risk_aversion=4.0),
            # A zero benchmark has to say it is not to be normalized, now that
            # the benchmark_weights shorthand is normalized like the spec.
            benchmark=BenchmarkSpec(
                kind="custom_weights",
                weights={a: 0.0 for a in returns.columns},
                normalize=False,
            ),
        ),
    ).result.weights
    # Same objective up to a constant shift, so the same portfolio.
    assert float((absolute - active).abs().max()) < 1e-5


def test_active_mean_variance_spends_its_tracking_error_budget(returns):
    cfg = _config(
        returns,
        optimizer=OptimizerSpec(name="active_mean_variance", risk_aversion=5.0),
        benchmark=BenchmarkSpec(kind="equal_weight"),
        max_tracking_error=0.03,
    )
    run = run_engine(returns, cfg)
    extras = run.result.extras
    assert extras["mode"] == "target_tracking_error"
    # Maximizing active return against a budget spends the whole budget.
    assert extras["expected_tracking_error"] == pytest.approx(0.03, abs=1e-4)
    assert _tracking_error(run) == pytest.approx(0.03, abs=1e-4)


def test_active_mean_variance_needs_a_benchmark(returns):
    cfg = _config(
        returns, optimizer=OptimizerSpec(name="active_mean_variance")
    )
    with pytest.raises(ConfigurationError, match="optimizes against a benchmark"):
        run_engine(returns, cfg)


def test_a_limit_without_a_benchmark_is_rejected(returns):
    cfg = _config(returns, max_tracking_error=0.03)
    with pytest.raises(ConfigurationError, match="without a benchmark"):
        run_engine(returns, cfg)


def test_a_method_that_cannot_bind_the_limit_still_reports_it(returns, caplog):
    cfg = _config(
        returns,
        optimizer=OptimizerSpec(name="hrp"),
        benchmark=BenchmarkSpec(kind="equal_weight"),
        max_tracking_error=0.001,
    )
    run = run_engine(returns, cfg)
    # Not enforced — but named in the compliance report rather than dropped.
    assert any("Tracking error" in v for v in run.result.violations)


def test_an_impossible_budget_reports_why(returns):
    cfg = _config(
        returns,
        benchmark=BenchmarkSpec(kind="single_asset", asset=returns.columns[0]),
        bounds={returns.columns[0]: [0.0, 0.10]},
        max_tracking_error=1e-5,
    )
    with pytest.raises(RuntimeError, match="infeasible|Solver|status"):
        run_engine(returns, cfg)


# ---------------------------------------------------------------------------
# Downstream plumbing
# ---------------------------------------------------------------------------


def test_the_run_carries_its_benchmark(returns):
    cfg = _config(returns, benchmark=BenchmarkSpec(kind="equal_weight"))
    run = run_engine(returns, cfg)
    assert run.benchmark is not None
    assert run.benchmark_label == "Equal weight (1/N)"
    assert run.benchmark_returns is not None
    assert "benchmark" in run.backtest_returns().columns
    assert run.assumptions()["benchmark"] == "Equal weight (1/N)"


def test_active_analytics_refuse_an_index_with_no_holdings(returns):
    external = pd.DataFrame({"INDEX": returns.mean(axis=1)})
    cfg = _config(
        returns, benchmark=BenchmarkSpec(kind="external", series_name="INDEX")
    )
    run = run_engine(returns, cfg, external_returns=external)
    assert run.benchmark is not None
    with pytest.raises(ValueError, match="no position-based benchmark"):
        run.active_risk_decomposition()


def test_the_workbook_carries_the_benchmark(returns):
    cfg = _config(returns, benchmark=BenchmarkSpec(kind="equal_weight"))
    run = run_engine(returns, cfg)
    sheets = run_sheets(run, riskfree_rate=0.02)
    assert "benchmark" in sheets
    assert "benchmark_weights" in sheets
    assert "performance_relative" in sheets
    active = sheets["benchmark_weights"]["active_weight"]
    assert active.sum() == pytest.approx(0.0, abs=1e-6)


def test_the_workbook_omits_relative_sheets_without_a_benchmark(returns):
    run = run_engine(returns, _config(returns))
    sheets = run_sheets(run, riskfree_rate=0.02)
    assert "benchmark" not in sheets
    assert "performance_relative" not in sheets


def test_widget_state_knows_which_methods_can_bind_the_limits():
    assert derive_widget_state("mean_variance")["benchmark_limits"]["enabled"] is True
    hrp = derive_widget_state("hrp")["benchmark_limits"]
    assert hrp["enabled"] is False
    assert hrp["tooltip"]
