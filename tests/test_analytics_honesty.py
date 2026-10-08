"""Numbers the analytics reported that were not the number their label named.

Each section pins one defect found in the 2026-10-08 review, with the
reproduction that exposed it. None of them raised: every one produced a
plausible figure — a drawdown, a Sharpe, a tracking error — that was simply
the wrong one, which is the kind of error nothing downstream catches.
"""

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

from optimization_engine.analytics.performance import (
    calmar_ratio,
    drawdown,
    summary_stats,
    time_under_water,
)
from optimization_engine.analytics.relative import relative_drawdown
from optimization_engine.analytics.risk import (
    drawdown_series,
    drawdown_table,
    max_drawdown_duration,
    ulcer_index,
)
from optimization_engine.backtest.sweep import SweepSpec, run_sweep
from optimization_engine.config import EngineConfig, OptimizerSpec

PPY = 252


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _reference_drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown measured the long way: wealth from 1.0, peak including 1.0."""
    wealth = np.cumprod(np.r_[1.0, 1.0 + returns.to_numpy(dtype=float)])
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    return pd.Series(dd[1:], index=returns.index)


# ---------------------------------------------------------------------------
# 1. A drawdown is measured from the capital invested, not from the first close
# ---------------------------------------------------------------------------
#
# The running peak used to start at the wealth *after* the first return, so a
# series that opened with a loss never counted it: [-10%, -10%, +5%, +1%]
# reported a -10% maximum drawdown against a true -19%.


@pytest.fixture()
def opens_losing() -> pd.Series:
    return pd.Series([-0.10, -0.10, 0.05, 0.01], index=_days(4))


def test_drawdown_counts_a_loss_on_the_first_period(opens_losing):
    dd = drawdown_series(opens_losing)
    assert float(dd.min()) == pytest.approx(-0.19)
    pd.testing.assert_series_equal(dd, _reference_drawdown(opens_losing))


def test_a_monotone_loser_is_underwater_by_its_whole_loss():
    r = pd.Series([-0.05] * 3, index=_days(3))
    assert float(drawdown_series(r).min()) == pytest.approx(0.95**3 - 1.0)


def test_a_series_that_opens_with_a_gain_is_unchanged():
    r = pd.Series([0.10, -0.20, -0.10, 0.05, 0.40, 0.01, -0.05, 0.02], index=_days(8))
    growth = np.log1p(r).cumsum()
    legacy = np.expm1(growth - growth.cummax())
    pd.testing.assert_series_equal(drawdown_series(r), legacy, check_names=False)
    pd.testing.assert_series_equal(drawdown_series(r), _reference_drawdown(r))


def test_summary_stats_and_calmar_inherit_the_true_drawdown(opens_losing):
    stats = summary_stats(opens_losing.to_frame("p"), PPY, riskfree_rate=0.0, extended=True)
    assert float(stats.loc["p", "Max Drawdown"]) == pytest.approx(-0.19)
    cagr = float((1.0 + opens_losing).prod() ** (PPY / len(opens_losing)) - 1.0)
    assert float(calmar_ratio(opens_losing, PPY)) == pytest.approx(cagr / 0.19)
    assert float(stats.loc["p", "Calmar Ratio"]) == pytest.approx(cagr / 0.19)


def test_the_drawdown_frame_starts_its_peak_at_the_starting_wealth(opens_losing):
    frame = drawdown(opens_losing, starting_wealth=1000.0)
    assert float(frame["Peaks"].iloc[0]) == pytest.approx(1000.0)
    assert float(frame["Drawdown"].iloc[0]) == pytest.approx(-0.10)
    implied = frame["Wealth"] / frame["Peaks"] - 1.0
    pd.testing.assert_series_equal(frame["Drawdown"], implied, check_names=False)


def test_ulcer_index_and_time_under_water_see_the_opening_loss(opens_losing):
    reference = _reference_drawdown(opens_losing)
    assert ulcer_index(opens_losing) == pytest.approx(float(np.sqrt((reference**2).mean())))
    assert time_under_water(opens_losing) == pytest.approx(1.0)


def test_drawdown_table_dates_an_episode_from_inception(opens_losing):
    table = drawdown_table(opens_losing)
    assert len(table) == 1
    episode = table.iloc[0]
    assert episode["max_drawdown"] == pytest.approx(-0.19)
    # The peak is the starting capital, which has no date in the index.
    assert pd.isna(episode["peak_date"])
    assert episode["peak_wealth"] == pytest.approx(1.0)
    assert episode["trough_date"] == opens_losing.index[1]
    assert episode["decline_periods"] == 2
    assert episode["total_periods"] == 4
    assert not episode["recovered"]
    assert max_drawdown_duration(opens_losing) == 4


def test_drawdown_table_still_dates_a_peak_inside_the_sample():
    r = pd.Series([0.10, -0.20, 0.30, 0.0], index=_days(4))
    episode = drawdown_table(r).iloc[0]
    assert episode["peak_date"] == r.index[0]
    assert episode["recovery_date"] == r.index[2]
    assert episode["decline_periods"] == 1
    assert episode["total_periods"] == 2


def test_relative_drawdown_counts_falling_behind_on_the_first_period():
    p = pd.Series([-0.10, 0.0, 0.0], index=_days(3))
    b = pd.Series([0.0, 0.0, 0.0], index=_days(3))
    assert float(relative_drawdown(p, b).min().iloc[0]) == pytest.approx(-0.10)


def test_the_sweep_column_reports_the_true_drawdown(opens_losing):
    base = EngineConfig(optimizer=OptimizerSpec(name="equal_weight"))
    results = run_sweep(
        base,
        SweepSpec(params={"optimizer.name": ["equal_weight"]}),
        lambda cfg: opens_losing,
        periods_per_year=PPY,
    )
    assert float(results.frame.loc[0, "max_drawdown"]) == pytest.approx(-0.19)


def test_the_underwater_charts_start_from_the_starting_capital(opens_losing):
    pytest.importorskip("plotly")
    from optimization_engine.reporting.plots import plot_drawdown, plot_relative_wealth

    underwater = plot_drawdown(opens_losing)
    assert min(underwater.data[0].y) == pytest.approx(-0.19)

    flat = pd.Series(0.0, index=opens_losing.index)
    relative = plot_relative_wealth(opens_losing, flat)
    high_water_mark = relative.data[0].y
    assert high_water_mark[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. A deflation that could not run says why, instead of "no trial count"
# ---------------------------------------------------------------------------
#
# build_tearsheet caught every exception from the deflation and set it to
# None, and the caveat then read "No trial count was supplied" — for a run
# whose caller had supplied forty.


@pytest.fixture()
def oos_run():
    from optimization_engine.backtest.runner import run_backtest
    from optimization_engine.backtest.spec import BacktestSpec

    rng = np.random.default_rng(7)
    assets = pd.DataFrame(
        rng.normal(0.0004, 0.01, (300, 2)), index=_days(300), columns=["A", "B"]
    )
    run = run_backtest(
        assets, pd.Series({"A": 0.5, "B": 0.5}), BacktestSpec(is_out_of_sample=True)
    )
    return run, assets


def _trial_caveats(sheet) -> list[str]:
    return [c for c in sheet.caveats if "trial" in c.lower()]


@pytest.mark.parametrize(
    "trials",
    [pd.Series([0.8]), pd.Series([np.nan, np.nan, np.nan])],
    ids=["one survivor", "no usable trial"],
)
def test_too_few_trial_sharpes_fall_back_to_the_sampling_variance(oos_run, trials):
    from optimization_engine.analytics.selection import deflated_sharpe_ratio
    from optimization_engine.backtest.tearsheet import build_tearsheet

    run, assets = oos_run
    sheet = build_tearsheet(run, assets, n_trials=40, trial_sharpes=trials)

    assert sheet.deflated_sharpe is not None
    assert sheet.deflated_sharpe.n_trials == 40
    expected = deflated_sharpe_ratio(
        run.returns, n_trials=40, periods_per_year=run.periods_per_year
    )
    assert sheet.deflated_sharpe.deflated == pytest.approx(expected.deflated)

    caveats = _trial_caveats(sheet)
    assert not any("No trial count was supplied" in c for c in caveats)
    assert any("sampling variance" in c and "40" in c for c in caveats)
    assert sheet.metadata["deflation_note"]


def test_a_deflation_that_cannot_run_says_why(oos_run, monkeypatch):
    import optimization_engine.analytics.selection as selection
    from optimization_engine.backtest.tearsheet import build_tearsheet

    def refuse(*args, **kwargs):
        raise ValueError("Need at least 3 observations; got 2.")

    monkeypatch.setattr(selection, "deflated_sharpe_ratio", refuse)
    run, assets = oos_run
    sheet = build_tearsheet(run, assets, n_trials=40)

    assert sheet.deflated_sharpe is None
    caveats = _trial_caveats(sheet)
    assert not any("No trial count was supplied" in c for c in caveats)
    assert any(
        "could not be deflated" in c and "Need at least 3 observations" in c and "40" in c
        for c in caveats
    )
    assert "Need at least 3 observations" in sheet.metadata["deflation_note"]


def test_an_unexpected_failure_in_the_deflation_is_not_swallowed(oos_run, monkeypatch):
    import optimization_engine.analytics.selection as selection
    from optimization_engine.backtest.tearsheet import build_tearsheet

    def crash(*args, **kwargs):
        raise RuntimeError("a bug, not an input the deflation cannot use")

    monkeypatch.setattr(selection, "deflated_sharpe_ratio", crash)
    run, assets = oos_run
    with pytest.raises(RuntimeError, match="a bug"):
        build_tearsheet(run, assets, n_trials=40)


def test_no_trial_count_still_says_so(oos_run):
    from optimization_engine.backtest.tearsheet import build_tearsheet

    run, assets = oos_run
    sheet = build_tearsheet(run, assets)
    assert sheet.deflated_sharpe is None
    assert any("No trial count was supplied" in c for c in sheet.caveats)
    assert sheet.metadata["deflation_note"] is None
