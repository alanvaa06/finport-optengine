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
