"""A column with no observations is NaN in every metric, not a crash.

``summary_stats`` raised ``IndexError`` on a frame with one all-NaN column:
``var_historic`` dropped the NaNs and handed ``np.percentile`` an empty array,
and ``cvar_historic`` reads its threshold from it. A walk-forward column that
never solved, or a benchmark with no overlap, took down the whole table,
including the columns that were fine.

``annualize_returns`` already returns NaN with no observations; the helpers
``summary_stats`` calls now follow the same rule, and the live columns come out
exactly as they do on their own.
"""

from __future__ import annotations

import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.analytics.performance import hit_rate, summary_stats
from optimization_engine.analytics.risk import (
    cvar_historic,
    max_drawdown_duration,
    omega_ratio,
    tail_ratio,
    var_gaussian,
    var_historic,
)

PPY = 252


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


@pytest.fixture(params=["all_nan", "no_rows"])
def empty(request) -> pd.Series:
    """The two shapes of "no observations": NaN rows, and no rows at all."""
    if request.param == "all_nan":
        return pd.Series(np.nan, index=_days(60))
    return pd.Series([], index=_days(0), dtype=float)


@pytest.fixture()
def repro() -> pd.DataFrame:
    """The reproduction in issue #40: one live column, one never observed."""
    return pd.DataFrame(
        {"a": np.random.default_rng(0).normal(0, 0.01, 60), "b": np.nan},
        index=_days(60),
    )


# ---------------------------------------------------------------------------
# 1. VaR and CVaR have no quantile to read without observations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [var_historic, cvar_historic, var_gaussian, partial(var_gaussian, modified=True)],
    ids=["var_historic", "cvar_historic", "var_gaussian", "var_cornish_fisher"],
)
def test_var_and_cvar_are_nan_without_observations(empty, metric):
    assert np.isnan(metric(empty))


@pytest.mark.parametrize("metric", [var_historic, cvar_historic])
def test_the_frame_path_blanks_only_the_empty_column(repro, metric):
    values = metric(repro)
    assert np.isnan(values["b"])
    assert values["a"] == pytest.approx(metric(repro["a"]))


# ---------------------------------------------------------------------------
# 2. The extended metrics say NaN, not infinity, zero or a warning
# ---------------------------------------------------------------------------
#
# These did not raise on an all-NaN column; they reported a number. Both of
# Omega's sums were zero, which read as gains with no losses: infinite. No
# drawdown episode read as never underwater: a duration of 0. The hit rate
# divided 0 by 0 with a RuntimeWarning, and the tail ratio raised on a series
# with no rows.


@pytest.mark.parametrize(
    "metric",
    [omega_ratio, tail_ratio, max_drawdown_duration, hit_rate],
    ids=["omega_ratio", "tail_ratio", "max_drawdown_duration", "hit_rate"],
)
def test_the_extended_metrics_are_nan_without_observations(empty, metric):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isnan(metric(empty))


def test_the_tail_ratio_reads_the_observations_a_column_has():
    # A NaN period is neither a gain nor a loss, as for var_historic: a late
    # listing used to make the whole ratio NaN.
    stream = pd.Series(np.random.default_rng(40).normal(0, 0.01, 300), index=_days(300))
    padded = stream.where(stream.index >= stream.index[100])
    assert tail_ratio(padded) == pytest.approx(tail_ratio(stream.iloc[100:]))


# ---------------------------------------------------------------------------
# 3. summary_stats blanks the empty column and leaves the others alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extended", [False, True])
def test_summary_stats_blanks_the_empty_column_and_keeps_the_rest(repro, extended):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        table = summary_stats(repro, periods_per_year=PPY, extended=extended)

    row = table.loc["b"]
    assert row.isna().all(), f"not NaN for a column with no data: {row[row.notna()].to_dict()}"
    alone = summary_stats(repro[["a"]], periods_per_year=PPY, extended=extended)
    pd.testing.assert_series_equal(table.loc["a"], alone.loc["a"])


@pytest.mark.parametrize("extended", [False, True])
def test_a_frame_with_no_rows_is_nan_throughout(repro, extended):
    # A benchmark with no overlap leaves no rows once the two are aligned.
    table = summary_stats(repro.iloc[:0], periods_per_year=PPY, extended=extended)
    assert list(table.index) == ["a", "b"]
    assert table.isna().all().all()
