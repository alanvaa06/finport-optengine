"""What the simulation layer must not quietly swallow.

Three failures that all looked like nothing from the outside: a target dated
on a day the market was shut, an opening solve that raised, and a result hash
whose rounding sat below the noise floor of the numbers it was rounding.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.backtest import (
    BacktestSpec,
    CostSpec,
    build_tearsheet,
    compute_result_hash,
    run_backtest,
    walk_forward_run,
)
from optimization_engine.backtest.results import (
    HASH_VERSION,
    empty_costs,
    empty_trades,
)
from optimization_engine.data.loader import prices_to_returns, sample_dataset

ASSETS = ["A", "B"]


@pytest.fixture(scope="module")
def business_days() -> pd.DatetimeIndex:
    """Forty business days from Monday 2024-01-01, so weekends are real gaps."""
    return pd.bdate_range("2024-01-01", periods=40)


@pytest.fixture(scope="module")
def flat_returns(business_days) -> pd.DataFrame:
    """A calm panel: this file is about the calendar, not about performance."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(0.0, 0.004, size=(len(business_days), len(ASSETS))),
        index=business_days,
        columns=ASSETS,
    )


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=400, seed=11))


@pytest.fixture(scope="module")
def equal_weights(returns) -> pd.Series:
    return pd.Series(1.0 / returns.shape[1], index=returns.columns)


# -- off-index schedule dates -----------------------------------------------


def test_offindex_schedule_dates_trade_next_bar(flat_returns, caplog):
    """A Sunday-stamped weekly schedule trades on Monday, not at inception.

    ``frequency="none"`` puts exactly one calendar mark in the run — the
    initial purchase — so a schedule whose dates never appear in the index
    used to produce a single trade and then buy-and-hold, silently. The
    targets were not lost, but the timing the caller asked for was.
    """
    sundays = pd.DatetimeIndex(["2024-01-07", "2024-01-14", "2024-01-21"])
    mondays = pd.DatetimeIndex(["2024-01-08", "2024-01-15", "2024-01-22"])
    assert not sundays.isin(flat_returns.index).any(), "the premise: not bars"
    assert mondays.isin(flat_returns.index).all()

    schedule = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [0.75, 0.25]], index=sundays, columns=ASSETS
    )
    spec = BacktestSpec(frequency="none", execution_lag=0)

    with caplog.at_level(logging.WARNING, logger="optimization_engine.backtest.runner"):
        run = run_backtest(flat_returns, schedule, spec)

    assert list(run.rebalance_dates) == list(mondays)
    moved = run.meta.notes["schedule_dates_moved"]
    assert moved == {
        "2024-01-07T00:00:00": "2024-01-08T00:00:00",
        "2024-01-14T00:00:00": "2024-01-15T00:00:00",
        "2024-01-21T00:00:00": "2024-01-22T00:00:00",
    }
    # Each Monday holds the target its Sunday asked for, not a later one.
    for monday, target in zip(mondays, schedule.to_numpy()):
        np.testing.assert_allclose(run.weights.loc[monday].to_numpy(), target)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "3 date(s) moved" in warnings[0].getMessage()


def test_in_index_schedule_dates_do_not_move(flat_returns):
    """The fix is idempotent: a date that is already a bar maps to itself."""
    mondays = pd.DatetimeIndex(["2024-01-08", "2024-01-15", "2024-01-22"])
    schedule = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [0.75, 0.25]], index=mondays, columns=ASSETS
    )
    run = run_backtest(flat_returns, schedule, BacktestSpec(frequency="none"))

    assert list(run.rebalance_dates) == list(mondays)
    assert "schedule_dates_moved" not in run.meta.notes
    assert "schedule_dates_collapsed" not in run.meta.notes


def test_two_offindex_dates_between_bars_record_their_collapse(flat_returns):
    """Saturday and Sunday land on the same Monday, and the later one wins.

    ``decisions`` is built through a set, so the collapse cannot be seen in
    the trade record at all. It has to be said out loud.
    """
    weekend = pd.DatetimeIndex(["2024-01-06", "2024-01-07"])
    schedule = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=weekend, columns=ASSETS)
    run = run_backtest(flat_returns, schedule, BacktestSpec(frequency="none"))

    monday = pd.Timestamp("2024-01-08")
    assert list(run.rebalance_dates) == [monday]
    assert run.meta.notes["schedule_dates_collapsed"] == {
        "2024-01-08T00:00:00": ["2024-01-06T00:00:00", "2024-01-07T00:00:00"]
    }
    # The freshest target the desk holds is the Sunday one.
    np.testing.assert_allclose(run.weights.loc[monday].to_numpy(), [0.0, 1.0])


def test_schedule_after_last_bar_is_dropped_and_noted(flat_returns):
    """There is no bar left to fill an order dated after the sample ends."""
    last_bar = flat_returns.index[-1]
    late = last_bar + pd.Timedelta(days=30)
    schedule = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=pd.DatetimeIndex([flat_returns.index[5], late]),
        columns=ASSETS,
    )
    run = run_backtest(flat_returns, schedule, BacktestSpec(frequency="none"))

    assert run.meta.notes["schedule_dates_dropped"] == [late.isoformat()]
    assert list(run.rebalance_dates) == [flat_returns.index[5]]
    # The dropped row never reaches the book, on any date.
    np.testing.assert_allclose(run.weights.iloc[-1].to_numpy()[1], 0.0, atol=1e-12)


# -- a failed first solve ---------------------------------------------------


def test_failed_first_solve_holds_cash_not_dropped(returns):
    """The opening failure is a cash period in the record, not a shorter one.

    Writing no schedule row meant the evaluation started at the first
    *success*, so every period the process could not trade — exactly the ones
    a fragile process gets wrong — disappeared from the track record.
    """
    calls = {"n": 0}

    def solver(window: pd.DataFrame) -> pd.Series:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("covariance would not invert")
        inverse = 1.0 / window.std()
        return inverse / inverse.sum()

    lookback, rebalance_every = 60, 20
    walk = walk_forward_run(
        returns,
        solver,
        lookback=lookback,
        rebalance_every=rebalance_every,
        spec=BacktestSpec(costs=CostSpec(commission_bps=10)),
    )

    assert len(walk.run.returns) == len(returns) - lookback
    assert walk.run.returns.index[0] == returns.index[lookback]

    opening = walk.weights_history.index[0]
    assert opening == returns.index[lookback]
    assert (walk.weights_history.loc[opening] == 0.0).all()
    assert walk.windows["status"].iloc[0].startswith("failed")
    assert "covariance would not invert" in walk.windows["status"].iloc[0]

    notes = walk.run.meta.notes
    assert notes["periods_in_cash_after_failed_solve"] == rebalance_every
    # Cash earns nothing and costs nothing; those periods are really in there.
    assert walk.run.returns.iloc[:rebalance_every].eq(0.0).all()

    sheet = build_tearsheet(
        walk.run, returns.loc[walk.run.returns.index], n_trials=4
    )
    assert any("held in cash" in caveat for caveat in sheet.caveats)


def test_a_successful_first_solve_is_unchanged(returns):
    """The window fix must be a no-op whenever the opening solve works."""

    def solver(window: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    walk = walk_forward_run(returns, solver, lookback=60, rebalance_every=20)
    assert walk.run.returns.index[0] == returns.index[60]
    assert len(walk.run.returns) == len(returns) - 60
    assert walk.run.meta.notes["periods_in_cash_after_failed_solve"] == 0


def test_every_solve_failing_is_still_refused(returns):
    """Cash on the first failure must not turn an all-failed run into a result."""

    def never(window: pd.DataFrame) -> pd.Series:
        raise RuntimeError("no solution anywhere")

    with pytest.raises(ValueError, match="Every walk-forward solve failed"):
        walk_forward_run(returns, never, lookback=60, rebalance_every=20)


# -- the result hash --------------------------------------------------------


def test_result_hash_stable_at_large_nav(returns, equal_weights):
    """Twelve significant figures, not twelve decimals.

    A NAV path starting at ten million carries about nine fractional decimals
    of float64 precision, so rounding to twelve *decimals* rounded nothing at
    all and a single ulp of BLAS disagreement changed the digest.
    """
    spec = BacktestSpec(
        frequency="monthly", initial_capital=1e7, costs=CostSpec(commission_bps=10)
    )
    first = run_backtest(returns, equal_weights, spec)
    second = run_backtest(returns, equal_weights, spec)
    assert first.meta.result_hash == second.meta.result_hash
    assert first.meta.hash_version == HASH_VERSION == 2

    position = len(first.nav) // 2
    original = float(first.nav.iloc[position])

    one_ulp = first.nav.copy()
    one_ulp.iloc[position] = np.nextafter(original, np.inf)
    assert float(one_ulp.iloc[position]) != original, "the premise: a real change"
    assert (
        compute_result_hash(one_ulp, first.trades, first.costs, first.weights)
        == first.meta.result_hash
    ), "one ulp is noise, not a different run"

    material = first.nav.copy()
    material.iloc[position] = original * (1.0 + 1e-9)
    assert (
        compute_result_hash(material, first.trades, first.costs, first.weights)
        != first.meta.result_hash
    ), "a part per billion is a different NAV path"


def test_result_hash_sees_weight_path():
    """NAV and trades do not pin down what the book was holding."""
    dates = pd.bdate_range("2024-01-01", periods=4)
    nav = pd.Series([1e6, 1.01e6, 1.02e6, 1.03e6], index=dates, name="nav")
    trades, costs = empty_trades(), empty_costs()

    balanced = pd.DataFrame(0.5, index=dates, columns=ASSETS)
    tilted = pd.DataFrame([[0.6, 0.4]] * 4, index=dates, columns=ASSETS)

    assert compute_result_hash(nav, trades, costs, balanced) == compute_result_hash(
        nav, trades, costs, balanced.copy()
    )
    assert compute_result_hash(nav, trades, costs, balanced) != compute_result_hash(
        nav, trades, costs, tilted
    )
    # And the same holdings under different names are a different run too.
    renamed = balanced.rename(columns={"B": "C"})
    assert compute_result_hash(nav, trades, costs, balanced) != compute_result_hash(
        nav, trades, costs, renamed
    )


def test_weight_path_hash_tolerates_one_ulp():
    """The weight path is rounded relatively, like everything else in there."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    nav = pd.Series([1e7, 1.01e7, 1.02e7], index=dates, name="nav")
    trades, costs = empty_trades(), empty_costs()

    held = pd.DataFrame(
        [[0.3, 0.7], [0.31, 0.69], [0.32, 0.68]], index=dates, columns=ASSETS
    )
    nudged = held.copy()
    nudged.iloc[1, 0] = np.nextafter(float(held.iloc[1, 0]), np.inf)
    assert float(nudged.iloc[1, 0]) != float(held.iloc[1, 0])

    assert compute_result_hash(nav, trades, costs, held) == compute_result_hash(
        nav, trades, costs, nudged
    )

    moved = held.copy()
    moved.iloc[1, 0] = float(held.iloc[1, 0]) * (1.0 + 1e-9)
    assert compute_result_hash(nav, trades, costs, held) != compute_result_hash(
        nav, trades, costs, moved
    )


# -- what a schedule's dates mean ---------------------------------------------


def test_lag_zero_holds_each_target_over_the_bar_it_is_dated_on(flat_returns):
    """Pins the semantics the docs now state: the index is the first holding bar.

    An oracle that knows each bar's winner and stamps it on that bar earns the
    winner's return at ``execution_lag=0`` — the schedule date is the first bar
    the target is held over, not the close it was decided on — and earns
    nothing special at ``execution_lag=1``.
    """
    winner = (flat_returns["A"] > flat_returns["B"]).astype(float)
    oracle = pd.DataFrame({"A": winner, "B": 1.0 - winner})
    free = BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        same_bar = run_backtest(flat_returns, oracle, free)
        lagged = run_backtest(flat_returns, oracle, free.with_(execution_lag=1))
    best = flat_returns.max(axis=1)
    np.testing.assert_allclose(same_bar.returns.to_numpy(), best.to_numpy())
    assert float(lagged.returns.mean()) < float(best.mean())


def test_a_dated_schedule_at_lag_zero_warns_that_it_must_not_use_its_own_bar(
    flat_returns,
):
    """The default lag is honest only for a schedule built before each bar."""
    schedule = pd.DataFrame(
        0.5, index=flat_returns.index[::5], columns=ASSETS
    )
    with pytest.warns(UserWarning, match="execution_lag=0"):
        run_backtest(flat_returns, schedule, BacktestSpec(frequency="none"))


def test_no_schedule_warning_where_the_dates_cannot_leak(flat_returns, returns):
    """A lag, a constant book, and the walk-forward's own schedule stay quiet."""
    schedule = pd.DataFrame(0.5, index=flat_returns.index[::5], columns=ASSETS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_backtest(
            flat_returns, schedule, BacktestSpec(frequency="none", execution_lag=1)
        )
        run_backtest(
            flat_returns, pd.Series(0.5, index=ASSETS), BacktestSpec(frequency="none")
        )
        # The walk-forward stamps each target on the first bar it is held,
        # from a window that ends the bar before: lag 0 is exact for it.
        walk_forward_run(
            returns,
            lambda window: pd.Series(1.0 / window.shape[1], index=window.columns),
            lookback=60,
            rebalance_every=60,
            spec=BacktestSpec(frequency="none"),
        )

