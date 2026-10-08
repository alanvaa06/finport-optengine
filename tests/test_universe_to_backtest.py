"""End to end: rules -> eligibility -> walk-forward -> tearsheet.

One panel, three membership events — two names that list part-way through and
one that stops printing — run through the whole stack with the expected trade
log written out by hand rather than read off the output. The point of writing
it by hand is that every number below follows from the rule and the calendar
alone:

* the screen is "printed a return on each of the five sessions before the
  decision", so a name whose first print is at row ``F`` is eligible from row
  ``F + 5`` and not one session earlier;
* the solve is equal weight over whatever it is shown, so each target is
  ``1 / breadth``;
* the decision calendar is every fifth row from row 10, and with a
  same-period fill each decision trades on its own bar.

Anything that moves those numbers has changed what the universe layer means,
which is exactly what this file is for.
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

from optimization_engine.backtest.spec import BacktestSpec, CostSpec
from optimization_engine.backtest.tearsheet import build_tearsheet
from optimization_engine.backtest.walkforward import walk_forward_run
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.universe import Eligibility

#: The four names, kept short so the expectation table below stays readable.
EQUITY = "US_Equity"
BONDS = "US_Treasuries"
GOLD = "Gold"
CREDIT = "IG_Credit"
ASSETS = [EQUITY, BONDS, GOLD, CREDIT]

#: Row of the return panel each event happens on.
GOLD_FIRST_PRINT = 12
CREDIT_FIRST_PRINT = 20
BONDS_LAST_PRINT = 39

#: The screen's window, in sessions.
SCREEN_WINDOW = 5

#: Walk-forward geometry. Decisions land on rows 10, 15, ... 55.
LOOKBACK = 10
REBALANCE_EVERY = 5
DECISION_ROWS = list(range(LOOKBACK, 60, REBALANCE_EVERY))

#: The universe, worked out by hand. A name is eligible on row ``j`` when it
#: printed on all of rows ``j-5 … j-1``: equities throughout (from row 5),
#: gold from ``12 + 5``, credit from ``20 + 5``, and bonds until ``39 + 1``.
EXPECTED_BREADTH = {
    10: [EQUITY, BONDS],
    15: [EQUITY, BONDS],
    20: [EQUITY, BONDS, GOLD],
    25: [EQUITY, BONDS, GOLD, CREDIT],
    30: [EQUITY, BONDS, GOLD, CREDIT],
    35: [EQUITY, BONDS, GOLD, CREDIT],
    # Row 40 is the first bar the bonds line did not print on. The screen still
    # admits it — its five-session window ends on row 39, its last good print —
    # and so does the delisting rule: the book chosen on row 40 is held over
    # row 40, so the silence it would need to see has not happened yet.
    40: [EQUITY, BONDS, GOLD, CREDIT],
    45: [EQUITY, GOLD, CREDIT],
    50: [EQUITY, GOLD, CREDIT],
    55: [EQUITY, GOLD, CREDIT],
}

#: The decision the delisting rule liquidates the bonds line on: the first
#: one whose prior bars include a missing print. Row 40 would be reading its
#: own bar, which is the look-ahead the rule must not have.
BONDS_DELISTED_AT = 45

#: What the optimizer is actually shown, once delisting is applied on top.
#: On this panel the two rules agree on every decision.
EXPECTED_SOLVED_ON = dict(EXPECTED_BREADTH)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    """The sample panel with two synthetic listings and one delisting."""
    prices = sample_dataset(n_periods=61, seed=17, assets=ASSETS)
    frame = prices_to_returns(prices)[ASSETS]
    assert len(frame) == 60
    frame.loc[frame.index[:GOLD_FIRST_PRINT], GOLD] = np.nan
    frame.loc[frame.index[:CREDIT_FIRST_PRINT], CREDIT] = np.nan
    frame.loc[frame.index[BONDS_LAST_PRINT + 1:], BONDS] = np.nan
    return frame


@pytest.fixture(scope="module")
def eligibility(returns) -> Eligibility:
    """Two screens, conjoined: it printed all week, and it is not in freefall."""
    printed = Eligibility.from_rolling(
        returns,
        window=SCREEN_WINDOW,
        agg="count",
        op=">=",
        value=SCREEN_WINDOW,
        name=f"printed on all of the last {SCREEN_WINDOW} sessions",
    )
    not_collapsing = Eligibility.from_rolling(
        returns,
        window=SCREEN_WINDOW,
        agg="mean",
        op=">",
        value=-0.5,
        name="not in freefall",
    )
    return printed & not_collapsing


@pytest.fixture(scope="module")
def walk(returns, eligibility):
    """The whole stack, run once."""
    seen: list[list[str]] = []

    def solve(window: pd.DataFrame) -> pd.Series:
        seen.append(list(window.columns))
        if window.shape[1] == 0:
            raise ValueError("nothing eligible to solve")
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    run = walk_forward_run(
        returns,
        solve,
        lookback=LOOKBACK,
        rebalance_every=REBALANCE_EVERY,
        spec=BacktestSpec(
            frequency="none",
            execution_lag=0,
            costs=CostSpec(commission_bps=10.0),
        ),
        universe=eligibility,
        universe_policy="exclude",
        delisting_grace=0,
    )
    return run, seen


def test_screen_admits_each_name_on_the_session_it_should(returns, eligibility):
    """The warm-up is missing, and the first eligible row is F + window."""
    frame = eligibility.frame
    # Nothing at all is known before the screen's window has filled.
    assert frame.iloc[:SCREEN_WINDOW].isna().all().all()
    assert bool(frame.iloc[SCREEN_WINDOW][EQUITY]) is True

    for asset, first in ((GOLD, GOLD_FIRST_PRINT), (CREDIT, CREDIT_FIRST_PRINT)):
        listed = first + SCREEN_WINDOW
        assert bool(frame.iloc[listed - 1][asset]) is False
        assert bool(frame.iloc[listed][asset]) is True

    assert bool(frame.iloc[BONDS_LAST_PRINT + 1][BONDS]) is True
    assert bool(frame.iloc[BONDS_LAST_PRINT + 2][BONDS]) is False

    breadth = eligibility.breadth()
    assert list(breadth.iloc[[10, 20, 25, 45]]) == [2, 3, 4, 3]


def test_explain_names_the_screen_that_kept_a_name_out(returns, eligibility):
    text = eligibility.explain(returns.index[10], GOLD)
    assert "not eligible" in text
    assert "printed on all of the last 5 sessions" in text


def test_every_solve_sees_exactly_the_eligible_names(walk):
    """No solve is shown a name before it lists or after it delists."""
    _, seen = walk
    assert len(seen) == len(DECISION_ROWS)
    for row, columns in zip(DECISION_ROWS, seen):
        assert columns == EXPECTED_SOLVED_ON[row], f"decision row {row}"


def test_weights_history_matches_the_hand_written_table(walk, returns):
    run, _ = walk
    history = run.weights_history
    assert list(history.index) == [returns.index[row] for row in DECISION_ROWS]
    for row in DECISION_ROWS:
        members = EXPECTED_SOLVED_ON[row]
        expected = pd.Series(0.0, index=ASSETS)
        expected[members] = 1.0 / len(members)
        actual = history.loc[returns.index[row], ASSETS]
        pd.testing.assert_series_equal(
            actual.astype(float),
            expected,
            check_names=False,
            atol=1e-12,
            obj=f"targets on row {row}",
        )


def test_delisting_is_recorded_with_its_last_print(walk, returns):
    run, _ = walk
    delistings = run.run.meta.notes["delistings"]
    assert set(delistings) == {BONDS}
    assert delistings[BONDS] == {
        "last_print": pd.Timestamp(returns.index[BONDS_LAST_PRINT]).isoformat(),
        "delisted_at": pd.Timestamp(returns.index[BONDS_DELISTED_AT]).isoformat(),
    }
    assert run.run.meta.notes["delisting_grace"] == 0


def test_universe_note_records_breadth_and_policy(walk, returns):
    run, _ = walk
    note = run.run.meta.notes["universe"]
    assert note["policy"] == "exclude"
    assert note["unknown_assets"] == []
    # The runner's breadth is the *screen's* verdict; the delisting rule is the
    # walk-forward's. Neither can see row 40's missing print from row 40, so on
    # this panel they agree on every decision.
    for row, members in EXPECTED_BREADTH.items():
        key = pd.Timestamp(returns.index[row]).isoformat()
        assert note["breadth"][key] == len(members), f"breadth on row {row}"
    assert note["min_breadth"] == 2
    # The walk-forward already zeroed everything the universe excluded, so the
    # replay finds nothing left to liquidate.
    assert note["liquidated"] == {}
    assert [row["n_eligible"] for _, row in run.windows.iterrows()] == [
        len(EXPECTED_SOLVED_ON[row]) for row in DECISION_ROWS
    ]


def test_trade_log_matches_the_hand_written_expectation(walk, returns):
    """Who trades on each decision, and on no other bar."""
    run, _ = walk
    trades = run.run.trades
    traded_dates = sorted(set(trades["date"]))
    assert traded_dates == [returns.index[row] for row in DECISION_ROWS]

    #: Every decision trades the names it holds or wants. The opening decision
    #: only buys; row 20 and row 25 add a name and trim the incumbents; row 40
    #: still holds the bonds line, and row 45 is the liquidation.
    expected_by_row = {
        10: {EQUITY: "buy", BONDS: "buy"},
        15: {EQUITY: None, BONDS: None},
        20: {EQUITY: "sell", BONDS: "sell", GOLD: "buy"},
        25: {EQUITY: "sell", BONDS: "sell", GOLD: "sell", CREDIT: "buy"},
        30: {EQUITY: None, BONDS: None, GOLD: None, CREDIT: None},
        35: {EQUITY: None, BONDS: None, GOLD: None, CREDIT: None},
        40: {EQUITY: None, BONDS: None, GOLD: None, CREDIT: None},
        45: {EQUITY: "buy", BONDS: "sell", GOLD: "buy", CREDIT: "buy"},
        50: {EQUITY: None, GOLD: None, CREDIT: None},
        55: {EQUITY: None, GOLD: None, CREDIT: None},
    }
    for row, expected in expected_by_row.items():
        date = returns.index[row]
        day = trades[trades["date"] == date]
        assert set(day["asset"]) == set(expected), f"assets traded on row {row}"
        for asset, side in expected.items():
            if side is None:
                continue
            actual = day.loc[day["asset"] == asset, "side"].iloc[0]
            assert actual == side, f"{asset} on row {row}"

    # The listings never trade early, and the delisting never trades late.
    assert trades.loc[trades["asset"] == GOLD, "date"].min() == returns.index[20]
    assert trades.loc[trades["asset"] == CREDIT, "date"].min() == returns.index[25]
    assert trades.loc[trades["asset"] == BONDS, "date"].max() == returns.index[
        BONDS_DELISTED_AT
    ]


def test_the_delisted_line_is_sold_whole_and_the_book_holds_cash(walk, returns):
    run, _ = walk
    liquidation = run.run.trades[
        (run.run.trades["asset"] == BONDS)
        & (run.run.trades["date"] == returns.index[BONDS_DELISTED_AT])
    ]
    assert len(liquidation) == 1
    assert liquidation["side"].iloc[0] == "sell"
    held_before = float(run.run.weights.loc[returns.index[BONDS_DELISTED_AT - 1], BONDS])
    assert held_before == pytest.approx(0.25, abs=0.02)
    # Sold at its last mark: the line has not printed since row 39, so the
    # loop has replayed the position flat and the sale goes through at that
    # close. ``weights`` reports the book at the *start* of row 44, so the sale
    # is that position plus one bar of drift -- but the whole of it, not a slice.
    assert float(liquidation["traded_weight"].iloc[0]) == pytest.approx(
        -held_before, abs=5e-3
    )
    assert float(
        run.run.weights.loc[returns.index[BONDS_DELISTED_AT]:, BONDS].abs().max()
    ) == 0.0


def test_nothing_is_renormalised_behind_the_optimizer(walk, returns):
    """The book sums to one because the solve did, not because the runner fixed it."""
    run, _ = walk
    sums = run.weights_history.sum(axis=1)
    assert float(sums.min()) == pytest.approx(1.0, abs=1e-12)
    assert float(sums.max()) == pytest.approx(1.0, abs=1e-12)


def test_tearsheet_assembles_over_the_universe_run(walk, returns):
    run, _ = walk
    evaluation = returns.loc[run.run.nav.index]
    tearsheet = build_tearsheet(run.run, evaluation)
    assert not tearsheet.performance.empty
    assert "re-solve" not in tearsheet.describe()  # it is a run summary, not a log
    assert len(tearsheet.describe()) > 0
    # Every name that was ever held shows up in the position episodes, and the
    # bonds line has a closed one -- the universe closed it.
    assert set(tearsheet.episodes["asset"]) == {EQUITY, BONDS, GOLD, CREDIT}
    bonds = tearsheet.episodes[tearsheet.episodes["asset"] == BONDS]
    assert bool(bonds["closed"].iloc[0]) is True
    assert pd.Timestamp(bonds["end"].iloc[0]) <= returns.index[BONDS_DELISTED_AT]


def test_no_lookahead_end_to_end(returns, eligibility):
    """Rewriting the panel after row 30 leaves every earlier decision alone."""
    cut = 30

    def solve(window: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    shuffled = returns.copy()
    rng = np.random.default_rng(4)
    tail = shuffled.iloc[cut + 1:]
    shuffled.iloc[cut + 1:] = tail.to_numpy()[rng.permutation(len(tail))]
    shuffled_eligibility = Eligibility.from_rolling(
        shuffled,
        window=SCREEN_WINDOW,
        agg="count",
        op=">=",
        value=SCREEN_WINDOW,
        name=f"printed on all of the last {SCREEN_WINDOW} sessions",
    ) & Eligibility.from_rolling(
        shuffled, window=SCREEN_WINDOW, agg="mean", op=">", value=-0.5,
        name="not in freefall",
    )

    kwargs = dict(
        lookback=LOOKBACK,
        rebalance_every=REBALANCE_EVERY,
        spec=BacktestSpec(
            frequency="none", execution_lag=0, costs=CostSpec(commission_bps=10.0)
        ),
        universe_policy="exclude",
        delisting_grace=0,
    )
    base = walk_forward_run(returns, solve, universe=eligibility, **kwargs)
    other = walk_forward_run(shuffled, solve, universe=shuffled_eligibility, **kwargs)

    date = returns.index[cut]
    pd.testing.assert_frame_equal(
        base.weights_history.loc[:date], other.weights_history.loc[:date]
    )
