"""The point-in-time universe: three-valued membership, and what it does to a run.

The property this file exists to pin is that "not evaluable" is a state of its
own. A rule with a warm-up period does not know whether a name was eligible on
day one, and answering ``False`` there is a look-ahead bug wearing a boolean:
it asserts a fact about a window that has not happened yet. Every test below
either checks that missing stays missing, or checks the one place it is allowed
to collapse into a hard answer -- and that the caller had to name the policy.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.backtest.runner import run_backtest
from optimization_engine.backtest.spec import BacktestSpec, CostSpec
from optimization_engine.backtest.walkforward import walk_forward_run
from optimization_engine.constraints import ConstraintLayer
from optimization_engine.universe import (
    Classification,
    Eligibility,
    Signal,
    UniverseError,
    point_in_time_mask,
)

T = True
F = False
NA = pd.NA

#: The nine ``(left, right)`` states of a three-valued logic, in a fixed order.
NINE = [
    (T, T), (T, F), (T, NA),
    (F, T), (F, F), (F, NA),
    (NA, T), (NA, F), (NA, NA),
]


def _dates(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _signal(rows: list[list[object]], assets: list[str]) -> Signal:
    return Signal(pd.DataFrame(rows, index=_dates(len(rows)), columns=assets))


def _states(frame: pd.DataFrame, asset: str) -> list[object]:
    """A column as plain ``True`` / ``False`` / ``None``, for readable asserts."""
    return [None if pd.isna(v) else bool(v) for v in frame[asset]]


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


def test_signal_kleene_truth_table():
    """All nine combinations of ``&`` and ``|``, plus ``~``, against Kleene.

    pandas' ``boolean`` dtype already implements this; the test is here to
    prove it empirically rather than to re-derive it, because the whole
    module rests on ``True & NA == NA`` while ``False & NA == False``.
    """
    left = Signal(
        pd.DataFrame(
            {"X": [a for a, _ in NINE]}, index=_dates(9), dtype="boolean"
        )
    )
    right = Signal(
        pd.DataFrame(
            {"X": [b for _, b in NINE]}, index=_dates(9), dtype="boolean"
        )
    )

    expected_and = [T, F, None, F, F, F, None, F, None]
    expected_or = [T, T, T, T, F, None, T, None, None]

    assert _states((left & right).frame, "X") == expected_and
    assert _states((left | right).frame, "X") == expected_or
    assert _states((~left).frame, "X") == [
        F, F, F, T, T, T, None, None, None,
    ]


def test_signal_refuses_a_non_date_index():
    with pytest.raises(UniverseError, match="dates"):
        Signal(pd.DataFrame({"X": [True, False]}, index=[0, 1]))


def test_signal_alignment_fills_with_missing_not_false():
    """A name only one side knows about is unknown to the pair, never False."""
    a = _signal([[T], [F]], ["A"])
    b = Signal(
        pd.DataFrame({"B": [T, T]}, index=_dates(2), dtype="boolean")
    )
    combined = a & b
    assert list(combined.assets) == ["A", "B"]
    # B is unknown to ``a``: True & NA is unknown, True & unknown-B likewise.
    assert _states(combined.frame, "B") == [None, None]
    # A is unknown to ``b``, but Kleene still decides the row where A is False:
    # nothing an unknown could turn out to be rescues a name already rejected.
    assert _states(combined.frame, "A") == [None, False]


def test_signal_as_of_reads_the_row_in_force():
    sig = _signal([[T], [F], [T]], ["A"])
    index = sig.index
    assert bool(sig.as_of(index[1])["A"]) is False
    # A date between two rows reads the earlier one, never the later.
    between = index[1] + pd.Timedelta(hours=6)
    assert bool(sig.as_of(between)["A"]) is False
    assert pd.isna(sig.as_of(index[0] - pd.Timedelta(days=1))["A"])


# ---------------------------------------------------------------------------
# Eligibility rules
# ---------------------------------------------------------------------------


def test_from_threshold_missing_input_is_missing_not_false():
    frame = pd.DataFrame(
        {"A": [1.0, np.nan, 3.0]}, index=_dates(3)
    )
    rule = Eligibility.from_threshold(frame, ">", 2.0)
    assert _states(rule.frame, "A") == [False, None, True]


def test_from_threshold_rejects_an_unknown_operator():
    frame = pd.DataFrame({"A": [1.0]}, index=_dates(1))
    with pytest.raises(UniverseError, match="operator"):
        Eligibility.from_threshold(frame, "=>", 1.0)


def test_from_rank_keeps_the_top_n_and_leaves_gaps_missing():
    frame = pd.DataFrame(
        [[3.0, 2.0, 1.0], [1.0, np.nan, 5.0]],
        index=_dates(2),
        columns=["A", "B", "C"],
    )
    rule = Eligibility.from_rank(frame, 2)
    assert _states(rule.frame, "A") == [True, True]
    # A missing characteristic is not ranked last, it is not ranked at all.
    assert _states(rule.frame, "B") == [True, None]
    assert _states(rule.frame, "C") == [False, True]
    tightest = Eligibility.from_rank(frame, 1)
    assert _states(tightest.frame, "A") == [True, False]
    assert _states(tightest.frame, "C") == [False, True]


def test_rolling_rule_first_rows_missing_not_false():
    """The warm-up rows have no window behind them, so they answer nothing.

    With a window of three periods strictly prior to the evaluation date, row
    3 is the first that can be evaluated at all. Rows 0-2 must be missing:
    ``False`` there would assert that the name failed a test that was never
    run.
    """
    frame = pd.DataFrame(
        {"A": [10.0, 10.0, 10.0, 10.0, 0.0, 0.0]}, index=_dates(6)
    )
    rule = Eligibility.from_rolling(frame, window=3, agg="mean", op=">", value=5.0)
    assert _states(rule.frame, "A") == [None, None, None, True, True, True]
    assert rule.frame["A"].isna().sum() == 3


def test_rolling_window_is_strictly_prior():
    """Changing the value *at* a date cannot change that date's verdict."""
    frame = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=_dates(5)
    )
    base = Eligibility.from_rolling(frame, window=2, agg="mean", op=">", value=0.5)
    poisoned = frame.copy()
    poisoned.iloc[3, 0] = -100.0
    after = Eligibility.from_rolling(poisoned, window=2, agg="mean", op=">", value=0.5)
    # Row 3 reads rows 1-2 only, so its own poisoned value is invisible to it.
    assert bool(base.frame.iat[3, 0]) is bool(after.frame.iat[3, 0]) is True
    # Row 4 reads rows 2-3, so it does see it.
    assert bool(after.frame.iat[4, 0]) is False


def test_rolling_rule_rejects_an_unknown_aggregation():
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=_dates(2))
    with pytest.raises(UniverseError, match="aggregation"):
        Eligibility.from_rolling(frame, window=2, agg="skew", op=">", value=0.0)


def test_rolling_rule_rejects_a_degenerate_window():
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=_dates(2))
    with pytest.raises(UniverseError, match="window"):
        Eligibility.from_rolling(frame, window=0, agg="mean", op=">", value=0.0)


# ---------------------------------------------------------------------------
# Hysteresis and reconstitution
# ---------------------------------------------------------------------------


def test_hysteresis_unknown_initial_propagates():
    """``initial=None`` means unknown, and unknown survives until evidence.

    Entry never fires and exit never fires, so nothing is ever learned: the
    membership stays missing for the whole history rather than defaulting to
    "out", which is a claim the data does not support.
    """
    entry = _signal([[F], [F], [F], [T]], ["A"])
    exits = _signal([[F], [F], [F], [F]], ["A"])
    unknown = Eligibility.with_hysteresis(entry, exits, initial=None)
    assert _states(unknown.frame, "A") == [None, None, None, True]

    out = Eligibility.with_hysteresis(entry, exits, initial=False)
    assert _states(out.frame, "A") == [False, False, False, True]


def test_hysteresis_holds_membership_until_exit_fires():
    entry = _signal([[T], [F], [F], [F]], ["A"])
    exits = _signal([[F], [F], [T], [F]], ["A"])
    held = Eligibility.with_hysteresis(entry, exits, initial=False)
    assert _states(held.frame, "A") == [True, True, False, False]


def test_hysteresis_unknown_exit_makes_membership_unknown():
    entry = _signal([[T], [F]], ["A"])
    exits = _signal([[F], [NA]], ["A"])
    held = Eligibility.with_hysteresis(entry, exits, initial=False)
    assert _states(held.frame, "A") == [True, None]


def test_hold_through_between_reconstitutions():
    """Membership is read on reconstitution dates and frozen in between."""
    dates = _dates(6)
    raw = Signal(
        pd.DataFrame(
            {"A": [F, T, F, F, T, F]}, index=dates, dtype="boolean"
        )
    )
    held = Eligibility.from_signal(raw, "raw").hold_through([dates[1], dates[4]])
    # Nothing before the first reconstitution has been evaluated at all.
    assert _states(held.frame, "A") == [None, True, True, True, True, True]


def test_hold_through_reads_the_last_row_on_or_before_a_reconstitution():
    dates = _dates(4)
    raw = Signal(
        pd.DataFrame({"A": [T, F, F, T]}, index=dates, dtype="boolean")
    )
    # A reconstitution stamped on a non-trading day reads the latest row before it.
    recon = [dates[1] + pd.Timedelta(hours=12)]
    held = Eligibility.from_signal(raw, "raw").hold_through(recon)
    assert _states(held.frame, "A") == [None, None, False, False]


# ---------------------------------------------------------------------------
# Collapsing to a hard mask
# ---------------------------------------------------------------------------


def test_to_mask_has_no_default_policy():
    """There is no safe collapse, so the signature must not offer one."""
    parameter = inspect.signature(Eligibility.to_mask).parameters["policy"]
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        Eligibility.from_signal(_signal([[T]], ["A"]), "r").to_mask()


def test_to_mask_policies_are_all_three_answers():
    elig = Eligibility.from_signal(_signal([[T], [F], [NA]], ["A"]), "r")
    assert list(elig.to_mask("exclude")["A"]) == [True, False, False]
    assert list(elig.to_mask("include")["A"]) == [True, False, True]
    with pytest.raises(UniverseError, match="not evaluable"):
        elig.to_mask("raise")
    with pytest.raises(UniverseError, match="policy"):
        elig.to_mask("whatever")


def test_to_mask_raise_passes_a_fully_evaluated_signal():
    elig = Eligibility.from_signal(_signal([[T], [F]], ["A"]), "r")
    assert list(elig.to_mask("raise")["A"]) == [True, False]


def test_breadth_counts_only_what_is_known():
    elig = Eligibility.from_signal(
        _signal([[T, T], [T, NA], [F, F]], ["A", "B"]), "r"
    )
    assert list(elig.breadth()) == [2, 1, 0]
    assert list(elig.unknown_count()) == [0, 1, 0]


def test_turnover_counts_entries_and_exits_and_ignores_unknowns():
    elig = Eligibility.from_signal(
        _signal([[F, T], [T, T], [T, NA], [F, F]], ["A", "B"]), "r"
    )
    frame = elig.turnover()
    assert list(frame["entries"]) == [0, 1, 0, 0]
    assert list(frame["exits"]) == [0, 0, 0, 1]
    assert list(frame["turnover"]) == [0, 1, 0, 1]


def test_explain_names_the_rule_that_excluded_the_name():
    liquidity = Eligibility.from_threshold(
        pd.DataFrame({"A": [10.0, 1.0]}, index=_dates(2)), ">", 5.0, name="liquidity"
    )
    size = Eligibility.from_threshold(
        pd.DataFrame({"A": [100.0, 100.0]}, index=_dates(2)), ">", 50.0, name="size"
    )
    combined = liquidity & size
    dates = combined.index
    excluded = combined.explain(dates[1], "A")
    assert "not eligible" in excluded
    assert "liquidity" in excluded
    assert "size" not in excluded
    admitted = combined.explain(dates[0], "A")
    assert "eligible" in admitted
    assert "liquidity" in admitted and "size" in admitted


def test_explain_reports_an_unevaluable_cell():
    rule = Eligibility.from_rolling(
        pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=_dates(3)),
        window=2,
        agg="mean",
        op=">",
        value=0.0,
    )
    text = rule.explain(rule.index[0], "A")
    assert "not evaluable" in text


def test_explain_refuses_an_unknown_asset():
    rule = Eligibility.from_signal(_signal([[T]], ["A"]), "r")
    with pytest.raises(UniverseError, match="ZZZ"):
        rule.explain(rule.index[0], "ZZZ")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classification_dated_requires_as_of():
    history = pd.DataFrame(
        {
            "asset": ["A", "A"],
            "label": ["Tech", "Industrials"],
            "effective_from": ["2020-01-01", "2022-06-30"],
        }
    )
    dated = Classification.from_history(history)
    with pytest.raises(UniverseError, match="as_of"):
        dated.label("A")
    assert dated.label("A", "2021-01-01") == "Tech"
    assert dated.label("A", "2023-01-01") == "Industrials"


def test_classification_before_first_record_is_none():
    history = pd.DataFrame(
        {"asset": ["A"], "label": ["Tech"], "effective_from": ["2020-01-01"]}
    )
    dated = Classification.from_history(history)
    assert dated.label("A", "2019-12-31") is None
    assert dated.label("UNKNOWN", "2021-01-01") is None


def test_classification_static_needs_no_date():
    static = Classification.static({"A": "Tech", "B": "Energy"})
    assert static.label("A") == "Tech"
    assert static.is_dated is False
    assert static.assignments() == {"A": "Tech", "B": "Energy"}


def test_classification_group_matrix_is_point_in_time():
    history = pd.DataFrame(
        {
            "asset": ["A", "A", "B"],
            "label": ["Tech", "Industrials", "Energy"],
            "effective_from": ["2020-01-01", "2022-06-30", "2021-01-01"],
        }
    )
    dated = Classification.from_history(history)
    early = dated.group_matrix("2021-06-30")
    # Only the buckets that existed on that date: Industrials had not happened.
    assert list(early.columns) == ["Energy", "Tech"]
    assert bool(early.loc["A", "Tech"]) is True
    assert bool(early.loc["B", "Energy"]) is True
    late = dated.group_matrix("2023-01-01")
    assert list(late.columns) == ["Energy", "Industrials"]
    assert bool(late.loc["A", "Industrials"]) is True
    # The 2019 view has no buckets at all: nothing had been classified yet.
    early_2019 = dated.group_matrix("2019-01-01")
    assert list(early_2019.columns) == []
    assert list(early_2019.index) == ["A", "B"]


def test_classification_from_history_refuses_a_missing_column():
    with pytest.raises(UniverseError, match="effective_from"):
        Classification.from_history(pd.DataFrame({"asset": ["A"], "label": ["T"]}))


def test_layer_from_classification_is_point_in_time():
    history = pd.DataFrame(
        {
            "asset": ["A", "A", "B"],
            "label": ["Tech", "Industrials", "Energy"],
            "effective_from": ["2020-01-01", "2022-06-30", "2021-01-01"],
        }
    )
    dated = Classification.from_history(history)
    early = ConstraintLayer.from_classification(
        dated, "2021-06-30", {"Tech": 0.4, "Energy": (0.0, 0.3)}
    )
    assert early.assignments == {"A": "Tech", "B": "Energy"}
    assert early.limits["Tech"] == (0.0, 0.4)
    late = ConstraintLayer.from_classification(
        dated, "2023-01-01", {"Industrials": 0.4}
    )
    assert late.assignments["A"] == "Industrials"


# ---------------------------------------------------------------------------
# Backtest integration
# ---------------------------------------------------------------------------


def _flat_returns(n: int, assets: list[str], seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0, 0.002, size=(n, len(assets))),
        index=_dates(n),
        columns=assets,
    )


def test_backtest_universe_policy_must_be_named():
    returns = _flat_returns(10, ["A", "B"])
    elig = Eligibility.from_signal(
        Signal(pd.DataFrame(True, index=returns.index, columns=returns.columns)),
        "all",
    )
    weights = pd.Series(0.5, index=returns.columns)
    with pytest.raises(ValueError, match="universe_policy"):
        run_backtest(returns, weights, universe=elig)


def test_backtest_excludes_ineligible_and_liquidates_next_bar():
    """A name that leaves the universe is sold on the next bar, and stays out.

    ``B`` is eligible until the fifth bar and ineligible from then on. The
    decision on that bar must carry a zero for ``B``; with a one-bar execution
    lag the sale prints on the bar after, and no later trade re-opens it.
    """
    assets = ["A", "B"]
    returns = _flat_returns(12, assets)
    membership = pd.DataFrame(
        {"A": [True] * 12, "B": [True] * 5 + [False] * 7},
        index=returns.index,
        dtype="boolean",
    )
    elig = Eligibility.from_signal(Signal(membership), "membership")
    targets = pd.DataFrame(
        0.5, index=returns.index[::2], columns=assets
    )
    spec = BacktestSpec(
        frequency="none", execution_lag=1, costs=CostSpec(commission_bps=0.0)
    )

    run = run_backtest(
        returns, targets, spec, universe=elig, universe_policy="exclude"
    )

    decision = returns.index[6]
    execution = returns.index[7]
    # ``targets`` is the schedule as handed in -- what was asked for. What the
    # universe did to it lives in the notes and in the realized weights.
    assert float(run.targets.loc[decision, "B"]) == pytest.approx(0.5)
    held_before = float(run.weights.loc[returns.index[6], "B"])
    assert held_before > 0.0
    # The book itself is flat in B from the execution bar on, and stays flat.
    assert float(run.weights.loc[execution:, "B"].abs().max()) == pytest.approx(0.0)

    liquidation = run.trades[
        (run.trades["asset"] == "B") & (run.trades["date"] == execution)
    ]
    assert len(liquidation) == 1
    assert liquidation["side"].iloc[0] == "sell"
    # The whole position goes, not a rebalancing slice of it: the book entered
    # this bar around 50% in B and leaves it at exactly zero.
    assert float(liquidation["traded_weight"].iloc[0]) < -0.49
    assert float(run.weights.loc[execution, "B"]) == 0.0
    later = run.trades[
        (run.trades["asset"] == "B") & (run.trades["date"] > execution)
    ]
    assert later.empty

    note = run.meta.notes["universe"]
    assert note["policy"] == "exclude"
    assert note["breadth"][pd.Timestamp(returns.index[0]).isoformat()] == 2
    assert note["breadth"][pd.Timestamp(decision).isoformat()] == 1
    assert note["liquidated"] == {"B": 3}
    # A's target is untouched: the runner never renormalises.
    assert float(run.targets.loc[decision, "A"]) == pytest.approx(0.5)


def test_backtest_unknown_assets_are_named_in_the_note():
    returns = _flat_returns(6, ["A", "B"])
    membership = pd.DataFrame(
        {"A": [True] * 6}, index=returns.index, dtype="boolean"
    )
    elig = Eligibility.from_signal(Signal(membership), "membership")
    run = run_backtest(
        returns,
        pd.Series(0.5, index=returns.columns),
        BacktestSpec(frequency="none"),
        universe=elig,
        universe_policy="exclude",
    )
    assert run.meta.notes["universe"]["unknown_assets"] == ["B"]


# ---------------------------------------------------------------------------
# Walk-forward integration
# ---------------------------------------------------------------------------


def _equal_weight(window: pd.DataFrame) -> pd.Series:
    if window.shape[1] == 0:
        raise ValueError("nothing to solve")
    return pd.Series(1.0 / window.shape[1], index=window.columns)


def test_walk_forward_universe_is_point_in_time():
    """A name eligible only from ``t`` is invisible to every earlier solve."""
    assets = ["A", "B", "C"]
    returns = _flat_returns(30, assets)
    seen: list[list[str]] = []

    def solve(window: pd.DataFrame) -> pd.Series:
        seen.append(list(window.columns))
        return _equal_weight(window)

    membership = pd.DataFrame(
        {
            "A": [True] * 30,
            "B": [True] * 30,
            "C": [False] * 20 + [True] * 10,
        },
        index=returns.index,
        dtype="boolean",
    )
    elig = Eligibility.from_signal(Signal(membership), "membership")

    wf = walk_forward_run(
        returns,
        solve,
        lookback=5,
        rebalance_every=5,
        spec=BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
        universe=elig,
        universe_policy="exclude",
    )

    listing = returns.index[20]
    for decision, columns in zip(wf.weights_history.index, seen):
        if decision < listing:
            assert "C" not in columns
        else:
            assert "C" in columns
    before = wf.weights_history.loc[wf.weights_history.index < listing, "C"]
    assert float(before.abs().max()) == 0.0
    after = wf.weights_history.loc[wf.weights_history.index >= listing, "C"]
    assert float(after.min()) > 0.0


def test_walk_forward_delisted_name_is_liquidated_and_never_traded_again():
    assets = ["A", "B"]
    returns = _flat_returns(30, assets)
    returns.loc[returns.index[18]:, "B"] = np.nan

    wf = walk_forward_run(
        returns,
        _equal_weight,
        lookback=5,
        rebalance_every=5,
        spec=BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
        delisting_grace=0,
    )

    delistings = wf.run.meta.notes["delistings"]
    assert set(delistings) == {"B"}
    assert delistings["B"]["last_print"] == pd.Timestamp(returns.index[17]).isoformat()
    delisted_at = pd.Timestamp(delistings["B"]["delisted_at"])
    assert delisted_at == returns.index[20]

    after = wf.weights_history.loc[wf.weights_history.index >= delisted_at, "B"]
    assert float(after.abs().max()) == 0.0
    trades = wf.run.trades
    later = trades[(trades["asset"] == "B") & (trades["date"] > delisted_at)]
    assert later.empty
    # The trade on the delisting decision is the whole position, sold at the
    # last mark: B has no print on that bar, so the loop replays it flat.
    liquidation = trades[(trades["asset"] == "B") & (trades["date"] == delisted_at)]
    assert len(liquidation) == 1
    assert liquidation["side"].iloc[0] == "sell"
    assert float(liquidation["traded_weight"].iloc[0]) < -0.4
    assert float(wf.run.weights.loc[delisted_at:, "B"].abs().max()) == 0.0


def test_delisting_is_sticky_once_declared():
    assets = ["A", "B"]
    returns = _flat_returns(40, assets)
    returns.loc[returns.index[18]:returns.index[26], "B"] = np.nan

    wf = walk_forward_run(
        returns,
        _equal_weight,
        lookback=5,
        rebalance_every=5,
        spec=BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
        delisting_grace=0,
    )
    delisted_at = pd.Timestamp(wf.run.meta.notes["delistings"]["B"]["delisted_at"])
    later = wf.weights_history.loc[wf.weights_history.index > delisted_at, "B"]
    assert float(later.abs().max()) == 0.0


def test_no_lookahead_in_delisting():
    """Shuffling everything after ``d`` cannot move a decision taken at ``d``."""
    assets = ["A", "B"]
    returns = _flat_returns(40, assets)
    returns.loc[returns.index[22]:, "B"] = np.nan

    cut = returns.index[20]
    shuffled = returns.copy()
    tail = shuffled.loc[shuffled.index > cut]
    rng = np.random.default_rng(99)
    permuted = tail.to_numpy()[rng.permutation(len(tail))]
    # Refill the delisted column too: the future must be unable to speak.
    permuted[:, 1] = np.linspace(0.01, 0.02, len(tail))
    shuffled.loc[shuffled.index > cut, :] = permuted

    kwargs = dict(
        lookback=5,
        rebalance_every=5,
        spec=BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
        delisting_grace=0,
    )
    base = walk_forward_run(returns, _equal_weight, **kwargs)
    other = walk_forward_run(shuffled, _equal_weight, **kwargs)

    left = base.weights_history.loc[:cut]
    right = other.weights_history.loc[:cut]
    pd.testing.assert_frame_equal(left, right)
    assert list(base.windows["status"])[: len(left)] == list(
        other.windows["status"]
    )[: len(left)]


def test_delisting_cannot_see_the_bar_its_decision_is_held_over():
    """A halt on the decision bar is news the decision did not have.

    The book chosen at ``t`` is held over ``t`` and its solve window ends on
    ``t - 1``. Measuring silence through ``t`` let ``delisting_grace=0`` sell
    X at its last mark on the very bar it halted, and so dodge the -50% it
    reopened at -- a sale nobody at the close of ``t - 1`` could have placed.
    """
    returns = pd.DataFrame(
        {"X": 0.001, "Y": 0.001}, index=_dates(40), dtype=float
    )
    halt = 20
    returns.iloc[halt, 0] = np.nan
    returns.iloc[halt + 1, 0] = -0.50

    def hold_x(window: pd.DataFrame) -> pd.Series:
        return pd.Series({"X": 1.0, "Y": 0.0}).reindex(window.columns).fillna(0.0)

    kwargs = dict(
        lookback=10,
        rebalance_every=10,
        spec=BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
    )
    honest = walk_forward_run(returns, hold_x, **kwargs)
    graced = walk_forward_run(returns, hold_x, delisting_grace=0, **kwargs)

    assert returns.index[halt] in graced.weights_history.index
    assert graced.metadata["delistings"] == {}
    assert float(graced.run.weights.loc[returns.index[halt + 1], "X"]) == pytest.approx(1.0)
    pd.testing.assert_series_equal(graced.returns, honest.returns)


def test_a_first_print_on_the_decision_bar_waits_for_the_next_decision():
    """The listing side of the same rule: a print on ``t`` is unknown at ``t``."""
    returns = _flat_returns(30, ["A", "B"])
    first_print = 10
    returns.loc[returns.index[:first_print], "B"] = np.nan
    seen: list[list[str]] = []

    def solve(window: pd.DataFrame) -> pd.Series:
        seen.append(list(window.columns))
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    walk_forward_run(
        returns,
        solve,
        lookback=5,
        rebalance_every=5,
        spec=BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
        delisting_grace=0,
    )
    # Decisions on rows 5, 10, 15, ...: B prints first on row 10, so the
    # decision held over row 10 cannot know it, and the next one can.
    assert seen[1] == ["A"]
    assert seen[2] == ["A", "B"]


def test_failed_solve_liquidates_the_carried_book_of_ineligible_names():
    """A solve that fails is not a licence to keep a name that left the universe."""
    assets = ["A", "B"]
    returns = _flat_returns(30, assets)
    calls = {"n": 0}

    def solve(window: pd.DataFrame) -> pd.Series:
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("no solution today")
        return _equal_weight(window)

    membership = pd.DataFrame(
        {"A": [True] * 30, "B": [True] * 15 + [False] * 15},
        index=returns.index,
        dtype="boolean",
    )
    elig = Eligibility.from_signal(Signal(membership), "membership")

    wf = walk_forward_run(
        returns,
        solve,
        lookback=5,
        rebalance_every=5,
        spec=BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
        universe=elig,
        universe_policy="exclude",
    )
    assert wf.n_failures > 0
    after = wf.weights_history.loc[wf.weights_history.index >= returns.index[15], "B"]
    assert float(after.abs().max()) == 0.0
    # A is carried forward untouched -- only the ineligible leg is sold.
    carried = wf.weights_history.loc[wf.weights_history.index >= returns.index[15], "A"]
    assert float(carried.min()) > 0.0
    assert wf.metadata["n_ineligible_carried_forward"] > 0


def test_walk_forward_records_an_empty_universe_as_a_failure():
    returns = _flat_returns(20, ["A", "B"])
    membership = pd.DataFrame(
        False, index=returns.index, columns=returns.columns, dtype="boolean"
    )
    membership.iloc[:6] = True
    elig = Eligibility.from_signal(Signal(membership), "membership")
    wf = walk_forward_run(
        returns,
        _equal_weight,
        lookback=5,
        rebalance_every=5,
        spec=BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
        universe=elig,
        universe_policy="exclude",
    )
    assert any("eligible" in reason for reason in wf.failures)


# ---------------------------------------------------------------------------
# The shared point-in-time helper
# ---------------------------------------------------------------------------


def test_point_in_time_mask_never_reads_forward():
    elig = Eligibility.from_signal(_signal([[F], [T]], ["A"]), "r")
    dates = pd.DatetimeIndex(
        [elig.index[0] - pd.Timedelta(days=1), elig.index[0], elig.index[1]]
    )
    mask = point_in_time_mask(elig, "exclude", dates, ["A"])
    assert list(mask["A"]) == [False, False, True]
    included = point_in_time_mask(elig, "include", dates, ["A"])
    assert list(included["A"]) == [True, False, True]
    with pytest.raises(UniverseError, match="not evaluable"):
        point_in_time_mask(elig, "raise", dates, ["A"])


# ---------------------------------------------------------------------------
# Same-bar rules: a verdict that reads its own date cannot trade that date
# ---------------------------------------------------------------------------

#: Row of the panel on which ``C`` loses 60% in a session.
CRASH = 15


def _crash_panel() -> pd.DataFrame:
    returns = pd.DataFrame(
        {"A": 0.001, "B": 0.001, "C": 0.001}, index=_dates(30), dtype=float
    )
    returns.iloc[CRASH, 2] = -0.60
    return returns


def _not_in_freefall(returns: pd.DataFrame) -> Eligibility:
    """The example rule from the rules-file docstring, on the run's returns."""
    return Eligibility.from_threshold(returns, ">", -0.5, name="not in freefall")


def test_a_same_bar_screen_cannot_see_the_bar_its_book_earns():
    """The headline look-ahead: with no lag the book chosen at ``t`` earns ``t``.

    A threshold on returns judges ``t`` on ``t``'s own return. Read as of
    ``t`` by a decision whose book is held over ``t``, it dropped every name
    on exactly the session it crashed. The screen may only act from the next
    decision, so ``C`` is held through its crash like it would be with no
    universe at all, and is dropped on the bar after.
    """
    returns = _crash_panel()
    kwargs = dict(
        lookback=5,
        rebalance_every=1,
        spec=BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
    )
    plain = walk_forward_run(returns, _equal_weight, **kwargs)
    screened = walk_forward_run(
        returns,
        _equal_weight,
        universe=_not_in_freefall(returns),
        universe_policy="exclude",
        **kwargs,
    )
    crash = returns.index[CRASH]
    held = float(screened.run.weights.loc[crash, "C"])
    assert held == pytest.approx(float(plain.run.weights.loc[crash, "C"]))
    assert held == pytest.approx(1.0 / 3.0)
    assert float(screened.weights_history.loc[returns.index[CRASH + 1], "C"]) == 0.0
    assert float(screened.returns.loc[crash]) == pytest.approx(
        float(plain.returns.loc[crash])
    )


def test_the_runner_reads_a_same_bar_screen_before_the_bar_at_lag_zero():
    """Same rule through ``run_backtest``: the replay is where the mask is read."""
    returns = _crash_panel()
    targets = pd.DataFrame(1.0 / 3.0, index=returns.index, columns=returns.columns)
    run = run_backtest(
        returns,
        targets,
        BacktestSpec(frequency="none", costs=CostSpec(commission_bps=0.0)),
        universe=_not_in_freefall(returns),
        universe_policy="exclude",
    )
    assert float(run.weights.loc[returns.index[CRASH], "C"]) == pytest.approx(1.0 / 3.0)
    assert float(run.weights.loc[returns.index[CRASH + 1], "C"]) == 0.0
    # The breadth note says what the decision on the crash bar could see.
    breadth = run.meta.notes["universe"]["breadth"]
    assert breadth[pd.Timestamp(returns.index[CRASH]).isoformat()] == 3
    assert breadth[pd.Timestamp(returns.index[CRASH + 1]).isoformat()] == 2


def test_a_lagged_run_still_reads_a_same_bar_screen_on_its_decision_date():
    """With one bar of lag the decision on ``t`` trades ``t + 1``: nothing moves."""
    returns = _crash_panel()
    targets = pd.DataFrame(1.0 / 3.0, index=returns.index, columns=returns.columns)
    run = run_backtest(
        returns,
        targets,
        BacktestSpec(
            frequency="none", execution_lag=1, costs=CostSpec(commission_bps=0.0)
        ),
        universe=_not_in_freefall(returns),
        universe_policy="exclude",
    )
    breadth = run.meta.notes["universe"]["breadth"]
    # The decision on the crash bar is the one that sees the crash ...
    assert breadth[pd.Timestamp(returns.index[CRASH]).isoformat()] == 2
    # ... and its sale fills on the next bar, after the crash was earned.
    assert float(run.weights.loc[returns.index[CRASH], "C"]) > 0.3
    assert float(run.weights.loc[returns.index[CRASH + 1], "C"]) == 0.0


def test_a_rolling_screen_is_read_on_its_own_date_at_lag_zero():
    """Rolling rules already end the bar before, so lag 0 reads them unchanged."""
    returns = _crash_panel()
    prior = Eligibility.from_rolling(returns, window=1, agg="min", op=">", value=-0.5)
    assert prior.same_bar is False
    mask = point_in_time_mask(
        prior, "exclude", returns.index, list(returns.columns), execution_lag=0
    )
    pd.testing.assert_frame_equal(mask, prior.to_mask("exclude"))


def test_same_bar_is_inherited_through_every_combinator():
    returns = _crash_panel()
    same = _not_in_freefall(returns)
    prior = Eligibility.from_rolling(returns, window=2, agg="mean", op=">", value=-1.0)
    ranked = Eligibility.from_rank(returns, 2)
    assert same.same_bar is True and ranked.same_bar is True
    assert prior.same_bar is False
    assert (prior & same).same_bar is True
    assert (prior | prior).same_bar is False
    assert (~same).same_bar is True
    assert same.hold_through([returns.index[3]]).same_bar is True
    assert Eligibility.with_hysteresis(prior, same, None).same_bar is True
    assert Eligibility.from_signal(same, "wrapped").same_bar is True
    assert Eligibility.from_signal(same.signal, "a bare signal").same_bar is False


# ---------------------------------------------------------------------------
# The rest of the public surface
# ---------------------------------------------------------------------------


def test_signal_constant_and_shape_helpers():
    dates = _dates(3)
    unknown = Signal.constant(None, dates, ["A", "B"])
    assert unknown.shape == (3, 2)
    assert len(unknown) == 3
    assert unknown.is_missing().to_numpy().all()
    assert "unknown=6" in repr(unknown)
    everything = Signal.constant(True, dates, ["A", "B"])
    assert everything.at(dates[0], "A") is True
    assert unknown.at(dates[0], "A") is None


def test_signal_at_refuses_a_date_or_asset_it_does_not_carry():
    sig = _signal([[T]], ["A"])
    with pytest.raises(UniverseError, match="not a date"):
        sig.at(sig.index[0] + pd.Timedelta(days=1), "A")
    with pytest.raises(UniverseError, match="not an asset"):
        sig.at(sig.index[0], "B")


def test_signal_reindex_leaves_new_cells_unknown_and_equals_compares_states():
    sig = _signal([[T], [F]], ["A"])
    wider = sig.reindex(assets=["A", "B"])
    assert _states(wider.frame, "B") == [None, None]
    assert wider.equals(wider) is True
    assert wider.equals(sig) is False


def test_signal_refuses_a_repeated_date():
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    with pytest.raises(UniverseError, match="same date twice"):
        Signal(pd.DataFrame({"A": [True, False]}, index=dates))


def test_signal_sorts_an_unordered_index():
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-01"])
    sig = Signal(pd.DataFrame({"A": [True, False]}, index=dates))
    assert list(sig.index) == list(pd.DatetimeIndex(["2024-01-01", "2024-01-02"]))
    assert _states(sig.frame, "A") == [False, True]


def test_signal_refuses_a_non_frame():
    with pytest.raises(UniverseError, match="DataFrame"):
        Signal(pd.Series([True]))


def test_eligibility_or_and_not_carry_their_provenance():
    cheap = Eligibility.from_threshold(
        pd.DataFrame({"A": [1.0]}, index=_dates(1)), "<", 5.0, name="cheap"
    )
    large = Eligibility.from_threshold(
        pd.DataFrame({"A": [1.0]}, index=_dates(1)), ">", 5.0, name="large"
    )
    either = cheap | large
    assert "cheap" in either.description and "large" in either.description
    assert bool(either.frame.iat[0, 0]) is True
    assert "cheap" in either.explain(either.index[0], "A")
    assert bool((~cheap).frame.iat[0, 0]) is False
    assert "not (" in (~cheap).description
    assert "Eligibility(" in repr(cheap)


def test_eligibility_as_of_reads_the_row_in_force():
    elig = Eligibility.from_signal(_signal([[T], [F]], ["A"]), "r")
    assert bool(elig.as_of(elig.index[1])["A"]) is False


def test_eligibility_rejects_a_non_numeric_characteristic():
    frame = pd.DataFrame({"A": ["large"]}, index=_dates(1))
    with pytest.raises(UniverseError, match="numeric"):
        Eligibility.from_threshold(frame, ">", 1.0)


def test_eligibility_from_rank_refuses_a_degenerate_n():
    frame = pd.DataFrame({"A": [1.0]}, index=_dates(1))
    with pytest.raises(UniverseError, match="top_n"):
        Eligibility.from_rank(frame, 0)


def test_classification_describe_and_labels():
    static = Classification.static({"A": "Tech", "B": "Energy"}, name="GICS sector")
    assert static.describe() == "GICS sector: 2 assets, static"
    assert static.labels() == ["Energy", "Tech"]
    assert static.assets == ["A", "B"]
    assert "GICS sector" in repr(static)


def test_classification_static_drops_blank_labels():
    static = Classification.static({"A": "Tech", "B": "", "C": None})
    assert static.assignments() == {"A": "Tech"}


def test_classification_refuses_a_mixed_dated_and_undated_asset():
    from optimization_engine.universe import LabelRecord

    with pytest.raises(UniverseError, match="both dated and undated"):
        Classification(
            {"A": [LabelRecord("Tech"), LabelRecord("Energy", pd.Timestamp("2020-01-01"))]}
        )


def test_classification_refuses_an_undated_label_change():
    frame = pd.DataFrame(
        {"asset": ["A"], "label": ["Tech"], "effective_from": [None]}
    )
    with pytest.raises(UniverseError, match="cannot be placed in time"):
        Classification.from_history(frame)


def test_walk_forward_refuses_a_negative_delisting_grace():
    returns = _flat_returns(20, ["A", "B"])
    with pytest.raises(ValueError, match="delisting_grace"):
        walk_forward_run(
            returns,
            _equal_weight,
            lookback=5,
            rebalance_every=5,
            delisting_grace=-1,
        )


def test_a_policy_with_no_universe_says_so(caplog):
    returns = _flat_returns(6, ["A", "B"])
    with caplog.at_level("WARNING"):
        run = run_backtest(
            returns,
            pd.Series(0.5, index=returns.columns),
            BacktestSpec(frequency="none"),
            universe_policy="exclude",
        )
    assert "decides nothing" in caplog.text
    assert "universe" not in run.meta.notes
