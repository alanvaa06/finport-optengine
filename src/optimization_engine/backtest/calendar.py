"""When the book trades: rebalance marks and execution dates.

Two distinct calendars, and conflating them is the classic way a backtest
buys at a price it could not have known. The *decision* calendar is when a
target is chosen; the *execution* calendar is when it is traded. They differ
by the execution lag, which is zero only in a simulation. A target traded
at position ``p`` is held from bar ``p`` on and earns that bar's return, so
at zero lag the decision date is itself the first holding bar.
"""

from __future__ import annotations

import pandas as pd

from optimization_engine.backtest.spec import FREQUENCY_ALIASES, RebalanceFrequency


def rebalance_dates(
    index: pd.DatetimeIndex, frequency: RebalanceFrequency
) -> pd.DatetimeIndex:
    """Dates on which a target is chosen at the given cadence.

    The first date is always included: that is the initial purchase, which
    happens whatever the cadence.

    Args:
        index: The simulation's date index.
        frequency: The cadence, e.g. ``"monthly"`` or ``"none"``.

    Returns:
        The subset of ``index`` on which a target is set.

    Raises:
        ValueError: If the frequency is not one the runner understands.
    """
    if len(index) == 0:
        return pd.DatetimeIndex([])
    if frequency == "none":
        return pd.DatetimeIndex([index[0]])
    if frequency == "daily":
        return pd.DatetimeIndex(index)
    if frequency not in FREQUENCY_ALIASES:
        raise ValueError(
            f"Unknown rebalance frequency {frequency!r}. "
            f"Available: {sorted(FREQUENCY_ALIASES)}"
        )
    alias = FREQUENCY_ALIASES[frequency]
    marks = pd.Series(index, index=index).resample(alias).last().dropna()
    dates = pd.DatetimeIndex(marks.values)
    if index[0] not in dates:
        dates = pd.DatetimeIndex([index[0]]).append(dates)
    return dates.unique().sort_values()


def execution_positions(
    index: pd.DatetimeIndex,
    decision_dates: pd.DatetimeIndex,
    execution_lag: int,
) -> dict[int, int]:
    """Map each decision to the position at which it is actually traded.

    A decision taken on a date that leaves fewer than ``execution_lag``
    periods of history behind it is dropped, not clamped to the last date:
    an order that could not be filled inside the sample was not filled.

    Args:
        index: The simulation's date index.
        decision_dates: The dates on which targets are chosen.
        execution_lag: How many periods pass between deciding and trading.
            ``1`` is the desk's default for a target decided on its date's
            close — you do not trade on a close you have not seen. A
            walk-forward schedule is already dated by its first holding bar,
            so ``0`` is exact for it and ``1`` adds a bar.

    Returns:
        ``{decision position: execution position}``, both indexing ``index``.
    """
    positions: dict[int, int] = {}
    lookup = {date: pos for pos, date in enumerate(index)}
    for date in decision_dates:
        decision_pos = lookup.get(date)
        if decision_pos is None:
            continue
        execution_pos = decision_pos + int(execution_lag)
        if execution_pos < len(index):
            positions[decision_pos] = execution_pos
    return positions


__all__ = ["execution_positions", "rebalance_dates"]
