"""Walk-forward evaluation: re-estimate, re-solve, hold forward, repeat.

An in-sample replay tells you how a weight vector would have done on the
returns it was fitted to. That is a description of the past, not a track
record. A walk-forward run evaluates the *process*: at each decision the
optimizer sees only returns strictly before the decision date, and the
weights it produces are held over the periods that follow. Nothing from the
evaluation window reaches the estimate.

**Two cadences, and they answer different questions.** ``rebalance_every`` is
how often the optimizer *re-solves* — how stale the desk is willing to let its
view get. ``rebalance_frequency`` is how often the book is *traded back* to
whichever target is current — how far the desk is willing to let the weights
drift away from that view. Real mandates separate them: a quarterly investment
committee running a monthly rebalancing discipline re-solves four times a year
and trades twelve. Collapsing the two ties the drift tolerance to the research
calendar for no reason other than that it was easier to implement, and it
understates the turnover of every policy that rebalances more often than it
re-optimizes.

The default is ``"none"``: trade only when a new target is solved. That is the
minimal reading of "hold the solution forward", and it is what this runner has
always done — a policy has to ask for the extra trading before it pays for it.

The runner is deliberately honest about failure. When a solve raises — an
infeasible constraint set on one window, a covariance that will not invert —
the previous book is carried forward, which is what a desk would actually do,
and the failure is recorded. Silently skipping the period would remove a real
cost from the track record and make a fragile process look robust.

When the failure is the *first* solve there is no previous book to carry, and
the answer is cash, not a shorter window. Starting the evaluation at the first
success would quietly delete every period the process could not trade —
exactly the periods a failure-prone process gets wrong — and hand back a track
record that begins on the strategy's first good day. The window always opens
at ``min_lookback``, the cash periods are in it at a zero return, and
``notes["periods_in_cash_after_failed_solve"]`` counts them.

**The universe, when there is one.** ``universe`` restricts what the optimizer
is even shown: the solve window's *columns* are masked before ``solve`` sees
it, which is the only way to do it — ``solve`` is a
``Callable[[DataFrame], Series]`` and widening that signature to carry
eligibility would break every closure already written against it. The reindex
that follows the solve zero-fills the names that were held back, so a book
never carries a weight in a name the optimizer was not allowed to look at.

Two decisions the runner has to take, and takes out loud:

*A failed solve does not suspend the mandate.* When the optimizer raises and
the previous book is carried forward, any name in it that has since left the
universe is still zeroed. Not being able to re-optimize is not a licence to
keep holding something the mandate no longer permits, and a desk in that
position sells the ineligible leg and leaves the proceeds in cash rather than
rescaling the rest. ``notes["n_ineligible_carried_forward"]`` counts it.

*Delisting is a claim about data, so it has to be quantified.* A name that has
stopped printing looks exactly like one on a long holiday until enough silence
has passed, and only the caller knows how much is enough — so
``delisting_grace`` has no default and delisting is simply not diagnosed
without one. When it is set, staleness is measured on ``returns`` strictly
**before** the decision date — the rows the solve window ends on — because the
book chosen on that date is held over it, and a halt on that very bar is news
the decision did not have. The verdict is sticky: a name declared delisted
stays out for the rest of the run rather than resurrecting on a late print.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from optimization_engine.backtest.results import RunResult
from optimization_engine.backtest.runner import resolve_universe_mask, run_backtest
from optimization_engine.backtest.spec import BacktestSpec, RebalanceFrequency
from optimization_engine.universe import Eligibility


@dataclass
class WalkForwardRun:
    """The out-of-sample track record of an optimization process.

    Attributes:
        run: The replayed result, tagged out-of-sample.
        weights_history: Target weights by decision date — one row per
            decision, not per trade, and a failed solve is a row like any
            other: the book it carries forward, or cash when there is none
            yet. With a trading cadence finer than the re-solve cadence the
            book trades more often than this frame has rows, and
            ``run.rebalance_dates`` is the record of that.
        windows: One row per decision — window bounds, length, and status.
            ``status`` is ``"ok"`` or starts with ``"failed: "`` and carries
            the exception text, so match it with ``.startswith``.
        failures: Human-readable reasons for each failed solve.
    """

    run: RunResult
    weights_history: pd.DataFrame
    windows: pd.DataFrame
    failures: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        """The net-of-cost return stream, taken from the underlying run."""
        return self.run.returns

    @property
    def n_resolves(self) -> int:
        """How many times the optimizer actually re-solved."""
        return len(self.weights_history)

    @property
    def n_rebalances(self) -> int:
        """Deprecated spelling of :attr:`n_resolves`.

        Kept because callers and saved notebooks use it, but the name is now
        ambiguous: with the two cadences separated, the book can rebalance
        without the optimizer re-solving. Count trades with
        :attr:`n_trade_dates`.
        """
        return self.n_resolves

    @property
    def n_trade_dates(self) -> int:
        """How many dates the book actually traded on.

        Equal to :attr:`n_resolves` when the trading cadence is ``"none"``,
        and larger when a calendar rebalance sits between re-solves.
        """
        return len(self.run.rebalance_dates)

    @property
    def n_failures(self) -> int:
        """How many windows failed to solve.

        A non-zero count is not fatal — the run carries on with the last usable
        weights — but it belongs in any read of the result, which is why
        :meth:`describe` reports it.
        """
        return len(self.failures)

    @property
    def rebalance_frequency(self) -> str:
        """The trading calendar this run was actually evaluated on."""
        return str(self.metadata.get("rebalance_frequency", "none"))

    def weight_stability(self) -> pd.Series:
        """Average absolute change in each asset's weight between re-solves.

        High values mean the optimizer is chasing estimation noise: the
        allocation is being rewritten every window on data that barely moved.
        This reads the solved targets, so it measures the optimizer, not the
        drift that the trading cadence corrects for.
        """
        if len(self.weights_history) < 2:
            return pd.Series(dtype=float)
        return self.weights_history.diff().abs().mean()

    def describe(self) -> str:
        """One line: re-solves, trade dates, trading cadence, failed solves.

        Returns:
            Something like ``"24 re-solves, 24 trade dates (trading only on
            re-solves); 0 failed solve(s)."``
        """
        trading = (
            "trading only on re-solves"
            if self.rebalance_frequency == "none"
            else f"{self.rebalance_frequency} rebalancing between re-solves"
        )
        return (
            f"{self.n_resolves} re-solves, {self.n_trade_dates} trade dates "
            f"({trading}); {self.n_failures} failed solve(s)."
        )


def _resolve_trading_cadence(
    spec: BacktestSpec | None, rebalance_frequency: RebalanceFrequency | None
) -> RebalanceFrequency:
    """Decide which calendar the book trades on, without overriding anyone.

    The old runner forced ``frequency="none"`` onto whatever spec it was
    handed, so a caller who asked for monthly rebalancing got buy-and-hold
    between re-solves and no signal that their request had been dropped. The
    rules here are the smallest set that never does that:

    * an explicit ``rebalance_frequency`` always wins;
    * otherwise a caller-supplied spec means what it says;
    * otherwise — no spec, no argument — trade only on re-solves.
    """
    if rebalance_frequency is not None:
        return rebalance_frequency
    if spec is not None:
        return spec.frequency
    return "none"


def _last_observation_positions(returns: pd.DataFrame) -> np.ndarray:
    """For each bar and asset, the position of its most recent print at or before it.

    Args:
        returns: The full return history.

    Returns:
        An integer ``(periods, assets)`` array. Entry ``[t, i]`` is the row
        position of asset ``i``'s latest non-missing return on or before bar
        ``t``, or ``-1`` when it has not printed at all yet. Reading only row
        ``t`` is what makes the staleness test free of look-ahead: nothing
        past ``t`` has been consulted to build it.
    """
    valid = returns.notna().to_numpy()
    positions = np.arange(len(returns.index), dtype=int)[:, None]
    seen = np.where(valid, positions, -1)
    return np.maximum.accumulate(seen, axis=0)


def walk_forward_run(
    returns: pd.DataFrame,
    solve: Callable[[pd.DataFrame], pd.Series],
    *,
    lookback: int,
    rebalance_every: int,
    spec: BacktestSpec | None = None,
    min_lookback: int | None = None,
    expanding: bool = False,
    rebalance_frequency: RebalanceFrequency | None = None,
    prices: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    universe: Eligibility | None = None,
    universe_policy: str | None = None,
    delisting_grace: int | None = None,
) -> WalkForwardRun:
    """Roll an estimation window forward, re-solving as it goes.

    Args:
        returns: Full periodic return history.
        solve: Callable taking a returns window and returning target weights.
        lookback: Estimation window length in periods.
        rebalance_every: Periods between **re-solves** — the re-optimization
            cadence. This is the one that decides how often the optimizer
            sees new data; it does not, on its own, decide how often the book
            trades.
        spec: Run description. ``is_out_of_sample`` is forced on, because it
            is. Its ``frequency`` is read as the trading cadence unless
            ``rebalance_frequency`` overrides it.
        min_lookback: Minimum window before the first solve. Defaults to
            ``lookback``.
        expanding: Grow the window from the start instead of rolling it.
        rebalance_frequency: How often the book is **traded back** to the
            most recent solved target between re-solves — the rebalancing
            cadence. ``"none"`` (the default when neither this nor a spec is
            given) trades only when a new target is solved and lets the
            weights drift in between. Any other cadence adds calendar trades
            on top of the re-solves; the target they trade to is always the
            freshest one available on that date, never a future one.
        prices: Close prices, needed only when the cost model prices capacity
            from traded volume.
        volumes: Traded volume per asset and period. Optional: without one the
            impact model uses its fixed participation rate, which is the only
            thing it can do for an index universe.
        universe: Point-in-time membership. The solve window's columns are
            restricted to the names eligible **as of the decision date**
            before ``solve`` is called, so the optimizer is never shown a name
            the mandate did not permit it to hold — and a name that becomes
            eligible at ``t`` is absent from every solve before ``t``. The
            same universe is handed to the replay, so calendar rebalances
            between re-solves respect it too, and ``run.meta.notes["universe"]``
            records the breadth at each decision.
        universe_policy: How a *not evaluable* cell is read — ``"exclude"``,
            ``"include"`` or ``"raise"``. Required whenever ``universe`` is
            given; there is no default.
        delisting_grace: How many bars of silence make a name delisted, or
            ``None`` (the default) to not diagnose delisting at all. With
            ``0``, a name that did not print on the bar before the decision
            is gone; with ``5``, a business week of silence is tolerated
            first. Staleness is measured on ``returns`` strictly before the
            decision date, like the solve window: the decision's book is held
            over the decision date, so that bar's print — or its absence —
            is not yet known. A delisted name is dropped from the
            solve window and its target forced to zero, which the replay
            executes as a sale at its last mark — see
            :func:`~optimization_engine.backtest.runner.run_backtest`. The
            verdict is sticky, and ``notes["delistings"]`` records the last
            print and the decision that liquidated it. A name that has not
            printed *at all* before a decision is likewise not investable at
            it, but it is not a delisting and not sticky: it enters on the
            first decision after its first print.

    Returns:
        The bundle. ``n_resolves`` counts optimizations, ``n_trade_dates``
        counts trades, and the two differ exactly when the cadences do.

    Raises:
        ValueError: If the parameters are degenerate, the history is too
            short to produce a single out-of-sample period, ``universe`` was
            given with no ``universe_policy``, or every solve failed.
        UniverseError: If the universe policy is unknown, or it is ``"raise"``
            and some name was not evaluable on some bar.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be at least 2 periods; got {lookback}.")
    if rebalance_every < 1:
        raise ValueError(
            f"rebalance_every must be at least 1 period; got {rebalance_every}."
        )
    min_lookback = min_lookback or lookback
    n = len(returns)
    if n <= min_lookback:
        raise ValueError(
            f"Need more than {min_lookback} observations to evaluate anything "
            f"out of sample; got {n}. Shorten the lookback or load more history."
        )

    trading = _resolve_trading_cadence(spec, rebalance_frequency)
    base = spec or BacktestSpec()
    spec = base.with_(frequency=trading, is_out_of_sample=True)

    schedule: dict[pd.Timestamp, pd.Series] = {}
    window_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    last_weights: pd.Series | None = None
    first_solved_position: int | None = None

    universe_mask: np.ndarray | None = None
    if universe is not None:
        universe_mask, _ = resolve_universe_mask(
            universe, universe_policy, pd.DatetimeIndex(returns.index), list(returns.columns)
        )
    last_seen: np.ndarray | None = None
    grace = 0
    if delisting_grace is not None:
        grace = int(delisting_grace)
        if grace < 0:
            raise ValueError(
                f"delisting_grace counts bars of silence and cannot be negative; got {grace}."
            )
        last_seen = _last_observation_positions(returns)
    delisted_ever = np.zeros(returns.shape[1], dtype=bool)
    delistings: dict[str, dict[str, str]] = {}
    n_ineligible_carried_forward = 0

    for position in range(min_lookback, n, rebalance_every):
        start = 0 if expanding else max(0, position - lookback)
        decision_date = returns.index[position]

        investable = np.ones(returns.shape[1], dtype=bool)
        if universe_mask is not None:
            investable &= universe_mask[position]
        if last_seen is not None:
            # The book chosen here is held over ``position`` itself, so what
            # the decision can know ends on the bar before — the same row the
            # solve window ends on. Reading ``position`` would let a halt on
            # this very bar sell the name at its last mark before anyone
            # could have known it halted.
            seen = last_seen[position - 1]
            never_printed = seen < 0
            stale = (~never_printed) & ((position - 1 - seen) > grace)
            for column_position in np.flatnonzero(stale & ~delisted_ever):
                delistings[str(returns.columns[column_position])] = {
                    "last_print": pd.Timestamp(
                        returns.index[int(seen[column_position])]
                    ).isoformat(),
                    "delisted_at": pd.Timestamp(decision_date).isoformat(),
                }
            delisted_ever |= stale
            investable &= ~(delisted_ever | never_printed)
        eligible = pd.Series(investable, index=returns.columns)

        try:
            if not investable.any():
                raise ValueError(
                    "no asset is eligible in the universe on this date, so "
                    "there is nothing to solve"
                )
            window = returns.iloc[start:position]
            if not investable.all():
                # Only pay for the column slice when something is masked, so a
                # run with no universe is byte-for-byte the run it always was.
                window = window.loc[:, eligible.to_numpy()]
            solved = (
                solve(window)
                .reindex(returns.columns)
                .fillna(0.0)
                .where(eligible, 0.0)
            )
            schedule[decision_date] = solved
            last_weights = solved
            if first_solved_position is None:
                first_solved_position = position
            status = "ok"
        except Exception as exc:  # noqa: BLE001 — a failed solve is a row, never a drop
            failures.append(f"{pd.Timestamp(decision_date).date()}: {exc}")
            status = f"failed: {exc}"
            # No prior book to carry forward means the desk holds cash. It does
            # not mean the track record starts later: a shortened window would
            # hide precisely the periods this process could not trade.
            carried = (
                last_weights
                if last_weights is not None
                else pd.Series(0.0, index=returns.columns)
            )
            # Failing to re-optimize is not a licence to keep a name the
            # mandate no longer permits. The ineligible leg is sold and the
            # proceeds sit in cash; the rest of the book is untouched, and
            # nothing is rescaled to hide the gap.
            n_ineligible_carried_forward += int(
                ((carried.abs() > 0.0) & ~eligible).sum()
            )
            schedule[decision_date] = carried.where(eligible, 0.0)
        window_rows.append(
            {
                "decision_date": decision_date,
                "window_start": returns.index[start],
                "window_end": returns.index[position - 1],
                "window_length": position - start,
                "n_eligible": int(investable.sum()),
                "status": status,
            }
        )

    if first_solved_position is None:
        # Every window failed. A book of nothing but cash is not a track record
        # of this process; it is a track record of the process never running.
        raise ValueError(
            "Every walk-forward solve failed; there is nothing to evaluate. "
            f"First error: {failures[0] if failures else 'unknown'}"
        )

    weights_history = pd.DataFrame(schedule).T.sort_index()
    # The evaluation always opens at the first decision date, whether or not
    # that decision produced weights. See the module docstring.
    evaluation = returns.loc[returns.index[min_lookback]:]
    periods_in_cash = int(first_solved_position - min_lookback)
    notes = {
        "lookback": int(lookback),
        "rebalance_every": int(rebalance_every),
        "rebalance_frequency": str(trading),
        "expanding": bool(expanding),
        "n_resolves": int(len(weights_history)),
        # The old key, kept so saved runs and reports keep reading.
        "n_rebalances": int(len(weights_history)),
        "n_failed_solves": int(len(failures)),
        # Evaluated periods the book held cash because the opening solves
        # failed before there was any book to carry forward.
        "periods_in_cash_after_failed_solve": periods_in_cash,
    }
    if universe is not None or delisting_grace is not None:
        # (asset, decision) pairs where a carried-forward book still held a
        # name the universe had dropped, and the runner sold it anyway.
        notes["n_ineligible_carried_forward"] = int(n_ineligible_carried_forward)
    if universe is not None:
        notes["universe_policy"] = str(universe_policy)
    if delisting_grace is not None:
        notes["delisting_grace"] = grace
        notes["delistings"] = delistings
    # The cost models see the whole history, not just the evaluated slice:
    # a decision made at the first evaluated date has years of returns behind
    # it, and pricing its impact off the slice alone would throw them away.
    run = run_backtest(
        evaluation,
        weights_history,
        spec,
        notes=notes,
        context_returns=returns,
        prices=prices,
        volumes=volumes,
        universe=universe,
        universe_policy=universe_policy if universe is not None else None,
    )

    return WalkForwardRun(
        run=run,
        weights_history=weights_history,
        windows=pd.DataFrame(window_rows),
        failures=tuple(failures),
        metadata=dict(notes),
    )


__all__ = ["WalkForwardRun", "walk_forward_run"]
