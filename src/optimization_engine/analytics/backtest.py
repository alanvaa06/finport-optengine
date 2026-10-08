"""Backtesting: weight drift, rebalancing, transaction costs, and walk-forward.

The default "backtest" in most optimizer demos applies the solved weights to
the same return history the optimizer was fitted on, and holds those weights
constant forever. Both halves of that are wrong in ways that flatter the
result:

* **Look-ahead.** The covariance and expected returns were estimated on the
  very returns being replayed. The optimizer already knew which assets won.
* **Costless continuous rebalancing.** Holding constant weights without
  saying so assumes the book is rebalanced every period, for free. Real
  weights drift with performance, and pulling them back costs money.

This module provides both an honest in-sample replay (:func:`backtest_weights`,
with explicit drift, rebalancing and costs) and a genuinely out-of-sample
:func:`walk_forward_backtest` that re-estimates and re-solves on a rolling
window and only ever holds positions forward in time.

Both are adapters. The simulation itself lives in
:mod:`optimization_engine.backtest`, whose stateless core also carries the
cost models, the execution calendar, the per-trade and per-date cost frames,
and the provenance hashes. What survives here is the compact result shape
most callers want — a return series, turnover, and costs — plus a
``.run`` handle back to the full bundle for the ones that want more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from optimization_engine.backtest.calendar import rebalance_dates
from optimization_engine.backtest.costs import CostModel
from optimization_engine.backtest.results import RunResult
from optimization_engine.backtest.runner import run_backtest
from optimization_engine.backtest.spec import (
    REBALANCE_DESCRIPTIONS,
    BacktestSpec,
    CostSpec,
    RebalanceFrequency,
)
from optimization_engine.backtest.walkforward import walk_forward_run

__all__ = [
    "REBALANCE_DESCRIPTIONS",
    "BacktestResult",
    "RebalanceFrequency",
    "WalkForwardResult",
    "backtest_weights",
    "compare_in_and_out_of_sample",
    "rebalance_dates",
    "walk_forward_backtest",
]


@dataclass
class BacktestResult:
    """Outcome of replaying a weight schedule over a return history.

    Attributes:
        returns: Portfolio returns, net of transaction costs.
        gross_returns: The same series before costs.
        weights: Actual held weights at the start of each period, after drift.
        turnover: Two-sided turnover, ``Σ|Δw|`` (buys plus sells),
            traded on each rebalance date. One-way is half of it.
        costs: Transaction cost charged on each rebalance date.
        rebalance_dates: When the book was scheduled to trade.
        is_out_of_sample: Whether the weights were chosen without seeing the
            returns they are evaluated on. The single most important caveat
            attached to any backtest number.
        metadata: Free-form run description (lookback, frequency, costs …).
        run: The full result bundle — per-trade costs, NAV, targets, and the
            spec and result hashes. ``None`` only on results built by hand.
    """

    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    rebalance_dates: pd.DatetimeIndex
    is_out_of_sample: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
    run: RunResult | None = None

    @property
    def total_turnover(self) -> float:
        """Sum of two-sided turnover, ``Σ|Δw|``, over the backtest, as a fraction of NAV."""
        return float(self.turnover.sum())

    @property
    def total_cost(self) -> float:
        """Every cost charged over the whole backtest, as a fraction of NAV."""
        return float(self.costs.sum())

    @property
    def annualized_turnover(self) -> float:
        """Two-sided turnover per year: ``Σ|Δw|``, buys plus sells.

        Desks usually quote one-way turnover, which is half of this.

        Returns:
            Total turnover divided by the run's length in years, or ``nan`` for a
            run too short to annualize.
        """
        periods_per_year = int(self.metadata.get("periods_per_year", 252))
        years = len(self.returns) / periods_per_year
        return float(self.turnover.sum() / years) if years > 0 else float("nan")

    def wealth(self, starting: float = 1.0) -> pd.Series:
        """The net-of-cost equity curve.

        Args:
            starting: Value at the first period. Defaults to ``1.0``, which makes
                the series read as a growth multiple.

        Returns:
            A series indexed like :attr:`returns`, compounding them from
            ``starting``.
        """
        return starting * (1 + self.returns).cumprod()

    def summary(self, periods_per_year: int | None = None, riskfree_rate: float = 0.0):
        """Standard performance summary of the net-of-cost return stream.

        Args:
            periods_per_year: Annualization basis. Defaults to the result's own.
            riskfree_rate: Annual risk-free rate for the ratio metrics, as a
                fraction; converted to a per-period rate internally.

        Returns:
            A one-row-per-statistic summary frame.
        """
        from optimization_engine.analytics.performance import summary_stats

        ppy = periods_per_year or int(self.metadata.get("periods_per_year", 252))
        return summary_stats(
            self.returns.to_frame("portfolio"),
            periods_per_year=ppy,
            riskfree_rate=riskfree_rate,
            extended=True,
        )

    def cost_drag(self, periods_per_year: int | None = None) -> float:
        """Annualized return given up to transaction costs.

        Args:
            periods_per_year: Annualization basis. Defaults to the result's own.

        Returns:
            The gross annualized return minus the net one, as a fraction.
        """
        from optimization_engine.analytics.performance import annualize_returns

        ppy = periods_per_year or int(self.metadata.get("periods_per_year", 252))
        gross = annualize_returns(self.gross_returns, ppy)
        net = annualize_returns(self.returns, ppy)
        return float(gross - net)

    def tca(self):
        """The transaction-cost panel, when the full bundle is available.

        Raises:
            ValueError: If this result was not produced by the simulation core.
        """
        from optimization_engine.backtest.tca import compute_tca

        if self.run is None:
            raise ValueError(
                "This result carries no run bundle, so there are no per-trade "
                "costs to analyze."
            )
        return compute_tca(self.run)


def _adapt(run: RunResult, marks: pd.DatetimeIndex, extra: dict) -> BacktestResult:
    """Project the full bundle onto the compact legacy result shape."""
    metadata: dict[str, object] = {
        "frequency": run.meta.spec.get("frequency"),
        "transaction_cost_bps": run.meta.spec.get("costs", {}).get("commission_bps", 0.0),
        "periods_per_year": run.meta.spec.get("periods_per_year", 252),
        "execution_lag": run.meta.spec.get("execution_lag", 0),
        "spec_hash": run.meta.spec_hash,
        "result_hash": run.meta.result_hash,
    }
    if run.meta.degradations:
        metadata["cost_degradations"] = list(run.meta.degradations)
    metadata.update(extra)
    return BacktestResult(
        returns=run.returns,
        gross_returns=run.gross_returns,
        weights=run.weights,
        turnover=run.turnover,
        costs=run.cost_series,
        rebalance_dates=marks,
        is_out_of_sample=run.meta.is_out_of_sample,
        metadata=metadata,
        run=run,
    )


def backtest_weights(
    returns: pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
    frequency: RebalanceFrequency = "monthly",
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
    is_out_of_sample: bool = False,
    *,
    spec: BacktestSpec | None = None,
    cost_model: CostModel | None = None,
    execution_lag: int = 0,
) -> BacktestResult:
    """Replay a weight schedule, letting positions drift between rebalances.

    Args:
        returns: Periodic asset returns.
        weights: Either one target vector held throughout, or a frame of
            target weights indexed by the date they become effective (used by
            the walk-forward runner).
        frequency: How often the book is traded back to target.
        transaction_cost_bps: One-way cost in basis points of traded notional.
            25 bps on 100% turnover costs 25 bps of NAV.
        periods_per_year: Observations per year, for annualizing turnover.
        is_out_of_sample: Tag the result; see :class:`BacktestResult`.
        spec: A full :class:`~optimization_engine.backtest.spec.BacktestSpec`,
            for the cost models and execution lag the four positional
            arguments above cannot express. It supersedes them entirely.
        cost_model: Override the model built from the spec's costs.
        execution_lag: Periods between a decision and its fill. Ignored when
            ``spec`` is given, which carries its own.

    Raises:
        ValueError: If ``returns`` is empty or the weights cover none of it.
    """
    if returns is None or returns.empty:
        raise ValueError("Cannot backtest on empty returns.")

    if spec is None:
        spec = BacktestSpec(
            frequency=frequency,
            costs=CostSpec.from_bps(transaction_cost_bps),
            execution_lag=execution_lag,
            periods_per_year=periods_per_year,
            is_out_of_sample=is_out_of_sample,
        )
    run = run_backtest(returns, weights, spec, cost_model=cost_model)
    marks = rebalance_dates(pd.DatetimeIndex(returns.index), spec.frequency)
    return _adapt(run, marks, {})


@dataclass
class WalkForwardResult:
    """Out-of-sample walk-forward evaluation of an optimization process."""

    backtest: BacktestResult
    weights_history: pd.DataFrame
    windows: pd.DataFrame
    failures: tuple[str, ...] = ()

    @property
    def returns(self) -> pd.Series:
        """The out-of-sample return stream, taken from the underlying backtest."""
        return self.backtest.returns

    @property
    def n_resolves(self) -> int:
        """How many times the optimizer actually re-solved."""
        return len(self.weights_history)

    @property
    def n_rebalances(self) -> int:
        """Deprecated spelling of :attr:`n_resolves`.

        Ambiguous now that the trading cadence is separable from the re-solve
        cadence: the book can rebalance without the optimizer re-solving.
        """
        return self.n_resolves

    @property
    def n_trade_dates(self) -> int:
        """How many dates the book actually traded on."""
        return len(self.backtest.rebalance_dates)

    def weight_stability(self) -> pd.Series:
        """Average absolute change in each asset's weight between re-solves.

        High values mean the optimizer is chasing estimation noise: the
        allocation is being rewritten every window on data that barely moved.
        """
        if len(self.weights_history) < 2:
            return pd.Series(dtype=float)
        return self.weights_history.diff().abs().mean()


def walk_forward_backtest(
    returns: pd.DataFrame,
    solve: Callable[[pd.DataFrame], pd.Series],
    lookback: int,
    rebalance_every: int,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
    min_lookback: int | None = None,
    expanding: bool = False,
    *,
    spec: BacktestSpec | None = None,
    rebalance_frequency: RebalanceFrequency | None = None,
) -> WalkForwardResult:
    """Re-estimate and re-solve on a rolling window, holding results forward.

    At each re-solve the optimizer sees only returns strictly *before* the
    decision date; the resulting weights are then held over the following
    ``rebalance_every`` periods. No information from the evaluation window
    reaches the estimate, which is what makes the resulting track record
    something other than a description of the past.

    Args:
        returns: Full periodic return history.
        solve: Callable taking a returns window and returning target weights.
            Typically wraps ``run_engine`` with a fixed config.
        lookback: Estimation window length in periods.
        rebalance_every: Periods between **re-solves** — the re-optimization
            cadence.
        transaction_cost_bps: One-way cost on traded notional.
        periods_per_year: Observations per year.
        min_lookback: Minimum window before the first solve. Defaults to
            ``lookback``.
        expanding: Use a growing window anchored at the start instead of a
            fixed-length rolling one.
        spec: A full backtest spec, for cost models and execution lag the
            scalar arguments cannot express. Its ``frequency`` is read as the
            trading cadence unless ``rebalance_frequency`` overrides it.
        rebalance_frequency: How often the book is **traded back** to the
            current target between re-solves. Defaults to ``"none"`` — trade
            only when a new target is solved. See
            :func:`~optimization_engine.backtest.walkforward.walk_forward_run`
            for why the two cadences are worth separating.

    Raises:
        ValueError: If the history is too short to produce a single
            out-of-sample period, or if every solve fails.
    """
    trading = rebalance_frequency
    if trading is None:
        # The scalar-argument form has never traded between re-solves, so its
        # own default spec must not smuggle a cadence in: adding turnover
        # silently would move every existing caller's cost figures.
        trading = spec.frequency if spec is not None else "none"
    if spec is None:
        spec = BacktestSpec(
            frequency=trading,
            costs=CostSpec.from_bps(transaction_cost_bps),
            periods_per_year=periods_per_year,
        )
    walk = walk_forward_run(
        returns,
        solve,
        lookback=lookback,
        rebalance_every=rebalance_every,
        spec=spec,
        min_lookback=min_lookback,
        expanding=expanding,
        rebalance_frequency=trading,
    )
    # The dates the book actually traded on. Reporting only the first one —
    # which this did — makes every walk-forward look like a single purchase,
    # and hides the calendar trades entirely once the cadences differ.
    backtest = _adapt(walk.run, walk.run.rebalance_dates, dict(walk.metadata))
    return WalkForwardResult(
        backtest=backtest,
        weights_history=walk.weights_history,
        windows=walk.windows,
        failures=walk.failures,
    )


def compare_in_and_out_of_sample(
    in_sample: pd.Series,
    out_of_sample: pd.Series,
    periods_per_year: int = 252,
    riskfree_rate: float = 0.0,
) -> pd.DataFrame:
    """Put the fitted and the walk-forward track records side by side.

    The gap between the two columns is the honest measure of how much of the
    backtest was optimization and how much was hindsight.

    Args:
        in_sample: The in-sample return stream — weights chosen knowing these
            returns.
        out_of_sample: The walk-forward stream, from the same process run
            without hindsight.
        periods_per_year: Annualization basis for both.
        riskfree_rate: Annual risk-free rate used by the ratio metrics.

    Returns:
        A frame with one row per statistic and a column per sample —
        ``"In-sample (fitted)"`` and ``"Out-of-sample (walk-forward)"`` —
        plus a ``"Degradation"`` column holding the first minus the second.
        Both columns are computed over the periods the two streams *share*:
        a walk-forward starts one lookback into the sample, and scoring the
        fitted stream over a longer window than the one it is compared with
        would not be a comparison.
    """
    from optimization_engine.analytics.performance import summary_stats

    frame = pd.concat(
        [
            in_sample.rename("In-sample (fitted)"),
            out_of_sample.rename("Out-of-sample (walk-forward)"),
        ],
        axis=1,
        join="inner",
    ).dropna(how="any")
    stats = summary_stats(
        frame, periods_per_year=periods_per_year, riskfree_rate=riskfree_rate
    ).T
    stats["Degradation"] = (
        stats["In-sample (fitted)"] - stats["Out-of-sample (walk-forward)"]
    )
    return stats
