"""Transaction-cost analysis: what the trading actually cost, and per what.

A single "total cost" number is almost useless. The same 40 basis points of
NAV means something different depending on whether it came from trading the
book over twice a year or twenty times, and whether it is commission — which
a better broker fixes — or impact, which only a smaller position does.

So the panel below reports the total, the split, and three ratios that
normalize it: cost per unit of traded notional, cost per rebalance, and the
annualized drag on returns. Ratios that cannot be computed come back as
``None`` with a reason attached rather than as zero or ``NaN``. Zero is a
claim about the world; ``None`` with "no trades were executed" is the truth.

The annualized drag here is the *approximate* form — total cost spread over
the sample and annualized — not a counterfactual re-run at zero cost. The
exact figure is on the result itself as
:meth:`~optimization_engine.backtest.results.RunResult.cost_drag`, which
compares the gross and net return streams the run already produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from optimization_engine.backtest.results import RunResult

_BPS = 10_000.0


@dataclass(frozen=True)
class TcaPanel:
    """Cost totals, their split, and the ratios that make them comparable.

    Attributes:
        total_cost: All costs charged, as a fraction of NAV.
        commission: The broker's share.
        slippage: Spread and market impact.
        total_turnover: Traded notional, buys plus sells (``Σ|Δw|``), as
            a multiple of NAV. One-way turnover is half of it.
        annualized_turnover: The same, per year.
        n_rebalances: Dates on which the book traded.
        cost_bps_of_notional: Cost per unit traded. The number to compare
            across strategies of different trading intensity.
        avg_cost_per_rebalance_bps: Cost of an average trip to the market.
        cost_drag_bps_annualized: Approximate annualized return given up.
        commission_share: Fraction of cost that is commission — high means a
            broker problem, low means a size problem.
        degradations: Cost components that switched themselves off during the
            run. Non-empty means these figures are a lower bound.
        reasons: Why any ``None`` field is ``None``.
    """

    total_cost: float
    commission: float
    slippage: float
    total_turnover: float
    annualized_turnover: float
    n_rebalances: int
    cost_bps_of_notional: float | None
    avg_cost_per_rebalance_bps: float | None
    cost_drag_bps_annualized: float | None
    commission_share: float | None
    degradations: tuple[str, ...] = ()
    reasons: dict[str, str] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """A one-column frame, for printing next to the performance summary."""
        rows = {
            "Total cost (% of NAV)": self.total_cost * 100.0,
            "  Commission": self.commission * 100.0,
            "  Slippage & impact": self.slippage * 100.0,
            "Commission share": self.commission_share,
            "Total turnover (x NAV)": self.total_turnover,
            "Annualized turnover (x/yr)": self.annualized_turnover,
            "Rebalances": float(self.n_rebalances),
            "Cost per notional (bps)": self.cost_bps_of_notional,
            "Cost per rebalance (bps)": self.avg_cost_per_rebalance_bps,
            "Annualized drag (bps)": self.cost_drag_bps_annualized,
        }
        return pd.DataFrame({"value": rows})

    def describe(self) -> str:
        """One line: turnover, cost per unit of notional, and annualized drag.

        Returns:
            A sentence, or a statement that there is no cost to analyze when the
            run never traded.
        """
        if self.cost_bps_of_notional is None:
            return "No trades were executed; there is no cost to analyze."
        drag = (
            "n/a"
            if self.cost_drag_bps_annualized is None
            else f"{self.cost_drag_bps_annualized:.1f} bps/yr"
        )
        return (
            f"Traded {self.annualized_turnover:.2f}x of the book per year at "
            f"{self.cost_bps_of_notional:.1f} bps of notional, costing {drag}."
        )


def compute_tca(run: RunResult) -> TcaPanel:
    """Build the cost panel from a run's own trade and cost frames.

    Sums are accumulated over the frames in their canonical order so the
    panel is reproducible for a given run, in the same way the result hash is.

    Args:
        run: The finished run to analyze.

    Returns:
        A :class:`TcaPanel` with turnover, cost per unit of notional, cost per
        rebalance and the annualized drag. Every ratio is ``None`` rather than
        zero when the run never traded.
    """
    reasons: dict[str, str] = {}
    commission = 0.0
    slippage = 0.0
    turnover = 0.0
    if not run.costs.empty:
        ordered = run.costs.sort_values("date", kind="mergesort")
        for row in ordered.itertuples(index=False):
            commission += float(row.commission)
            slippage += float(row.slippage)
            turnover += float(row.turnover)
    total = commission + slippage
    n_rebalances = int(len(run.costs))

    if turnover > 0.0:
        cost_bps_of_notional: float | None = total / turnover * _BPS
    else:
        cost_bps_of_notional = None
        reasons["cost_bps_of_notional"] = (
            "no traded notional — cost per unit traded is undefined"
        )

    if n_rebalances > 0:
        avg_cost_per_rebalance_bps: float | None = total / n_rebalances * _BPS
    else:
        avg_cost_per_rebalance_bps = None
        reasons["avg_cost_per_rebalance_bps"] = (
            "no rebalances — average cost per trip is undefined"
        )

    periods = len(run.returns)
    if periods >= 1:
        years = periods / float(run.periods_per_year)
        cost_drag_bps_annualized: float | None = (
            total / years * _BPS if years > 0 else None
        )
    else:
        cost_drag_bps_annualized = None
    if cost_drag_bps_annualized is None:
        reasons["cost_drag_bps_annualized"] = (
            "no evaluated periods — annualized drag is undefined"
        )

    if total > 0.0:
        commission_share: float | None = commission / total
    else:
        commission_share = None
        reasons["commission_share"] = "zero total cost — the split is undefined"

    return TcaPanel(
        total_cost=total,
        commission=commission,
        slippage=slippage,
        total_turnover=turnover,
        annualized_turnover=run.annualized_turnover,
        n_rebalances=n_rebalances,
        cost_bps_of_notional=cost_bps_of_notional,
        avg_cost_per_rebalance_bps=avg_cost_per_rebalance_bps,
        cost_drag_bps_annualized=cost_drag_bps_annualized,
        commission_share=commission_share,
        degradations=tuple(run.meta.degradations),
        reasons=reasons,
    )


def cost_by_asset(run: RunResult) -> pd.DataFrame:
    """Where the money went, by name.

    Concentrated cost is usually a liquidity story: one illiquid position
    being rebalanced against, over and over, by an optimizer that has no idea
    it is expensive.

    Args:
        run: The finished run to analyze.

    Returns:
        One row per traded asset with its turnover, its total cost as a
        fraction of NAV, and its share of the run's whole cost.
    """
    if run.trades.empty:
        return pd.DataFrame(
            columns=["traded_notional", "commission", "slippage", "cost", "n_trades"]
        )
    frame = run.trades.copy()
    frame["traded_notional"] = frame["traded_weight"].abs()
    grouped = frame.groupby("asset", sort=True).agg(
        traded_notional=("traded_notional", "sum"),
        commission=("commission", "sum"),
        slippage=("slippage", "sum"),
        cost=("cost", "sum"),
        n_trades=("cost", "size"),
    )
    return grouped.sort_values("cost", ascending=False)


__all__ = ["TcaPanel", "compute_tca", "cost_by_asset"]
