"""One object that answers the questions a backtest is asked in a meeting.

The frames a run produces are the raw material; a tearsheet is the reading.
It puts performance, drawdowns, costs, position statistics, and — when the
run came out of a search — the selection-bias correction side by side, so the
Sharpe ratio never appears without the three things that qualify it: what it
cost to trade, how much of it survives out of sample, and how many
configurations were tried before this one.

Assembly is deliberately lazy about what it is given. A tearsheet built from
an in-sample run says so and skips the out-of-sample panel rather than
inventing one. A tearsheet with no sweep behind it reports no deflation. The
absence is stated, not filled in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from optimization_engine.backtest.positions import (
    PositionStats,
    compute_position_stats,
    episodes_frame,
    position_episodes,
)
from optimization_engine.backtest.results import RunResult
from optimization_engine.backtest.tca import TcaPanel, compute_tca, cost_by_asset
from optimization_engine.stress import Shock, StressReport, stress_test


@dataclass
class Tearsheet:
    """The assembled reading of one run.

    Attributes:
        run: The run this describes.
        performance: Summary statistics of the net return stream.
        drawdowns: The worst drawdowns, with their dates and recovery.
        tca: The cost panel.
        costs_by_asset: Where the cost was concentrated.
        positions: Round-trip statistics over position episodes.
        episodes: The episodes themselves.
        deflated_sharpe: Selection-bias correction, when trials were counted.
        overfitting: CSCV report, when a grid was run.
        stress: What the supplied shocks do to the book the run ended on,
            when any were supplied. A backward-looking track record says
            nothing about a shock that has not happened yet; this is the one
            forward-looking panel here, and its absence is stated in
            ``describe()`` by simply not appearing.
        caveats: What a reader must know before quoting any number here.
    """

    run: RunResult
    performance: pd.DataFrame
    drawdowns: pd.DataFrame
    tca: TcaPanel
    costs_by_asset: pd.DataFrame
    positions: PositionStats
    episodes: pd.DataFrame
    deflated_sharpe: Any = None
    overfitting: Any = None
    stress: StressReport | None = None
    caveats: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Every panel as a frame, ready for the Excel exporter.

        Returns:
            One frame per panel. ``stress`` and ``stress_contributions``
            appear only when the tearsheet was built with shocks, so a run
            with none exports the same sheets it always did.
        """
        frames = {
            "performance": self.performance,
            "drawdowns": self.drawdowns,
            "tca": self.tca.to_frame(),
            "costs_by_asset": self.costs_by_asset,
            "positions": self.positions.to_frame(),
            "episodes": self.episodes,
            "weights": self.run.weights,
            "nav": self.run.nav.to_frame("nav"),
            "trades": self.run.trades,
            "caveats": pd.DataFrame({"caveat": list(self.caveats)}),
        }
        if self.stress is not None:
            frames["stress"] = self.stress.to_frame()
            frames["stress_contributions"] = self.stress.contributions_frame()
        return frames

    def describe(self) -> str:
        """The whole tearsheet as text: run, costs, selection bias, caveats.

        Returns:
            One line per section, newline-joined. The caveats are part of the
            output rather than an appendix — an in-sample run says so here.
        """
        lines = [self.run.describe(), self.tca.describe()]
        if self.deflated_sharpe is not None:
            lines.append(self.deflated_sharpe.describe())
        if self.overfitting is not None:
            lines.append(self.overfitting.describe())
        if self.stress is not None:
            lines.append(self.stress.describe())
        lines.extend(f"Caveat: {caveat}" for caveat in self.caveats)
        return "\n".join(lines)


def _caveats(
    run: RunResult,
    deflated: Any,
    overfitting: Any,
    deflation_note: str | None = None,
    deflation_failed: bool = False,
) -> tuple[str, ...]:
    caveats: list[str] = []
    if not run.meta.is_out_of_sample:
        caveats.append(
            "These weights were chosen knowing these returns. The figures "
            "describe a fit, not a track record."
        )
    costs = run.meta.spec.get("costs", {})
    linear_bps = float(costs.get("commission_bps", 0.0)) + float(
        costs.get("slippage_bps", 0.0)
    )
    impact_eta = float(costs.get("impact_coefficient", 0.0))
    if linear_bps == 0.0 and impact_eta == 0.0:
        caveats.append(
            "Trading was modelled as free. Every turnover figure below is a "
            "cost this backtest did not charge."
        )
    elif linear_bps == 0.0:
        # Impact alone was charged. Calling that "free" contradicts the cost
        # line right above it; the honest statement is the narrower one.
        caveats.append(
            "Only market impact was charged - no commission and no spread. "
            "The cost below is what size did to the price, not what the trade "
            "cost to place."
        )
    if impact_eta > 0.0 and costs.get("impact_participation_source") == "adv":
        caveats.append(
            "Market impact was priced against traded volume where the panel "
            "carried it. Any asset without volume was charged at the fixed "
            "participation rate instead - those trades are named in the "
            "degradation notes."
        )
    if int(run.meta.spec.get("execution_lag", 0)) == 0:
        caveats.append(
            "Orders fill on the close before the bar each target is dated on, "
            "and the target earns that bar: honest only if it used nothing "
            "from that bar. A real desk trades at least one period after it "
            "decides, at a price it has not seen."
        )
    if run.meta.degradations:
        caveats.append(
            f"{len(run.meta.degradations)} cost component(s) could not be computed "
            "and were charged as zero; the reported cost is a lower bound."
        )
    cash_periods = int(run.meta.notes.get("periods_in_cash_after_failed_solve", 0) or 0)
    if cash_periods > 0:
        caveats.append(
            f"The first {cash_periods} period(s) were held in cash because the "
            "opening solve failed and there was no book to carry forward. They "
            "are in this track record at a zero return, not dropped from it."
        )
    if deflation_failed:
        caveats.append(
            f"{deflation_note} The Sharpe ratio is undeflated; if this "
            "configuration was chosen from several, it is optimistic."
        )
    elif deflated is None:
        caveats.append(
            "No trial count was supplied, so the Sharpe ratio is undeflated. If "
            "this configuration was chosen from several, it is optimistic."
        )
    elif deflation_note:
        caveats.append(deflation_note)
    if overfitting is not None and getattr(overfitting, "pbo", 0.0) > 0.5:
        caveats.append(
            "The in-sample winner lands below the out-of-sample median more "
            "often than not: the selection is no better than picking at random."
        )
    return tuple(caveats)


def build_tearsheet(
    run: RunResult,
    returns: pd.DataFrame,
    *,
    riskfree_rate: float = 0.0,
    n_trials: int | None = None,
    trial_sharpes: pd.Series | None = None,
    overfitting: Any = None,
    top_drawdowns: int = 5,
    shocks: Sequence[Shock] = (),
    stress_cov_matrix: pd.DataFrame | None = None,
) -> Tearsheet:
    """Assemble the full reading of a run.

    Args:
        run: The simulation to describe.
        returns: The asset returns it was replayed on, for position episodes.
        riskfree_rate: Annual rate for the Sharpe and Sortino ratios.
        n_trials: How many configurations were tried before this one. Supply
            it and the Sharpe gets deflated; leave it out and the tearsheet
            says the number is undeflated rather than pretending otherwise.
            A deflation that cannot run on these inputs leaves
            ``deflated_sharpe`` at ``None`` and says why in the caveats and in
            ``metadata["deflation_note"]``.
        trial_sharpes: The trials' annualized Sharpes, when available. Their
            dispersion is what the deflation actually uses. Fewer than two
            finite values have no dispersion; the deflation then falls back to
            this run's own sampling variance, as
            :func:`~optimization_engine.analytics.selection.deflated_sharpe_ratio`
            does when no trials are kept, and a caveat says so.
        overfitting: A pre-computed CSCV report from a sweep.
        top_drawdowns: How many drawdown episodes to table.
        shocks: Stress scenarios to apply to the book the run ended on — the
            last row of ``run.weights``, which is the only book a reader can
            still trade. Leave it empty and the tearsheet carries no stress
            panel rather than an empty one.
        stress_cov_matrix: The covariance the stressed volatilities are
            measured against, in whatever annualization the caller works in.
            Omit it and the stress panel reports P&L only: this function will
            not silently estimate a covariance from ``returns`` and present
            the result as if it had been given one.

    Returns:
        The assembled :class:`Tearsheet`.

    Raises:
        StressError: If ``shocks`` name an asset the final book does not hold,
            or ``stress_cov_matrix`` does not cover it.
    """
    from optimization_engine.analytics.risk import drawdown_table

    performance = run.summary(riskfree_rate=riskfree_rate)
    drawdowns = drawdown_table(run.returns, top=top_drawdowns)

    deflated = None
    deflation_note: str | None = None
    deflation_failed = False
    if n_trials is not None and n_trials >= 1:
        from optimization_engine.analytics import selection

        trials = trial_sharpes
        if trials is not None:
            supplied = pd.Series(trials, dtype=float)
            usable = int(np.isfinite(supplied.to_numpy()).sum())
            if usable < 2:
                # The documented fallback of ``deflated_sharpe_ratio`` when
                # the trials were not kept. A sweep that left one survivor
                # still tried every cell, and deflating against that count
                # with a cruder dispersion beats not deflating at all.
                deflation_note = (
                    f"Only {usable} of the {len(supplied)} trial Sharpe(s) supplied "
                    "is usable, which has no dispersion, so the deflation against "
                    f"{int(n_trials)} trials used this run's own sampling variance "
                    "rather than the spread of the trials."
                )
                trials = None
        try:
            deflated = selection.deflated_sharpe_ratio(
                run.returns,
                n_trials=int(n_trials),
                trial_sharpes=trials,
                riskfree_rate=riskfree_rate,
                periods_per_year=run.periods_per_year,
            )
        except ValueError as exc:
            # The one exception the deflation raises for inputs it cannot use.
            # It used to be swallowed with everything else, and the caveat
            # then said no trial count had been supplied. Anything other than
            # a ValueError is a bug, and travels.
            deflated = None
            deflation_failed = True
            deflation_note = (
                f"A trial count of {int(n_trials)} was supplied, but the Sharpe "
                f"ratio could not be deflated: {exc}"
            )

    episodes = position_episodes(run.weights, returns)

    stress = None
    stress_as_of = None
    if len(shocks) and not run.weights.empty:
        final_book = run.weights.iloc[-1]
        stress_as_of = str(run.weights.index[-1])
        stress = stress_test(final_book, list(shocks), cov_matrix=stress_cov_matrix)

    return Tearsheet(
        run=run,
        performance=performance,
        drawdowns=drawdowns,
        tca=compute_tca(run),
        costs_by_asset=cost_by_asset(run),
        positions=compute_position_stats(run, returns),
        episodes=episodes_frame(episodes),
        deflated_sharpe=deflated,
        overfitting=overfitting,
        stress=stress,
        caveats=_caveats(
            run, deflated, overfitting, deflation_note, deflation_failed
        ),
        metadata={
            "spec_hash": run.meta.spec_hash,
            "result_hash": run.meta.result_hash,
            "n_trials": n_trials,
            "deflation_note": deflation_note,
            "stress_as_of": stress_as_of,
        },
    )


__all__ = ["Tearsheet", "build_tearsheet"]
