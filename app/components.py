"""Reusable Streamlit rendering blocks for the optimization app.

Kept separate from ``streamlit_app.py`` so the page script stays a readable
sequence of steps rather than a wall of chart plumbing. Every ``render_*``
here is a pure render function: it takes data and draws it, and never mutates
session state.

Beside them sit the small pure transforms the page needs *before* it can draw
anything — an editable grid turned into stress scenarios and back, an
eligibility frame turned into the three state codes its heatmap colours. Those
live here rather than inline in the page for one reason: a function that
returns a value can be tested in milliseconds, and a page that draws one
cannot. ``tests/test_ui_stress_universe`` calls them directly;
``tests/test_app_layers`` runs the page that uses them.

Nothing here imports the engine at module level, so this module loads with
Streamlit alone; the one function that needs it imports inside its own body.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

PERCENT_COLUMNS = {
    "Annualized Return", "Annualized Vol", "Max Drawdown",
    "Annualized Excess", "Annualized T.E.", "Expected Return", "Expected Vol",
    "Historic VaR(5%)", "Cornish-Fisher VaR(5%)", "Historic CVaR(5%)",
    "In-sample (fitted)", "Out-of-sample (walk-forward)", "Degradation",
    # Absolute and relative performance additions.
    "Best Period", "Worst Period", "Alpha (annualized)", "Residual Vol",
    "Downside T.E.", "Worst Relative Drawdown", "M-squared", "Treynor Ratio",
    # Calendar-period tables, whose columns are the series themselves.
    "Portfolio", "Benchmark", "Excess",
}
RATIO_COLUMNS = {
    "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Omega Ratio",
    "Tail Ratio", "Skewness", "Kurtosis", "Ulcer Index", "Beta",
    "Information Ratio", "Up Capture", "Down Capture", "Capture", "Sharpe",
    "Martin Ratio", "Gain-to-Pain", "Win/Loss Ratio", "Appraisal Ratio",
    "Correlation", "R-squared", "Up Beta", "Down Beta", "Beta Asymmetry",
    "Alpha t-stat",
}
#: Reported as percentages of *periods* rather than of money.
SHARE_COLUMNS = {
    "Hit Rate", "Prob. Sharpe > 0", "Time Under Water", "Batting Average",
    "Up Number Ratio", "Down Number Ratio", "Prob. Excess > 0", "Active Share",
}

#: Signed-quantity colours, taken from the library's validated palette so a
#: figure drawn here sits beside one drawn by ``reporting.plots`` without a
#: second red or a second green appearing.
LOSS_COLOR = "#e34948"
GAIN_COLOR = "#1baf7a"

#: The three states of an eligibility frame, and the colour each one gets.
#:
#: The point of the whole universe module is that *not evaluable* is neither
#: eligible nor ineligible — a rule with a warm-up has not screened a name, it
#: has not reached it. Drawing that on a two-colour scale collapses it onto
#: whichever end the scale happens to put it at, which is the exact error the
#: three-valued logic exists to prevent, so the third state gets a hue of its
#: own rather than a shade of an existing one.
#:
#: The choice: a desaturated warm grey for *not eligible* (a screen looked and
#: said no), a saturated amber for *not evaluable* (nobody looked — the
#: attention-getting colour, because it is the state that needs a decision),
#: and the palette's aqua-green for *eligible*. Amber and green are adjacent
#: slots of ``reporting.plots.PALETTE``, which is validated for
#: colour-vision-deficiency separation as a pair; the grey is separated from
#: both by chroma rather than hue, so it survives any CVD. Hover text names
#: the state in words as well, so the reading never rests on colour alone.
ELIGIBILITY_STATES = ("Not eligible", "Not evaluable", "Eligible")
ELIGIBILITY_COLORS = ("#c3c2b7", "#eda100", "#1baf7a")


def format_table(df: pd.DataFrame) -> Any:
    """Style a frame by column *name*, so new metrics format themselves.

    The previous approach passed a hand-written dict of column formats; every
    metric added to ``summary_stats`` then rendered as a raw float until
    someone remembered to update the dict in three places.
    """
    fmt: dict[str, str] = {}
    for col in df.columns:
        name = str(col)
        if name in PERCENT_COLUMNS or name.startswith(("Historic ", "Cornish")):
            fmt[col] = "{:.2%}"
        elif name in RATIO_COLUMNS:
            fmt[col] = "{:.3f}"
        elif name in ("Hit Rate", "Prob. Sharpe > 0"):
            fmt[col] = "{:.1%}"
        elif pd.api.types.is_float_dtype(df[col]):
            fmt[col] = "{:.4f}"
    try:
        return df.style.format(fmt)
    except Exception:
        return df


def metric_row(items: Iterable[tuple[str, str, str | None]]) -> None:
    """Render a row of ``st.metric`` cards from ``(label, value, help)``."""
    items = list(items)
    if not items:
        return
    for col, (label, value, helptext) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=helptext)


def pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}%}"


def num(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Diagnostics panels
# ---------------------------------------------------------------------------


def render_data_quality(report, expanded_default: bool = False) -> None:
    """Summarize a :class:`DataQualityReport` as a badge plus detail."""
    n_err = len(report.errors)
    n_warn = len(report.warnings)

    if n_err:
        st.error(
            f"**{n_err} data problem(s) that will corrupt the optimization.** "
            "Fix these before trusting any result below."
        )
    elif n_warn:
        st.warning(
            f"**{n_warn} data caveat(s).** The optimizer will run, but these "
            "affect what the answer means."
        )
    else:
        st.success("Data checks passed — no gaps, stale runs, or thin samples found.")

    if report.issues:
        with st.expander(
            f"Data-quality detail ({n_err} error, {n_warn} warning)"
            if n_err or n_warn else "Data-quality detail",
            expanded=expanded_default or bool(n_err),
        ):
            rows = [
                {
                    "Severity": i.severity.title(),
                    "Asset": i.asset or "—",
                    "Finding": i.message,
                    "What to do": i.suggestion,
                }
                for i in report.issues
            ]
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True
            )


def render_feasibility(report, show_ok: bool = True) -> bool:
    """Render a :class:`FeasibilityReport`. Returns True when solvable.

    This is the block that replaces ``Solver failed: status=infeasible`` with
    a sentence naming the constraint and the fix.
    """
    if report is None:
        return True
    if report.is_feasible and not report.issues:
        if show_ok:
            msg = "Constraints are feasible."
            if report.min_return is not None and report.max_return is not None:
                msg += (
                    f" Reachable expected return: "
                    f"{report.min_return:.2%} to {report.max_return:.2%}"
                )
                if report.min_variance_return is not None:
                    msg += (
                        f" (efficient above {report.min_variance_return:.2%})."
                    )
            st.success(msg)
        return True

    if not report.is_feasible:
        st.error("**These constraints have no solution.** Nothing will solve until one of these changes:")
        for issue in report.fatal_issues:
            st.markdown(f"- **{issue.message}**  \n  ↳ {issue.suggestion}")
    for issue in report.warnings:
        st.warning(f"{issue.message}  \n↳ {issue.suggestion}")
    return report.is_feasible


def render_covariance_diagnostics(diag) -> None:
    """Show whether the covariance estimate is fit to optimize against."""
    if diag is None:
        return
    metric_row(
        [
            ("Observations (T)", f"{diag.n_observations:,}", None),
            ("Assets (N)", str(diag.n_assets), None),
            (
                "T / N",
                num(diag.observations_per_asset, 1),
                "Below ~10, covariance estimates are dominated by noise.",
            ),
            (
                "Condition number",
                f"{diag.condition_number:.3g}",
                "Largest over smallest eigenvalue. Above 1e4 the matrix is "
                "poorly conditioned and mean-variance weights get unstable.",
            ),
        ]
    )
    for warning in diag.warnings:
        st.warning(warning)


def render_compliance(result) -> None:
    """Banner stating whether the solved weights respect every constraint."""
    diag = result.extras.get("diagnostics")
    status = result.extras.get("solver_status", "unknown")
    if result.violations:
        st.error(
            "**The solved weights breach constraints you set.** "
            + " · ".join(result.violations)
        )
    elif status == "optimal_inaccurate":
        st.warning(
            "The solver converged only to loose tolerance "
            "(`optimal_inaccurate`). Treat the weights as approximate — "
            "usually a sign of a near-singular covariance or nearly-binding "
            "constraints."
        )
    elif diag is not None:
        st.success(
            f"Solved with {result.extras.get('solver', 'the default solver')} "
            f"in {result.extras.get('solve_seconds', 0):.2f}s — every bound, "
            "group budget and exposure limit is respected."
        )

    ignored = result.extras.get("ignored_constraints")
    if ignored:
        st.info(
            f"This method cannot enforce: {', '.join(ignored)}. "
            "The setting was kept but not imposed."
        )
    note = result.extras.get("bounds_note") or result.extras.get("fallback_reason")
    if note:
        distance = result.extras.get("projection_distance")
        if distance is not None and distance > 0.10:
            st.warning(note)
        else:
            st.info(note)
    rb_note = result.extras.get("risk_budget_note")
    if rb_note:
        st.info(rb_note)
    cvar_note = result.extras.get("cvar_note")
    if cvar_note:
        st.info(cvar_note)


def render_portfolio_diagnostics(diag) -> None:
    """Concentration and diversification cards."""
    if diag is None:
        return
    metric_row(
        [
            ("Positions", str(diag.n_positions), "Weights above 1bp."),
            (
                "Effective N",
                num(diag.effective_n, 1),
                "Inverse Herfindahl: how many equally-weighted positions this "
                "book is really worth. Far below the position count means the "
                "portfolio is concentrated.",
            ),
            (
                "Effective N (risk)",
                num(diag.effective_n_risk, 1),
                "The same measure applied to risk contributions. A 60/40 book "
                "is diversified by capital and concentrated in equity risk.",
            ),
            (
                "Diversification ratio",
                num(diag.diversification_ratio, 2),
                "Weighted-average asset volatility over portfolio volatility. "
                "1.0 means correlations bought you nothing.",
            ),
        ]
    )
    extra: list[tuple[str, str, str | None]] = [
        ("Largest position", pct(diag.max_weight), None),
        ("Gross exposure", pct(diag.gross_exposure), "Sum of absolute weights."),
    ]
    if abs(diag.short_exposure) > 1e-9:
        extra.append(("Short exposure", pct(diag.short_exposure), None))
    if diag.turnover is not None:
        extra.append(
            (
                "Turnover vs. previous",
                pct(diag.turnover),
                "Two-sided: buys plus sells, Σ|Δw|. Half of it is the "
                "fraction of the book that changes hands.",
            )
        )
    metric_row(extra)


def render_projection_distance(result) -> None:
    """Show how far the mandate moved a projection-based method's answer."""
    distance = result.extras.get("projection_distance")
    if distance is None:
        return
    metric_row(
        [
            (
                "Moved by constraints",
                pct(distance),
                "The fraction of the book the mandate shifted away from "
                "this method's own allocation, counted one way: ½·Σ|Δw|, "
                "half what a two-sided turnover figure shows for the same "
                "move. Large values mean the constraints, not the method, "
                "produced the result.",
            )
        ]
    )


def render_method_card(req) -> None:
    """Explain what the chosen optimizer does and what it assumes."""
    st.markdown(f"**{req.display_name}** — {req.summary}")
    with st.expander("What this method assumes", expanded=False):
        st.markdown(f"**Use it when:** {req.when_to_use}")
        if req.assumptions:
            st.markdown("**It assumes:**")
            for a in req.assumptions:
                st.markdown(f"- {a}")
        st.caption(req.bounds_note)
        needs = []
        if req.requires_mu:
            needs.append("expected returns")
        if req.requires_cov:
            needs.append("a covariance matrix")
        if req.requires_returns:
            needs.append("the full return history")
        if needs:
            st.caption(f"Inputs used: {', '.join(needs)}.")


def render_benchmark(run) -> None:
    """State which benchmark is in force, and what it does and does not offer.

    Rendered wherever relative numbers appear, because "information ratio
    0.6" is not a fact until the reader knows what it is 0.6 against.
    """
    bench = getattr(run, "benchmark", None)
    if bench is None:
        st.info(
            "No benchmark is set. Absolute numbers say what happened; only a "
            "benchmark says whether the process earned its fee. Choose one in "
            "the sidebar under **6 · Benchmark**."
        )
        return

    bits = [f"**{bench.label}**"]
    if bench.spec.kind != "external":
        bits.append(
            "rebalanced every period"
            if bench.spec.rebalance == "periodic"
            else "bought and held"
        )
    limits = []
    cfg = getattr(run, "config", None)
    if cfg is not None and cfg.max_tracking_error is not None:
        limits.append(f"tracking error ≤ {cfg.max_tracking_error:.2%}")
    if cfg is not None and cfg.max_active_share is not None:
        limits.append(f"active share ≤ {cfg.max_active_share:.0%}")
    line = " · ".join(bits)
    if limits:
        line += f" — optimized subject to {', '.join(limits)}"
    st.caption(line)
    if not bench.has_weights:
        st.caption(
            "This benchmark is a return series with no holdings in the "
            "universe, so active share and the active-risk decomposition are "
            "unavailable. Every return-based metric below still applies."
        )


def render_assumptions(assumptions: dict[str, Any]) -> None:
    """Print every modelling choice a result rests on."""
    labels = {
        "optimizer": "Optimizer",
        "objective_mode": "Objective",
        "covariance_estimator": "Covariance estimator",
        "ewma_lambda": "EWMA λ",
        "expected_returns_method": "Expected-returns method",
        "risk_free_rate": "Risk-free rate",
        "risk_aversion": "Risk aversion λ",
        "target_return": "Target return",
        "target_volatility": "Target volatility",
        "periods_per_year": "Periods per year",
        "base_currency": "Base currency",
        "sample_start": "Sample start",
        "sample_end": "Sample end",
        "n_observations": "Observations",
        "n_assets": "Assets",
        "long_only": "Long only",
        "fully_invested": "Fully invested",
        "leverage_cap": "Gross-exposure cap",
        "turnover_limit": "Turnover budget",
        "solver": "Solver",
        "solver_status": "Solver status",
    }
    rows = [
        {"Assumption": labels.get(k, k), "Value": _fmt_assumption(k, v)}
        for k, v in assumptions.items()
        if v is not None
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _fmt_assumption(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key in ("risk_free_rate", "target_return", "target_volatility") and isinstance(
        value, (int, float)
    ):
        return f"{value:.2%}"
    return str(value)


def render_frontier_health(frontier) -> None:
    """Report failed sweep points and the reachable return range."""
    if frontier is None:
        return
    if frontier.reachable_range is not None:
        lo, hi = frontier.reachable_range
        gmv = (
            frontier.min_variance["expected_return"]
            if frontier.min_variance is not None
            else None
        )
        caption = (
            f"Under these constraints, expected return can range from "
            f"{lo:.2%} to {hi:.2%}"
        )
        if gmv is not None:
            caption += f"; only the part above {gmv:.2%} is efficient."
        else:
            caption += "."
        st.caption(caption)
    if frontier.n_failed:
        with st.expander(
            f"{frontier.n_failed} frontier point(s) did not solve", expanded=False
        ):
            for f in frontier.failures[:10]:
                st.text(f)


# ---------------------------------------------------------------------------
# Stress scenarios
# ---------------------------------------------------------------------------

#: Columns of the long-form shock grid. One row is one ``(scenario, asset)``
#: move, which is the shape a scenario is actually written in — "equities
#: −32%, treasuries +8%" — rather than a dense asset × scenario matrix that is
#: mostly zeros and cannot hold a name the book does not carry.
SHOCK_TABLE_COLUMNS = ("Scenario", "Asset", "Shock return")

#: Columns of the per-scenario grid beside it. These are properties of the
#: scenario, not of any one leg, so repeating them on every row would let two
#: rows of the same scenario disagree about the same fact.
SCENARIO_TABLE_COLUMNS = ("Covariance ×", "Notes")


def empty_shock_table() -> pd.DataFrame:
    """An empty long-form shock grid with the right dtypes.

    Returns:
        A zero-row frame carrying :data:`SHOCK_TABLE_COLUMNS`. The dtypes
        matter: an object column would let Streamlit's editor offer a text box
        where a number belongs.
    """
    return pd.DataFrame(
        {
            "Scenario": pd.Series(dtype="string"),
            "Asset": pd.Series(dtype="string"),
            "Shock return": pd.Series(dtype="float"),
        }
    )


def empty_scenario_table() -> pd.DataFrame:
    """An empty per-scenario grid, indexed by scenario name."""
    return pd.DataFrame(
        {
            "Covariance ×": pd.Series(dtype="float"),
            "Notes": pd.Series(dtype="string"),
        },
        index=pd.Index([], dtype="object", name="Scenario"),
    )


def _clean_text(value: Any) -> str:
    """A cell as trimmed text, with every flavour of blank reading as ``""``."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def align_scenario_table(
    shock_rows: pd.DataFrame, previous: pd.DataFrame | None
) -> pd.DataFrame:
    """The per-scenario grid for whatever scenarios the shock grid now names.

    Rebuilt from the shock grid on every run rather than edited in parallel
    with it: a scenario renamed in one table and not the other is how a
    covariance multiplier ends up attached to a scenario nobody is running.
    Values already set survive by name, new scenarios arrive blank, and a
    scenario whose last leg was deleted drops out.

    Args:
        shock_rows: The long-form grid.
        previous: The per-scenario grid as it was, or ``None``.

    Returns:
        A frame indexed by the scenario names present in ``shock_rows``, in
        first-seen order, carrying :data:`SCENARIO_TABLE_COLUMNS`.
    """
    names: list[str] = []
    for _, row in shock_rows.iterrows():
        name = _clean_text(row.get("Scenario"))
        if name and name not in names:
            names.append(name)
    frame = empty_scenario_table().reindex(pd.Index(names, name="Scenario"))
    frame["Notes"] = frame["Notes"].astype("object")
    if previous is not None and len(previous.index):
        for name in names:
            if name in previous.index:
                row = previous.loc[name]
                frame.loc[name, "Covariance ×"] = row.get("Covariance ×")
                frame.loc[name, "Notes"] = _clean_text(row.get("Notes"))
    frame["Notes"] = frame["Notes"].fillna("")
    return frame


def shock_dicts_from_tables(
    shock_rows: pd.DataFrame, scenario_rows: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    """The two grids as the plain mappings ``Shock.from_dict`` reads.

    Pure, and deliberately not validating: it reports what the analyst typed
    and lets :mod:`optimization_engine.stress` say whether it is a scenario.
    A row missing a scenario name, an asset or a number is not a scenario with
    a hole in it — it is a row still being typed — so it is skipped.

    Args:
        shock_rows: The long-form grid.
        scenario_rows: The per-scenario grid, or ``None``.

    Returns:
        One mapping per scenario, in first-seen order, with ``returns`` in the
        order the legs were typed. A leg naming an asset twice in the same
        scenario keeps the last value, which is what an editor's user means by
        typing it again.
    """
    ordered: list[str] = []
    returns: dict[str, dict[str, float]] = {}
    for _, row in shock_rows.iterrows():
        name = _clean_text(row.get("Scenario"))
        asset = _clean_text(row.get("Asset"))
        raw = row.get("Shock return")
        if not name or not asset or raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        if name not in returns:
            ordered.append(name)
            returns[name] = {}
        returns[name][asset] = value

    payload: list[dict[str, Any]] = []
    for name in ordered:
        entry: dict[str, Any] = {
            "name": name,
            "returns": returns[name],
            "covariance_scale": None,
            "notes": "",
        }
        if scenario_rows is not None and name in scenario_rows.index:
            row = scenario_rows.loc[name]
            scale = row.get("Covariance ×")
            if scale is not None and not pd.isna(scale):
                entry["covariance_scale"] = float(scale)
            entry["notes"] = _clean_text(row.get("Notes"))
        payload.append(entry)
    return payload


def tables_from_shocks(shocks: Iterable[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The inverse: scenarios back into the two grids.

    Used by the YAML upload and by loading a saved preset, so a scenario that
    arrived as a file is edited in the same grid as one typed by hand.

    Args:
        shocks: :class:`~optimization_engine.stress.Shock` objects, or
            anything exposing ``name``, ``returns``, ``covariance_scale`` and
            ``notes``.

    Returns:
        ``(shock_rows, scenario_rows)``. A scenario whose ``covariance_scale``
        is a full matrix comes back with a blank multiplier — the grid holds a
        scalar, and silently flattening a matrix to one number would be a
        different scenario wearing the same name.
    """
    rows: list[dict[str, Any]] = []
    meta: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for shock in shocks:
        name = str(getattr(shock, "name", "")).strip()
        if not name:
            continue
        for asset, value in dict(getattr(shock, "returns", {}) or {}).items():
            rows.append(
                {
                    "Scenario": name,
                    "Asset": str(asset),
                    "Shock return": float(value),
                }
            )
        scale = getattr(shock, "covariance_scale", None)
        meta[name] = {
            "Covariance ×": (
                float(scale) if isinstance(scale, (int, float)) and not isinstance(scale, bool)
                else np.nan
            ),
            "Notes": str(getattr(shock, "notes", "") or ""),
        }
        if name not in order:
            order.append(name)

    shock_rows = (
        pd.DataFrame(rows, columns=list(SHOCK_TABLE_COLUMNS))
        if rows
        else empty_shock_table()
    )
    scenario_rows = empty_scenario_table().reindex(pd.Index(order, name="Scenario"))
    scenario_rows["Notes"] = scenario_rows["Notes"].astype("object")
    for name in order:
        scenario_rows.loc[name, "Covariance ×"] = meta[name]["Covariance ×"]
        scenario_rows.loc[name, "Notes"] = meta[name]["Notes"]
    scenario_rows["Notes"] = scenario_rows["Notes"].fillna("")
    return shock_rows, scenario_rows


def unheld_shocked_assets(
    shock_dicts: Iterable[dict[str, Any]], held: Iterable[Any]
) -> dict[str, list[str]]:
    """Which scenarios name assets this book cannot hold, and which names.

    The same question :func:`~optimization_engine.stress.stress_test` refuses
    on, asked ahead of it so the page can warn before the run rather than only
    explain afterwards. It does not decide anything: the refusal is still the
    library's.

    Args:
        shock_dicts: Scenario mappings, as :func:`shock_dicts_from_tables`
            produces.
        held: The book's assets.

    Returns:
        ``scenario -> sorted unheld names``, only for scenarios that name one.
    """
    universe = {str(a) for a in held}
    out: dict[str, list[str]] = {}
    for entry in shock_dicts:
        missing = sorted(
            {str(a) for a in (entry.get("returns") or {})} - universe
        )
        if missing:
            out[str(entry.get("name", ""))] = missing
    return out


def validated_shock_dicts(
    shock_dicts: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split scenario mappings into the ones that are scenarios and the rest.

    A grid is edited a cell at a time, so at any moment some row may not yet
    describe anything — a covariance multiplier typed as ``-2`` while the
    minus sign is still the only character in the box. Letting that reach
    :class:`~optimization_engine.config.EngineConfig` would make the whole
    page fall over on a keystroke, and dropping it without a word would be the
    silent-swallow this library refuses everywhere else. So it is separated
    out and *named*, and the caller is expected to show the message.

    Note what this does **not** check: whether the book can hold the assets a
    scenario names. That is
    :func:`~optimization_engine.stress.stress_test`'s refusal to make, not
    this function's, and pre-empting it here would move a decision out of the
    library and into a text box.

    Args:
        shock_dicts: Scenario mappings, as :func:`shock_dicts_from_tables`
            produces.

    Returns:
        ``(usable, problems)`` — the mappings that build a
        :class:`~optimization_engine.stress.Shock`, in order, and one sentence
        per mapping that does not.
    """
    from optimization_engine.stress import Shock, StressError

    usable: list[dict[str, Any]] = []
    problems: list[str] = []
    for entry in shock_dicts:
        try:
            Shock.from_dict(entry)
        except StressError as exc:
            problems.append(f"{entry.get('name') or '(unnamed)'}: {exc}")
        else:
            usable.append(entry)
    return usable, problems


def render_stress_report(report: Any) -> None:
    """A stress report: the worst case, every scenario, and what drove it.

    Args:
        report: A :class:`~optimization_engine.stress.StressReport`.
    """
    if report is None:
        return
    worst = report.worst
    metric_row(
        [
            ("Worst scenario", worst.name, "Lowest P&L of the scenarios applied."),
            (
                "Worst-case P&L",
                pct(worst.pnl),
                "As a fraction of book value, over one period.",
            ),
            (
                "Largest contributor",
                str(worst.largest_contributor or "—"),
                f"{pct(worst.largest_contribution)} of book value, on its own.",
            ),
            (
                "Volatility under it",
                pct(worst.stressed_volatility)
                if worst.stressed_volatility is not None
                else "—",
                "The book's volatility under the scenario's own covariance"
                + (
                    f" — ×{worst.volatility_ratio:.2f} the unstressed figure."
                    if worst.volatility_ratio is not None
                    else "."
                ),
            ),
        ]
    )

    st.markdown("**Every scenario, worst first**")
    frame = report.to_frame()
    display = frame.copy()
    for col in ("pnl", "stressed_volatility", "base_volatility", "largest_contribution"):
        if col in display:
            display[col] = display[col].map(
                lambda v: "—" if pd.isna(v) else f"{v:.2%}"
            )
    if "volatility_ratio" in display:
        display["volatility_ratio"] = display["volatility_ratio"].map(
            lambda v: "—" if pd.isna(v) else f"×{v:.2f}"
        )
    display = display.rename(
        columns={
            "pnl": "P&L",
            "stressed_volatility": "Stressed vol",
            "base_volatility": "Base vol",
            "volatility_ratio": "Vol ratio",
            "largest_contributor": "Largest contributor",
            "largest_contribution": "Its contribution",
            "ignored_assets": "Dropped (not held)",
            "notes": "Notes",
        }
    )
    st.dataframe(display, width="stretch")

    dropped = [s for s in report.scenarios if s.ignored_assets]
    if dropped:
        st.warning(
            "These scenarios were applied with names this book does not hold "
            "dropped, so the loss shown is smaller than the scenario "
            "describes: "
            + " · ".join(
                f"**{s.name}** ({', '.join(s.ignored_assets)})" for s in dropped
            )
        )

    st.markdown("**What produced the worst case**")
    st.caption(
        f"Each bar is wᵢ·rᵢ. They sum to {worst.pnl:.2%} exactly — that is an "
        "identity, not an approximation, which is what makes the loss "
        "attributable to a position rather than merely reported."
    )
    st.plotly_chart(
        plot_stress_contributions(worst.contributions, worst.name),
        width="stretch",
    )

    with st.expander("Per-asset contributions, every scenario", expanded=False):
        contributions = report.contributions_frame()
        st.dataframe(contributions.style.format("{:.2%}"), width="stretch")
        st.caption(
            "Rows are scenarios worst-first, columns are positions. Every row "
            "sums to that scenario's P&L."
        )

    with st.expander("The report as text", expanded=False):
        st.text(report.describe())


def plot_stress_contributions(contributions: pd.Series, scenario: str) -> Any:
    """Signed P&L contributions for one scenario, largest loss first.

    Args:
        contributions: ``asset -> wᵢ·rᵢ``, in fraction-of-book units.
        scenario: The scenario's name, for the title.

    Returns:
        A Plotly figure.
    """
    import plotly.graph_objects as go

    ordered = contributions.sort_values()
    values = ordered.to_numpy(dtype=float) * 100.0
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[str(a) for a in ordered.index],
            orientation="h",
            marker_color=[LOSS_COLOR if v < 0 else GAIN_COLOR for v in values],
            hovertemplate="%{y}: %{x:.2f}% of book<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"P&L contributions — {scenario}",
        xaxis_title="% of book value",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
        height=max(280, 26 * len(ordered) + 120),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Point-in-time universe
# ---------------------------------------------------------------------------

#: Numeric encoding of the three states for the heatmap. Ordered
#: ineligible → unknown → eligible, but drawn as three flat bands rather than
#: a ramp, so "not evaluable" never reads as "half eligible".
_STATE_CODES = {False: 0, None: 1, True: 2}


def eligibility_state_codes(frame: pd.DataFrame) -> pd.DataFrame:
    """A three-valued eligibility frame as integer state codes.

    Pure, and the single place the encoding lives, so the heatmap's colours
    and the counts underneath it cannot disagree about which code is which.

    Args:
        frame: A ``boolean``-dtype ``date × asset`` frame, as carried by an
            :class:`~optimization_engine.universe.eligibility.Eligibility`.

    Returns:
        An ``int`` frame on the same axes: ``0`` not eligible, ``1`` not
        evaluable, ``2`` eligible — the indices of
        :data:`ELIGIBILITY_STATES`.
    """
    values = frame.to_numpy(dtype=object)
    codes = np.full(values.shape, _STATE_CODES[None], dtype=int)
    is_missing = pd.isna(frame).to_numpy(dtype=bool)
    truthy = np.zeros(values.shape, dtype=bool)
    truthy[~is_missing] = values[~is_missing].astype(bool)
    codes[~is_missing & truthy] = _STATE_CODES[True]
    codes[~is_missing & ~truthy] = _STATE_CODES[False]
    return pd.DataFrame(codes, index=frame.index, columns=frame.columns)


def eligibility_state_counts(frame: pd.DataFrame) -> dict[str, int]:
    """How many cells are in each of the three states.

    Args:
        frame: A three-valued eligibility frame.

    Returns:
        ``{state label: count}`` over every label in
        :data:`ELIGIBILITY_STATES`, zeros included — a state that never occurs
        is a fact about the universe, not a row to omit.
    """
    codes = eligibility_state_codes(frame).to_numpy()
    return {
        label: int((codes == index).sum())
        for index, label in enumerate(ELIGIBILITY_STATES)
    }


def thin_rows(frame: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, int]:
    """Every ``n``-th row of a frame, keeping the last one.

    A daily universe over eight years is two thousand rows against thirteen
    columns; drawing every one of them makes a figure that is slower to render
    and no easier to read. Sampling — rather than aggregating — is deliberate:
    any rule for collapsing a fortnight of states into one cell has to decide
    what a fortnight containing both an eligible and an unknown day *is*, and
    every answer to that misrepresents something. A sampled row is a row that
    really occurred, and the caller says so.

    Args:
        frame: The frame to thin.
        max_rows: The most rows to keep. Below 1 is treated as 1.

    Returns:
        ``(thinned, step)``. ``step`` is 1 when nothing was dropped.
    """
    limit = max(int(max_rows), 1)
    if len(frame.index) <= limit:
        return frame, 1
    step = int(np.ceil(len(frame.index) / limit))
    kept = frame.iloc[::step]
    if frame.index[-1] not in kept.index:
        kept = pd.concat([kept, frame.iloc[[-1]]])
    return kept, step


def plot_eligibility_heatmap(
    frame: pd.DataFrame, title: str = "Eligibility"
) -> Any:
    """The eligibility frame as a three-state heatmap.

    Args:
        frame: A three-valued ``date × asset`` frame.
        title: Figure title.

    Returns:
        A Plotly figure whose colour scale has exactly three flat bands and a
        colour bar labelled in words.
    """
    import plotly.graph_objects as go

    codes = eligibility_state_codes(frame)
    labels = np.array(ELIGIBILITY_STATES, dtype=object)[codes.to_numpy()]
    # Flat bands: each colour is repeated across its third of the scale, so
    # nothing is interpolated between two states that have no midpoint.
    colorscale = []
    for index, color in enumerate(ELIGIBILITY_COLORS):
        colorscale.append([index / len(ELIGIBILITY_COLORS), color])
        colorscale.append([(index + 1) / len(ELIGIBILITY_COLORS), color])
    fig = go.Figure(
        go.Heatmap(
            z=codes.to_numpy(),
            x=[str(c) for c in codes.columns],
            y=codes.index,
            customdata=labels,
            colorscale=colorscale,
            zmin=-0.5,
            zmax=2.5,
            xgap=1,
            ygap=0,
            colorbar=dict(
                tickmode="array",
                tickvals=list(range(len(ELIGIBILITY_STATES))),
                ticktext=list(ELIGIBILITY_STATES),
                title="",
            ),
            hovertemplate="%{x}<br>%{y|%Y-%m-%d}: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
        height=520,
    )
    return fig


def plot_universe_breadth(breadth: pd.Series, unknown: pd.Series) -> Any:
    """Eligible names and unevaluated names per date, on one axis.

    Drawn together on purpose: breadth alone looks like a universe shrinking,
    and the unknown count is what says whether it shrank or was never
    measured.

    Args:
        breadth: Eligible names per date.
        unknown: Not-evaluable names per date.

    Returns:
        A Plotly figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=breadth.index, y=breadth.to_numpy(), name="Eligible",
            mode="lines", line=dict(color=ELIGIBILITY_COLORS[2], width=2),
            hovertemplate="%{x|%Y-%m-%d}: %{y} eligible<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=unknown.index, y=unknown.to_numpy(), name="Not evaluable",
            mode="lines", line=dict(color=ELIGIBILITY_COLORS[1], width=2),
            hovertemplate="%{x|%Y-%m-%d}: %{y} not evaluable<extra></extra>",
        )
    )
    fig.update_layout(
        title="Breadth, and what nobody evaluated",
        yaxis_title="Names",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
        height=320,
    )
    return fig


def plot_universe_turnover(turnover: pd.DataFrame) -> Any:
    """Entries and exits per date as opposed bars.

    Args:
        turnover: The frame from
            :meth:`~optimization_engine.universe.eligibility.Eligibility.turnover`,
            with ``entries``, ``exits`` and ``turnover`` columns.

    Returns:
        A Plotly figure. Exits are drawn below the axis so a reconstitution
        that swapped one name for another reads as a swap rather than as two
        unrelated events.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=turnover.index, y=turnover["entries"].to_numpy(), name="Entries",
            marker_color=ELIGIBILITY_COLORS[2],
            hovertemplate="%{x|%Y-%m-%d}: %{y} entered<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=turnover.index, y=-turnover["exits"].to_numpy(), name="Exits",
            marker_color=LOSS_COLOR,
            hovertemplate="%{x|%Y-%m-%d}: %{customdata} left<extra></extra>",
            customdata=turnover["exits"].to_numpy(),
        )
    )
    fig.update_layout(
        title="Membership changes",
        yaxis_title="Names (exits below the axis)",
        barmode="relative",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
        height=300,
    )
    return fig


def describe_policy_cost(
    policy: str | None, cells: int, bars: int, names: Iterable[Any]
) -> str:
    """One sentence naming what the collapse policy — not a screen — decided.

    The library refuses to pick a policy; a page that picks one owes the
    reader the size of what it picked. This is the app's copy of the sentence
    the CLI prints on stderr, for the same reason and in the same words.

    Args:
        policy: ``"exclude"``, ``"include"``, ``"raise"``, or ``None`` when the
            reader has not chosen yet.
        cells: How many ``(bar, asset)`` cells the policy decides.
        bars: Over how many bars they are spread.
        names: The assets any of them touch.

    Returns:
        The sentence.
    """
    listed = [str(n) for n in names]
    if policy is None:
        if not cells:
            return (
                "Every date/asset cell was evaluable, so whichever policy you "
                "choose decides nothing here."
            )
        return (
            f"{cells:,} date/asset cell(s) across {bars:,} bar(s) and "
            f"{len(listed)} name(s) were not evaluable. Choose a policy above "
            "to say what they mean; until then nothing collapses them."
        )
    if not cells:
        return (
            "Every date/asset cell was evaluable, so the "
            f"{policy!r} policy decides nothing."
        )
    verdict = {
        "exclude": "reads them as ineligible",
        "include": "admits them",
        "raise": "will stop any run on them",
    }[policy]
    shown = ", ".join(listed[:5]) + (" …" if len(listed) > 5 else "")
    return (
        f"{cells:,} date/asset cell(s) across {bars:,} bar(s) and "
        f"{len(listed)} name(s) were not evaluable, and the {policy!r} "
        f"policy — not a screen — {verdict}: {shown}."
    )


def render_universe_notes(notes: Any) -> None:
    """What a run recorded about the universe it was run under.

    Reads ``meta.notes["universe"]`` and ``meta.notes["delistings"]``, which
    are where the runner writes down the things a return series cannot show:
    how wide the investable set was at each decision, which held names the
    universe forced out, which names it had never heard of, and which stopped
    printing.

    Args:
        notes: The run's ``meta.notes`` mapping.
    """
    notes = dict(notes or {})
    block = notes.get("universe")
    if not isinstance(block, dict):
        st.info(
            "This run recorded no universe notes, which means it was run on "
            "the panel's columns — the survivors of today's data."
        )
        return

    breadth = block.get("breadth") or {}
    liquidated = block.get("liquidated") or {}
    unknown_assets = list(block.get("unknown_assets") or [])
    metric_row(
        [
            ("Collapse policy", str(block.get("policy", "—")), None),
            (
                "Decisions",
                f"{int(block.get('n_decisions', 0)):,}",
                "Dates a target was chosen under the universe.",
            ),
            (
                "Narrowest decision",
                f"{int(block.get('min_breadth', 0)):,} names",
                "The fewest eligible names any decision had to work with.",
            ),
            (
                "Forced liquidations",
                f"{int(block.get('n_liquidations', 0)):,}",
                "Times a held position was sold because the universe no "
                "longer admitted it.",
            ),
        ]
    )

    if unknown_assets:
        st.warning(
            "The universe says nothing at all about "
            f"{', '.join(str(a) for a in unknown_assets)}. Those names were "
            "resolved by the policy rather than by any rule."
        )

    if liquidated:
        st.markdown("**Names the universe took out of the book**")
        st.dataframe(
            pd.DataFrame(
                {"Times liquidated": pd.Series(liquidated)}
            ).sort_values("Times liquidated", ascending=False),
            width="stretch",
        )

    if breadth:
        series = pd.Series(
            {pd.Timestamp(k): int(v) for k, v in breadth.items()}
        ).sort_index()
        st.markdown("**Eligible names at each decision**")
        st.line_chart(series.rename("Eligible names"))

    delistings = notes.get("delistings")
    if isinstance(delistings, dict):
        if delistings:
            st.markdown("**Delistings**")
            st.caption(
                "A name that stopped printing for longer than the grace "
                "period. Separate from the screen: the universe says what the "
                "mandate permits, this says what still trades."
            )
            st.dataframe(
                pd.DataFrame(delistings).T.rename(
                    columns={
                        "last_print": "Last print",
                        "delisted_at": "Declared delisted",
                    }
                ),
                width="stretch",
            )
        else:
            st.caption(
                "Delisting was diagnosed and nothing was found: every name "
                "kept printing for the whole run."
            )
