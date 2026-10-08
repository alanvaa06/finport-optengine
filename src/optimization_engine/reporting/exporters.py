"""Excel exporter for optimization results.

The workbook is meant to be read by someone who was not in the room when the
optimization ran, so :func:`run_sheets` assembles the assumptions, the
estimation diagnostics and the data-quality findings alongside the weights.
A weights tab on its own is a number without its provenance.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from optimization_engine._optional import require

#: Excel refuses sheet names longer than this.
_SHEET_NAME_LIMIT = 31


def write_excel_report(
    path: str | Path, sheets: Mapping[str, pd.DataFrame | pd.Series]
) -> Path:
    """Write a multi-sheet Excel workbook from a mapping of name → frame.

    ``None`` entries are skipped, Series are promoted to single-column frames,
    and names are truncated to Excel's 31-character limit. Truncation can
    collide, so a numeric suffix is appended rather than letting one sheet
    silently overwrite another.

    Args:
        path: Where to write the ``.xlsx``.
        sheets: ``sheet name -> frame``, in the order they should appear.

    Returns:
        The path written, as a :class:`~pathlib.Path`.

    Raises:
        MissingDependencyError: If xlsxwriter is not installed. Install it
            with ``finport-optengine[excel]``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    # Checked before opening the writer: pandas resolves the engine by
    # name and its own ImportError names xlsxwriter without saying which
    # extra ships it.
    require("xlsxwriter", extra="excel", purpose="writing Excel workbooks")
    with pd.ExcelWriter(p, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if df is None:
                continue
            if isinstance(df, pd.Series):
                df = df.to_frame()
            sheet = unique_sheet_name(name, used)
            used.add(sheet)
            df.to_excel(writer, sheet_name=sheet, index=True)
    return p


def unique_sheet_name(name: str, used: set[str]) -> str:
    """Excel-safe sheet name for ``name``, distinct from everything in ``used``.

    Public because the UI writes its own workbook and needs the same
    de-duplication: two names that differ only past character 31 would
    otherwise land on the same sheet and one would be lost.

    Args:
        name: The desired sheet name. Truncated to Excel's 31-character limit
            and stripped of the characters Excel rejects.
        used: Names already taken. Not mutated — add the result yourself.

    Returns:
        A name that is valid for Excel and not in ``used``, disambiguated with
        a numeric suffix if needed.

    Raises:
        ValueError: If no unique name could be found within the length limit.
    """
    base = str(name)[:_SHEET_NAME_LIMIT]
    if base not in used:
        return base
    for i in range(2, 100):
        suffix = f"_{i}"
        candidate = base[: _SHEET_NAME_LIMIT - len(suffix)] + suffix
        if candidate not in used:
            return candidate
    raise ValueError(f"Cannot find a unique Excel sheet name for {name!r}.")


#: Kept for callers that imported the private name.
_unique_sheet_name = unique_sheet_name


def performance_sheets(report, prefix: str = "") -> dict[str, pd.DataFrame]:
    """Workbook contents for one :class:`PerformanceReport`.

    Args:
        report: The report to lay out.
        prefix: Prepended to every sheet name, so an out-of-sample report can
            sit alongside the fitted one without either overwriting the other.
    """
    frames = report.to_frames()
    if not prefix:
        return dict(frames)
    return {f"{prefix}{name}": frame for name, frame in frames.items()}


def run_sheets(
    run,
    riskfree_rate: float = 0.0,
    data_quality=None,
    walk_forward=None,
    frontier_uncertainty=None,
    performance=None,
) -> dict[str, pd.DataFrame]:
    """Assemble the standard workbook contents for an :class:`EngineRun`.

    Shared by the CLI and the Streamlit app so both produce the same report.

    Args:
        run: The :class:`~optimization_engine.engine.EngineRun` to describe.
        riskfree_rate: Annual rate used for the performance summary.
        data_quality: Optional :class:`DataQualityReport` for the input panel.
        walk_forward: Optional :class:`WalkForwardResult` to include the
            out-of-sample track record and its degradation table.
        frontier_uncertainty: Optional :class:`FrontierUncertainty` to include
            the resampled confidence band and per-asset weight dispersion.
        performance: Optional :class:`PerformanceReport`. When the run has a
            benchmark and none is passed, one is built from the run so that a
            workbook exported without extra arguments still carries the
            relative numbers.

    Returns:
        ``sheet name -> frame``. When a performance report cannot be built —
        a benchmark sharing no dates with the walk-forward, say — its sheets
        are left out, an ``omitted_sheets`` sheet names them with the reason,
        and a ``UserWarning`` is issued.
    """
    omitted: list[dict[str, str]] = []
    sheets: dict[str, pd.DataFrame] = {
        "weights": run.result.weights.to_frame("weight"),
        "summary": pd.DataFrame(
            [
                {
                    "expected_return": run.result.expected_return,
                    "expected_volatility": run.result.expected_volatility,
                    "sharpe_ratio": run.result.sharpe_ratio,
                }
            ]
        ),
        "assumptions": pd.DataFrame(
            [{"assumption": k, "value": v} for k, v in run.assumptions().items()]
        ),
        "risk_decomposition": run.risk_decomposition(),
        "expected_returns": run.expected_returns.to_frame("annualized"),
        "cov_matrix": run.cov_matrix,
        "absolute_summary": run.absolute_summary(
            riskfree_rate=riskfree_rate, extended=True
        ),
    }

    if run.benchmark is not None:
        sheets["benchmark"] = run.benchmark.summary()
        benchmark_weights = run.benchmark.weights_frame()
        if benchmark_weights is not None:
            # The active weights are the whole point of showing the benchmark
            # in a workbook, so they are computed here rather than left to the
            # reader's spreadsheet arithmetic.
            benchmark_weights = benchmark_weights.copy()
            portfolio = run.result.weights.reindex(benchmark_weights.index).fillna(0.0)
            benchmark_weights["portfolio_weight"] = portfolio
            benchmark_weights["active_weight"] = (
                portfolio - benchmark_weights["benchmark_weight"]
            )
            sheets["benchmark_weights"] = benchmark_weights
        if performance is None:
            try:
                performance = run.performance(riskfree_rate=riskfree_rate)
            except (ValueError, KeyError) as exc:
                # A benchmark that shares no dates with the panel must not take
                # the whole workbook down with it — but the sheets it costs are
                # named, not silently missing.
                performance = None
                _omit(omitted, "performance_*", exc)
    if performance is not None:
        sheets.update(performance_sheets(performance))

    exposures = run.layer_exposures()
    if not exposures.empty:
        # The policy as realized, not as written: which bucket on which layer
        # actually stopped the optimizer is the first thing a committee asks.
        sheets["allocation_layers"] = exposures.set_index(["layer", "bucket"])

    if run.diagnostics is not None:
        sheets["portfolio_diagnostics"] = pd.DataFrame(
            [
                {"metric": k, "value": v}
                for k, v in run.diagnostics.to_dict().items()
                if not isinstance(v, list)
            ]
        )
    if run.covariance_diagnostics is not None:
        diag = run.covariance_diagnostics
        sheets["estimation_diagnostics"] = pd.DataFrame(
            [
                {"metric": "n_assets", "value": diag.n_assets},
                {"metric": "n_observations", "value": diag.n_observations},
                {"metric": "observations_per_asset", "value": diag.observations_per_asset},
                {"metric": "condition_number", "value": diag.condition_number},
                {"metric": "min_eigenvalue", "value": diag.min_eigenvalue},
                {"metric": "effective_observations", "value": diag.effective_observations},
                *[{"metric": "warning", "value": w} for w in diag.warnings],
            ]
        )
    if run.feasibility is not None and run.feasibility.issues:
        sheets["feasibility"] = pd.DataFrame(
            [
                {
                    "severity": "fatal" if i.fatal else "warning",
                    "finding": i.message,
                    "suggestion": i.suggestion,
                }
                for i in run.feasibility.issues
            ]
        )
    if data_quality is not None:
        sheets["data_quality"] = (
            pd.DataFrame(
                [
                    {
                        "severity": i.severity,
                        "asset": i.asset or "",
                        "finding": i.message,
                        "suggestion": i.suggestion,
                    }
                    for i in data_quality.issues
                ]
            )
            if data_quality.issues
            else pd.DataFrame([{"severity": "info", "finding": "No issues found."}])
        )
    if run.frontier is not None:
        sheets["frontier_summary"] = run.frontier.summary
        sheets["frontier_weights"] = run.frontier.weights
        if (
            run.frontier.group_weights is not None
            and not run.frontier.group_weights.empty
        ):
            sheets["frontier_groups"] = run.frontier.group_weights
    if frontier_uncertainty is not None:
        sheets["frontier_uncertainty"] = frontier_uncertainty.quantiles
        sheets["weight_dispersion"] = frontier_uncertainty.weight_dispersion.to_frame(
            "weight_sd_across_draws"
        )
    if walk_forward is not None:
        sheets["walk_forward_weights"] = walk_forward.weights_history
        sheets["walk_forward_windows"] = walk_forward.windows
        sheets["in_vs_out_of_sample"] = run.in_vs_out_of_sample(
            walk_forward, riskfree_rate
        )
        if run.benchmark is not None:
            # The out-of-sample track record measured against the same
            # benchmark. This is the pair a reader should compare — a fitted
            # information ratio next to a walk-forward one — and putting them
            # in the same workbook is what makes the comparison happen.
            try:
                sheets.update(
                    performance_sheets(
                        run.performance(
                            riskfree_rate=riskfree_rate,
                            returns_override=walk_forward.returns,
                        ),
                        prefix="oos_",
                    )
                )
            except (ValueError, KeyError) as exc:
                _omit(omitted, "oos_performance_*", exc)
    if omitted:
        sheets["omitted_sheets"] = pd.DataFrame(omitted, columns=["sheets", "reason"])
    return sheets


def _omit(omitted: list[dict[str, str]], sheets: str, exc: Exception) -> None:
    """Record sheets the workbook leaves out, and warn, so the gap is visible.

    These used to be dropped with nothing to show for it: a reader of the
    workbook could not tell a report with no relative panel from one whose
    relative panel failed.
    """
    reason = f"{type(exc).__name__}: {exc}"
    omitted.append({"sheets": sheets, "reason": reason})
    warnings.warn(
        f"The report omits the {sheets} sheets: {reason}", UserWarning, stacklevel=3
    )
