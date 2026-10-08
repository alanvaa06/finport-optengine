"""JSON payloads for machine consumers.

The Excel report and the CLI's printed tables are written for a person who
will read them. This module is the other audience: a script, a pipeline, or
an agent that has to *act* on a result and cannot parse a formatted table
without guessing.

The distinction that matters is a contract versus a dump. Serialising the
result objects with ``dataclasses.asdict`` would be a dump: every internal
rename becomes a silent breaking change for anyone parsing it, and the shape
depends on which optional pieces happened to be computed. So each function
here builds an explicit dictionary, keys are chosen for the reader rather
than inherited from the attribute they came from, and a key that is absent
this run is present as ``null`` rather than missing — a consumer can test a
value, never a key.

Every payload carries ``schema_version``. It is the only promise this module
makes: within a major version, a key that exists keeps its meaning and keys
are only added. See ``SCHEMA_VERSION``.

Floats are plain Python floats and never NumPy scalars, because
``json.dumps`` cannot serialise the latter and the failure surfaces at the
worst moment — after a long solve, with the result already computed.
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import numpy as np
import pandas as pd

#: Bumped major on a breaking change to any payload below, minor when keys
#: are added. Consumers should check the major and refuse a version they do
#: not know rather than parsing optimistically.
#:
#: ``2.1`` adds ``audit`` to the optimize payload — the mandate audit, as
#: structured violations rather than the sentences ``diagnostics.violations``
#: has always carried.
#:
#: ``2.2`` adds ``stress`` to the optimize and backtest payloads. ``--stress``
#: already reached the console and the workbook; a machine caller reading
#: ``--json`` could not see the scenarios at all, which made the one number
#: worth automating an alert on — the worst case — the one number the
#: structured output omitted.
#:
#: ``2.3`` adds four keys. ``data_source``, on the check, optimize and
#: backtest payloads: which panel the numbers were computed on, and whether
#: it is synthetic. The CLI used to fall back to the built-in sample when no
#: price file was named, and nothing in the payload could tell those weights
#: apart from weights on market data. ``data_quality``, on optimize and
#: backtest, with ``usable``, ``clean`` and ``findings`` added inside the one
#: ``check`` already carried: a run on a panel with an error-level finding
#: exited 0 and its document never mentioned it; the findings reached stderr
#: only. ``ingest``, on all three: the request the panel was fetched with,
#: its window resolved — an ingest with no end date ends on the day it runs.
#: And ``condition_number_infinite``, on the covariance diagnostics: an
#: infinite condition number — a singular estimate — has to be ``null`` in
#: strict JSON, which is also what "not computed" looks like; the flag
#: tells the two apart.
SCHEMA_VERSION = "2.3"


def _num(value: Any) -> float | None:
    """A JSON-safe float, or ``None`` where JSON has no way to say it.

    NaN and infinity are the reason this exists. Both are legal Python
    floats and legal in ``json.dumps`` output by default, but ``NaN`` is not
    valid JSON — a strict parser on the other end rejects the whole
    document. A metric that could not be computed becomes ``null``.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _series(values: pd.Series | None) -> dict[str, float | None] | None:
    """A pandas Series as ``{label: value}``, keys stringified."""
    if values is None:
        return None
    return {str(k): _num(v) for k, v in values.items()}


def _key(value: Any) -> str:
    """A mapping key as a string, with dates in the same form as values.

    ``str`` on a Timestamp gives ``"2020-01-05 00:00:00"`` while
    ``_jsonable`` gives ``"2020-01-05T00:00:00"``, so a note keyed by date
    would spell its dates one way and its values another.
    """
    if isinstance(value, (pd.Timestamp, datetime.date, datetime.datetime)):
        return pd.Timestamp(value).isoformat()
    return str(value)


def _jsonable(value: Any) -> Any:
    """Any note value as something ``json.dumps`` will accept.

    Backtest notes are open-ended: the runner records counts, lists of dates
    and date-to-date mappings there, and nothing constrains what a caller
    adds. Timestamps become ISO strings, NumPy scalars become Python
    numbers, and containers are walked; anything left becomes ``str``, on
    the grounds that a readable approximation beats a document a strict
    parser rejects.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _num(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _num(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime.date, datetime.datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, dict):
        return {_key(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, pd.Index, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Series):
        return {_key(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _notes(notes: Any) -> dict[str, Any]:
    """Backtest notes as a JSON-safe mapping, values included.

    This used to run through ``_strings``, which iterates a mapping and so
    kept the keys and silently discarded every value — a reader learned that
    something had been recorded but never what.
    """
    if not notes:
        return {}
    if isinstance(notes, dict):
        return {_key(k): _jsonable(v) for k, v in notes.items()}
    return {str(n): None for n in notes}


def _strings(values: Any) -> list[str]:
    """Any iterable of message-ish things as a list of strings."""
    if not values:
        return []
    return [str(v) for v in values]


def portfolio_diagnostics_payload(diagnostics: Any) -> dict[str, Any] | None:
    """Concentration and exposure diagnostics.

    ``effective_n`` against ``effective_n_risk`` is the pair worth reading
    together: the first counts positions by their share of gross capital, the
    second by the size of their risk contribution — a hedge counts, whichever
    way it points — and the gap between them is what a weights table hides.
    Both lie between 1 and the number of assets.

    Args:
        diagnostics: A
            :class:`~optimization_engine.optimizers.diagnostics.PortfolioDiagnostics`,
            or ``None``.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if diagnostics is None:
        return None
    return {
        "n_positions": int(diagnostics.n_positions),
        "gross_exposure": _num(diagnostics.gross_exposure),
        "net_exposure": _num(diagnostics.net_exposure),
        "long_exposure": _num(diagnostics.long_exposure),
        "short_exposure": _num(diagnostics.short_exposure),
        "max_weight": _num(diagnostics.max_weight),
        "herfindahl": _num(diagnostics.herfindahl),
        "effective_n": _num(diagnostics.effective_n),
        "effective_n_risk": _num(diagnostics.effective_n_risk),
        "diversification_ratio": _num(diagnostics.diversification_ratio),
        "turnover": _num(diagnostics.turnover),
        "violations": _strings(diagnostics.violations),
    }


def audit_payload(report: Any) -> dict[str, Any] | None:
    """Whether the solved book obeys the mandate, breach by breach.

    ``diagnostics.violations`` already carried this, as prose. A consumer that
    wanted the size of a breach — to decide whether 20.4% against a 20% cap is
    worth a phone call — had to parse it back out of a sentence. Here each
    violation is an object with the limit, the actual figure and the distance
    between them as numbers, and the sentence is kept alongside as ``message``
    for whoever was only ever going to print it.

    ``clean`` is lifted to the top for the same reason ``feasible`` is on the
    feasibility payload: a consumer branches on one boolean rather than on the
    emptiness of a list. Note that a clean audit means every limit that *could*
    be checked was met — a tracking-error budget needs a covariance matrix, and
    a solve that had none did not look.

    Args:
        report: An
            :class:`~optimization_engine.optimizers.audit.AuditReport`, or
            ``None`` when the solve was asked not to run one.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if report is None:
        return None
    violations = [
        {
            "kind": str(getattr(violation, "kind", "")),
            "label": str(getattr(violation, "label", "")),
            "limit": _num(getattr(violation, "limit", None)),
            "actual": _num(getattr(violation, "actual", None)),
            "magnitude": _num(getattr(violation, "magnitude", None)),
            "message": str(
                violation.describe()
                if hasattr(violation, "describe")
                else violation
            ),
        }
        for violation in (getattr(report, "violations", None) or ())
    ]
    # The report's own verdict, not one derived from the list — the same
    # reasoning as `feasible` above, and cheap insurance against the day the
    # two stop coinciding.
    clean = getattr(report, "is_clean", None)
    return {
        "clean": bool(clean) if clean is not None else not violations,
        "tolerance": _num(getattr(report, "tolerance", None)),
        "violations": violations,
    }


def stress_payload(report: Any, *, as_of: Any = None) -> dict[str, Any] | None:
    """What each named scenario does to the book, worst case first.

    Structured for the same reason ``feasibility.issues`` and
    ``audit.violations`` are: the console prints
    :meth:`~optimization_engine.stress.StressReport.describe`, and a consumer
    that wanted the worst scenario's loss had to parse it back out of a
    sentence. Here every scenario is an object, ``pnl`` is a number, and
    ``contributions`` carries the per-asset decomposition that sums to it —
    the identity that makes the report auditable is preserved in the payload
    rather than only in the prose.

    ``worst`` is lifted to the top, like ``feasible`` and ``clean`` above, so a
    consumer can threshold on one number without sorting the list itself.

    ``scenarios`` is ordered worst-first, matching
    :meth:`~optimization_engine.stress.StressReport.by_severity` and the
    printed report; a consumer that wants the scenario's own declaration order
    has it in the config it supplied.

    Args:
        report: A :class:`~optimization_engine.stress.StressReport`, or
            ``None`` when no scenarios were configured or the run was not
            asked to apply them. The two are different claims, and neither is
            "the book is safe".
        as_of: The date of the book that was stressed, when it is a book from
            a point in a simulation rather than the solve's own weights.
            ``null`` means the weights this payload already carries.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if report is None:
        return None
    ordered = list(report.by_severity())
    worst = ordered[0] if ordered else None
    metadata = dict(getattr(report, "metadata", None) or {})
    stamp = as_of
    if isinstance(stamp, str):
        # The tearsheet records this date as ``str(Timestamp)``, which spells
        # it "2020-01-05 00:00:00" while every other date in these documents
        # is ISO. One document, one date format.
        try:
            stamp = pd.Timestamp(stamp)
        except (TypeError, ValueError):
            pass
    return {
        # Which book this is about. A tearsheet stresses the holdings the run
        # ended on, not the ones a solve produced, and a consumer comparing
        # two documents needs to know that before it compares the numbers.
        "as_of": _key(stamp) if stamp is not None else None,
        # The scenario that decides whether anyone reads the rest.
        "worst_scenario": str(worst.name) if worst is not None else None,
        "worst_pnl": _num(worst.pnl) if worst is not None else None,
        "worst_contributor": (
            str(worst.largest_contributor)
            if worst is not None and worst.largest_contributor is not None
            else None
        ),
        "base_volatility": _num(getattr(report, "base_volatility", None)),
        # What the run was told to do with a shock naming an asset the book
        # cannot hold: "raise" (nothing was dropped, because a scenario that
        # named an unheld name would have stopped the run) or "ignore" (some
        # may have been, and `ignored_assets` below says which).
        "unknown_asset_policy": str(metadata.get("unknown_assets", "raise")),
        "n_scenarios": len(ordered),
        "scenarios": [
            {
                "name": str(scenario.name),
                # A fraction of book value: -0.18 is "the book loses 18%".
                "pnl": _num(scenario.pnl),
                "stressed_volatility": _num(scenario.stressed_volatility),
                "base_volatility": _num(scenario.base_volatility),
                "volatility_ratio": _num(scenario.volatility_ratio),
                "largest_contributor": (
                    str(scenario.largest_contributor)
                    if scenario.largest_contributor is not None
                    else None
                ),
                "largest_contribution": _num(scenario.largest_contribution),
                # wᵢ·rᵢ per asset. Sums to `pnl` by construction, which is
                # what makes the number attributable rather than merely
                # reported.
                "contributions": _series(scenario.contributions),
                "ignored_assets": _strings(scenario.ignored_assets),
                "notes": str(scenario.notes or ""),
            }
            for scenario in ordered
        ],
    }


def data_quality_payload(quality: Any) -> dict[str, Any] | None:
    """What was wrong with the panel before anything was estimated from it.

    ``findings`` is the structured form — one object per issue, with the
    severity, the code and the asset as fields a consumer can filter on.
    ``errors`` and ``issues`` are the string lists ``check`` has always
    carried, kept with their meaning. ``usable`` is false when any finding is
    an error, which ``optimize`` and ``backtest`` proceed past unless
    ``--strict``; a consumer that wants the same rule reads that one boolean.

    Args:
        quality: A :class:`~optimization_engine.data.quality.DataQualityReport`,
            read off the panel before alignment, or ``None``.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if quality is None:
        return None
    issues = list(getattr(quality, "issues", None) or ())
    errors = getattr(quality, "errors", None)
    return {
        "usable": bool(getattr(quality, "is_usable", not errors)),
        "clean": bool(getattr(quality, "is_clean", not issues)),
        "errors": _strings(errors),
        "issues": _strings(issues),
        "findings": [
            {
                "severity": str(getattr(issue, "severity", "")),
                "code": str(getattr(issue, "code", "")),
                "asset": (
                    str(issue.asset) if getattr(issue, "asset", None) is not None else None
                ),
                "message": str(getattr(issue, "message", "")),
                "suggestion": str(getattr(issue, "suggestion", "") or ""),
            }
            for issue in issues
        ],
        "n_common_periods": int(getattr(quality, "n_common_periods", 0) or 0),
        "common_start": str(getattr(quality, "common_start", None) or "") or None,
        "common_end": str(getattr(quality, "common_end", None) or "") or None,
    }


def ingest_payload(result: Any) -> dict[str, Any] | None:
    """The request a panel was fetched with, its window resolved.

    An ingest with no ``end`` ends today and one with no ``start`` counts back
    from that, so the same flags fetch a different window each day. The
    resolved dates were hashed into the cache key and written nowhere a reader
    of the result could see them; here they are, with the fingerprint and
    whether the panel came from the cache — a cached panel is as old as its
    entry, not as old as the run.

    Args:
        result: An :class:`~optimization_engine.ingest.IngestResult`, or
            ``None`` when the panel did not come through an ingest.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if result is None:
        return None
    request = result.request
    entry = getattr(result, "cache_entry", None)
    return {
        **request.to_dict(),
        "fingerprint": request.fingerprint(),
        "from_cache": bool(getattr(result, "from_cache", False)),
        "cache_age_seconds": _num(entry.age_seconds) if entry is not None else None,
    }


def covariance_diagnostics_payload(diagnostics: Any) -> dict[str, Any] | None:
    """Whether the covariance estimate is worth the weights built on it.

    ``observations_per_asset`` below roughly 10 and a large
    ``condition_number`` together mean the optimiser is fitting noise, which
    no amount of solver precision fixes.

    A singular estimate has an infinite condition number, which strict JSON
    cannot carry: ``condition_number`` is then ``null`` and
    ``condition_number_infinite`` is ``true``. A ``null`` with the flag
    ``false`` means the number could not be computed.

    Args:
        diagnostics: A
            :class:`~optimization_engine.data.covariance.CovarianceDiagnostics`,
            or ``None``.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if diagnostics is None:
        return None
    return {
        "n_assets": int(diagnostics.n_assets),
        "n_observations": int(diagnostics.n_observations),
        "observations_per_asset": _num(diagnostics.observations_per_asset),
        "condition_number": _num(diagnostics.condition_number),
        "condition_number_infinite": bool(
            isinstance(diagnostics.condition_number, (int, float))
            and math.isinf(diagnostics.condition_number)
        ),
        "min_eigenvalue": _num(diagnostics.min_eigenvalue),
        "is_psd": bool(diagnostics.is_psd),
        "effective_observations": _num(diagnostics.effective_observations),
        "warnings": _strings(diagnostics.warnings),
    }


def feasibility_payload(report: Any) -> dict[str, Any] | None:
    """Whether the mandate could be satisfied at all, and over what range.

    ``feasible`` is lifted to the top of the object so a consumer can branch
    on one boolean instead of interpreting an issue list.

    Args:
        report: A
            :class:`~optimization_engine.optimizers.feasibility.FeasibilityReport`,
            or ``None``.

    Returns:
        A JSON-serializable dict, or ``None`` when nothing was supplied.
    """
    if report is None:
        return None
    # Structured, not stringified. This used to run through `_strings`, which
    # put the dataclass repr into the document — a consumer wanting the code
    # had to parse `FeasibilityIssue(code='...', ...)` out of prose.
    issues = [
        {
            "code": str(getattr(issue, "code", "")),
            "severity": str(getattr(issue, "severity", "")),
            "message": str(getattr(issue, "message", "")),
            "suggestion": str(getattr(issue, "suggestion", "") or ""),
        }
        for issue in (getattr(report, "issues", None) or ())
    ]
    # `is_feasible` is the report's own verdict; deriving it from an empty
    # issue list would silently disagree the day the two stop coinciding —
    # and they already do, now that a warning is an issue that does not
    # make the mandate impossible.
    feasible = getattr(report, "is_feasible", None)
    reachable = getattr(report, "reachable_return", None)
    return {
        "feasible": bool(feasible) if feasible is not None else not issues,
        "issues": issues,
        "stage_reached": str(getattr(report, "stage_reached", "structural")),
        "reachable_return": [_num(reachable[0]), _num(reachable[1])] if reachable else None,
        "min_return": _num(getattr(report, "min_return", None)),
        "max_return": _num(getattr(report, "max_return", None)),
        "min_variance_return": _num(getattr(report, "min_variance_return", None)),
    }


def optimization_payload(
    run: Any,
    *,
    output_path: str | None = None,
    alignment: Any = None,
    quality: Any = None,
    ingest: Any = None,
    data_source: Any = None,
) -> dict[str, Any]:
    """The full result of one solve: weights, and what they rest on.

    Args:
        run: An :class:`~optimization_engine.engine.EngineRun`.
        output_path: Where the workbook was written, when one was.
        alignment: The action log from
            :func:`~optimization_engine.data.quality.align_panel` — one
            sentence per change made to the panel before it was
            differenced. Empty means the panel needed no changes, which
            is a different claim from "nobody looked".
        quality: The raw panel's data-quality report. ``data_quality`` is
            ``null`` without one, which is "not reported", not "clean".
        ingest: The :class:`~optimization_engine.ingest.IngestResult` the
            panel came from; ``ingest`` is ``null`` for a file or the sample.
        data_source: Where the prices came from — see
            :func:`data_source_payload`.

    Returns:
        A JSON-serialisable dict. ``weights`` maps asset name to weight;
        every other top-level key is either a nested object of the same
        shape as its ``*_payload`` builder, or ``null``.
    """
    result = run.result
    extras = dict(getattr(result, "extras", {}) or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "optimize",
        "optimizer": extras.get("optimizer"),
        # Which solver actually answered, and how it terminated. A run that
        # fell through to a later solver in the chain is not wrong, but it
        # is worth knowing about.
        "solver": extras.get("solver"),
        "solver_status": extras.get("solver_status"),
        "weights": _series(result.weights),
        "metrics": {
            "expected_return": _num(result.expected_return),
            "expected_volatility": _num(result.expected_volatility),
            "sharpe_ratio": _num(result.sharpe_ratio),
        },
        "diagnostics": portfolio_diagnostics_payload(getattr(run, "diagnostics", None)),
        # The compliance half of the diagnostics above, structured: same
        # breaches, as numbers a consumer can threshold on rather than
        # sentences it has to parse. `null` means the solve ran no audit,
        # which is not the same claim as an empty violation list.
        "audit": audit_payload(getattr(result, "audit", None)),
        "covariance": covariance_diagnostics_payload(
            getattr(run, "covariance_diagnostics", None)
        ),
        "feasibility": feasibility_payload(getattr(run, "feasibility", None)),
        # What named bad days do to this book, when the run was asked. `null`
        # means no scenarios were applied — not that none of them hurt.
        "stress": stress_payload(getattr(run, "stress", None)),
        "warnings": _strings(getattr(run, "warnings", None)),
        # What the sample the numbers above were computed on had to lose
        # to become rectangular. A late-listing asset truncates every
        # other series, and that is invisible in the weights.
        "alignment": _strings(alignment),
        # And what was wrong with it before that. An asset missing a quarter
        # of its history is an error finding that the run proceeds past.
        "data_quality": data_quality_payload(quality),
        "ingest": ingest_payload(ingest),
        "data_source": data_source_payload(data_source),
        "output_path": output_path,
    }


def check_payload(
    quality: Any,
    feasibility: Any,
    covariance: Any = None,
    *,
    alignment: Any = None,
    ingest: Any = None,
    data_source: Any = None,
) -> dict[str, Any]:
    """A pre-flight verdict: can this mandate be solved, and on what data.

    Deliberately narrower than :func:`optimization_payload` — ``check`` runs
    *before* a solve, so there are no weights to report and claiming any
    would describe something that was never computed.

    ``ready`` is the single field a caller needs to branch on, and it
    mirrors the command's exit code: both data-quality errors and an
    infeasible constraint set make it false.

    Args:
        quality: A :class:`~optimization_engine.data.quality.DataQualityReport`.
        feasibility: A ``FeasibilityReport`` from ``analyze_feasibility``.
        covariance: Covariance diagnostics, when they were computed.
        alignment: The action log from
            :func:`~optimization_engine.data.quality.align_panel`.
        ingest: The ingest the panel came from, as for
            :func:`optimization_payload`.
        data_source: Where the prices came from — see
            :func:`data_source_payload`.
    """
    feas = feasibility_payload(feasibility)
    quality_errors = _strings(getattr(quality, "errors", None))
    ready = not quality_errors and bool(feas["feasible"] if feas else True)
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "check",
        "ready": ready,
        "data_quality": data_quality_payload(quality),
        "ingest": ingest_payload(ingest),
        # `data_quality` describes the panel as it arrived; `alignment`
        # describes what was done to it afterwards. Reading the first
        # without the second says what was wrong, not what was kept.
        "alignment": _strings(alignment),
        "data_source": data_source_payload(data_source),
        "feasibility": feas,
        "covariance": covariance_diagnostics_payload(covariance),
    }


def data_source_payload(source: Any) -> dict[str, Any] | None:
    """Which panel a result was computed on, in a fixed shape.

    Args:
        source: A mapping with ``kind`` — ``"file"``, ``"provider"``,
            ``"yahoo"`` or ``"sample"`` — and ``synthetic``, plus whichever of
            ``path``, ``provider`` and ``identifiers`` the kind has. ``None``
            when the caller did not record a source.

    Returns:
        All five keys, ``null`` where the kind has no value, so a consumer
        tests values rather than keys. ``synthetic`` is the one to branch on:
        it is true for the built-in sample panel and for the ``sample``
        provider, and a result computed on either describes no market.
    """
    if source is None:
        return None
    identifiers = source.get("identifiers")
    return {
        "kind": str(source["kind"]),
        "synthetic": bool(source.get("synthetic", False)),
        "path": None if source.get("path") is None else str(source["path"]),
        "provider": None if source.get("provider") is None else str(source["provider"]),
        "identifiers": None if identifiers is None else [str(i) for i in identifiers],
    }


def _frame(frame: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    """A DataFrame as a list of row objects, index carried as ``"index"``.

    Row objects rather than a column-major dict because a consumer that
    wants one row should not have to zip several parallel arrays and trust
    that they stayed aligned.
    """
    if frame is None or frame.empty:
        return None
    rows = []
    for idx, row in frame.iterrows():
        record: dict[str, Any] = {"index": str(idx)}
        for col, value in row.items():
            record[str(col)] = _num(value) if isinstance(value, (int, float, np.number)) else str(value)
        rows.append(record)
    return rows


def backtest_payload(
    result: Any,
    *,
    tearsheet: Any = None,
    output_path: str | None = None,
    alignment: Any = None,
    quality: Any = None,
    ingest: Any = None,
    data_source: Any = None,
) -> dict[str, Any]:
    """What a simulated run of the process actually produced.

    The two hashes are the reason a machine caller should read this rather
    than the workbook: two runs carrying the same ``spec_hash`` were asked
    the same question, and ``result_hash`` says whether they got the same
    answer. That is the cheapest way to tell a genuine change from a
    re-run.

    ``degradations`` is the honest part of the record — where the
    simulation had to fall back (a missing volatility estimate dropping
    market impact to the linear charge, say). A run with degradations is
    not wrong, but its costs are optimistic in a specific, named way.

    Args:
        result: A :class:`~optimization_engine.backtest.results.RunResult`.
        tearsheet: The built tearsheet, when one was.
        output_path: Where the workbook was written, when one was.
        alignment: The action log from
            :func:`~optimization_engine.data.quality.align_panel`. A
            simulated track record that starts three years late because
            one asset listed late is not the same track record.
        quality: The raw panel's data-quality report, as for
            :func:`optimization_payload`.
        ingest: The ingest the panel came from, as for
            :func:`optimization_payload`.
        data_source: Where the prices came from — see
            :func:`data_source_payload`.
    """
    meta = getattr(result, "meta", None)
    sheet_metadata = getattr(tearsheet, "metadata", None)
    stress_as_of = (
        sheet_metadata.get("stress_as_of")
        if isinstance(sheet_metadata, dict)
        else None
    )
    metrics = None
    if tearsheet is not None:
        performance = getattr(tearsheet, "performance", None)
        if performance is not None and not performance.empty:
            # One row, named for the book it describes; a machine caller
            # wants the metrics, not the row label.
            metrics = {
                str(k): _num(v) for k, v in performance.iloc[0].items()
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "backtest",
        "spec_hash": getattr(meta, "spec_hash", None),
        "result_hash": getattr(meta, "result_hash", None),
        "window": {
            "n_periods": int(getattr(meta, "n_periods", 0) or 0),
            "n_assets": int(getattr(meta, "n_assets", 0) or 0),
            "start": str(getattr(meta, "start", None) or "") or None,
            "end": str(getattr(meta, "end", None) or "") or None,
            "is_out_of_sample": bool(getattr(meta, "is_out_of_sample", False)),
        },
        "degradations": _strings(getattr(meta, "degradations", None)),
        "notes": _notes(getattr(meta, "notes", None)),
        "alignment": _strings(alignment),
        "data_quality": data_quality_payload(quality),
        "ingest": ingest_payload(ingest),
        "data_source": data_source_payload(data_source),
        "metrics": metrics,
        # The tearsheet applies the configured shocks to the book the run
        # ended on, so this is the stress of what would actually be held
        # tomorrow rather than of the average of the whole track record.
        "stress": stress_payload(
            getattr(tearsheet, "stress", None), as_of=stress_as_of
        ),
        "output_path": output_path,
    }


def describe_payload(req: Any) -> dict[str, Any]:
    """One optimizer's contract: what it needs, and what it will honour.

    This is the payload an agent reads *before* building a config, to find
    out whether the method it wants can express the mandate at all — a
    turnover budget handed to a method with ``supports_turnover`` false is
    silently ignored, not rejected.

    Args:
        req: A
            :class:`~optimization_engine.optimizers.requirements.MethodRequirements`.

    Returns:
        A JSON-serializable dict of what the method requires, supports and
        assumes.
    """
    inputs = {
        "expected_returns": bool(req.requires_mu),
        "covariance": bool(req.requires_cov),
        "return_history": bool(req.requires_returns),
        "benchmark": bool(req.requires_benchmark),
    }
    supports = {
        "target_return": bool(req.supports_target_return),
        "target_volatility": bool(req.supports_target_volatility),
        "risk_aversion": bool(req.supports_risk_aversion),
        "risk_free_rate": bool(req.supports_risk_free_rate),
        "group_bounds": bool(req.supports_group_bounds),
        "frontier": bool(req.supports_frontier),
        "turnover": bool(req.supports_turnover),
        "benchmark_limits": bool(req.supports_benchmark_limits),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "describe",
        "name": req.name,
        "label": getattr(req, "label", None),
        "summary": getattr(req, "summary", None),
        "when_to_use": getattr(req, "when_to_use", None),
        "risk_measure": getattr(req, "risk_measure", None),
        "bounds_mode": getattr(req, "bounds_mode", None),
        "assumptions": _strings(getattr(req, "assumptions", None)),
        "requires": inputs,
        "supports": supports,
    }
