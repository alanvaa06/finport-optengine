"""The held-out segment, and the audit log that keeps it held out.

Walk-forward analysis controls look-ahead within a run. It does not control
the researcher. Over weeks of work the same history gets seen dozens of
times, the grid gets re-run after each disappointing result, and the final
number is the maximum of a search whose size nobody wrote down. That is
overfitting at the level of the project rather than the model, and no
statistic computed on the same data can detect it.

The defence is a segment of history that is never touched until the work is
finished, and a record of every time it was. This module provides both:

* :func:`gate_returns` physically truncates the data before any run can see
  it, and :func:`assert_within_holdout` fails loudly if untruncated data
  reaches a gated path anyway. Enforcement at the loader beats enforcement by
  convention, because convention is what erodes at 6pm on a Friday.
* :func:`final_holdout_run` is the single sanctioned way to evaluate on the
  held-out segment. Every invocation is appended to a JSONL log, and repeat
  visits earn flags: ``REPEATED`` when the same specification has been run
  before, ``SHIFTED_HOLDOUT`` when the same strategy has been evaluated
  against a *different* boundary — the tell that the boundary was moved until
  the answer improved.

The flags do not block anything. They make the second look visible, which is
all a diagnostic can honestly do.

The log is meant to be **committed**. Selection bias accumulates across people
as much as across sessions — two analysts looking once each is two trials on
the same segment — and a log that lives only on one laptop is a control the
controlled party can reset by cloning the repository fresh. The default path
is under ``runs/``, which this repository ignores except for this one file;
``.gitattributes`` gives it a union merge so concurrent appends from two
branches do not conflict.

That it is committed is also why ``strategy`` should stay a *description* and
never a payload: whatever a caller puts in that mapping is hashed, written to
the log, and published with it.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

#: Where the audit trail lives unless a caller says otherwise.
DEFAULT_AUDIT_PATH = Path("runs") / "holdout_audit.jsonl"

#: The same specification, evaluated on the held-out segment more than once.
REPEATED = "REPEATED"

#: The same strategy, evaluated against a boundary that moved.
SHIFTED_HOLDOUT = "SHIFTED_HOLDOUT"


class HoldoutViolationError(RuntimeError):
    """Data past the holdout boundary reached a gated path."""


def gate_returns(
    returns: pd.DataFrame, holdout_after: pd.Timestamp | str
) -> pd.DataFrame:
    """Everything at or before the boundary; the future is simply not there.

    Truncation rather than a flag: code cannot accidentally read rows that
    were never handed to it.

    Args:
        returns: The full history.
        holdout_after: The boundary date. Rows at or before it are visible;
            everything after is the holdout.

    Returns:
        A view of ``returns`` ending at the boundary inclusive.
    """
    boundary = pd.Timestamp(holdout_after)
    return returns.loc[returns.index <= boundary]


def holdout_segment(
    returns: pd.DataFrame, holdout_after: pd.Timestamp | str
) -> pd.DataFrame:
    """The segment after the boundary — the part nothing may have seen.

    Args:
        returns: The full history.
        holdout_after: The boundary date, exclusive.

    Returns:
        A view of ``returns`` beginning after the boundary.
    """
    boundary = pd.Timestamp(holdout_after)
    return returns.loc[returns.index > boundary]


def assert_within_holdout(
    returns: pd.DataFrame, holdout_after: pd.Timestamp | str, name: str = "returns"
) -> None:
    """Raise if any row sits past the boundary.

    Args:
        returns: The frame to check.
        holdout_after: The boundary date. Rows at or before it are fine.
        name: What to call the frame in the error message.

    Raises:
        HoldoutViolationError: If the frame carries post-boundary rows.
    """
    boundary = pd.Timestamp(holdout_after)
    if returns.empty:
        return
    last = pd.Timestamp(returns.index.max())
    if last > boundary:
        raise HoldoutViolationError(
            f"{name} carries {last.date()}, past the holdout boundary "
            f"{boundary.date()} - a gated run must never see this row."
        )


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def read_audit_log(audit_path: Path | str = DEFAULT_AUDIT_PATH) -> list[dict[str, Any]]:
    """Every recorded visit to the held-out segment, oldest first.

    Args:
        audit_path: The log to read. Defaults to the repository's own, which
            is committed on purpose: a log any one researcher can reset by
            cloning fresh is a decorative control.

    Returns:
        One dict per recorded look, or an empty list when the log does not
        exist yet.
    """
    path = Path(audit_path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class HoldoutOutcome:
    """One audited evaluation on the held-out segment.

    Attributes:
        returns: The evaluated return stream.
        flags: ``REPEATED`` / ``SHIFTED_HOLDOUT``, or empty on a first look.
        audit_row: Exactly what was appended to the log.
        summary: Performance of the held-out stream.
    """

    returns: pd.Series
    flags: tuple[str, ...]
    audit_row: dict[str, Any]
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def is_first_look(self) -> bool:
        """Whether this is the first evaluation of this strategy on this holdout.

        Returns:
            ``True`` when no flag was raised. Anything else means the audit log
            has seen this strategy — or a shifted boundary for it — before.
        """
        return not self.flags

    def describe(self) -> str:
        """What the audit log makes of this look.

        Returns:
            On a first look, a reminder that a second look is a second trial. On a
            repeat, the flags spelled out: whether this exact specification has
            been evaluated on the holdout before, whether the boundary moved, or
            both.
        """
        if self.is_first_look:
            return (
                "First look at the held-out segment. Whatever it says, it says "
                "once - a second look is a second trial."
            )
        notes = []
        if REPEATED in self.flags:
            notes.append(
                "this exact specification has been evaluated on the holdout before"
            )
        if SHIFTED_HOLDOUT in self.flags:
            notes.append(
                "this strategy has been evaluated against a different holdout "
                "boundary - the boundary moved"
            )
        return "Holdout flags: " + "; ".join(notes) + "."


def final_holdout_run(
    returns: pd.DataFrame,
    holdout_after: pd.Timestamp | str,
    evaluate: Callable[[pd.DataFrame], pd.Series],
    *,
    strategy: dict[str, Any] | None = None,
    label: str = "final",
    audit_path: Path | str = DEFAULT_AUDIT_PATH,
    periods_per_year: int = 252,
    clock: Callable[[], _datetime.datetime] | None = None,
) -> HoldoutOutcome:
    """Evaluate one strategy on the held-out segment, and write it down.

    There is deliberately no grid form. A holdout you can sweep is not a
    holdout.

    Args:
        returns: The full history, including the held-out segment.
        holdout_after: The boundary. Everything after it is the holdout.
        evaluate: Takes the held-out returns and produces a return stream.
        strategy: A serializable description of what is being evaluated. It
            identifies the *strategy* across boundaries, which is what makes
            ``SHIFTED_HOLDOUT`` detectable. Keep it to identifiers — a config
            path, a spec hash, an optimizer name. The log is committed, so
            anything put here is published with it.
        label: A human-readable name for the log.
        audit_path: Where to append. Opened in append mode only.
        periods_per_year: Annualization basis for the summary.
        clock: Injectable clock, for tests.

    Raises:
        ValueError: If the boundary leaves no held-out segment.
    """
    segment = holdout_segment(returns, holdout_after)
    boundary = pd.Timestamp(holdout_after)
    if segment.empty:
        raise ValueError(
            f"No held-out segment: the boundary {boundary.date()} leaves nothing "
            f"before the last observation {pd.Timestamp(returns.index.max()).date()}."
        )

    description = dict(strategy or {})
    family_hash = _fingerprint(description)
    spec_hash = _fingerprint(
        {"strategy": description, "holdout_after": boundary.isoformat()}
    )

    existing = read_audit_log(audit_path)
    flags: list[str] = []
    if any(row.get("spec_hash") == spec_hash for row in existing):
        flags.append(REPEATED)
    if any(
        row.get("family_hash") == family_hash
        and row.get("holdout_after") != boundary.isoformat()
        for row in existing
    ):
        flags.append(SHIFTED_HOLDOUT)

    stream = evaluate(segment)
    from optimization_engine.analytics.performance import summary_stats

    summary = summary_stats(
        stream.to_frame("holdout"),
        periods_per_year=periods_per_year,
        riskfree_rate=0.0,
        extended=True,
    )

    now = clock() if clock is not None else _datetime.datetime.now(_datetime.timezone.utc)
    audit_row: dict[str, Any] = {
        "timestamp": now.isoformat(),
        "label": label,
        "spec_hash": spec_hash,
        "family_hash": family_hash,
        "holdout_after": boundary.isoformat(),
        "holdout_start": pd.Timestamp(segment.index[0]).isoformat(),
        "holdout_end": pd.Timestamp(segment.index[-1]).isoformat(),
        "n_periods": int(len(stream)),
        "flags": list(flags),
        "strategy": description,
    }
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append-only by construction: there is no rewrite path in this module.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_row, sort_keys=True, default=str) + "\n")

    return HoldoutOutcome(
        returns=stream, flags=tuple(flags), audit_row=audit_row, summary=summary
    )


__all__ = [
    "DEFAULT_AUDIT_PATH",
    "REPEATED",
    "SHIFTED_HOLDOUT",
    "HoldoutOutcome",
    "HoldoutViolationError",
    "assert_within_holdout",
    "final_holdout_run",
    "gate_returns",
    "holdout_segment",
    "read_audit_log",
]
