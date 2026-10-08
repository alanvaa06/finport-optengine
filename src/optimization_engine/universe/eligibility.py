"""Membership rules, and the one place their unknowns become a hard answer.

An :class:`Eligibility` is a :class:`~optimization_engine.universe.signal.Signal`
that remembers where it came from. The rules that build one — a threshold on a
characteristic, a cross-sectional rank, a rolling window, hysteresis around an
entry/exit pair, a reconstitution calendar — all preserve the third state, so a
name whose data has not started yet is *unknown*, not *out*.

Two things are worth being explicit about, because they are where look-ahead
and silent shrinkage get in.

**Rolling windows are strictly prior.** :meth:`Eligibility.from_rolling`
evaluates date ``t`` on the ``window`` rows *before* ``t`` and never on ``t``
itself. Row ``window`` is therefore the first that can be evaluated at all;
rows ``0 … window-1`` come back *not evaluable*. (The design note calls these
"the first ``window-1`` rows", counting the warm-up a window that included the
evaluation date would burn. The rule implemented here excludes the evaluation
date, so the warm-up is one row longer — and the first evaluable row index
equals ``window``.)

**Threshold and rank rules read their own date.** A verdict from
:meth:`Eligibility.from_threshold` or :meth:`Eligibility.from_rank` on ``t``
is computed from the characteristic on ``t`` — the close of ``t``, for a
return or a price — so it is known only once ``t`` is over. A simulation with
no execution lag holds the book chosen on ``t`` over ``t`` itself, and reading
such a verdict there would drop a name on exactly the session it crashed.
:attr:`Eligibility.same_bar` records which verdicts read their own date, and
:func:`point_in_time_mask` reads those strictly *before* each date when the
lag is zero. Rolling rules are already known before ``t`` opens and are read
on ``t`` at any lag; with a lag of one or more every verdict is read on the
decision date, because the book it chooses is not held until the bar after.

**Collapsing is a decision, not a detail.** :meth:`Eligibility.to_mask` is the
only function in the package that turns three states into two, and it has no
default policy. ``"exclude"`` silently shrinks the book at every warm-up;
``"include"`` silently admits names nothing has screened; ``"raise"`` stops
every run that has a warm-up period at all. There is no safe answer, so the
caller names one.
"""

from __future__ import annotations

import operator as _operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from optimization_engine.universe.signal import (
    BOOLEAN_DTYPE,
    Signal,
    UniverseError,
    _as_date_index,
)

#: The three ways a *not evaluable* cell may be turned into a hard boolean.
#: ``"exclude"`` reads unknown as "not eligible", ``"include"`` as "eligible",
#: and ``"raise"`` refuses to guess.
MASK_POLICIES = ("exclude", "include", "raise")

#: Comparison operators :meth:`Eligibility.from_threshold` understands.
COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {
    ">": _operator.gt,
    ">=": _operator.ge,
    "<": _operator.lt,
    "<=": _operator.le,
    "==": _operator.eq,
    "!=": _operator.ne,
}

#: Rolling aggregations :meth:`Eligibility.from_rolling` understands. All but
#: ``"count"`` need a full window of observations; ``"count"`` is the number of
#: observations, so a partly empty window is exactly what it measures.
ROLLING_AGGS = ("mean", "median", "sum", "min", "max", "std", "var", "count")

_VERDICTS = {True: "eligible", False: "not eligible", None: "not evaluable"}


def _signal_of(value: Any) -> Signal:
    """The signal behind an eligibility, a signal, or a raw frame."""
    if isinstance(value, Eligibility):
        return value.signal
    if isinstance(value, Signal):
        return value
    if isinstance(value, pd.DataFrame):
        return Signal(value)
    raise UniverseError(
        "Expected an Eligibility, a Signal or a DataFrame; got "
        f"{type(value).__name__}."
    )


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """A characteristic frame as floats, or a message saying why it is not one."""
    try:
        return frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise UniverseError(
            f"A membership rule thresholds a numeric characteristic: {exc}"
        ) from exc


def _compare(frame: pd.DataFrame, op: str, value: Any) -> pd.DataFrame:
    """A numeric frame against a threshold, with missing inputs left unknown.

    Args:
        frame: The characteristic, one column per asset.
        op: One of :data:`COMPARISONS`.
        value: The threshold, in the characteristic's own units.

    Returns:
        A ``boolean``-dtype frame, *not evaluable* wherever the input was.

    Raises:
        UniverseError: On an operator this module does not implement, or a
            characteristic that is not numeric — a rule cannot threshold text.
    """
    if op not in COMPARISONS:
        raise UniverseError(
            f"{op!r} is not a comparison operator this rule understands. "
            f"Available: {sorted(COMPARISONS)}."
        )
    numeric = _numeric(frame)
    raw = COMPARISONS[op](numeric, value).astype(BOOLEAN_DTYPE)
    return raw.where(numeric.notna())


def collapse(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Turn a three-valued frame into a hard boolean one, under a named policy.

    The single collapse point of the package. Every caller that needs a plain
    ``bool`` frame comes through here, so there is exactly one line to read to
    find out what happened to the unknowns.

    Args:
        frame: A ``boolean``-dtype frame, as carried by a
            :class:`~optimization_engine.universe.signal.Signal`.
        policy: One of :data:`MASK_POLICIES`. ``"exclude"`` reads unknown as
            not eligible, ``"include"`` as eligible, ``"raise"`` refuses.

    Returns:
        A frame of NumPy ``bool``, same axes.

    Raises:
        UniverseError: If ``policy`` is not one of the three, or if it is
            ``"raise"`` and any cell was not evaluable — the message names the
            first few and how many there are in total.
    """
    if policy not in MASK_POLICIES:
        raise UniverseError(
            f"{policy!r} is not a mask policy. Choose one of {list(MASK_POLICIES)}: "
            "'exclude' drops names nothing has evaluated, 'include' admits "
            "them, 'raise' refuses to guess."
        )
    missing = frame.isna()
    if policy == "raise":
        rows, columns = np.nonzero(missing.to_numpy())
        total = int(len(rows))
        if total:
            where = [
                f"{pd.Timestamp(frame.index[row]).date()}/{frame.columns[column]}"
                for row, column in list(zip(rows, columns))[:5]
            ]
            raise UniverseError(
                f"{total} cell(s) are not evaluable and the policy is 'raise': "
                f"{', '.join(where)}"
                f"{' …' if total > len(where) else ''}. A rule with a warm-up "
                "period always has some; pass 'exclude' or 'include' to say "
                "what they mean."
            )
        return frame.astype(bool)
    return frame.fillna(policy == "include").astype(bool)


def point_in_time_mask(
    universe: Any,
    policy: str,
    dates: Iterable[Any],
    assets: Sequence[Any],
    *,
    execution_lag: int | None = None,
) -> pd.DataFrame:
    """The eligibility in force on each of ``dates``, as a hard boolean frame.

    This is the bridge between a universe definition and a simulation. For
    each requested date it reads the most recent evaluation *at or before* it —
    never the next one — so a run replayed on a calendar the universe was not
    defined on cannot see forward. Requested dates before the first evaluation,
    and requested assets the universe has never heard of, come back unknown and
    are then resolved by ``policy``.

    One exception, and it is the one that matters for a simulation: with
    ``execution_lag=0`` the book chosen on a date is held over that same bar,
    so a universe whose verdicts read their own date's data
    (:attr:`Eligibility.same_bar`) is read strictly *before* each date — the
    latest verdict that existed when the book was chosen.

    Args:
        universe: An :class:`Eligibility`, a
            :class:`~optimization_engine.universe.signal.Signal`, or a frame
            one can be built from.
        policy: One of :data:`MASK_POLICIES`.
        dates: The dates to evaluate on — normally a simulation's index.
        assets: The columns to report, in order — normally the return frame's.
        execution_lag: The lag of the simulation the mask is for, in bars.
            ``0`` reads a same-bar universe strictly before each date; a
            positive lag, or ``None`` (the default, for inspection rather
            than simulation), reads every universe at or before it.

    Returns:
        A ``bool`` frame indexed by ``dates`` with one column per asset.

    Raises:
        UniverseError: On an unknown policy, or under ``"raise"`` when
            anything was not evaluable.
    """
    signal = _signal_of(universe)
    columns = [str(a) for a in assets]
    wanted = _as_date_index(dates)
    values = signal.frame.reindex(columns=columns).astype(BOOLEAN_DTYPE)
    strictly_before = (
        execution_lag is not None
        and int(execution_lag) == 0
        and isinstance(universe, Eligibility)
        and universe.same_bar
    )
    side = "left" if strictly_before else "right"
    positions = (
        np.asarray(values.index.searchsorted(wanted.to_numpy(), side=side)) - 1
    )
    cells = np.empty((len(wanted), len(columns)), dtype=object)
    cells[:] = pd.NA
    known = positions >= 0
    if known.any() and len(values.index):
        cells[known] = values.to_numpy(dtype=object)[positions[known]]
    frame = pd.DataFrame(cells, index=wanted, columns=columns).astype(BOOLEAN_DTYPE)
    return collapse(frame, policy)


@dataclass(frozen=True, eq=False)
class Rule:
    """One node of the provenance tree behind an :class:`Eligibility`.

    A rule is what :meth:`Eligibility.explain` walks to answer "why is this
    name out?" with the name of the screen that excluded it rather than a
    shrug.

    Attributes:
        description: What this node says, in words — the label a caller
            passed, or a generated one naming the characteristic, the
            operator and the threshold.
        signal: This node's own verdict, over its own axes.
        operator: ``"leaf"``, ``"and"``, ``"or"``, ``"not"``, ``"hysteresis"``
            or ``"hold_through"``.
        operands: The nodes this one combines, empty for a leaf.
        same_bar: Whether this node's verdict on ``t`` reads data stamped
            ``t`` — so it is known only at ``t``'s close — rather than only
            earlier rows. True for threshold and rank leaves, and for any
            node with such an operand.
    """

    description: str
    signal: Signal
    operator: str = "leaf"
    operands: tuple[Rule, ...] = field(default_factory=tuple)
    same_bar: bool = False


def _state_at(rule: Rule, stamp: pd.Timestamp, asset: str) -> bool | None:
    """One node's verdict at a date and asset, or ``None`` when it has none."""
    frame = rule.signal.frame
    if stamp not in frame.index or asset not in frame.columns:
        return None
    value = frame.at[stamp, asset]
    return None if pd.isna(value) else bool(value)


def _reason(rule: Rule, stamp: pd.Timestamp, asset: str) -> str:
    """The clause of ``rule`` that decided ``asset`` on ``stamp``."""
    if rule.operator in ("and", "or") and rule.operands:
        decisive = False if rule.operator == "and" else True
        state = _state_at(rule, stamp, asset)
        if state is decisive:
            for operand in rule.operands:
                if _state_at(operand, stamp, asset) is decisive:
                    return _reason(operand, stamp, asset)
        if state is None:
            for operand in rule.operands:
                if _state_at(operand, stamp, asset) is None:
                    return f"{_reason(operand, stamp, asset)} could not be evaluated"
        joiner = " and " if rule.operator == "and" else " or "
        return joiner.join(_reason(o, stamp, asset) for o in rule.operands)
    if rule.operator == "not" and rule.operands:
        return f"not ({_reason(rule.operands[0], stamp, asset)})"
    if rule.operator == "hysteresis" and len(rule.operands) == 2:
        entry, exits = rule.operands
        if _state_at(entry, stamp, asset) is True:
            return f"the entry rule fired: {_reason(entry, stamp, asset)}"
        if _state_at(exits, stamp, asset) is True:
            return f"the exit rule fired: {_reason(exits, stamp, asset)}"
        return (
            f"membership carried forward under hysteresis "
            f"(entry: {_reason(entry, stamp, asset)}; "
            f"exit: {_reason(exits, stamp, asset)})"
        )
    if rule.operator == "hold_through" and rule.operands:
        return (
            f"{_reason(rule.operands[0], stamp, asset)}, held since the most "
            "recent reconstitution"
        )
    return rule.description


@dataclass(frozen=True, eq=False)
class Eligibility:
    """Point-in-time membership of an investable universe.

    Wraps a :class:`~optimization_engine.universe.signal.Signal` together with
    the rule tree that produced it, so the verdict on any one name on any one
    date can be explained rather than merely reported.

    Attributes:
        rule: The provenance tree. Its ``signal`` is this eligibility's
            verdict.
    """

    rule: Rule

    # -- construction -------------------------------------------------------

    @classmethod
    def from_signal(cls, signal: Any, description: str) -> Eligibility:
        """Wrap an already-computed signal, naming the rule it represents.

        Args:
            signal: A :class:`~optimization_engine.universe.signal.Signal`, an
                :class:`Eligibility`, or a frame one can be built from — for
                example an index-membership panel loaded from a vendor file.
            description: What the signal means, for :meth:`explain`.

        Returns:
            The :class:`Eligibility`. A frame or a bare signal is taken to be
            known before each of its dates opens — what a membership file
            records — so it is read on its own date at any lag; a panel
            screened on its own date's data should be shifted one row before
            it is wrapped. An :class:`Eligibility` keeps its
            :attr:`same_bar`.
        """
        same_bar = isinstance(signal, Eligibility) and signal.same_bar
        return cls(Rule(str(description), _signal_of(signal), same_bar=same_bar))

    @classmethod
    def from_threshold(
        cls,
        series_frame: pd.DataFrame,
        op: str,
        value: float,
        *,
        name: str | None = None,
    ) -> Eligibility:
        """Eligible where a characteristic clears a threshold on that date.

        Args:
            series_frame: The characteristic, ``date × asset``, in whatever
                units the threshold is quoted in — currency for a market
                capitalisation, shares or currency for a volume, a fraction
                for a weight.
            op: One of :data:`COMPARISONS`.
            value: The threshold, in the same units as ``series_frame``.
            name: Label for :meth:`explain`. Defaults to a generated one.

        Returns:
            An :class:`Eligibility`, *not evaluable* wherever the
            characteristic was missing. Its verdict on ``t`` reads ``t``, so
            it is :attr:`same_bar`: a run with no execution lag acts on it
            from the next bar.

        Raises:
            UniverseError: On an unknown operator, or a frame whose index is
                not dates.
        """
        signal = Signal(_compare(series_frame, op, value))
        label = name or f"characteristic {op} {value:g}"
        return cls(Rule(str(label), signal, same_bar=True))

    @classmethod
    def from_rank(
        cls,
        series_frame: pd.DataFrame,
        top_n: int,
        *,
        name: str | None = None,
    ) -> Eligibility:
        """Eligible where a characteristic ranks in the top ``n`` on that date.

        Ranking is cross-sectional and descending: the largest value ranks
        first. Ties are broken by column order, which is arbitrary but
        deterministic — two identical panels always produce the same universe.

        Args:
            series_frame: The characteristic, ``date × asset``.
            top_n: How many names to admit per date. Must be at least 1.
            name: Label for :meth:`explain`. Defaults to a generated one.

        Returns:
            An :class:`Eligibility`, *not evaluable* for any name whose
            characteristic was missing on that date. A missing value is not
            ranked last: it is not ranked at all. Like a threshold, it is
            :attr:`same_bar`.

        Raises:
            UniverseError: If ``top_n`` is below 1, or the index is not dates.
        """
        if int(top_n) < 1:
            raise UniverseError(f"top_n must be at least 1; got {top_n}.")
        numeric = _numeric(series_frame)
        ranks = numeric.rank(axis=1, ascending=False, method="first")
        raw = (ranks <= int(top_n)).astype(BOOLEAN_DTYPE).where(numeric.notna())
        label = name or f"top {int(top_n)} by characteristic"
        return cls(Rule(str(label), Signal(raw), same_bar=True))

    @classmethod
    def from_rolling(
        cls,
        frame: pd.DataFrame,
        window: int,
        agg: str,
        op: str,
        value: float,
        *,
        name: str | None = None,
    ) -> Eligibility:
        """Eligible where a rolling statistic of the *prior* window clears a threshold.

        The window ends on the period **before** the evaluation date and never
        includes it, which is what makes this rule usable as a decision screen:
        a name is admitted on ``t`` on evidence that existed before ``t``.

        Args:
            frame: The characteristic, ``date × asset``, in its own units.
            window: Window length in periods, at least 1. The window for date
                ``t`` is the ``window`` rows before ``t``.
            agg: One of :data:`ROLLING_AGGS`. Everything except ``"count"``
                needs a full window of observations and is *not evaluable*
                when the window has a gap; ``"count"`` counts the
                observations, so a gappy window is what it is measuring.
            op: One of :data:`COMPARISONS`.
            value: The threshold, in the aggregated characteristic's units.
            name: Label for :meth:`explain`. Defaults to a generated one.

        Returns:
            An :class:`Eligibility` whose first ``window`` rows are *not
            evaluable* — there is no complete prior window behind them, and
            answering ``False`` there would assert a test that never ran.

        Raises:
            UniverseError: On a window below 1, an unknown aggregation, or an
                unknown operator.
        """
        span = int(window)
        if span < 1:
            raise UniverseError(f"A rolling window must be at least 1 period; got {window}.")
        if agg not in ROLLING_AGGS:
            raise UniverseError(
                f"{agg!r} is not a rolling aggregation this rule understands. "
                f"Available: {list(ROLLING_AGGS)}."
            )
        numeric = _numeric(frame)
        rolled = getattr(numeric.shift(1).rolling(span, min_periods=span), agg)()
        # The first ``span`` rows have no complete prior window whatever the
        # aggregation says: ``count`` in particular happily reports a partial
        # one. Blank them, so every aggregation warms up identically.
        if len(rolled.index):
            rolled.iloc[:span] = np.nan
        label = name or f"rolling {agg} over {span} prior periods {op} {value:g}"
        return cls(Rule(str(label), Signal(_compare(rolled, op, value))))

    @classmethod
    def with_hysteresis(
        cls,
        entry: Any,
        exit: Any,
        initial: bool | None,
    ) -> Eligibility:
        """Membership that is easier to keep than to gain.

        ``member[t] = entry[t] | (member[t-1] & ~exit[t])`` evaluated under
        Kleene logic, which is what stops an index from churning on a name
        oscillating around one threshold: entry needs the entry rule to fire,
        while staying in only needs the exit rule *not* to.

        Args:
            entry: The rule that admits a name.
            exit: The rule that ejects one. A name is dropped only when this
                fires; a name neither entering nor exiting keeps whatever it
                had.
            initial: Membership before the first date. ``None`` means unknown,
                and unknown propagates: until either rule fires the answer
                stays *not evaluable* rather than defaulting to "out".

        Returns:
            The :class:`Eligibility`. Its axes are the union of the two rules'.

        Raises:
            UniverseError: If either argument is not a signal-like object.
        """
        entry_signal = _signal_of(entry)
        exit_signal = _signal_of(exit)
        left, right = entry_signal.align(exit_signal)
        not_exit = ~right
        previous = pd.Series(
            pd.NA if initial is None else bool(initial),
            index=left.columns,
            dtype=BOOLEAN_DTYPE,
        )
        cells = np.empty((len(left.index), len(left.columns)), dtype=object)
        cells[:] = pd.NA
        for position in range(len(left.index)):
            current = left.iloc[position] | (previous & not_exit.iloc[position])
            cells[position] = current.to_numpy(dtype=object)
            previous = current
        frame = pd.DataFrame(cells, index=left.index, columns=left.columns).astype(
            BOOLEAN_DTYPE
        )
        entry_rule = _rule_of(entry, "entry rule")
        exit_rule = _rule_of(exit, "exit rule")
        return cls(
            Rule(
                f"hysteresis(entry={entry_rule.description}, "
                f"exit={exit_rule.description}, initial={initial})",
                Signal(frame),
                operator="hysteresis",
                operands=(entry_rule, exit_rule),
                same_bar=entry_rule.same_bar or exit_rule.same_bar,
            )
        )

    # -- shape --------------------------------------------------------------

    @property
    def signal(self) -> Signal:
        """The three-valued membership this eligibility resolves to."""
        return self.rule.signal

    @property
    def frame(self) -> pd.DataFrame:
        """The membership frame, on the ``boolean`` dtype. Treat as read-only."""
        return self.rule.signal.frame

    @property
    def index(self) -> pd.DatetimeIndex:
        """The dates membership is evaluated on."""
        return self.rule.signal.index

    @property
    def assets(self) -> pd.Index:
        """The asset labels this eligibility covers."""
        return self.rule.signal.assets

    @property
    def description(self) -> str:
        """The rule tree's top-level label."""
        return self.rule.description

    @property
    def same_bar(self) -> bool:
        """Whether a verdict on ``t`` reads ``t``'s own data anywhere in the tree.

        True when any threshold or rank rule is in it. Such a verdict is known
        only at ``t``'s close, so a simulation with no execution lag reads it
        from the bar before; see :func:`point_in_time_mask`.
        """
        return self.rule.same_bar

    def __repr__(self) -> str:
        """``Eligibility(<description>, dates=…, assets=…)``."""
        shape = self.rule.signal.shape
        return (
            f"Eligibility({self.rule.description!r}, dates={shape[0]}, "
            f"assets={shape[1]})"
        )

    # -- logic --------------------------------------------------------------

    def __and__(self, other: Any) -> Eligibility:
        """Both rules must admit the name. Unknown survives unless one says no."""
        right = _rule_of(other, "rule")
        return Eligibility(
            Rule(
                f"({self.rule.description}) and ({right.description})",
                self.signal & right.signal,
                operator="and",
                operands=(self.rule, right),
                same_bar=self.rule.same_bar or right.same_bar,
            )
        )

    def __or__(self, other: Any) -> Eligibility:
        """Either rule admits the name. Unknown survives unless one says yes."""
        right = _rule_of(other, "rule")
        return Eligibility(
            Rule(
                f"({self.rule.description}) or ({right.description})",
                self.signal | right.signal,
                operator="or",
                operands=(self.rule, right),
                same_bar=self.rule.same_bar or right.same_bar,
            )
        )

    def __invert__(self) -> Eligibility:
        """The complement. What was unknown stays unknown."""
        return Eligibility(
            Rule(
                f"not ({self.rule.description})",
                ~self.signal,
                operator="not",
                operands=(self.rule,),
                same_bar=self.rule.same_bar,
            )
        )

    # -- calendars ----------------------------------------------------------

    def hold_through(self, dates: Iterable[Any]) -> Eligibility:
        """Evaluate membership only on reconstitution dates and hold it between.

        A real index does not re-screen its constituents daily. This freezes
        the verdict of the most recent reconstitution and carries it forward,
        so a name that dips below a threshold mid-quarter stays in until the
        quarter is reviewed.

        Args:
            dates: The reconstitution dates. Each one reads the membership row
                in force at or before it, so a review stamped on a non-trading
                day reads the previous session.

        Returns:
            An :class:`Eligibility` on this one's dates. Rows before the first
            reconstitution are *not evaluable*: membership had not been
            reviewed yet, which is not the same as being out.

        Raises:
            UniverseError: If ``dates`` cannot be read as dates.
        """
        frame = self.frame
        reviews = _as_date_index(dates).sort_values().unique()
        source = np.asarray(
            frame.index.searchsorted(np.asarray(reviews), side="right")
        ) - 1
        applies = (
            np.asarray(
                np.searchsorted(
                    np.asarray(reviews), frame.index.to_numpy(), side="right"
                )
            )
            - 1
        )
        values = frame.to_numpy(dtype=object)
        cells = np.empty(values.shape, dtype=object)
        cells[:] = pd.NA
        for position in range(len(frame.index)):
            review = int(applies[position])
            if review < 0:
                continue
            row = int(source[review])
            if row < 0:
                continue
            cells[position] = values[row]
        held = pd.DataFrame(cells, index=frame.index, columns=frame.columns).astype(
            BOOLEAN_DTYPE
        )
        return Eligibility(
            Rule(
                f"({self.rule.description}) held through "
                f"{len(reviews)} reconstitution(s)",
                Signal(held),
                operator="hold_through",
                operands=(self.rule,),
                same_bar=self.rule.same_bar,
            )
        )

    # -- reading ------------------------------------------------------------

    def as_of(self, date: Any) -> pd.Series:
        """Membership in force on ``date`` — the latest evaluation at or before it.

        Args:
            date: Any date-like value; it need not be an evaluation date.

        Returns:
            A ``boolean``-dtype :class:`pandas.Series` indexed by asset, all
            *not evaluable* when ``date`` precedes the first evaluation.
        """
        return self.signal.as_of(date)

    def to_mask(self, policy: str) -> pd.DataFrame:
        """Collapse to a hard boolean frame under an explicitly named policy.

        There is deliberately **no default**. ``"exclude"`` quietly shrinks the
        book across every warm-up period, ``"include"`` quietly admits names no
        rule has screened, and ``"raise"`` stops any run whose rules have a
        warm-up at all. Which of those is wrong depends on the mandate, so the
        caller says which one they meant.

        Args:
            policy: One of :data:`MASK_POLICIES`.

        Returns:
            A ``bool`` frame on the same axes.

        Raises:
            UniverseError: On an unknown policy, or under ``"raise"`` when any
                cell was not evaluable.
        """
        return collapse(self.frame, policy)

    def breadth(self) -> pd.Series:
        """How many names are eligible on each date.

        Returns:
            An integer :class:`pandas.Series` indexed by date, counting only
            cells that evaluated to ``True``. Unknown cells are *not* counted
            — see :meth:`unknown_count` for those, and :meth:`to_mask` for a
            count under a stated policy.
        """
        counts = self.frame.fillna(False).astype(bool).sum(axis=1)
        return counts.astype(int).rename("breadth")

    def unknown_count(self) -> pd.Series:
        """How many names could not be evaluated on each date.

        Returns:
            An integer :class:`pandas.Series` indexed by date. A large count
            outside the warm-up usually means the characteristic panel has
            gaps the rule cannot see through.
        """
        return self.frame.isna().sum(axis=1).astype(int).rename("unknown")

    def turnover(self) -> pd.DataFrame:
        """Entries, exits and their sum per date, in names.

        Only transitions between two *evaluated* states are counted: a name
        going from unknown to eligible has not entered the universe, it has
        merely become knowable, and counting it as an entry would inflate the
        turnover of every rule with a warm-up.

        Returns:
            A frame indexed by date with integer columns ``entries``,
            ``exits`` and ``turnover`` (their sum). The first row is always
            zero — there is no previous date to have changed from. For the
            other reading, in which the warm-up boundary *is* an entry because
            that is the day the desk first had a universe to trade, diff a
            collapsed mask instead: ``self.to_mask(policy).astype(int).diff()``.
        """
        frame = self.frame
        is_true = frame.fillna(False).astype(bool)
        is_false = (~frame).fillna(False).astype(bool)
        previous_true = is_true.shift(1).fillna(False).astype(bool)
        previous_false = is_false.shift(1).fillna(False).astype(bool)
        entries = (is_true & previous_false).sum(axis=1).astype(int)
        exits = (is_false & previous_true).sum(axis=1).astype(int)
        return pd.DataFrame(
            {"entries": entries, "exits": exits, "turnover": entries + exits}
        )

    def explain(self, date: Any, asset: str) -> str:
        """Why one name was in or out on one date, naming the rule that decided.

        Args:
            date: The date to explain. Reads the evaluation in force at or
                before it, and says so when that is an earlier date.
            asset: The name to explain.

        Returns:
            A sentence: the verdict (``eligible`` / ``not eligible`` /
            ``not evaluable``) and the clause of the rule tree responsible —
            for a conjunction, the screen that rejected the name; for a
            disjunction, the one that admitted it.

        Raises:
            UniverseError: If the asset is not in this universe, or the date
                precedes the first evaluation.
        """
        label = str(asset)
        if label not in self.assets:
            raise UniverseError(
                f"{label!r} is not an asset this universe covers. "
                f"It has {len(self.assets)} name(s)."
            )
        stamp = pd.Timestamp(date)
        index = self.index
        position = int(index.searchsorted(stamp, side="right")) - 1
        if position < 0:
            raise UniverseError(
                f"{stamp.date()} is before the first evaluation "
                f"({pd.Timestamp(index[0]).date() if len(index) else 'none'}), "
                "so there is nothing to explain."
            )
        evaluated = pd.Timestamp(index[position])
        state = _state_at(self.rule, evaluated, label)
        text = (
            f"{label} on {stamp.date()}: {_VERDICTS[state]} — "
            f"{_reason(self.rule, evaluated, label)}."
        )
        if evaluated != stamp:
            text += (
                f" (Evaluated on {evaluated.date()}, the most recent "
                f"evaluation on or before {stamp.date()}.)"
            )
        return text


def _rule_of(value: Any, fallback: str) -> Rule:
    """The rule tree behind an operand, wrapping a bare signal in a leaf."""
    if isinstance(value, Eligibility):
        return value.rule
    return Rule(fallback, _signal_of(value))


__all__ = [
    "COMPARISONS",
    "MASK_POLICIES",
    "ROLLING_AGGS",
    "Eligibility",
    "Rule",
    "collapse",
    "point_in_time_mask",
]
