"""Three-valued membership over time: ``True``, ``False``, and *not evaluable*.

Every membership rule in this library has a warm-up. "Eligible if the average
daily volume over the last 63 days exceeded ten million" says nothing at all
about day one, and the honest answer there is not ``False`` — it is *nothing*.
Writing ``False`` asserts that the name failed a test that was never run, and
because a hard boolean cannot carry that distinction, the assertion survives
every downstream operation: it excludes the name, shrinks the book, and looks
exactly like a genuine liquidity screen.

:class:`Signal` keeps the third state. It is a ``date × asset`` frame on
pandas' nullable ``boolean`` dtype, where ``pd.NA`` means *this rule could not
be evaluated on that date*, never *no*. The operators are Kleene's
three-valued logic, which pandas already implements on that dtype:

======  ======  ==========  ==========
``a``   ``b``   ``a & b``   ``a | b``
======  ======  ==========  ==========
T       T       T           T
T       F       F           T
T       NA      **NA**      **T**
F       T       F           T
F       F       F           F
F       NA      **F**       **NA**
NA      T       NA          T
NA      F       **F**       NA
NA      NA      NA          NA
======  ======  ==========  ==========

The two rows worth memorising are ``False & NA = False`` and
``True | NA = True``: an unknown cannot rescue a name a hard rule already
rejected, and it cannot exclude one a hard rule already admitted. Everything
else stays unknown, and stays unknown all the way to
:meth:`~optimization_engine.universe.eligibility.Eligibility.to_mask`, which is
the only place in the package where the third state collapses — under a policy
the caller had to name.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

#: The pandas dtype that carries the three states. Nothing else will do: the
#: NumPy ``bool`` dtype has no room for "not evaluable".
BOOLEAN_DTYPE = "boolean"


class UniverseError(ValueError):
    """A universe definition, or a question asked of one, that cannot be answered.

    Subclasses :class:`ValueError`, so callers already catching that keep
    working. Raised for a signal whose index is not dates, an unknown
    comparison operator or rolling aggregation, a mask policy that is not one
    of the three, a point-in-time label asked for without a date, and a
    ``"raise"`` collapse that met a cell nothing had evaluated.
    """


def _as_date_index(index: Any) -> pd.DatetimeIndex:
    """Coerce an index to dates, refusing the coercions that lie.

    Args:
        index: Anything a pandas index can be built from.

    Returns:
        A :class:`pandas.DatetimeIndex`.

    Raises:
        UniverseError: If the index is numeric — reading integers as
            nanoseconds since the epoch is never what anyone meant — or if
            pandas cannot parse it as dates at all.
    """
    idx = pd.Index(index)
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    if pd.api.types.is_numeric_dtype(idx):
        raise UniverseError(
            "A signal is indexed by dates; got a numeric index "
            f"({idx.dtype}). Reading those as nanoseconds since the epoch "
            "would silently invent a calendar."
        )
    try:
        return pd.DatetimeIndex(pd.to_datetime(idx))
    except (TypeError, ValueError) as exc:
        raise UniverseError(
            f"A signal is indexed by dates; this index is not one: {exc}"
        ) from exc


def to_boolean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize any frame into the three-valued form a signal is stored in.

    Args:
        frame: A ``date × asset`` frame of booleans, of ``0``/``1``, or of the
            nullable ``boolean`` dtype. Anything missing — ``NaN``, ``None``,
            ``pd.NA`` — becomes *not evaluable*.

    Returns:
        A copy on the ``boolean`` dtype, indexed by dates in ascending order
        with string column labels.

    Raises:
        UniverseError: If the index is not dates, or carries a duplicate
            date — an as-of lookup on a duplicated date has no single answer.
    """
    if not isinstance(frame, pd.DataFrame):
        raise UniverseError(
            f"A signal is built from a DataFrame; got {type(frame).__name__}."
        )
    out = frame.copy()
    out.index = _as_date_index(out.index)
    out.columns = pd.Index([str(c) for c in out.columns])
    if out.index.has_duplicates:
        duplicated = sorted(
            {str(d) for d in out.index[out.index.duplicated()]}
        )
        raise UniverseError(
            "A signal cannot carry the same date twice - an as-of lookup on "
            f"one would have no single answer. Repeated: {duplicated}."
        )
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    return out.astype(BOOLEAN_DTYPE)


class Signal:
    """A ``date × asset`` membership frame with three states.

    Attributes are read-only views on one immutable idea: what a rule said
    about each name on each date, including "nothing".

    Args:
        frame: The membership, as booleans, ``0``/``1``, or the nullable
            ``boolean`` dtype. Missing entries mean *not evaluable*.

    Raises:
        UniverseError: If ``frame`` is not a frame, its index is not dates, or
            a date appears twice.
    """

    __slots__ = ("_frame",)

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = to_boolean_frame(frame)

    # -- construction -------------------------------------------------------

    @classmethod
    def constant(
        cls,
        value: bool | None,
        index: Iterable[Any],
        assets: Iterable[Any],
    ) -> Signal:
        """A signal that says the same thing everywhere.

        Args:
            value: ``True``, ``False``, or ``None`` for *not evaluable*.
            index: The dates.
            assets: The column labels.

        Returns:
            The constant :class:`Signal`. Useful as the identity element when
            folding a list of rules, and as an explicit "we know nothing yet".
        """
        columns = [str(a) for a in assets]
        dates = _as_date_index(index)
        filled = pd.NA if value is None else bool(value)
        return cls(
            pd.DataFrame(filled, index=dates, columns=columns, dtype=BOOLEAN_DTYPE)
        )

    # -- shape --------------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """The membership frame itself, on the ``boolean`` dtype.

        Returned by reference for the sake of large panels; treat it as
        read-only. Every operation on this class produces a new signal rather
        than mutating one.
        """
        return self._frame

    @property
    def index(self) -> pd.DatetimeIndex:
        """The dates the signal is evaluated on, ascending."""
        return pd.DatetimeIndex(self._frame.index)

    @property
    def assets(self) -> pd.Index:
        """The asset labels, in the order the frame carries them."""
        return self._frame.columns

    @property
    def shape(self) -> tuple[int, int]:
        """``(dates, assets)``."""
        return self._frame.shape

    def __len__(self) -> int:
        """How many dates the signal covers."""
        return len(self._frame.index)

    def __repr__(self) -> str:
        """``Signal(dates=…, assets=…, unknown=…)`` — shape plus what is unknown."""
        return (
            f"Signal(dates={self.shape[0]}, assets={self.shape[1]}, "
            f"unknown={int(self._frame.isna().to_numpy().sum())})"
        )

    # -- logic --------------------------------------------------------------

    def align(self, other: Signal) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Both frames on the union of the two date and asset axes.

        Alignment is a union rather than an intersection on purpose: a date or
        a name only one side has heard of is *unknown* to the pair, and
        dropping it would quietly narrow the universe instead of admitting the
        gap.

        Args:
            other: The signal to align with.

        Returns:
            The two frames, same shape, missing cells set to *not evaluable*.
        """
        dates = self._frame.index.union(other._frame.index)
        assets = self._frame.columns.union(other._frame.columns)
        left = self._frame.reindex(index=dates, columns=assets).astype(BOOLEAN_DTYPE)
        right = other._frame.reindex(index=dates, columns=assets).astype(BOOLEAN_DTYPE)
        return left, right

    def _coerce(self, other: Any) -> Signal:
        if isinstance(other, Signal):
            return other
        if isinstance(other, pd.DataFrame):
            return Signal(other)
        raise UniverseError(
            "A signal combines with another signal or a DataFrame; got "
            f"{type(other).__name__}."
        )

    def __and__(self, other: Any) -> Signal:
        """Kleene conjunction. ``False & NA`` is ``False``; ``True & NA`` is unknown."""
        left, right = self.align(self._coerce(other))
        return Signal(left & right)

    def __or__(self, other: Any) -> Signal:
        """Kleene disjunction. ``True | NA`` is ``True``; ``False | NA`` is unknown."""
        left, right = self.align(self._coerce(other))
        return Signal(left | right)

    def __invert__(self) -> Signal:
        """Kleene negation. The negation of unknown is unknown."""
        return Signal(~self._frame)

    # -- reading ------------------------------------------------------------

    def is_missing(self) -> pd.DataFrame:
        """Where the signal could not be evaluated, as a hard boolean frame."""
        return self._frame.isna()

    def at(self, date: Any, asset: str) -> bool | None:
        """One cell, as ``True``, ``False`` or ``None``.

        Args:
            date: A date present in the index.
            asset: A column label.

        Returns:
            The state, with *not evaluable* rendered as ``None`` so the result
            can be compared with ``is``.

        Raises:
            UniverseError: If the date or the asset is not in the signal.
        """
        stamp = pd.Timestamp(date)
        if stamp not in self._frame.index:
            raise UniverseError(f"{stamp.isoformat()} is not a date this signal covers.")
        if str(asset) not in self._frame.columns:
            raise UniverseError(f"{asset!r} is not an asset this signal covers.")
        value = self._frame.at[stamp, str(asset)]
        return None if pd.isna(value) else bool(value)

    def as_of(self, date: Any) -> pd.Series:
        """The row in force on ``date``: the latest evaluation at or before it.

        This is the point-in-time read, and the only one the backtest runners
        use. A date that falls between two evaluations reads the earlier one;
        a date before the first evaluation reads *not evaluable* for every
        name, because nothing had been evaluated yet.

        Args:
            date: Any date-like value. It need not be in the index.

        Returns:
            One ``boolean``-dtype :class:`pandas.Series` indexed by asset.
        """
        stamp = pd.Timestamp(date)
        position = int(self._frame.index.searchsorted(stamp, side="right")) - 1
        if position < 0:
            return pd.Series(
                pd.NA, index=self._frame.columns, dtype=BOOLEAN_DTYPE, name=stamp
            )
        row = self._frame.iloc[position].copy()
        row.name = stamp
        return row

    def reindex(
        self,
        index: Iterable[Any] | None = None,
        assets: Sequence[Any] | None = None,
    ) -> Signal:
        """A signal on different axes, with anything new left *not evaluable*.

        Args:
            index: New dates. ``None`` keeps the current ones. Dates the
                signal does not carry come back unknown — this is a
                re-labelling, not an as-of lookup; use :meth:`as_of` for that.
            assets: New column labels. ``None`` keeps the current ones.

        Returns:
            The reindexed :class:`Signal`.
        """
        frame = self._frame
        if index is not None:
            frame = frame.reindex(index=_as_date_index(index))
        if assets is not None:
            frame = frame.reindex(columns=[str(a) for a in assets])
        return Signal(frame.astype(BOOLEAN_DTYPE))

    def equals(self, other: Signal) -> bool:
        """Whether two signals carry identical states on identical axes.

        Args:
            other: The signal to compare with.

        Returns:
            ``True`` when the frames are equal cell for cell, treating
            *not evaluable* as equal to itself.
        """
        return self._frame.equals(self._coerce(other)._frame)


__all__ = ["BOOLEAN_DTYPE", "Signal", "UniverseError", "to_boolean_frame"]
