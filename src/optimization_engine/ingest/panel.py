"""The panel every provider produces and every consumer reads.

:class:`PricePanel` is the boundary of the ingest layer. Above it sit adapters
that know about HTTP, vendor JSON and ticker grammars; below it sits an engine
that knows about none of that and sees one shape: a date index, a column per
identifier, a frame per field, and a metadata record saying where each series
came from and what it is.

The validation in :meth:`PricePanel.validate` is the point of the class. A
price panel can be structurally perfect and economically impossible — a low
above its high, a negative volume, a close that jumps 400% because a split was
applied to one field and not another. Those survive a ``read_csv`` untouched
and reach the optimizer as a low-volatility asset with a spectacular Sharpe.
Catching them here costs one pass over the data and saves an allocation nobody
can explain.

The engine still consumes plain ``DataFrame``s; :meth:`PricePanel.prices` hands
one over. The panel is what carries the *provenance* alongside it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import PanelValidationError

#: Fields every adapter is required to return on one common adjustment scale,
#: and therefore the only ones the session-range check may compare.
#: ``m_close_raw`` is excluded by construction: it is the unadjusted print, so
#: on a dividend-paying stock it sits outside an adjusted high/low by design.
SCALE_CONSISTENT_FIELDS: tuple[str, ...] = (F.OPEN, F.CLOSE)


@dataclass(frozen=True)
class SeriesMeta:
    """Where one column came from and what it represents.

    Attributes:
        identifier: The name the column carries in the panel — the engine-side
            symbol, not necessarily the one the provider was asked for.
        provider_symbol: What the provider was actually queried with. Differs
            from ``identifier`` whenever the catalog translated a symbol
            (``SP500`` becomes ``^GSPC`` on Yahoo and ``SP500`` on FRED).
        provider: Registered name of the provider that served the series.
        kind: What sort of instrument this is. Drives whether missing volume
            is a defect and whether the series is tradeable at all.
        currency: ISO code the series is quoted in, or ``None`` when the
            provider does not say.
        name: Human-readable description, when published.
        exchange: Listing venue, when published.
    """

    identifier: str
    provider_symbol: str
    provider: str
    kind: F.InstrumentKind = F.InstrumentKind.UNKNOWN
    currency: str | None = None
    name: str | None = None
    exchange: str | None = None

    def describe(self) -> str:
        """One line naming the instrument, its kind, and where it came from.

        Returns:
            Something like ``"S&P 500 (index · USD · NYSE) via stooq"``. The
            display name is used when the provider supplied one, and the
            engine-side identifier otherwise.
        """
        label = self.name or self.identifier
        venue = f" | {self.exchange}" if self.exchange else ""
        ccy = f" | {self.currency}" if self.currency else ""
        return f"{label} ({self.kind.value}{ccy}{venue}) via {self.provider}"


@dataclass(frozen=True)
class PricePanel:
    """A multi-field, multi-identifier price panel with its provenance.

    Attributes:
        frames: Field name (from :mod:`~optimization_engine.ingest.fields`) to
            a DataFrame indexed by date with one column per identifier. Always
            contains :data:`~optimization_engine.ingest.fields.CLOSE`.
        meta: Per-identifier provenance, keyed by column name.
    """

    frames: Mapping[str, pd.DataFrame]
    meta: Mapping[str, SeriesMeta] = field(default_factory=dict)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, pd.DataFrame],
        meta: Mapping[str, SeriesMeta] | None = None,
        *,
        validate: bool = True,
    ) -> PricePanel:
        """Build a panel from per-field frames, normalizing index and columns.

        Args:
            frames: Field name to DataFrame. Frames need not share an index or
                a column order; both are reconciled here.
            meta: Optional provenance per identifier.
            validate: Run :meth:`validate` before returning. Off only for
                tests that deliberately construct a broken panel.

        Returns:
            A panel whose frames all share one sorted date index and one
            column order.

        Raises:
            PanelValidationError: If the input is empty, names an unknown
                field, or fails validation.
        """
        if not frames:
            raise PanelValidationError("A price panel needs at least one field frame.")
        unknown = sorted(set(frames) - set(F.MARKET_FIELDS))
        if unknown:
            raise PanelValidationError(
                f"Unknown field(s) in panel: {', '.join(unknown)}."
            )
        if F.CLOSE not in frames:
            raise PanelValidationError(
                f"A price panel must carry {F.CLOSE!r}; got {sorted(frames)}."
            )

        # One index and one column order across every field, so that
        # ``panel.frames[a].loc[d, c]`` and ``panel.frames[b].loc[d, c]``
        # always describe the same observation.
        columns = list(dict.fromkeys(frames[F.CLOSE].columns))
        index = frames[F.CLOSE].index
        for name, frame in frames.items():
            if name == F.CLOSE:
                continue
            index = index.union(frame.index)
        # ``freq`` is dropped deliberately: a panel built from ``bdate_range``
        # carries one and the same panel read back from Parquet does not, which
        # would make two identical panels compare unequal.
        index = pd.DatetimeIndex(
            pd.DatetimeIndex(pd.to_datetime(index)).sort_values().unique(),
            freq=None,
        )

        aligned: dict[str, pd.DataFrame] = {}
        for name in F.MARKET_FIELDS:
            if name not in frames:
                continue
            frame = frames[name].copy()
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
            frame = frame[~frame.index.duplicated(keep="last")]
            frame = frame.reindex(index=index, columns=columns)
            frame.index.name = "date"
            aligned[name] = frame.astype("float64")

        panel = cls(frames=aligned, meta=dict(meta or {}))
        if validate:
            panel.validate()
        return panel

    @classmethod
    def from_prices(
        cls,
        prices: pd.DataFrame,
        meta: Mapping[str, SeriesMeta] | None = None,
        *,
        validate: bool = True,
    ) -> PricePanel:
        """Wrap a plain close-price frame as a close-only panel.

        Args:
            prices: Prices indexed by date, one column per identifier.
            meta: Per-identifier provenance. Absent identifiers get none.
            validate: Run the structural and economic checks.

        Returns:
            A close-only :class:`PricePanel`.

        Raises:
            PanelValidationError: If ``validate`` is set and the frame fails a
                check — a non-datetime or unsorted index, or non-positive
                prices.
        """
        return cls.from_frames({F.CLOSE: prices}, meta, validate=validate)

    # -- accessors --------------------------------------------------------

    @property
    def identifiers(self) -> tuple[str, ...]:
        """Engine-side names carried by this panel, in column order."""
        return tuple(self.frames[F.CLOSE].columns)

    @property
    def available_fields(self) -> tuple[str, ...]:
        """Which market fields this panel actually carries.

        Returns:
            The subset of the field vocabulary present in :attr:`frames`, in the
            vocabulary's own order. Always includes ``CLOSE``.
        """
        return tuple(f for f in F.MARKET_FIELDS if f in self.frames)

    @property
    def index(self) -> pd.DatetimeIndex:
        """The panel's date index, shared by every field frame."""
        return pd.DatetimeIndex(self.frames[F.CLOSE].index)

    def prices(self) -> pd.DataFrame:
        """The total-return close panel — what the engine optimizes on."""
        return self.frames[F.CLOSE].copy()

    def frame(self, name: str) -> pd.DataFrame | None:
        """One field's frame, or ``None`` when the panel does not carry it.

        Args:
            name: A field from
                :mod:`~optimization_engine.ingest.fields`, e.g. ``VOLUME``.

        Returns:
            A copy of that field's frame, or ``None``. A copy rather than a view,
            so a caller cannot mutate the panel by accident.
        """
        found = self.frames.get(name)
        return None if found is None else found.copy()

    def volumes(self) -> pd.DataFrame | None:
        """Traded volume, or ``None`` when no identifier reports any.

        The distinction matters to the cost model: ``None`` means "this
        universe has no volume, price the impact some other way", while a
        frame of NaNs would mean "volume exists but is missing here".
        """
        volume = self.frames.get(F.VOLUME)
        if volume is None or volume.dropna(how="all").empty:
            return None
        return volume.copy()

    @property
    def has_volume(self) -> bool:
        """Whether any identifier reports traded volume.

        ``False`` is a fact about the universe, not a gap in it: an index panel
        never carries volume, and the cost model prices impact differently rather
        than failing.
        """
        return self.volumes() is not None

    def kinds(self) -> dict[str, F.InstrumentKind]:
        """Instrument kind per identifier.

        Returns:
            ``identifier -> InstrumentKind``, covering every column. An identifier
            with no metadata maps to ``UNKNOWN`` rather than being left out.
        """
        return {
            identifier: self.meta[identifier].kind
            if identifier in self.meta
            else F.InstrumentKind.UNKNOWN
            for identifier in self.identifiers
        }

    @property
    def tradeable(self) -> tuple[str, ...]:
        """Identifiers that can actually be held at their quoted price."""
        return tuple(i for i, k in self.kinds().items() if k.is_tradeable)

    def coverage(self) -> pd.DataFrame:
        """One row per identifier: what arrived, from where, over what span.

        This is the table the app renders after an ingest run, and the
        fastest way to see that one series is three years shorter than the
        rest before that fact shows up as a suspiciously stable covariance.
        """
        close = self.frames[F.CLOSE]
        rows = []
        for identifier in self.identifiers:
            series = close[identifier].dropna()
            meta = self.meta.get(identifier)
            present = [
                name
                for name in self.available_fields
                if not self.frames[name][identifier].dropna().empty
            ]
            rows.append(
                {
                    "identifier": identifier,
                    "provider": meta.provider if meta else "-",
                    "symbol": meta.provider_symbol if meta else identifier,
                    "kind": (meta.kind if meta else F.InstrumentKind.UNKNOWN).value,
                    "currency": (meta.currency if meta else None) or "-",
                    "observations": int(series.shape[0]),
                    "first_date": series.index.min() if not series.empty else pd.NaT,
                    "last_date": series.index.max() if not series.empty else pd.NaT,
                    "fields": ", ".join(f.removeprefix("m_") for f in present),
                    "has_volume": F.VOLUME in present,
                }
            )
        return pd.DataFrame(rows).set_index("identifier")

    # -- transformation ---------------------------------------------------

    def select(self, identifiers: Iterable[str]) -> PricePanel:
        """A panel restricted to ``identifiers``, preserving their order.

        Args:
            identifiers: The subset to keep, in the order the result should carry.

        Returns:
            A new panel over that subset, with its metadata narrowed to match.

        Raises:
            PanelValidationError: If an identifier is not in the panel.
        """
        wanted = list(dict.fromkeys(str(i) for i in identifiers))
        missing = [i for i in wanted if i not in self.frames[F.CLOSE].columns]
        if missing:
            raise PanelValidationError(
                f"Identifier(s) not in panel: {', '.join(missing)}."
            )
        return PricePanel(
            frames={name: frame[wanted].copy() for name, frame in self.frames.items()},
            meta={i: m for i, m in self.meta.items() if i in wanted},
        )

    def merge(self, other: PricePanel) -> PricePanel:
        """Union two panels along both axes.

        Used when a universe is served by more than one provider — equities
        from one, the index they are benchmarked against from another. Fields
        present in only one side are kept, with NaN for the identifiers the
        other side supplied.

        Args:
            other: The panel to merge in. On an identifier collision it wins
                **outright**: its column replaces the left one rather than filling
                the left one's gaps. Filling gaps would splice two providers'
                price levels into a single series — one vendor's 100 next to
                another's 10 — producing return spikes that never happened, under
                a provenance record naming one source. A series comes from one
                provider or the other.

        Returns:
            A new panel over the union of both universes and both date ranges.
        """
        names = [f for f in F.MARKET_FIELDS if f in self.frames or f in other.frames]
        index = self.index.union(other.index).sort_values()
        columns = list(dict.fromkeys([*self.identifiers, *other.identifiers]))
        overridden = set(other.identifiers)

        merged: dict[str, pd.DataFrame] = {}
        for name in names:
            left = self.frames.get(name)
            right = other.frames.get(name)
            frame = pd.DataFrame(index=index, columns=columns, dtype="float64")
            if left is not None:
                kept = [c for c in left.columns if c not in overridden]
                if kept:
                    frame[kept] = left[kept].reindex(index=index)
            if right is not None:
                frame[list(right.columns)] = right.reindex(index=index)
            frame.index.name = "date"
            merged[name] = frame

        return PricePanel.from_frames(merged, {**self.meta, **other.meta})

    def with_meta(self, meta: Mapping[str, SeriesMeta]) -> PricePanel:
        """A copy of this panel with additional or replaced series metadata.

        Args:
            meta: ``identifier -> SeriesMeta`` entries to merge in. Existing keys
                are overwritten; the frames are untouched.

        Returns:
            A new :class:`PricePanel`. The original is unchanged.
        """
        return replace(self, meta={**self.meta, **dict(meta)})

    # -- validation -------------------------------------------------------

    def validate(self) -> None:
        """Reject panels that are structurally or economically impossible.

        Checked, in order:

        * the index is a strictly increasing ``DatetimeIndex``;
        * every field frame shares that index and column order;
        * price fields are strictly positive and finite;
        * volume is non-negative;
        * ``low <= min(open, close, high)`` and ``high >= max(open, close)``,
          within a small tolerance for vendors that round each field
          independently;
        * at least one identifier has at least two observations, since a
          single price yields no return.

        Raises:
            PanelValidationError: On the first failure, naming the field, the
                identifier and the date so the report points at one cell.
        """
        close = self.frames.get(F.CLOSE)
        if close is None:
            raise PanelValidationError(f"Panel is missing the {F.CLOSE!r} frame.")
        if not isinstance(close.index, pd.DatetimeIndex):
            raise PanelValidationError("Panel index must be a DatetimeIndex.")
        if not close.index.is_monotonic_increasing:
            raise PanelValidationError("Panel index must be sorted ascending.")
        if close.index.has_duplicates:
            duplicated = close.index[close.index.duplicated()].unique()[:3]
            raise PanelValidationError(
                f"Panel index has duplicate dates: {list(duplicated)}."
            )

        columns = list(close.columns)
        for name, frame in self.frames.items():
            if not frame.index.equals(close.index):
                raise PanelValidationError(
                    f"Field {name!r} does not share the panel's date index."
                )
            if list(frame.columns) != columns:
                raise PanelValidationError(
                    f"Field {name!r} does not share the panel's identifiers."
                )

        for name, frame in self.frames.items():
            values = frame.to_numpy(dtype="float64", copy=False)
            finite = np.isfinite(values)
            if name in F.PRICE_FIELDS:
                bad = finite & (values <= 0.0)
                if bad.any():
                    self._raise_cell(name, frame, bad, "must be strictly positive")
            elif name == F.VOLUME:
                bad = finite & (values < 0.0)
                if bad.any():
                    self._raise_cell(name, frame, bad, "must be non-negative")
            # +/-inf reaches here as non-finite; NaN is legitimate (a gap),
            # infinity never is.
            infinite = np.isinf(values)
            if infinite.any():
                self._raise_cell(name, frame, infinite, "is infinite")

        self._validate_ohlc_ordering()

        usable = (close.notna().sum(axis=0) >= 2).any()
        if not usable:
            raise PanelValidationError(
                "No identifier has two or more observations; there is no return "
                "series to build."
            )

    def _validate_ohlc_ordering(self) -> None:
        """Check the intraday range brackets the prices inside it.

        Only fields on the *same* adjustment scale may be compared, and that
        is the whole subtlety here. :data:`~optimization_engine.ingest.fields.CLOSE`
        is a total-return series while
        :data:`~optimization_engine.ingest.fields.CLOSE_RAW` is the unadjusted
        print, so on any stock that has paid a dividend the two are legitimately
        several percent apart — and comparing the raw close against an adjusted
        session range would reject a perfectly good panel. Adapters put open,
        high, low and close on one scale (see
        :data:`SCALE_CONSISTENT_FIELDS`); the raw close is deliberately
        excluded.

        Vendors round each field independently, so an exact comparison flags
        harmless last-digit noise. The tolerance is relative to the price
        level: 10 bps of the high, which is far below any real crossing and
        far above any rounding.
        """
        high = self.frames.get(F.HIGH)
        low = self.frames.get(F.LOW)
        if high is None or low is None:
            return
        tolerance = 1e-3

        crossed = (low - high) > tolerance * high.abs()
        if crossed.to_numpy().any():
            self._raise_cell(
                F.LOW, low, crossed.to_numpy(), "exceeds the session high"
            )

        for name in SCALE_CONSISTENT_FIELDS:
            inner = self.frames.get(name)
            if inner is None:
                continue
            above = (inner - high) > tolerance * high.abs()
            if above.to_numpy().any():
                self._raise_cell(name, inner, above.to_numpy(), "exceeds the session high")
            below = (low - inner) > tolerance * low.abs()
            if below.to_numpy().any():
                self._raise_cell(name, inner, below.to_numpy(), "falls below the session low")

    @staticmethod
    def _raise_cell(
        name: str, frame: pd.DataFrame, mask: np.ndarray, problem: str
    ) -> None:
        rows, cols = np.nonzero(mask)
        row, col = int(rows[0]), int(cols[0])
        date = frame.index[row]
        identifier = frame.columns[col]
        value = frame.iat[row, col]
        count = int(mask.sum())
        suffix = f" ({count} such values in total)" if count > 1 else ""
        raise PanelValidationError(
            f"{name} for {identifier} on {pd.Timestamp(date).date()} {problem}: "
            f"got {value!r}{suffix}."
        )


__all__ = ["SCALE_CONSISTENT_FIELDS", "PricePanel", "SeriesMeta"]
