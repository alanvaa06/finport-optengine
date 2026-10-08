"""Local files — the provider for data that never came from an API.

Most real mandates run on a panel somebody maintains by hand: a spreadsheet of
fund NAVs, an extract from a custodian, a series that exists nowhere public.
Making that a *provider* rather than a special case means such a panel gets
the same validation, the same coverage table and the same provenance record as
a Yahoo pull, and can be merged with one.

Two layouts are accepted, distinguished by their columns:

* **Wide** — one row per date, one column per identifier, values are closes.
  This is what the engine's existing loaders produce and what most people have.
* **Long** — one row per (date, identifier), with named ``open`` / ``high`` /
  ``low`` / ``close`` / ``volume`` columns. The only way to express OHLCV in a
  flat file, and what a database export looks like.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from optimization_engine._optional import require
from optimization_engine.ingest import fields as F
from optimization_engine.ingest.adjust import rescale_frames_to_adjusted
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    ProviderConfigurationError,
    ProviderResponseError,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta
from optimization_engine.ingest.providers.base import PriceProvider, ProviderCapabilities
from optimization_engine.ingest.spec import IngestRequest

#: Long-format column names, lower-cased, mapped onto the vocabulary.
_LONG_COLUMNS = {
    "open": F.OPEN,
    "high": F.HIGH,
    "low": F.LOW,
    "close": F.CLOSE,
    "adj_close": F.CLOSE,
    "adjclose": F.CLOSE,
    "adjusted_close": F.CLOSE,
    "close_raw": F.CLOSE_RAW,
    "volume": F.VOLUME,
    "vwap": F.VWAP,
}

#: Which source column wins when a file offers more than one candidate for the
#: same field, most-preferred first. Without this the winner is whichever
#: column the file happens to list first — so the conventional
#: ``…,close,adj_close,…`` ordering would discard the adjusted series and hand
#: the optimizer a price return labelled as a total return. When both are
#: present the loser is kept as ``m_close_raw`` rather than thrown away.
_CLOSE_PREFERENCE: tuple[str, ...] = (
    "adj_close", "adjclose", "adjusted_close", "close",
)

_IDENTIFIER_COLUMNS = ("identifier", "ticker", "symbol", "asset", "id")
_DATE_COLUMNS = ("date", "datetime", "timestamp", "fecha")


class LocalFile(PriceProvider):
    """A price panel read from a CSV, Excel or Parquet file on disk."""

    name = "file"
    description = (
        "A panel you already have: CSV, Excel or Parquet, in wide "
        "(date × asset) or long (date, identifier, OHLCV) layout."
    )

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        sheet_name: str | int | None = "Precios",
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Bind the provider to one file on disk.

        Args:
            path: The CSV, Excel or Parquet file to read. May be ``None`` here and
                supplied later, but a fetch without it is refused.
            sheet_name: Worksheet to read from an Excel workbook. Ignored for the
                other formats.
            api_key: Unused; accepted so every provider constructs the same way.
            timeout: Unused; this provider makes no network call.
        """
        super().__init__(api_key=api_key, timeout=timeout)
        self._path = Path(path) if path is not None else None
        self._sheet_name = sheet_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Every market field, no key, no network, any symbol the file names.

        Returns:
            Capabilities advertising offline batch reads. Instrument kind is
            ``UNKNOWN``: a file says nothing about what its columns are.
        """
        return ProviderCapabilities(
            fields=frozenset(F.MARKET_FIELDS),
            intervals=frozenset({"1d", "1wk", "1mo"}),
            kinds=frozenset({F.InstrumentKind.UNKNOWN}),
            requires_key=False,
            supports_batch=True,
            max_batch_size=10_000,
            accepts_any_symbol=True,
            is_offline=True,
            notes="No network. What the file says is what you get.",
        )

    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        """One identifier, by way of :meth:`fetch_batch`.

        The file is read whole either way, so there is nothing to save by
        fetching a single column.

        Args:
            identifier: The engine-side name.
            request: The run's window, interval and requested fields.

        Returns:
            A single-column panel. See :meth:`fetch_batch` for what can be raised.
        """
        return self.fetch_batch((identifier,), request)

    def fetch_batch(
        self, identifiers: tuple[str, ...], request: IngestRequest
    ) -> PricePanel:
        """Read the file and return the requested identifiers as a panel.

        Both layouts are accepted: wide (dates down, one column per asset) and
        long (a date column, an identifier column, and OHLCV columns). The
        layout is detected from the column names rather than declared.

        Args:
            identifiers: Engine-side names to pull out of the file.
            request: The run's window and requested fields. The window trims the
                file's own history; it never extends it.

        Returns:
            A validated panel over whichever of ``identifiers`` the file carries.

        Raises:
            ProviderConfigurationError: If no path was set, the path is not a
                file, or the extension is not one of ``.csv``, ``.xlsx``,
                ``.xls``, ``.xlsm`` or ``.parquet``.
        """
        if self._path is None:
            raise ProviderConfigurationError(
                "The file provider needs a path: LocalFile(path=...) or "
                "`--file-path` on the command line."
            )
        if not self._path.is_file():
            raise ProviderConfigurationError(f"No such file: {self._path}")

        raw = self._read()
        frames = _to_field_frames(raw, source=self._path)
        frames = _slice_window(frames, request)
        frames = _select_identifiers(frames, identifiers, source=self._path)

        meta = {
            identifier: SeriesMeta(
                identifier=identifier,
                provider_symbol=identifier,
                provider=self.name,
                kind=F.InstrumentKind.UNKNOWN,
                # Unknown, like the kind. Stamping the request's *target*
                # currency here told the conversion step there was nothing
                # to convert, and a peso file came back labelled dollars.
                currency=None,
                name=str(self._path.name),
            )
            for identifier in frames[F.CLOSE].columns
        }
        return PricePanel.from_frames(frames, meta)

    def _read(self) -> pd.DataFrame:
        suffix = self._path.suffix.lower()  # type: ignore[union-attr]
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(self._path)
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            require("openpyxl", extra="excel", purpose="reading Excel workbooks")
            return pd.read_excel(self._path, sheet_name=self._sheet_name)
        if suffix == ".parquet":
            return pd.read_parquet(self._path)
        raise ProviderConfigurationError(
            f"Unsupported file extension {suffix!r}. "
            "Use .csv, .xlsx, .xls, .xlsm or .parquet."
        )


def _to_field_frames(raw: pd.DataFrame, *, source: Path) -> dict[str, pd.DataFrame]:
    """Detect the layout and return homogenized per-field frames."""
    lowered = {str(c).strip().lower(): c for c in raw.columns}
    identifier_column = next(
        (lowered[c] for c in _IDENTIFIER_COLUMNS if c in lowered), None
    )
    date_column = next((lowered[c] for c in _DATE_COLUMNS if c in lowered), None)

    if identifier_column is not None and date_column is not None:
        return _from_long(raw, date_column, identifier_column, source=source)

    # Wide layout: the first column is the date index and every remaining
    # numeric column is one asset's close.
    frame = raw.copy()
    index_column = date_column or frame.columns[0]
    index = pd.DatetimeIndex(pd.to_datetime(frame[index_column], errors="coerce"))
    frame = frame.drop(columns=[index_column])
    frame = frame.loc[index.notna()]
    index = index[index.notna()]
    frame.index = index
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").sort_index()
    if numeric.empty or numeric.shape[1] == 0:
        raise ProviderResponseError(
            f"{source.name} has no numeric price columns after parsing its "
            "first column as dates."
        )
    return {F.CLOSE: numeric}


def _from_long(
    raw: pd.DataFrame, date_column: object, identifier_column: object, *, source: Path
) -> dict[str, pd.DataFrame]:
    """Pivot a long-format file into one frame per field."""
    frame = raw.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame[date_column], errors="coerce"))
    frame = frame.loc[index.notna()].copy()
    frame["__date__"] = index[index.notna()]
    frame["__id__"] = frame[identifier_column].astype(str).str.strip().str.upper()

    by_lower = {str(c).strip().lower(): c for c in raw.columns}

    def pivot(column: object) -> pd.DataFrame:
        """Pivot one long-format column into a date x identifier frame.

        Args:
            column: The source column to pivot.

        Returns:
            A frame indexed by date with one column per identifier. Values are
            coerced to numeric, so a non-numeric cell becomes NaN rather than
            poisoning the column's dtype, and duplicate ``(date, identifier)``
            rows keep the last.
        """
        values = pd.to_numeric(frame[column], errors="coerce")
        pivoted = (
            pd.DataFrame({"__date__": frame["__date__"], "__id__": frame["__id__"], "v": values})
            .pivot_table(index="__date__", columns="__id__", values="v", aggfunc="last")
            .sort_index()
        )
        pivoted.columns.name = None
        pivoted.index.name = "date"
        return pivoted

    # Resolve the close first, by preference rather than by file order.
    close_candidates = [name for name in _CLOSE_PREFERENCE if name in by_lower]
    if not close_candidates:
        raise ProviderResponseError(
            f"{source.name} is in long format but has no close column "
            f"(looked for: {', '.join(sorted(_LONG_COLUMNS))})."
        )

    frames: dict[str, pd.DataFrame] = {F.CLOSE: pivot(by_lower[close_candidates[0]])}
    if close_candidates[0] != "close" and "close" in by_lower:
        # The unadjusted print is worth keeping — it is what turns weights into
        # share counts — but only the genuinely raw column qualifies. Two
        # spellings of the *adjusted* close (``adj_close`` and ``adjclose``)
        # would otherwise produce a "raw" series that is itself adjusted.
        frames[F.CLOSE_RAW] = pivot(by_lower["close"])

    for lower, column in by_lower.items():
        target = _LONG_COLUMNS.get(lower)
        if target is None or target in frames or lower in _CLOSE_PREFERENCE:
            continue
        pivoted = pivot(column)
        if target is F.VOLUME and not (pivoted.fillna(0.0) > 0).any().any():
            continue
        frames[target] = pivoted

    # A terminal export is almost always a raw OHLCV row with an adj_close
    # column bolted on the end, so the range needs the same reconciliation a
    # provider's does.
    if F.CLOSE_RAW not in frames and _has_adjusted_close(by_lower):
        has_range = any(f in frames for f in (F.OPEN, F.HIGH, F.LOW))
        if has_range:
            raise ProviderResponseError(
                f"{source.name} has an adjusted close but no raw `close` "
                "column, so its open/high/low cannot be put on the same scale "
                "— a dividend-adjusted close does not sit inside an "
                "unadjusted day's range. Add the raw `close` column, or drop "
                "the open/high/low columns and load closes alone."
            )
    return rescale_frames_to_adjusted(frames)


def _has_adjusted_close(by_lower: dict[str, object]) -> bool:
    """Whether the file's close column is an adjusted one."""
    return any(name in by_lower for name in ("adj_close", "adjclose", "adjusted_close"))


def _slice_window(
    frames: dict[str, pd.DataFrame], request: IngestRequest
) -> dict[str, pd.DataFrame]:
    """Trim every frame to the request's window.

    A file usually holds more history than the run wants. Trimming here rather
    than downstream keeps the panel honest: its index is the window that was
    asked for, and the coverage table reports what the file actually had
    inside it.
    """
    start = pd.Timestamp(request.start)
    end = pd.Timestamp(request.end)
    return {
        name: frame.loc[(frame.index >= start) & (frame.index <= end)]
        for name, frame in frames.items()
    }


def _select_identifiers(
    frames: dict[str, pd.DataFrame], identifiers: tuple[str, ...], *, source: Path
) -> dict[str, pd.DataFrame]:
    """Keep the requested identifiers, matching case-insensitively."""
    close = frames[F.CLOSE]
    by_upper = {str(c).strip().upper(): c for c in close.columns}
    # Files are written by people, so a column may be ``Cash`` where the run
    # asked for ``CASH``. Match without regard to case, then present the
    # column under the name the caller used.
    pairs = [
        (by_upper[i.strip().upper()], i)
        for i in identifiers
        if i.strip().upper() in by_upper
    ]
    if not pairs:
        raise IdentifierNotFoundError(
            f"{source.name} has none of the requested identifiers "
            f"({', '.join(identifiers)}). It contains: "
            f"{', '.join(sorted(by_upper)[:12])}"
            f"{'…' if len(by_upper) > 12 else ''}."
        )
    wanted = [original for original, _ in pairs]
    renamed = dict(pairs)
    return {
        name: frame.reindex(columns=wanted).rename(columns=renamed)
        for name, frame in frames.items()
    }


__all__ = ["LocalFile"]
