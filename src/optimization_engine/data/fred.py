"""FRED (Federal Reserve Economic Data) loader.

Pulls daily/weekly/monthly economic series straight from the public
``fredgraph.csv`` endpoint:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>

No API key is required for this endpoint. To keep the dependency
surface flat we use ``pandas.read_csv`` directly — tests mock the
``_fetch_fred_csv`` helper so they run offline.

Common series:

* Risk-free / rates: ``DGS3MO``, ``DGS1``, ``DGS5``, ``DGS10``,
  ``DGS30``, ``DFF``, ``EFFR``, ``CPIAUCSL``.
* FX (USD vs.): ``DEXMXUS``, ``DEXUSEU``, ``DEXUSUK``, ``DEXJPUS``,
  ``DEXCAUS``, ``DEXSZUS``, ``DEXUSAL``, ``DEXBZUS``, ``DEXKOUS``,
  ``DEXINUS``, ``DEXCHUS``.
* Equity / vol: ``VIXCLS``, ``SP500``, ``NASDAQCOM``.
"""

from __future__ import annotations

import io
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence

import pandas as pd

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# FRED series IDs are uppercase letters, digits, underscore. Length cap
# is generous; the longest known IDs are around 20 chars.
_SERIES_ID_PATTERN = re.compile(r"^[A-Z0-9_]{1,30}$")

_USER_AGENT = "optimization-engine/0.2 (+https://github.com/alanvaa06/finport-optengine)"

#: Ceiling on one series' CSV body. Generous by three orders of magnitude for
#: real data, and the difference between a bad endpoint costing a request and
#: it costing the process.
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class FREDError(RuntimeError):
    """Raised when FRED can't be reached or input/output is invalid."""


def _validate_series_ids(series_ids: Sequence[str]) -> list[str]:
    if not series_ids:
        raise FREDError("At least one FRED series id is required.")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in series_ids:
        if not isinstance(raw, str):
            raise FREDError(f"Series id must be a string; got {type(raw).__name__}")
        sid = raw.strip().upper()
        if not _SERIES_ID_PATTERN.match(sid):
            raise FREDError(
                f"Rejected FRED series id {raw!r}: only A-Z, 0-9, _ are allowed."
            )
        if sid not in seen:
            seen.add(sid)
            cleaned.append(sid)
    return cleaned


def _fetch_fred_csv(
    series_id: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch a single FRED series as a CSV. Mocked in tests.

    Returns a single-column DataFrame indexed by the series' DATE column.
    """
    params = [f"id={series_id}"]
    if start is not None:
        params.append(f"cosd={pd.Timestamp(start).date().isoformat()}")
    if end is not None:
        params.append(f"coed={pd.Timestamp(end).date().isoformat()}")
    url = f"{_FRED_CSV_URL}?{'&'.join(params)}"

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # A single FRED series is at most a few hundred kilobytes. Reading
            # without a bound lets a malfunctioning or hostile endpoint hold
            # the process until it runs out of memory.
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.URLError as exc:
        raise FREDError(f"Could not fetch FRED series {series_id}: {exc}") from exc

    if len(payload) > _MAX_RESPONSE_BYTES:
        raise FREDError(
            f"FRED returned more than {_MAX_RESPONSE_BYTES // (1024 * 1024)} MB "
            f"for {series_id}; refusing to read further."
        )

    try:
        df = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:  # malformed CSV / HTML error page
        raise FREDError(f"Could not parse FRED response for {series_id}: {exc}") from exc

    if df.empty:
        raise FREDError(f"FRED returned an empty payload for {series_id}.")

    date_col = next((c for c in df.columns if c.lower() in ("date", "observation_date")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    df.index.name = "date"

    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise FREDError(f"FRED response for {series_id} has no value column.")

    series = pd.to_numeric(df[value_cols[0]], errors="coerce")
    return series.to_frame(name=series_id)


def load_fred_series(
    series_ids: str | Iterable[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    drop_missing: bool = True,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Download one or more FRED series and combine them into a single frame.

    Args:
        series_ids: Either a single series id (``"DGS10"``) or an
            iterable of ids. Strings can also be comma-/whitespace-
            separated.
        start: Optional inclusive start date.
        end: Optional inclusive end date.
        drop_missing: Drop rows where all series are NaN. FRED uses
            ``"."`` for missing observations; those become NaN here.
        timeout: HTTP timeout per series.

    Returns:
        A wide DataFrame with one column per series, indexed by date.
    """
    if isinstance(series_ids, str):
        parts = [t for t in re.split(r"[\s,]+", series_ids) if t]
    else:
        parts = list(series_ids)
    cleaned = _validate_series_ids(parts)

    frames: list[pd.DataFrame] = []
    for sid in cleaned:
        frames.append(_fetch_fred_csv(sid, start=start, end=end, timeout=timeout))

    out = pd.concat(frames, axis=1).sort_index()
    if drop_missing:
        out = out.dropna(how="all")
    return out


def load_risk_free_rate(
    series_id: str = "DGS10",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Fetch a single rate series and return it as decimals.

    FRED publishes rates in percent; this divides by 100 so they can be used
    directly in the engine — ``0.045`` for a 4.5% yield.

    Args:
        series_id: The FRED series, e.g. ``"DGS10"`` for the 10-year Treasury
            constant-maturity yield.
        start: First date to fetch. ``None`` takes the series' own start.
        end: Last date to fetch. ``None`` takes today.

    Returns:
        The rate as a decimal fraction, indexed by date.

    Raises:
        FREDError: On a malformed series id, an unreachable FRED, or a series
            that came back empty.
    """
    df = load_fred_series([series_id], start=start, end=end)
    return df[series_id] / 100.0
