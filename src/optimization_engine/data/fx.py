"""Foreign-exchange handling.

Converts a multi-currency price panel into a single base currency so
the optimizer sees a homogeneous universe.

Conventions:

* ``fx_to_base[ccy]`` is the multiplier that converts **one unit** of
  ``ccy`` into the base currency. ``fx_to_base[base] == 1.0``.
* Conversion is applied at the price level:
  ``price_base[t] = price_ccy[t] * fx_to_base[ccy][t]``.
  Returns are then computed from the converted prices, which yields the
  correct currency-hedged-return-free total return for the base
  investor.

Sources:

* FRED — default. Most major USD pairs are published as ``DEXxxUS`` /
  ``DEXUSxx`` series. We invert when the published quote is the wrong
  way around for our base.
* Direct user input — pass a DataFrame to ``convert_prices_to_base``
  via ``fx_rates``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from optimization_engine.data.fred import FREDError, load_fred_series

# Pre-computed mapping of USD pairs to FRED series IDs. The boolean
# indicates whether the series is quoted as "X per USD" (True, so to get
# X→USD we invert) or "USD per X" (False, so the value is X→USD already).
_FRED_USD_PAIRS: dict[str, tuple[str, bool]] = {
    "MXN": ("DEXMXUS", True),    # Mexican Peso per 1 USD
    "EUR": ("DEXUSEU", False),   # USD per 1 EUR
    "GBP": ("DEXUSUK", False),   # USD per 1 GBP
    "JPY": ("DEXJPUS", True),    # JPY per 1 USD
    "CAD": ("DEXCAUS", True),    # CAD per 1 USD
    "CHF": ("DEXSZUS", True),    # CHF per 1 USD
    "AUD": ("DEXUSAL", False),   # USD per 1 AUD
    "BRL": ("DEXBZUS", True),    # BRL per 1 USD
    "KRW": ("DEXKOUS", True),    # KRW per 1 USD
    "INR": ("DEXINUS", True),    # INR per 1 USD
    "CNY": ("DEXCHUS", True),    # CNY per 1 USD
    "HKD": ("DEXHKUS", True),    # HKD per 1 USD
    "SEK": ("DEXSDUS", True),    # SEK per 1 USD
    "NOK": ("DEXNOUS", True),    # NOK per 1 USD
    "DKK": ("DEXDNUS", True),    # DKK per 1 USD
    "ZAR": ("DEXSFUS", True),    # ZAR per 1 USD
    "TWD": ("DEXTAUS", True),    # TWD per 1 USD
    "SGD": ("DEXSIUS", True),    # SGD per 1 USD
    "MYR": ("DEXMAUS", True),    # MYR per 1 USD
}


SUPPORTED_CURRENCIES = sorted({"USD", *_FRED_USD_PAIRS.keys()})


class FXError(RuntimeError):
    """Raised when FX conversion can't be completed."""


def supported_currencies() -> list[str]:
    """Currencies the built-in FRED FX source can handle (out of the box)."""
    return list(SUPPORTED_CURRENCIES)


def fetch_fx_to_usd(
    currencies: Iterable[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch X→USD rates from FRED for the given currencies.

    Args:
        currencies: ISO codes to fetch. See :func:`supported_currencies`.
        start: First date to fetch. ``None`` takes FRED's own start.
        end: Last date to fetch. ``None`` takes today.

    Returns:
        One column per currency, indexed by date, holding the multiplier that
        converts one unit of that currency into USD. USD itself is included as
        a constant ``1.0`` column.

    Raises:
        FXError: If a currency has no built-in FRED mapping, or the fetch
            failed.
    """
    cleaned = _normalize_currencies(currencies)
    requested_usd = "USD" in cleaned
    needed = sorted({c for c in cleaned if c != "USD"})
    bad = [c for c in needed if c not in _FRED_USD_PAIRS]
    if bad:
        raise FXError(
            f"No built-in FRED mapping for currencies: {bad}. "
            f"Supported: {supported_currencies()}"
        )

    if not needed:
        # USD-only request: return a single 1.0 column on today's date.
        if requested_usd:
            today = pd.DatetimeIndex([pd.Timestamp.today().normalize()])
            return pd.DataFrame({"USD": [1.0]}, index=today)
        return pd.DataFrame()

    series_ids = [_FRED_USD_PAIRS[c][0] for c in needed]
    try:
        raw = load_fred_series(series_ids, start=start, end=end)
    except FREDError as exc:
        raise FXError(f"FRED fetch failed: {exc}") from exc

    out = pd.DataFrame(index=raw.index)
    for ccy in needed:
        sid, inverted = _FRED_USD_PAIRS[ccy]
        column = raw[sid]
        out[ccy] = (1.0 / column) if inverted else column
    if requested_usd:
        out["USD"] = 1.0
    return out


def fetch_fx_to_base(
    currencies: Iterable[str],
    base: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch X→base rates by triangulating through USD.

    Args:
        currencies: ISO codes to fetch. See :func:`supported_currencies` for
            what the built-in FRED source can serve.
        base: ISO code of the target currency.
        start: First date to fetch. ``None`` takes FRED's own start.
        end: Last date to fetch. ``None`` takes today.

    Returns:
        One column per currency, indexed by date, holding the multiplier that
        converts one unit of that currency into ``base``. ``base`` itself is
        included as a constant ``1.0`` column.

    Raises:
        FXError: If FRED returned nothing for the requested currencies, or the
            base→USD leg could not be sourced.
    """
    base = base.upper()
    cleaned = _normalize_currencies(currencies)
    universe = sorted({base, *cleaned})

    if universe == [base]:
        return pd.DataFrame({base: [1.0]}, index=pd.DatetimeIndex([pd.Timestamp.today().normalize()]))

    fx_to_usd = fetch_fx_to_usd(universe, start=start, end=end)
    if fx_to_usd.empty:
        raise FXError(f"FRED returned no FX data for currencies={universe}.")

    if base == "USD":
        out = fx_to_usd.copy()
        out["USD"] = 1.0
    else:
        if base not in fx_to_usd.columns:
            raise FXError(f"Could not source base->USD rate for {base}.")
        base_to_usd = fx_to_usd[base]
        out = fx_to_usd.div(base_to_usd, axis=0)
        out[base] = 1.0
    return out[[c for c in universe if c in out.columns]].sort_index()


#: How many leading rows a price panel may run ahead of its FX history and
#: still be valued at the first known rate. Covers a request that starts on
#: a US holiday, where FRED has no print; anything longer is refused rather
#: than valued with a rate from the future.
MAX_LEADING_FX_GAP = 5

#: How many business days a price may be valued at the last known rate before
#: the rate counts as stale and the conversion is refused. FRED publishes the
#: H.10 rates once a week, so a daily panel run before Monday's release is
#: normally a week ahead of its newest rate, more around a holiday; two weeks
#: without a print is a series that stopped, and carrying it forward would
#: strip the currency move out of every return after it.
MAX_STALE_FX_DAYS = 10


def convert_prices_to_base(
    prices: pd.DataFrame,
    asset_currency: Mapping[str, str],
    base: str,
    fx_rates: pd.DataFrame | None = None,
    fill: str = "ffill",
) -> pd.DataFrame:
    """Convert a multi-currency price panel into a single base currency.

    Args:
        prices: Price panel (rows = dates, cols = assets). The index must be a
            ``DatetimeIndex``.
        asset_currency: ``asset -> ISO currency code``. Assets missing from
            this map are assumed to already be in ``base``.
        base: ISO code of the desired base currency.
        fx_rates: Pre-fetched X→base rates. When ``None``, they are fetched
            from FRED over the range of ``prices``.
        fill: How to fill FX gaps when business-day calendars do not align.
            ``"ffill"`` (the default) values each price date at the last
            rate on or before it, which is the only direction that uses no
            future information — for at most :data:`MAX_STALE_FX_DAYS`
            business days, past which the rate is stale and the conversion
            is refused.
            A gap *before* the first known rate cannot be forward-filled;
            it is back-filled from the first rate for at most
            :data:`MAX_LEADING_FX_GAP` rows — the case of a request that
            starts on a holiday — and refused beyond that, because valuing
            a month of prices at a rate from a month later is look-ahead.
            ``"bfill"`` back-fills without limit; ``None`` skips filling.

    Returns:
        A new frame of the same shape as ``prices``, valued in ``base``.

    Raises:
        FXError: If the price index is not a ``DatetimeIndex``, a rate for
            one of the panel's currencies is unavailable, the rate history
            starts more than :data:`MAX_LEADING_FX_GAP` rows after the prices
            do, or (under ``"ffill"``) a price date's newest rate is more
            than :data:`MAX_STALE_FX_DAYS` business days old.
        ValueError: On a ``fill`` other than ``"ffill"``, ``"bfill"`` or
            ``None``.
    """
    base = base.upper()
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise FXError("Price index must be a DatetimeIndex.")

    needed = {asset_currency.get(a, base).upper() for a in prices.columns}
    if needed == {base}:
        return prices.copy()

    if fx_rates is None:
        fx_rates = fetch_fx_to_base(
            sorted(needed),
            base=base,
            start=prices.index.min(),
            end=prices.index.max(),
        )

    fx_rates = fx_rates.copy()
    fx_rates.index = pd.to_datetime(fx_rates.index)
    fx_rates = fx_rates.sort_index()
    # Fill over both calendars, then read the price dates: an as-of join.
    # Reindexing onto the price dates first discards every rate that falls
    # between two of them, so a month-end on a weekend took the previous
    # month-end's rate instead of that Friday's.
    both = fx_rates.reindex(fx_rates.index.union(prices.index))
    if fill == "ffill":
        aligned = both.ffill().reindex(prices.index).bfill(limit=MAX_LEADING_FX_GAP)
    elif fill == "bfill":
        aligned = both.bfill().reindex(prices.index)
    elif fill is None:
        aligned = both.reindex(prices.index)
    else:
        raise ValueError(f"fill must be 'ffill', 'bfill' or None; got {fill!r}.")

    out = prices.copy()
    for asset in prices.columns:
        ccy = asset_currency.get(asset, base).upper()
        if ccy == base:
            continue
        if ccy not in aligned.columns:
            raise FXError(f"FX rate for {ccy}->{base} not available.")
        rate = aligned[ccy].astype(float)
        if rate.isna().any():
            first_rate = fx_rates[ccy].dropna().index.min()
            raise FXError(
                f"The {ccy}->{base} rate history starts {first_rate.date()}, "
                f"more than {MAX_LEADING_FX_GAP} rows after the prices do "
                f"({prices.index.min().date()}). Trim the panel to where the "
                "rate exists, or pass fx_rates that cover it - filling the "
                "gap from a later rate would value those prices with "
                "information from the future."
            )
        if fill == "ffill":
            _refuse_stale_rate(fx_rates[ccy], prices.index, f"{ccy}->{base}")
        out[asset] = prices[asset].astype(float) * rate
    return out


def _refuse_stale_rate(rates: pd.Series, dates: pd.DatetimeIndex, pair: str) -> None:
    """Raise when a date would be valued at a rate too old to stand for it.

    Age is counted in business days strictly after the rate's own date, up to
    and including the price date — so a Friday rate for a Sunday price is zero
    days old, and for the Monday after, one. Dates before the first rate are
    the leading gap's business, not this check's.
    """
    known = rates.dropna().index
    position = known.searchsorted(dates, side="right") - 1
    covered = position >= 0
    if not covered.any():
        return
    used = known[position[covered]]
    day = np.timedelta64(1, "D")
    age = np.busday_count(
        used.values.astype("datetime64[D]") + day,
        dates[covered].values.astype("datetime64[D]") + day,
    )
    stale = age > MAX_STALE_FX_DAYS
    if not stale.any():
        return
    first = int(np.argmax(stale))
    raise FXError(
        f"{int(stale.sum())} price date(s) from {dates[covered][first].date()} "
        f"would be valued at a {pair} rate more than {MAX_STALE_FX_DAYS} "
        f"business days old - the newest rate before that date is from "
        f"{used[first].date()}. Trim the panel to where the rate exists, or "
        "pass fx_rates that cover it; carrying a stopped series forward "
        "removes the currency move from every return after it."
    )


def _normalize_currencies(currencies: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for c in currencies:
        if not isinstance(c, str):
            raise FXError(f"Currency must be a string; got {type(c).__name__}")
        code = c.strip().upper()
        if not code or len(code) != 3 or not code.isalpha():
            raise FXError(f"Invalid ISO 4217 currency code: {c!r}")
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out
