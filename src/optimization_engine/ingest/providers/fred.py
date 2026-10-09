"""FRED — index levels with no key, and no volume by construction.

The St. Louis Fed publishes ``SP500``, ``NASDAQCOM``, ``DJIA``, ``VIXCLS`` and
hundreds of other series through a keyless CSV endpoint the library already
speaks (:mod:`optimization_engine.data.fred`). For index-level work that makes
it the most dependable source available: no account, no quota, no unofficial
API that changes shape between releases.

It is also the clearest statement of why the volume path has to be optional.
FRED has never published a volume figure and never will; a universe served
from here is a universe where the square-root impact model has no ADV to work
with. That is a fact about index data, not a defect in the request, so this
adapter simply does not advertise
:data:`~optimization_engine.ingest.fields.VOLUME` in its capabilities and the
service plans around it.

One guard rail: FRED's rate series (``DGS10``, ``DFF``, ``EFFR``) are levels
in percent, not investable prices. Treating one as an asset produces a return
series with no meaning — and, once a rate goes to zero, an infinite one. The
adapter refuses those by name rather than letting them through as an asset
that happens to have a fantastic Sharpe ratio.
"""

from __future__ import annotations

from optimization_engine.data.fred import FREDError, load_fred_series
from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    ProviderConfigurationError,
    ProviderTransientError,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta
from optimization_engine.ingest.providers.base import PriceProvider, ProviderCapabilities
from optimization_engine.ingest.spec import IngestRequest

#: Series that are yields or policy rates published in percent. Not prices.
_RATE_PREFIXES = ("DGS", "DFF", "EFFR", "FEDFUNDS", "TB3", "GS", "MORTGAGE", "SOFR")
_RATE_SERIES = frozenset({"DFF", "EFFR", "SOFR", "FEDFUNDS", "IORB", "DPRIME"})

#: Series known to be equity index levels, so the panel can label them as such
#: rather than falling back to UNKNOWN.
_INDEX_SERIES = frozenset(
    {
        "SP500", "NASDAQCOM", "DJIA", "WILL5000IND", "NASDAQ100",
        "VIXCLS", "VXNCLS", "DJTA", "DJUA", "RU2000PR",
    }
)

#: Interval -> the pandas resampling rule used to downsample daily FRED data.
_RESAMPLE_RULES = {"1wk": "W-FRI", "1mo": "ME"}


class Fred(PriceProvider):
    """Keyless index levels and macro series from FRED."""

    name = "fred"
    description = (
        "Keyless daily index levels (S&P 500, Nasdaq, VIX) from the St. Louis "
        "Fed. Levels only - no OHLC, no volume."
    )

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Close only, keyless, batched up to 25 series.

        Returns:
            Capabilities advertising ``CLOSE`` and nothing else. FRED publishes a
            single level per date, and claiming OHLC or volume here would let a
            request pass preflight and come back short a column.
        """
        return ProviderCapabilities(
            # Deliberately close-only: FRED publishes a single level per date.
            # Advertising OHLC or volume here would let a request pass
            # preflight and then come back short a column.
            fields=frozenset({F.CLOSE}),
            intervals=frozenset({"1d", "1wk", "1mo"}),
            kinds=frozenset({F.InstrumentKind.INDEX, F.InstrumentKind.UNKNOWN}),
            requires_key=False,
            supports_batch=True,
            max_batch_size=25,
            signup_url="https://fred.stlouisfed.org",
            notes="Index levels without an account. Never carries volume.",
        )

    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        """One series, by way of :meth:`fetch_batch`.

        Args:
            identifier: The FRED series id.
            request: The run's window, interval and requested fields.

        Returns:
            A single-column panel. See :meth:`fetch_batch` for what can be raised.
        """
        return self.fetch_batch((identifier,), request)

    def fetch_batch(
        self, identifiers: tuple[str, ...], request: IngestRequest
    ) -> PricePanel:
        # FRED series ids are upper-case; the column keeps whatever the caller
        # asked for, so the panel matches the requested universe exactly.
        """Fetch several FRED series as one panel.

        Series ids are upper-cased for the request and the columns are renamed
        back to whatever the caller asked for, so the panel matches the requested
        universe exactly. Interest-rate series are rejected rather than returned:
        a rate is not a price, and compounding one as if it were produces a
        plausible-looking curve that means nothing.

        Args:
            identifiers: FRED series ids.
            request: The run's window and interval.

        Returns:
            A close-only panel over the requested series.

        Raises:
            ProviderConfigurationError: If any identifier is an interest-rate
                series.
            IdentifierNotFoundError: If FRED has no observations for a series.
        """
        requested_by_id = {s.strip().upper(): s for s in identifiers}
        series_ids = list(requested_by_id)
        rejected = [s for s in series_ids if is_rate_series(s)]
        if rejected:
            raise ProviderConfigurationError(
                f"FRED series {', '.join(rejected)} are interest rates in percent, "
                "not investable price levels - optimizing on them produces "
                "meaningless returns. Use them as a risk-free rate instead "
                "(`optengine fred`, or `load_risk_free_rate`)."
            )

        try:
            frame = load_fred_series(
                series_ids, start=request.start, end=request.end
            )
        except FREDError as exc:
            raise ProviderTransientError(f"FRED request failed: {exc}") from None

        frame = frame.reindex(columns=series_ids)
        frame = frame.dropna(how="all")
        if frame.empty:
            raise IdentifierNotFoundError(
                f"FRED returned no observations for {', '.join(series_ids)} "
                f"between {request.start} and {request.end}."
            )

        rule = _RESAMPLE_RULES.get(request.interval)
        if rule is not None:
            # FRED publishes these daily; taking the last print in each bucket
            # matches how every other provider defines a weekly or monthly close.
            frame = frame.resample(rule).last().dropna(how="all")

        alive = [s for s in series_ids if s in frame.columns and not frame[s].dropna().empty]
        if not alive:
            raise IdentifierNotFoundError(
                f"FRED returned only empty series for {', '.join(series_ids)}."
            )
        frame = frame[alive].rename(columns=requested_by_id)

        meta = {
            requested_by_id[series_id]: SeriesMeta(
                identifier=requested_by_id[series_id],
                provider_symbol=series_id,
                provider=self.name,
                kind=classify(series_id),
                currency="USD",
            )
            for series_id in alive
        }
        return PricePanel.from_frames({F.CLOSE: frame.astype("float64")}, meta)


def is_rate_series(series_id: str) -> bool:
    """Whether a FRED series id names a yield or policy rate.

    Args:
        series_id: A FRED series id, case-insensitive.

    Returns:
        ``True`` for a rate. Prefix matching alone would sweep in legitimate
        index series that happen to start with the same letters, so the check
        requires either an exact hit on a known rate series or a rate prefix
        followed by a maturity code.
    """
    cleaned = series_id.strip().upper()
    if cleaned in _RATE_SERIES:
        return True
    if cleaned in _INDEX_SERIES:
        return False
    for prefix in _RATE_PREFIXES:
        remainder = cleaned.removeprefix(prefix)
        if remainder != cleaned and remainder and len(remainder) <= 4:
            return True
    return False


def classify(series_id: str) -> F.InstrumentKind:
    """What kind of instrument a FRED series id names.

    Args:
        series_id: A FRED series id, case-insensitive.

    Returns:
        ``INDEX`` for the known equity and volatility indices, ``FX`` for the
        ``DEX`` exchange-rate family, ``RATE`` for anything matching the
        interest-rate prefixes, and ``UNKNOWN`` otherwise.
    """
    cleaned = series_id.strip().upper()
    if cleaned in _INDEX_SERIES:
        return F.InstrumentKind.INDEX
    if cleaned.startswith("DEX"):
        return F.InstrumentKind.FX
    if is_rate_series(cleaned):
        return F.InstrumentKind.RATE
    return F.InstrumentKind.UNKNOWN


__all__ = ["Fred", "classify", "is_rate_series"]
