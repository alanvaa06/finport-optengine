"""Synthetic data — the provider that makes the whole pipeline testable offline.

Every other adapter needs a network and, half the time, a key. That is fine in
production and useless in a test suite, a CI run, a tutorial, or the first
thirty seconds someone spends with the app. This provider generates a panel
with the same shape, the same validation and the same provenance record as a
real one, from a seed.

It generates a full OHLCV panel rather than closes alone, which matters for
one specific reason: it is the only way to exercise the volume-aware cost path
without a paid data subscription. Set the universe to
:data:`INDEX_IDENTIFIERS` instead and it produces index levels with no volume
at all, so both halves of the liquidity story can be tested in the same suite.

The prices come from :func:`~optimization_engine.data.loader.sample_dataset`,
so the correlation structure is the one the rest of the library's examples
already use; only the intraday range and the volume are added here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimization_engine.data.loader import sample_dataset
from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import IdentifierNotFoundError
from optimization_engine.ingest.panel import PricePanel, SeriesMeta
from optimization_engine.ingest.providers.base import PriceProvider, ProviderCapabilities
from optimization_engine.ingest.spec import IngestRequest

#: Names that make the provider emit index levels — no volume, kind INDEX —
#: so the volume-free path has something to run against.
INDEX_IDENTIFIERS = frozenset({"SP500", "NDX", "IPC", "STOXX50", "MSCIWORLD", "^GSPC"})

#: How many bars of each interval to draw before resampling.
_BARS_PER_YEAR = {"1d": 252, "1wk": 52, "1mo": 12}


class Sample(PriceProvider):
    """Deterministic synthetic OHLCV, seeded by the requested universe."""

    name = "sample"
    description = (
        "Deterministic synthetic panel with a realistic cross-asset "
        "correlation structure. No network, no key - the default for demos "
        "and tests."
    )

    def __init__(
        self,
        *,
        seed: int = 42,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Fix the seed this provider generates from.

        Args:
            seed: Base seed. Combined with the requested universe, so the same
                request always produces the same panel and two different universes
                do not share a path.
            api_key: Unused; accepted so every provider constructs the same way.
            timeout: Unused; this provider makes no network call.
        """
        super().__init__(api_key=api_key, timeout=timeout)
        self._seed = int(seed)

    @property
    def capabilities(self) -> ProviderCapabilities:
        """OHLCV, no key, no network, any symbol.

        Returns:
            Capabilities advertising offline batch generation for up to 200
            identifiers at any supported interval.
        """
        return ProviderCapabilities(
            fields=frozenset({F.OPEN, F.HIGH, F.LOW, F.CLOSE, F.VOLUME}),
            intervals=frozenset(_BARS_PER_YEAR),
            kinds=frozenset(
                {F.InstrumentKind.EQUITY, F.InstrumentKind.INDEX, F.InstrumentKind.UNKNOWN}
            ),
            requires_key=False,
            supports_batch=True,
            max_batch_size=200,
            accepts_any_symbol=True,
            is_offline=True,
            notes="Reproducible from a seed; identical output for identical input.",
        )

    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        """One identifier, by way of :meth:`fetch_batch`.

        Args:
            identifier: The name to generate.
            request: The run's window, interval and requested fields.

        Returns:
            A single-column panel. See :meth:`fetch_batch` for what can be raised.
        """
        return self.fetch_batch((identifier,), request)

    def fetch_batch(
        self, identifiers: tuple[str, ...], request: IngestRequest
    ) -> PricePanel:
        """Generate a synthetic panel for the requested universe.

        The series carry a realistic cross-asset correlation structure rather
        than independent noise, so a covariance estimated from them is
        ill-conditioned in the way a real one is.

        Args:
            identifiers: Names to generate. Any string is accepted.
            request: The run's window and interval, which set the date index.

        Returns:
            A validated OHLCV panel, identical for identical inputs.

        Raises:
            IdentifierNotFoundError: If no identifiers were given.
        """
        if not identifiers:
            raise IdentifierNotFoundError("The sample provider needs an identifier.")

        dates = _business_index(request)
        if len(dates) < 2:
            raise IdentifierNotFoundError(
                f"The requested window ({request.start} to {request.end}) holds "
                f"fewer than two {request.interval} bars."
            )

        closes = _synthetic_closes(list(identifiers), dates, seed=self._seed)
        kinds = {
            identifier: (
                F.InstrumentKind.INDEX
                if identifier.upper() in INDEX_IDENTIFIERS
                else F.InstrumentKind.EQUITY
            )
            for identifier in identifiers
        }

        frames: dict[str, pd.DataFrame] = {F.CLOSE: closes}
        rng = np.random.default_rng(self._seed + 7)
        if {F.OPEN, F.HIGH, F.LOW} & set(request.fields):
            frames.update(_synthetic_range(closes, rng))
        if F.VOLUME in request.fields:
            volume = _synthetic_volume(closes, kinds, rng)
            if volume is not None:
                frames[F.VOLUME] = volume

        frames = {name: frame for name, frame in frames.items() if name in {*request.fields, F.CLOSE}}
        meta = {
            identifier: SeriesMeta(
                identifier=identifier,
                provider_symbol=identifier,
                provider=self.name,
                kind=kinds[identifier],
                currency="USD",
                name=f"Synthetic {kinds[identifier].value}",
            )
            for identifier in identifiers
        }
        return PricePanel.from_frames(frames, meta)


def _business_index(request: IngestRequest) -> pd.DatetimeIndex:
    """The bar dates covered by the request's window at its interval."""
    start = pd.Timestamp(request.start)
    end = pd.Timestamp(request.end)
    if request.interval == "1d":
        return pd.bdate_range(start=start, end=end)
    freq = {"1wk": "W-FRI", "1mo": "ME"}[request.interval]
    return pd.DatetimeIndex(pd.date_range(start=start, end=end, freq=freq))


def _synthetic_closes(
    identifiers: list[str], dates: pd.DatetimeIndex, *, seed: int
) -> pd.DataFrame:
    """Draw a correlated close panel of the right shape.

    ``sample_dataset`` supplies the correlation structure under its own asset
    names. An identifier that matches one of those names gets that series
    exactly — which makes this provider a drop-in for the generator, so a
    config written against ``Cash`` and ``US_Equity`` keeps meaning what it
    meant. Anything else is assigned a column positionally and perturbed, so
    that asking for forty tickers does not produce forty copies of thirteen
    series.
    """
    base = sample_dataset(n_periods=max(len(dates), 3), seed=seed)
    columns = list(base.columns)
    by_lower = {name.lower(): name for name in columns}

    frame = base.iloc[-len(dates):, :].reset_index(drop=True)
    out = pd.DataFrame(index=dates, dtype="float64")
    for position, identifier in enumerate(identifiers):
        named = by_lower.get(identifier.strip().lower())
        if named is not None:
            # Hand the generator's own series back untouched: perturbing it
            # would change the volatilities and correlations the rest of the
            # library's examples are tuned to, and would make a low-volatility
            # asset like cash several times more volatile than it should be.
            out[identifier] = frame[named].to_numpy(dtype="float64")
            continue

        series = frame[columns[position % len(columns)]].to_numpy(dtype="float64")
        # Recycled columns would otherwise be perfectly collinear, which makes
        # every covariance estimate singular. The wobble is scaled to the
        # series' own volatility so a quiet asset stays quiet.
        rng = np.random.default_rng(seed + 1_000 + position)
        step_vol = float(np.nanstd(np.diff(np.log(series)))) or 0.001
        wobble = np.cumsum(rng.normal(0.0, 0.25 * step_vol, size=series.shape[0]))
        out[identifier] = series * float(1.0 + 0.05 * position) * np.exp(wobble)
    return out


def _synthetic_range(
    closes: pd.DataFrame, rng: np.random.Generator
) -> dict[str, pd.DataFrame]:
    """Build an open/high/low envelope that always brackets the close."""
    values = closes.to_numpy(dtype="float64")
    spread = np.abs(rng.normal(0.0, 0.008, size=values.shape))
    gap = rng.normal(0.0, 0.004, size=values.shape)

    opens = values * (1.0 + gap)
    highs = np.maximum(values, opens) * (1.0 + spread)
    lows = np.minimum(values, opens) * (1.0 - spread)
    return {
        F.OPEN: pd.DataFrame(opens, index=closes.index, columns=closes.columns),
        F.HIGH: pd.DataFrame(highs, index=closes.index, columns=closes.columns),
        F.LOW: pd.DataFrame(lows, index=closes.index, columns=closes.columns),
    }


def _synthetic_volume(
    closes: pd.DataFrame,
    kinds: dict[str, F.InstrumentKind],
    rng: np.random.Generator,
) -> pd.DataFrame | None:
    """Log-normal turnover for tradeable names; nothing for indices.

    Returning ``None`` when every identifier is an index is the behaviour that
    matters: the panel then reports ``has_volume == False`` and the backtest
    picks its volume-free cost path, exactly as it would on real index data.
    """
    tradeable = [i for i, kind in kinds.items() if kind.has_volume]
    if not tradeable:
        return None

    volume = pd.DataFrame(
        float("nan"), index=closes.index, columns=closes.columns, dtype="float64"
    )
    for position, identifier in enumerate(tradeable):
        level = 1e6 * (1.0 + position)
        noise = rng.lognormal(mean=0.0, sigma=0.35, size=len(closes.index))
        volume[identifier] = np.round(level * noise)
    return volume


__all__ = ["INDEX_IDENTIFIERS", "Sample"]
