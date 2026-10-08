"""The spine: one function that turns a request into a validated panel.

Everything above this module is a description of what is wanted; everything
below is a description of where it lives. :func:`ingest` is the only place
that knows both, and it is deliberately the only place in the ingest layer
that has any control flow worth reading:

1. resolve the provider and let it reject what it cannot serve, before any
   network call;
2. translate engine-side names into that provider's symbols;
3. answer from the cache when the request has been served before;
4. fetch — one batched call where the API supports it, a bounded thread pool
   otherwise — with each identifier's failure isolated from the rest;
5. validate, convert currency, and check the volume policy;
6. return the panel *and* a log of exactly what happened to every identifier.

That last point is the design's whole opinion. A pipeline that returns only
data invites you to trust it; one that returns data plus a per-identifier
account of what arrived, what was skipped, and why, makes the gaps impossible
to miss. Eleven of twelve assets loading is not a success, and the caller
should not have to compare column counts to find out.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from optimization_engine.data.fx import FXError, convert_prices_to_base
from optimization_engine.ingest import catalog
from optimization_engine.ingest import fields as F
from optimization_engine.ingest.cache import CacheEntry, PanelCache
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    IngestError,
    PanelValidationError,
    ProviderConfigurationError,
    ProviderCredentialsError,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta
from optimization_engine.ingest.providers.base import PriceProvider
from optimization_engine.ingest.registry import get_provider
from optimization_engine.ingest.spec import IngestRequest

_LOG = logging.getLogger(__name__)

#: Outcomes an identifier can end a run in.
STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_FAILED = "failed"
STATUS_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class IdentifierOutcome:
    """What happened to one identifier during a run.

    Attributes:
        identifier: The engine-side name that was requested.
        symbol: The provider symbol it was translated to.
        status: One of :data:`STATUS_OK`, :data:`STATUS_MISSING`,
            :data:`STATUS_FAILED`, :data:`STATUS_UNSUPPORTED`.
        observations: Non-null closes that arrived.
        message: Why, when the status is not ``ok``.
    """

    identifier: str
    symbol: str
    status: str
    observations: int = 0
    message: str = ""

    @property
    def ok(self) -> bool:
        """Whether this identifier's data arrived."""
        return self.status == STATUS_OK


@dataclass(frozen=True)
class IngestResult:
    """A panel plus the full account of how it was assembled.

    Attributes:
        panel: The validated, homogenized data.
        request: The request that produced it, for the record.
        outcomes: One entry per requested identifier, in request order.
        warnings: Run-level notes that belong to no single identifier —
            currency fallbacks, absent volume, a cache hit.
        elapsed_seconds: Wall-clock duration of the fetch.
        from_cache: Whether the panel was served from disk.
        cache_entry: Details of the cache hit, when there was one.
    """

    panel: PricePanel
    request: IngestRequest
    outcomes: tuple[IdentifierOutcome, ...] = ()
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    from_cache: bool = False
    cache_entry: CacheEntry | None = None

    @property
    def prices(self) -> pd.DataFrame:
        """The close panel, ready for the engine."""
        return self.panel.prices()

    @property
    def volumes(self) -> pd.DataFrame | None:
        """Traded volume, or ``None`` when this universe has none."""
        return self.panel.volumes()

    @property
    def loaded(self) -> tuple[str, ...]:
        """Identifiers that arrived, in request order."""
        return tuple(o.identifier for o in self.outcomes if o.ok)

    @property
    def failed(self) -> tuple[IdentifierOutcome, ...]:
        """The outcomes that did not arrive, each carrying its own reason.

        Returns:
            Full :class:`IdentifierOutcome` records rather than bare names, so a
            caller can tell a missing ticker from a provider that timed out.
        """
        return tuple(o for o in self.outcomes if not o.ok)

    @property
    def is_complete(self) -> bool:
        """Whether every requested identifier arrived."""
        return all(o.ok for o in self.outcomes)

    def summary(self) -> str:
        """One line for a log or a status bar."""
        got = len(self.loaded)
        asked = len(self.outcomes)
        source = "cache" if self.from_cache else self.request.provider
        span = ""
        if not self.panel.index.empty:
            span = (
                f", {self.panel.index.min().date()}→{self.panel.index.max().date()}"
            )
        return (
            f"{got}/{asked} identifiers from {source} "
            f"({len(self.panel.index):,} rows{span}) in {self.elapsed_seconds:.1f}s"
        )

    def report(self) -> pd.DataFrame:
        """The per-identifier outcome table the CLI and app render."""
        return pd.DataFrame(
            [
                {
                    "identifier": o.identifier,
                    "symbol": o.symbol,
                    "status": o.status,
                    "observations": o.observations,
                    "message": o.message,
                }
                for o in self.outcomes
            ]
        ).set_index("identifier")


def ingest(
    request: IngestRequest,
    *,
    provider: PriceProvider | None = None,
    api_key: str | None = None,
    use_cache: bool = True,
    **provider_options: Any,
) -> IngestResult:
    """Fetch, homogenize and validate a price panel.

    Args:
        request: What to fetch.
        provider: An already-built provider. When ``None``, one is created
            from ``request.provider`` through the registry. Passing an
            instance is how tests inject a fake without touching the registry.
        api_key: Overrides the environment for this run only.
        use_cache: Read from and write to ``request.cache_dir``. Ignored when
            no cache directory is set.
        **provider_options: Extra keyword arguments for the provider factory
            (``path=`` for the file provider, ``seed=`` for the sample one).

    Returns:
        An :class:`IngestResult` holding the panel and a per-identifier log.

    Raises:
        ProviderNotFoundError: The provider name is not registered.
        ProviderConfigurationError: The request asks for something the
            provider cannot serve, or nothing at all could be loaded.
        ProviderCredentialsError: A required key is missing or rejected.
    """
    started = time.perf_counter()
    source = provider or get_provider(
        request.provider, api_key=api_key, **provider_options
    )
    source.preflight(request)

    symbol_by_identifier, unsupported = catalog.translate(
        request.identifiers,
        source.name,
        passthrough=source.capabilities.accepts_any_symbol,
    )
    warnings: list[str] = []

    # Resolved before the cache branch, because a warm run must describe the
    # same universe a cold one does — a collision the cold path names would
    # otherwise come back as "the provider returned nothing" on every re-run.
    duplicates = _duplicate_symbols(request.identifiers, symbol_by_identifier, unsupported)
    unsupported = tuple([*unsupported, *duplicates])

    cache = (
        PanelCache(request.cache_dir, request.cache_ttl_seconds)
        if (use_cache and request.cache_dir)
        else None
    )
    # The provider options are part of what was fetched — ``path=`` decides
    # which file the ``file`` provider reads — so they are part of the key,
    # and so is the provider's own token for state the request cannot see.
    cache_key = _cache_key(request, provider_options, source.cache_token())
    if cache is not None:
        cached = cache.load(cache_key)
        if cached is not None:
            panel, entry = cached
            # The fingerprint covers what was fetched, not what the caller
            # will accept — ``require_volume`` is deliberately outside it, so
            # a permissive run and a strict one share an entry. The policy
            # therefore has to be re-applied to the cached panel, or a strict
            # request would pass simply because a lax one ran first.
            cache_warnings = [f"Served from cache, written {entry.age_label}."]
            cache_warnings.extend(
                _translation_notes(source.name, unsupported, duplicates)
            )
            # What the fetch said about the panel itself — an undeclared
            # currency, a conversion — is as true of the cached copy.
            cache_warnings.extend(entry.notes)
            cache_warnings.extend(_volume_notes(panel, request))
            outcomes = _outcomes_from_panel(
                panel, request, symbol_by_identifier, unsupported, None, duplicates
            )
            return IngestResult(
                panel=panel,
                request=request,
                outcomes=outcomes,
                warnings=tuple(cache_warnings),
                elapsed_seconds=time.perf_counter() - started,
                from_cache=True,
                cache_entry=entry,
            )

    fetchable = tuple(i for i in request.identifiers if i not in unsupported)
    warnings.extend(_translation_notes(source.name, unsupported, duplicates))
    if not fetchable:
        raise ProviderConfigurationError(
            f"None of the requested identifiers can be served by {source.name}: "
            f"{', '.join(request.identifiers)}."
        )

    panels, failures = _fetch(source, fetchable, symbol_by_identifier, request)
    if not panels:
        first = next(iter(failures.values()), "no data returned")
        raise ProviderConfigurationError(
            f"{source.name} returned nothing for any of "
            f"{', '.join(fetchable)}. First failure: {first}"
        )

    panel = panels[0]
    for extra in panels[1:]:
        panel = panel.merge(extra)
    panel = _apply_catalog_metadata(panel, source.name)
    panel = _reorder(panel, request.identifiers)

    fx_degraded = False
    panel_notes: list[str] = []
    if request.currency:
        panel, currency_note, fx_degraded = _convert_currency(panel, request.currency)
        if currency_note:
            warnings.append(currency_note)
            panel_notes.append(currency_note)

    warnings.extend(_volume_notes(panel, request))
    outcomes = _outcomes_from_panel(
        panel, request, symbol_by_identifier, unsupported, failures, duplicates
    )

    if cache is not None:
        # Only a panel that is what the request asked for goes into the
        # cache. One with a failed or missing identifier would come back on
        # every warm run for the length of the TTL as "the provider returned
        # no series", and never be retried; one whose currency conversion
        # degraded would be served under a fingerprint that claims the base
        # currency while holding native quotes.
        not_cached = _uncacheable_reason(outcomes, fx_degraded)
        if not_cached:
            warnings.append(f"Not cached: {not_cached}")
        elif not cache.store(cache_key, panel, notes=panel_notes):
            warnings.append(
                "Not cached: the entry could not be written (see the log). An "
                "older entry for this request, if one exists, is still served "
                "until it expires."
            )

    return IngestResult(
        panel=panel,
        request=request,
        outcomes=outcomes,
        warnings=tuple(warnings),
        elapsed_seconds=time.perf_counter() - started,
    )


def _translation_notes(
    provider: str, unsupported: tuple[str, ...], duplicates: tuple[str, ...]
) -> list[str]:
    """Run-level notes about identifiers that never reached the provider.

    Shared by the cold and warm paths so a cached run reports the same
    universe as the fetch that filled it.
    """
    notes: list[str] = []
    if duplicates:
        notes.append(
            f"{', '.join(duplicates)} resolve to a symbol another identifier "
            f"already claims on {provider}, so they name the same instrument "
            "twice. Only the first was fetched."
        )
    no_symbol = tuple(i for i in unsupported if i not in duplicates)
    if no_symbol:
        notes.append(
            f"{provider} has no symbol for {', '.join(no_symbol)}; "
            "they were skipped. Try another provider, or pass the ticker "
            "your provider uses directly."
        )
    return notes


def _duplicate_symbols(
    identifiers: tuple[str, ...],
    symbol_by_identifier: Mapping[str, str],
    unsupported: tuple[str, ...],
) -> tuple[str, ...]:
    """Identifiers whose provider symbol another identifier already claims.

    ``SP500`` and ``^GSPC`` are the same Yahoo series. Fetching both makes one
    column that two identifiers want, and the loser is reported as though the
    provider had returned nothing for it — which is not what happened. Naming
    the collision is the honest answer.
    """
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for identifier in identifiers:
        if identifier in unsupported:
            continue
        symbol = symbol_by_identifier.get(identifier, identifier)
        if symbol in seen:
            duplicates.append(identifier)
        else:
            seen[symbol] = identifier
    return tuple(duplicates)


def _fetch(
    source: PriceProvider,
    identifiers: tuple[str, ...],
    symbol_by_identifier: Mapping[str, str],
    request: IngestRequest,
) -> tuple[list[PricePanel], dict[str, str]]:
    """Run the fetch, batched or concurrent, isolating per-identifier failures.

    Returns the panels that came back and a map of identifier to failure
    message. A credentials failure is re-raised rather than recorded: every
    other identifier is about to fail the same way, and eight identical 401s
    are a worse error report than one.
    """
    caps = source.capabilities
    if caps.supports_batch:
        return _fetch_batched(source, identifiers, symbol_by_identifier, request)

    panels: list[PricePanel] = []
    failures: dict[str, str] = {}
    workers = max(1, min(int(request.max_workers), len(identifiers)))

    if workers == 1:
        for identifier in identifiers:
            panel, message = _fetch_single(
                source, identifier, symbol_by_identifier, request
            )
            if panel is not None:
                panels.append(panel)
            elif message:
                failures[identifier] = message
        return panels, failures

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="ingest"
    ) as pool:
        futures = {
            pool.submit(
                _fetch_single, source, identifier, symbol_by_identifier, request
            ): identifier
            for identifier in identifiers
        }
        for future in concurrent.futures.as_completed(futures):
            identifier = futures[future]
            panel, message = future.result()
            if panel is not None:
                panels.append(panel)
            elif message:
                failures[identifier] = message
    return panels, failures


def _fetch_batched(
    source: PriceProvider,
    identifiers: tuple[str, ...],
    symbol_by_identifier: Mapping[str, str],
    request: IngestRequest,
) -> tuple[list[PricePanel], dict[str, str]]:
    """Fetch in provider-sized chunks, renaming symbols back to engine names."""
    size = max(1, int(source.capabilities.max_batch_size))
    panels: list[PricePanel] = []
    failures: dict[str, str] = {}

    for start in range(0, len(identifiers), size):
        chunk = identifiers[start : start + size]
        symbols = tuple(symbol_by_identifier[i] for i in chunk)
        try:
            panel = source.fetch_batch(symbols, request)
        except ProviderCredentialsError:
            raise
        except IngestError as exc:
            # A whole chunk failing is still scoped to its identifiers; the
            # other chunks may well succeed.
            for identifier in chunk:
                failures[identifier] = str(exc)
            continue
        except Exception as exc:
            # The same rule as the per-identifier path: an unclassified
            # exception's text has been vetted by nobody, and this string
            # travels to a log, the CLI's stderr and the browser.
            _LOG.exception(
                "Unexpected failure fetching %s from %s",
                ", ".join(chunk), source.name,
            )
            message = (
                f"{type(exc).__name__} in the {source.name} adapter "
                "(see the log for the full traceback)"
            )
            for identifier in chunk:
                failures[identifier] = message
            continue
        panels.append(_rename_to_identifiers(panel, chunk, symbol_by_identifier))

    return panels, failures


def _fetch_single(
    source: PriceProvider,
    identifier: str,
    symbol_by_identifier: Mapping[str, str],
    request: IngestRequest,
) -> tuple[PricePanel | None, str]:
    """Fetch one identifier, converting expected failures into a message.

    Runs on a worker thread. Only :class:`ProviderCredentialsError` escapes,
    because it is fatal for the whole run.
    """
    symbol = symbol_by_identifier[identifier]
    try:
        panel = source.fetch_one(symbol, request)
    except ProviderCredentialsError:
        raise
    except IdentifierNotFoundError as exc:
        return None, str(exc)
    except IngestError as exc:
        return None, str(exc)
    except Exception as exc:  # a provider bug must not take down the run
        _LOG.exception("Unexpected failure fetching %s from %s", identifier, source.name)
        # Only the type is reported. An unclassified exception is by
        # definition one whose message nobody has vetted, and the messages
        # that reach here travel to a log, the CLI's stderr and the browser.
        # A provider puts its API key in a request header.
        return None, (
            f"{type(exc).__name__} in the {source.name} adapter "
            "(see the log for the full traceback)"
        )
    return _rename_to_identifiers(panel, (identifier,), symbol_by_identifier), ""


def _rename_to_identifiers(
    panel: PricePanel,
    identifiers: Sequence[str],
    symbol_by_identifier: Mapping[str, str],
) -> PricePanel:
    """Rename provider symbols back to engine-side names.

    This is what makes the catalog worth having: the panel that leaves this
    function is labelled ``SP500`` whether the bytes came from Yahoo's
    ``^GSPC`` or FRED's ``SP500``.
    """
    by_symbol = {symbol_by_identifier[i]: i for i in identifiers}
    close_columns = list(panel.frames[F.CLOSE].columns)
    rename = {c: by_symbol[c] for c in close_columns if c in by_symbol}
    if not rename:
        return panel

    frames = {
        name: frame.rename(columns=rename) for name, frame in panel.frames.items()
    }
    meta = {}
    for symbol, record in panel.meta.items():
        identifier = by_symbol.get(symbol, symbol)
        meta[identifier] = SeriesMeta(
            identifier=identifier,
            provider_symbol=record.provider_symbol,
            provider=record.provider,
            kind=record.kind,
            currency=record.currency,
            name=record.name,
            exchange=record.exchange,
        )
    return PricePanel(frames=frames, meta=meta)


def _apply_catalog_metadata(panel: PricePanel, provider: str) -> PricePanel:
    """Let the catalog correct a provider's guesses about kind and currency.

    A provider infers ``INDEX`` from a leading caret and gives up otherwise.
    The catalog *knows*, for the instruments it covers, so it wins — which is
    what makes ``IPC`` come back as an MXN index rather than an unknown
    instrument in an unknown currency.
    """
    updates: dict[str, SeriesMeta] = {}
    for identifier in panel.identifiers:
        entry = catalog.lookup(identifier)
        if entry is None:
            continue
        current = panel.meta.get(identifier)
        updates[identifier] = SeriesMeta(
            identifier=identifier,
            provider_symbol=current.provider_symbol if current else identifier,
            provider=current.provider if current else provider,
            kind=entry.kind,
            currency=entry.currency,
            name=entry.name,
            exchange=current.exchange if current else None,
        )
    return panel.with_meta(updates) if updates else panel


def _reorder(panel: PricePanel, identifiers: tuple[str, ...]) -> PricePanel:
    """Present the panel in the order the universe was written in."""
    present = [i for i in identifiers if i in panel.identifiers]
    extra = [i for i in panel.identifiers if i not in present]
    return panel.select([*present, *extra])


def _cache_key(
    request: IngestRequest,
    provider_options: Mapping[str, Any],
    token: str | None = None,
) -> str:
    """The request's fingerprint, extended by whatever else shapes the data.

    Without the options the ``file`` provider served file A's panel for file
    B, and the ``sample`` provider one seed's panel for another, whenever the
    identifiers and window matched; without the provider's token it served
    file A's panel from before an edit. A request with neither keeps the bare
    fingerprint, so existing cache entries stay valid.
    """
    fingerprint = request.fingerprint()
    if not provider_options and token is None:
        return fingerprint
    payload: dict[str, Any] = {"fingerprint": fingerprint, "options": provider_options}
    if token is not None:
        payload["token"] = token
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _uncacheable_reason(
    outcomes: tuple[IdentifierOutcome, ...], fx_degraded: bool
) -> str:
    """Why this panel must not be cached, or ``""`` when it may be."""
    if fx_degraded:
        return "the currency conversion fell back to native quotes."
    not_ok = [o.identifier for o in outcomes if o.status in (STATUS_FAILED, STATUS_MISSING)]
    if not_ok:
        return (
            f"{', '.join(not_ok)} did not arrive, and a cached panel would not "
            "be retried."
        )
    return ""


def _convert_currency(panel: PricePanel, base: str) -> tuple[PricePanel, str, bool]:
    """Convert every price field into ``base``.

    Volume is left alone — it counts shares, not money — and every price field
    is converted with the same rates, so the OHLC ordering that
    :meth:`PricePanel.validate` checks survives the conversion.

    A conversion that cannot be sourced degrades to the native quote with a
    warning rather than failing the run: a panel in mixed currencies that says
    so is more useful than no panel at all. The third value says whether that
    happened, so the caller can keep the degraded panel out of the cache.
    """
    known = {
        identifier: record.currency
        for identifier, record in panel.meta.items()
        if record.currency
    }
    unknown = [i for i in panel.identifiers if i not in known]
    asset_currency = {**known, **{i: base for i in unknown}}
    if set(asset_currency.values()) <= {base}:
        if unknown:
            return panel, (
                f"{', '.join(unknown)} do not say what currency they are quoted "
                f"in, so they were left as they arrived rather than converted "
                f"to {base}. If any of them is not already in {base}, its "
                "returns are not comparable with the rest."
            ), False
        return panel, "", False

    converted: dict[str, pd.DataFrame] = {}
    try:
        for name, frame in panel.frames.items():
            if name not in F.PRICE_FIELDS:
                converted[name] = frame
                continue
            converted[name] = convert_prices_to_base(frame, asset_currency, base)
    except (FXError, PanelValidationError) as exc:
        return panel, (
            f"Could not convert to {base} ({exc}); series are shown in their "
            "native currencies."
        ), True

    # Only series whose currency was actually known get stamped with the base.
    # One with no declared currency was *assumed* to be in the base already;
    # labelling it as converted would turn that assumption into a claim.
    updated = {
        identifier: SeriesMeta(
            identifier=record.identifier,
            provider_symbol=record.provider_symbol,
            provider=record.provider,
            kind=record.kind,
            currency=base if identifier in known else None,
            name=record.name,
            exchange=record.exchange,
        )
        for identifier, record in panel.meta.items()
    }
    foreign = sorted({c for i, c in asset_currency.items() if c != base and i in known})
    note = f"Converted {', '.join(foreign)} into {base} using FRED daily rates."
    if unknown:
        note += (
            f" {', '.join(unknown)} declare no currency and were left as they "
            f"arrived — they are only comparable if they were already in {base}."
        )
    return PricePanel.from_frames(converted, updated), note, False


def _volume_notes(panel: PricePanel, request: IngestRequest) -> list[str]:
    """Report on volume, and enforce ``require_volume`` when it is set.

    The default is permissive on purpose. An index universe has no volume and
    the backtest is built to price impact without one; failing here would make
    the most common index workflow impossible. ``require_volume=True`` flips
    that for strategies whose costs genuinely depend on turnover, where a
    silent fallback to a fixed participation rate would flatter the result.

    Raises:
        ProviderConfigurationError: When ``require_volume`` is set and an
            instrument that should have volume did not report any.
    """
    if not request.wants_volume:
        return []

    volume = panel.frames.get(F.VOLUME)
    kinds = panel.kinds()
    expected = [i for i, kind in kinds.items() if kind.has_volume]
    missing = [
        identifier
        for identifier in expected
        if volume is None or volume[identifier].dropna().empty
    ]
    volume_free = [i for i, kind in kinds.items() if not kind.has_volume]

    if missing and request.require_volume:
        raise ProviderConfigurationError(
            f"require_volume is set, but {', '.join(missing)} came back without "
            f"volume from {request.provider}. Either choose a provider that "
            "publishes it, or clear require_volume and let the backtest price "
            "impact from a fixed participation rate."
        )

    notes: list[str] = []
    if missing:
        notes.append(
            f"No volume for {', '.join(missing)}; the backtest will price "
            "their impact from the fixed participation rate instead."
        )
    if volume_free:
        notes.append(
            f"{', '.join(volume_free)} are index or rate levels and carry no "
            "volume by construction — expected, not a gap."
        )
    return notes


def _outcomes_from_panel(
    panel: PricePanel,
    request: IngestRequest,
    symbol_by_identifier: Mapping[str, str],
    unsupported: tuple[str, ...],
    failures: Mapping[str, str] | None = None,
    duplicates: tuple[str, ...] = (),
) -> tuple[IdentifierOutcome, ...]:
    """Build the per-identifier log, in the order the universe was requested."""
    failures = failures or {}
    close = panel.frames[F.CLOSE]
    outcomes: list[IdentifierOutcome] = []

    for identifier in request.identifiers:
        symbol = symbol_by_identifier.get(identifier, identifier)
        if identifier in unsupported:
            claimed = duplicates and identifier in duplicates
            outcomes.append(
                IdentifierOutcome(
                    identifier=identifier,
                    symbol=symbol if claimed else "—",
                    status=STATUS_UNSUPPORTED,
                    message=(
                        f"Resolves to {symbol!r}, which another identifier in "
                        "this universe already claims."
                        if claimed
                        else f"{request.provider} publishes no symbol for this instrument."
                    ),
                )
            )
            continue
        if identifier in failures:
            outcomes.append(
                IdentifierOutcome(
                    identifier=identifier,
                    symbol=symbol,
                    status=STATUS_FAILED,
                    message=failures[identifier],
                )
            )
            continue
        if identifier not in close.columns:
            outcomes.append(
                IdentifierOutcome(
                    identifier=identifier,
                    symbol=symbol,
                    status=STATUS_MISSING,
                    message="The provider returned no series for this identifier.",
                )
            )
            continue

        observations = int(close[identifier].notna().sum())
        if observations == 0:
            outcomes.append(
                IdentifierOutcome(
                    identifier=identifier,
                    symbol=symbol,
                    status=STATUS_MISSING,
                    observations=0,
                    message="Every observation in the window was missing.",
                )
            )
            continue
        outcomes.append(
            IdentifierOutcome(
                identifier=identifier,
                symbol=symbol,
                status=STATUS_OK,
                observations=observations,
            )
        )
    return tuple(outcomes)


__all__ = [
    "STATUS_FAILED",
    "STATUS_MISSING",
    "STATUS_OK",
    "STATUS_UNSUPPORTED",
    "IdentifierOutcome",
    "IngestResult",
    "ingest",
]
