"""The contract every price provider implements, and the plumbing they share.

A provider's job is narrow on purpose: take an
:class:`~optimization_engine.ingest.spec.IngestRequest` and one identifier,
and return a :class:`~optimization_engine.ingest.panel.PricePanel` whose
columns are named in the homogenized vocabulary. Everything else — retries,
concurrency, caching, currency conversion, the run log — belongs to the
service above, so a new provider is one class and a parse function, not a
pipeline.

Two pieces make that work.

:class:`ProviderCapabilities` is a provider's honest self-description: which
fields it can serve, which intervals, which instrument kinds, whether it needs
a key. The service reads it *before* fetching and fails a request that asks
for something impossible — asking Stooq for volume on an index, or FRED for
intraday bars — with a message naming the provider and the gap, instead of a
panel that is quietly missing a column.

:class:`PriceProvider` carries the shared HTTP behaviour: bounded retries with
exponential backoff on the failures that are worth retrying, an error taxonomy
mapped from status codes, and a hard rule that no exception message may ever
contain the request URL, because provider keys travel in query strings.

Implementations must be thread-safe. The service fetches identifiers
concurrently, and the only state a provider is expected to hold is its own
configuration.
"""

from __future__ import annotations

import abc
import json
import logging
import random
import threading
import time
import typing
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    IngestError,
    ProviderConfigurationError,
    ProviderCredentialsError,
    ProviderResponseError,
    ProviderTransientError,
)
from optimization_engine.ingest.panel import PricePanel
from optimization_engine.ingest.spec import IngestRequest

_LOG = logging.getLogger(__name__)

_USER_AGENT = (
    "optimization-engine/0.3 "
    "(+https://github.com/alanvaa06/finport-optengine)"
)

#: Statuses worth trying again: the server is overloaded, throttling, or a
#: gateway blinked. Everything else in 4xx is the request's own fault and
#: retrying it just burns the rate limit.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Hard ceiling on a single response body. A day of daily bars for one symbol
#: is kilobytes; anything approaching this is a provider malfunction or a
#: deliberate attempt to exhaust memory, and either way the run is better off
#: failing than swallowing it. Eight identifiers fetch concurrently, so the
#: real ceiling is eight times this.
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


#: Headers that authenticate the caller to one specific host, and must not
#: travel to another. Compared lower-cased.
_CREDENTIAL_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """A redirect handler that will not hand credentials to a new host.

    urllib's default carries every request header across a redirect, including
    ``Authorization``. A provider that is compromised — or merely
    misconfigured — can therefore bounce the client to a host of its choosing
    and be handed the key. It can also redirect to ``http://``, putting the
    key on the wire in clear.

    Both are refused here. A redirect that stays on the same https host is
    followed normally, because that is what a legitimate one looks like.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Follow a redirect, refusing the two that would leak the key.

        Args:
            req: The original request.
            fp: The response body.
            code: The redirect status code.
            msg: The status message.
            headers: The response headers.
            newurl: Where the server wants to send us.

        Returns:
            The redirected request, with every credential header stripped when the
            new host differs from the original. Header names are matched
            case-insensitively, because urllib stores them capitalized and
            removing them by the caller's own spelling silently misses.

        Raises:
            urllib.error.HTTPError: If the redirect targets a non-https scheme,
                which would put the key on the wire in clear.
        """
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https":
            raise urllib.error.HTTPError(
                req.full_url, code,
                f"refused a redirect to a non-https scheme ({parsed.scheme!r})",
                headers, fp,
            )

        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None

        if parsed.netloc.lower() != urllib.parse.urlsplit(req.full_url).netloc.lower():
            # Cross-host: the new host is not the one the credential was
            # issued for, so it does not get to see it. Matched case-
            # insensitively — urllib stores header names capitalized, so
            # removing them by the spelling the caller used silently misses.
            for name in list(redirected.headers):
                if name.lower() in _CREDENTIAL_HEADERS:
                    redirected.headers.pop(name, None)
        return redirected


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can actually do, declared rather than discovered.

    Attributes:
        fields: The homogenized fields it can serve.
        intervals: The bar sizes it publishes.
        kinds: The instrument kinds it covers.
        requires_key: Whether an API key is mandatory.
        supports_batch: Whether one request can carry several identifiers.
            When true the service calls :meth:`PriceProvider.fetch_batch` once
            instead of :meth:`PriceProvider.fetch_one` per identifier.
        max_batch_size: Cap on identifiers per batched request.
        accepts_any_symbol: Whether the provider serves whatever name it is
            given. True for the synthetic and file providers, whose universe
            is whatever the caller says it is; the symbol catalog is then
            bypassed rather than reporting a catalog instrument as
            unsupported.
        is_offline: Whether a fetch touches the network at all. A caller can
            run an offline provider without asking permission — nothing
            leaves the machine and nothing can be rate-limited — which is
            what lets the app land on working data instead of an empty page.
        signup_url: Where a user goes to get a key.
        rate_limit_per_minute: Documented request ceiling, or ``None``.
        notes: One line the UI shows under the provider's name.
    """

    fields: frozenset[str] = frozenset(F.PRICE_ONLY)
    intervals: frozenset[str] = frozenset({"1d"})
    kinds: frozenset[F.InstrumentKind] = frozenset({F.InstrumentKind.UNKNOWN})
    requires_key: bool = False
    supports_batch: bool = False
    max_batch_size: int = 1
    accepts_any_symbol: bool = False
    is_offline: bool = False
    signup_url: str | None = None
    rate_limit_per_minute: int | None = None
    notes: str = ""

    @property
    def serves_volume(self) -> bool:
        """Whether this provider publishes traded volume at all."""
        return F.VOLUME in self.fields

    def missing_fields(self, requested: tuple[str, ...]) -> tuple[str, ...]:
        """Requested fields this provider cannot serve, in request order.

        Args:
            requested: The fields a request asks for.

        Returns:
            The subset this provider does not publish. Empty means the request is
            servable — which is what :meth:`PriceProvider.preflight` checks before
            any network call.
        """
        return tuple(f for f in requested if f not in self.fields)


class PriceProvider(abc.ABC):
    """Base class for every price source.

    Subclasses implement :meth:`fetch_one` (and :meth:`fetch_batch` when the
    API supports multi-symbol requests) plus :attr:`capabilities`, and get the
    HTTP behaviour below for free.
    """

    #: Registered name. Used in configs, the CLI, and the run log.
    name: str = "base"
    #: One-line description shown in the provider picker.
    description: str = ""

    _MAX_ATTEMPTS = 4
    _BACKOFF_BASE_SECONDS = 0.4
    _TIMEOUT_SECONDS = 30.0
    _MAX_BYTES = _MAX_RESPONSE_BYTES
    _shared_opener: typing.ClassVar[urllib.request.OpenerDirector | None] = None
    _opener_lock: typing.ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, *, api_key: str | None = None, timeout: float | None = None) -> None:
        """Store the credentials and timeout this instance will fetch with.

        Args:
            api_key: The provider's key, or ``None`` for a keyless source. It is
                held privately and travels in a header, never in a URL, and is
                never logged or raised in an exception.
            timeout: Per-request timeout in seconds. Defaults to the class's own.
        """
        self._api_key = api_key
        self._timeout = float(timeout) if timeout is not None else self._TIMEOUT_SECONDS

    # -- contract ---------------------------------------------------------

    @property
    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """What this provider can serve. Read before every fetch."""

    @abc.abstractmethod
    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        """Return a single-identifier panel.

        Args:
            identifier: The provider-side symbol, already translated by the
                catalog and validated by the request.
            request: The run's window, interval and requested fields.

        Returns:
            A panel with exactly one column, named ``identifier``, carrying at
            least :data:`~optimization_engine.ingest.fields.CLOSE`.

        Raises:
            IdentifierNotFoundError: The provider has no such symbol.
            ProviderCredentialsError: The key is missing or rejected.
            ProviderTransientError: A retryable failure survived the retries.
            ProviderResponseError: The payload could not be parsed.
        """

    def fetch_batch(self, identifiers: tuple[str, ...], request: IngestRequest) -> PricePanel:
        """Return a panel for several identifiers in one call.

        The default fans out to :meth:`fetch_one` and merges, which is correct
        for every provider and optimal for none — providers whose API takes a
        symbol list override this and set
        :attr:`ProviderCapabilities.supports_batch`.

        Args:
            identifiers: Provider-side symbols, already translated by the catalog.
            request: The run's window, interval and requested fields.

        Returns:
            A panel over every identifier that returned data.

        Raises:
            ProviderConfigurationError: If ``identifiers`` is empty.
            IdentifierNotFoundError: Propagated from :meth:`fetch_one`.
            ProviderCredentialsError: Propagated from :meth:`fetch_one`.
            ProviderTransientError: Propagated from :meth:`fetch_one`.
            ProviderResponseError: Propagated from :meth:`fetch_one`.
        """
        panel: PricePanel | None = None
        for identifier in identifiers:
            fetched = self.fetch_one(identifier, request)
            panel = fetched if panel is None else panel.merge(fetched)
        if panel is None:
            raise ProviderConfigurationError("fetch_batch requires at least one identifier.")
        return panel

    def validate_credentials(self) -> bool | None:
        """Whether the configured key is usable.

        Returns ``None`` for providers that need no key. The default answers
        from configuration alone; providers with a cheap validation endpoint
        override it to actually ask.
        """
        if not self.capabilities.requires_key:
            return None
        return bool(self._api_key)

    def preflight(self, request: IngestRequest) -> None:
        """Reject a request this provider cannot serve, before any network call.

        Args:
            request: The request to check against :attr:`capabilities`.

        Raises:
            ProviderConfigurationError: The interval or a requested field is
                outside :attr:`capabilities`.
            ProviderCredentialsError: A key is required and none is configured.
        """
        caps = self.capabilities
        if request.interval not in caps.intervals:
            raise ProviderConfigurationError(
                f"{self.name} does not publish {request.interval!r} bars "
                f"(supports: {', '.join(sorted(caps.intervals))})."
            )
        missing = caps.missing_fields(request.fields)
        if missing:
            raise ProviderConfigurationError(
                f"{self.name} cannot serve {', '.join(missing)} "
                f"(serves: {', '.join(sorted(caps.fields))}). "
                "Drop the field, or choose a provider that publishes it."
            )
        if caps.requires_key and not self._api_key:
            from optimization_engine.ingest.credentials import env_var_for

            where = f" Get one at {caps.signup_url}." if caps.signup_url else ""
            raise ProviderCredentialsError(
                f"{self.name} requires an API key. Set {env_var_for(self.name)} "
                f"in the environment or in a .env file.{where}"
            )

    # -- shared HTTP ------------------------------------------------------

    @classmethod
    def _opener(cls) -> urllib.request.OpenerDirector:
        """The shared opener, built once, with the safe redirect handler.

        Built lazily and cached on :class:`PriceProvider` itself so every
        provider and every thread shares one — ``OpenerDirector`` is stateless
        per request.
        """
        if PriceProvider._shared_opener is None:
            with PriceProvider._opener_lock:
                if PriceProvider._shared_opener is None:
                    PriceProvider._shared_opener = urllib.request.build_opener(
                        _SafeRedirectHandler()
                    )
        return PriceProvider._shared_opener

    def _get_text(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        secret_params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        endpoint: str = "endpoint",
    ) -> str:
        """GET a URL with bounded retries, returning the body as text.

        Args:
            url: Base URL, without a query string.
            params: Query parameters safe to name in an error message.
            secret_params: Query parameters that must never be logged or
                raised — API keys go here, and only here.
            headers: Extra request headers. Values are never logged.
            endpoint: Short label used in error messages instead of the URL.

        Raises:
            ProviderCredentialsError: On 401 or 403.
            IdentifierNotFoundError: On 404.
            ProviderTransientError: When retries are exhausted.
            ProviderResponseError: On a non-retryable failure.
        """
        query = {**(params or {}), **(secret_params or {})}
        full_url = f"{url}?{urllib.parse.urlencode(query)}" if query else url
        request_headers = {"User-Agent": _USER_AGENT, **(headers or {})}

        last_error: Exception | None = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                req = urllib.request.Request(full_url, headers=request_headers)
                with self._opener().open(req, timeout=self._timeout) as response:
                    # One byte past the cap is enough to know it was exceeded,
                    # without materializing whatever else was coming.
                    payload = response.read(self._MAX_BYTES + 1)
                    if len(payload) > self._MAX_BYTES:
                        raise ProviderResponseError(
                            f"{self.name} returned more than "
                            f"{self._MAX_BYTES // (1024 * 1024)} MB on {endpoint}; "
                            "refusing to read further."
                        )
                    charset = _safe_charset(response.headers.get_content_charset())
                    return payload.decode(charset, errors="replace")
            except IngestError:
                # Already sanitized and already classified — the cap above,
                # or a nested call. Do not re-wrap it.
                raise
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                if status in (401, 403):
                    raise ProviderCredentialsError(
                        f"{self.name} rejected the API key on {endpoint} "
                        f"(HTTP {status}). Check the key and its entitlements."
                    ) from None
                if status == 404:
                    raise IdentifierNotFoundError(
                        f"{self.name} has no data at {endpoint} (HTTP 404)."
                    ) from None
                if status == 402:
                    raise ProviderCredentialsError(
                        f"{self.name} says this data needs a paid plan "
                        f"(HTTP 402 on {endpoint})."
                    ) from None
                if status not in _RETRYABLE_STATUS:
                    # ``exc`` stringifies with the URL, which carries the key.
                    raise ProviderResponseError(
                        f"{self.name} returned HTTP {status} on {endpoint}."
                    ) from None
                last_error = ProviderTransientError(
                    f"{self.name} returned HTTP {status} on {endpoint}."
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # Only the exception *type* is safe to report: URLError's
                # message can embed the request URL.
                last_error = ProviderTransientError(
                    f"{self.name} could not reach {endpoint} "
                    f"({type(exc).__name__})."
                )
            except Exception as exc:
                # The backstop, and the reason it exists: anything raised in
                # here that is not one of the cases above — a header the http
                # client rejects, a codec that does not exist — stringifies
                # with whatever it was handed, which can be the request's own
                # headers. Those carry the API key. Only the type escapes.
                raise ProviderResponseError(
                    f"{self.name} failed on {endpoint} ({type(exc).__name__})."
                ) from None

            if attempt < self._MAX_ATTEMPTS:
                self._sleep_before_retry(attempt, endpoint)

        raise last_error or ProviderTransientError(
            f"{self.name} failed on {endpoint} after {self._MAX_ATTEMPTS} attempts."
        )

    def _get_json(self, url: str, **kwargs: object) -> object:
        """GET a URL and parse the body as JSON."""
        endpoint = str(kwargs.get("endpoint", "endpoint"))
        body = self._get_text(url, **kwargs)  # type: ignore[arg-type]
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"{self.name} returned a body that is not JSON on {endpoint}: {exc}."
            ) from None

    def _sleep_before_retry(self, attempt: int, endpoint: str) -> None:
        """Exponential backoff with jitter.

        The jitter matters when the service fetches eight identifiers at once
        and they all hit the same 429: without it they retry in lockstep and
        get throttled again together.
        """
        delay = self._BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        delay += random.uniform(0.0, self._BACKOFF_BASE_SECONDS)
        _LOG.warning(
            "%s: retrying %s (attempt %d of %d) in %.2fs",
            self.name, endpoint, attempt + 1, self._MAX_ATTEMPTS, delay,
        )
        time.sleep(delay)


def _safe_charset(declared: str | None) -> str:
    """Resolve a response's declared charset, falling back to UTF-8.

    A provider controls this header, and an unknown codec name makes
    ``bytes.decode`` raise ``LookupError`` from inside the request path. The
    payload is worth more than the label, so an unusable label is ignored.
    """
    if not declared:
        return "utf-8"
    try:
        import codecs

        codecs.lookup(declared)
    except (LookupError, TypeError):
        return "utf-8"
    return declared


@dataclass(frozen=True)
class ProviderInfo:
    """A provider's registry entry, renderable without instantiating it."""

    name: str
    description: str
    capabilities: ProviderCapabilities
    aliases: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "PriceProvider",
    "ProviderCapabilities",
    "ProviderInfo",
]
