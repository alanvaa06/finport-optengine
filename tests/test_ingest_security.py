"""Tests for the properties that must hold when a key or a hostile response
is in play.

Each of these encodes an invariant the rest of the layer is documented as
having: a key never reaches a message, a credential never crosses a host
boundary, and a provider cannot make the client read forever. They are here
because every one of them was, at some point, false.
"""

from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.ingest import credentials  # noqa: E402
from optimization_engine.ingest.errors import (  # noqa: E402
    ProviderCredentialsError,
    ProviderResponseError,
)
from optimization_engine.ingest.providers.base import (  # noqa: E402
    PriceProvider,
    ProviderCapabilities,
    _safe_charset,
    _SafeRedirectHandler,
)
from optimization_engine.ingest.service import ingest  # noqa: E402
from optimization_engine.ingest.spec import IngestRequest  # noqa: E402

SECRET = "sk-live-SUPERSECRETVALUE-0123456789"


class _Provider(PriceProvider):
    name = "probe"
    description = "test double"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(requires_key=True)

    def fetch_one(self, identifier, request):  # pragma: no cover - unused
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Keys must not survive into any message
# ---------------------------------------------------------------------------


def test_a_key_with_a_line_break_is_refused_without_being_echoed(monkeypatch):
    # A wrapped paste. Left alone it reaches the HTTP client, which raises a
    # ValueError whose message is the whole header — key included.
    monkeypatch.setenv("OPTENGINE_API_KEY_TIINGO", f"sk-live\n{SECRET}")
    with pytest.raises(ProviderCredentialsError) as caught:
        credentials.resolve_api_key("tiingo")
    assert SECRET not in str(caught.value)
    assert "OPTENGINE_API_KEY_TIINGO" in str(caught.value)


@pytest.mark.parametrize("bad", ["\r", "\n", "\t", "\x1b", "\x7f"])
def test_every_control_character_is_refused(monkeypatch, bad):
    monkeypatch.setenv("OPTENGINE_API_KEY_FMP", f"abc{bad}def")
    with pytest.raises(ProviderCredentialsError):
        credentials.resolve_api_key("fmp")


def test_a_null_byte_is_refused_on_the_explicit_key_path():
    # The environment cannot carry a NUL, but a caller passing a key directly
    # can, and it breaks the HTTP client the same way.
    with pytest.raises(ProviderCredentialsError):
        credentials.resolve_api_key("fmp", "abc\0def")


def test_surrounding_whitespace_is_stripped_rather_than_refused(monkeypatch):
    # A trailing newline from a heredoc or an editor is not a malformed key,
    # it is an ordinary one with whitespace around it.
    monkeypatch.setenv("OPTENGINE_API_KEY_FMP", f"  {SECRET}\n")
    assert credentials.resolve_api_key("fmp") == SECRET


def test_key_status_reports_a_malformed_key_without_raising(monkeypatch):
    # Runs on every Streamlit rerun; a bad key is a thing to report, not to
    # crash on. The break has to be *inside* the value — one at the end is
    # ordinary whitespace and is simply stripped.
    monkeypatch.setenv("OPTENGINE_API_KEY_FMP", f"sk-live\n{SECRET}")
    status = credentials.key_status("fmp", required=True)
    assert status.malformed
    assert not status.ready
    assert SECRET not in status.label


def test_an_unclassified_request_failure_reports_only_its_type():
    # The backstop. Anything raised inside the request path that is not an
    # HTTP or socket error stringifies with what it was handed — which can be
    # the request headers, which carry the key.
    provider = _Provider(api_key=SECRET)

    class _Exploding:
        def open(self, *_args, **_kwargs):
            raise ValueError(f"Invalid header value b'Token {SECRET}'")

    PriceProvider._shared_opener = _Exploding()
    try:
        with pytest.raises(ProviderResponseError) as caught:
            provider._get_text("https://example.test/x", endpoint="probe endpoint")
    finally:
        PriceProvider._shared_opener = None

    message = str(caught.value)
    assert SECRET not in message
    assert "ValueError" in message


def test_the_service_does_not_relay_an_adapter_exception_message():
    class Leaky(PriceProvider):
        name = "leaky"
        description = "test double"

        @property
        def capabilities(self):
            return ProviderCapabilities(accepts_any_symbol=True)

        def fetch_one(self, identifier, request):
            raise ValueError(f"boom with {SECRET}")

    from optimization_engine.ingest.errors import ProviderConfigurationError

    request = IngestRequest(identifiers=["AAA"], provider="leaky", period="1y")
    with pytest.raises(ProviderConfigurationError) as caught:
        ingest(request, provider=Leaky())
    assert SECRET not in str(caught.value)
    assert "ValueError" in str(caught.value)


def test_a_short_secret_is_masked_completely():
    assert credentials.mask("abcdefghi") == "*" * 9
    assert "abcdefghi" not in credentials.mask("abcdefghi")
    # A real 32-character key can afford a recognizable hint.
    hinted = credentials.mask(SECRET)
    assert hinted.startswith("sk-") and hinted.endswith("6789")
    assert "SUPERSECRET" not in hinted


# ---------------------------------------------------------------------------
# Redirects must not carry credentials to a new host
# ---------------------------------------------------------------------------


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"Authorization": f"Token {SECRET}", "User-Agent": "test"}
    )


def _redirect(handler, from_url: str, to_url: str):
    return handler.redirect_request(
        _request(from_url),
        io.BytesIO(b""),
        302,
        "Found",
        {"location": to_url},
        to_url,
    )


def test_a_same_host_redirect_keeps_the_credential():
    handler = _SafeRedirectHandler()
    redirected = _redirect(
        handler, "https://api.example.test/a", "https://api.example.test/b"
    )
    assert redirected.get_header("Authorization") == f"Token {SECRET}"


def test_a_cross_host_redirect_drops_the_credential():
    # A compromised or misconfigured provider must not be able to bounce the
    # client somewhere else and be handed the key.
    handler = _SafeRedirectHandler()
    redirected = _redirect(
        handler, "https://api.example.test/a", "https://attacker.test/collect"
    )
    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("Cookie") is None


def test_a_downgrade_to_http_is_refused():
    handler = _SafeRedirectHandler()
    with pytest.raises(urllib.error.HTTPError, match="non-https"):
        _redirect(handler, "https://api.example.test/a", "http://api.example.test/a")


def test_a_redirect_to_a_file_url_is_refused():
    handler = _SafeRedirectHandler()
    with pytest.raises(urllib.error.HTTPError, match="non-https"):
        _redirect(handler, "https://api.example.test/a", "file:///etc/passwd")


# ---------------------------------------------------------------------------
# A provider cannot make the client read forever
# ---------------------------------------------------------------------------


class _Response:
    """Minimal stand-in for what ``opener.open`` returns."""

    def __init__(self, payload: bytes, charset: str | None = "utf-8") -> None:
        self._payload = payload
        self.headers = _Headers(charset)

    def read(self, size: int | None = None) -> bytes:
        return self._payload if size is None else self._payload[:size]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Headers:
    def __init__(self, charset: str | None) -> None:
        self._charset = charset

    def get_content_charset(self) -> str | None:
        return self._charset


class _Opener:
    def __init__(self, response) -> None:
        self._response = response

    def open(self, *_args, **_kwargs):
        return self._response


def _with_opener(response):
    provider = _Provider(api_key=SECRET)
    PriceProvider._shared_opener = _Opener(response)
    return provider


def test_a_response_over_the_cap_is_refused():
    provider = _with_opener(_Response(b"x" * 4096))
    provider._MAX_BYTES = 1024
    try:
        with pytest.raises(ProviderResponseError, match="refusing to read further"):
            provider._get_text("https://example.test/x", endpoint="big endpoint")
    finally:
        PriceProvider._shared_opener = None


def test_a_response_at_the_cap_is_accepted():
    provider = _with_opener(_Response(b"y" * 1024))
    provider._MAX_BYTES = 1024
    try:
        assert provider._get_text("https://example.test/x") == "y" * 1024
    finally:
        PriceProvider._shared_opener = None


def test_the_cap_is_not_retried_as_if_it_were_transient():
    # Reading four times what the cap allows is not an improvement on reading
    # it once, so the size failure must escape the retry loop immediately.
    calls = {"n": 0}

    class _Counting(_Opener):
        def open(self, *args, **kwargs):
            calls["n"] += 1
            return super().open(*args, **kwargs)

    provider = _Provider(api_key=SECRET)
    provider._MAX_BYTES = 8
    PriceProvider._shared_opener = _Counting(_Response(b"z" * 64))
    try:
        with pytest.raises(ProviderResponseError):
            provider._get_text("https://example.test/x")
    finally:
        PriceProvider._shared_opener = None
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# A provider-supplied charset cannot break the request path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared", [None, "", "not-a-real-codec", "utf-8", "latin-1"]
)
def test_an_unusable_declared_charset_falls_back_to_utf8(declared):
    assert _safe_charset(declared) in {"utf-8", declared}
    # Whatever it returns must be a codec that actually exists.
    "x".encode(_safe_charset(declared))


def test_a_bogus_charset_header_does_not_fail_the_fetch():
    provider = _with_opener(_Response(b"payload", charset="totally-made-up"))
    try:
        assert provider._get_text("https://example.test/x") == "payload"
    finally:
        PriceProvider._shared_opener = None
