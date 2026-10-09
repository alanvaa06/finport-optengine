"""Tests for the ingest orchestrator: registry, credentials, cache, service.

The provider adapters are stubbed out here — what is under test is the layer
that plans a run, isolates its failures, records what happened, and enforces
the volume policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.ingest import catalog, credentials  # noqa: E402
from optimization_engine.ingest import fields as F  # noqa: E402
from optimization_engine.ingest.cache import PanelCache  # noqa: E402
from optimization_engine.ingest.errors import (  # noqa: E402
    IdentifierNotFoundError,
    ProviderConfigurationError,
    ProviderCredentialsError,
    ProviderNotFoundError,
    ProviderTransientError,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta  # noqa: E402
from optimization_engine.ingest.providers.base import (  # noqa: E402
    PriceProvider,
    ProviderCapabilities,
)
from optimization_engine.ingest.registry import (  # noqa: E402
    available_providers,
    describe_providers,
    get_provider,
    register_provider,
    resolve_name,
)
from optimization_engine.ingest.service import (  # noqa: E402
    STATUS_FAILED,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    ingest,
)
from optimization_engine.ingest.spec import IngestRequest  # noqa: E402

# ---------------------------------------------------------------------------
# Stub providers
# ---------------------------------------------------------------------------


def _panel_for(identifier: str, *, volume: bool, kind=F.InstrumentKind.EQUITY):
    index = pd.bdate_range("2024-01-01", periods=8)
    close = pd.DataFrame({identifier: np.linspace(100.0, 108.0, 8)}, index=index)
    frames = {F.CLOSE: close}
    if volume:
        frames[F.VOLUME] = pd.DataFrame(
            {identifier: np.full(8, 1_000_000.0)}, index=index
        )
    return PricePanel.from_frames(
        frames,
        {identifier: SeriesMeta(identifier, identifier, "stub", kind, "USD")},
    )


class StubProvider(PriceProvider):
    """A provider whose behaviour per identifier is scripted by the test."""

    name = "stub"
    description = "test double"

    def __init__(self, *, failures=(), volume=True, batch=False, api_key=None, **_):
        super().__init__(api_key=api_key)
        self.failures = dict(failures)
        self.volume = volume
        self.batch = batch
        self.calls: list[tuple[str, ...]] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        served = {F.CLOSE, F.VOLUME} if self.volume else {F.CLOSE}
        return ProviderCapabilities(
            fields=frozenset(served),
            intervals=frozenset({"1d"}),
            requires_key=False,
            supports_batch=self.batch,
            max_batch_size=2,
            accepts_any_symbol=True,
        )

    def fetch_one(self, identifier, request):
        self.calls.append((identifier,))
        problem = self.failures.get(identifier)
        if problem is not None:
            raise problem
        return _panel_for(identifier, volume=self.volume and F.VOLUME in request.fields)

    def fetch_batch(self, identifiers, request):
        if not self.batch:
            return super().fetch_batch(identifiers, request)
        self.calls.append(tuple(identifiers))
        panel = None
        for identifier in identifiers:
            problem = self.failures.get(identifier)
            if problem is not None:
                raise problem
            one = _panel_for(
                identifier, volume=self.volume and F.VOLUME in request.fields
            )
            panel = one if panel is None else panel.merge(one)
        return panel


def _request(identifiers=("AAA", "BBB"), **kwargs) -> IngestRequest:
    kwargs.setdefault("provider", "stub")
    kwargs.setdefault("start", "2024-01-01")
    kwargs.setdefault("end", "2024-01-31")
    return IngestRequest(identifiers=list(identifiers), **kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_builtin_providers_are_registered():
    assert {"sample", "yahoo", "stooq", "fred", "fmp", "tiingo", "file"} <= set(
        available_providers()
    )


def test_aliases_resolve():
    assert resolve_name("yfinance") == "yahoo"
    assert resolve_name("financial_modeling_prep") == "fmp"
    assert resolve_name("SYNTHETIC") == "sample"


def test_unknown_provider_suggests_the_nearest_name():
    with pytest.raises(ProviderNotFoundError, match="Did you mean 'yahoo'"):
        resolve_name("yaho")


def test_registering_over_a_builtin_needs_an_explicit_replace():
    with pytest.raises(ValueError, match="already registered"):
        register_provider("yahoo", StubProvider)


def test_a_third_party_provider_can_be_registered_and_used():
    register_provider(
        "stub_registry_test", StubProvider, description="test", replace=True
    )
    assert isinstance(get_provider("stub_registry_test"), StubProvider)


def test_describe_providers_reports_readiness_without_building_anything():
    rows = {row["provider"]: row for row in describe_providers()}
    assert rows["sample"]["ready"] is True
    assert rows["sample"]["requires_key"] is False
    assert rows["fred"]["serves_volume"] is False
    assert rows["fmp"]["requires_key"] is True
    # Fields come back in OHLCV order rather than alphabetically.
    assert rows["yahoo"]["fields"][0] == F.OPEN


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_key_is_read_from_the_conventional_variable(monkeypatch):
    monkeypatch.setenv("OPTENGINE_API_KEY_FMP", "abcdef123456")
    assert credentials.resolve_api_key("fmp") == "abcdef123456"


def test_explicit_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("OPTENGINE_API_KEY_FMP", "from-env")
    assert credentials.resolve_api_key("fmp", "explicit") == "explicit"


def test_placeholder_values_count_as_no_key(monkeypatch):
    for placeholder in ("", "  ", "your_api_key", "CHANGEME", "none"):
        monkeypatch.setenv("OPTENGINE_API_KEY_FMP", placeholder)
        assert credentials.resolve_api_key("fmp") is None


def test_mask_never_reveals_a_short_secret():
    assert credentials.mask("short") == "*****"
    assert credentials.mask(None) == "-"
    masked = credentials.mask("sk-live-abcdefghijklmnop")
    assert "abcdefghijkl" not in masked
    assert masked.endswith("mnop")


def test_key_status_reports_presence_not_value(monkeypatch):
    monkeypatch.setenv("OPTENGINE_API_KEY_TIINGO", "supersecrettoken")
    status = credentials.key_status("tiingo", required=True)
    assert status.present and status.ready
    assert "supersecrettoken" not in status.label


def test_dotenv_does_not_override_a_real_environment_variable(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('OPTENGINE_API_KEY_FMP="from-file"\nexport OTHER=plain\n')
    monkeypatch.setenv("OPTENGINE_API_KEY_FMP", "from-env")
    credentials.load_dotenv(env)
    assert credentials.resolve_api_key("fmp") == "from-env"
    import os

    assert os.environ["OTHER"] == "plain"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_translates_one_name_into_each_provider_symbol():
    yahoo, _ = catalog.translate(("SP500",), "yahoo")
    fred, _ = catalog.translate(("SP500",), "fred")
    stooq, _ = catalog.translate(("SP500",), "stooq")
    assert yahoo["SP500"] == "^GSPC"
    assert fred["SP500"] == "SP500"
    assert stooq["SP500"] == "^spx"


def test_catalog_passes_unknown_tickers_straight_through():
    resolved, unsupported = catalog.translate(("MSFT",), "yahoo")
    assert resolved["MSFT"] == "MSFT"
    assert unsupported == ()


def test_catalog_reports_an_instrument_a_provider_cannot_serve():
    _, unsupported = catalog.translate(("IPC",), "fred")
    assert unsupported == ("IPC",)


def test_passthrough_skips_translation_entirely():
    resolved, unsupported = catalog.translate(("SP500",), "sample", passthrough=True)
    assert resolved == {"SP500": "SP500"}
    assert unsupported == ()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def test_a_successful_run_reports_every_identifier():
    result = ingest(_request(), provider=StubProvider())
    assert result.is_complete
    assert result.loaded == ("AAA", "BBB")
    assert all(o.status == STATUS_OK for o in result.outcomes)
    assert all(o.observations == 8 for o in result.outcomes)


def test_one_missing_identifier_does_not_lose_the_others():
    stub = StubProvider(failures={"BBB": IdentifierNotFoundError("no such symbol")})
    result = ingest(_request(("AAA", "BBB", "CCC")), provider=stub)

    assert result.loaded == ("AAA", "CCC")
    assert not result.is_complete
    failed = {o.identifier: o for o in result.failed}
    assert failed["BBB"].status == STATUS_FAILED
    assert "no such symbol" in failed["BBB"].message


def test_a_provider_bug_is_contained_to_its_identifier():
    stub = StubProvider(failures={"BBB": RuntimeError("boom with a secret in it")})
    result = ingest(_request(("AAA", "BBB")), provider=stub)
    assert result.loaded == ("AAA",)
    # The type is reported; the message is not. An unclassified exception's
    # text has been vetted by nobody, and this string reaches a log, the CLI's
    # stderr and the browser.
    assert "RuntimeError" in result.failed[0].message
    assert "boom with a secret" not in result.failed[0].message


def test_a_credentials_failure_aborts_the_whole_run():
    # Every other identifier is about to fail the same way; eight identical
    # 401s are a worse report than one.
    stub = StubProvider(failures={"BBB": ProviderCredentialsError("bad key")})
    with pytest.raises(ProviderCredentialsError):
        ingest(_request(("AAA", "BBB")), provider=stub)


def test_a_run_where_nothing_loads_raises_rather_than_returning_an_empty_panel():
    stub = StubProvider(
        failures={
            "AAA": ProviderTransientError("down"),
            "BBB": ProviderTransientError("down"),
        }
    )
    with pytest.raises(ProviderConfigurationError, match="returned nothing"):
        ingest(_request(("AAA", "BBB")), provider=stub)


def test_the_panel_is_ordered_as_the_universe_was_written():
    result = ingest(_request(("CCC", "AAA", "BBB")), provider=StubProvider())
    assert result.panel.identifiers == ("CCC", "AAA", "BBB")


def test_concurrent_and_sequential_fetches_agree():
    sequential = ingest(
        _request(("AAA", "BBB", "CCC"), max_workers=1), provider=StubProvider()
    )
    concurrent = ingest(
        _request(("AAA", "BBB", "CCC"), max_workers=4), provider=StubProvider()
    )
    pd.testing.assert_frame_equal(sequential.prices, concurrent.prices)


def test_a_batch_provider_is_called_in_chunks_of_its_declared_size():
    stub = StubProvider(batch=True)
    ingest(_request(("AAA", "BBB", "CCC")), provider=stub)
    assert [len(call) for call in stub.calls] == [2, 1]


def test_a_failing_chunk_does_not_take_down_the_other_chunks():
    stub = StubProvider(batch=True, failures={"AAA": IdentifierNotFoundError("gone")})
    result = ingest(_request(("AAA", "BBB", "CCC")), provider=stub)
    # AAA and BBB share a chunk, so both are reported failed; CCC survives.
    assert result.loaded == ("CCC",)
    assert {o.identifier for o in result.failed} == {"AAA", "BBB"}


def test_an_instrument_the_provider_cannot_serve_is_reported_not_attempted():
    class NoIndexProvider(StubProvider):
        name = "fred"

        @property
        def capabilities(self):
            return ProviderCapabilities(
                fields=frozenset({F.CLOSE}), intervals=frozenset({"1d"})
            )

    result = ingest(
        _request(("SP500", "IPC"), provider="fred"), provider=NoIndexProvider()
    )
    statuses = {o.identifier: o.status for o in result.outcomes}
    assert statuses["SP500"] == STATUS_OK
    assert statuses["IPC"] == STATUS_UNSUPPORTED
    assert any("no symbol for IPC" in note for note in result.warnings)


# ---------------------------------------------------------------------------
# Volume policy — the reason indices work
# ---------------------------------------------------------------------------


def test_a_volume_free_universe_loads_by_default_and_says_so():
    class IndexProvider(StubProvider):
        def fetch_one(self, identifier, request):
            return _panel_for(identifier, volume=False, kind=F.InstrumentKind.INDEX)

    result = ingest(
        _request(("SP500", "IPC"), fields=(F.CLOSE, F.VOLUME), provider="stub"),
        provider=IndexProvider(volume=True),
    )
    assert result.is_complete
    assert result.volumes is None
    assert any("carry no volume by construction" in n for n in result.warnings)


def test_require_volume_fails_a_run_that_silently_lost_it():
    class NoVolumeProvider(StubProvider):
        def fetch_one(self, identifier, request):
            return _panel_for(identifier, volume=False, kind=F.InstrumentKind.EQUITY)

    with pytest.raises(ProviderConfigurationError, match="require_volume is set"):
        ingest(
            _request(("AAA",), fields=(F.CLOSE, F.VOLUME), require_volume=True),
            provider=NoVolumeProvider(volume=True),
        )


def test_require_volume_is_satisfied_when_volume_arrives():
    result = ingest(
        _request(("AAA",), fields=(F.CLOSE, F.VOLUME), require_volume=True),
        provider=StubProvider(volume=True),
    )
    assert result.volumes is not None


def test_missing_volume_on_a_tradeable_name_warns_without_failing():
    class MixedProvider(StubProvider):
        def fetch_one(self, identifier, request):
            return _panel_for(
                identifier, volume=identifier == "AAA", kind=F.InstrumentKind.EQUITY
            )

    result = ingest(
        _request(("AAA", "BBB"), fields=(F.CLOSE, F.VOLUME)), provider=MixedProvider(volume=True)
    )
    assert result.is_complete
    assert any("No volume for BBB" in note for note in result.warnings)


def test_volume_is_not_mentioned_when_it_was_never_requested():
    result = ingest(_request(("AAA",), fields=F.PRICE_ONLY), provider=StubProvider())
    assert not any("volume" in note.lower() for note in result.warnings)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_a_cached_panel_is_returned_without_a_second_fetch(tmp_path):
    stub = StubProvider()
    request = _request(cache_dir=str(tmp_path))

    first = ingest(request, provider=stub)
    calls_after_first = len(stub.calls)
    second = ingest(request, provider=stub)

    assert not first.from_cache
    assert second.from_cache
    assert len(stub.calls) == calls_after_first
    pd.testing.assert_frame_equal(first.prices, second.prices)
    assert second.is_complete


def test_changing_the_request_misses_the_cache(tmp_path):
    stub = StubProvider()
    ingest(_request(("AAA",), cache_dir=str(tmp_path)), provider=stub)
    result = ingest(_request(("AAA", "BBB"), cache_dir=str(tmp_path)), provider=stub)
    assert not result.from_cache


def _age_entry(cache: PanelCache, key: str, seconds: float) -> None:
    """Rewrite an entry's manifest so it looks ``seconds`` old."""
    import io
    import json
    import zipfile

    path = cache.path_for(key)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}

    manifest = json.loads(members["manifest.json"].decode("utf-8"))
    manifest["written_at"] = manifest["written_at"] - seconds
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    path.write_bytes(buffer.getvalue())


def test_an_entry_past_its_ttl_is_a_miss(tmp_path):
    stub = StubProvider()
    request = _request(cache_dir=str(tmp_path), cache_ttl_seconds=3600)
    ingest(request, provider=stub)

    cache = PanelCache(tmp_path, ttl_seconds=3600)
    key = request.fingerprint()
    assert cache.load(key) is not None

    _age_entry(cache, key, seconds=7200)
    assert cache.load(key) is None


def test_a_zero_ttl_disables_expiry_rather_than_expiring_instantly(tmp_path):
    stub = StubProvider()
    request = _request(cache_dir=str(tmp_path), cache_ttl_seconds=0)
    ingest(request, provider=stub)

    cache = PanelCache(tmp_path, ttl_seconds=0)
    _age_entry(cache, request.fingerprint(), seconds=10 * 365 * 24 * 3600)
    assert cache.load(request.fingerprint()) is not None


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path):
    stub = StubProvider()
    request = _request(cache_dir=str(tmp_path))
    ingest(request, provider=stub)

    PanelCache(tmp_path).path_for(request.fingerprint()).write_bytes(b"not a zip")

    result = ingest(request, provider=stub)
    assert not result.from_cache
    assert result.is_complete


def test_a_truncated_entry_is_a_miss_not_a_crash(tmp_path):
    # What an interrupted write would have produced, if it could reach the
    # entry's own name — which is exactly what the temporary name prevents.
    stub = StubProvider()
    request = _request(cache_dir=str(tmp_path))
    ingest(request, provider=stub)

    path = PanelCache(tmp_path).path_for(request.fingerprint())
    payload = path.read_bytes()
    path.write_bytes(payload[: len(payload) // 2])

    assert PanelCache(tmp_path).load(request.fingerprint()) is None
    assert ingest(request, provider=stub).is_complete


def test_no_cache_directory_means_nothing_is_written(tmp_path):
    ingest(_request(), provider=StubProvider())
    assert list(tmp_path.iterdir()) == []


def test_cache_round_trips_provenance(tmp_path):
    cache = PanelCache(tmp_path)
    panel = _panel_for("AAA", volume=True, kind=F.InstrumentKind.INDEX)
    assert cache.store("k1", panel)

    loaded, entry = cache.load("k1")
    assert loaded.meta["AAA"].kind is F.InstrumentKind.INDEX
    assert loaded.meta["AAA"].currency == "USD"
    assert entry.key == "k1"
    pd.testing.assert_frame_equal(loaded.prices(), panel.prices())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_summary_and_report_describe_the_run():
    stub = StubProvider(failures={"BBB": IdentifierNotFoundError("gone")})
    result = ingest(_request(("AAA", "BBB")), provider=stub)

    assert "1/2 identifiers" in result.summary()
    report = result.report()
    assert list(report.index) == ["AAA", "BBB"]
    assert report.loc["BBB", "status"] == STATUS_FAILED


# ---------------------------------------------------------------------------
# The volume policy has to survive the cache
# ---------------------------------------------------------------------------


class _NoVolumeProvider(StubProvider):
    def fetch_one(self, identifier, request):
        self.calls.append((identifier,))
        return _panel_for(identifier, volume=False, kind=F.InstrumentKind.EQUITY)


def test_require_volume_is_enforced_on_a_cache_hit_too(tmp_path):
    # The fingerprint covers what was fetched, not what the caller will
    # accept, so a permissive run and a strict one share an entry. Without
    # re-applying the policy, running the lax one first would make the strict
    # one pass.
    lax = _request(("AAA",), fields=(F.CLOSE, F.VOLUME), cache_dir=str(tmp_path))
    ingest(lax, provider=_NoVolumeProvider(volume=True))

    strict = _request(
        ("AAA",), fields=(F.CLOSE, F.VOLUME), cache_dir=str(tmp_path),
        require_volume=True,
    )
    assert strict.fingerprint() == lax.fingerprint()
    with pytest.raises(ProviderConfigurationError, match="require_volume is set"):
        ingest(strict, provider=_NoVolumeProvider(volume=True))


def test_a_cache_hit_still_reports_its_volume_warnings(tmp_path):
    request = _request(("AAA",), fields=(F.CLOSE, F.VOLUME), cache_dir=str(tmp_path))
    ingest(request, provider=_NoVolumeProvider(volume=True))
    second = ingest(request, provider=_NoVolumeProvider(volume=True))

    assert second.from_cache
    assert any("No volume for AAA" in note for note in second.warnings)


# ---------------------------------------------------------------------------
# Two identifiers naming one instrument
# ---------------------------------------------------------------------------


def test_two_identifiers_resolving_to_one_symbol_are_named_as_such():
    # SP500 and ^GSPC are the same Yahoo series. Reporting the loser as "the
    # provider returned nothing" describes something that did not happen.
    class YahooLike(StubProvider):
        name = "yahoo"

        @property
        def capabilities(self):
            return ProviderCapabilities(
                fields=frozenset({F.CLOSE}), intervals=frozenset({"1d"})
            )

    result = ingest(
        _request(("SP500", "^GSPC"), provider="yahoo"), provider=YahooLike()
    )
    statuses = {o.identifier: o for o in result.outcomes}
    assert statuses["SP500"].status == STATUS_OK
    assert statuses["^GSPC"].status == STATUS_UNSUPPORTED
    assert "already claims" in statuses["^GSPC"].message
    assert any("same instrument twice" in note for note in result.warnings)


def test_distinct_symbols_are_not_treated_as_duplicates():
    result = ingest(_request(("AAA", "BBB", "CCC")), provider=StubProvider())
    assert result.is_complete


# ---------------------------------------------------------------------------
# Currency conversion must not claim what it did not do
# ---------------------------------------------------------------------------


def _panel_with_currency(identifier: str, currency: str | None):
    index = pd.bdate_range("2024-01-01", periods=8)
    close = pd.DataFrame({identifier: np.linspace(100.0, 108.0, 8)}, index=index)
    return PricePanel.from_frames(
        {F.CLOSE: close},
        {identifier: SeriesMeta(identifier, identifier, "stub", currency=currency)},
    )


def test_a_series_with_no_declared_currency_is_not_stamped_as_converted():
    from optimization_engine.ingest.service import _convert_currency

    panel = _panel_with_currency("AAA", None)
    converted, note, degraded = _convert_currency(panel, "USD")
    assert not degraded
    # It was assumed to be in USD, not converted to it. Labelling it USD turns
    # an assumption into a claim.
    assert converted.meta["AAA"].currency is None
    assert "do not say what currency" in note


def test_a_series_with_a_declared_matching_currency_needs_no_note():
    from optimization_engine.ingest.service import _convert_currency

    converted, note, degraded = _convert_currency(_panel_with_currency("AAA", "USD"), "USD")
    assert not degraded
    assert note == ""
    assert converted.meta["AAA"].currency == "USD"


def test_a_cache_hit_reports_a_symbol_collision_the_same_way_a_cold_run_does(tmp_path):
    # The warm path used to return before the collision was even detected, so
    # a re-run described the loser as "the provider returned no series" —
    # exactly the misreport the check exists to prevent.
    class YahooLike(StubProvider):
        name = "yahoo"

        @property
        def capabilities(self):
            return ProviderCapabilities(
                fields=frozenset({F.CLOSE}), intervals=frozenset({"1d"})
            )

    request = _request(("SP500", "^GSPC"), provider="yahoo", cache_dir=str(tmp_path))
    cold = ingest(request, provider=YahooLike())
    warm = ingest(request, provider=YahooLike())

    assert warm.from_cache
    cold_gspc = next(o for o in cold.outcomes if o.identifier == "^GSPC")
    warm_gspc = next(o for o in warm.outcomes if o.identifier == "^GSPC")
    assert warm_gspc.status == cold_gspc.status == STATUS_UNSUPPORTED
    assert "already claims" in warm_gspc.message
    assert any("same instrument twice" in note for note in warm.warnings)


def test_a_cache_hit_reports_an_unsupported_instrument_too(tmp_path):
    class NoIndexProvider(StubProvider):
        name = "fred"

        @property
        def capabilities(self):
            return ProviderCapabilities(
                fields=frozenset({F.CLOSE}), intervals=frozenset({"1d"})
            )

    request = _request(("SP500", "IPC"), provider="fred", cache_dir=str(tmp_path))
    ingest(request, provider=NoIndexProvider())
    warm = ingest(request, provider=NoIndexProvider())
    assert warm.from_cache
    assert any("no symbol for IPC" in note for note in warm.warnings)


def test_a_batched_provider_bug_does_not_relay_its_message_either():
    # `_fetch_single` got the catch-all first; `_fetch_batched` needed the
    # same rule, and reaches the browser through the same render path.
    class LeakyBatch(StubProvider):
        name = "leaky_batch"

        @property
        def capabilities(self):
            return ProviderCapabilities(
                fields=frozenset({F.CLOSE}), intervals=frozenset({"1d"}),
                supports_batch=True, max_batch_size=4, accepts_any_symbol=True,
            )

        def fetch_batch(self, identifiers, request):
            raise ValueError("boom carrying sk-live-SECRET")

    with pytest.raises(ProviderConfigurationError) as caught:
        ingest(_request(("AAA", "BBB"), provider="leaky_batch"), provider=LeakyBatch())
    assert "sk-live-SECRET" not in str(caught.value)
    assert "ValueError" in str(caught.value)


# ---------------------------------------------------------------------------
# Publishing a cache entry has to be atomic
# ---------------------------------------------------------------------------


def test_concurrent_writers_of_one_key_all_succeed(tmp_path):
    """The reason an entry is a single file.

    With a directory per entry, publishing means removing the old one and then
    renaming the new one into place. Two writers racing through that gap leave
    one of them with ``Directory not empty`` — a lost cache write on every
    collision — and readers with a window in which the entry does not exist.
    ``os.replace`` on a file has neither problem.
    """
    import concurrent.futures

    cache = PanelCache(tmp_path)
    panel = _panel_for("AAA", volume=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: cache.store("samekey", panel), range(16)))

    assert all(results), f"{results.count(False)} of 16 concurrent writes failed"

    loaded = cache.load("samekey")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded[0].prices(), panel.prices())


def test_a_concurrent_writer_never_hides_the_entry_from_a_reader(tmp_path):
    """A reader must see the old entry or the new one, never neither."""
    import threading

    cache = PanelCache(tmp_path)
    panel = _panel_for("AAA", volume=True)
    cache.store("samekey", panel)

    stop = threading.Event()
    misses: list[int] = []

    def read_until_stopped() -> None:
        while not stop.is_set():
            if cache.load("samekey") is None:
                misses.append(1)

    reader = threading.Thread(target=read_until_stopped)
    reader.start()
    try:
        for _ in range(40):
            cache.store("samekey", panel)
    finally:
        stop.set()
        reader.join(timeout=10)

    assert not misses, f"the entry vanished {len(misses)} times mid-republish"


def test_an_interrupted_write_leaves_nothing_a_later_run_would_trust(
    tmp_path, monkeypatch
):
    cache = PanelCache(tmp_path)

    def explode(stream, panel, *_):
        # Fail partway through serialization, once the temporary file exists.
        stream.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(PanelCache, "_write_archive", staticmethod(explode))
    assert cache.store("samekey", _panel_for("AAA", volume=True)) is False

    monkeypatch.undo()
    assert cache.load("samekey") is None
    assert not cache.path_for("samekey").exists()
    # And no temporary file was left lying around.
    assert list(tmp_path.iterdir()) == []


def test_a_refused_publish_is_retried_rather_than_reported_as_a_failure(
    tmp_path, monkeypatch
):
    """Windows refuses ``os.replace`` while anyone else holds the target open.

    ``PermissionError`` (``WinError 5``) there means "come back in a moment",
    not "this write failed": a reader mid-``load`` is enough to trigger it,
    which is why the two concurrency tests in this section lose a handful of
    their writers on Windows. POSIX renames straight over an open file, so the
    refusal has to be simulated — there is no way to provoke it on the
    platform CI runs on.
    """
    from optimization_engine.ingest import cache as cache_module

    cache = PanelCache(tmp_path)
    panel = _panel_for("AAA", volume=True)

    real_replace = cache_module.os.replace
    attempts = []

    def refuse_the_first_two(src, dst):
        attempts.append(1)
        if len(attempts) <= 2:
            raise PermissionError(5, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(cache_module.os, "replace", refuse_the_first_two)
    assert cache.store("samekey", panel) is True
    monkeypatch.undo()

    assert len(attempts) == 3, "the third attempt should have been the one that landed"
    loaded = cache.load("samekey")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded[0].prices(), panel.prices())
    # The staging file was renamed away, not left behind next to the entry.
    assert list(tmp_path.iterdir()) == [cache.path_for("samekey")]


def test_a_publish_lost_to_another_writer_of_the_same_key_is_still_a_success(
    tmp_path, monkeypatch
):
    """Losing the race to a writer of the *same* key is a hit, not a failure.

    The key is the request fingerprint, which covers everything that affects
    the data, so the entry the other writer published is the entry this call
    would have written. What the caller asked for is an entry under this key,
    and there is one.
    """
    from optimization_engine.ingest import cache as cache_module

    cache = PanelCache(tmp_path)
    panel = _panel_for("AAA", volume=True)
    rival = PanelCache(tmp_path)
    real_replace = cache_module.os.replace
    attempts = []
    published = {}

    def always_refuse(src, dst):
        attempts.append(1)
        if len(attempts) == 1:
            # The writer that wins the race publishes while this call backs
            # off. An entry already there *before* the call would not count:
            # see the held-open test below.
            monkeypatch.setattr(cache_module.os, "replace", real_replace)
            assert rival.store("samekey", panel) is True
            published["bytes"] = cache.path_for("samekey").read_bytes()
            monkeypatch.setattr(cache_module.os, "replace", always_refuse)
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(cache_module.os, "replace", always_refuse)
    assert cache.store("samekey", panel) is True
    monkeypatch.undo()

    assert len(attempts) == 5, "five attempts before conceding the race"
    # The entry the rival published is untouched — no half-published mix of
    # the two — and the staging file this call built is gone.
    assert cache.path_for("samekey").read_bytes() == published["bytes"]
    assert list(tmp_path.iterdir()) == [cache.path_for("samekey")]
    assert cache.load("samekey") is not None


def test_a_publish_that_is_never_permitted_and_lands_nothing_reports_failure(
    tmp_path, monkeypatch
):
    """No entry afterwards is a failed write, retries or not — and no leftovers."""
    from optimization_engine.ingest import cache as cache_module

    def always_refuse(src, dst):
        raise PermissionError(5, "Access is denied")

    cache = PanelCache(tmp_path)
    monkeypatch.setattr(cache_module.os, "replace", always_refuse)
    assert cache.store("samekey", _panel_for("AAA", volume=True)) is False
    monkeypatch.undo()

    assert cache.load("samekey") is None
    assert not cache.path_for("samekey").exists()
    assert list(tmp_path.iterdir()) == []


def test_clear_removes_entries_and_the_directories_version_one_wrote(tmp_path):
    cache = PanelCache(tmp_path)
    cache.store("k1", _panel_for("AAA", volume=True))
    cache.store("k2", _panel_for("BBB", volume=True))
    # A version-1 entry, which is now a miss but still takes up space.
    (tmp_path / "legacy_key").mkdir()
    (tmp_path / "legacy_key" / "manifest.json").write_text("{}")

    assert cache.clear() == 3
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# What the cache refuses to keep (review fixes I1, I2, I3)
# ---------------------------------------------------------------------------


def test_a_partially_failed_run_is_not_cached_and_is_retried(tmp_path):
    stub = StubProvider(failures={"BBB": ProviderTransientError("429")})
    request = _request(cache_dir=str(tmp_path))
    first = ingest(request, provider=stub)
    assert not first.is_complete
    assert any(note.startswith("Not cached:") for note in first.warnings)

    stub.failures.clear()  # the provider recovers
    second = ingest(request, provider=stub)
    # Before the fix this came back from the cache with BBB relabelled
    # "missing" for the length of the TTL, and the provider was never asked.
    assert not second.from_cache
    assert second.is_complete
    third = ingest(request, provider=stub)
    assert third.from_cache and third.is_complete


def test_a_degraded_currency_conversion_is_not_cached(tmp_path, monkeypatch):
    from optimization_engine.data.fx import FXError
    from optimization_engine.ingest import service

    class Foreign(StubProvider):
        def fetch_one(self, identifier, request):
            self.calls.append((identifier,))
            return _panel_with_currency(identifier, "MXN")

    attempts = []

    def flaky(frame, asset_currency, base, **_):
        attempts.append(base)
        if len(attempts) == 1:
            raise FXError("FRED down")
        return frame * 0.05

    monkeypatch.setattr(service, "convert_prices_to_base", flaky)
    request = _request(("AAA",), currency="USD", cache_dir=str(tmp_path))
    degraded = ingest(request, provider=Foreign())
    assert any("Could not convert" in note for note in degraded.warnings)
    assert any(note.startswith("Not cached:") for note in degraded.warnings)
    assert degraded.panel.meta["AAA"].currency == "MXN"

    healthy = ingest(request, provider=Foreign())
    assert not healthy.from_cache
    assert healthy.panel.meta["AAA"].currency == "USD"


def test_provider_options_are_part_of_the_cache_key(tmp_path):
    stub = StubProvider()
    request = _request(("AAA",), cache_dir=str(tmp_path))
    first = ingest(request, provider=stub, path="a.csv")
    other = ingest(request, provider=stub, path="b.csv")
    same = ingest(request, provider=stub, path="a.csv")
    assert not first.from_cache
    assert not other.from_cache  # a different file is different data
    assert same.from_cache


# ---------------------------------------------------------------------------
# What a warm run has to say, and when an entry is stale (review I2, I3 residue)
# ---------------------------------------------------------------------------


def test_a_cache_hit_repeats_the_currency_warning_the_cold_run_gave(tmp_path):
    """The panel is the same, so what was wrong with it still is."""

    class Undeclared(StubProvider):
        def fetch_one(self, identifier, request):
            self.calls.append((identifier,))
            return _panel_with_currency(identifier, None)

    request = _request(("AAA",), currency="USD", cache_dir=str(tmp_path))
    cold = ingest(request, provider=Undeclared())
    warm = ingest(request, provider=Undeclared())

    note = "do not say what currency"
    assert any(note in w for w in cold.warnings)
    assert warm.from_cache
    assert any(note in w for w in warm.warnings), warm.warnings


def test_an_edited_file_is_read_again_rather_than_served_from_cache(tmp_path):
    """The key named the file; it has to name what the file held, too."""
    import os

    index = pd.bdate_range("2024-01-01", periods=20)
    path = tmp_path / "panel.csv"
    pd.DataFrame({"AAA": np.linspace(100.0, 120.0, 20)}, index=index).to_csv(
        path, index_label="date"
    )
    request = IngestRequest(
        identifiers=("AAA",), provider="file", start="2024-01-01",
        end="2024-02-28", cache_dir=str(tmp_path / "cache"),
    )
    first = ingest(request, path=str(path))
    assert ingest(request, path=str(path)).from_cache

    pd.DataFrame({"AAA": np.linspace(300.0, 360.0, 20)}, index=index).to_csv(
        path, index_label="date"
    )
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    edited = ingest(request, path=str(path))

    assert not edited.from_cache
    assert float(edited.prices["AAA"].iloc[0]) == pytest.approx(300.0)
    assert float(first.prices["AAA"].iloc[0]) == pytest.approx(100.0)


def test_a_timezone_aware_panel_round_trips_through_the_cache(tmp_path):
    """``store`` said True and every ``load`` missed, logged at debug only."""
    index = pd.bdate_range("2024-01-01", periods=8).tz_localize("America/New_York")
    panel = PricePanel.from_frames(
        {F.CLOSE: pd.DataFrame({"AAA": np.linspace(10.0, 17.0, 8)}, index=index)}
    )
    cache = PanelCache(tmp_path)

    assert cache.store("tz", panel) is True
    loaded = cache.load("tz")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded[0].prices(), panel.prices())


def test_a_publish_refused_while_an_older_entry_is_held_open_reports_failure(
    tmp_path, monkeypatch
):
    """An entry that predates the call is not this call's result.

    Windows refuses ``os.replace`` while a reader holds the target open. The
    entry left in place is then the *old* one, and reporting the write as a
    success told the caller its panel was cached when it was not.
    """
    from optimization_engine.ingest import cache as cache_module

    cache = PanelCache(tmp_path)
    old = _panel_for("AAA", volume=False)
    assert cache.store("samekey", old) is True
    held = cache.path_for("samekey").read_bytes()

    def refuse(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(cache_module.os, "replace", refuse)
    assert cache.store("samekey", _panel_for("BBB", volume=False)) is False
    monkeypatch.undo()

    assert cache.path_for("samekey").read_bytes() == held
    assert list(tmp_path.iterdir()) == [cache.path_for("samekey")]


def test_a_run_whose_panel_could_not_be_cached_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(PanelCache, "store", lambda self, *a, **k: False)
    result = ingest(_request(("AAA",), cache_dir=str(tmp_path)), provider=StubProvider())
    assert any(note.startswith("Not cached:") for note in result.warnings)
