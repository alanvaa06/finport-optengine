"""Tests for the provider adapters, against recorded payloads.

Every provider is exercised offline: HTTP is replaced at the base class's
``_get_text`` / ``_get_json`` seam, and yfinance is replaced with a fake
module. What is being tested is the translation — vendor schema in,
homogenized vocabulary out — plus the two behaviours that are easy to get
wrong and invisible when wrong: which column is the total-return close, and
what happens to volume that does not exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.ingest import fields as F  # noqa: E402
from optimization_engine.ingest.errors import (  # noqa: E402
    IdentifierNotFoundError,
    ProviderConfigurationError,
    ProviderCredentialsError,
    ProviderResponseError,
)
from optimization_engine.ingest.providers.file import LocalFile  # noqa: E402
from optimization_engine.ingest.providers.fmp import FinancialModelingPrep  # noqa: E402
from optimization_engine.ingest.providers.fred import Fred, is_rate_series  # noqa: E402
from optimization_engine.ingest.providers.sample import Sample  # noqa: E402
from optimization_engine.ingest.providers.stooq import Stooq, classify  # noqa: E402
from optimization_engine.ingest.providers.tiingo import Tiingo  # noqa: E402
from optimization_engine.ingest.providers.yahoo import Yahoo  # noqa: E402
from optimization_engine.ingest.spec import IngestRequest  # noqa: E402


def _request(identifiers=("AAA",), **kwargs) -> IngestRequest:
    kwargs.setdefault("start", "2024-01-01")
    kwargs.setdefault("end", "2024-01-10")
    return IngestRequest(identifiers=list(identifiers), **kwargs)


# ---------------------------------------------------------------------------
# Preflight — rejecting the impossible before the network
# ---------------------------------------------------------------------------


def test_preflight_rejects_a_field_the_provider_cannot_serve():
    provider = Fred()
    with pytest.raises(ProviderConfigurationError, match="cannot serve"):
        provider.preflight(_request(("SP500",), provider="fred", fields=F.OHLCV))


def test_preflight_rejects_an_interval_the_provider_does_not_publish():
    provider = FinancialModelingPrep(api_key="k")
    with pytest.raises(ProviderConfigurationError, match="1wk"):
        provider.preflight(_request(provider="fmp", interval="1wk"))


def test_preflight_demands_a_key_and_names_the_variable():
    provider = FinancialModelingPrep(api_key=None)
    with pytest.raises(ProviderCredentialsError, match="OPTENGINE_API_KEY_FMP"):
        provider.preflight(_request(provider="fmp"))


def test_preflight_passes_for_a_keyless_provider():
    Stooq().preflight(_request(("^SPX",), provider="stooq", fields=F.OHLC))


# ---------------------------------------------------------------------------
# Stooq
# ---------------------------------------------------------------------------

_STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,100.0,101.0,99.5,100.5,1200
2024-01-03,100.5,102.0,100.0,101.8,1500
2024-01-04,101.8,102.5,101.0,102.2,900
"""

_STOOQ_INDEX_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,4700.0,4720.0,4690.0,4710.0,0
2024-01-03,4710.0,4750.0,4705.0,4745.0,0
"""


def test_stooq_parses_ohlcv(monkeypatch):
    provider = Stooq()
    monkeypatch.setattr(provider, "_get_text", lambda *a, **k: _STOOQ_CSV)
    panel = provider.fetch_one("AAPL.US", _request(("AAPL.US",), fields=F.OHLCV))

    assert panel.identifiers == ("AAPL.US",)
    assert set(panel.available_fields) == {F.OPEN, F.HIGH, F.LOW, F.CLOSE, F.VOLUME}
    assert float(panel.prices().iloc[1, 0]) == pytest.approx(101.8)
    assert panel.has_volume


def test_stooq_drops_the_zero_volume_column_an_index_returns(monkeypatch):
    provider = Stooq()
    monkeypatch.setattr(provider, "_get_text", lambda *a, **k: _STOOQ_INDEX_CSV)
    panel = provider.fetch_one("^SPX", _request(("^SPX",), fields=F.OHLCV))

    # Zero is not "no liquidity" — it is "this instrument has no volume".
    assert F.VOLUME not in panel.available_fields
    assert not panel.has_volume
    assert panel.meta["^SPX"].kind is F.InstrumentKind.INDEX


def test_stooq_reports_an_unknown_symbol_as_not_found(monkeypatch):
    provider = Stooq()
    monkeypatch.setattr(provider, "_get_text", lambda *a, **k: "No data")
    with pytest.raises(IdentifierNotFoundError):
        provider.fetch_one("NOPE", _request(("NOPE",)))


def test_stooq_surfaces_its_rate_limit_distinctly(monkeypatch):
    provider = Stooq()
    monkeypatch.setattr(provider, "_get_text", lambda *a, **k: "Exceeded the daily hits limit")
    with pytest.raises(ProviderResponseError, match="daily limit"):
        provider.fetch_one("AAPL.US", _request(("AAPL.US",)))


def test_stooq_symbol_classification():
    assert classify("^spx") is F.InstrumentKind.INDEX
    assert classify("EURUSD") is F.InstrumentKind.FX
    assert classify("AAPL.US") is F.InstrumentKind.EQUITY


# ---------------------------------------------------------------------------
# Financial Modeling Prep
# ---------------------------------------------------------------------------

_FMP_PAYLOAD = {
    "symbol": "AAPL",
    "currency": "USD",
    "historical": [
        {"date": "2024-01-04", "open": 182.0, "high": 183.1, "low": 180.9,
         "close": 181.9, "adjClose": 181.5, "volume": 50_000_000, "vwap": 181.8},
        {"date": "2024-01-03", "open": 184.2, "high": 185.9, "low": 183.4,
         "close": 184.3, "adjClose": 183.9, "volume": 58_000_000, "vwap": 184.5},
    ],
}


def test_fmp_maps_adjclose_to_the_total_return_close(monkeypatch):
    provider = FinancialModelingPrep(api_key="k")
    monkeypatch.setattr(provider, "_get_json", lambda *a, **k: _FMP_PAYLOAD)
    panel = provider.fetch_one("AAPL", _request(("AAPL",), fields=F.OHLCV))

    # adjClose is the total-return series; close is the raw print. Swapping
    # them silently drops every dividend from the backtest.
    assert float(panel.prices().loc["2024-01-03", "AAPL"]) == pytest.approx(183.9)
    assert float(panel.frame(F.CLOSE_RAW).loc["2024-01-03", "AAPL"]) == pytest.approx(184.3)
    assert panel.meta["AAPL"].currency == "USD"


def test_fmp_sorts_its_descending_rows_ascending(monkeypatch):
    provider = FinancialModelingPrep(api_key="k")
    monkeypatch.setattr(provider, "_get_json", lambda *a, **k: _FMP_PAYLOAD)
    panel = provider.fetch_one("AAPL", _request(("AAPL",)))
    assert panel.index.is_monotonic_increasing


def test_fmp_accepts_the_newer_bare_list_payload(monkeypatch):
    provider = FinancialModelingPrep(api_key="k")
    monkeypatch.setattr(
        provider, "_get_json", lambda *a, **k: _FMP_PAYLOAD["historical"]
    )
    panel = provider.fetch_one("AAPL", _request(("AAPL",)))
    assert panel.identifiers == ("AAPL",)


def test_fmp_reports_an_empty_history_as_not_found(monkeypatch):
    provider = FinancialModelingPrep(api_key="k")
    monkeypatch.setattr(provider, "_get_json", lambda *a, **k: {"historical": []})
    with pytest.raises(IdentifierNotFoundError):
        provider.fetch_one("NOPE", _request(("NOPE",)))


def test_fmp_relays_a_business_error_message(monkeypatch):
    provider = FinancialModelingPrep(api_key="k")
    monkeypatch.setattr(
        provider, "_get_json", lambda *a, **k: {"Error Message": "Limit reached"}
    )
    with pytest.raises(ProviderResponseError, match="Limit reached"):
        provider.fetch_one("AAPL", _request(("AAPL",)))


# ---------------------------------------------------------------------------
# Tiingo
# ---------------------------------------------------------------------------

_TIINGO_PAYLOAD = [
    {"date": "2024-01-03T00:00:00.000Z", "close": 184.3, "adjClose": 183.9,
     "adjOpen": 184.0, "adjHigh": 185.5, "adjLow": 183.0, "adjVolume": 58_000_000},
    {"date": "2024-01-04T00:00:00.000Z", "close": 181.9, "adjClose": 181.5,
     "adjOpen": 181.7, "adjHigh": 182.8, "adjLow": 180.6, "adjVolume": 50_000_000},
]


def test_tiingo_strips_the_timezone_from_its_index(monkeypatch):
    provider = Tiingo(api_key="k")
    monkeypatch.setattr(provider, "_get_json", lambda *a, **k: _TIINGO_PAYLOAD)
    panel = provider.fetch_one("AAPL", _request(("AAPL",), fields=F.OHLCV))

    # A tz-aware index cannot be compared with any other provider's dates.
    assert panel.index.tz is None
    assert list(panel.index) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]


def test_tiingo_sends_its_token_in_a_header_not_the_url(monkeypatch):
    provider = Tiingo(api_key="secret-token")
    captured: dict = {}

    def fake_get_json(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _TIINGO_PAYLOAD

    monkeypatch.setattr(provider, "_get_json", fake_get_json)
    provider.fetch_one("AAPL", _request(("AAPL",)))

    assert "secret-token" not in captured["url"]
    assert "secret-token" not in json.dumps(captured.get("params", {}))
    assert captured["headers"]["Authorization"] == "Token secret-token"


def test_tiingo_relays_a_detail_error(monkeypatch):
    provider = Tiingo(api_key="k")
    monkeypatch.setattr(
        provider, "_get_json", lambda *a, **k: {"detail": "Not found"}
    )
    with pytest.raises(ProviderResponseError, match="Not found"):
        provider.fetch_one("NOPE", _request(("NOPE",)))


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------


def test_fred_refuses_rate_series_as_assets(monkeypatch):
    provider = Fred()
    with pytest.raises(ProviderConfigurationError, match="interest rates"):
        provider.fetch_batch(("DGS10",), _request(("DGS10",), provider="fred"))


def test_rate_detection_does_not_sweep_in_index_series():
    assert is_rate_series("DGS10")
    assert is_rate_series("DFF")
    assert is_rate_series("EFFR")
    assert not is_rate_series("SP500")
    assert not is_rate_series("NASDAQCOM")
    assert not is_rate_series("VIXCLS")


def test_fred_returns_columns_under_the_requested_name(monkeypatch):
    from optimization_engine.ingest.providers import fred as fred_module

    index = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame({"SP500": np.linspace(4700, 4750, 5)}, index=index)
    monkeypatch.setattr(fred_module, "load_fred_series", lambda *a, **k: frame)

    panel = Fred().fetch_batch(("sp500",), _request(("sp500",), provider="fred"))
    assert panel.identifiers == ("sp500",)
    assert panel.meta["sp500"].provider_symbol == "SP500"
    assert panel.meta["sp500"].kind is F.InstrumentKind.INDEX


def test_fred_never_advertises_volume():
    assert not Fred().capabilities.serves_volume
    assert F.VOLUME not in Fred().capabilities.fields


# ---------------------------------------------------------------------------
# Yahoo
# ---------------------------------------------------------------------------


class _FakeYFinance:
    """A stand-in for ``yfinance`` returning the grouped-column layout."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.calls: list[dict] = []

    def download(self, **kwargs):
        self.calls.append(kwargs)
        return self._frame


def _yahoo_frame(symbols, *, volume) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=4)
    blocks = {}
    for position, symbol in enumerate(symbols):
        base = 100.0 + 10 * position
        blocks[("Open", symbol)] = np.linspace(base, base + 3, 4)
        blocks[("High", symbol)] = np.linspace(base + 1, base + 4, 4)
        blocks[("Low", symbol)] = np.linspace(base - 1, base + 2, 4)
        blocks[("Close", symbol)] = np.linspace(base + 0.5, base + 3.5, 4)
        blocks[("Volume", symbol)] = np.full(4, volume[symbol], dtype=float)
    frame = pd.DataFrame(blocks, index=index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def test_yahoo_batches_the_whole_universe_in_one_download(monkeypatch):
    from optimization_engine.data import yahoo as yahoo_module

    fake = _FakeYFinance(_yahoo_frame(["SPY", "AGG"], volume={"SPY": 1e7, "AGG": 5e6}))
    monkeypatch.setattr(yahoo_module, "_import_yfinance", lambda: fake)

    panel = Yahoo().fetch_batch(("SPY", "AGG"), _request(("SPY", "AGG"), fields=F.OHLCV))
    assert len(fake.calls) == 1
    assert set(panel.identifiers) == {"SPY", "AGG"}
    assert panel.has_volume


def test_yahoo_blanks_the_zero_volume_yahoo_reports_for_an_index(monkeypatch):
    from optimization_engine.data import yahoo as yahoo_module

    fake = _FakeYFinance(
        _yahoo_frame(["SPY", "^GSPC"], volume={"SPY": 1e7, "^GSPC": 0.0})
    )
    monkeypatch.setattr(yahoo_module, "_import_yfinance", lambda: fake)

    panel = Yahoo().fetch_batch(
        ("SPY", "^GSPC"), _request(("SPY", "^GSPC"), fields=F.OHLCV)
    )
    volume = panel.volumes()
    assert volume is not None
    assert volume["SPY"].notna().all()
    # Zero volume on an index would read as "untradeable" to a cost model.
    assert volume["^GSPC"].dropna().empty


def test_yahoo_advances_the_exclusive_end_date_by_one_day(monkeypatch):
    from optimization_engine.data import yahoo as yahoo_module

    fake = _FakeYFinance(_yahoo_frame(["SPY"], volume={"SPY": 1e7}))
    monkeypatch.setattr(yahoo_module, "_import_yfinance", lambda: fake)

    Yahoo().fetch_batch(("SPY",), _request(("SPY",), start="2024-01-01", end="2024-01-10"))
    assert fake.calls[0]["end"].startswith("2024-01-11")


def test_yahoo_renames_upper_cased_columns_back_to_the_request(monkeypatch):
    from optimization_engine.data import yahoo as yahoo_module

    fake = _FakeYFinance(_yahoo_frame(["SPY"], volume={"SPY": 1e7}))
    monkeypatch.setattr(yahoo_module, "_import_yfinance", lambda: fake)

    panel = Yahoo().fetch_batch(("spy",), _request(("spy",)))
    assert panel.identifiers == ("spy",)
    assert panel.meta["spy"].provider_symbol == "SPY"


def test_yahoo_symbol_classification():
    from optimization_engine.ingest.providers.yahoo import classify as yahoo_classify

    assert yahoo_classify("^GSPC") is F.InstrumentKind.INDEX
    assert yahoo_classify("^TNX") is F.InstrumentKind.RATE
    assert yahoo_classify("EURUSD=X") is F.InstrumentKind.FX
    assert yahoo_classify("BTC-USD") is F.InstrumentKind.CRYPTO
    assert yahoo_classify("ES=F") is F.InstrumentKind.COMMODITY
    assert yahoo_classify("SPY") is F.InstrumentKind.UNKNOWN


# ---------------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------------


def test_file_reads_a_wide_panel(tmp_path):
    index = pd.bdate_range("2024-01-01", periods=6)
    frame = pd.DataFrame(
        {"AAA": np.linspace(100, 105, 6), "BBB": np.linspace(50, 55, 6)}, index=index
    )
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index_label="date")

    provider = LocalFile(path=path)
    panel = provider.fetch_batch(
        ("AAA", "BBB"), _request(("AAA", "BBB"), start="2024-01-01", end="2024-01-31")
    )
    assert panel.identifiers == ("AAA", "BBB")
    assert len(panel.index) == 6


def test_file_reads_a_long_panel_with_volume(tmp_path):
    rows = []
    for day in pd.bdate_range("2024-01-01", periods=4):
        for name, base in (("AAA", 100.0), ("BBB", 50.0)):
            rows.append(
                {
                    "date": day, "identifier": name,
                    "open": base, "high": base + 1, "low": base - 1,
                    "close": base + 0.5, "volume": 1_000_000,
                }
            )
    path = tmp_path / "long.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    panel = LocalFile(path=path).fetch_batch(
        ("AAA", "BBB"),
        _request(("AAA", "BBB"), start="2024-01-01", end="2024-01-31", fields=F.OHLCV),
    )
    assert panel.has_volume
    assert set(panel.available_fields) >= {F.OPEN, F.HIGH, F.LOW, F.CLOSE, F.VOLUME}


def test_file_matches_columns_case_insensitively_but_keeps_the_asked_name(tmp_path):
    index = pd.bdate_range("2024-01-01", periods=4)
    pd.DataFrame({"Cash": np.linspace(100, 101, 4)}, index=index).to_csv(
        tmp_path / "p.csv", index_label="date"
    )
    panel = LocalFile(path=tmp_path / "p.csv").fetch_batch(
        ("CASH",), _request(("CASH",), start="2024-01-01", end="2024-01-31")
    )
    assert panel.identifiers == ("CASH",)


def test_file_lists_what_it_does_contain_when_nothing_matches(tmp_path):
    index = pd.bdate_range("2024-01-01", periods=4)
    pd.DataFrame({"AAA": np.linspace(100, 101, 4)}, index=index).to_csv(
        tmp_path / "p.csv", index_label="date"
    )
    with pytest.raises(IdentifierNotFoundError, match="AAA"):
        LocalFile(path=tmp_path / "p.csv").fetch_batch(
            ("ZZZ",), _request(("ZZZ",), start="2024-01-01", end="2024-01-31")
        )


def test_a_file_does_not_claim_the_currency_it_was_asked_to_convert_into(tmp_path):
    """A file says nothing about its currency, so the panel must not either.

    Every series used to be tagged with the *target* currency, which made
    the conversion step see nothing to convert: a peso series asked for in
    dollars came back unconverted, labelled USD, with no warning.
    """
    from optimization_engine.ingest import ingest

    index = pd.bdate_range("2024-01-01", periods=6)
    pd.DataFrame({"WALMEX": np.linspace(60, 61, 6)}, index=index).to_csv(
        tmp_path / "mx.csv", index_label="date"
    )
    request = _request(("WALMEX",), provider="file", end="2024-01-31", currency="USD")

    result = ingest(request, path=str(tmp_path / "mx.csv"), use_cache=False)

    assert result.panel.meta["WALMEX"].currency is None
    assert any("do not say what currency" in note for note in result.warnings)


def test_file_without_a_path_says_so():
    with pytest.raises(ProviderConfigurationError, match="needs a path"):
        LocalFile().fetch_batch(("AAA",), _request(("AAA",)))


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def test_sample_is_deterministic():
    request = _request(("AAA", "BBB"), provider="sample", period="1y", start=None, end=None)
    first = Sample(seed=7).fetch_batch(("AAA", "BBB"), request).prices()
    second = Sample(seed=7).fetch_batch(("AAA", "BBB"), request).prices()
    pd.testing.assert_frame_equal(first, second)


def test_sample_emits_volume_for_tradeable_names_only():
    request = _request(
        ("AAA", "SP500"), provider="sample", period="1y", start=None, end=None,
        fields=F.OHLCV,
    )
    panel = Sample().fetch_batch(("AAA", "SP500"), request)
    volume = panel.volumes()
    assert volume is not None
    assert volume["AAA"].notna().any()
    assert volume["SP500"].dropna().empty


def test_sample_returns_no_volume_at_all_for_a_pure_index_universe():
    request = _request(
        ("SP500", "IPC"), provider="sample", period="1y", start=None, end=None,
        fields=F.OHLCV,
    )
    panel = Sample().fetch_batch(("SP500", "IPC"), request)
    assert panel.volumes() is None
    assert not panel.has_volume


def test_sample_reproduces_the_generator_series_unperturbed():
    from optimization_engine.data.loader import sample_dataset

    request = _request(
        ("US_Equity", "Cash"), provider="sample", period="1y", start=None, end=None
    )
    panel = Sample(seed=42).fetch_batch(("US_Equity", "Cash"), request)
    reference = sample_dataset(n_periods=len(panel.index), seed=42)

    # Within the generator's own universe the series must arrive untouched:
    # perturbing them would change cash from a 0.5% vol asset into something
    # else entirely.
    np.testing.assert_allclose(
        panel.prices()["Cash"].to_numpy(),
        reference["Cash"].to_numpy()[-len(panel.index):],
    )


def test_sample_keeps_recycled_columns_from_being_collinear():
    from optimization_engine.data.loader import sample_dataset

    width = sample_dataset(n_periods=10).shape[1]
    names = [f"A{i}" for i in range(width + 3)]
    request = _request(names, provider="sample", period="2y", start=None, end=None)
    prices = Sample().fetch_batch(tuple(names), request).prices()

    returns = prices.pct_change().dropna()
    # The recycled column must not be a perfect copy of the one it reuses.
    assert abs(float(returns.iloc[:, 0].corr(returns.iloc[:, width]))) < 0.999


# ---------------------------------------------------------------------------
# Adjustment coherence — the panel must not mix two price scales
# ---------------------------------------------------------------------------

#: A payload with ~3% of accumulated dividends: entirely ordinary over a
#: two-year window, and enough to put the adjusted close outside the
#: unadjusted session range.
_FMP_DIVIDEND_PAYLOAD = {
    "symbol": "KO",
    "currency": "USD",
    "historical": [
        {"date": "2020-01-03", "open": 100.0, "high": 100.5, "low": 99.5,
         "close": 100.0, "adjClose": 97.0, "volume": 1_000_000, "vwap": 100.1},
        {"date": "2020-01-02", "open": 99.0, "high": 99.8, "low": 98.5,
         "close": 99.5, "adjClose": 96.5, "volume": 1_100_000, "vwap": 99.2},
    ],
}


def test_fmp_ohlcv_survives_an_ordinary_dividend_adjustment(monkeypatch):
    provider = FinancialModelingPrep(api_key="k")
    monkeypatch.setattr(provider, "_get_json", lambda *a, **k: _FMP_DIVIDEND_PAYLOAD)
    panel = provider.fetch_one("KO", _request(("KO",), fields=F.OHLCV))
    assert panel.identifiers == ("KO",)


def test_fmp_puts_the_session_range_on_the_adjusted_scale(monkeypatch):
    provider = FinancialModelingPrep(api_key="k")
    monkeypatch.setattr(provider, "_get_json", lambda *a, **k: _FMP_DIVIDEND_PAYLOAD)
    panel = provider.fetch_one("KO", _request(("KO",), fields=F.OHLCV))

    # FMP publishes no adjusted high, so it is reconstructed from the day's
    # own adjClose/close ratio. Without that the adjusted close of 97 would
    # sit below an unadjusted low of 99.5.
    high = float(panel.frame(F.HIGH).loc["2020-01-03", "KO"])
    low = float(panel.frame(F.LOW).loc["2020-01-03", "KO"])
    close = float(panel.prices().loc["2020-01-03", "KO"])
    assert low <= close <= high
    assert high == pytest.approx(100.5 * (97.0 / 100.0))
    # The raw print is preserved untouched alongside it.
    assert float(panel.frame(F.CLOSE_RAW).loc["2020-01-03", "KO"]) == pytest.approx(100.0)


def test_fmp_asks_for_the_full_payload_even_for_a_close_only_request(monkeypatch):
    # ``serietype=line`` returns only the *unadjusted* close, which would then
    # be labelled as a total-return series — the default request silently
    # dropping every dividend.
    provider = FinancialModelingPrep(api_key="k")
    captured: dict = {}

    def fake(url, **kwargs):
        captured.update(kwargs)
        return _FMP_DIVIDEND_PAYLOAD

    monkeypatch.setattr(provider, "_get_json", fake)
    panel = provider.fetch_one("KO", _request(("KO",)))

    assert "serietype" not in captured.get("params", {})
    # And the close that comes back is the adjusted one.
    assert float(panel.prices().loc["2020-01-03", "KO"]) == pytest.approx(97.0)


def test_fmp_drops_a_bar_whose_adjustment_ratio_is_unknowable(monkeypatch):
    payload = {
        "symbol": "X",
        "historical": [
            # No ``close`` on the first row: the ratio for that date is
            # unknowable, and a wrong scale factor is worse than an unscaled
            # bar.
            {"date": "2020-01-02", "open": 10.0, "high": 10.5, "low": 9.5,
             "adjClose": 10.0},
            {"date": "2020-01-03", "open": 10.0, "high": 10.5, "low": 9.5,
             "close": 10.0, "adjClose": 10.0},
        ],
    }
    provider = FinancialModelingPrep(api_key="k")
    monkeypatch.setattr(provider, "_get_json", lambda *a, **k: payload)
    panel = provider.fetch_one("X", _request(("X",), fields=F.OHLC))

    # Leaving that bar on the raw scale beside an adjusted close is not a
    # smaller error — it is a low above its own close, and it would take the
    # whole panel down. An unknown range is a gap, which the panel carries.
    assert pd.isna(panel.frame(F.HIGH).loc["2020-01-02", "X"])
    assert float(panel.frame(F.HIGH).loc["2020-01-03", "X"]) == pytest.approx(10.5)
    # The close itself is untouched: only the range depended on the ratio.
    assert panel.prices().loc["2020-01-02", "X"] == pytest.approx(10.0)


def test_a_zero_price_print_still_fails_the_panel(monkeypatch):
    # A zero close is a bad tick, and the panel's job is to say so rather than
    # let it through as a -100% return.
    payload = {
        "symbol": "X",
        "historical": [
            {"date": "2020-01-02", "close": 0.0, "adjClose": 10.0},
            {"date": "2020-01-03", "close": 10.0, "adjClose": 10.0},
        ],
    }
    provider = FinancialModelingPrep(api_key="k")
    monkeypatch.setattr(provider, "_get_json", lambda *a, **k: payload)
    from optimization_engine.ingest.errors import PanelValidationError

    with pytest.raises(PanelValidationError, match="strictly positive"):
        provider.fetch_one("X", _request(("X",), fields=(F.CLOSE, F.CLOSE_RAW)))


def test_yahoos_raw_close_may_sit_outside_its_adjusted_range(monkeypatch):
    from optimization_engine.data import yahoo as yahoo_module

    adjusted = _yahoo_frame(["KO"], volume={"KO": 1e6})
    unadjusted = adjusted.copy()
    for field in ("Open", "High", "Low", "Close"):
        unadjusted[(field, "KO")] = unadjusted[(field, "KO")] * 1.04

    class _TwoShot:
        def __init__(self):
            self.calls = 0

        def download(self, **kwargs):
            self.calls += 1
            return adjusted if kwargs.get("auto_adjust") else unadjusted

    monkeypatch.setattr(yahoo_module, "_import_yfinance", lambda: _TwoShot())
    panel = Yahoo().fetch_batch(
        ("KO",), _request(("KO",), fields=(*F.OHLC, F.CLOSE_RAW))
    )
    assert float(panel.frame(F.CLOSE_RAW).iloc[0, 0]) > float(panel.frame(F.HIGH).iloc[0, 0])


# ---------------------------------------------------------------------------
# Local files: which close wins must not depend on column order
# ---------------------------------------------------------------------------


def _write_long(path, order):
    rows = []
    for day, raw, adj in zip(
        pd.bdate_range("2024-01-01", periods=4), [100, 100, 100, 100], [97, 98, 99, 100]
    ):
        rows.append({"date": day, "ticker": "AAA", "open": raw, "high": raw,
                     "low": raw, "close": raw, "adj_close": adj, "volume": 1000})
    pd.DataFrame(rows)[order].to_csv(path, index=False)


@pytest.mark.parametrize(
    "order",
    [
        ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
        ["date", "ticker", "open", "high", "low", "adj_close", "close", "volume"],
    ],
)
def test_a_long_file_prefers_the_adjusted_close_whatever_the_column_order(tmp_path, order):
    path = tmp_path / "long.csv"
    _write_long(path, order)
    panel = LocalFile(path=path).fetch_batch(
        ("AAA",), _request(("AAA",), start="2024-01-01", end="2024-01-31", fields=F.OHLCV)
    )
    # The adjusted series rises 97 → 100; the raw one is flat. Taking the flat
    # one would report a 0% return for an asset that returned 3%.
    assert panel.prices()["AAA"].tolist() == [97.0, 98.0, 99.0, 100.0]
    assert panel.frame(F.CLOSE_RAW)["AAA"].tolist() == [100.0] * 4


def test_a_long_file_with_only_a_plain_close_still_works(tmp_path):
    rows = [
        {"date": day, "ticker": "AAA", "close": price}
        for day, price in zip(pd.bdate_range("2024-01-01", periods=4), [10, 11, 12, 13])
    ]
    path = tmp_path / "plain.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    panel = LocalFile(path=path).fetch_batch(
        ("AAA",), _request(("AAA",), start="2024-01-01", end="2024-01-31")
    )
    assert panel.prices()["AAA"].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert panel.frame(F.CLOSE_RAW) is None


def test_a_long_file_with_an_adjusted_close_and_a_range_but_no_raw_close_says_why(tmp_path):
    # There is no way to put the range on the close's scale without the raw
    # print, and silently loading a contradictory panel is the worst option.
    rows = [
        {"date": day, "ticker": "AAA", "open": 100.0, "high": 101.0,
         "low": 99.0, "adj_close": adj}
        for day, adj in zip(pd.bdate_range("2024-01-01", periods=4), [97, 98, 99, 100])
    ]
    path = tmp_path / "adjusted_only.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ProviderResponseError, match="no raw `close` column"):
        LocalFile(path=path).fetch_batch(
            ("AAA",),
            _request(("AAA",), start="2024-01-01", end="2024-01-31", fields=F.OHLC),
        )


def test_a_long_file_with_an_adjusted_close_and_no_range_loads_fine(tmp_path):
    rows = [
        {"date": day, "ticker": "AAA", "adj_close": adj}
        for day, adj in zip(pd.bdate_range("2024-01-01", periods=4), [97, 98, 99, 100])
    ]
    path = tmp_path / "closes_only.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    panel = LocalFile(path=path).fetch_batch(
        ("AAA",), _request(("AAA",), start="2024-01-01", end="2024-01-31")
    )
    assert panel.prices()["AAA"].tolist() == [97.0, 98.0, 99.0, 100.0]


def test_two_spellings_of_the_adjusted_close_do_not_invent_a_raw_one(tmp_path):
    rows = [
        {"date": day, "ticker": "AAA", "adj_close": adj, "adjclose": adj}
        for day, adj in zip(pd.bdate_range("2024-01-01", periods=4), [97, 98, 99, 100])
    ]
    path = tmp_path / "two_adjusted.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    panel = LocalFile(path=path).fetch_batch(
        ("AAA",), _request(("AAA",), start="2024-01-01", end="2024-01-31")
    )
    # Both columns are adjusted; calling one of them "raw" would be a lie that
    # later turns into the wrong share count.
    assert panel.frame(F.CLOSE_RAW) is None
