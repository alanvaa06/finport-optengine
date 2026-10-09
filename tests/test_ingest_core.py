"""Tests for the ingest vocabulary, panel entity and request spec.

No network anywhere: the panel and the request are pure data structures, and
these tests are about the invariants they promise to enforce.
"""

from __future__ import annotations

import datetime as dt
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
    PanelValidationError,
    ProviderConfigurationError,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta  # noqa: E402
from optimization_engine.ingest.spec import IngestRequest  # noqa: E402


def _dates(n: int = 10) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _close(columns=("AAA", "BBB"), n: int = 10) -> pd.DataFrame:
    index = _dates(n)
    data = {
        name: np.linspace(100.0, 110.0, n) + offset
        for offset, name in enumerate(columns)
    }
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# Field vocabulary
# ---------------------------------------------------------------------------


def test_normalize_fields_orders_canonically_and_always_keeps_close():
    assert F.normalize_fields([F.VOLUME, F.OPEN]) == (F.OPEN, F.CLOSE, F.VOLUME)


def test_normalize_fields_deduplicates():
    assert F.normalize_fields([F.CLOSE, F.CLOSE, F.CLOSE]) == (F.CLOSE,)


def test_normalize_fields_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown field"):
        F.normalize_fields(["m_close", "m_dividend_yield"])


def test_index_kinds_have_no_volume_and_are_not_tradeable():
    assert not F.InstrumentKind.INDEX.has_volume
    assert not F.InstrumentKind.INDEX.is_tradeable
    assert not F.InstrumentKind.RATE.is_tradeable
    assert F.InstrumentKind.EQUITY.has_volume
    assert F.InstrumentKind.EQUITY.is_tradeable


# ---------------------------------------------------------------------------
# Panel construction and validation
# ---------------------------------------------------------------------------


def test_from_frames_aligns_index_and_columns_across_fields():
    close = _close()
    # A high frame that is short two days and has its columns the other way
    # round: both are things a real provider does.
    high = (close * 1.01).iloc[:-2][["BBB", "AAA"]]
    panel = PricePanel.from_frames({F.CLOSE: close, F.HIGH: high})

    assert panel.frames[F.HIGH].index.equals(panel.frames[F.CLOSE].index)
    assert list(panel.frames[F.HIGH].columns) == list(panel.frames[F.CLOSE].columns)
    assert panel.frames[F.HIGH].iloc[-1].isna().all()


def test_panel_requires_a_close_frame():
    with pytest.raises(PanelValidationError, match="must carry"):
        PricePanel.from_frames({F.OPEN: _close()})


def test_panel_rejects_non_positive_prices():
    close = _close()
    close.iloc[3, 1] = 0.0
    with pytest.raises(PanelValidationError, match="strictly positive"):
        PricePanel.from_frames({F.CLOSE: close})


def test_panel_rejects_negative_volume():
    close = _close()
    volume = pd.DataFrame(1e6, index=close.index, columns=close.columns)
    volume.iloc[2, 0] = -5.0
    with pytest.raises(PanelValidationError, match="non-negative"):
        PricePanel.from_frames({F.CLOSE: close, F.VOLUME: volume})


def test_panel_rejects_a_low_above_its_high():
    close = _close()
    high = close * 1.01
    low = close * 0.99
    low.iloc[4, 0] = float(high.iloc[4, 0]) * 1.05
    with pytest.raises(PanelValidationError, match="exceeds the session high"):
        PricePanel.from_frames({F.CLOSE: close, F.HIGH: high, F.LOW: low})


def test_panel_rejects_a_close_outside_the_session_range():
    close = _close()
    high = close * 1.01
    low = close * 0.99
    high.iloc[5, 1] = float(close.iloc[5, 1]) * 0.90
    low.iloc[5, 1] = float(close.iloc[5, 1]) * 0.80
    with pytest.raises(PanelValidationError, match="exceeds the session high"):
        PricePanel.from_frames({F.CLOSE: close, F.HIGH: high, F.LOW: low})


def test_ohlc_tolerance_absorbs_independent_vendor_rounding():
    # A high one part in ten thousand below the close is rounding, not a
    # crossed market, and must not fail the panel.
    close = _close()
    high = close * (1 - 1e-5)
    low = close * 0.99
    panel = PricePanel.from_frames({F.CLOSE: close, F.HIGH: high, F.LOW: low})
    assert panel.identifiers == ("AAA", "BBB")


def test_panel_rejects_infinite_values():
    close = _close()
    close.iloc[1, 0] = np.inf
    with pytest.raises(PanelValidationError, match="infinite"):
        PricePanel.from_frames({F.CLOSE: close})


def test_panel_accepts_nan_gaps():
    close = _close()
    close.iloc[2, 0] = np.nan
    panel = PricePanel.from_frames({F.CLOSE: close})
    assert panel.prices().iloc[2, 0] != panel.prices().iloc[2, 0]  # NaN


def test_panel_needs_two_observations_somewhere():
    close = pd.DataFrame({"AAA": [100.0]}, index=_dates(1))
    with pytest.raises(PanelValidationError, match="two or more observations"):
        PricePanel.from_frames({F.CLOSE: close})


def test_panel_rejects_duplicate_dates():
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03"])
    close = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=index)
    # ``from_frames`` de-duplicates on the way in, so this checks the
    # validator itself against a panel built around it.
    panel = PricePanel(frames={F.CLOSE: close})
    with pytest.raises(PanelValidationError, match="duplicate dates"):
        panel.validate()


# ---------------------------------------------------------------------------
# Volume semantics
# ---------------------------------------------------------------------------


def test_volumes_returns_none_when_no_identifier_reports_any():
    close = _close()
    volume = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    panel = PricePanel.from_frames({F.CLOSE: close, F.VOLUME: volume})
    assert panel.volumes() is None
    assert not panel.has_volume


def test_volumes_survives_when_only_some_identifiers_have_it():
    close = _close()
    volume = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    volume["AAA"] = 1e6
    panel = PricePanel.from_frames({F.CLOSE: close, F.VOLUME: volume})
    assert panel.has_volume
    assert panel.volumes()["BBB"].dropna().empty


def test_coverage_reports_provenance_and_volume_availability():
    close = _close()
    panel = PricePanel.from_frames(
        {F.CLOSE: close},
        {
            "AAA": SeriesMeta("AAA", "^GSPC", "yahoo", F.InstrumentKind.INDEX, "USD"),
        },
    )
    coverage = panel.coverage()
    assert coverage.loc["AAA", "provider"] == "yahoo"
    assert coverage.loc["AAA", "symbol"] == "^GSPC"
    assert coverage.loc["AAA", "kind"] == "index"
    assert not bool(coverage.loc["AAA", "has_volume"])
    # An identifier with no metadata still gets a row rather than vanishing.
    assert coverage.loc["BBB", "provider"] == "-"


def test_tradeable_excludes_indices_and_rates():
    close = _close(("SPY", "SP500", "DGS10"))
    panel = PricePanel.from_frames(
        {F.CLOSE: close},
        {
            "SPY": SeriesMeta("SPY", "SPY", "yahoo", F.InstrumentKind.ETF),
            "SP500": SeriesMeta("SP500", "^GSPC", "yahoo", F.InstrumentKind.INDEX),
            "DGS10": SeriesMeta("DGS10", "DGS10", "fred", F.InstrumentKind.RATE),
        },
    )
    assert panel.tradeable == ("SPY",)


# ---------------------------------------------------------------------------
# Panel algebra
# ---------------------------------------------------------------------------


def test_merge_unions_identifiers_and_fields():
    left = PricePanel.from_frames({F.CLOSE: _close(("AAA",))})
    right_close = _close(("BBB",))
    right = PricePanel.from_frames(
        {F.CLOSE: right_close, F.VOLUME: right_close * 1000.0}
    )
    merged = left.merge(right)

    assert set(merged.identifiers) == {"AAA", "BBB"}
    assert F.VOLUME in merged.frames
    # AAA has no volume; the union must leave it missing, not zero.
    assert merged.frames[F.VOLUME]["AAA"].dropna().empty


def test_merge_lets_the_right_hand_panel_win_a_collision():
    left = PricePanel.from_frames({F.CLOSE: _close(("AAA",))})
    right = PricePanel.from_frames({F.CLOSE: _close(("AAA",)) * 2.0})
    merged = left.merge(right)
    assert float(merged.prices()["AAA"].iloc[0]) == pytest.approx(
        float(right.prices()["AAA"].iloc[0])
    )


def test_select_preserves_the_requested_order():
    panel = PricePanel.from_frames({F.CLOSE: _close(("AAA", "BBB", "CCC"))})
    assert panel.select(["CCC", "AAA"]).identifiers == ("CCC", "AAA")


def test_select_rejects_an_unknown_identifier():
    panel = PricePanel.from_frames({F.CLOSE: _close()})
    with pytest.raises(PanelValidationError, match="not in panel"):
        panel.select(["ZZZ"])


# ---------------------------------------------------------------------------
# Request spec
# ---------------------------------------------------------------------------


def test_request_parses_a_delimited_identifier_string():
    request = IngestRequest(identifiers="SPY, AGG  GLD,", provider="sample")
    assert request.identifiers == ("SPY", "AGG", "GLD")


def test_request_preserves_identifier_case():
    # The identifier becomes a column name and therefore a config key; mangling
    # it would silently break every constraint written against it.
    request = IngestRequest(identifiers=["US_Equity", "Cash"], provider="sample")
    assert request.identifiers == ("US_Equity", "Cash")


def test_request_deduplicates_case_insensitively():
    request = IngestRequest(identifiers=["spy", "SPY"], provider="sample")
    assert request.identifiers == ("spy",)


def test_request_rejects_hostile_identifiers():
    for bad in ["../etc/passwd", "SPY AGG/../x", "a" * 40, "sp y"]:
        with pytest.raises(ProviderConfigurationError):
            IngestRequest(identifiers=[bad], provider="sample")


def test_request_requires_at_least_one_identifier():
    with pytest.raises(ProviderConfigurationError, match="At least one identifier"):
        IngestRequest(identifiers=[], provider="sample")


def test_request_resolves_period_into_a_window():
    request = IngestRequest(identifiers=["AAA"], provider="sample", period="1y")
    assert request.end == dt.date.today()
    assert (request.end - request.start).days == 365


def test_request_rejects_a_backwards_window():
    with pytest.raises(ProviderConfigurationError, match="strictly before"):
        IngestRequest(
            identifiers=["AAA"], provider="sample",
            start="2024-01-01", end="2023-01-01",
        )


def test_request_rejects_an_unknown_interval():
    with pytest.raises(ProviderConfigurationError, match="Unsupported interval"):
        IngestRequest(identifiers=["AAA"], provider="sample", interval="5m")


def test_request_rejects_a_malformed_currency():
    with pytest.raises(ProviderConfigurationError, match="three-letter"):
        IngestRequest(identifiers=["AAA"], provider="sample", currency="dollars")


def test_fingerprint_tracks_data_affecting_fields_only():
    base = IngestRequest(
        identifiers=["AAA"], provider="sample", start="2024-01-01", end="2024-06-01"
    )
    same_data = IngestRequest(
        identifiers=["AAA"], provider="sample", start="2024-01-01", end="2024-06-01",
        max_workers=4, cache_dir="/tmp/whatever", metadata={"note": "hi"},
    )
    other_data = base.for_provider("stooq")

    assert base.fingerprint() == same_data.fingerprint()
    assert base.fingerprint() != other_data.fingerprint()


def test_periods_per_year_follows_the_interval():
    assert IngestRequest(identifiers=["A"], interval="1d").periods_per_year == 252
    assert IngestRequest(identifiers=["A"], interval="1wk").periods_per_year == 52
    assert IngestRequest(identifiers=["A"], interval="1mo").periods_per_year == 12


def test_round_trip_through_dict():
    request = IngestRequest(
        identifiers=["SPY", "AGG"], provider="yahoo", start="2020-01-01",
        end="2021-01-01", fields=F.OHLCV, currency="usd", require_volume=True,
    )
    rebuilt = IngestRequest.from_dict(request.to_dict())
    assert rebuilt.fingerprint() == request.fingerprint()
    assert rebuilt.require_volume is True
    assert rebuilt.currency == "USD"


def test_from_dict_rejects_an_unknown_option():
    with pytest.raises(ProviderConfigurationError, match="Unknown ingest option"):
        IngestRequest.from_dict({"identifiers": ["A"], "tickers": ["B"]})


# ---------------------------------------------------------------------------
# Adjustment scales — the session-range check must not compare two of them
# ---------------------------------------------------------------------------


def test_a_raw_close_outside_an_adjusted_range_is_not_an_error():
    # After a few percent of accumulated dividends the unadjusted print sits
    # legitimately outside the adjusted day's high and low. Rejecting that
    # would make an ordinary OHLCV request for any dividend payer impossible.
    close = _close(("AAA",))
    panel = PricePanel.from_frames(
        {
            F.CLOSE: close,
            F.HIGH: close * 1.005,
            F.LOW: close * 0.995,
            F.CLOSE_RAW: close * 1.04,
        }
    )
    assert panel.frame(F.CLOSE_RAW) is not None


def test_an_adjusted_close_outside_its_own_range_is_still_an_error():
    # The check still has teeth for fields that do share a scale.
    close = _close(("AAA",))
    high = close * 1.005
    low = close * 0.995
    high.iloc[3, 0] = float(close.iloc[3, 0]) * 0.90
    low.iloc[3, 0] = float(close.iloc[3, 0]) * 0.80
    with pytest.raises(PanelValidationError, match="exceeds the session high"):
        PricePanel.from_frames({F.CLOSE: close, F.HIGH: high, F.LOW: low})


def test_a_crossed_high_and_low_is_always_an_error():
    close = _close(("AAA",))
    high = close * 1.005
    low = close * 1.05
    with pytest.raises(PanelValidationError, match="exceeds the session high"):
        PricePanel.from_frames({F.CLOSE: close, F.HIGH: high, F.LOW: low})


# ---------------------------------------------------------------------------
# Merge must not splice two providers' price levels together
# ---------------------------------------------------------------------------


def test_merge_replaces_a_colliding_column_rather_than_filling_its_gaps():
    index = _dates(5)
    left = PricePanel.from_frames(
        {F.CLOSE: pd.DataFrame({"SPY": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=index)},
        {"SPY": SeriesMeta("SPY", "SPY", "providerA")},
    )
    right = PricePanel.from_frames(
        {F.CLOSE: pd.DataFrame({"SPY": [100.0, 101.0, np.nan, 103.0, 104.0]}, index=index)},
        {"SPY": SeriesMeta("SPY", "SPY", "providerB")},
    )
    merged = left.merge(right)

    # Filling the gap from the left panel would splice a 12 into a series
    # around 100 — a fabricated -88% / +758% round trip, under a provenance
    # record naming one provider.
    assert merged.prices()["SPY"].isna().sum() == 1
    assert merged.prices()["SPY"].dropna().between(99.0, 105.0).all()
    assert merged.meta["SPY"].provider == "providerB"


def test_merge_still_keeps_columns_only_one_side_has():
    left = PricePanel.from_frames({F.CLOSE: _close(("AAA",))})
    right = PricePanel.from_frames({F.CLOSE: _close(("BBB",))})
    merged = left.merge(right)
    assert set(merged.identifiers) == {"AAA", "BBB"}
    assert merged.prices()["AAA"].notna().all()
