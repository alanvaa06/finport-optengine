"""Tests for FX conversion."""

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

from optimization_engine.data import fx  # noqa: E402
from optimization_engine.data.fx import (  # noqa: E402
    FXError,
    _normalize_currencies,
    convert_prices_to_base,
    fetch_fx_to_base,
    fetch_fx_to_usd,
    supported_currencies,
)

# ---------------------------------------------------------------------------
# Currency code validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["USD", "MXN", "EUR", "GBP"])
def test_normalize_accepts_iso(code: str):
    assert _normalize_currencies([code.lower()]) == [code]


@pytest.mark.parametrize("bad", ["", "US", "USDX", "1.0", "U$D", None])
def test_normalize_rejects_invalid(bad):
    with pytest.raises((FXError, TypeError)):
        _normalize_currencies([bad])


def test_normalize_dedupes():
    assert _normalize_currencies(["USD", "usd", "MXN"]) == ["USD", "MXN"]


def test_supported_currencies_contains_majors():
    supported = supported_currencies()
    for code in ["USD", "MXN", "EUR", "GBP", "JPY"]:
        assert code in supported


# ---------------------------------------------------------------------------
# fetch_fx_to_usd / fetch_fx_to_base — mocked FRED
# ---------------------------------------------------------------------------


def _fred_fixture(values: dict[str, float], n: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame(values, index=dates)


def test_fetch_fx_to_usd_inverts_xperusd(monkeypatch):
    # MXN: DEXMXUS = 17 MXN per 1 USD → MXN→USD = 1/17
    raw = _fred_fixture({"DEXMXUS": 17.0})
    monkeypatch.setattr(fx, "load_fred_series", lambda ids, **kw: raw[ids])

    out = fetch_fx_to_usd(["MXN"])
    assert "MXN" in out.columns
    assert pytest.approx(out["MXN"].iloc[0], rel=1e-6) == 1.0 / 17.0


def test_fetch_fx_to_usd_passes_through_usdperx(monkeypatch):
    # EUR: DEXUSEU = 1.10 USD per 1 EUR → EUR→USD = 1.10
    raw = _fred_fixture({"DEXUSEU": 1.10})
    monkeypatch.setattr(fx, "load_fred_series", lambda ids, **kw: raw[ids])

    out = fetch_fx_to_usd(["EUR"])
    assert pytest.approx(out["EUR"].iloc[0]) == 1.10


def test_fetch_fx_to_usd_rejects_unsupported(monkeypatch):
    monkeypatch.setattr(fx, "load_fred_series", lambda *a, **kw: pd.DataFrame())
    with pytest.raises(FXError, match="No built-in FRED mapping"):
        fetch_fx_to_usd(["XYZ"])


def test_fetch_fx_to_base_triangulates_via_usd(monkeypatch):
    # base = MXN, also need EUR.
    # DEXMXUS = 17 (MXN per USD), DEXUSEU = 1.10 (USD per EUR)
    # Expected: MXN→MXN = 1, EUR→MXN = 1.10 * 17 = 18.7
    raw = _fred_fixture({"DEXMXUS": 17.0, "DEXUSEU": 1.10})

    def fake_load(ids, **kw):
        return raw[list(ids)]

    monkeypatch.setattr(fx, "load_fred_series", fake_load)

    out = fetch_fx_to_base(["EUR", "MXN", "USD"], base="MXN")
    assert pytest.approx(out["MXN"].iloc[0]) == 1.0
    assert pytest.approx(out["EUR"].iloc[0], rel=1e-6) == 1.10 * 17.0
    assert pytest.approx(out["USD"].iloc[0], rel=1e-6) == 17.0


def test_fetch_fx_to_base_usd_keeps_one_constant(monkeypatch):
    raw = _fred_fixture({"DEXMXUS": 17.0})
    monkeypatch.setattr(fx, "load_fred_series", lambda ids, **kw: raw[ids])

    out = fetch_fx_to_base(["MXN", "USD"], base="USD")
    assert (out["USD"] == 1.0).all()
    assert pytest.approx(out["MXN"].iloc[0], rel=1e-6) == 1.0 / 17.0


# ---------------------------------------------------------------------------
# convert_prices_to_base — math correctness + edge cases
# ---------------------------------------------------------------------------


def test_convert_noop_when_all_assets_in_base(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame({"SPY": np.linspace(100, 110, 10)}, index=dates)
    out = convert_prices_to_base(prices, asset_currency={"SPY": "USD"}, base="USD")
    pd.testing.assert_frame_equal(out, prices)


def test_convert_applies_per_asset_currency(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame(
        {
            "SPY": np.full(10, 100.0),    # USD
            "MEXBOL": np.full(10, 50000.0),  # MXN
        },
        index=dates,
    )

    fx_rates = pd.DataFrame(
        {"USD": np.full(10, 1.0), "MXN": np.full(10, 1.0 / 17.0)},
        index=dates,
    )

    out = convert_prices_to_base(
        prices,
        asset_currency={"SPY": "USD", "MEXBOL": "MXN"},
        base="USD",
        fx_rates=fx_rates,
    )
    np.testing.assert_array_equal(out["SPY"].values, prices["SPY"].values)
    np.testing.assert_allclose(out["MEXBOL"].values, 50000.0 / 17.0)


def test_convert_forward_fills_fx_gaps():
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame({"X": np.full(10, 100.0)}, index=dates)
    fx_rates = pd.DataFrame(
        {"MXN": [1.0 / 17.0] + [np.nan] * 9},
        index=[dates[0]] + list(dates[1:]),
    )
    out = convert_prices_to_base(
        prices, asset_currency={"X": "MXN"}, base="USD", fx_rates=fx_rates
    )
    np.testing.assert_allclose(out["X"].values, np.full(10, 100.0 / 17.0))


def test_convert_rejects_missing_fx_currency():
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"X": np.full(5, 100.0)}, index=dates)
    fx_rates = pd.DataFrame({"EUR": np.full(5, 1.10)}, index=dates)
    with pytest.raises(FXError, match="MXN.*USD"):
        convert_prices_to_base(
            prices, asset_currency={"X": "MXN"}, base="USD", fx_rates=fx_rates
        )


def test_convert_requires_datetime_index():
    prices = pd.DataFrame({"X": [100, 101]}, index=[0, 1])
    with pytest.raises(FXError, match="DatetimeIndex"):
        convert_prices_to_base(
            prices, asset_currency={"X": "MXN"}, base="USD", fx_rates=pd.DataFrame()
        )


# ---------------------------------------------------------------------------
# End-to-end: convert + run engine.
# ---------------------------------------------------------------------------


def test_convert_then_optimize(monkeypatch):
    from optimization_engine.config import EngineConfig, OptimizerSpec
    from optimization_engine.data.loader import prices_to_returns
    from optimization_engine.engine import apply_fx_conversion, run_engine

    dates = pd.bdate_range("2024-01-02", periods=300)
    rng = np.random.default_rng(0)
    spy = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
    mex = 50000.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(dates)))
    prices = pd.DataFrame({"SPY": spy, "MEX": mex}, index=dates)

    fx_rates = pd.DataFrame(
        {"USD": np.full(len(dates), 1.0), "MXN": np.full(len(dates), 1.0 / 17.0)},
        index=dates,
    )

    cfg = EngineConfig(
        expected_returns={"SPY": 0.07, "MEX": 0.08},
        bounds={"SPY": [0.0, 1.0], "MEX": [0.0, 1.0]},
        currencies={"SPY": "USD", "MEX": "MXN"},
        base_currency="USD",
        optimizer=OptimizerSpec(name="risk_parity"),
    )

    converted = apply_fx_conversion(prices, cfg, fx_rates=fx_rates)
    returns = prices_to_returns(converted)
    run = run_engine(returns, cfg)
    assert run.result.weights.sum() == pytest.approx(1.0, abs=1e-3)


def test_a_short_leading_fx_gap_is_valued_at_the_first_known_rate():
    dates = pd.bdate_range("2024-01-01", periods=10)  # starts on a US holiday
    prices = pd.DataFrame({"X": np.full(10, 100.0)}, index=dates)
    fx_rates = pd.DataFrame({"MXN": np.full(9, 1.0 / 17.0)}, index=dates[1:])
    out = convert_prices_to_base(
        prices, asset_currency={"X": "MXN"}, base="USD", fx_rates=fx_rates
    )
    np.testing.assert_allclose(out["X"].values, np.full(10, 100.0 / 17.0))


def test_a_long_leading_fx_gap_is_refused_rather_than_filled_from_the_future():
    dates = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame({"X": np.full(30, 100.0)}, index=dates)
    late = dates[fx.MAX_LEADING_FX_GAP + 1 :]
    fx_rates = pd.DataFrame({"MXN": np.full(len(late), 1.0 / 17.0)}, index=late)
    # Back-filling a month of prices from a rate observed a month later
    # values them with information from the future.
    with pytest.raises(FXError, match="starts"):
        convert_prices_to_base(
            prices, asset_currency={"X": "MXN"}, base="USD", fx_rates=fx_rates
        )
    # The caller can still ask for it explicitly.
    out = convert_prices_to_base(
        prices, asset_currency={"X": "MXN"}, base="USD", fx_rates=fx_rates, fill="bfill"
    )
    np.testing.assert_allclose(out["X"].values, np.full(30, 100.0 / 17.0))


# ---------------------------------------------------------------------------
# Which rate a price date is valued at
# ---------------------------------------------------------------------------


def test_a_price_date_the_rate_series_skips_takes_the_last_rate_before_it():
    """An as-of join, not a reindex followed by a fill.

    Reindexing the rates onto the price dates first throws away every rate
    that falls between two price dates, so a month-end on a weekend took the
    rate of the *previous month-end* rather than that Friday's. On two years
    of month-end prices, 7 of 24 months were valued a month stale and the
    FX part of the monthly returns correlated 0.40 with the truth.
    """
    month_ends = pd.date_range("2023-01-31", "2024-12-31", freq="ME")
    prices = pd.DataFrame({"EU": 100.0}, index=month_ends)
    days = pd.bdate_range("2023-01-02", "2025-01-10")
    eur = pd.Series(np.linspace(1.00, 1.24, len(days)), index=days)

    out = convert_prices_to_base(
        prices, {"EU": "EUR"}, "USD", fx_rates=pd.DataFrame({"EUR": eur})
    )

    as_of = eur.reindex(eur.index.union(month_ends)).ffill().reindex(month_ends)
    assert (month_ends.dayofweek >= 5).sum() == 7
    np.testing.assert_allclose(out["EU"].values, 100.0 * as_of.values)

