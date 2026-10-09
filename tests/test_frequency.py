"""Annualization follows the data's own spacing, not a default of 252.

A monthly panel annualized on 252 periods reports a 6% volatility as 29% and
an 8% return as 170%. The ingest request carried ``periods_per_year`` for its
interval and nothing read it; the CLI used the config's 252 whatever the
dates said. These tests pin the order in which the factor is decided — a
stated value, the ingest interval, the dates, the default — and that a
stated value or an interval the dates contradict is refused, not overridden.
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

from optimization_engine.cli import main  # noqa: E402
from optimization_engine.data.covariance import covariance_matrix  # noqa: E402
from optimization_engine.data.frequency import (  # noqa: E402
    FrequencyMismatchError,
    infer_periods_per_year,
    resolve_periods_per_year,
)
from optimization_engine.data.loader import prices_to_returns, sample_dataset  # noqa: E402


@pytest.fixture(scope="module")
def monthly() -> pd.DataFrame:
    daily = sample_dataset(n_periods=252 * 10, seed=1)[["US_Equity", "Gold", "US_Treasuries"]]
    return daily.resample("ME").last()


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("B", 252), ("D", 252), ("W-FRI", 52), ("2W-FRI", 26), ("ME", 12), ("MS", 12),
        ("QE", 4), ("6ME", 2), ("6MS", 2), ("YE", 1),
    ],
)
def test_the_spacing_of_the_dates_names_the_frequency(freq, expected):
    index = pd.date_range("2015-01-01", periods=40, freq=freq)
    assert infer_periods_per_year(index) == expected


def test_a_biweekly_series_is_not_annualized_as_daily():
    """The repro: a fortnightly NAV fell between the bands and got 252."""
    biweekly = pd.date_range("2020-01-03", periods=60, freq="2W-FRI")
    ppy, note = resolve_periods_per_year(biweekly)
    assert ppy == 26
    assert "biweekly" in note


def test_a_semiannual_series_annualizes_on_two():
    semiannual = pd.date_range("2000-06-30", periods=30, freq="6ME")
    ppy, note = resolve_periods_per_year(semiannual)
    assert ppy == 2
    assert "semiannual" in note
    with pytest.raises(FrequencyMismatchError, match="periods_per_year: 2"):
        resolve_periods_per_year(semiannual, stated=4)


def test_semimonthly_dates_may_state_twenty_four():
    """1st-and-15th dates sit in the biweekly band; 24 is what they are."""
    semimonthly = pd.date_range("2020-01-01", periods=48, freq="SMS")
    assert resolve_periods_per_year(semimonthly, stated=24) == (24, "")
    with pytest.raises(FrequencyMismatchError, match="periods_per_year: 26"):
        resolve_periods_per_year(semimonthly, stated=252)


def test_seven_day_daily_data_is_daily_and_may_state_365():
    """Crypto trades every day: still daily, and 365 is a statement about it."""
    calendar_days = pd.date_range("2020-01-01", periods=400, freq="D")
    assert infer_periods_per_year(calendar_days) == 252
    assert resolve_periods_per_year(calendar_days, stated=365) == (365, "")


def test_too_few_or_irregular_dates_infer_nothing():
    assert infer_periods_per_year(pd.date_range("2020-01-01", periods=2, freq="ME")) is None
    assert infer_periods_per_year(pd.RangeIndex(10)) is None
    irregular = pd.DatetimeIndex(["2020-01-01", "2020-01-20", "2020-03-01", "2020-04-15"])
    assert infer_periods_per_year(irregular) is None


def test_a_stated_value_wins_when_the_dates_agree_with_it():
    daily = pd.bdate_range("2020-01-01", periods=60)
    # Seven-day markets annualize on 365; that is a statement about daily data.
    assert resolve_periods_per_year(daily, stated=365) == (365, "")


def test_a_stated_value_the_dates_contradict_is_refused(monthly):
    with pytest.raises(FrequencyMismatchError, match="periods_per_year: 12"):
        resolve_periods_per_year(monthly.index, stated=252)


def test_the_interval_decides_when_the_config_is_silent():
    weekly = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    ppy, note = resolve_periods_per_year(weekly, interval="1wk")
    assert ppy == 52
    assert "1wk" in note


def test_an_interval_the_dates_contradict_is_refused(monthly):
    with pytest.raises(FrequencyMismatchError, match="'1d'"):
        resolve_periods_per_year(monthly.index, interval="1d")


def test_with_nothing_stated_the_dates_decide(monthly):
    ppy, note = resolve_periods_per_year(monthly.index)
    assert ppy == 12
    assert "monthly" in note
    assert resolve_periods_per_year(pd.RangeIndex(10), default=252) == (252, "")


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _optimize_json(capsys, argv):
    code = main([*argv, "--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_a_monthly_file_is_annualized_on_twelve(monthly, tmp_path, capsys):
    """The repro: 28.7% reported for a book whose volatility is 6.3%."""
    csv = tmp_path / "monthly.csv"
    monthly.to_csv(csv, index_label="date")
    config = tmp_path / "cfg.yaml"
    config.write_text("optimizer: min_variance\ncovariance_method: sample\n")

    code, payload, err = _optimize_json(
        capsys,
        ["optimize", "--config", str(config), "--prices", str(csv),
         "--output", str(tmp_path / "o.xlsx")],
    )

    assert code == 0
    weights = pd.Series(payload["weights"])
    returns = prices_to_returns(monthly).dropna()
    cov = covariance_matrix(returns, method="sample", periods_per_year=12)
    expected = float(np.sqrt(weights @ cov.loc[weights.index, weights.index] @ weights))
    assert payload["metrics"]["expected_volatility"] == pytest.approx(expected, rel=1e-6)
    assert payload["metrics"]["expected_volatility"] < 0.10
    assert "12 periods per year" in err


def test_the_ingest_interval_reaches_the_annualization(monthly, tmp_path, capsys):
    csv = tmp_path / "monthly.csv"
    monthly.to_csv(csv, index_label="date")
    config = tmp_path / "cfg.yaml"
    config.write_text("optimizer: min_variance\ncovariance_method: sample\n")

    code, payload, _ = _optimize_json(
        capsys,
        ["optimize", "--config", str(config), "--provider", "file",
         "--file-path", str(csv), "--identifiers", "US_Equity,Gold,US_Treasuries",
         "--ingest-interval", "1mo", "--ingest-period", "20y",
         "--output", str(tmp_path / "o.xlsx")],
    )
    assert code == 0
    assert payload["metrics"]["expected_volatility"] < 0.10


@pytest.mark.parametrize("command", ["optimize", "check", "backtest"])
def test_a_config_that_contradicts_the_dates_exits_two(monthly, tmp_path, capsys, command):
    csv = tmp_path / "monthly.csv"
    monthly.to_csv(csv, index_label="date")
    config = tmp_path / "cfg.yaml"
    config.write_text("optimizer: min_variance\nperiods_per_year: 252\n")
    argv = [command, "--config", str(config), "--prices", str(csv)]
    if command == "optimize":
        argv += ["--output", str(tmp_path / "o.xlsx")]

    code, payload, err = _optimize_json(capsys, argv)

    assert code == 2
    assert "periods_per_year: 252" in payload["error"]
    assert "monthly" in err


def test_a_daily_panel_with_the_default_is_unchanged(tmp_path, capsys):
    config = tmp_path / "cfg.yaml"
    config.write_text("optimizer: min_variance\n")
    code, _, err = _optimize_json(
        capsys,
        ["optimize", "--config", str(config), "--sample", "--output", str(tmp_path / "o.xlsx")],
    )
    assert code == 0
    assert "periods per year" not in err
