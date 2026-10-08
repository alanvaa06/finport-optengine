"""Tests for the ingest-related command-line surface.

Offline throughout: everything runs against the ``sample`` provider, which
needs neither a key nor a network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.cli import main  # noqa: E402


def test_providers_lists_every_provider_and_its_key_state(capsys):
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    for name in ("sample", "yahoo", "stooq", "fred", "fmp", "tiingo", "file"):
        assert name in out
    assert "OPTENGINE_API_KEY_FMP" in out
    # FRED's inability to serve volume is stated, not implied.
    assert "index-style levels only" in out


def test_providers_json_is_machine_readable(capsys):
    assert main(["providers", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    by_name = {row["provider"]: row for row in rows}
    assert by_name["fred"]["serves_volume"] is False
    assert by_name["sample"]["ready"] is True
    # A key value must never appear, only whether one is set.
    assert "key_present" in by_name["fmp"]


def test_providers_never_prints_a_key(capsys, monkeypatch):
    monkeypatch.setenv("OPTENGINE_API_KEY_FMP", "sk-live-do-not-print-me")
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "do-not-print-me" not in out
    assert "sk-" in out  # the masked prefix is fine


def test_ingest_writes_a_panel_and_prints_its_coverage(tmp_path, capsys):
    output = tmp_path / "prices.csv"
    code = main([
        "ingest", "--provider", "sample",
        "--identifiers", "AAA,BBB",
        "--ingest-period", "1y",
        "--output", str(output),
    ])
    assert code == 0
    assert output.is_file()

    frame = pd.read_csv(output, index_col=0, parse_dates=True)
    assert list(frame.columns) == ["AAA", "BBB"]
    out = capsys.readouterr().out
    assert "2/2 identifiers from sample" in out
    assert "has_volume" in out


def test_ingest_writes_volume_when_the_universe_has_it(tmp_path):
    prices = tmp_path / "p.csv"
    volume = tmp_path / "v.csv"
    code = main([
        "ingest", "--provider", "sample", "--identifiers", "AAA,BBB",
        "--ingest-period", "1y", "--ingest-fields", "ohlcv",
        "--output", str(prices), "--volume-output", str(volume),
    ])
    assert code == 0
    assert volume.is_file()
    written = pd.read_csv(volume, index_col=0, parse_dates=True)
    assert list(written.columns) == ["AAA", "BBB"]
    assert written.notna().any().any()


def test_parquet_output_without_an_engine_names_the_install(tmp_path, capsys, monkeypatch):
    # pandas raises a two-paragraph ImportError listing two packages it tried.
    # What the caller needs is the one command that fixes it.
    def no_engine(*_args, **_kwargs):
        raise ImportError("Unable to find a usable engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_engine)
    code = main([
        "ingest", "--provider", "sample", "--identifiers", "AAA",
        "--ingest-period", "1y", "--output", str(tmp_path / "p.parquet"),
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "pyarrow" in err
    assert '.[data]' in err


def test_ingest_says_so_rather_than_writing_an_empty_volume_file(tmp_path, capsys):
    prices = tmp_path / "p.csv"
    volume = tmp_path / "v.csv"
    code = main([
        "ingest", "--provider", "sample", "--identifiers", "SP500,IPC",
        "--ingest-period", "1y", "--ingest-fields", "ohlcv",
        "--output", str(prices), "--volume-output", str(volume),
    ])
    assert code == 0
    assert prices.is_file()
    # An index universe has no volume; writing an empty file would be worse
    # than saying why there is nothing to write.
    assert not volume.exists()
    assert "No volume to write" in capsys.readouterr().out


def test_ingest_requires_a_provider_and_a_universe(capsys):
    assert main(["ingest", "--identifiers", "AAA", "--output", "x.csv"]) == 2
    assert "--provider is required" in capsys.readouterr().err
    assert main(["ingest", "--provider", "sample", "--output", "x.csv"]) == 2
    assert "--identifiers is required" in capsys.readouterr().err


def test_ingest_reports_an_unknown_provider(capsys):
    code = main([
        "ingest", "--provider", "nosuchsource", "--identifiers", "AAA",
        "--output", "x.csv",
    ])
    assert code == 2
    assert "Unknown data provider" in capsys.readouterr().err


def test_ingest_rejects_an_unsupported_output_extension(tmp_path, capsys):
    code = main([
        "ingest", "--provider", "sample", "--identifiers", "AAA",
        "--ingest-period", "1y", "--output", str(tmp_path / "p.json"),
    ])
    assert code == 2
    assert "Unsupported output extension" in capsys.readouterr().err


def test_ingest_honours_a_cache_directory(tmp_path, capsys):
    cache = tmp_path / "cache"
    args = [
        "ingest", "--provider", "sample", "--identifiers", "AAA",
        "--ingest-period", "1y", "--cache-dir", str(cache),
        "--output", str(tmp_path / "p.csv"),
    ]
    assert main(args) == 0
    assert cache.is_dir() and any(cache.iterdir())

    capsys.readouterr()
    assert main(args) == 0
    assert "from cache" in capsys.readouterr().out


def test_no_cache_directory_writes_nothing(tmp_path):
    main([
        "ingest", "--provider", "sample", "--identifiers", "AAA",
        "--ingest-period", "1y", "--output", str(tmp_path / "p.csv"),
    ])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.csv"]


def test_a_key_pasted_into_env_file_is_loaded(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPTENGINE_API_KEY_FMP", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPTENGINE_API_KEY_FMP=abc123\n")
    assert main(["providers", "--env-file", str(env)]) == 0
    out = capsys.readouterr().out
    assert "Key set" in out
    assert "abc123" not in out


def test_ingest_returns_a_nonzero_code_for_a_partial_load(tmp_path, capsys):
    # A file holding only one of the two requested series: the panel is still
    # written, but a partial load must not report itself as a clean success.
    index = pd.bdate_range("2024-01-01", periods=30)
    source = tmp_path / "source.csv"
    pd.DataFrame({"AAA": range(100, 130)}, index=index).to_csv(
        source, index_label="date"
    )

    code = main([
        "ingest", "--provider", "file", "--file-path", str(source),
        "--identifiers", "AAA,BBB",
        "--ingest-start", "2024-01-01", "--ingest-end", "2024-03-01",
        "--output", str(tmp_path / "p.csv"),
    ])
    assert code == 1
    out = capsys.readouterr().out
    assert "1/2 identifiers" in out
    assert "BBB" in out


def test_backtest_accepts_the_adv_liquidity_flags_without_volume(tmp_path, capsys):
    # The headline case: a capacity-aware backtest asked for on data that has
    # no volume must still run, and must say what it fell back to.
    config = tmp_path / "config.yaml"
    config.write_text(
        "periods_per_year: 252\n"
        "optimizer:\n"
        "  name: equal_weight\n"
        "expected_returns:\n"
        "  AAA: 0.05\n"
        "  BBB: 0.04\n"
    )
    code = main([
        "backtest", "--config", str(config),
        "--provider", "sample", "--identifiers", "AAA,BBB",
        "--ingest-period", "3y",
        "--impact-eta", "0.5",
        "--impact-participation-source", "adv",
        # Capacity is a currency amount, so ADV pricing needs a fund size.
        "--initial-capital", "100000000",
        "--lookback", "120", "--rebalance-every", "60",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "no volume panel available" in out
    assert "fixed participation rate" in out


def test_backtest_refuses_adv_pricing_without_a_fund_size(tmp_path, capsys):
    # The default initial_capital of 1.0 makes every market infinitely deep,
    # so the run would complete and report a cost of roughly nothing.
    config = tmp_path / "config.yaml"
    config.write_text(
        "periods_per_year: 252\n"
        "optimizer:\n"
        "  name: equal_weight\n"
        "expected_returns:\n"
        "  AAA: 0.05\n"
        "  BBB: 0.04\n"
    )
    code = main([
        "backtest", "--config", str(config),
        "--provider", "sample", "--identifiers", "AAA,BBB",
        "--ingest-period", "1y",
        "--impact-eta", "0.5", "--impact-participation-source", "adv",
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "real fund size" in err
    assert "--initial-capital" in err


def test_the_config_currency_map_still_converts_what_the_ingest_could_not(
    tmp_path, capsys, monkeypatch
):
    """Only series the ingest actually converted are exempt from the config.

    ``--ingest-currency`` used to make the CLI set ``config.currencies`` aside
    wholesale and announce that the panel had been converted on ingest —
    even when the file provider had converted nothing. A peso series was then
    optimized as if it were in dollars, with exit code 0.
    """
    import numpy as np

    from optimization_engine.data import fx

    days = pd.bdate_range("2021-01-04", periods=300)
    rng = np.random.default_rng(9)
    pd.DataFrame(
        {
            "date": days,
            "WALMEX": 60 * np.exp(np.cumsum(rng.normal(0, 0.012, len(days)))),
            "SPY": 400 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))),
        }
    ).to_csv(tmp_path / "mx.csv", index=False)
    config = tmp_path / "fx.yaml"
    config.write_text(
        "\n".join(
            [
                "optimizer: min_variance",
                "covariance_method: sample",
                "base_currency: USD",
                "currencies:",
                "  WALMEX: MXN",
                "  SPY: USD",
            ]
        )
    )
    asked = []

    def fake_rates(currencies, base, start=None, end=None):
        asked.append(sorted(currencies))
        rates = pd.DataFrame(
            {"MXN": np.linspace(0.050, 0.055, len(days)), "USD": 1.0}, index=days
        )
        return rates[[c for c in ("MXN", "USD") if c in currencies]]

    monkeypatch.setattr(fx, "fetch_fx_to_base", fake_rates)

    code = main(
        [
            "optimize", "--config", str(config),
            "--provider", "file", "--file-path", str(tmp_path / "mx.csv"),
            "--identifiers", "WALMEX,SPY", "--ingest-currency", "USD",
            "--ingest-start", "2021-01-01", "--ingest-end", "2022-12-31",
            "--output", str(tmp_path / "out.xlsx"),
        ]
    )
    err = capsys.readouterr().err

    assert code == 0
    assert asked == [["MXN", "USD"]], "the config's MXN entry was never applied"
    assert "converted to USD on ingest" not in err
