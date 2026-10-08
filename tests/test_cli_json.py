"""The `--json` contract, held to what a machine consumer needs from it.

These tests are deliberately about *shape*, not values. A consumer parsing
this output cares that `weights` is an object of floats and that
`feasibility.feasible` is a boolean it can branch on; it does not care what
the optimizer decided today. Asserting on numbers here would make the suite
fail every time an estimator improves, which trains people to update the
expected values without reading them.

The one thing worth guarding hardest is that stdout parses at all. Every
command prints as it works, and a single stray `print` reaching stdout in
JSON mode breaks every caller — silently, because a truncated document
usually still looks like output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optimization_engine.cli import main  # noqa: E402
from optimization_engine.reporting.payloads import SCHEMA_VERSION  # noqa: E402

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "example_multi_asset.yaml")


def _run(capsys, argv: list[str]) -> tuple[int, dict]:
    """Run the CLI and parse stdout, asserting the streams stayed separate."""
    code = main(argv)
    captured = capsys.readouterr()
    # The real assertion: stdout is a single JSON document and nothing else.
    # json.loads is strict about trailing content, so a leaked print fails
    # here rather than corrupting a downstream parse.
    payload = json.loads(captured.out)
    return code, payload


def test_describe_json_reports_the_optimizer_contract(capsys):
    code, payload = _run(capsys, ["describe", "risk_parity", "--json"])
    assert code == 0
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["command"] == "describe"
    assert payload["name"] == "risk_parity"
    # `requires` and `supports` are what an agent reads before building a
    # config: a turnover budget handed to a method that does not support one
    # is ignored, not rejected, so the flag has to be discoverable up front.
    assert set(payload["requires"]) == {
        "expected_returns",
        "covariance",
        "return_history",
        "benchmark",
    }
    assert all(isinstance(v, bool) for v in payload["requires"].values())
    assert all(isinstance(v, bool) for v in payload["supports"].values())


def test_describe_json_on_an_unknown_name_still_emits_json(capsys):
    """A failure a caller can parse beats a failure it has to guess at."""
    code, payload = _run(capsys, ["describe", "no_such_optimizer", "--json"])
    assert code != 0
    assert payload["command"] == "describe"
    assert payload["error"]
    assert payload["exit_code"] == code


def test_check_json_answers_ready_with_a_boolean(capsys):
    code, payload = _run(capsys, ["check", "--config", CONFIG, "--sample", "--json"])
    assert code == 0
    assert payload["command"] == "check"
    # One boolean to branch on, mirroring the exit code.
    assert payload["ready"] is True
    assert payload["feasibility"]["feasible"] is True
    assert isinstance(payload["data_quality"]["errors"], list)
    assert payload["covariance"]["is_psd"] is True
    assert payload["covariance"]["n_assets"] > 0


def test_optimize_json_carries_weights_and_the_evidence(capsys, tmp_path):
    out = tmp_path / "report.xlsx"
    code, payload = _run(
        capsys,
        ["optimize", "--config", CONFIG, "--sample", "--output", str(out), "--json"],
    )
    assert code == 0
    assert payload["command"] == "optimize"

    weights = payload["weights"]
    assert weights and all(isinstance(v, float) for v in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-6

    # The claim this library makes is that weights alone are not a result.
    # If these ever come back None on a successful solve, the payload has
    # stopped carrying the thing that distinguishes it.
    assert payload["diagnostics"] is not None
    assert payload["covariance"] is not None
    assert payload["diagnostics"]["effective_n"] is not None
    assert payload["diagnostics"]["effective_n_risk"] is not None
    assert payload["solver"], "the solver that answered should be named"
    assert payload["output_path"] == str(out)


def test_backtest_json_carries_the_hashes_that_identify_the_run(capsys):
    code, payload = _run(
        capsys,
        [
            "backtest",
            "--config", CONFIG,
            "--sample",
            "--lookback", "504",
            "--rebalance-every", "252",
            "--json",
        ],
    )
    assert code == 0
    assert payload["command"] == "backtest"
    # Without these two a caller cannot tell a real change from a re-run,
    # which is the whole reason to read this instead of the workbook.
    assert payload["spec_hash"]
    assert payload["result_hash"]
    assert payload["window"]["n_periods"] > 0
    assert isinstance(payload["degradations"], list)
    # No --output was passed, so there is no workbook to point at. This also
    # covers the path where the writer never runs.
    assert payload["output_path"] is None


@pytest.mark.parametrize(
    "argv",
    [
        ["describe", "risk_parity", "--json"],
        ["check", "--config", CONFIG, "--sample", "--json"],
    ],
)
def test_narration_goes_to_stderr_not_stdout(capsys, argv):
    """Human narration is preserved — it just moves off the parsed stream."""
    main(argv)
    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout is exactly one document
    assert captured.err.strip(), "the human-readable output should still exist"


def test_every_payload_declares_its_schema_version(capsys):
    """A consumer must be able to refuse a version it does not know."""
    for argv in (
        ["describe", "risk_parity", "--json"],
        ["check", "--config", CONFIG, "--sample", "--json"],
    ):
        _, payload = _run(capsys, argv)
        assert payload["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize(
    ("argv", "expected_type"),
    [
        (["optimize", "--config", "/no/such/file.yaml", "--sample", "--json"],
         "FileNotFoundError"),
        (["check", "--config", "/no/such/file.yaml", "--sample", "--json"],
         "FileNotFoundError"),
        (["backtest", "--config", "/no/such/file.yaml", "--sample", "--json"],
         "FileNotFoundError"),
    ],
)
def test_a_raised_exception_still_emits_json(capsys, argv, expected_type):
    """The half of the contract that a returned exit code does not cover.

    `_emit_json` originally caught only a command that *returned* non-zero.
    A command that *raised* — an unreadable config being the cheapest way in
    — printed a traceback and left stdout empty, which is exactly the "no
    output versus output I could not parse" ambiguity this mode exists to
    remove. The payload has to survive the exception, not just the failure.
    """
    code, payload = _run(capsys, argv)
    assert code != 0
    assert payload["exit_code"] == code
    assert payload["schema_version"] == SCHEMA_VERSION
    # The type and message, so a caller can tell a missing file from a bad
    # one without scraping the traceback.
    assert expected_type in payload["error"]


def test_the_traceback_survives_on_stderr(capsys):
    """Catching the exception must not cost the human their diagnosis."""
    main(["optimize", "--config", "/no/such/file.yaml", "--sample", "--json"])
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert "Traceback" in captured.err
    assert "FileNotFoundError" in captured.err


def test_a_returned_failure_carries_its_reason(capsys, tmp_path):
    # A command that *returns* a non-zero code, as opposed to raising, used
    # to emit "the command exited before producing a result" — true, and
    # useless. The reason it printed to stderr now travels in the payload.
    config = tmp_path / "wrong.yaml"
    config.write_text("expected_returns:\n  NOT_IN_PANEL: 0.05\noptimizer: min_variance\n")
    code, payload = _run(capsys, ["optimize", "--config", str(config), "--sample", "--json"])
    assert code == 2
    assert payload["exit_code"] == 2
    assert "no expected returns" in payload["error"]


def test_backtest_notes_carry_their_values_not_only_their_keys():
    """A note's value must survive serialization.

    ``notes`` went through the same helper as the message lists, which
    iterates a mapping and keeps only its keys — so a reader of ``--json``
    learned that ``schedule_dates_moved`` had been recorded but never which
    dates moved where.
    """
    import datetime

    import numpy as np
    import pandas as pd

    from optimization_engine.reporting.payloads import _notes

    out = _notes(
        {
            "periods_in_cash_after_failed_solve": np.int64(3),
            "schedule_dates_moved": {
                pd.Timestamp("2020-01-05"): pd.Timestamp("2020-01-06")
            },
            "missing_returns": ["AAA", "BBB"],
            "cutoff": datetime.date(2020, 1, 5),
            "ratio": np.float64(0.5),
        }
    )
    assert out["periods_in_cash_after_failed_solve"] == 3
    assert out["schedule_dates_moved"] == {
        "2020-01-05T00:00:00": "2020-01-06T00:00:00"
    }
    assert out["missing_returns"] == ["AAA", "BBB"]
    assert out["cutoff"] == "2020-01-05T00:00:00"
    assert out["ratio"] == 0.5
    # The whole point: it survives a strict encoder.
    json.dumps(out)


def test_backtest_notes_are_empty_rather_than_absent():
    from optimization_engine.reporting.payloads import _notes

    assert _notes(None) == {}
    assert _notes({}) == {}


# ---------------------------------------------------------------------------
# What the data looked like, and where it came from
# ---------------------------------------------------------------------------


@pytest.fixture
def flawed_panel(tmp_path):
    """An unadjusted 4:1 split and an asset missing every other day for half
    its history (25% interior gaps, an error by the quality report's rule)."""
    import numpy as np
    import pandas as pd

    days = pd.bdate_range("2021-01-04", periods=600)
    rng = np.random.default_rng(4)
    split = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    split[300:] /= 4
    gappy = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    gappy[100:400:2] = np.nan
    steady = 80 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    path = tmp_path / "flawed.csv"
    pd.DataFrame(
        {"date": days, "SPLIT": split, "GAPPY": gappy, "OK": steady}
    ).to_csv(path, index=False)
    config = tmp_path / "cfg.yaml"
    config.write_text("optimizer: min_variance\n")
    return path, config


def _codes(payload) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for finding in payload["data_quality"]["findings"]:
        found.setdefault(finding["code"], set()).add(finding["asset"])
    return found


@pytest.mark.parametrize("command", ["optimize", "backtest", "check"])
def test_every_payload_carries_the_data_quality_findings(
    capsys, tmp_path, flawed_panel, command
):
    """A run on flawed data exited 0 and its JSON never mentioned the flaws.

    The console printed "Data error — GAPPY ..." on stderr; a machine reading
    stdout saw a clean-looking book.
    """
    path, config = flawed_panel
    argv = [command, "--config", str(config), "--prices", str(path), "--json"]
    if command == "optimize":
        argv += ["--output", str(tmp_path / "o.xlsx")]
    if command == "backtest":
        argv += ["--lookback", "252", "--rebalance-every", "63"]

    _, payload = _run(capsys, argv)

    quality = payload["data_quality"]
    assert quality["usable"] is False
    assert quality["clean"] is False
    codes = _codes(payload)
    assert codes["interior_gaps"] == {"GAPPY"}
    assert codes["extreme_returns"] == {"SPLIT"}
    gaps = next(f for f in quality["findings"] if f["code"] == "interior_gaps")
    assert gaps["severity"] == "error"
    assert gaps["message"] and gaps["suggestion"]
    # check's original keys keep their meaning.
    assert isinstance(quality["errors"], list) and quality["errors"]
    assert quality["n_common_periods"] > 0


def test_a_clean_panel_reports_no_findings_rather_than_no_key(capsys, tmp_path):
    _, payload = _run(
        capsys,
        ["optimize", "--config", CONFIG, "--sample", "--output", str(tmp_path / "o.xlsx"), "--json"],
    )
    assert payload["data_quality"]["usable"] is True
    assert isinstance(payload["data_quality"]["findings"], list)


@pytest.mark.parametrize("command", ["optimize", "backtest", "check"])
def test_the_resolved_ingest_window_is_in_the_payload(capsys, tmp_path, command):
    """An ingest with no end date ends today, and the document has to say so.

    The resolved window was hashed into the cache key and recorded nowhere a
    reader of the result could see it, so two runs a day apart produced
    different numbers from what looked like the same request.
    """
    import datetime as dt

    from optimization_engine.ingest.spec import IngestRequest

    config = tmp_path / "cfg.yaml"
    config.write_text("optimizer: min_variance\n")
    argv = [
        command, "--config", str(config), "--provider", "sample",
        "--identifiers", "US_Equity,Gold,US_Treasuries", "--ingest-period", "3y",
        "--json",
    ]
    if command == "optimize":
        argv += ["--output", str(tmp_path / "o.xlsx")]
    if command == "backtest":
        argv += ["--lookback", "252", "--rebalance-every", "63"]

    _, payload = _run(capsys, argv)

    window = payload["ingest"]
    expected = IngestRequest(
        identifiers=("US_Equity", "Gold", "US_Treasuries"), provider="sample", period="3y"
    )
    assert window["provider"] == "sample"
    assert window["end"] == dt.date.today().isoformat()
    assert window["start"] == expected.start.isoformat()
    assert window["interval"] == "1d"
    assert window["fingerprint"] == expected.fingerprint()
    assert window["from_cache"] is False


def test_a_panel_that_was_not_ingested_says_so(capsys, tmp_path):
    _, payload = _run(
        capsys,
        ["optimize", "--config", CONFIG, "--sample", "--output", str(tmp_path / "o.xlsx"), "--json"],
    )
    assert payload["ingest"] is None
