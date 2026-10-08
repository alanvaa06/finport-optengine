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


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    "argv",
    [
        ["optimize", "--config", CONFIG, "--sample", "--json"],
        ["backtest", "--config", CONFIG, "--sample", "--json"],
    ],
)
def test_a_raised_exception_still_emits_json(capsys, monkeypatch, argv):
    """The half of the contract that a returned exit code does not cover.

    `_emit_json` originally caught only a command that *returned* non-zero.
    A command that *raised* printed a traceback and left stdout empty, which
    is exactly the "no output versus output I could not parse" ambiguity this
    mode exists to remove. The payload has to survive the exception, not just
    the failure.

    The exception is injected rather than provoked. An unreadable config used
    to be the cheapest way in, and it is now an input error with exit 2 —
    which is what this branch is *not* for. It is the net under a defect.
    """
    monkeypatch.setattr("optimization_engine.cli.run_engine", _boom)
    code, payload = _run(capsys, argv)
    assert code == 1
    assert payload["exit_code"] == code
    assert payload["schema_version"] == SCHEMA_VERSION
    # The type and message, so a caller can tell one failure from another
    # without scraping the traceback.
    assert payload["error"] == "RuntimeError: boom"


def test_the_traceback_survives_on_stderr(capsys, monkeypatch):
    """Catching the exception must not cost the human their diagnosis."""
    monkeypatch.setattr("optimization_engine.cli.run_engine", _boom)
    main(["optimize", "--config", CONFIG, "--sample", "--json"])
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert "Traceback" in captured.err
    assert "RuntimeError: boom" in captured.err


@pytest.mark.parametrize("command", ["optimize", "check", "backtest"])
def test_a_missing_config_is_an_input_error_not_a_crash(capsys, command):
    """`docs/ERRORS.md` promises exit 2 and no traceback for a bad config.

    The config was loaded outside every handler, so a missing file fell into
    `_emit_json`'s net and came back as exit 1 with a traceback — the code
    that means "the engine ran and the answer is no".
    """
    code = main([command, "--config", "/no/such/file.yaml", "--sample", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert payload["exit_code"] == 2
    # Still enough to tell a missing file from a malformed one.
    assert "FileNotFoundError" in payload["error"]
    assert "file.yaml" in payload["error"]
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A misspelt key is refused by name.
        ("max_tracking_eror: 0.03\n", "max_tracking_eror"),
        # Not YAML at all.
        ("bounds: [unclosed\n", "not valid YAML"),
        # A known key holding a value of the wrong type.
        ("ewma_lambda: not-a-number\n", "ValueError"),
    ],
)
def test_a_malformed_config_is_exit_2_with_the_reason(capsys, tmp_path, text, expected):
    config = tmp_path / "bad.yaml"
    config.write_text(text)
    code = main(["check", "--config", str(config), "--sample", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert expected in payload["error"]
    assert "Traceback" not in captured.err


def test_a_missing_price_file_is_exit_2_with_the_reason(capsys, tmp_path):
    missing = tmp_path / "nope.csv"
    code = main(["optimize", "--config", CONFIG, "--prices", str(missing), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert "nope.csv" in payload["error"]
    assert "Traceback" not in captured.err


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


@pytest.mark.parametrize("command", ["optimize", "check", "backtest"])
def test_no_data_source_is_refused_rather_than_filled_with_the_sample(capsys, command):
    """Without --prices the commands used to solve on the synthetic panel.

    `optimize --config c.yaml --json` exited 0 with plausible weights and no
    word anywhere — stdout, stderr or payload — that none of it was market
    data. A forgotten flag is the most likely way to get there.
    """
    code, payload = _run(capsys, [command, "--config", CONFIG, "--json"])
    assert code == 2
    assert "--prices" in payload["error"] and "--sample" in payload["error"]


def test_two_data_sources_are_refused(capsys, tmp_path):
    # --sample used to win silently over --prices, which is the same
    # substitution reached by naming too much rather than too little.
    csv = tmp_path / "prices.csv"
    csv.write_text("date,A\n2024-01-01,1\n")
    code, payload = _run(
        capsys,
        ["optimize", "--config", CONFIG, "--sample", "--prices", str(csv), "--json"],
    )
    assert code == 2
    assert "--prices" in payload["error"] and "--sample" in payload["error"]


@pytest.mark.parametrize(
    "argv",
    [
        ["check", "--config", CONFIG, "--sample", "--json"],
        ["optimize", "--config", CONFIG, "--sample", "--json"],
        ["backtest", "--config", CONFIG, "--sample",
         "--lookback", "504", "--rebalance-every", "252", "--json"],
    ],
)
def test_every_payload_names_its_data_source(capsys, argv):
    code, payload = _run(capsys, argv)
    assert code == 0
    assert payload["data_source"] == {
        "kind": "sample",
        "synthetic": True,
        "path": None,
        "provider": None,
        "identifiers": None,
    }


def test_a_price_file_is_named_as_the_source(capsys, tmp_path):
    from optimization_engine.data.loader import sample_dataset

    csv = tmp_path / "prices.csv"
    sample_dataset(n_periods=400).to_csv(csv)
    code, payload = _run(capsys, ["check", "--config", CONFIG, "--prices", str(csv), "--json"])
    assert code == 0
    assert payload["data_source"]["kind"] == "file"
    assert payload["data_source"]["path"] == str(csv)
    assert payload["data_source"]["synthetic"] is False


def _refusal_argv(case: str, tmp_path: Path) -> list[str]:
    """A command line that ends in each of the refusals the CLI returns."""
    import yaml

    impossible = tmp_path / "impossible.yaml"
    impossible.write_text("bounds:\n  US_Equity: [0.6, 1.0]\n  Cash: [0.6, 1.0]\n")
    data = yaml.safe_load(Path(CONFIG).read_text())
    assets = list(data["expected_returns"])
    data["optimizer"] = {"name": "hrp"}
    data["previous_weights"] = {a: (1.0 if a == assets[0] else 0.0) for a in assets}
    data["turnover_limit"] = 0.01
    unhonoured = tmp_path / "hrp.yaml"
    unhonoured.write_text(yaml.safe_dump(data))
    shocks = tmp_path / "shocks.yaml"
    shocks.write_text(
        yaml.safe_dump({"shocks": [{"name": "bad", "returns": {"NOT_IN_PANEL": -0.1}}]})
    )
    universe = tmp_path / "universe.yaml"
    universe.write_text(
        yaml.safe_dump({"rules": [{"kind": "rolling", "panel": "returns", "windwo": 3}]})
    )
    short = ["--lookback", "252", "--rebalance-every", "252"]
    return {
        "solver_failure": ["optimize", "--config", str(impossible), "--sample"],
        "infeasible_under_strict": [
            "optimize", "--config", str(impossible), "--sample", "--strict",
        ],
        "mandate_violation": [
            "optimize", "--config", str(unhonoured), "--sample", "--strict-mandate",
        ],
        "unknown_optimizer": ["describe", "nope"],
        "optimize_stress": [
            "optimize", "--config", CONFIG, "--sample", "--stress", str(shocks),
        ],
        "unreadable_stress_file": [
            "optimize", "--config", CONFIG, "--sample",
            "--stress", str(tmp_path / "nowhere.yaml"),
        ],
        "backtest_stress": [
            "backtest", "--config", CONFIG, "--sample", *short, "--stress", str(shocks),
        ],
        "unreadable_universe": [
            "backtest", "--config", CONFIG, "--sample", *short,
            "--universe", str(universe),
        ],
        "bad_spec": [
            "backtest", "--config", CONFIG, "--sample", "--execution-lag", "-1",
        ],
    }[case] + ["--json"]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("solver_failure", "Optimization failed"),
        ("infeasible_under_strict", "Minimum weights sum to"),
        ("mandate_violation", "this method did not satisfy it"),
        ("unknown_optimizer", "Unknown optimizer 'nope'"),
        ("optimize_stress", "NOT_IN_PANEL"),
        ("unreadable_stress_file", "Could not read stress scenarios"),
        ("backtest_stress", "NOT_IN_PANEL"),
        ("unreadable_universe", "windwo"),
        ("bad_spec", "execution_lag"),
    ],
)
def test_every_refusal_carries_its_reason_into_the_payload(
    capsys, tmp_path, case, expected
):
    """A returned 2 used to emit "the command exited before producing a result".

    True, and useless to a caller: the infeasible mandate, the solver that
    gave up, the breached limit and the unknown method name were all printed
    to stderr and none of them reached the document the caller parses.
    """
    code, payload = _run(capsys, _refusal_argv(case, tmp_path))
    assert code == 2
    assert payload["exit_code"] == 2
    assert expected in payload["error"], payload["error"]


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
