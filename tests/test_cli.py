"""CLI behaviour: validation, reporting, and exit codes."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.cli import main
from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.reporting.exporters import (
    _unique_sheet_name,
    run_sheets,
    write_excel_report,
)

EXAMPLE_CONFIG = ROOT / "config" / "example_multi_asset.yaml"


@pytest.fixture
def feasible_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def infeasible_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    # Cap everything at 5%: the caps cannot sum to a fully-invested book.
    data["bounds"] = {a: [0.0, 0.05] for a in data["expected_returns"]}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_list_optimizers_describes_each_method(capsys):
    assert main(["list-optimizers"]) == 0
    out = capsys.readouterr().out
    assert "mean_variance" in out
    # Not just a bare list of names any more.
    assert "variance" in out.lower() and len(out.splitlines()) >= 10


def test_describe_reports_assumptions(capsys):
    assert main(["describe", "hrp"]) == 0
    out = capsys.readouterr().out
    assert "Hierarchical Risk Parity" in out
    assert "Assumptions:" in out
    assert "Group budgets cannot be enforced" in out


def test_describe_rejects_an_unknown_method(capsys):
    assert main(["describe", "not_a_method"]) == 2
    assert "Available" in capsys.readouterr().err


def test_check_passes_on_a_feasible_config(feasible_config, capsys):
    assert main(["check", "--config", str(feasible_config), "--sample"]) == 0
    out = capsys.readouterr().out
    assert "Ready to optimize." in out
    assert "Reachable expected return" in out
    assert "T/N" in out


def test_check_fails_and_names_the_constraint(infeasible_config, capsys):
    # 2 is the mandate-is-impossible code; 1 stays for unusable data.
    assert main(["check", "--config", str(infeasible_config), "--sample"]) == 2
    captured = capsys.readouterr()
    assert "cannot reach 100% invested" in captured.out
    assert "Raise the caps" in captured.out
    assert "Not ready to optimize." in captured.err


def test_a_misspelt_config_key_is_exit_2_not_a_traceback(tmp_path, capsys):
    # The human-mode half of the contract: the same failure without --json
    # used to leak a traceback and exit 1.
    config = tmp_path / "typo.yaml"
    config.write_text("max_tracking_eror: 0.03\n")
    assert main(["check", "--config", str(config), "--sample"]) == 2
    err = capsys.readouterr().err
    assert "max_tracking_eror" in err
    assert "Traceback" not in err


def test_optimize_writes_a_report_with_provenance(feasible_config, tmp_path, capsys):
    out_path = tmp_path / "report.xlsx"
    assert (
        main(
            [
                "optimize", "--config", str(feasible_config), "--sample",
                "--output", str(out_path),
            ]
        )
        == 0
    )
    assert out_path.exists()
    sheets = pd.read_excel(out_path, sheet_name=None)
    assert {"weights", "assumptions", "risk_decomposition", "data_quality"} <= set(sheets)
    out = capsys.readouterr().out
    assert "effective N" in out


def test_strict_optimize_refuses_an_infeasible_config(
    infeasible_config, tmp_path, capsys
):
    out_path = tmp_path / "report.xlsx"
    code = main(
        [
            "optimize", "--config", str(infeasible_config), "--sample",
            "--strict", "--output", str(out_path),
        ]
    )
    assert code == 2
    assert "cannot reach 100% invested" in capsys.readouterr().err
    assert not out_path.exists()


def test_optimize_with_walk_forward_reports_degradation(
    feasible_config, tmp_path, capsys
):
    out_path = tmp_path / "wf.xlsx"
    assert (
        main(
            [
                "optimize", "--config", str(feasible_config), "--sample",
                "--walk-forward", "--lookback", "504", "--rebalance-every", "126",
                "--cost-bps", "10", "--output", str(out_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Walk-forward" in out and "out of sample" in out
    sheets = pd.read_excel(out_path, sheet_name=None)
    assert "in_vs_out_of_sample" in sheets


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


def test_run_sheets_carries_the_evidence():
    returns = prices_to_returns(sample_dataset(252 * 4, seed=5))
    cfg = EngineConfig(
        bounds={a: [0.0, 0.3] for a in returns.columns},
        optimizer=OptimizerSpec(name="min_variance"),
    )
    run = run_engine(returns, cfg, build_frontier=True, n_frontier_points=6)
    sheets = run_sheets(run)
    for expected in (
        "weights", "assumptions", "risk_decomposition",
        "portfolio_diagnostics", "estimation_diagnostics",
        "frontier_summary", "frontier_weights",
    ):
        assert expected in sheets, expected


def test_long_sheet_names_do_not_collide(tmp_path):
    long_a = "a_very_long_sheet_name_that_exceeds_limits_one"
    long_b = "a_very_long_sheet_name_that_exceeds_limits_two"
    frame = pd.DataFrame({"x": [1, 2]})
    out = write_excel_report(tmp_path / "r.xlsx", {long_a: frame, long_b: frame})
    sheets = pd.read_excel(out, sheet_name=None)
    assert len(sheets) == 2, "truncated names silently overwrote each other"


def test_unique_sheet_name_suffixes_collisions():
    used = {"abc"}
    assert _unique_sheet_name("abc", used) != "abc"
    assert _unique_sheet_name("xyz", used) == "xyz"


def test_write_excel_report_skips_none_and_promotes_series(tmp_path):
    out = write_excel_report(
        tmp_path / "r.xlsx",
        {"a": pd.Series([1.0, 2.0], name="v"), "b": None},
    )
    sheets = pd.read_excel(out, sheet_name=None)
    assert set(sheets) == {"a"}


# ---------------------------------------------------------------------------
# Benchmark flags
# ---------------------------------------------------------------------------


def test_optimize_reports_against_a_benchmark(feasible_config, tmp_path, capsys):
    out_path = tmp_path / "bench.xlsx"
    code = main(
        [
            "optimize", "--config", str(feasible_config), "--sample",
            "--benchmark", "equal_weight", "--output", str(out_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "vs Equal weight (1/N)" in out
    assert "IR" in out and "T.E." in out
    sheets = pd.read_excel(out_path, sheet_name=None)
    assert "benchmark" in sheets
    assert "performance_relative" in sheets


def test_optimize_accepts_an_asset_name_as_the_benchmark(
    feasible_config, tmp_path, capsys
):
    code = main(
        [
            "optimize", "--config", str(feasible_config), "--sample",
            "--benchmark", "US_Equity", "--output", str(tmp_path / "a.xlsx"),
        ]
    )
    assert code == 0
    assert "vs US_Equity" in capsys.readouterr().out


def test_optimize_rejects_an_unknown_benchmark(feasible_config, tmp_path, capsys):
    code = main(
        [
            "optimize", "--config", str(feasible_config), "--sample",
            "--benchmark", "NOT_A_THING", "--output", str(tmp_path / "b.xlsx"),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "neither a benchmark kind nor an asset" in err


def test_a_tracking_error_budget_binds_from_the_command_line(
    feasible_config, tmp_path
):
    import numpy as np

    data = yaml.safe_load(feasible_config.read_text())
    # The example config targets a return the tight budget cannot reach;
    # this test is about the budget, so the target comes off.
    data["optimizer"].pop("target_return", None)
    path = tmp_path / "no_target.yaml"
    path.write_text(yaml.safe_dump(data))

    out_path = tmp_path / "te.xlsx"
    assert (
        main(
            [
                "optimize", "--config", str(path), "--sample",
                "--benchmark", "equal_weight",
                "--max-tracking-error", "0.02",
                "--output", str(out_path),
            ]
        )
        == 0
    )
    sheets = pd.read_excel(out_path, sheet_name=None)
    weights = sheets["benchmark_weights"].set_index(
        sheets["benchmark_weights"].columns[0]
    )
    active = weights["active_weight"].values
    cov = sheets["cov_matrix"].set_index(sheets["cov_matrix"].columns[0])
    realized = float(np.sqrt(max(active @ cov.values @ active, 0.0)))
    assert realized <= 0.02 + 1e-4


def test_an_unreachable_target_under_a_budget_names_the_cause(
    feasible_config, tmp_path, capsys
):
    # The example config targets 7%; a 4% tracking-error budget puts that
    # out of reach, and the message should say so rather than "infeasible".
    code = main(
        [
            "optimize", "--config", str(feasible_config), "--sample",
            "--benchmark", "equal_weight", "--max-tracking-error", "0.04",
            "--output", str(tmp_path / "c.xlsx"),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "Target return" in err
    assert "tracking-error or active-share budget is in force" in err


def test_backtest_prices_the_trading_and_states_its_caveats(
    feasible_config, capsys
):
    assert (
        main(
            [
                "backtest",
                "--config", str(feasible_config),
                "--sample",
                "--lookback", "252",
                "--rebalance-every", "252",
                "--commission-bps", "8",
                "--slippage-bps", "4",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "out-of-sample" in out
    assert "bps of notional" in out
    # The run modelled costs and a lag, so those caveats are gone; nobody
    # supplied a trial count, so that one stays.
    assert "modelled as free" not in out
    assert "undeflated" in out


def test_backtest_sweep_counts_the_trials_it_ran(feasible_config, tmp_path, capsys):
    output = tmp_path / "backtest.xlsx"
    assert (
        main(
            [
                "backtest",
                "--config", str(feasible_config),
                "--sample",
                "--lookback", "252",
                "--rebalance-every", "252",
                "--commission-bps", "10",
                "--sweep", "optimizer.name=min_variance,equal_weight",
                "--output", str(output),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "2 cells" in out
    assert "trial(s)" in out, "the deflation must name the trial count"
    assert output.exists()
    assert "sweep" in pd.ExcelFile(output).sheet_names


def test_backtest_holdout_flags_a_second_look(feasible_config, tmp_path, capsys):
    audit = tmp_path / "audit.jsonl"
    argv = [
        "backtest",
        "--config", str(feasible_config),
        "--sample",
        "--lookback", "252",
        "--rebalance-every", "252",
        "--holdout", "2024-01-01",
        "--audit-log", str(audit),
    ]
    assert main(argv) == 0
    assert "First look" in capsys.readouterr().out
    assert main(argv) == 0
    assert "evaluated on the holdout before" in capsys.readouterr().out
    assert len(audit.read_text().splitlines()) == 2


def test_backtest_rejects_a_malformed_sweep(feasible_config, capsys):
    assert (
        main(
            [
                "backtest",
                "--config", str(feasible_config),
                "--sample",
                "--lookback", "252",
                "--rebalance-every", "252",
                "--sweep", "optimizer.name",
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "Sweep skipped" in err and "PATH=V1,V2" in err


# ---------------------------------------------------------------------------
# check and optimize see the same mandate (review fix E1)
# ---------------------------------------------------------------------------


def test_check_and_optimize_accept_the_same_config_without_expected_returns(
    tmp_path, capsys
):
    # A method that needs no mu, and a config that supplies none: the vector
    # is estimated from history. `check` called this ready while `optimize`
    # refused it with "no expected returns matching the price columns".
    config = tmp_path / "rp.yaml"
    config.write_text("optimizer:\n  name: risk_parity\n")
    assert main(["check", "--config", str(config), "--sample"]) == 0
    assert "Ready to optimize." in capsys.readouterr().out
    out_path = tmp_path / "rp.xlsx"
    assert (
        main(["optimize", "--config", str(config), "--sample", "--output", str(out_path)])
        == 0
    )
    assert "no expected returns" not in capsys.readouterr().err
    assert out_path.exists()


def test_check_honours_the_benchmark_flags_optimize_does(feasible_config, capsys):
    # A tracking-error budget tight enough to be unreachable has to fail the
    # pre-flight the same way it fails the solve, or `check` is validating a
    # different mandate.
    code = main(
        [
            "check", "--config", str(feasible_config), "--sample",
            "--benchmark", "equal_weight", "--max-tracking-error", "0.0001",
        ]
    )
    captured = capsys.readouterr()
    # 2, not 1: an impossible mandate and unusable data are different
    # problems and a script should be able to tell them apart.
    assert code == 2, captured.out + captured.err
    assert "Not ready to optimize." in captured.err


def test_backtest_no_longer_seeds_zero_expected_returns(tmp_path, capsys):
    # A return target with no explicit mu was infeasible before the walk-
    # forward even started, because the seed solve saw a vector of zeros
    # rather than the history-derived one every window would then use.
    config = tmp_path / "target.yaml"
    config.write_text(
        "optimizer:\n  name: mean_variance\n  target_return: 0.04\n"
    )
    assert main(["backtest", "--config", str(config), "--sample", "--lookback", "504",
                 "--rebalance-every", "252"]) == 0
    assert "initial solve failed" not in capsys.readouterr().err


def test_accept_inaccurate_reaches_the_config_on_every_solving_subcommand(
    feasible_config, monkeypatch
):
    """The flag exists on all three, and each one writes it onto the config.

    ``optimize`` and ``backtest`` pass it into the solve through the optimizer
    they build; ``check`` has no optimizer and opens the scope around its own
    feasibility LPs. All three read it off the same field, so a config file
    that sets ``accept_inaccurate: true`` and a command line that passes
    ``--accept-inaccurate`` cannot disagree.
    """
    from optimization_engine import cli

    seen: list[bool] = []
    real = cli._apply_estimator_flags

    def spy(config, args):
        real(config, args)
        seen.append(config.optimizer.accept_inaccurate)

    monkeypatch.setattr(cli, "_apply_estimator_flags", spy)

    assert main(["check", "--config", str(feasible_config), "--sample"]) == 0
    assert main(
        ["check", "--config", str(feasible_config), "--sample", "--accept-inaccurate"]
    ) == 0
    assert seen == [False, True]

    for command in ("optimize", "backtest"):
        parser_args = [command, "--config", str(feasible_config), "--sample"]
        # Only the parser is exercised here: running a full backtest twice
        # more to re-check one boolean is not worth the minute it costs.
        parsed = cli._build_parser().parse_args([*parser_args, "--accept-inaccurate"])
        assert parsed.accept_inaccurate is True
        assert cli._build_parser().parse_args(parser_args).accept_inaccurate is False


def test_accept_inaccurate_does_not_undo_a_config_that_asked_for_it(
    tmp_path, feasible_config
):
    """Its absence on the command line is silence, not a veto."""
    import yaml

    from optimization_engine import cli
    from optimization_engine.config import load_config

    data = yaml.safe_load(feasible_config.read_text())
    data["optimizer"]["accept_inaccurate"] = True
    path = tmp_path / "loose.yaml"
    path.write_text(yaml.safe_dump(data))

    config = load_config(path)
    args = cli._build_parser().parse_args(
        ["optimize", "--config", str(path), "--sample"]
    )
    cli._apply_estimator_flags(config, args)
    assert config.optimizer.accept_inaccurate is True


# ---------------------------------------------------------------------------
# --stress
# ---------------------------------------------------------------------------


@pytest.fixture
def shocks_file(tmp_path: Path) -> Path:
    """One scenario over two names the shipped example config holds."""
    path = tmp_path / "shocks.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "shocks": [
                    {
                        "name": "risk_off",
                        "returns": {"US_Equity": -0.20, "Gold": 0.05},
                        "covariance_scale": 2.0,
                        "notes": "a 2008-shaped day",
                    }
                ],
            }
        )
    )
    return path


def test_optimize_stresses_the_book_it_solved(feasible_config, shocks_file, capsys):
    assert (
        main(
            ["optimize", "--config", str(feasible_config), "--sample",
             "--stress", str(shocks_file)]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Loaded 1 stress scenario(s)" in out
    assert "risk_off" in out
    assert "of book value" in out
    assert "a 2008-shaped day" in out


def test_backtest_stresses_the_book_the_walk_forward_ended_on(
    feasible_config, shocks_file, capsys
):
    assert (
        main(
            ["backtest", "--config", str(feasible_config), "--sample",
             "--lookback", "252", "--rebalance-every", "252",
             "--stress", str(shocks_file)]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "risk_off" in out and "of book value" in out


def test_an_unreadable_stress_file_is_an_exit_code_not_a_traceback(
    feasible_config, tmp_path, capsys
):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"shocks": [{"name": "x", "retruns": {"A": -0.1}}]}))
    assert (
        main(["optimize", "--config", str(feasible_config), "--sample",
              "--stress", str(bad)])
        == 2
    )
    err = capsys.readouterr().err
    assert "Could not read stress scenarios" in err and "retruns" in err

    missing = tmp_path / "nowhere.yaml"
    assert (
        main(["optimize", "--config", str(feasible_config), "--sample",
              "--stress", str(missing)])
        == 2
    )
    assert "Could not read stress scenarios" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --strict-mandate
# ---------------------------------------------------------------------------


def test_strict_mandate_is_the_flag_form_of_the_config_key(feasible_config):
    """Both commands carry it, and it is one-way: absent means silent."""
    from optimization_engine import cli
    from optimization_engine.config import load_config

    for command in ("optimize", "backtest"):
        args = cli._build_parser().parse_args(
            [command, "--config", str(feasible_config), "--sample", "--strict-mandate"]
        )
        assert args.strict_mandate is True
        config = load_config(feasible_config)
        cli._apply_estimator_flags(config, args)
        assert config.strict_mandate is True

    # Absent, it does not switch off a config that already asked to refuse.
    config = load_config(feasible_config)
    config.strict_mandate = True
    plain = cli._build_parser().parse_args(
        ["optimize", "--config", str(feasible_config), "--sample"]
    )
    assert plain.strict_mandate is False
    cli._apply_estimator_flags(config, plain)
    assert config.strict_mandate is True


def test_a_refused_book_exits_two_with_the_breach_named(tmp_path, capsys):
    """HRP under a turnover budget it cannot honour, refused rather than reported.

    Without the CLI's own handler this is the one failure ``--strict-mandate``
    exists to produce and the only one that would leak a traceback, since
    ``MandateViolationError`` is a ``ValueError`` and the ``except
    SolverFailure`` clause never sees it.
    """
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    assets = list(data["expected_returns"])
    data["optimizer"] = {"name": "hrp"}
    data["previous_weights"] = {a: (1.0 if a == assets[0] else 0.0) for a in assets}
    data["turnover_limit"] = 0.01
    path = tmp_path / "hrp.yaml"
    path.write_text(yaml.safe_dump(data))

    assert (
        main(["optimize", "--config", str(path), "--sample", "--strict-mandate"]) == 2
    )
    err = capsys.readouterr().err
    assert "breach the mandate" in err
    assert "Turnover" in err
    assert "Traceback" not in err
    assert "this method did not satisfy it" in err


def test_backtests_anchor_solve_is_refused_without_a_traceback(tmp_path, capsys):
    """The one refusal a backtest cannot record as a failed window.

    Every per-window refusal is caught inside the walk-forward. The anchor
    solve ``_cmd_backtest`` makes before it is outside that loop, so it needs
    its own handler or the flag leaks a traceback and exits 1.
    """
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    assets = list(data["expected_returns"])
    data["optimizer"] = {"name": "hrp"}
    data["previous_weights"] = {a: (1.0 if a == assets[0] else 0.0) for a in assets}
    data["turnover_limit"] = 0.01
    path = tmp_path / "hrp-backtest.yaml"
    path.write_text(yaml.safe_dump(data))

    assert (
        main(
            ["backtest", "--config", str(path), "--sample",
             "--lookback", "252", "--rebalance-every", "252", "--strict-mandate"]
        )
        == 2
    )
    err = capsys.readouterr().err
    assert "The initial solve was refused" in err
    assert "Turnover" in err
    assert "Traceback" not in err


def test_a_shock_outside_the_panel_is_refused_on_both_commands(
    feasible_config, tmp_path, capsys
):
    """Refused, not zeroed — the same posture as an out-of-universe BL view."""
    path = tmp_path / "oops.yaml"
    path.write_text(
        yaml.safe_dump({"shocks": [{"name": "bad", "returns": {"NOT_IN_PANEL": -0.1}}]})
    )
    for argv in (
        ["optimize", "--config", str(feasible_config), "--sample",
         "--stress", str(path)],
        ["backtest", "--config", str(feasible_config), "--sample",
         "--lookback", "252", "--rebalance-every", "252", "--stress", str(path)],
    ):
        assert main(argv) == 2, argv
        err = capsys.readouterr().err
        assert "Stress test failed" in err and "NOT_IN_PANEL" in err
        assert "Traceback" not in err


# ---------------------------------------------------------------------------
# --universe
# ---------------------------------------------------------------------------


@pytest.fixture
def universe_file(tmp_path: Path) -> Path:
    """A screen over the run's own return panel — no data files beside it."""
    path = tmp_path / "universe.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "combine": "all",
                "rules": [
                    {
                        "kind": "rolling", "panel": "returns", "window": 21,
                        "agg": "count", "op": ">=", "value": 21,
                        "name": "printed on all of the last 21 sessions",
                    }
                ],
            }
        )
    )
    return path


def test_backtest_reads_a_universe_and_says_what_the_policy_decided(
    feasible_config, universe_file, capsys
):
    assert (
        main(
            ["backtest", "--config", str(feasible_config), "--sample",
             "--lookback", "252", "--rebalance-every", "252",
             "--universe", str(universe_file)]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "printed on all of the last 21 sessions" in captured.out
    # The count of what the default policy — not a screen — decided, on stderr
    # with the alignment log and for the same reason.
    assert "not evaluable" in captured.err
    assert "'exclude' policy" in captured.err
    assert "reads them as ineligible" in captured.err


def test_the_universe_policy_defaults_to_exclude_and_raise_stops_the_run(
    feasible_config, universe_file, capsys
):
    from optimization_engine import cli

    parsed = cli._build_parser().parse_args(
        ["backtest", "--config", str(feasible_config), "--sample"]
    )
    assert parsed.universe_policy == "exclude"

    assert (
        main(
            ["backtest", "--config", str(feasible_config), "--sample",
             "--lookback", "252", "--rebalance-every", "252",
             "--universe", str(universe_file), "--universe-policy", "raise"]
        )
        == 2
    )
    err = capsys.readouterr().err
    assert "Universe failed" in err
    assert "not evaluable" in err
    assert "Traceback" not in err


def test_the_universe_reaches_the_json_payload(
    feasible_config, universe_file, tmp_path, capsys
):
    """``meta.notes["universe"]`` with its *values*, not just its keys."""
    import json

    assert (
        main(
            ["backtest", "--json", "--config", str(feasible_config), "--sample",
             "--lookback", "252", "--rebalance-every", "252",
             "--universe", str(universe_file), "--delisting-grace", "5"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    notes = payload["notes"]

    universe = notes["universe"]
    assert universe["policy"] == "exclude"
    assert universe["n_decisions"] >= 1
    assert universe["min_breadth"] >= 1
    assert all(isinstance(v, int) for v in universe["breadth"].values())
    assert universe["unknown_assets"] == []
    # The separate opt-in lands beside it, and only because it was asked for.
    assert notes["delisting_grace"] == 5
    assert "delistings" in notes


def test_delisting_grace_is_absent_from_the_notes_unless_asked_for(
    feasible_config, capsys
):
    import json

    assert (
        main(
            ["backtest", "--json", "--config", str(feasible_config), "--sample",
             "--lookback", "252", "--rebalance-every", "252"]
        )
        == 0
    )
    notes = json.loads(capsys.readouterr().out)["notes"]
    assert "delisting_grace" not in notes
    assert "universe" not in notes


def test_an_unreadable_universe_file_is_an_exit_code_not_a_traceback(
    feasible_config, tmp_path, capsys
):
    bad = tmp_path / "bad-universe.yaml"
    bad.write_text(
        yaml.safe_dump(
            {"rules": [{"kind": "rolling", "panel": "returns", "windwo": 3,
                        "agg": "mean", "op": ">", "value": 0.0}]}
        )
    )
    assert (
        main(["backtest", "--config", str(feasible_config), "--sample",
              "--universe", str(bad)])
        == 2
    )
    err = capsys.readouterr().err
    assert "Could not read the universe" in err and "windwo" in err
    assert "Traceback" not in err


def test_a_universe_over_a_characteristic_panel_comes_in_through_the_rules_file(
    feasible_config, tmp_path, capsys
):
    """The ADV screen the CLI has no other route for.

    ``_prepare_inputs`` loads prices and nothing else, so the panel's path is
    written into the rules file and resolved against that file's directory.
    """
    import json

    from optimization_engine.data.loader import prices_to_returns, sample_dataset

    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    assets = list(data["expected_returns"])
    returns = prices_to_returns(sample_dataset(assets=assets))
    adv = pd.DataFrame(2.0e7, index=returns.index, columns=assets)
    thin = assets[-1]
    adv[thin] = 1.0e5
    adv.to_csv(tmp_path / "adv.csv")

    rules = tmp_path / "adv-universe.yaml"
    rules.write_text(
        yaml.safe_dump(
            {
                "panels": {"adv": "adv.csv"},
                "rules": [
                    {"kind": "threshold", "panel": "adv", "op": ">=",
                     "value": 1.0e7, "name": "ADV over $10m"}
                ],
            }
        )
    )

    assert (
        main(
            ["backtest", "--json", "--config", str(feasible_config), "--sample",
             "--lookback", "252", "--rebalance-every", "252",
             "--universe", str(rules)]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "Panels read from disk: adv" in captured.err
    universe = json.loads(captured.out)["notes"]["universe"]
    # Every name but the thin one, on every decision.
    assert set(universe["breadth"].values()) == {len(assets) - 1}


# ---------------------------------------------------------------------------
# A piped run on a legacy console encoding
# ---------------------------------------------------------------------------


def _run_on_cp1252(argv: list[str], cwd: Path):
    """Run the CLI in a child process whose stdio is cp1252, as on Windows.

    A subprocess rather than ``capsys`` because the failure lives in the real
    stream: pytest's capture encodes UTF-8 and would let every character
    through. ``PYTHONIOENCODING`` is what a piped run on a Windows machine
    with a legacy code page gets by default, and it reproduces the same thing
    on any platform.
    """
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SRC), env.get("PYTHONPATH", "")) if p
    )
    return subprocess.run(
        [sys.executable, "-m", "optimization_engine.cli", *argv],
        cwd=cwd,
        env=env,
        capture_output=True,
        timeout=600,
    )


@pytest.mark.parametrize(
    "argv",
    [
        # "α" in a summary line.
        ["list-optimizers"],
        # "δ" in one of the assumptions.
        ["describe", "black_litterman"],
    ],
)
def test_a_piped_run_on_a_legacy_encoding_does_not_crash(argv, tmp_path):
    proc = _run_on_cp1252(argv, tmp_path)
    assert b"Traceback" not in proc.stderr, proc.stderr.decode("cp1252", "replace")
    assert proc.returncode == 0


def test_optimize_writes_its_report_through_a_legacy_encoding(tmp_path):
    """The example config marks a binding bucket with an arrow cp1252 lacks.

    It used to raise ``UnicodeEncodeError`` half-way through the console
    report, so the workbook was never written and the exit code was 1.
    """
    out = tmp_path / "report.xlsx"
    proc = _run_on_cp1252(
        ["optimize", "--config", str(EXAMPLE_CONFIG), "--sample", "--output", str(out)],
        tmp_path,
    )
    assert b"Traceback" not in proc.stderr, proc.stderr.decode("cp1252", "replace")
    assert proc.returncode == 0
    assert out.exists()
    # The character is escaped rather than lost, so the line still says what
    # it said.
    assert b"\u2190binding" in proc.stdout
