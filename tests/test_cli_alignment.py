"""What the CLI does about assets with different histories, and what it says.

Every solving command starts from `_prepare_inputs`, which used to make the
panel rectangular with a bare `prices_to_returns(prices).dropna(how="any")`.
That single call is the most expensive silent decision in the pipeline: one
asset that listed three years after the rest truncates the *estimation
sample* for every other asset, and the only trace of it was an error string
claiming "after alignment" for a step that never happened.

These tests are about disclosure. For the case the fix is named after — one
asset that lists late — `align_panel(method="common")` keeps exactly the rows
the old call kept, so no number moves; what moves is that the run now names
what it dropped, on stderr for a person and under `alignment` for a parser,
and that `optimize`, `backtest` and `check` all inherit it because they share
one input path. (An *interior* gap is the one case where the sample itself
differs: aligning before differencing books the move across the gap as one
period instead of discarding two. That is the split `prices_to_returns`
documents and hands to the alignment step, and it is what the app has always
done.)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.cli import main  # noqa: E402
from optimization_engine.data.loader import prices_to_returns, sample_dataset  # noqa: E402
from optimization_engine.data.quality import align_panel  # noqa: E402

#: How many leading periods the late-listing asset is missing. Large enough
#: that a truncated sample is unmistakable in the window, small enough that
#: the remaining history still supports a walk-forward.
N_LATE = 500

LOOKBACK = 504


@pytest.fixture
def late_listing(tmp_path: Path):
    """A panel where one asset starts late, written where the CLI can read it.

    Returns the raw panel, the CSV path and a minimal config path. The config
    carries no `expected_returns` block on purpose: naming assets there would
    filter the universe before alignment and confuse which step dropped what.
    """
    prices = sample_dataset()
    prices.loc[prices.index[:N_LATE], prices.columns[0]] = float("nan")
    csv = tmp_path / "late_listing.csv"
    prices.to_csv(csv)
    config = tmp_path / "minimal.yaml"
    config.write_text("optimizer: risk_parity\nperiods_per_year: 252\n")
    return prices, csv, config


def _run_json(capsys, argv: list[str]) -> tuple[int, dict, str]:
    """Run the CLI in JSON mode, returning the code, the payload and stderr."""
    code = main(argv)
    captured = capsys.readouterr()
    # Strict: a stray print on stdout fails here rather than downstream.
    return code, json.loads(captured.out), captured.err


def test_cli_backtest_reports_alignment(capsys, late_listing):
    """The named acceptance test: the log is emitted, and it is load-bearing.

    Two claims, and the second is the one that matters. The log being
    non-empty only proves something was written; that the simulated track
    record starts exactly where an independent `align_panel` call puts it
    proves the log describes the sample the backtest actually ran on.
    """
    prices, csv, config = late_listing
    code, payload, _ = _run_json(
        capsys,
        [
            "backtest",
            "--config", str(config),
            "--prices", str(csv),
            "--lookback", str(LOOKBACK),
            "--rebalance-every", "252",
            "--json",
        ],
    )
    assert code == 0

    actions = payload["alignment"]
    assert actions, "a panel with a late listing has something to report"
    assert all(isinstance(a, str) for a in actions)
    assert str(N_LATE) in " ".join(actions), "the count of dropped dates is the point"

    # Reproduce the alignment through the public function. The CLI must have
    # done this and nothing else — a different method, or an extra drop, shows
    # up as a different log.
    aligned, expected_actions = align_panel(prices, method="common")
    assert actions == expected_actions

    # And the run started where that alignment says it does. `window.start`
    # is the first evaluated period, `LOOKBACK` rows into the aligned sample.
    returns = prices_to_returns(aligned).dropna(how="any")
    assert payload["window"]["start"] == str(returns.index[LOOKBACK])
    # The unaligned panel would have started this run roughly two years
    # earlier; the assertion above is only meaningful because it does not.
    assert returns.index[LOOKBACK] > prices.index[LOOKBACK]


@pytest.mark.parametrize("command", ["check", "backtest"])
def test_alignment_reaches_every_command_sharing_the_input_path(
    capsys, late_listing, command
):
    """`_prepare_inputs` is shared, so the disclosure cannot be per-command.

    `optimize` is covered separately — it needs an `--output` path — but the
    three commands build their inputs in one place and a fix that reached
    only the one named in the spec would be a fix in name only.
    """
    prices, csv, config = late_listing
    argv = [command, "--config", str(config), "--prices", str(csv), "--json"]
    if command == "backtest":
        argv += ["--lookback", str(LOOKBACK), "--rebalance-every", "252"]
    _, payload, _ = _run_json(capsys, argv)
    _, expected = align_panel(prices, method="common")
    assert payload["alignment"] == expected


def test_optimize_reports_alignment(capsys, late_listing, tmp_path):
    prices, csv, config = late_listing
    out = tmp_path / "report.xlsx"
    code, payload, _ = _run_json(
        capsys,
        [
            "optimize",
            "--config", str(config),
            "--prices", str(csv),
            "--output", str(out),
            "--json",
        ],
    )
    assert code == 0
    _, expected = align_panel(prices, method="common")
    assert payload["alignment"] == expected


def test_alignment_is_narrated_on_stderr_without_breaking_stdout(
    capsys, late_listing
):
    """The log is unconditional, and it is not on the parsed stream.

    There is no `--verbose` to hide behind: a truncated estimation sample is
    not an advanced-mode detail, and stdout under `--json` belongs to the
    caller's parser.
    """
    _, csv, config = late_listing
    _, payload, err = _run_json(
        capsys,
        ["check", "--config", str(config), "--prices", str(csv), "--json"],
    )
    assert "Alignment:" in err
    assert str(N_LATE) in err
    # Belt and braces: `_run_json` already parsed stdout, so a leaked line
    # would have raised. This pins that the narration is why we checked.
    assert "Alignment:" not in json.dumps(payload["alignment"])


def test_alignment_is_narrated_without_json_too(capsys, late_listing):
    """A human running `check` with no flags sees it as well."""
    _, csv, config = late_listing
    main(["check", "--config", str(config), "--prices", str(csv)])
    assert "Alignment:" in capsys.readouterr().err


def test_a_complete_panel_reports_an_empty_log(capsys):
    """Empty means "nothing was dropped", not "nobody looked".

    The key is always present, so a consumer can test the value rather than
    probing for the key — which is the contract `payloads` documents.
    """
    config = ROOT / "config" / "example_multi_asset.yaml"
    _, payload, _ = _run_json(
        capsys, ["check", "--config", str(config), "--sample", "--json"]
    )
    assert payload["alignment"] == []


def test_no_overlap_still_fails_with_a_message_that_is_now_true(capsys, tmp_path):
    """"No usable returns after alignment" used to describe a step never run.

    Two assets whose histories do not overlap at all leave nothing behind.
    The refusal is the same; what changed is that alignment really happened
    and the log says how much it removed before giving up.
    """
    prices = sample_dataset(n_periods=60)
    first, second = prices.columns[0], prices.columns[1]
    prices = prices[[first, second]].copy()
    prices.loc[prices.index[30:], first] = float("nan")
    prices.loc[prices.index[:30], second] = float("nan")
    csv = tmp_path / "disjoint.csv"
    prices.to_csv(csv)
    config = tmp_path / "minimal.yaml"
    config.write_text("optimizer: risk_parity\nperiods_per_year: 252\n")

    code, payload, err = _run_json(
        capsys, ["check", "--config", str(config), "--prices", str(csv), "--json"]
    )
    assert code == 2
    assert "No usable returns after alignment" in payload["error"]
    assert "Alignment:" in err


def test_the_script_aligns_the_way_the_cli_does(tmp_path):
    """`scripts/run_optimization.py` carried a verbatim copy of the same line.

    Run as a subprocess because that is the only way it is ever used, and
    because importing it would not exercise the `__main__` path where the
    duplicate lived.
    """
    prices = sample_dataset()
    prices.loc[prices.index[:N_LATE], prices.columns[0]] = float("nan")
    csv = tmp_path / "late_listing.csv"
    prices.to_csv(csv)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_optimization.py"),
            "--config", str(ROOT / "config" / "example_multi_asset.yaml"),
            "--prices", str(csv),
            "--output", str(tmp_path / "report.xlsx"),
        ],
        capture_output=True,
        # Both ends pinned to one encoding. Left to the defaults, the child
        # writes in whatever PYTHONIOENCODING says and this side decodes in the
        # locale's; on Windows those disagree (utf-8 vs cp1252) and the dash
        # in the message arrives as mojibake.
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        encoding="utf-8",
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "Alignment —" in proc.stderr
    assert str(N_LATE) in proc.stderr


def test_an_interior_gap_is_resolved_by_alignment_not_by_dropping_returns(
    capsys, tmp_path
):
    """The one case where the sample itself changes, pinned so it stays chosen.

    Differencing first leaves NaN at the gap *and* at the period after it,
    so `dropna(how="any")` quietly cost two observations for one missing
    price. Aligning first drops the gap date and books the move across it as
    a single period. `prices_to_returns` documents that split and hands the
    gap to "the alignment step"; this asserts the CLI now has one.
    """
    prices = sample_dataset(n_periods=300)
    gap_row, gap_col = 150, prices.columns[1]
    prices.loc[prices.index[gap_row], gap_col] = float("nan")
    csv = tmp_path / "interior_gap.csv"
    prices.to_csv(csv)
    config = tmp_path / "minimal.yaml"
    config.write_text("optimizer: risk_parity\nperiods_per_year: 252\n")

    _, payload, _ = _run_json(
        capsys, ["check", "--config", str(config), "--prices", str(csv), "--json"]
    )

    aligned, _ = align_panel(prices, method="common")
    kept = len(prices_to_returns(aligned).dropna(how="any"))
    discarded = len(prices_to_returns(prices).dropna(how="any"))
    assert kept == discarded + 1, "one missing price used to cost two returns"
    assert payload["covariance"]["n_observations"] == kept


# ---------------------------------------------------------------------------
# A point-in-time backtest walks the panel alignment would have cut
# ---------------------------------------------------------------------------

#: Rows of the point-in-time panel: one name lists late, one stops printing,
#: and one has a single missing price in the middle of its life.
PIT_ROWS = 700
PIT_LISTS = 150
PIT_DELISTS = 450
PIT_GAP = 300


@pytest.fixture
def listings_and_delistings(tmp_path: Path):
    """A panel whose common window is [late listing, first delisting]."""
    prices = sample_dataset(n_periods=PIT_ROWS)[
        ["US_Equity", "US_Treasuries", "Gold", "IG_Credit"]
    ].copy()
    prices.loc[prices.index[:PIT_LISTS], "IG_Credit"] = float("nan")
    prices.loc[prices.index[PIT_DELISTS:], "Gold"] = float("nan")
    prices.loc[prices.index[PIT_GAP], "US_Treasuries"] = float("nan")
    csv = tmp_path / "listings.csv"
    prices.to_csv(csv)
    config = tmp_path / "minimal.yaml"
    config.write_text("optimizer: equal_weight\nperiods_per_year: 252\n")
    return prices, csv, config


def test_delisting_grace_walks_the_whole_panel_not_the_common_window(
    capsys, listings_and_delistings
):
    """Alignment used to cut the run off on the day the first name delisted.

    ``--delisting-grace`` exists to sell a name that stopped printing, and it
    could never fire: ``align_panel("common")`` had already trimmed the panel
    to the dates every name printed on, so the backtest ended on the delisting
    and ``notes.delistings`` was always empty. The walk-forward now runs over
    every date; only a gap *inside* a name's life is aligned away, as before.
    """
    prices, csv, config = listings_and_delistings
    code, payload, err = _run_json(
        capsys,
        [
            "backtest",
            "--config", str(config),
            "--prices", str(csv),
            "--lookback", "100",
            "--rebalance-every", "50",
            "--delisting-grace", "5",
            "--json",
        ],
    )
    assert code == 0
    assert payload["window"]["end"] == str(prices.index[-1])
    # Starts ``lookback`` returns into the panel, not into the common window.
    assert payload["window"]["start"] == str(prices.index[101])
    delistings = payload["notes"]["delistings"]
    assert set(delistings) == {"Gold"}
    assert delistings["Gold"]["last_print"] == prices.index[PIT_DELISTS - 1].isoformat()

    # The alignment the initial solve used is still reported, word for word,
    # and the walk-forward's own panel is reported beside it.
    _, common = align_panel(prices, method="common")
    assert payload["alignment"][: len(common)] == common
    walk_line = payload["alignment"][len(common)]
    assert "walk-forward" in walk_line and "interior" in walk_line
    assert "Alignment:" in err


def test_a_universe_is_built_on_the_unaligned_panel(
    capsys, listings_and_delistings, tmp_path
):
    """The screen sees the late listing arrive instead of a panel that began with it."""
    prices, csv, config = listings_and_delistings
    rules = tmp_path / "printed.yaml"
    rules.write_text(
        yaml.safe_dump(
            {"rules": [{"kind": "rolling", "panel": "returns", "window": 100,
                        "agg": "count", "op": ">=", "value": 100}]}
        )
    )
    code, payload, _ = _run_json(
        capsys,
        [
            "backtest",
            "--config", str(config),
            "--prices", str(csv),
            "--lookback", "100",
            "--rebalance-every", "50",
            "--universe", str(rules),
            "--json",
        ],
    )
    assert code == 0
    assert payload["window"]["end"] == str(prices.index[-1])
    breadth = payload["notes"]["universe"]["breadth"]
    # Before the late name has a full window the screen admits three names,
    # after the delisted one's window runs dry it admits three again, and in
    # between all four. A panel cut to the common window only ever had four.
    assert set(breadth.values()) == {3, 4}
    # Every window the screen admits is complete, so nothing failed to solve.
    assert payload["notes"]["n_failed_solves"] == 0

