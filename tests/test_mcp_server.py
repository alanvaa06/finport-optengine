"""The MCP server's tool surface, exercised through the server.

Calling the decorated functions directly would test the bodies and skip
everything that makes this a server: schema generation from the signatures,
argument coercion, and the error wrapping that decides whether a caller sees
a useful message or "Error executing tool optimize".

The SDK needs Python 3.10, one minor above this package's floor, so the
whole module skips where it is unavailable rather than failing the suite on
3.9.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mcp.server.mcpserver", reason="the `mcp` extra is not installed")

from optimization_engine.mcp_server import ToolError, mcp  # noqa: E402

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "example_multi_asset.yaml")


def call(name: str, args: dict):
    """Invoke a tool the way a client would, returning its structured result."""
    result = asyncio.run(mcp.call_tool(name, args))
    assert not result.is_error, f"{name} reported an error"
    return result.structured_content


def failure(name: str, args: dict) -> str:
    """Invoke a tool expecting it to fail, returning the message a client sees."""
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 — SDK wraps the type
        asyncio.run(mcp.call_tool(name, args))
    return str(excinfo.value)


def test_every_tool_is_registered_with_a_schema():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert set(tools) == {
        "list_optimizers",
        "describe_optimizer",
        "check_mandate",
        "optimize",
        "backtest",
    }
    # A tool whose parameters did not make it into the schema is unusable:
    # the client has nothing to fill in.
    assert "name" in tools["describe_optimizer"].input_schema["properties"]
    assert "sample" in tools["optimize"].input_schema["properties"]
    # The description is what a model reads when choosing a tool, so an
    # empty one is a silent failure rather than a cosmetic one.
    assert all(t.description for t in tools.values())


def test_list_optimizers_enumerates_the_methods():
    payload = call("list_optimizers", {})
    names = [o["name"] for o in payload["optimizers"]]
    assert "risk_parity" in names
    assert "max_sharpe" in names
    assert all(o["summary"] for o in payload["optimizers"])


def test_describe_optimizer_reports_the_contract():
    payload = call("describe_optimizer", {"name": "risk_parity"})
    assert payload["name"] == "risk_parity"
    assert payload["requires"]["covariance"] is True
    assert isinstance(payload["supports"]["turnover"], bool)


def test_optimize_returns_weights_and_the_evidence():
    payload = call("optimize", {"sample": True, "optimizer": "risk_parity"})
    weights = payload["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert payload["solver"]
    # Weights without these are what this engine exists not to return.
    assert payload["diagnostics"]["effective_n"] is not None
    assert payload["diagnostics"]["effective_n_risk"] is not None
    assert payload["covariance"]["is_psd"] is True


def test_check_and_optimize_agree_about_the_same_mandate():
    """The regression this pair exists for.

    `check_mandate` used to derive expected returns as zeros when the config
    carried no `expected_returns` block, while `optimize` derived them from
    the return history. The check therefore validated a mandate the solve
    never saw: a reachable range of exactly zero to zero, against a solve
    that returned a real expected return outside it.
    """
    checked = call("check_mandate", {"sample": True, "optimizer": "max_sharpe"})
    lo = checked["feasibility"]["min_return"]
    hi = checked["feasibility"]["max_return"]
    assert lo is not None and hi is not None
    assert hi > lo, "a degenerate range means the two are out of step again"

    solved = call("optimize", {"sample": True, "optimizer": "max_sharpe"})
    achieved = solved["metrics"]["expected_return"]
    assert lo - 1e-9 <= achieved <= hi + 1e-9, (
        f"the solve returned {achieved}, outside the {lo}..{hi} range check promised"
    )


def test_backtest_returns_the_hashes_that_identify_the_run():
    payload = call(
        "backtest",
        {"sample": True, "optimizer": "risk_parity", "lookback": 504, "rebalance_every": 252},
    )
    assert payload["spec_hash"]
    assert payload["result_hash"]
    assert payload["window"]["n_periods"] > 0


def test_a_config_path_is_honoured():
    payload = call("check_mandate", {"config_path": CONFIG, "sample": True})
    assert payload["ready"] is True


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, "No data given"),
        ({"sample": True, "prices_path": "/nope.csv"}, "not both"),
        ({"sample": True, "config_path": "/nope.yaml"}, "No such config file"),
        ({"prices_path": "/nope.csv"}, "No such price file"),
    ],
)
def test_anticipated_failures_keep_their_message(args, expected):
    """A `ToolError` reaches the client; anything else is wrapped and lost.

    The SDK turns an unanticipated exception into `UnexpectedToolError` with
    the text replaced by "Error executing tool optimize" — a failure an
    agent can neither act on nor explain. Every reachable bad-input path
    therefore has to raise `ToolError` specifically.
    """
    assert expected in failure("optimize", args)


def test_an_unknown_optimizer_names_the_alternatives():
    message = failure("describe_optimizer", {"name": "definitely_not_a_method"})
    assert "definitely_not_a_method" in message
    # A dead end that lists the valid names is one call from recovery.
    assert "risk_parity" in message


def test_tool_error_is_the_sdk_class():
    """Guards the import in `mcp_server`, which is what makes the above work."""
    assert issubclass(ToolError, Exception)
    assert ToolError.__name__ == "ToolError"


def test_an_optimizer_override_keeps_the_rest_of_the_mandate():
    from optimization_engine.config import load_config
    from optimization_engine.mcp_server import _config

    original = load_config(CONFIG).optimizer
    overridden = _config(CONFIG, "max_sharpe").optimizer
    assert overridden.name == "max_sharpe"
    # Replacing the whole spec solved max-Sharpe against a cash rate of zero
    # on a config that said otherwise, and dropped the return target with it.
    assert overridden.risk_free_rate == original.risk_free_rate
    assert overridden.target_return == original.target_return
    assert overridden.risk_aversion == original.risk_aversion


def test_an_infeasible_mandate_fails_with_its_report(tmp_path):
    import yaml

    data = yaml.safe_load(Path(CONFIG).read_text())
    data["bounds"] = {a: [0.0, 0.05] for a in data["expected_returns"]}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    # The engine's default lets the solver fail instead of raising the
    # feasibility report, which made the ToolError branch unreachable and
    # handed the client a message-less wrapped exception.
    message = failure("optimize", {"config_path": str(bad), "sample": True})
    assert "no solution" in message
    assert "100%" in message or "cap" in message.lower()


def test_backtest_is_shaped_by_the_config_it_is_handed(tmp_path):
    daily = call("backtest", {"sample": True, "optimizer": "risk_parity"})
    calendar_config = tmp_path / "calendar.yaml"
    # A basis the daily dates allow (seven-day markets): a monthly 12 on this
    # panel is now refused as a contradiction rather than simulated.
    calendar_config.write_text("periods_per_year: 365\noptimizer: risk_parity\n")
    calendar = call(
        "backtest",
        {"config_path": str(calendar_config), "sample": True, "lookback": 504, "rebalance_every": 63},
    )
    # The spec hash covers the annualization basis and the trading cadence;
    # the config's basis used to be ignored here, so the two agreed.
    assert daily["spec_hash"] != calendar["spec_hash"]
    assert daily["window"]["n_periods"] > 0


def test_every_payload_carries_the_alignment_log(tmp_path):
    """The transport promise: the same payload the CLI's `--json` emits.

    `_panel` used to make the panel rectangular with a bare
    `dropna(how="any")`, so a client that handed over a file with one
    late-listing asset got a book estimated on a truncated sample with no
    way to find that out. There is no stdout to narrate on here — this
    server speaks the protocol over stdio — so the log has to be in the
    payload or nowhere.
    """
    import pandas as pd

    from optimization_engine.data.loader import sample_dataset
    from optimization_engine.data.quality import align_panel

    prices = sample_dataset()
    prices.loc[prices.index[:500], prices.columns[0]] = float("nan")
    csv = tmp_path / "late_listing.csv"
    prices.to_csv(csv)
    _, expected = align_panel(pd.read_csv(csv, index_col=0, parse_dates=True), "common")
    assert expected, "the fixture is only useful if something is dropped"

    args = {"prices_path": str(csv), "optimizer": "risk_parity"}
    assert call("check_mandate", args)["alignment"] == expected
    assert call("optimize", args)["alignment"] == expected
    assert call("backtest", {**args, "lookback": 504, "rebalance_every": 252})[
        "alignment"
    ] == expected


def test_a_complete_panel_reports_an_empty_alignment_log():
    """Present and empty, so a client can test the value rather than the key."""
    assert call("check_mandate", {"sample": True})["alignment"] == []


def test_every_tool_reports_the_raw_panels_data_quality(tmp_path):
    """optimize never looked at data quality; check looked after aligning.

    Alignment removes the very gaps the report exists to name, so a panel
    with an asset missing a quarter of its history came back from both tools
    without a word about it. The report is now read off the raw panel, as the
    CLI reads it, and travels in every payload.
    """
    import numpy as np
    import pandas as pd

    days = pd.bdate_range("2021-01-04", periods=600)
    rng = np.random.default_rng(4)
    gappy = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    gappy[100:400:2] = np.nan
    steady = 80 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    csv = tmp_path / "gappy.csv"
    pd.DataFrame({"date": days, "GAPPY": gappy, "OK": steady}).to_csv(csv, index=False)

    args = {"prices_path": str(csv), "optimizer": "min_variance"}
    for tool, extra in (
        ("optimize", {}),
        ("check_mandate", {}),
        ("backtest", {"lookback": 252, "rebalance_every": 63}),
    ):
        quality = call(tool, {**args, **extra})["data_quality"]
        assert quality["usable"] is False, tool
        assert {f["code"] for f in quality["findings"]} >= {"interior_gaps"}, tool


def test_a_monthly_file_is_annualized_on_twelve_and_a_contradiction_refused(tmp_path):
    """The CLI's rule, over the protocol: the dates decide, a contradiction fails."""
    from optimization_engine.data.loader import sample_dataset

    daily = sample_dataset(n_periods=252 * 10, seed=1)[["US_Equity", "Gold"]]
    csv = tmp_path / "monthly.csv"
    daily.resample("ME").last().to_csv(csv, index_label="date")

    solved = call("optimize", {"prices_path": str(csv), "optimizer": "min_variance"})
    assert solved["metrics"]["expected_volatility"] < 0.15

    stated = tmp_path / "daily.yaml"
    stated.write_text("periods_per_year: 252\noptimizer: min_variance\n")
    message = failure("optimize", {"prices_path": str(csv), "config_path": str(stated)})
    assert "periods_per_year: 12" in message
