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
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mcp.server.mcpserver", reason="the `mcp` extra is not installed")

from optimization_engine.mcp_server import ToolError, mcp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "config" / "example_multi_asset.yaml")


@pytest.fixture(autouse=True)
def _readable_roots(tmp_path, monkeypatch):
    """Let the tools read this test's files and the repository's examples.

    The server reads only under its allowed roots, which default to the
    working directory; a test's ``tmp_path`` is outside it. The first root
    is where a relative path resolves.
    """
    monkeypatch.setenv(
        "OPTENGINE_MCP_ROOTS", os.pathsep.join([str(tmp_path), str(ROOT)])
    )


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
        ({"sample": True, "config_path": "nope.yaml"}, "No such config file"),
        ({"prices_path": "nope.csv"}, "No such price file"),
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


# ---------------------------------------------------------------------------
# What the path-taking tools may read, and what their errors may say
# ---------------------------------------------------------------------------


def test_a_path_outside_the_allowed_roots_is_refused(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    secret = elsewhere / "prices.csv"
    secret.write_text("date,A\n2024-01-01,1\n")
    monkeypatch.setenv("OPTENGINE_MCP_ROOTS", str(ROOT))
    message = failure("optimize", {"prices_path": str(secret)})
    assert "outside the directories" in message
    assert "OPTENGINE_MCP_ROOTS" in message


def test_a_relative_path_cannot_climb_out_of_a_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.yaml").write_text("optimizer: risk_parity\n")
    monkeypatch.setenv("OPTENGINE_MCP_ROOTS", str(root))
    message = failure("check_mandate", {"sample": True, "config_path": "../outside.yaml"})
    assert "outside the directories" in message


@pytest.mark.parametrize(
    "path",
    [
        "\\\\fileserver\\share\\prices.csv",
        "//fileserver/share/prices.csv",
        "\\\\?\\C:\\prices.csv",
    ],
)
def test_a_network_or_device_path_is_refused_before_it_is_touched(path):
    # On Windows, merely stat-ing a UNC path makes the OS authenticate to that
    # host over SMB with the user's credentials.
    message = failure("optimize", {"prices_path": path})
    assert "network or device path" in message


def test_an_unsupported_extension_is_refused_without_reading_the_file(
    tmp_path, monkeypatch
):
    key = tmp_path / "id_rsa"
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nFAKE\n")
    reads = []
    original = Path.read_text

    def spy(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    message = failure("check_mandate", {"sample": True, "config_path": str(key)})
    assert ".yaml" in message
    assert reads == []


def test_an_oversized_file_is_refused_before_it_is_read(tmp_path, monkeypatch):
    import optimization_engine.mcp_server as server

    big = tmp_path / "big.yaml"
    big.write_text("optimizer: risk_parity\n" + "# padding\n" * 50)
    monkeypatch.setattr(server, "MAX_CONFIG_BYTES", 64)
    message = failure("check_mandate", {"sample": True, "config_path": str(big)})
    assert "bytes" in message


@pytest.mark.parametrize(
    ("name", "text"),
    [
        # Malformed YAML: the parser's message quotes the offending line.
        ("creds.yaml", "aws_secret_access_key: AKIA_SECRET_1\n  broken: [SECRET_LINE_2\n"),
        # A known key holding a value of the wrong type: float() quotes it.
        ("c2.yaml", "ewma_lambda: sk-live-SECRETVALUE\n"),
        # A JSON list: its items came back as "unknown config keys".
        ("list.json", '["ghp_SECRET_TOKEN_IN_LIST", "another-SECRET"]'),
    ],
)
def test_a_config_error_does_not_quote_the_file(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    message = failure("check_mandate", {"sample": True, "config_path": str(path)})
    assert "SECRET" not in message
    # Still says which file and what kind of problem.
    assert name in message


def test_a_price_file_error_does_not_quote_the_file(tmp_path):
    csv = tmp_path / "passwords.csv"
    csv.write_text(
        "name,url,username,password\nSECRET_ROW,https://example.invalid,alice,hunter2\n"
    )
    message = failure("optimize", {"prices_path": str(csv)})
    assert "SECRET_ROW" not in message
    assert "passwords.csv" in message


# ---------------------------------------------------------------------------
# Failures that used to reach the client as "Error executing tool X"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["check_mandate", "optimize", "backtest"])
def test_an_unknown_optimizer_override_names_the_alternatives(tool):
    message = failure(tool, {"sample": True, "optimizer": "nope"})
    assert "nope" in message and "risk_parity" in message


def test_a_shock_outside_the_panel_keeps_its_reason(tmp_path):
    config = tmp_path / "stress.yaml"
    config.write_text(
        "optimizer: equal_weight\nstress:\n  - name: crash\n"
        "    returns: {NOT_AN_ASSET: -0.3}\n"
    )
    message = failure("backtest", {"sample": True, "config_path": str(config)})
    assert "NOT_AN_ASSET" in message


def test_an_invalid_backtest_spec_keeps_its_reason():
    # Called directly: the schema now refuses negative costs before the body
    # runs, and this is the body's own handler for every other spec error.
    from optimization_engine.mcp_server import backtest

    with pytest.raises(ToolError, match="commission_bps"):
        backtest(sample=True, optimizer="equal_weight", commission_bps=-500.0)


def test_a_text_column_is_named_by_position_not_by_content(tmp_path):
    csv = tmp_path / "tokens.csv"
    csv.write_text("date,token\n2024-01-01,ghp_SECRET_X\n2024-01-02,ghp_SECRET_Y\n")
    message = failure("check_mandate", {"prices_path": str(csv)})
    assert "not numeric" in message
    assert "SECRET" not in message


# ---------------------------------------------------------------------------
# Compute limits
# ---------------------------------------------------------------------------


def test_the_schema_bounds_the_backtest_arguments():
    schema = {t.name: t for t in asyncio.run(mcp.list_tools())}["backtest"].input_schema
    props = schema["properties"]

    def minimum(prop):
        options = prop.get("anyOf", [prop])
        return next(o["minimum"] for o in options if "minimum" in o)

    assert minimum(props["rebalance_every"]) >= 1
    assert minimum(props["lookback"]) >= 2
    assert minimum(props["commission_bps"]) == 0
    assert minimum(props["slippage_bps"]) == 0


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # Zero used to be read as "use the default" and run silently.
        ({"rebalance_every": 0}, "greater than or equal to 1"),
        ({"commission_bps": -500.0}, "greater than or equal to 0"),
    ],
)
def test_out_of_range_arguments_are_refused_by_the_schema(args, expected):
    message = failure("backtest", {"sample": True, "optimizer": "equal_weight", **args})
    assert expected in message


def test_a_walk_forward_with_too_many_resolves_is_refused():
    # A re-solve on every bar of the sample panel is ~1,500 solves: minutes of
    # a blocked server for one call.
    message = failure(
        "backtest", {"sample": True, "optimizer": "min_variance", "rebalance_every": 1}
    )
    assert "re-solves" in message


def test_a_panel_wider_than_the_limit_is_refused(monkeypatch):
    import optimization_engine.mcp_server as server

    monkeypatch.setattr(server, "MAX_ASSETS", 5)
    message = failure("optimize", {"sample": True})
    assert "13 assets" in message and "5" in message


def test_root_on_the_command_line_overrides_the_environment(tmp_path, monkeypatch):
    import optimization_engine.mcp_server as server

    # Restored after the test, so the override does not leak into the others.
    monkeypatch.setattr(server, "_command_line_roots", None)
    monkeypatch.setattr(server.mcp, "run", lambda **kwargs: None)
    server.main(["--root", str(tmp_path)])
    assert server.allowed_roots() == (tmp_path.resolve(),)


def test_every_tool_names_its_data_source(tmp_path):
    """The CLI's `data_source`, filled in over the protocol too."""
    from optimization_engine.data.loader import sample_dataset

    csv = tmp_path / "prices.csv"
    sample_dataset(n_periods=600)[["US_Equity", "Gold", "US_Treasuries"]].to_csv(
        csv, index_label="date"
    )
    for tool, extra in (
        ("optimize", {}),
        ("check_mandate", {}),
        ("backtest", {"lookback": 252, "rebalance_every": 63}),
    ):
        synthetic = call(tool, {"sample": True, "optimizer": "risk_parity", **extra})
        assert synthetic["data_source"] == {
            "kind": "sample", "synthetic": True, "path": None,
            "provider": None, "identifiers": None,
        }, tool
        named = call(tool, {"prices_path": str(csv), "optimizer": "risk_parity", **extra})
        assert named["data_source"]["kind"] == "file", tool
        assert named["data_source"]["path"] == str(csv), tool
        assert named["data_source"]["synthetic"] is False, tool
