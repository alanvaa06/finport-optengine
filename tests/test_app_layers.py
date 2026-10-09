"""End-to-end check that the layered-constraint builder is actually wired up.

The pure logic is covered in ``test_ui_state``; what this file protects is the
wiring — that clicking "Add layer" in the real app produces a layer the solve
honours, and that a saved scenario brings its policy back. Those are the two
things that break silently when the page script is refactored, and neither is
visible to a unit test.

Skipped when Streamlit is not installed (it is an optional ``[ui]`` extra).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from optimization_engine.ui_state import layer_states_to_layers  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from layer_editor import _current_bucket_weights  # noqa: E402

APP = ROOT / "app" / "streamlit_app.py"

ASSET_CLASS = {
    "US_Equity": "Equity",
    "Intl_Equity": "Equity",
    "EM_Equity": "Equity",
    "Real_Estate": "Alternatives",
    "Infra": "Alternatives",
    "Commodities": "Commodities",
    "Gold": "Commodities",
    "US_Treasuries": "Fixed Income",
    "TIPS": "Fixed Income",
    "IG_Credit": "Fixed Income",
    "HY_Credit": "Fixed Income",
    "EM_Debt": "Fixed Income",
    "Cash": "Cash",
}
SUB_CLASS = {
    "US_Equity": "DM Equity",
    "Intl_Equity": "DM Equity",
    "EM_Equity": "EM Equity",
    "US_Treasuries": "DM Fixed Income",
    "TIPS": "DM Fixed Income",
    "IG_Credit": "DM Fixed Income",
    "HY_Credit": "DM Fixed Income",
    "EM_Debt": "EM Fixed Income",
}
CLASS_CAPS = {
    "Equity": 0.60,
    "Fixed Income": 0.30,
    "Commodities": 0.10,
    "Alternatives": 0.15,
    "Cash": 0.10,
}


def _no_exception(at: AppTest) -> None:
    assert not at.exception, [str(e.value) for e in at.exception]


@pytest.fixture(scope="module")
def app() -> AppTest:
    """The app with asset classes named and their budgets set, before layers."""
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    _no_exception(at)

    table = at.session_state["config_table"].copy()
    table["Group"] = [ASSET_CLASS[a] for a in table.index]
    at.session_state["config_table"] = table
    at.run()

    bounds = at.session_state["group_bounds"].copy()
    for group, cap in CLASS_CAPS.items():
        bounds.loc[group, "Max Weight"] = cap
    at.session_state["group_bounds"] = bounds
    at.run()
    _no_exception(at)
    return at


def test_adding_a_layer_from_the_menu_produces_one_the_solve_honours(app):
    app.selectbox(key="layer_preset_choice").set_value(
        "Sub-asset class (DM / EM)"
    ).run()
    app.button(key="layer_add_btn").click().run()
    _no_exception(app)

    states = app.session_state["constraint_layer_states"]
    assert len(states) == 1
    states[0]["assignments"] = {
        a: SUB_CLASS.get(a, "—") for a in app.session_state["config_table"].index
    }
    app.run()
    _no_exception(app)

    next(b for b in app.button if b.label == "Optimize portfolio").click().run()
    _no_exception(app)

    run = app.session_state["last_run"]
    assert run is not None, "the solve produced nothing"
    assert run.result.is_compliant, run.result.violations
    assert [lyr.name for lyr in run.constraint_layers] == [
        "Asset class",
        "Sub-asset class",
    ]

    weights = run.result.weights
    equity = sum(w for a, w in weights.items() if ASSET_CLASS[a] == "Equity")
    em = sum(w for a, w in weights.items() if SUB_CLASS.get(a) == "EM Equity")
    assert equity <= 0.60 + 1e-6
    assert em <= 0.20 + 1e-6

    exposures = run.layer_exposures()
    assert set(exposures["layer"]) == {"Asset class", "Sub-asset class"}
    assert exposures["binding"].any()


def test_a_saved_scenario_brings_its_whole_policy_back(app):
    """Including the basis and parent — a mandate must reopen as it was saved."""
    states = app.session_state["constraint_layer_states"]
    assert states, "depends on the layer added by the previous test"
    uid = states[0]["uid"]
    app.radio(key=f"layer_basis_{uid}").set_value("parent").run()
    _no_exception(app)
    states = app.session_state["constraint_layer_states"]
    states[0]["limits"] = {
        "DM Equity": (0.0, 1.0),
        "EM Equity": (0.0, 0.30),
        "DM Fixed Income": (0.0, 1.0),
        "EM Fixed Income": (0.0, 0.25),
    }
    app.run()
    saved = [lyr.to_dict() for lyr in layer_states_to_layers(states)]
    assert saved[0]["basis"] == "parent"
    assert saved[0]["parent"] == "Asset class"

    app.text_input(key="scn_new_name").set_value("Layered mandate").run()
    app.button(key="scn_save").click().run()
    _no_exception(app)
    assert "Layered mandate" in app.session_state["scenarios"]

    app.session_state["constraint_layer_states"] = []
    app.run()
    app.selectbox(key="scn_select").set_value("Layered mandate").run()
    app.button(key="scn_load").click().run()
    app.run()
    _no_exception(app)

    restored = app.session_state["constraint_layer_states"]
    assert [
        lyr.to_dict() for lyr in layer_states_to_layers(restored)
    ] == saved


def test_removing_a_layer_removes_it_from_the_solve(app):
    app.selectbox(key="layer_preset_choice").set_value("Region").run()
    app.button(key="layer_add_btn").click().run()
    _no_exception(app)
    states = app.session_state["constraint_layer_states"]
    assert [s["name"] for s in states][-1] == "Region"

    app.button(key=f"layer_del_{states[-1]['uid']}").click().run()
    _no_exception(app)
    assert "Region" not in [
        s["name"] for s in app.session_state["constraint_layer_states"]
    ]


def test_the_live_column_states_the_book_in_the_units_of_the_limit_beside_it(app):
    """A number in the wrong units next to a limit invites the very mistake
    the column exists to prevent, so both bases are checked explicitly."""
    run = app.session_state["last_run"]
    assert run is not None, "depends on the solve in the first test"
    weights = run.result.weights
    state = dict(app.session_state["constraint_layer_states"][0])
    state["assignments"] = {
        a: SUB_CLASS.get(a, "—") for a in app.session_state["config_table"].index
    }

    equity = sum(w for a, w in weights.items() if ASSET_CLASS[a] == "Equity")
    em = sum(w for a, w in weights.items() if SUB_CLASS.get(a) == "EM Equity")
    assert equity > 0.01 and em > 0.001

    state["basis"] = "portfolio"
    absolute = _current_bucket_weights(state, weights)
    assert absolute["EM Equity"] == pytest.approx(100 * em)

    state["basis"] = "parent"
    state["parent"] = "Asset class"
    relative = _current_bucket_weights(state, weights, parent_assignments=ASSET_CLASS)
    assert relative["EM Equity"] == pytest.approx(100 * em / equity)


def test_the_live_column_is_absent_when_there_is_nothing_to_compare_against(app):
    state = dict(app.session_state["constraint_layer_states"][0])
    assert _current_bucket_weights(state, None) is None
    state["basis"] = "parent"
    # Relative units cannot be formed without the parent's assignments.
    assert _current_bucket_weights(
        state, app.session_state["last_run"].result.weights
    ) is None


def test_the_backtest_tab_prices_the_trading_it_shows():
    """The cost controls are wired to the simulation, not decoration.

    The backtest tab only renders once a solve has happened, so this drives
    the real page through an optimization first. What it protects is the
    wiring: that moving the commission, slippage, impact and lag controls
    reaches the simulation core, and that the cost panel renders off the
    result rather than off a separate calculation that can drift from it.
    """
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    _no_exception(at)

    optimize = [b for b in at.button if b.label == "Optimize portfolio"]
    assert optimize, "the page no longer offers an Optimize button"
    optimize[0].click().run()
    _no_exception(at)
    assert at.session_state["last_run"] is not None

    # Charge for the trading: commission, spread, and impact that scales.
    sliders = {s.label: s for s in at.slider}
    sliders["Commission (bps, one-way)"].set_value(15)
    sliders["Slippage (bps, one-way)"].set_value(5)
    sliders["Market impact (eta)"].set_value(0.5)
    at.run()
    _no_exception(at)

    from optimization_engine.backtest import BacktestSpec, CostSpec, compute_tca

    run = at.session_state["last_run"]
    spec = BacktestSpec(
        frequency="monthly",
        costs=CostSpec(commission_bps=15.0, slippage_bps=5.0, impact_coefficient=0.5),
        periods_per_year=run.config.periods_per_year,
    )
    panel = compute_tca(run.simulate(spec))
    assert panel.total_cost > 0
    # Impact rides on top of the linear charge, so the realized rate exceeds it.
    assert panel.cost_bps_of_notional > 20.0


# ---------------------------------------------------------------------------
# 💥 Stress and 🌍 Universe
# ---------------------------------------------------------------------------
#
# The pure halves of both tabs live in ``test_ui_stress_universe``; what these
# protect is the wiring — that a scenario typed into the grid reaches
# ``stress_test`` on the solved book, that the library's refusals arrive as
# messages instead of stack traces, and that the collapse policy the page
# offers is the one the run is actually made under.


@pytest.fixture(scope="module")
def solved_app() -> AppTest:
    """A fresh page with one solve behind it, shared by the tabs below.

    Both tabs need a book: a scenario is applied to weights, and a universe is
    applied to a run. Solving once and reusing it keeps the integration checks
    to a handful of seconds.
    """
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    _no_exception(at)
    next(b for b in at.button if b.label == "Optimize portfolio").click().run()
    _no_exception(at)
    assert at.session_state["last_run"] is not None
    return at


def _set_shocks(at: AppTest, rows: list[tuple[str, str, float]]) -> None:
    """Put a long-form scenario grid into the page and rerun it."""
    import pandas as pd

    at.session_state["stress_shock_table"] = pd.DataFrame(
        {
            "Scenario": [r[0] for r in rows],
            "Asset": [r[1] for r in rows],
            "Shock return": [r[2] for r in rows],
        }
    )
    at.run()


def _metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def _plotly_specs(at: AppTest) -> list[dict]:
    """Every Plotly figure the page emitted, as the JSON it emitted.

    ``AppTest`` does not model ``st.plotly_chart``, so the figure is read back
    off the element's proto — which is the serialized figure itself, and
    therefore exactly what a browser would draw.
    """
    import json

    return [
        json.loads(chart.proto.spec)
        for chart in at.get("plotly_chart")
        if chart.proto.spec
    ]


def test_the_stress_tab_reports_the_worst_scenario_and_what_drove_it(solved_app):
    """A scenario typed into the grid reaches the solved book."""
    held = [
        a
        for a, w in solved_app.session_state["last_run"].result.weights.items()
        if abs(w) > 1e-4
    ]
    assert len(held) >= 2, "the default solve holds too little to stress"
    worst_name, mild_name = "Deep drawdown", "Mild wobble"
    _set_shocks(
        solved_app,
        [
            (worst_name, held[0], -0.40),
            (worst_name, held[1], -0.20),
            (mild_name, held[0], -0.02),
        ],
    )
    _no_exception(solved_app)

    cards = _metrics(solved_app)
    assert cards["Worst scenario"] == worst_name
    assert cards["Largest contributor"] in held
    # The number on the card is the report's, not a re-derivation of it.
    from optimization_engine.stress import stress_test

    run = solved_app.session_state["last_run"]
    report = stress_test(
        run.result.weights,
        list(run.config.stress) or [],
        cov_matrix=run.cov_matrix,
    ) if run.config.stress else None
    assert report is None or report.worst.name == worst_name


def test_a_shock_on_a_name_the_book_cannot_hold_is_a_message_not_a_crash(solved_app):
    """And the scenario is still there afterwards — refused, not deleted."""
    _set_shocks(
        solved_app,
        [("Wider than the book", "US_Equity", -0.30),
         ("Wider than the book", "NOT_IN_THIS_BOOK", -0.50)],
    )
    _no_exception(solved_app)

    errors = [e.value for e in solved_app.error]
    assert any("does not hold" in message for message in errors), errors
    assert any("NOT_IN_THIS_BOOK" in message for message in errors), errors

    # Not silently dropped: the leg is still in the grid, and it still reaches
    # the mandate, so saving the preset saves the scenario as written.
    table = solved_app.session_state["stress_shock_table"]
    assert "NOT_IN_THIS_BOOK" in set(table["Asset"])
    solved_app.text_input(key="scn_new_name").set_value("Refused scenario").run()
    solved_app.button(key="scn_save").click().run()
    _no_exception(solved_app)
    saved = solved_app.session_state["scenarios"]["Refused scenario"].config
    assert [s.name for s in saved.stress] == ["Wider than the book"]
    assert "NOT_IN_THIS_BOOK" in saved.stress[0].returns


def test_applying_it_anyway_records_what_was_dropped(solved_app):
    """The library's escape hatch, wired to a control that says what it costs."""
    _set_shocks(
        solved_app,
        [("Wider than the book", "EM_Equity", -0.30),
         ("Wider than the book", "NOT_IN_THIS_BOOK", -0.50)],
    )
    solved_app.radio(key="stress_unknown_assets").set_value("ignore").run()
    _no_exception(solved_app)

    assert not [e.value for e in solved_app.error]
    warnings = [w.value for w in solved_app.warning]
    assert any("NOT_IN_THIS_BOOK" in w for w in warnings), warnings
    assert any("smaller than the scenario" in w for w in warnings), warnings
    solved_app.radio(key="stress_unknown_assets").set_value("raise").run()


def test_scenarios_survive_a_saved_preset(solved_app):
    """`EngineConfig.stress` is part of the mandate, so it must reopen."""
    from components import empty_scenario_table, empty_shock_table

    _set_shocks(
        solved_app,
        [("Rates repricing", "IG_Credit", -0.16),
         ("Rates repricing", "EM_Equity", -0.20)],
    )
    meta = solved_app.session_state["stress_scenario_table"]
    meta.loc["Rates repricing", "Covariance ×"] = 2.25
    meta.loc["Rates repricing", "Correlation shift"] = 0.5
    meta.loc["Rates repricing", "Notes"] = "2022-shaped"
    solved_app.session_state["stress_scenario_table"] = meta
    solved_app.run()

    solved_app.text_input(key="scn_new_name").set_value("Stressed mandate").run()
    solved_app.button(key="scn_save").click().run()
    _no_exception(solved_app)
    saved = solved_app.session_state["scenarios"]["Stressed mandate"].config
    assert [s.to_dict() for s in saved.stress] == [
        {
            "name": "Rates repricing",
            "returns": {"IG_Credit": -0.16, "EM_Equity": -0.20},
            "covariance_scale": 2.25,
            "notes": "2022-shaped",
            "correlation_shift": 0.5,
        }
    ]

    solved_app.session_state["stress_shock_table"] = empty_shock_table()
    solved_app.session_state["stress_scenario_table"] = empty_scenario_table()
    solved_app.run()
    solved_app.selectbox(key="scn_select").set_value("Stressed mandate").run()
    solved_app.button(key="scn_load").click().run()
    solved_app.run()
    _no_exception(solved_app)

    restored = solved_app.session_state["stress_shock_table"]
    assert sorted(restored["Asset"]) == ["EM_Equity", "IG_Credit"]
    restored_meta = solved_app.session_state["stress_scenario_table"]
    assert float(restored_meta.loc["Rates repricing", "Covariance ×"]) == 2.25
    assert float(restored_meta.loc["Rates repricing", "Correlation shift"]) == 0.5
    assert restored_meta.loc["Rates repricing", "Notes"] == "2022-shaped"

    # Leave the page as the other tests expect to find it.
    solved_app.session_state["stress_shock_table"] = empty_shock_table()
    solved_app.session_state["stress_scenario_table"] = empty_scenario_table()
    solved_app.run()


def test_the_shipped_shock_file_loads_into_the_page(solved_app):
    """`config/shocks.yaml` is what the README's --stress line points at.

    Uploading it is the same journey a desk takes: a scenario library written
    once, run from the CLI and read in the app, with neither one having to be
    retyped into the other.
    """
    from components import empty_scenario_table, empty_shock_table

    payload = (ROOT / "config" / "shocks.yaml").read_bytes()
    solved_app.file_uploader(key="stress_upload").set_value(
        ("shocks.yaml", payload, "text/yaml")
    ).run()
    _no_exception(solved_app)

    from optimization_engine.stress import load_shocks

    expected = load_shocks(ROOT / "config" / "shocks.yaml")
    rows = solved_app.session_state["stress_shock_table"]
    assert set(rows["Scenario"]) == {s.name for s in expected}
    meta = solved_app.session_state["stress_scenario_table"]
    assert list(meta.index) == [s.name for s in expected]

    # And they reach the book: every scenario names only assets the sample
    # panel carries, so the report renders rather than being refused.
    assert not [e.value for e in solved_app.error]
    cards = _metrics(solved_app)
    assert cards["Worst scenario"] in {s.name for s in expected}

    solved_app.file_uploader(key="stress_upload").set_value(None).run()
    solved_app.session_state["stress_shock_table"] = empty_shock_table()
    solved_app.session_state["stress_scenario_table"] = empty_scenario_table()
    solved_app.run()


def _capture_downloads(monkeypatch) -> dict[str, bytes]:
    """Record every file the page hands a download button, by file name.

    ``AppTest`` keeps a download's bytes in a media store it throws away at
    the end of each run, so they are recorded on the way in — which is
    exactly the file a browser would save.
    """
    from streamlit.runtime.media_file_manager import MediaFileManager

    captured: dict[str, bytes] = {}
    original = MediaFileManager.add

    def add(self, path_or_data, *args, **kwargs):
        name = kwargs.get("file_name")
        if name is not None and isinstance(path_or_data, bytes):
            captured[name] = path_or_data
        return original(self, path_or_data, *args, **kwargs)

    monkeypatch.setattr(MediaFileManager, "add", add)
    return captured


def _stress_report_text(at: AppTest) -> str:
    """The "report as text" block the Stress tab printed."""
    texts = [t.value for t in at.text if t.value.startswith("Stressed ")]
    assert len(texts) == 1, texts
    return texts[0]


def test_a_correlation_shift_survives_upload_edit_download_and_run(
    solved_app, monkeypatch
):
    """The grid used to drop it, so a breakdown scenario only scaled vols."""
    import pandas as pd
    from components import empty_scenario_table, empty_shock_table

    from optimization_engine.stress import (
        Shock,
        dump_shocks_yaml,
        load_shocks_yaml,
        stress_test,
    )

    downloads = _capture_downloads(monkeypatch)
    uploaded = [
        Shock(
            "Correlation breakdown",
            {"US_Equity": -0.12, "Intl_Equity": -0.14, "US_Treasuries": -0.02},
            covariance_scale=2.25,
            correlation_shift=0.5,
        ),
        Shock("Vols only", {"US_Equity": -0.05}, covariance_scale=2.25),
    ]
    solved_app.file_uploader(key="stress_upload").set_value(
        ("breakdown.yaml", dump_shocks_yaml(uploaded).encode("utf-8"), "text/yaml")
    ).run()
    _no_exception(solved_app)

    meta = solved_app.session_state["stress_scenario_table"]
    assert meta.loc["Correlation breakdown", "Correlation shift"] == 0.5
    assert pd.isna(meta.loc["Vols only", "Correlation shift"])

    # Edit: a stronger breakdown, and a shift typed onto the other scenario.
    meta.loc["Correlation breakdown", "Correlation shift"] = 0.8
    meta.loc["Vols only", "Correlation shift"] = 0.25
    solved_app.session_state["stress_scenario_table"] = meta
    solved_app.run()
    _no_exception(solved_app)
    assert not [e.value for e in solved_app.error]

    # Download: the file carries what the grid now says.
    saved = load_shocks_yaml(downloads["shocks.yaml"].decode("utf-8"))
    assert {s.name: s.correlation_shift for s in saved} == {
        "Correlation breakdown": 0.8,
        "Vols only": 0.25,
    }
    assert {s.name: s.covariance_scale for s in saved} == {
        "Correlation breakdown": 2.25,
        "Vols only": 2.25,
    }

    # Run: the report on the page is the library's report for those shocks,
    # and not the one it gives with the shifts left out.
    run = solved_app.session_state["last_run"]
    expected = stress_test(run.result.weights, list(saved), cov_matrix=run.cov_matrix)
    unshifted = stress_test(
        run.result.weights,
        [Shock(s.name, s.returns, s.covariance_scale, s.notes) for s in saved],
        cov_matrix=run.cov_matrix,
    )
    page = _stress_report_text(solved_app)
    assert page == expected.describe()
    assert page != unshifted.describe()

    solved_app.file_uploader(key="stress_upload").set_value(None).run()
    solved_app.session_state["stress_shock_table"] = empty_shock_table()
    solved_app.session_state["stress_scenario_table"] = empty_scenario_table()
    solved_app.run()


def test_the_universe_tab_draws_three_states_not_two(solved_app):
    """The heatmap has to say "nobody evaluated this", or it says nothing."""
    from components import ELIGIBILITY_COLORS, ELIGIBILITY_STATES

    cards = _metrics(solved_app)
    for state in ELIGIBILITY_STATES:
        assert state in cards, cards
        assert int(cards[state].split()[0].replace(",", "")) > 0, (state, cards)

    # Read the figure the page actually emitted, not one rebuilt here: the
    # question is whether the *page* draws three states, and a two-colour
    # scale would still pass every assertion made off a fresh figure.
    heatmaps = [
        trace
        for spec in _plotly_specs(solved_app)
        for trace in spec["data"]
        if trace.get("type") == "heatmap"
        and {str(color).lower() for _, color in trace.get("colorscale", [])}
        == {c.lower() for c in ELIGIBILITY_COLORS}
    ]
    assert heatmaps, "the eligibility heatmap is not on the page"
    trace = heatmaps[0]
    # Three flat bands rather than a ramp: each colour repeats at both ends of
    # its third of the scale, so nothing is interpolated between two states
    # that have no midpoint between them.
    assert len(trace["colorscale"]) == 2 * len(ELIGIBILITY_COLORS)
    # And all three states are really on the figure. Read off ``customdata``,
    # which is the hover text: the state is named in words as well as
    # coloured, so the distinction survives a reader who cannot see the hue.
    drawn = {cell for row in trace["customdata"] for cell in row}
    assert drawn == set(ELIGIBILITY_STATES), drawn


def test_the_policy_is_not_chosen_for_the_reader(solved_app):
    """No default, and the page says what a choice would cost before it is made."""
    assert solved_app.radio(key="universe_policy").value is None
    prompts = [i.value for i in solved_app.info]
    assert any("Choose a policy above" in p for p in prompts), prompts
    assert any(
        "Choose a collapse policy above" in p for p in prompts
    ), prompts


def test_the_policy_selector_changes_what_the_collapse_costs(solved_app):
    """Same rules, same book, different policy — and a different run."""
    solved_app.radio(key="universe_policy").set_value("exclude").run()
    _no_exception(solved_app)
    excluded_line = [
        i.value for i in solved_app.info if "not evaluable" in i.value
    ]
    assert excluded_line and "'exclude'" in excluded_line[0]
    assert "reads them as ineligible" in excluded_line[0]

    solved_app.button(key="run_universe").click().run()
    _no_exception(solved_app)
    under_exclude = solved_app.session_state["universe_run_notes"]["universe"]

    solved_app.radio(key="universe_policy").set_value("include").run()
    included_line = [
        i.value for i in solved_app.info if "not evaluable" in i.value
    ]
    assert included_line and "admits them" in included_line[0]

    solved_app.button(key="run_universe").click().run()
    _no_exception(solved_app)
    under_include = solved_app.session_state["universe_run_notes"]["universe"]

    assert under_exclude["policy"] == "exclude"
    assert under_include["policy"] == "include"
    # The warm-up is where the two disagree: 'exclude' has nothing eligible
    # there, 'include' has everything.
    assert under_include["min_breadth"] > under_exclude["min_breadth"]
    assert under_include["n_liquidations"] < under_exclude["n_liquidations"]


def test_the_universe_run_surfaces_the_notes_only_it_records(solved_app):
    """Breadth, forced liquidations and delistings, from the run's own notes."""
    solved_app.radio(key="universe_policy").set_value("exclude").run()
    solved_app.radio(key="universe_mode").set_value("walk_forward").run()
    solved_app.button(key="run_universe").click().run()
    _no_exception(solved_app)

    notes = solved_app.session_state["universe_run_notes"]
    assert set(notes["universe"]) >= {
        "policy", "breadth", "liquidated", "unknown_assets", "n_liquidations"
    }
    # Delisting is a separate opt-in and only the re-solving path diagnoses it.
    assert "delistings" in notes
    cards = _metrics(solved_app)
    assert cards["Collapse policy"] == "exclude"
    assert "Forced liquidations" in cards
    assert any(
        "kept printing for the whole run" in c.value for c in solved_app.caption
    )
    solved_app.radio(key="universe_mode").set_value("replay").run()


def test_the_explain_inspector_answers_for_one_name_on_one_date(solved_app):
    asset = solved_app.selectbox(key="universe_explain_asset").value
    answers = [
        i.value
        for i in solved_app.info
        if i.value.startswith(f"{asset} on ")
    ]
    assert answers, [i.value for i in solved_app.info]
    assert any(
        verdict in answers[0]
        for verdict in ("eligible", "not eligible", "not evaluable")
    )


@pytest.mark.parametrize(
    "rules_text",
    [
        # A misspelt key. Refused rather than defaulted: a rules file that
        # loads cleanly with a typo in it is a *different mandate* silently
        # substituted for the one that was signed off.
        "schema_version: 1\nrules:\n  - {kind: rolling, panel: returns, windwo: 63}\n",
        # Not YAML at all — an unterminated flow mapping, the shape a
        # half-typed rule takes.
        "rules:\n  - {kind: rolling,\n",
        # Empty, and a list where a mapping belongs.
        "",
        "- just\n- a\n- list\n",
    ],
)
def test_broken_rules_are_a_message_beside_the_box_that_broke_them(
    solved_app, rules_text
):
    """Every way of getting the document wrong ends on the page, not in a trace."""
    solved_app.session_state["universe_rules_text"] = rules_text
    solved_app.run()
    _no_exception(solved_app)
    errors = [e.value for e in solved_app.error]
    assert any("do not describe a universe" in message for message in errors), errors


def test_the_universe_tab_takes_no_path_to_a_file_on_the_server(solved_app):
    """`streamlit run` listens on every interface by default.

    A free-text "Rules file on disk" box let anyone who could reach the page
    have the server read any file it could see, and the parse error quoted
    what it found — a fake ``~/.aws/credentials`` came back with its key.
    """
    labels = [t.label for t in solved_app.text_input]
    assert "Rules file on disk" not in labels


def test_a_rules_document_naming_a_panel_file_is_refused_unread(solved_app, tmp_path):
    """Typed or uploaded rules could point ``panels:`` at any path on the server.

    The same file-read as the path box, one level down, and the pandas error
    for a file that was not a panel quoted its first cell.
    """
    secret = tmp_path / "clients.csv"
    secret.write_text("client,iban\nACME-SECRET-CLIENT,ES00-SECRET\n")
    solved_app.session_state["universe_rules_text"] = (
        "schema_version: 1\n"
        f"panels:\n  adv: {{path: '{secret.as_posix()}'}}\n"
        "rules:\n  - {kind: threshold, panel: adv, op: '>', value: 0}\n"
    )
    solved_app.run()
    _no_exception(solved_app)
    errors = [e.value for e in solved_app.error]
    assert any("panels" in message for message in errors), errors
    assert not any("SECRET" in message for message in errors), errors
