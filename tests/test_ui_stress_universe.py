"""Pure logic behind the 💥 Stress and 🌍 Universe tabs.

The page itself is exercised in ``test_app_layers``; what lives here is
everything the page does *before* it draws anything — turning two editable
grids into scenarios and back, deciding which of the three eligibility states
a cell is in, and wording what a collapse policy just decided. Those are the
parts that can be wrong without raising, so they get a test that runs in
milliseconds rather than one that needs a page render.

The shipped example files are checked here too. ``config/shocks.yaml`` and
``config/universe.yaml`` are referenced from the README and are the first
thing anyone runs; an example that no longer parses, or one whose universe
never produces a *not evaluable* cell, is a broken example either way.

Skipped when Streamlit is not installed (``app/components.py`` imports it).
"""

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

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

sys.path.insert(0, str(ROOT / "app"))

from components import (  # noqa: E402
    ELIGIBILITY_COLORS,
    ELIGIBILITY_STATES,
    align_scenario_table,
    describe_policy_cost,
    eligibility_state_codes,
    eligibility_state_counts,
    empty_scenario_table,
    empty_shock_table,
    plot_eligibility_heatmap,
    shock_dicts_from_tables,
    tables_from_shocks,
    thin_rows,
    unheld_shocked_assets,
    validated_shock_dicts,
)

from optimization_engine.data.loader import (  # noqa: E402
    SAMPLE_UNIVERSE,
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.stress import (  # noqa: E402
    load_shocks,
    shocks_from_dicts,
    stress_test,
)
from optimization_engine.universe.rules import (  # noqa: E402
    count_unresolved,
    load_universe_rules,
)

SHOCKS_EXAMPLE = ROOT / "config" / "shocks.yaml"
UNIVERSE_EXAMPLE = ROOT / "config" / "universe.yaml"


@pytest.fixture(scope="module")
def sample_returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=520))


# ---------------------------------------------------------------------------
# The scenario grids
# ---------------------------------------------------------------------------


def test_the_grids_round_trip_the_yaml_the_cli_reads():
    """A file loaded, edited in the grid and written back is the same file.

    This is the whole reason the grid is long-form: a scenario has to survive
    the trip through the UI unchanged, or the app and the CLI are two
    different tools that happen to share a flag name.
    """
    shocks = load_shocks(SHOCKS_EXAMPLE)
    rows, meta = tables_from_shocks(shocks)
    rebuilt = shocks_from_dicts(shock_dicts_from_tables(rows, meta))
    assert [s.to_dict() for s in rebuilt] == [s.to_dict() for s in shocks]


def test_a_row_still_being_typed_is_not_a_scenario_with_a_hole_in_it():
    rows = pd.DataFrame(
        {
            "Scenario": ["Crash", "Crash", "", "Nameless leg", None],
            "Asset": ["US_Equity", "", "Gold", None, "Cash"],
            "Shock return": [-0.3, -0.2, -0.1, -0.4, np.nan],
        }
    )
    payload = shock_dicts_from_tables(rows, empty_scenario_table())
    assert [entry["name"] for entry in payload] == ["Crash"]
    assert payload[0]["returns"] == {"US_Equity": -0.3}


def test_the_last_value_wins_when_a_leg_is_typed_twice():
    rows = pd.DataFrame(
        {
            "Scenario": ["Crash", "Crash"],
            "Asset": ["US_Equity", "US_Equity"],
            "Shock return": [-0.3, -0.4],
        }
    )
    payload = shock_dicts_from_tables(rows, empty_scenario_table())
    assert payload[0]["returns"] == {"US_Equity": -0.4}


def test_scenario_order_is_the_order_they_were_first_named():
    rows = pd.DataFrame(
        {
            "Scenario": ["Second", "First", "Second"],
            "Asset": ["Gold", "US_Equity", "Cash"],
            "Shock return": [0.1, -0.3, 0.0],
        }
    )
    assert [e["name"] for e in shock_dicts_from_tables(rows, None)] == [
        "Second",
        "First",
    ]


def test_the_per_scenario_grid_follows_a_rename_rather_than_shadowing_it():
    """A multiplier must not outlive the name it was attached to.

    The failure this prevents is silent: rename a scenario in the leg grid,
    leave a stale row behind in the per-scenario grid, and the run carries a
    covariance multiplier belonging to a scenario nobody is running.
    """
    rows = pd.DataFrame(
        {
            "Scenario": ["Crash", "Squeeze"],
            "Asset": ["US_Equity", "Gold"],
            "Shock return": [-0.3, -0.1],
        }
    )
    meta = align_scenario_table(rows, None)
    meta.loc["Crash", "Covariance ×"] = 4.0
    meta.loc["Crash", "Correlation shift"] = 0.5
    meta.loc["Crash", "Notes"] = "2008-shaped"

    renamed = rows.copy()
    renamed.loc[0, "Scenario"] = "Drawdown"
    after = align_scenario_table(renamed, meta)

    assert list(after.index) == ["Drawdown", "Squeeze"]
    assert pd.isna(after.loc["Drawdown", "Covariance ×"])
    assert pd.isna(after.loc["Drawdown", "Correlation shift"])
    assert after.loc["Drawdown", "Notes"] == ""


def test_a_multiplier_survives_an_edit_that_does_not_rename_it():
    rows = pd.DataFrame(
        {"Scenario": ["Crash"], "Asset": ["US_Equity"], "Shock return": [-0.3]}
    )
    meta = align_scenario_table(rows, None)
    meta.loc["Crash", "Covariance ×"] = 4.0
    meta.loc["Crash", "Correlation shift"] = 0.5
    meta.loc["Crash", "Notes"] = "2008-shaped"

    extended = pd.concat(
        [
            rows,
            pd.DataFrame(
                {
                    "Scenario": ["Crash"],
                    "Asset": ["US_Treasuries"],
                    "Shock return": [0.08],
                }
            ),
        ],
        ignore_index=True,
    )
    after = align_scenario_table(extended, meta)
    assert after.loc["Crash", "Covariance ×"] == 4.0
    assert after.loc["Crash", "Correlation shift"] == 0.5
    payload = shock_dicts_from_tables(extended, after)
    assert payload[0]["covariance_scale"] == 4.0
    assert payload[0]["correlation_shift"] == 0.5
    assert payload[0]["notes"] == "2008-shaped"


def test_a_correlation_shift_round_trips_through_the_grids():
    """It used to be dropped: a breakdown scenario came back scaling vols only."""
    from optimization_engine.stress import Shock

    shocks = [
        Shock("Breakdown", {"US_Equity": -0.2}, covariance_scale=2.25, correlation_shift=0.5),
        Shock("Shift only", {"Gold": -0.05}, correlation_shift=0.0),
        Shock("Plain", {"US_Equity": -0.1}, covariance_scale=1.5),
    ]
    rows, meta = tables_from_shocks(shocks)
    assert meta.loc["Breakdown", "Correlation shift"] == 0.5
    assert meta.loc["Shift only", "Correlation shift"] == 0.0
    assert pd.isna(meta.loc["Plain", "Correlation shift"])

    rebuilt = shocks_from_dicts(shock_dicts_from_tables(rows, meta))
    assert [s.to_dict() for s in rebuilt] == [s.to_dict() for s in shocks]


def test_a_blank_correlation_shift_is_no_shift_rather_than_zero():
    """Blank means the scenario says nothing about correlations."""
    rows = pd.DataFrame(
        {"Scenario": ["Crash"], "Asset": ["US_Equity"], "Shock return": [-0.3]}
    )
    payload = shock_dicts_from_tables(rows, align_scenario_table(rows, None))
    assert "correlation_shift" not in payload[0]
    assert shocks_from_dicts(payload)[0].correlation_shift is None


def _library_refusal(entry: dict) -> str:
    """The message ``Shock.from_dict`` raises for ``entry``, word for word."""
    from optimization_engine.stress import Shock, StressError

    with pytest.raises(StressError) as caught:
        Shock.from_dict(entry)
    return str(caught.value)


def test_a_correlation_shift_outside_zero_to_one_is_refused_in_the_library_s_words():
    rows = pd.DataFrame(
        {"Scenario": ["Too far"], "Asset": ["US_Equity"], "Shock return": [-0.3]}
    )
    meta = align_scenario_table(rows, None)
    meta.loc["Too far", "Correlation shift"] = 1.5
    payload = shock_dicts_from_tables(rows, meta)

    usable, problems = validated_shock_dicts(payload)
    assert usable == []
    assert problems == [f"Too far: {_library_refusal(payload[0])}"]


def test_something_that_is_not_a_scenario_is_named_rather_than_dropped():
    """A negative covariance multiple is refused, and the page is told why."""
    rows = pd.DataFrame(
        {
            "Scenario": ["Fine", "Broken"],
            "Asset": ["US_Equity", "US_Equity"],
            "Shock return": [-0.3, -0.3],
        }
    )
    meta = align_scenario_table(rows, None)
    meta.loc["Broken", "Covariance ×"] = -2.0
    usable, problems = validated_shock_dicts(shock_dicts_from_tables(rows, meta))
    assert [entry["name"] for entry in usable] == ["Fine"]
    assert len(problems) == 1
    assert "Broken" in problems[0]
    assert "negative" in problems[0]


def test_an_empty_grid_is_no_scenarios_and_no_complaints():
    usable, problems = validated_shock_dicts(
        shock_dicts_from_tables(empty_shock_table(), empty_scenario_table())
    )
    assert usable == []
    assert problems == []


def test_unheld_names_are_reported_per_scenario():
    payload = [
        {"name": "Wide", "returns": {"US_Equity": -0.3, "TSLA": -0.5, "AAPL": -0.4}},
        {"name": "Narrow", "returns": {"US_Equity": -0.3}},
    ]
    assert unheld_shocked_assets(payload, ["US_Equity", "Cash"]) == {
        "Wide": ["AAPL", "TSLA"]
    }


def test_a_matrix_covariance_scale_comes_back_blank_rather_than_flattened():
    """Silently averaging a matrix into one number is a different scenario."""
    from optimization_engine.stress import Shock

    matrix = {"A": {"A": 0.04, "B": 0.01}, "B": {"A": 0.01, "B": 0.09}}
    shock = Shock(name="Regime", returns={"A": -0.2}, covariance_scale=matrix)
    _, meta = tables_from_shocks([shock])
    assert pd.isna(meta.loc["Regime", "Covariance ×"])


# ---------------------------------------------------------------------------
# The shipped example scenarios
# ---------------------------------------------------------------------------


def test_the_shipped_shocks_apply_to_the_sample_book():
    """`config/shocks.yaml` is what the README's --stress line points at."""
    shocks = load_shocks(SHOCKS_EXAMPLE)
    assert len(shocks) >= 3
    book = pd.Series(
        1.0 / len(SAMPLE_UNIVERSE), index=pd.Index(list(SAMPLE_UNIVERSE))
    )
    report = stress_test(book, list(shocks))
    for scenario in report.scenarios:
        assert scenario.pnl == pytest.approx(
            float(scenario.contributions.sum()), abs=1e-12
        )
    assert report.worst.pnl < 0.0


# ---------------------------------------------------------------------------
# The three states
# ---------------------------------------------------------------------------


def _three_state_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [True, False, None],
            "B": [None, True, True],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
        dtype="boolean",
    )


def test_each_state_gets_its_own_code():
    codes = eligibility_state_codes(_three_state_frame())
    assert codes.loc[codes.index[0], "A"] == ELIGIBILITY_STATES.index("Eligible")
    assert codes.loc[codes.index[1], "A"] == ELIGIBILITY_STATES.index("Not eligible")
    assert codes.loc[codes.index[2], "A"] == ELIGIBILITY_STATES.index("Not evaluable")
    assert set(np.unique(codes.to_numpy())) == {0, 1, 2}


def test_counts_report_a_state_that_never_occurs_rather_than_omitting_it():
    all_true = pd.DataFrame(
        {"A": [True, True]},
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
        dtype="boolean",
    )
    counts = eligibility_state_counts(all_true)
    assert set(counts) == set(ELIGIBILITY_STATES)
    assert counts["Eligible"] == 2
    assert counts["Not evaluable"] == 0


def test_not_evaluable_is_not_a_shade_of_not_eligible():
    """The heatmap's third state must be a colour of its own.

    A two-colour scale reads *not evaluable* as whichever end it lands on,
    which is exactly the collapse the universe module exists to refuse. This
    pins three distinct colours, and pins that the unknown band is not simply
    an interpolation between the other two.
    """
    assert len(set(ELIGIBILITY_COLORS)) == 3

    fig = plot_eligibility_heatmap(_three_state_frame())
    trace = fig.data[0]
    colors = {color for _, color in trace.colorscale}
    assert colors == set(ELIGIBILITY_COLORS)
    # Three flat bands, not a ramp: each colour is repeated at both ends of
    # its third of the scale, so nothing is interpolated between two states
    # that have no midpoint.
    assert len(trace.colorscale) == 2 * len(ELIGIBILITY_COLORS)
    assert set(np.unique(np.asarray(trace.z))) == {0, 1, 2}
    # And the reading never rests on colour alone.
    assert "Not evaluable" in np.asarray(trace.customdata).ravel().tolist()


def test_thin_rows_keeps_the_last_bar_it_was_given():
    frame = pd.DataFrame(
        {"A": range(100)}, index=pd.date_range("2020-01-01", periods=100, freq="D")
    )
    thinned, step = thin_rows(frame, 10)
    assert step == 10
    assert len(thinned) <= 11
    assert thinned.index[-1] == frame.index[-1]
    assert thinned.index.is_monotonic_increasing


def test_thin_rows_leaves_a_small_frame_alone():
    frame = pd.DataFrame(
        {"A": range(5)}, index=pd.date_range("2020-01-01", periods=5, freq="D")
    )
    thinned, step = thin_rows(frame, 400)
    assert step == 1
    assert thinned.equals(frame)


# ---------------------------------------------------------------------------
# The collapse policy
# ---------------------------------------------------------------------------


def test_the_policy_sentence_names_the_policy_and_what_it_decided():
    sentence = describe_policy_cost("exclude", 881, 116, ["A", "B", "C"])
    assert "881" in sentence
    assert "116" in sentence
    assert "'exclude'" in sentence
    assert "not a screen" in sentence


def test_with_no_policy_chosen_nothing_is_claimed_to_have_collapsed():
    sentence = describe_policy_cost(None, 881, 116, ["A"])
    assert "Choose a policy" in sentence
    assert "exclude" not in sentence


def test_a_universe_with_nothing_unresolved_says_the_policy_decides_nothing():
    assert "decides nothing" in describe_policy_cost("include", 0, 0, [])
    assert "decides nothing" in describe_policy_cost(None, 0, 0, [])


# ---------------------------------------------------------------------------
# The shipped example universe
# ---------------------------------------------------------------------------


def test_the_shipped_universe_needs_no_data_beside_it(sample_returns):
    """It reads only the panels the *run* supplies, so it always runs.

    An example that screened on ADV would need an ADV panel this repository
    does not ship, which makes it an example nobody can execute.
    """
    rules = load_universe_rules(UNIVERSE_EXAMPLE)
    assert rules.panels == ()
    assert set(rules.panel_names) <= {"returns", "prices"}


def test_the_shipped_universe_produces_all_three_states(sample_returns):
    """Including *not evaluable* — the state the whole module is about.

    A screen with no warm-up would draw a two-colour heatmap and teach the
    reader the wrong thing about the module they just opened.
    """
    universe = load_universe_rules(UNIVERSE_EXAMPLE).build(returns=sample_returns)
    counts = eligibility_state_counts(universe.frame)
    for state in ELIGIBILITY_STATES:
        assert counts[state] > 0, f"the example never produces {state!r}"


def test_the_shipped_universe_leaves_the_policy_something_to_decide(sample_returns):
    universe = load_universe_rules(UNIVERSE_EXAMPLE).build(returns=sample_returns)
    cells, bars, names = count_unresolved(
        universe, sample_returns.index, list(sample_returns.columns)
    )
    assert cells > 0 and bars > 0 and names
    # And it is a screen, not a pass-through: something is actually excluded.
    assert int(universe.breadth().min()) < sample_returns.shape[1]


def test_the_shipped_universe_is_a_band_rather_than_a_threshold(sample_returns):
    """Hysteresis is what stops a name churning on a threshold it sits on."""
    rules = load_universe_rules(UNIVERSE_EXAMPLE)
    assert rules.has_hysteresis
    assert rules.exit_rules
    assert rules.initial is None
