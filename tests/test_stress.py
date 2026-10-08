"""Shock-based stress testing: the P&L, its decomposition, and what it refuses.

The identity that matters here is that per-asset contributions sum to the
scenario's P&L — not approximately, but to the accuracy of a float sum, because
they are the terms of it. Everything else is a guard against a stress report
that quietly understates the loss it was asked to compute.
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

from optimization_engine.stress import (  # noqa: E402
    UNKNOWN_ASSET_POLICIES,
    ScenarioStress,
    Shock,
    StressError,
    StressReport,
    stress_test,
)

ASSETS = ["EQ_US", "EQ_EU", "BOND", "GOLD"]


@pytest.fixture
def weights() -> pd.Series:
    return pd.Series({"EQ_US": 0.45, "EQ_EU": 0.25, "BOND": 0.20, "GOLD": 0.10})


@pytest.fixture
def cov_matrix() -> pd.DataFrame:
    """An annualized covariance over the same four names.

    Built from volatilities and a correlation matrix so the numbers are
    recognizable: 18%, 20%, 6% and 15% annualized.
    """
    vol = np.array([0.18, 0.20, 0.06, 0.15])
    corr = np.array(
        [
            [1.00, 0.85, -0.20, 0.05],
            [0.85, 1.00, -0.15, 0.05],
            [-0.20, -0.15, 1.00, 0.25],
            [0.05, 0.05, 0.25, 1.00],
        ]
    )
    cov = corr * np.outer(vol, vol)
    return pd.DataFrame(cov, index=ASSETS, columns=ASSETS)


@pytest.fixture
def shocks() -> list[Shock]:
    return [
        Shock(
            name="2008",
            returns={"EQ_US": -0.38, "EQ_EU": -0.44, "BOND": 0.05, "GOLD": 0.06},
            covariance_scale=4.0,
            notes="Global financial crisis, peak-to-trough.",
        ),
        Shock(
            name="Rates +200bp",
            returns={"BOND": -0.12, "EQ_US": -0.05, "EQ_EU": -0.05},
        ),
        Shock(name="Melt-up", returns={"EQ_US": 0.15, "EQ_EU": 0.18, "GOLD": -0.08}),
    ]


# ---------------------------------------------------------------------------
# The four acceptance tests named in the spec
# ---------------------------------------------------------------------------


def test_stress_contributions_sum(weights: pd.Series, shocks: list[Shock], cov_matrix):
    """Per-asset contributions add to the scenario P&L, to 1e-12.

    They are the terms of the sum, so this is an identity rather than an
    agreement between two calculations. The tolerance is there for float
    summation order alone.
    """
    report = stress_test(weights, shocks, cov_matrix=cov_matrix)
    assert len(report.scenarios) == len(shocks)
    for scenario in report.scenarios:
        assert float(scenario.contributions.sum()) == pytest.approx(
            scenario.pnl, abs=1e-12
        )
        # And the contributions are per-asset, over the whole book.
        assert list(scenario.contributions.index) == list(weights.index)

    # The frame view carries the same identity, row by row.
    frame = report.contributions_frame()
    for name, row in frame.iterrows():
        expected = report.to_frame().loc[name, "pnl"]
        assert float(row.sum()) == pytest.approx(float(expected), abs=1e-12)


def test_stress_worst_named(weights: pd.Series, shocks: list[Shock], cov_matrix):
    """``worst`` names the scenario, its P&L, and the position that drove it."""
    report = stress_test(weights, shocks, cov_matrix=cov_matrix)
    worst = report.worst

    # 2008 is the deepest of the three by construction.
    assert worst.name == "2008"
    expected = 0.45 * -0.38 + 0.25 * -0.44 + 0.20 * 0.05 + 0.10 * 0.06
    assert worst.pnl == pytest.approx(expected, abs=1e-12)
    assert worst.pnl == min(s.pnl for s in report.scenarios)

    # EQ_US: 45% of the book falling 38% is the largest single line.
    assert worst.largest_contributor == "EQ_US"
    assert worst.largest_contribution == pytest.approx(0.45 * -0.38, abs=1e-12)
    assert worst.name in report.describe()
    assert "EQ_US" in report.describe()


def test_stress_missing_asset_is_zero_shock(weights: pd.Series):
    """An asset the *shock* does not name is unmoved, not undefined.

    This is the spec's stated semantics for a shock mapping that is narrower
    than the book: absent means 0.0. The opposite direction — a shock naming
    an asset the *book* does not hold — is a different question, and is
    covered by ``test_stress_shock_on_an_unheld_asset_raises`` below.
    """
    partial = Shock(name="Equity only", returns={"EQ_US": -0.20})
    report = stress_test(weights, [partial])
    scenario = report.scenarios[0]

    assert scenario.pnl == pytest.approx(0.45 * -0.20, abs=1e-12)
    assert scenario.contributions["EQ_EU"] == 0.0
    assert scenario.contributions["BOND"] == 0.0
    assert scenario.contributions["GOLD"] == 0.0
    assert set(scenario.contributions.index) == set(weights.index)
    # Nothing was dropped: the book is fully covered by the (implicit) zeros.
    assert scenario.ignored_assets == ()


def test_stress_covariance_scale(weights: pd.Series, cov_matrix: pd.DataFrame):
    """A scalar multiplier scales the variance, so volatility scales by its root."""
    quiet = Shock(name="No risk clause", returns={"EQ_US": -0.10})
    doubled = Shock(name="Vol doubles", returns={"EQ_US": -0.10}, covariance_scale=4.0)
    halved = Shock(name="Vol halves", returns={"EQ_US": -0.10}, covariance_scale=0.25)

    report = stress_test(weights, [quiet, doubled, halved], cov_matrix=cov_matrix)
    base = report.base_volatility
    assert base is not None and base > 0.0

    by_name = {s.name: s for s in report.scenarios}
    # No clause: risk stays where it was.
    assert by_name["No risk clause"].stressed_volatility == pytest.approx(base, rel=1e-12)
    assert by_name["No risk clause"].volatility_ratio == pytest.approx(1.0, rel=1e-12)
    # ×4 on the covariance is ×2 on the volatility.
    assert by_name["Vol doubles"].stressed_volatility == pytest.approx(2 * base, rel=1e-12)
    assert by_name["Vol doubles"].volatility_ratio == pytest.approx(2.0, rel=1e-12)
    assert by_name["Vol halves"].volatility_ratio == pytest.approx(0.5, rel=1e-12)

    # The P&L is untouched by the risk clause: they are separate statements.
    for scenario in report.scenarios:
        assert scenario.pnl == pytest.approx(0.45 * -0.10, abs=1e-12)

    # And without a covariance, every volatility field is None rather than 0.
    silent = stress_test(weights, [doubled])
    assert silent.base_volatility is None
    assert silent.scenarios[0].stressed_volatility is None
    assert silent.scenarios[0].volatility_ratio is None


# ---------------------------------------------------------------------------
# The unknown-asset posture
# ---------------------------------------------------------------------------


def test_stress_shock_on_an_unheld_asset_raises(weights: pd.Series):
    """A shock on a name the book cannot hold is refused, and says why.

    The same defect as a Black-Litterman view on an asset outside the
    universe, which this engine also refuses: the scenario's loss on that name
    cannot reach the portfolio, so the P&L reported is quietly smaller than
    the scenario describes and nothing in the output says so.
    """
    shock = Shock(name="EM crisis", returns={"EQ_EM": -0.45, "EQ_US": -0.10})
    with pytest.raises(StressError, match="EQ_EM"):
        stress_test(weights, [shock])


def test_stress_unheld_asset_can_be_ignored_but_is_recorded(weights: pd.Series):
    """The explicit way out narrows the scenario visibly, never silently."""
    shock = Shock(name="EM crisis", returns={"EQ_EM": -0.45, "EQ_US": -0.10})
    report = stress_test(weights, [shock], unknown_assets="ignore")
    scenario = report.scenarios[0]

    assert scenario.ignored_assets == ("EQ_EM",)
    assert scenario.pnl == pytest.approx(0.45 * -0.10, abs=1e-12)
    assert "EQ_EM" in scenario.describe()
    assert "EQ_EM" in report.to_frame().loc["EM crisis", "ignored_assets"]
    assert report.metadata["unknown_assets"] == "ignore"


def test_stress_rejects_an_unknown_policy(weights: pd.Series):
    shock = Shock(name="X", returns={"EQ_US": -0.1})
    with pytest.raises(StressError, match="unknown_assets"):
        stress_test(weights, [shock], unknown_assets="drop")
    assert UNKNOWN_ASSET_POLICIES == ("raise", "ignore")


# ---------------------------------------------------------------------------
# Ordering and reporting
# ---------------------------------------------------------------------------


def test_stress_describe_orders_scenarios_worst_first(
    weights: pd.Series, shocks: list[Shock], cov_matrix
):
    report = stress_test(weights, shocks, cov_matrix=cov_matrix)
    ordered = report.by_severity()
    assert [s.pnl for s in ordered] == sorted(s.pnl for s in report.scenarios)
    assert ordered[0].name == "2008"
    assert ordered[-1].name == "Melt-up"

    lines = report.describe().splitlines()
    positions = [next(i for i, ln in enumerate(lines) if s.name in ln) for s in ordered]
    assert positions == sorted(positions)

    # The frame views agree with describe()'s order.
    assert list(report.to_frame().index) == [s.name for s in ordered]
    assert list(report.contributions_frame().index) == [s.name for s in ordered]
    assert list(report.contributions_frame().columns) == list(weights.index)

    # And the scenarios themselves keep the order they were given in.
    assert [s.name for s in report.scenarios] == [s.name for s in shocks]


def test_stress_report_is_the_documented_shape(weights: pd.Series, cov_matrix):
    report = stress_test(weights, [Shock("X", {"EQ_US": -0.1})], cov_matrix=cov_matrix)
    assert isinstance(report, StressReport)
    assert isinstance(report.worst, ScenarioStress)
    frame = report.to_frame()
    assert frame.index.name == "scenario"
    assert list(frame.columns) == [
        "pnl",
        "stressed_volatility",
        "base_volatility",
        "volatility_ratio",
        "largest_contributor",
        "largest_contribution",
        "ignored_assets",
        "notes",
    ]


def test_a_levered_book_stresses_as_written():
    """Weights need not sum to 1; nothing renormalizes them behind the caller."""
    book = pd.Series({"EQ_US": 1.5, "BOND": -0.5})
    report = stress_test(book, [Shock("Selloff", {"EQ_US": -0.10, "BOND": 0.02})])
    assert report.scenarios[0].pnl == pytest.approx(1.5 * -0.10 + -0.5 * 0.02, abs=1e-12)


def test_weights_accept_a_plain_mapping():
    report = stress_test({"A": 0.6, "B": 0.4}, [Shock("X", {"A": -0.5})])
    assert report.worst.pnl == pytest.approx(-0.30, abs=1e-12)
    assert list(report.weights.index) == ["A", "B"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_stress_with_no_shocks_raises(weights: pd.Series):
    """An empty report reads exactly like a passing one."""
    with pytest.raises(StressError, match="no shocks"):
        stress_test(weights, [])


def test_duplicate_shock_names_raise(weights: pd.Series):
    twice = [Shock("Same", {"EQ_US": -0.1}), Shock("Same", {"BOND": -0.1})]
    with pytest.raises(StressError, match="Duplicate"):
        stress_test(weights, twice)


def test_an_empty_book_raises():
    with pytest.raises(StressError, match="empty book"):
        stress_test(pd.Series(dtype=float), [Shock("X", {"A": -0.1})])


def test_a_non_finite_weight_raises():
    book = pd.Series({"A": 0.5, "B": float("nan")})
    with pytest.raises(StressError, match="not finite"):
        stress_test(book, [Shock("X", {"A": -0.1})], unknown_assets="ignore")


def test_a_non_finite_shock_raises():
    with pytest.raises(StressError, match="finite one-period return"):
        Shock("X", {"A": float("inf")})


def test_an_unnamed_shock_raises():
    with pytest.raises(StressError, match="needs a name"):
        Shock("   ", {"A": -0.1})


def test_a_negative_covariance_scale_raises():
    with pytest.raises(StressError, match="not a covariance matrix"):
        Shock("X", {"A": -0.1}, covariance_scale=-1.0)


def test_a_covariance_that_misses_a_position_raises(weights: pd.Series):
    partial = pd.DataFrame(
        np.eye(2) * 0.04, index=["EQ_US", "EQ_EU"], columns=["EQ_US", "EQ_EU"]
    )
    with pytest.raises(StressError, match="does not cover"):
        stress_test(weights, [Shock("X", {"EQ_US": -0.1})], cov_matrix=partial)


def test_a_shock_matrix_that_misses_a_position_raises(weights, cov_matrix):
    partial = cov_matrix.loc[["EQ_US", "EQ_EU"], ["EQ_US", "EQ_EU"]]
    shock = Shock("X", {"EQ_US": -0.1}, covariance_scale=partial)
    with pytest.raises(StressError, match="does not cover"):
        stress_test(weights, [shock], cov_matrix=cov_matrix)


def test_an_asymmetric_shock_matrix_raises():
    bad = pd.DataFrame([[0.04, 0.01], [0.03, 0.09]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(StressError, match="not symmetric"):
        Shock("X", {"A": -0.1}, covariance_scale=bad)


def test_a_non_psd_shock_matrix_raises(weights: pd.Series, cov_matrix: pd.DataFrame):
    """A matrix that implies a negative variance would report imaginary risk."""
    bad = pd.DataFrame(
        [[1.0, 5.0], [5.0, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    # Refused where it is written now, not only for a book that happens to
    # load on the negative eigenvector.
    with pytest.raises(StressError, match="not positive semi-definite"):
        Shock("X", {"A": -0.1}, covariance_scale=bad)


# ---------------------------------------------------------------------------
# The full-matrix form
# ---------------------------------------------------------------------------


def test_a_full_stressed_covariance_replaces_the_base(weights, cov_matrix):
    """Correlations going to 1 in a crisis is the case a scalar cannot express."""
    vol = pd.Series({"EQ_US": 0.36, "EQ_EU": 0.40, "BOND": 0.12, "GOLD": 0.30})
    ones = np.ones((len(ASSETS), len(ASSETS)))
    crisis = pd.DataFrame(
        ones * np.outer(vol.loc[ASSETS], vol.loc[ASSETS]), index=ASSETS, columns=ASSETS
    )
    shock = Shock("Correlations to 1", {"EQ_US": -0.30}, covariance_scale=crisis)
    report = stress_test(weights, [shock], cov_matrix=cov_matrix)
    scenario = report.scenarios[0]

    # Perfect correlation: portfolio vol is the weighted sum of the vols.
    expected = float((weights.loc[ASSETS] * vol.loc[ASSETS]).sum())
    assert scenario.stressed_volatility == pytest.approx(expected, rel=1e-10)
    assert scenario.stressed_volatility > report.base_volatility
    assert "covariance replaced" in shock.describe()


def test_a_nested_mapping_is_accepted_as_a_matrix():
    matrix = {"A": {"A": 0.04, "B": 0.00}, "B": {"A": 0.00, "B": 0.09}}
    shock = Shock("X", {"A": -0.1}, covariance_scale=matrix)
    assert isinstance(shock.covariance_scale, pd.DataFrame)
    book = pd.Series({"A": 1.0, "B": 0.0})
    base = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
    report = stress_test(book, [shock], cov_matrix=base)
    assert report.scenarios[0].stressed_volatility == pytest.approx(0.2, rel=1e-12)


# ---------------------------------------------------------------------------
# Engine and tearsheet wiring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    from optimization_engine.data.loader import prices_to_returns, sample_dataset

    return prices_to_returns(sample_dataset(n_periods=252 * 3, seed=11))


@pytest.fixture(scope="module")
def engine_config(panel: pd.DataFrame):
    from optimization_engine.config import EngineConfig, OptimizerSpec

    return EngineConfig(
        expected_returns={a: 0.06 for a in panel.columns},
        bounds={a: [0.0, 1.0] for a in panel.columns},
        optimizer=OptimizerSpec(name="equal_weight"),
    )


def _panel_shock(panel: pd.DataFrame) -> Shock:
    return Shock(
        name="Everything falls",
        returns={a: -0.20 for a in panel.columns},
        covariance_scale=4.0,
    )


def test_run_engine_does_not_stress_unless_asked(panel, engine_config):
    """The default is off, because the walk-forward solvers call this per window."""
    from optimization_engine.engine import run_engine

    # ``EngineConfig.stress`` is set here as an attribute so this test pins the
    # engine's side of the seam whether or not the config field has landed.
    engine_config.stress = (_panel_shock(panel),)
    try:
        run = run_engine(panel, engine_config, check_feasibility=False)
        assert run.stress is None
    finally:
        del engine_config.stress


def test_run_engine_attaches_the_stress_report_when_asked(panel, engine_config):
    from optimization_engine.engine import run_engine

    engine_config.stress = (_panel_shock(panel),)
    try:
        run = run_engine(panel, engine_config, check_feasibility=False, run_stress=True)
    finally:
        del engine_config.stress

    assert run.stress is not None
    assert run.stress.worst.name == "Everything falls"
    # An equal-weight book fully invested in a -20% shock loses 20%.
    assert run.stress.worst.pnl == pytest.approx(-0.20, abs=1e-9)
    assert run.stress.base_volatility is not None
    assert run.stress.worst.volatility_ratio == pytest.approx(2.0, rel=1e-9)


def test_run_stress_with_no_configured_shocks_is_a_no_op(panel, engine_config):
    from optimization_engine.engine import run_engine

    run = run_engine(panel, engine_config, check_feasibility=False, run_stress=True)
    assert run.stress is None


def test_configured_shocks_refuses_a_non_shock(engine_config):
    from optimization_engine.engine import configured_shocks

    assert configured_shocks(engine_config) == ()
    engine_config.stress = ["not a shock"]
    try:
        with pytest.raises(StressError, match="written as a mapping"):
            configured_shocks(engine_config)
    finally:
        del engine_config.stress


def test_the_tearsheet_carries_a_stress_section(panel, engine_config):
    from optimization_engine.engine import run_engine

    run = run_engine(panel, engine_config, check_feasibility=False)
    shock = _panel_shock(panel)
    sheet = run.tearsheet(shocks=[shock])

    assert sheet.stress is not None
    assert sheet.stress.worst.name == "Everything falls"
    assert "Everything falls" in sheet.describe()
    frames = sheet.to_frames()
    assert "stress" in frames
    assert "stress_contributions" in frames
    assert sheet.metadata["stress_as_of"] is not None

    # No shocks, no panel — the absence is not an empty table.
    plain = run.tearsheet(shocks=())
    assert plain.stress is None
    assert "stress" not in plain.to_frames()
    assert plain.metadata["stress_as_of"] is None


# ---------------------------------------------------------------------------
# The serialization seam `EngineConfig.stress` and `--stress` are wired through
# ---------------------------------------------------------------------------


def test_a_shock_round_trips_through_its_dict_form(shocks: list[Shock]):
    from optimization_engine.stress import shocks_from_dicts, shocks_to_dicts

    back = shocks_from_dicts(shocks_to_dicts(shocks))
    assert [s.name for s in back] == [s.name for s in shocks]
    assert [s.returns for s in back] == [s.returns for s in shocks]
    assert back[0].covariance_scale == 4.0
    assert back[0].notes == shocks[0].notes

    # And a full matrix survives the trip as a nested mapping.
    matrix = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    once = Shock("M", {"A": -0.1}, covariance_scale=matrix)
    twice = Shock.from_dict(once.to_dict())
    pd.testing.assert_frame_equal(twice.covariance_scale, matrix)


def test_shocks_from_dicts_passes_shock_objects_through(shocks: list[Shock]):
    from optimization_engine.stress import shocks_from_dicts

    assert shocks_from_dicts(shocks) == tuple(shocks)
    assert shocks_from_dicts(None) == ()
    assert shocks_from_dicts([]) == ()
    with pytest.raises(StressError, match="Duplicate"):
        shocks_from_dicts([{"name": "X", "returns": {}}, {"name": "X", "returns": {}}])


def test_an_unknown_shock_key_is_refused():
    with pytest.raises(StressError, match="Unknown shock key"):
        Shock.from_dict({"name": "X", "returns": {"A": -0.1}, "covarance_scale": 2.0})


def test_shocks_load_from_a_yaml_file(tmp_path: Path):
    from optimization_engine.stress import dump_shocks_yaml, load_shocks

    path = tmp_path / "shocks.yaml"
    path.write_text(
        """
schema_version: 1
shocks:
  - name: Equity crash
    notes: -30% on both equity sleeves
    covariance_scale: 4.0
    returns:
      EQ_US: -0.30
      EQ_EU: -0.30
  - name: Rates spike
    returns:
      BOND: -0.10
""",
        encoding="utf-8",
    )
    loaded = load_shocks(path)
    assert [s.name for s in loaded] == ["Equity crash", "Rates spike"]
    assert loaded[0].covariance_scale == 4.0
    assert loaded[1].returns == {"BOND": -0.10}

    # The writer's own output reloads unchanged.
    round_tripped = tmp_path / "again.yaml"
    round_tripped.write_text(dump_shocks_yaml(loaded), encoding="utf-8")
    assert [s.name for s in load_shocks(round_tripped)] == [s.name for s in loaded]


def test_a_bare_list_of_shocks_is_accepted(tmp_path: Path):
    """The shape a person writes by hand, not only the shape the writer emits."""
    from optimization_engine.stress import load_shocks

    path = tmp_path / "shocks.yaml"
    path.write_text(
        "- name: Only one\n  returns: {A: -0.5}\n", encoding="utf-8"
    )
    assert [s.name for s in load_shocks(path)] == ["Only one"]


def test_a_shocks_file_with_the_wrong_shape_is_refused(tmp_path: Path):
    from optimization_engine.stress import load_shocks

    path = tmp_path / "shocks.yaml"
    path.write_text("schema_version: 1\nscenarios: []\n", encoding="utf-8")
    with pytest.raises(StressError, match="needs a 'shocks' key"):
        load_shocks(path)

    path.write_text("schema_version: 9\nshocks: []\n", encoding="utf-8")
    with pytest.raises(StressError, match="schema_version"):
        load_shocks(path)

    path.write_text("", encoding="utf-8")
    with pytest.raises(StressError, match="empty"):
        load_shocks(path)


def test_the_module_exports_what_it_documents():
    import optimization_engine.stress as module

    assert module.__all__ == sorted(module.__all__)
    for name in module.__all__:
        assert hasattr(module, name)
