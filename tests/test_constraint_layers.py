"""Layered (multi-level) allocation constraints.

The mandate under test throughout is the one an allocator actually writes:
at most 60% equity, 30% fixed income, 10% commodities; inside equity, a limit
on developed and emerging; and a currency limit cutting across both.
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

from optimization_engine.config import EngineConfig, OptimizerSpec, load_config, save_config
from optimization_engine.constraints import (
    BASIS_PARENT,
    ConstraintLayer,
    LayerConfigurationError,
    currency_layer,
    effective_layers,
    layer_breaches,
    layer_exposures,
    layer_from_mapping,
)
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.diagnostics import check_constraints
from optimization_engine.optimizers.feasibility import analyze_feasibility

TOL = 1e-6

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
CURRENCY = {
    "US_Equity": "USD",
    "Intl_Equity": "EUR",
    "EM_Equity": "EUR",
    "Real_Estate": "USD",
    "Infra": "USD",
    "Commodities": "EUR",
    "Gold": "EUR",
    "US_Treasuries": "USD",
    "TIPS": "USD",
    "IG_Credit": "USD",
    "HY_Credit": "USD",
    "EM_Debt": "EUR",
    "Cash": "USD",
}

CLASS_BUDGETS = {
    "Equity": [0.0, 0.60],
    "Fixed Income": [0.0, 0.30],
    "Commodities": [0.0, 0.10],
    "Alternatives": [0.0, 0.20],
    "Cash": [0.0, 0.15],
}


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=11))


def _config(returns: pd.DataFrame, layers, method: str = "mean_variance") -> EngineConfig:
    expected = (1 + returns).prod() ** (252 / len(returns)) - 1
    return EngineConfig(
        expected_returns=expected.to_dict(),
        groups=ASSET_CLASS,
        group_bounds=CLASS_BUDGETS,
        constraint_layers=list(layers),
        optimizer=OptimizerSpec(name=method, risk_free_rate=0.02, risk_aversion=2.0),
    )


def _sub_layer(**kwargs) -> ConstraintLayer:
    return layer_from_mapping(
        "Sub-asset class",
        SUB_CLASS,
        {
            "DM Equity": 0.40,
            "EM Equity": 0.20,
            "DM Fixed Income": 0.20,
            "EM Fixed Income": 0.10,
        },
        **kwargs,
    )


def _fx_layer() -> ConstraintLayer:
    return currency_layer(
        "FX exposure", CURRENCY, "USD", local_max=0.70, foreign_max=0.30
    )


def _bucket_weight(weights: pd.Series, assignments: dict, bucket: str) -> float:
    return float(sum(w for a, w in weights.items() if assignments.get(a) == bucket))


# ---------------------------------------------------------------------------
# The layer object itself
# ---------------------------------------------------------------------------


def test_a_bare_number_limit_is_read_as_a_cap_with_no_floor():
    layer = layer_from_mapping("L", {"A": "X"}, {"X": 0.25})
    assert layer.limits["X"] == (0.0, 0.25)


def test_a_layer_round_trips_through_its_dict_form():
    layer = _sub_layer()
    assert ConstraintLayer.from_dict(layer.to_dict()) == layer


def test_a_relative_layer_without_a_parent_is_rejected_at_construction():
    with pytest.raises(LayerConfigurationError, match="names no parent"):
        ConstraintLayer(name="L", assignments={"A": "X"}, limits={"X": (0, 1)},
                        basis=BASIS_PARENT)


def test_an_unknown_basis_is_rejected():
    with pytest.raises(LayerConfigurationError, match="basis"):
        ConstraintLayer(name="L", basis="sideways")


def test_the_legacy_grouping_becomes_the_first_layer():
    cons = PortfolioConstraints(
        groups=ASSET_CLASS,
        group_bounds={k: tuple(v) for k, v in CLASS_BUDGETS.items()},
        constraint_layers=(_fx_layer(),),
    )
    names = [lyr.name for lyr in cons.layers]
    assert names == ["Asset class", "FX exposure"]
    assert cons.has_layer_limits


def test_layers_survive_a_yaml_round_trip(tmp_path, returns):
    config = _config(returns, [_sub_layer(), _fx_layer()])
    path = tmp_path / "policy.yaml"
    save_config(config, path)
    back = load_config(path)
    assert [lyr.to_dict() for lyr in back.constraint_layers] == [
        lyr.to_dict() for lyr in config.constraint_layers
    ]


def test_a_currency_layer_splits_on_the_base_currency():
    layer = currency_layer("FX", CURRENCY, "USD", foreign_max=0.30)
    assert layer.assignments["Intl_Equity"] == "Foreign FX"
    assert layer.assignments["US_Equity"] == "Local FX (USD)"
    assert layer.limits["Foreign FX"] == (0.0, 0.30)


# ---------------------------------------------------------------------------
# Enforcement inside the solve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["mean_variance", "min_variance", "max_sharpe", "risk_parity", "cvar"],
)
def test_three_layers_bind_at_once_for_the_hard_methods(returns, method):
    config = _config(returns, [_sub_layer(), _fx_layer()], method=method)
    run = run_engine(returns, config)
    w = run.result.weights

    assert _bucket_weight(w, ASSET_CLASS, "Equity") <= 0.60 + TOL
    assert _bucket_weight(w, ASSET_CLASS, "Fixed Income") <= 0.30 + TOL
    assert _bucket_weight(w, ASSET_CLASS, "Commodities") <= 0.10 + TOL
    assert _bucket_weight(w, SUB_CLASS, "EM Equity") <= 0.20 + TOL
    assert _bucket_weight(w, SUB_CLASS, "DM Equity") <= 0.40 + TOL
    fx = _fx_layer().assignments
    assert _bucket_weight(w, fx, "Foreign FX") <= 0.30 + TOL
    assert run.result.is_compliant, run.result.violations


@pytest.mark.parametrize(
    "method", ["hrp", "herc", "equal_weight", "inverse_vol", "max_diversification"]
)
def test_the_projection_methods_are_moved_into_the_layered_policy(returns, method):
    """These allocate first and constrain after, but must still end compliant."""
    config = _config(returns, [_sub_layer(), _fx_layer()], method=method)
    with pytest.warns(UserWarning) if method in ("hrp", "herc") else _noop():
        run = run_engine(returns, config)
    w = run.result.weights
    assert _bucket_weight(w, ASSET_CLASS, "Equity") <= 0.60 + 1e-4
    assert _bucket_weight(w, SUB_CLASS, "EM Equity") <= 0.20 + 1e-4
    assert _bucket_weight(w, _fx_layer().assignments, "Foreign FX") <= 0.30 + 1e-4


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("method", ["mean_variance", "max_sharpe", "risk_parity"])
def test_a_percent_of_parent_limit_binds_against_the_solved_parent(returns, method):
    """The point of the relative basis: the cap moves with the equity sleeve.

    ``max_sharpe`` is the interesting case — it solves a homogeneous
    reformulation, and a limit that did not carry through the change of
    variables would silently vanish there.
    """
    layer = layer_from_mapping(
        "Sub-asset class",
        SUB_CLASS,
        {
            "DM Equity": 0.75,
            "EM Equity": 0.25,
            "DM Fixed Income": 0.80,
            "EM Fixed Income": 0.30,
        },
        basis=BASIS_PARENT,
        parent="Asset class",
    )
    run = run_engine(returns, _config(returns, [layer], method=method))
    w = run.result.weights
    equity = _bucket_weight(w, ASSET_CLASS, "Equity")
    em = _bucket_weight(w, SUB_CLASS, "EM Equity")
    assert equity > 0.01, "the test needs a non-trivial equity sleeve"
    assert em <= 0.25 * equity + 1e-6
    assert run.result.is_compliant, run.result.violations


def test_a_percent_of_parent_cap_is_looser_than_the_same_number_of_the_book(returns):
    """25% of a 40% sleeve is 10% of the book, and the solver should see that."""
    relative = layer_from_mapping(
        "Sub", SUB_CLASS, {"EM Equity": 0.25}, basis=BASIS_PARENT, parent="Asset class"
    )
    absolute = layer_from_mapping("Sub", SUB_CLASS, {"EM Equity": 0.25})
    em_rel = _bucket_weight(
        run_engine(returns, _config(returns, [relative])).result.weights,
        SUB_CLASS, "EM Equity",
    )
    em_abs = _bucket_weight(
        run_engine(returns, _config(returns, [absolute])).result.weights,
        SUB_CLASS, "EM Equity",
    )
    assert em_rel < em_abs - 1e-4


def test_a_layer_left_uncapped_changes_nothing(returns):
    free = layer_from_mapping("Sub", SUB_CLASS, {b: 1.0 for b in set(SUB_CLASS.values())})
    with_layer = run_engine(returns, _config(returns, [free])).result.weights
    without = run_engine(returns, _config(returns, [])).result.weights
    pd.testing.assert_series_equal(with_layer, without, atol=1e-6)


def test_an_impossible_layer_is_named_before_the_solve(returns):
    """Caps of 5% over the whole universe cannot fund a fully-invested book."""
    layer = layer_from_mapping(
        "FX", {a: "Foreign" for a in ASSET_CLASS}, {"Foreign": 0.05}
    )
    config = _config(returns, [layer])
    report = analyze_feasibility(
        list(returns.columns),
        PortfolioConstraints(
            groups=ASSET_CLASS,
            group_bounds={k: tuple(v) for k, v in CLASS_BUDGETS.items()},
            constraint_layers=(layer,),
        ),
    )
    assert not report.is_feasible
    codes = {i.code for i in report.issues}
    assert "group_maxes_below_budget" in codes
    assert "FX" in report.describe()
    del config


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_exposures_restate_a_relative_cap_as_a_share_of_the_book():
    weights = pd.Series({"A": 0.30, "B": 0.20, "C": 0.50})
    parent = layer_from_mapping("Asset class", {"A": "Eq", "B": "Eq", "C": "FI"}, {})
    child = layer_from_mapping(
        "Sub", {"A": "DM", "B": "EM"}, {"EM": 0.50}, basis=BASIS_PARENT,
        parent="Asset class",
    )
    table = layer_exposures(weights, [parent, child]).set_index("bucket")
    assert table.loc["EM", "max"] == pytest.approx(0.50)
    # Half of the 50% equity sleeve.
    assert table.loc["EM", "effective_max"] == pytest.approx(0.25)
    assert table.loc["EM", "headroom"] == pytest.approx(0.05)


def test_a_zero_floor_met_by_a_zero_weight_is_not_reported_as_binding():
    weights = pd.Series({"A": 0.0, "B": 1.0})
    layer = layer_from_mapping("L", {"A": "X", "B": "Y"}, {"X": 0.10, "Y": 1.0})
    table = layer_exposures(weights, [layer]).set_index("bucket")
    assert not bool(table.loc["X", "binding"])
    assert bool(table.loc["Y", "binding"])


def test_a_breach_on_any_layer_is_reported_with_the_layer_that_owns_it():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    cons = PortfolioConstraints(
        groups={"A": "Eq", "B": "FI"},
        group_bounds={"Eq": (0.0, 0.30)},
        constraint_layers=(layer_from_mapping("FX", {"A": "Foreign"}, {"Foreign": 0.10}),),
    )
    labels = [v.label for v in check_constraints(weights, cons)]
    assert "Asset class | Eq upper bound" in labels
    assert "FX | Foreign upper bound" in labels

    breaches = layer_breaches(weights, cons.layers)
    assert {b[1] for b in breaches} == {"max"}


def test_the_run_reports_which_bucket_is_binding(returns):
    run = run_engine(returns, _config(returns, [_sub_layer(), _fx_layer()]))
    table = run.layer_exposures()
    assert set(table["layer"]) == {"Asset class", "Sub-asset class", "FX exposure"}
    assert table["binding"].any()
    assert "constraint_layers" in run.assumptions()


def test_risk_shares_can_be_read_off_any_layer(returns):
    run = run_engine(returns, _config(returns, [_fx_layer()]))
    shares = run.layer_risk_contributions("FX exposure")
    assert set(shares.index) <= set(_fx_layer().assignments.values())
    assert float(shares.sum()) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="No layer named"):
        run.layer_risk_contributions("Nope")


def test_the_workbook_carries_the_realized_policy(tmp_path, returns):
    from optimization_engine.reporting.exporters import run_sheets

    run = run_engine(returns, _config(returns, [_sub_layer()]))
    sheets = run_sheets(run)
    assert "allocation_layers" in sheets
    assert not sheets["allocation_layers"].empty


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


def test_a_bucket_straddling_two_parents_is_refused_rather_than_guessed(returns):
    """"40% of the parent" has no meaning when the members sit in two parents."""
    straddling = layer_from_mapping(
        "Sub",
        {"US_Equity": "Mixed", "US_Treasuries": "Mixed"},
        {"Mixed": 0.5},
        basis=BASIS_PARENT,
        parent="Asset class",
    )
    cons = PortfolioConstraints(
        groups=ASSET_CLASS,
        group_bounds={k: tuple(v) for k, v in CLASS_BUDGETS.items()},
        constraint_layers=(straddling,),
    )
    report = analyze_feasibility(list(returns.columns), cons)
    assert "ambiguous_parent_bucket" in {i.code for i in report.issues}

    with pytest.raises(LayerConfigurationError, match="more than one"):
        run_engine(returns, _config(returns, [straddling]))


def test_a_relative_layer_pointing_at_nothing_is_named(returns):
    orphan = layer_from_mapping(
        "Sub", SUB_CLASS, {"EM Equity": 0.2}, basis=BASIS_PARENT, parent="Ghost layer"
    )
    cons = PortfolioConstraints(
        groups=ASSET_CLASS,
        group_bounds={k: tuple(v) for k, v in CLASS_BUDGETS.items()},
        constraint_layers=(orphan,),
    )
    report = analyze_feasibility(list(returns.columns), cons)
    assert "missing_parent_layer" in {i.code for i in report.issues}


def test_sub_caps_that_cannot_fill_their_sleeve_are_flagged_as_a_warning(returns):
    """30/30 inside a sleeve the allocator means to fill is almost always a typo."""
    layer = layer_from_mapping(
        "Sub",
        {"US_Equity": "DM", "Intl_Equity": "DM", "EM_Equity": "EM"},
        {"DM": 0.30, "EM": 0.30},
        basis=BASIS_PARENT,
        parent="Asset class",
    )
    cons = PortfolioConstraints(
        groups=ASSET_CLASS,
        group_bounds={k: tuple(v) for k, v in CLASS_BUDGETS.items()},
        constraint_layers=(layer,),
    )
    report = analyze_feasibility(list(returns.columns), cons)
    issue = next(i for i in report.issues if i.code == "relative_caps_below_parent")
    assert not issue.fatal
    assert "60%" in issue.message


def test_effective_layers_accepts_the_mapping_form_from_yaml():
    cons = PortfolioConstraints(
        constraint_layers=[
            {"name": "FX", "assignments": {"A": "Local"}, "limits": {"Local": [0, 0.7]}}
        ]
    )
    layers = effective_layers(cons)
    assert layers[0].limits["Local"] == (0.0, 0.7)


def test_an_empty_bucket_with_a_floor_is_fatal_and_without_one_is_not(returns):
    def report_for(floor):
        layer = ConstraintLayer(
            name="Sub", assignments=SUB_CLASS, limits={"Nobody": (floor, 0.5)}
        )
        return analyze_feasibility(
            list(returns.columns),
            PortfolioConstraints(constraint_layers=(layer,)),
        )

    assert not report_for(0.10).is_feasible
    assert report_for(0.0).is_feasible


def test_weights_outside_the_universe_do_not_confuse_the_exposures():
    """A layer naming assets the panel does not hold reports the rest correctly."""
    weights = pd.Series({"A": 0.6, "B": 0.4})
    layer = layer_from_mapping("L", {"A": "X", "Z": "X", "B": "Y"}, {"X": 0.7})
    table = layer_exposures(weights, [layer]).set_index("bucket")
    assert table.loc["X", "weight"] == pytest.approx(0.6)
    assert np.isnan(table.loc["Y", "max"])


# ---------------------------------------------------------------------------
# Floors, which are the other half of every layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mean_variance", "max_sharpe", "risk_parity"])
def test_a_bucket_floor_is_honoured_as_a_share_of_the_portfolio(returns, method):
    layer = ConstraintLayer(
        name="Sub-asset class",
        assignments=SUB_CLASS,
        limits={"EM Equity": (0.08, 0.20), "DM Fixed Income": (0.05, 0.25)},
    )
    run = run_engine(returns, _config(returns, [layer], method=method))
    w = run.result.weights
    assert _bucket_weight(w, SUB_CLASS, "EM Equity") >= 0.08 - TOL
    assert _bucket_weight(w, SUB_CLASS, "DM Fixed Income") >= 0.05 - TOL
    assert run.result.is_compliant, run.result.violations


@pytest.mark.parametrize("method", ["mean_variance", "max_sharpe"])
def test_a_floor_stated_as_a_share_of_the_parent_scales_with_it(returns, method):
    """"At least 70% of the equity sleeve must be developed."""
    layer = ConstraintLayer(
        name="Sub-asset class",
        assignments=SUB_CLASS,
        limits={"DM Equity": (0.70, 1.0), "EM Equity": (0.0, 0.30)},
        basis=BASIS_PARENT,
        parent="Asset class",
    )
    run = run_engine(returns, _config(returns, [layer], method=method))
    w = run.result.weights
    equity = _bucket_weight(w, ASSET_CLASS, "Equity")
    assert equity > 0.01
    assert _bucket_weight(w, SUB_CLASS, "DM Equity") >= 0.70 * equity - 1e-6
    assert run.result.is_compliant, run.result.violations


def test_two_buckets_sharing_a_parent_each_get_their_own_limit(returns):
    """The parent row repeats, once per child, and the caps must not merge."""
    layer = ConstraintLayer(
        name="Equity split",
        assignments={
            "US_Equity": "US", "Intl_Equity": "Non-US", "EM_Equity": "Non-US"
        },
        limits={"US": (0.0, 0.60), "Non-US": (0.0, 0.40)},
        basis=BASIS_PARENT,
        parent="Asset class",
    )
    run = run_engine(returns, _config(returns, [layer]))
    w = run.result.weights
    equity = _bucket_weight(w, ASSET_CLASS, "Equity")
    assert w["US_Equity"] <= 0.60 * equity + 1e-6
    assert w["Intl_Equity"] + w["EM_Equity"] <= 0.40 * equity + 1e-6
    assert run.result.is_compliant, run.result.violations


def test_a_child_whose_parent_holds_nothing_is_dropped_rather_than_forced(returns):
    """A parent bucket outside the universe leaves the child at zero, not broken."""
    layer = ConstraintLayer(
        name="Sub",
        assignments={"US_Equity": "DM Equity"},
        limits={"DM Equity": (0.0, 0.50), "Ghost": (0.0, 0.30)},
        basis=BASIS_PARENT,
        parent="Asset class",
    )
    run = run_engine(returns, _config(returns, [layer]))
    equity = _bucket_weight(run.result.weights, ASSET_CLASS, "Equity")
    assert run.result.weights["US_Equity"] <= 0.50 * equity + 1e-6
