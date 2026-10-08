"""Black-Litterman must check, impose and report the problem it actually solves.

Three places used to disagree with the solve:

* the pre-flight built its posterior with the configured δ even when the solve
  implied δ from the market's Sharpe ratio, so a reachable return target was
  reported as fatal;
* the registry said Black-Litterman could not impose a tracking-error budget
  while the solve imposed one, measured against the posterior covariance
  ``Σ + M`` rather than ``Σ`` like every other method and like the audit.
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

from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.engine import run_engine
from optimization_engine.optimizers.factory import (
    effective_expected_returns,
)

NAMES = ["a", "b", "c"]
VOL = np.array([0.10, 0.15, 0.20])
CORR = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    sigma = np.outer(VOL, VOL) * CORR
    draws = rng.multivariate_normal(np.array([0.06, 0.08, 0.10]) / 252, sigma / 252, 800)
    return pd.DataFrame(
        draws, columns=NAMES, index=pd.bdate_range("2018-01-01", periods=800)
    )


def _calibrated(**extra) -> OptimizerSpec:
    return OptimizerSpec(
        name="black_litterman",
        bl_calibrate_risk_aversion=True,
        bl_market_return=0.12,
        risk_aversion=2.5,
        risk_free_rate=0.02,
        **extra,
    )


def test_the_preflight_posterior_is_the_one_the_solve_optimizes(history):
    """With δ implied from the market, the pre-flight used the configured 2.5.

    The solve implied δ = 6.95 from a 12% market return, so its posterior was
    [7.7%, 11.9%, 16.3%] while the pre-flight checked targets against
    [4.1%, 5.6%, 7.2%]. Both now come from one computation.
    """
    cfg = EngineConfig(optimizer=_calibrated(), covariance_method="sample")
    run = run_engine(history, cfg)

    implied = run.result.extras["implied_risk_aversion"]
    assert implied == pytest.approx(6.95, abs=0.01)
    preflight = effective_expected_returns(cfg, run.cov_matrix, run.expected_returns)
    pd.testing.assert_series_equal(
        preflight,
        run.result.extras["bl_posterior_returns"],
        check_names=False,
    )


def test_a_target_the_solve_reaches_is_not_called_unreachable(history):
    """The 16.22% target sits inside the solve's posterior range (max 16.32%).

    The pre-flight measured it against the uncalibrated posterior, whose
    maximum is 7.2%, and reported ``target_return_too_high`` as fatal — so
    ``raise_on_infeasible`` refused a mandate the solve then met exactly.
    """
    probe = run_engine(
        history, EngineConfig(optimizer=_calibrated(), covariance_method="sample")
    )
    target = float(probe.result.extras["bl_posterior_returns"].max()) - 0.001
    cfg = EngineConfig(
        optimizer=_calibrated(target_return=target), covariance_method="sample"
    )

    run = run_engine(history, cfg, raise_on_infeasible=True)

    assert run.feasibility.is_feasible, run.feasibility.describe()
    assert run.result.expected_return == pytest.approx(target, abs=1e-6)


def test_black_litterman_binds_a_tracking_error_budget_against_sigma(history, caplog):
    """The registry said the limit was not imposed; the solve imposed it on Σ + M.

    The sub-solve handed the mandate to a mean-variance optimizer built on the
    posterior covariance, so the 1% budget was measured on ``Σ + M`` — tighter
    than the ``Σ`` the audit and every other method measure it on. The book
    came back at 0.98% against ``Σ``, inside the limit but not at it, while the
    factory warned that the limit "will be reported ... but not enforced".
    """
    from optimization_engine.optimizers.requirements import requirements_for

    equal = {a: 1.0 / len(NAMES) for a in NAMES}
    cfg = EngineConfig(
        optimizer=OptimizerSpec(name="black_litterman", bl_views={"c": 0.30}),
        benchmark_weights=equal,
        max_tracking_error=0.01,
        covariance_method="sample",
    )
    with caplog.at_level("WARNING"):
        run = run_engine(history, cfg)

    assert requirements_for("black_litterman").supports_benchmark_limits
    assert not [r for r in caplog.records if "cannot impose" in r.getMessage()]
    active = run.result.weights.values - np.array([equal[a] for a in NAMES])
    tracking_error = float(np.sqrt(active @ run.cov_matrix.values @ active))
    assert tracking_error == pytest.approx(0.01, abs=1e-5), "the budget binds on Σ"
    assert run.result.audit.is_clean, run.result.audit.describe()
