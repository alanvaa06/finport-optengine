"""Max-Sharpe must say why it has no answer, and must find one when it exists.

The tangency portfolio is solved on a ray: minimize ``y'Σy`` subject to
``(μ − rf)'y = 1`` and normalize ``w = y/Σy``. That transform has three ways to
go wrong that look alike from outside — every one used to surface as a bare
"infeasible" or as weights nobody should hold:

* no allocation the mandate allows earns more than cash, so no portfolio has a
  positive Sharpe ratio and the ray has nowhere to point;
* the best excess return is positive but tiny, so ``(μ − rf)'y = 1`` needs a
  ray so long the solver's tolerances call it infeasible; and
* with no finite box, the ratio keeps rising as the book levers up and no
  finite portfolio attains it.
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

from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.mean_variance import MaxSharpeOptimizer


def test_unbounded_tangency_is_refused_rather_than_levered_without_limit():
    """With no finite box, a long-short tangency can have no maximizer.

    When ``1'Σ⁻¹(μ − rf) < 0`` the stationary point on the budget hyperplane is
    the *minimum* Sharpe ratio, and the maximum is only approached as the book
    levers up without limit. The ray solve then pins ``κ = Σy`` at its floor of
    1e-8 and ``y/κ`` came back as ±1.9e9 — reported ``optimal``. The degenerate
    check that existed for this compared ``κ`` against 1e-10, below the floor
    the program itself imposes, so it could never fire.
    """
    names = ["p", "q"]
    vols = np.array([0.10, 0.20])
    corr = np.array([[1.0, 0.95], [0.95, 1.0]])
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=names, columns=names)
    mu = pd.Series([0.01, 0.05], index=names)
    assert np.ones(2) @ np.linalg.inv(cov.values) @ mu.values < 0, "fixture drifted"

    optimizer = MaxSharpeOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        constraints=PortfolioConstraints(
            long_only=False, bounds={a: (-np.inf, np.inf) for a in names}
        ),
    )
    with pytest.raises(SolverFailure) as raised:
        optimizer.optimize()

    assert raised.value.status == "unbounded"
    assert "leverage" in str(raised.value)
