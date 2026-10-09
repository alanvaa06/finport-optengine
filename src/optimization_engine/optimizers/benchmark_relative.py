"""Optimization against a benchmark rather than against cash.

Most of the engine solves for the best *absolute* portfolio and only then asks
how it compares to an index. A mandate written in relative terms — "beat this
benchmark by 150bp at no more than 3% tracking error" — is a different
problem, and solving the absolute one and hoping is how a portfolio ends up
with a perfectly good Sharpe ratio and an active risk its owner never agreed
to.

The objective here is Grinold-Kahn's: choose active weights ``x = w − b`` to
maximize ``α'x − λ·x'Σx``, where ``α`` is expected return in excess of the
benchmark's. The constraint machinery is shared with every other optimizer,
so bounds, group budgets, turnover and exposure limits all still hold.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from optimization_engine.optimizers._cvxpy_helpers import (
    build_constraints,
    solve_problem,
)
from optimization_engine.optimizers.base import BaseOptimizer


class ActiveMeanVarianceOptimizer(BaseOptimizer):
    """Mean-variance in *active* space: return and risk measured vs a benchmark.

    Two modes, determined by the constraints:

    * ``max_tracking_error`` set → maximize expected active return subject to
      ``√(x'Σx) ≤ TE*``. The mandate form: spend the active-risk budget as
      well as the forecasts allow.
    * otherwise                  → maximize ``α'x − λ·x'Σx``, trading expected
      active return against active variance at the configured risk aversion.

    Setting the benchmark to the portfolio's own solution space matters: with
    ``b = 0`` this reduces exactly to :class:`MeanVarianceOptimizer`, which is
    the sanity check that the active formulation is not doing something
    different behind the same numbers.
    """

    name = "active_mean_variance"
    bounds_mode = "hard"

    def __init__(self, *args, risk_aversion: float = 1.0, **kwargs) -> None:
        """Set the active risk-aversion coefficient.

        Args:
            *args: Passed to :class:`~optimization_engine.optimizers.base.BaseOptimizer`.
            risk_aversion: Aversion to *active* risk — the coefficient on
                ``(w − b)'Σ(w − b)``. At ``b = 0`` the problem reduces exactly to
                the absolute one.
            **kwargs: Passed to the base class.

        Raises:
            ValueError: If ``risk_aversion`` is negative, which would reward
                active risk and leave the problem unbounded.
        """
        super().__init__(*args, **kwargs)
        self.risk_aversion = float(risk_aversion)
        if self.risk_aversion < 0:
            raise ValueError(
                f"risk_aversion must be non-negative; got {risk_aversion}. "
                "A negative coefficient rewards active risk and makes the "
                "problem unbounded."
            )

    def _benchmark(self) -> np.ndarray:
        b = self.constraints.benchmark_vector(self.assets)
        if b is None:
            raise ValueError(
                "active_mean_variance optimizes against a benchmark, but none "
                "was set. Choose a benchmark before running it - with no "
                "index to be active against, use mean_variance instead."
            )
        return b

    def _solve(self) -> np.ndarray:
        mu = self._mu_vector()
        sigma = self._sigma_matrix()
        if mu is None or sigma is None:
            raise ValueError(
                "active_mean_variance needs both expected_returns and cov_matrix"
            )
        benchmark = self._benchmark()
        n = len(self.assets)
        w = cp.Variable(n)
        active = w - benchmark
        sigma_psd = cp.psd_wrap(sigma)
        # Expected return in excess of the benchmark's. Subtracting the
        # benchmark's own expected return is a constant shift of the
        # objective, but it is what makes the reported alpha a real alpha.
        alpha = mu - float(mu @ benchmark)

        if self.constraints.max_tracking_error is not None:
            mode = "target_tracking_error"
            objective = cp.Maximize(alpha @ active)
            # The budget is already imposed by build_constraints; stating the
            # mode here keeps the reported diagnostics honest about which
            # problem was solved.
        else:
            mode = "active_utility"
            objective = cp.Maximize(
                alpha @ active - self.risk_aversion * cp.quad_form(active, sigma_psd)
            )

        cons = build_constraints(w, self.assets, self.constraints, cov_matrix=sigma)
        problem = cp.Problem(objective, cons)
        info = solve_problem(problem)
        if w.value is None:
            raise RuntimeError(
                f"Solver failed for {self.name}: status={problem.status}. The "
                "tracking-error budget and the weight bounds may be mutually "
                "impossible - a benchmark holding an asset the bounds cap "
                "below its index weight forces a minimum tracking error."
            )

        solved = np.asarray(w.value).flatten()
        x = solved - benchmark
        te = float(np.sqrt(max(float(x @ sigma @ x), 0.0)))
        active_return = float(alpha @ x)
        self._diagnostics.update(info.as_dict())
        self._diagnostics.update(
            {
                "mode": mode,
                "expected_active_return": active_return,
                "expected_tracking_error": te,
                "implied_information_ratio": (
                    active_return / te if te > 1e-12 else float("nan")
                ),
                "active_share": float(np.abs(x).sum() / 2.0),
                "benchmark_expected_return": float(mu @ benchmark),
            }
        )
        return solved


__all__ = ["ActiveMeanVarianceOptimizer"]
