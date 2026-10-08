"""Risk parity / risk-budgeting optimizer.

Implements Equal Risk Contribution (ERC) and arbitrary risk-budgeting via
the convex log-barrier formulation of Spinu (2013) / Maillard et al. (2010):

    minimize   ½ y' Σ y − Σ b_i log(y_i)

At the optimum the risk contributions of the normalized weights
``w = y / Σy`` are proportional to ``b``. The formulation is convex and so
has a unique solution, unlike the fixed-point iteration it replaces.

Note on the budget vector: ``b_i`` is asset *i*'s target share of total
portfolio risk. Because the Euler decomposition splits volatility (and
variance) into the same proportions, "share of volatility" and "share of
variance" name the same number here.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from optimization_engine.optimizers._cvxpy_helpers import (
    bounds_arrays,
    layer_constraints,
    solve_problem,
)
from optimization_engine.optimizers.base import BaseOptimizer


class RiskParityOptimizer(BaseOptimizer):
    """Equal Risk Contribution / Risk Budgeting optimizer.

    Pass a ``risk_budget`` mapping ``asset -> target share of total risk``
    (it is renormalized to sum to 1). If absent, defaults to ERC.

    The solve reports how closely the target budget was actually met: bounds
    and group budgets can make an exact risk parity unreachable, and when
    that happens the analyst should see the gap rather than assume parity.
    """

    name = "risk_parity"
    bounds_mode = "constrained"

    def __init__(self, *args, risk_budget: dict[str, float] | None = None, **kwargs) -> None:
        """Set the risk budget the solve will target.

        Args:
            *args: Passed to :class:`~optimization_engine.optimizers.base.BaseOptimizer`.
            risk_budget: ``asset -> target share of total risk``. Shares are
                normalized, so they need not sum to one. ``None`` means equal risk
                contribution. Every asset needs a strictly positive share: the
                log-barrier formulation has no answer for a zero one, so drop such
                an asset from the universe instead.
            **kwargs: Passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.risk_budget = risk_budget

    def _budget_vector(self) -> np.ndarray:
        n = len(self.assets)
        if not self.risk_budget:
            return np.ones(n) / n
        b = np.array([float(self.risk_budget.get(a, 0.0)) for a in self.assets])
        if (b < 0).any():
            raise ValueError(
                "Risk budgets must be non-negative; a negative target risk "
                "share has no interpretation."
            )
        if b.sum() <= 0:
            raise ValueError("Risk budget must sum to a positive number")
        if (b <= 0).any():
            zero = [a for a, v in zip(self.assets, b) if v <= 0]
            raise ValueError(
                f"Risk budget is zero for {zero}. The log-barrier formulation "
                "needs a strictly positive budget for every asset — give them a "
                "small positive share, or drop them from the universe."
            )
        total = b.sum()
        if abs(total - 1.0) > 1e-6:
            warnings.warn(
                f"Risk budgets sum to {total:.4f}, not 1; renormalizing.",
                stacklevel=3,
            )
        return b / total

    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        n = len(self.assets)
        b = self._budget_vector()

        lb_arr, ub_arr = bounds_arrays(self.assets, self.constraints)
        if (lb_arr < 0).any():
            raise ValueError(
                "Risk parity requires non-negative weights: risk contributions "
                "are only well defined for a long-only book. Set every minimum "
                "weight to 0 or above."
            )
        # Strict positivity for the log-barrier; tighten lb a hair if zero.
        lb_pos = np.maximum(lb_arr, 1e-8)

        ignored: list[str] = []
        if self.constraints.turnover_limit is not None:
            warnings.warn(
                "Risk parity ignores the turnover budget: the log-barrier "
                "solve works on a scaled ray where turnover is not well "
                "defined.",
                stacklevel=3,
            )
            ignored.append("turnover_limit")
        if not self.constraints.fully_invested:
            # The weights are the ray normalized to sum to one, so an open
            # budget never reaches the answer. It used to be dropped without
            # a word, where max-Sharpe and max-diversification report it.
            warnings.warn(
                "Risk parity ignores fully_invested=False: risk shares are "
                "defined on the normalized book, and the weights it returns "
                "always sum to one. Size the result against cash separately.",
                stacklevel=3,
            )
            ignored.append("fully_invested")
        if ignored:
            self._diagnostics["ignored_constraints"] = ignored

        y = cp.Variable(n, pos=True)
        sigma_psd = cp.psd_wrap(sigma)
        total = cp.sum(y)
        cons: list = [
            y >= cp.multiply(lb_pos, total),
            y <= cp.multiply(ub_arr, total),
        ]
        # ``y`` is the unnormalized ray again, so every layer's bucket budget
        # scales by the same total — including a percent-of-parent limit,
        # where both sides scale and the ratio is preserved exactly.
        cons.extend(
            layer_constraints(y, self.assets, self.constraints, scale=total)
        )

        objective = cp.Minimize(0.5 * cp.quad_form(y, sigma_psd) - b @ cp.log(y))
        problem = cp.Problem(objective, cons)
        # The log term makes this exponential-cone; CLARABEL and SCS handle it,
        # ECOS/OSQP do not, so the chain is narrowed here.
        info = solve_problem(problem, solvers=("CLARABEL", "SCS", "ECOS"))
        if y.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        self._diagnostics.update(info.as_dict())

        w = np.array(y.value) / float(np.sum(y.value))
        # Clamp tiny floating drift back into the box.
        w = np.clip(w, lb_arr, ub_arr)
        w = w / w.sum() if w.sum() > 0 else w

        self._record_budget_error(w, sigma, b)
        return w

    def _record_budget_error(
        self, w: np.ndarray, sigma: np.ndarray, b: np.ndarray
    ) -> None:
        """Compare achieved risk shares with the target budget."""
        total_var = float(w @ sigma @ w)
        if total_var <= 0:
            return
        achieved = w * (sigma @ w) / total_var
        error = np.abs(achieved - b)
        self._diagnostics["risk_budget_target"] = pd.Series(b, index=self.assets)
        self._diagnostics["risk_budget_achieved"] = pd.Series(achieved, index=self.assets)
        self._diagnostics["risk_budget_max_error"] = float(error.max())
        self._diagnostics["risk_budget_mean_error"] = float(error.mean())
        if error.max() > 1e-3:
            self._diagnostics["risk_budget_note"] = (
                f"Largest gap between target and achieved risk share is "
                f"{error.max():.2%} — the weight bounds or group budgets stop "
                "the portfolio from reaching exact risk parity."
            )

    def risk_contributions(self, weights: np.ndarray | pd.Series) -> pd.Series:
        """Per-asset share of total portfolio variance for a weight vector.

        Args:
            weights: The weights to decompose, as fractions of the book.

        Returns:
            One share per asset, summing to 1. Compare against the target risk
            budget to see how closely parity was actually reached — bounds and
            bucket budgets can make an exact one unreachable.
        """
        sigma = self._sigma_matrix()
        w = np.asarray(weights).flatten()
        total = float(w @ sigma @ w)
        marginal = sigma @ w
        rc = w * marginal / total if total > 0 else np.zeros_like(w)
        return pd.Series(rc, index=self.assets)
