"""Mean-variance family: minimum variance, target-return MV, max Sharpe."""

from __future__ import annotations

import warnings
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from optimization_engine.optimizers._cvxpy_helpers import (
    SolverFailure,
    build_constraints,
    build_scaled_constraints,
    homogeneous_ignored_constraints,
    solve_problem,
)
from optimization_engine.optimizers.base import BaseOptimizer

# Backwards-compatible alias: several modules imported ``_solve_problem``
# from here before solver dispatch moved into ``_cvxpy_helpers``.
_solve_problem = solve_problem
_SOLVER_FALLBACK = ["CLARABEL", "ECOS", "SCS", "OSQP"]

#: How close ``μ'w`` has to sit to the return target before the floor counts
#: as binding. Absolute floor for targets near zero, relative above it.
_TARGET_SLACK_ATOL = 1e-7
_TARGET_SLACK_RTOL = 1e-5

#: A tangency ray whose ``κ = Σy`` comes back within a hundredfold of the 1e-8
#: floor ``build_scaled_constraints`` imposes is resting on that floor, not
#: describing a portfolio.
_KAPPA_DEGENERATE = 1e-6


class MinVarianceOptimizer(BaseOptimizer):
    """Global Minimum-Variance portfolio (no return target).

    The one mean-variance portfolio that needs no expected-return vector,
    and therefore the one that is immune to the estimation error that
    dominates ``μ``. That robustness is why it is the natural anchor for
    the efficient frontier.
    """

    name = "min_variance"
    bounds_mode = "hard"

    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        n = len(self.assets)
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
        constraints = build_constraints(
            w, self.assets, self.constraints, cov_matrix=sigma
        )
        problem = cp.Problem(objective, constraints)
        info = solve_problem(problem)
        if w.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        self._diagnostics.update(info.as_dict())
        if not self.constraints.fully_invested:
            # The open budget is honoured, not ignored — the pre-flight relies
            # on this solve for the volatility floor of an open-budget mandate,
            # where holding cash is how a low target is met. But the answer is
            # then the smallest book the floors allow, which came back as
            # weights near 1e-4 with nothing to say so.
            invested = float(np.sum(w.value))
            self._diagnostics["invested_fraction"] = invested
            self._diagnostics["budget_note"] = (
                f"With an open budget the lowest-variance book invests "
                f"{invested:.2%} of capital — only what the weight floors "
                "force. Set fully_invested=True for the minimum-variance "
                "portfolio, or use mean_variance to trade return for risk."
            )
        return w.value


class MeanVarianceOptimizer(BaseOptimizer):
    """Markowitz mean-variance optimizer.

    Three modes determined by ``constraints``:

    * ``target_return`` set        → minimize variance s.t. ``μ'w ≥ R*``
    * ``target_volatility`` set    → maximize ``μ'w`` s.t. ``√(w'Σw) ≤ σ*``
    * neither set                  → maximize ``μ'w − λ·w'Σw`` (utility)

    The return target is an **inequality**, not an equality. With ``μ'w = R*``
    a target below the global minimum-variance return returns a point on the
    dominated lower branch — the same volatility with less return — and says
    nothing about it. Under ``μ'w ≥ R*`` the minimizer sits on the efficient
    branch by construction: any such target simply returns the
    minimum-variance portfolio. Whether the target actually bound is reported
    as ``extras["target_return_binding"]``, with the realized slack in
    ``extras["target_return_slack"]``.

    The volatility target is imposed as ``w'Σw ≤ σ*²`` — a convex quadratic
    constraint, so the solve stays a QP rather than becoming a second-order
    cone problem.
    """

    name = "mean_variance"
    bounds_mode = "hard"

    def __init__(self, *args, risk_aversion: float = 1.0, **kwargs) -> None:
        """Set the risk-aversion coefficient; everything else is the base class's.

        Args:
            *args: Passed to :class:`~optimization_engine.optimizers.base.BaseOptimizer`.
            risk_aversion: The ``λ`` in ``μ'w − λ·w'Σw``, used only in utility
                mode — that is, when neither a target return nor a target
                volatility is set. Higher means more risk-averse.
            **kwargs: Passed to the base class.

        Raises:
            ValueError: If ``risk_aversion`` is negative, which would reward risk
                and leave the problem unbounded.
        """
        super().__init__(*args, **kwargs)
        self.risk_aversion = float(risk_aversion)
        #: μ as the solve saw it, kept so the post-solve slack can be measured
        #: without calling ``_mu_vector()`` a second time — it warns about
        #: missing entries, and it should warn once per solve, not twice.
        self._solved_mu: np.ndarray | None = None
        #: The covariance a tracking-error budget is measured on, when it is
        #: not the one being optimized. Black-Litterman sets it: its sub-solve
        #: optimizes against the posterior ``Σ + M``, but active risk is
        #: measured on ``Σ`` by every other method and by the audit.
        self._tracking_cov: pd.DataFrame | None = None
        if self.risk_aversion < 0:
            raise ValueError(
                f"risk_aversion must be non-negative; got {risk_aversion}. "
                "A negative coefficient rewards risk and makes the problem "
                "unbounded."
            )

    def _solve(self) -> np.ndarray:
        mu = self._mu_vector()
        sigma = self._sigma_matrix()
        if mu is None or sigma is None:
            raise ValueError("Mean-variance needs both expected_returns and cov_matrix")
        n = len(self.assets)
        w = cp.Variable(n)
        sigma_psd = cp.psd_wrap(sigma)

        target_return: float | None = None
        if self.constraints.target_return is not None:
            mode = "target_return"
            target_return = float(self.constraints.target_return)
            objective = cp.Minimize(cp.quad_form(w, sigma_psd))
            extra = [mu @ w >= target_return]
        elif self.constraints.target_volatility is not None:
            mode = "target_volatility"
            target_vol = float(self.constraints.target_volatility)
            if target_vol <= 0:
                raise ValueError(
                    f"target_volatility must be positive; got {target_vol}."
                )
            objective = cp.Maximize(mu @ w)
            extra = [cp.quad_form(w, sigma_psd) <= target_vol**2]
        else:
            mode = "utility"
            if self.risk_aversion == 0:
                warnings.warn(
                    "risk_aversion=0 makes this a pure return-maximization: the "
                    "result will sit at the corner of the constraint set and "
                    "ignore risk entirely.",
                    stacklevel=3,
                )
            objective = cp.Maximize(mu @ w - self.risk_aversion * cp.quad_form(w, sigma_psd))
            extra = []

        tracking = sigma
        if self._tracking_cov is not None:
            tracking = (
                self._tracking_cov.reindex(index=self.assets, columns=self.assets)
                .to_numpy(dtype=float)
            )
        cons = build_constraints(
            w, self.assets, self.constraints, extra, cov_matrix=tracking
        )
        problem = cp.Problem(objective, cons)
        info = solve_problem(problem)
        if w.value is None:
            raise RuntimeError(
                f"Solver failed for {self.name}: status={problem.status}. "
                "Constraints may be infeasible (check bounds and target)."
            )
        self._diagnostics.update(info.as_dict())
        self._diagnostics["mode"] = mode
        self._solved_mu = np.asarray(mu, dtype=float)
        return w.value

    def _post_solve_diagnostics(self, weights: pd.Series) -> dict[str, Any]:
        """Base diagnostics, plus whether the return floor actually bound.

        Measured here rather than in ``_solve`` because the base class calls
        this *after* ``_clean_weights``: a slack computed on the raw solver
        vector would describe a portfolio that is not the one reported.
        """
        diag = super()._post_solve_diagnostics(weights)
        target_return = self.constraints.target_return
        if target_return is None or self._solved_mu is None:
            return diag
        diag.update(
            self._target_slack(weights.values, float(target_return), self._solved_mu)
        )
        return diag

    @staticmethod
    def _target_slack(
        weights: np.ndarray, target_return: float, mu: np.ndarray
    ) -> dict[str, Any]:
        """Describe the return floor: what it asked for, and what it got.

        A non-binding floor means the answer is the minimum-variance
        portfolio and the target had no influence on it — worth saying out
        loud, because the caller asked for a return they did not get.
        """
        achieved = float(np.asarray(weights, dtype=float) @ mu)
        slack = achieved - target_return
        # Solver tolerance, not economics, decides whether an active floor
        # lands a hair either side of zero.
        tol = max(_TARGET_SLACK_ATOL, abs(target_return) * _TARGET_SLACK_RTOL)
        binding = bool(slack <= tol)
        diag: dict[str, Any] = {
            "target_return": target_return,
            "target_return_achieved": achieved,
            "target_return_slack": slack,
            "target_return_binding": binding,
        }
        if not binding:
            diag["target_return_note"] = (
                f"The {target_return:.2%} return floor never bound: minimum "
                f"variance alone already earns {achieved:.2%}. This is the "
                "minimum-variance portfolio, not a portfolio built to that "
                "target."
            )
        return diag


class MaxSharpeOptimizer(BaseOptimizer):
    """Maximum Sharpe ratio (tangency) portfolio.

    Solved as the standard homogeneous reformulation: minimize ``y'Σy``
    subject to ``(μ − rf)·y = 1``, then renormalize ``w = y / Σy``.

    Bounds, group budgets, layer limits, a gross-exposure cap and the
    benchmark-relative budgets are scaled by ``κ = Σy`` so they stay exact.

    Three things do not survive the transform, and all are reported rather
    than silently dropped: a turnover budget (affine, not homogeneous), an
    open budget (``fully_invested=False`` — the ray fixes a direction, and
    the result always sums to one), and the case where the tangency ray
    points the wrong way because no allocation the mandate allows earns more
    than the risk-free rate — :class:`NoPositiveExcessReturnError`, which a
    long-short book whose assets all trail cash does not trigger when a
    spread between them beats it.
    """

    name = "max_sharpe"
    bounds_mode = "hard"

    def _solve(self) -> np.ndarray:
        mu = self._mu_vector()
        sigma = self._sigma_matrix()
        if mu is None or sigma is None:
            raise ValueError("Max-Sharpe needs both expected_returns and cov_matrix")
        rf = self.risk_free_rate
        excess = mu - rf
        # Whether *any* allocation beats cash is a question about the feasible
        # set, not about the assets one at a time. Asked per asset, it refused
        # every long-short book whose assets all trail cash (review item O10),
        # and passed a box that caps the only asset above it — whose ray then
        # came back "infeasible", naming no constraint because none is at fault.
        best = self._best_feasible_excess(excess, sigma)
        if best is not None and best <= 0:
            raise NoPositiveExcessReturnError(best, rf)

        ignored = homogeneous_ignored_constraints(self.constraints, "Max-Sharpe")
        if ignored:
            self._diagnostics["ignored_constraints"] = ignored

        n = len(self.assets)
        y = cp.Variable(n)
        kappa = cp.Variable(nonneg=True)

        sigma_psd = cp.psd_wrap(sigma)
        objective = cp.Minimize(cp.quad_form(y, sigma_psd))
        # The ray is normalized by the best feasible excess rather than by 1.
        # The transform is homogeneous, so the weights are the same, but
        # ``κ = Σy`` now sits near one: with an excess of 1e-7 the unscaled
        # ray needed κ ≈ 1e7, past the feasibility tolerance of CLARABEL, SCS
        # and OSQP alike — all three answered "infeasible".
        scaled = excess / best if best is not None and np.isfinite(best) else excess
        cons = [scaled @ y == 1]
        cons += build_scaled_constraints(
            y, kappa, self.assets, self.constraints, cov_matrix=sigma
        )

        problem = cp.Problem(objective, cons)
        info = solve_problem(problem)
        if y.value is None or kappa.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        scale = float(kappa.value)
        if scale <= _KAPPA_DEGENERATE:
            # ``build_scaled_constraints`` floors κ at 1e-8, so κ sitting on
            # that floor is the only way this branch is reached — and with a
            # finite box it cannot be, since ``(μ − rf)'y = 1`` then forces κ
            # away from zero. Without one, κ on the floor means the Sharpe
            # ratio is still rising as ``y/κ`` grows: a supremum, not a maximum.
            raise SolverFailure(
                "unbounded",
                info.attempts,
                detail=(
                    "The Sharpe ratio keeps rising as the book levers up, so "
                    "no finite portfolio attains the maximum. Add per-asset "
                    "bounds or a leverage cap."
                ),
            )
        self._diagnostics.update(info.as_dict())
        return np.array(y.value) / scale

    def _best_feasible_excess(
        self, excess: np.ndarray, sigma: np.ndarray
    ) -> float | None:
        """The highest excess return any allocation the ray solve allows earns.

        Measured over what the ray actually imposes — the budget, the box, the
        layers, a gross cap and the benchmark budgets, but not a turnover
        budget or an open budget, which it cannot carry — with the feasibility
        module's machinery: the knapsack closed form under box and budget, two
        LPs otherwise.

        Returns:
            The best excess return, or ``None`` when it cannot be measured: an
            infinite bound (the LP has no finite answer to compare against
            zero), an infeasible mandate (the solve says so in its own terms)
            or a solver that could not answer.
        """
        from dataclasses import replace

        from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays
        from optimization_engine.optimizers.feasibility import reachable_return_range

        ray = replace(
            self.constraints,
            fully_invested=True,
            turnover_limit=None,
            previous_weights=None,
        )
        lb, ub = bounds_arrays(self.assets, ray)
        if not (np.isfinite(lb).all() and np.isfinite(ub).all()):
            return None
        found = reachable_return_range(
            pd.Series(excess, index=self.assets),
            ray,
            self.assets,
            pd.DataFrame(sigma, index=self.assets, columns=self.assets),
        )
        return None if found is None else float(found[1])


class NoPositiveExcessReturnError(ValueError):
    """No allocation the mandate allows earns more than the risk-free rate.

    The tangency portfolio is undefined there: every Sharpe ratio on offer is
    at or below zero, so there is nothing to maximize. A ``ValueError``, like
    the per-asset refusal it replaces.

    Attributes:
        best_excess_return: The highest excess return any feasible allocation
            earns, in the units of the expected returns.
        risk_free_rate: The rate it was measured against.
    """

    def __init__(self, best_excess_return: float, risk_free_rate: float) -> None:
        """Build the error from the best excess return the mandate allows.

        Args:
            best_excess_return: The maximum of ``(μ − rf)'w`` over the
                feasible set.
            risk_free_rate: The risk-free rate.
        """
        self.best_excess_return = float(best_excess_return)
        self.risk_free_rate = float(risk_free_rate)
        super().__init__(
            "No allocation these constraints allow earns more than the "
            f"risk-free rate ({self.risk_free_rate:.2%}): the best feasible "
            f"excess return is {self.best_excess_return:.2%}, so no portfolio "
            "has a positive Sharpe ratio and maximizing it is infeasible. "
            "Lower the risk-free rate, revisit the expected returns, or loosen "
            "the caps on the assets that do beat cash."
        )
