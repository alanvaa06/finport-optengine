"""Maximum Diversification optimizer (Choueifaty & Coignard, 2008).

Maximizes the diversification ratio:

    DR(w) = (Σ_i w_i · σ_i) / √(w' Σ w)

A higher DR means weighted-average asset volatility is being meaningfully
reduced by correlation. The classic transformation reduces this to:

    minimize   y' Σ y      subject to σ' y = 1, y ≥ 0
    weights    w = y / Σ y

Because the objective and the normalization are homogeneous, per-asset and
group bounds carry over exactly when scaled by ``κ = Σy`` — so this solves
the *constrained* max-DR problem rather than solving it unconstrained and
projecting afterwards, which can cost several points of diversification
ratio whenever the caps actually bind.

That scaled solve is the answer whenever it succeeds. Two things can stop it,
and they are not the same thing: a mandate with no feasible allocation at all
is a property of the *problem* and raises, while a numerical failure inside
the solver falls back to the unconstrained solve plus a projection. The
fallback is honest about itself — it reports ``solver_status`` of
``"fallback_projection"``, the distance the projection moved the book, and the
constraints it could not carry — because a projected result that breaks its own
tracking-error budget labelled ``"optimal"`` is worse than no result.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from optimization_engine.optimizers._bounds import project_to_constraints
from optimization_engine.optimizers._cvxpy_helpers import (
    SolverFailure,
    build_scaled_constraints,
    homogeneous_ignored_constraints,
    solve_problem,
)
from optimization_engine.optimizers.base import BaseOptimizer


class MaxDiversificationOptimizer(BaseOptimizer):
    """Maximum diversification-ratio portfolio.

    Bounds are hard on the scaled solve, which is the path taken unless the
    solver fails numerically; the projection fallback holds them only
    approximately and re-labels the result's ``bounds_mode`` when it runs.
    Hence the registry's ``"hard_or_projected"``.
    """

    name = "max_diversification"
    bounds_mode = "hard"

    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        std = np.sqrt(np.diag(sigma))
        if not (std > 0).all():
            zero = [a for a, s in zip(self.assets, std) if s <= 0]
            raise ValueError(
                f"Zero-variance asset(s) {zero}: the diversification ratio is "
                "undefined when an asset has no volatility. Drop them from the "
                "universe."
            )

        ignored = homogeneous_ignored_constraints(
            self.constraints, "Max-diversification"
        )
        if ignored:
            self._diagnostics["ignored_constraints"] = ignored

        n = len(self.assets)
        y = cp.Variable(n)
        kappa = cp.Variable(nonneg=True)
        sigma_psd = cp.psd_wrap(sigma)
        objective = cp.Minimize(cp.quad_form(y, sigma_psd))
        cons = [std @ y == 1]
        cons += build_scaled_constraints(
            y, kappa, self.assets, self.constraints, cov_matrix=sigma
        )

        problem = cp.Problem(objective, cons)
        try:
            info = solve_problem(problem)
        except SolverFailure as exc:
            # Infeasible and unbounded are properties of the *problem*, not of
            # the solver: no amount of retrying or projecting makes an
            # unreachable mandate reachable. Projecting anyway is how a book
            # that breaks its own tracking-error budget used to come back
            # labelled "optimal". A refused ``optimal_inaccurate`` is not a
            # numerical failure either: the chain found an answer and the
            # caller declined it, and a projection of the *unconstrained*
            # solve is a less verified answer than the one refused.
            if exc.status in {"infeasible", "unbounded", "optimal_inaccurate"}:
                raise
            return self._fallback_projection(sigma, std, exc)
        except Exception as exc:
            # Anything else — a modelling error, a backend blowing up — is a
            # numerical failure as far as the caller is concerned, so it earns
            # the fallback rather than losing the solve entirely.
            return self._fallback_projection(sigma, std, exc)

        if y.value is None or kappa.value is None or float(kappa.value) <= 1e-10:
            return self._fallback_projection(
                sigma, std, RuntimeError("degenerate scaled solution")
            )
        self._diagnostics.update(info.as_dict())
        return np.array(y.value) / float(kappa.value)

    def _fallback_projection(
        self, sigma: np.ndarray, std: np.ndarray, cause: Exception
    ) -> np.ndarray:
        """Unconstrained solve + projection, for a *numerical* failure only.

        Never reached for an infeasible or unbounded mandate: no projection
        makes an unreachable mandate reachable, so those raise instead.

        This is strictly worse than the constrained solve — it is a last
        resort, and it says so in the diagnostics so the result is never
        mistaken for the true constrained optimum.

        Args:
            sigma: Annualized covariance, aligned to ``self.assets``.
            std: Per-asset volatilities, the diagonal of ``sigma`` rooted.
            cause: What failed, quoted verbatim in ``fallback_reason``.

        Returns:
            The projected weights.
        """
        n = len(self.assets)
        y = cp.Variable(n, nonneg=True)
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma))), [std @ y == 1]
        )
        info = solve_problem(problem)
        if y.value is None:
            raise RuntimeError(
                f"Max-diversification failed: {cause}"
            ) from cause
        w = np.array(y.value)
        total = w.sum()
        if total <= 0:
            raise RuntimeError("Degenerate Max-Diversification solution")
        w, distance = project_to_constraints(w / total, self.assets, self.constraints)
        # Deliberately *not* ``info.as_dict()``: that record describes the
        # unconstrained solve above, and merging it wholesale is what used to
        # stamp ``solver_status="optimal"`` — and an objective value nobody's
        # portfolio has — onto a projected, mandate-breaching result. Only the
        # fields that stay true of the answer actually returned are kept.
        self._diagnostics.update(
            {
                "solver": info.solver,
                "solver_status": "fallback_projection",
                "solve_seconds": info.solve_seconds,
                "solvers_attempted": list(info.attempts),
                "dropped_constraints": self._dropped_by_projection(),
            }
        )
        self._diagnostics["projection_distance"] = distance
        self._diagnostics["fallback_reason"] = (
            f"Exact bounded solve failed ({cause}); bounds were applied by "
            "projection, so the diversification ratio is below the true "
            "constrained optimum."
        )
        # On this solve's diagnostics, not on the instance: setting
        # ``self.bounds_mode`` outlived the solve and labelled every later,
        # exact solve on the same optimizer as projected.
        self._diagnostics["bounds_mode"] = "soft_iterated"
        return w

    def _dropped_by_projection(self) -> list[str]:
        """Mandate items the projection fallback cannot carry, in a fixed order.

        The fallback re-imposes the mandate by solving ``min ‖x − w‖²`` subject
        to it, so most of it survives. Two things do not, and naming them here
        saves the reader a trip through ``_bounds.py``:

        * ``max_tracking_error`` is stripped unconditionally
          (``_bounds._without_turnover``), because the projection is not handed
          a covariance matrix and active risk cannot be written from weights
          alone.
        * ``leverage`` survives only on the projection's CVXPY branch, taken
          when a bucket budget or an active-share cap is set. Without either,
          the projection clips and redistributes, which is blind to gross
          exposure.

        ``max_active_share`` is deliberately absent: setting it is exactly what
        forces the CVXPY branch, so the projection *does* honour it. Listing it
        would be the same kind of false claim this diagnostic exists to stop. A
        turnover budget is dropped too, but the ray-space solve never carried
        it either, so it is already named in ``ignored_constraints``.

        Returns:
            The constraint field names, empty when the projection carried the
            whole mandate.
        """
        constraints = self.constraints
        dropped: list[str] = []
        if constraints.max_tracking_error is not None:
            dropped.append("max_tracking_error")
        projection_is_exact = (
            constraints.has_layer_limits or constraints.max_active_share is not None
        )
        if constraints.leverage is not None and not projection_is_exact:
            dropped.append("leverage")
        return dropped
