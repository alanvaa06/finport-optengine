"""Naive baselines: equal weight (1/N) and inverse volatility.

Neither method has an opinion about constraints — they produce a weight
vector and the mandate is applied afterwards, by projecting onto the closest
feasible allocation. The distance moved is reported, because a 1/N book that
had to travel 15% of its weight to satisfy the mandate is no longer really
1/N, and the analyst should be told rather than left to notice.
"""

from __future__ import annotations

import numpy as np

from optimization_engine.optimizers._bounds import project_to_constraints
from optimization_engine.optimizers.base import BaseOptimizer, zero_variance_assets


class _ProjectedOptimizer(BaseOptimizer):
    """Shared plumbing for methods that allocate first and constrain after."""

    bounds_mode = "soft_iterated"

    def _project(self, w: np.ndarray) -> np.ndarray:
        projected, distance = project_to_constraints(w, self.assets, self.constraints)
        self._diagnostics["projection_distance"] = distance
        if distance > 1e-6:
            self._diagnostics["bounds_note"] = (
                f"Constraints moved {distance:.2%} of the book away from the "
                f"raw {self.name} allocation. Above roughly 10%, the mandate "
                "rather than the method is producing the answer."
            )
        return projected


class EqualWeightOptimizer(_ProjectedOptimizer):
    """Allocate 1/N to each asset, then project into the constraint set."""

    name = "equal_weight"

    def _solve(self) -> np.ndarray:
        n = len(self.assets)
        if n == 0:
            raise ValueError("Equal weight needs at least one asset.")
        return self._project(np.ones(n) / n)


class InverseVolatilityOptimizer(_ProjectedOptimizer):
    """Weights inversely proportional to per-asset volatility (no correlations)."""

    name = "inverse_vol"

    def _solve(self) -> np.ndarray:
        """Weight each asset by ``1/σ`` and renormalize.

        Raises:
            ValueError: If any asset has zero variance, exactly or to rounding
                (see ``zero_variance_assets``). ``1/σ`` is undefined
                there, and the old behaviour — weight 0 — silently dropped the
                name from the book while still reporting it as part of the
                universe. A degenerate column is a data problem, so it is
                named and raised rather than absorbed.
        """
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        degenerate = zero_variance_assets(self.assets, np.diag(sigma))
        if degenerate:
            raise ValueError(
                f"Inverse-volatility weights are undefined for "
                f"{len(degenerate)} zero-variance asset(s): "
                f"{', '.join(map(str, degenerate))}. 1/sigma does not exist there, "
                "and weighting them zero would drop them from the book without "
                "saying so. Drop the asset(s) from the universe, or check the "
                "price history for a constant series."
            )
        std = np.sqrt(np.diag(sigma))
        return self._project((1.0 / std) / (1.0 / std).sum())
