"""Mean-CDaR optimizer (Chekhlov, Uryasev & Zabarankin, 2005).

Variance and CVaR are both computed period by period: reorder the return
history at random and neither changes. Drawdown is the one risk measure a
client actually experiences that does *not* have that property — it depends
on the sequence, because it asks how far below a previous peak the book went
and for how long. A portfolio can have unremarkable volatility and CVaR and
still spend three years underwater.

Conditional Drawdown at Risk is the drawdown analogue of CVaR: the average of
the worst ``α`` fraction of drawdowns along the path. Chekhlov, Uryasev and
Zabarankin show it is a coherent risk measure and, crucially, that it is
*linear-programmable* — the same Rockafellar-Uryasev trick that makes
mean-CVaR a linear program works here on the drawdown series:

    minimize    ζ + 1/(α·T) · Σ_t z_t
    subject to  z_t ≥ u_t − y_t − ζ,      z_t ≥ 0
                u_t ≥ y_t,  u_t ≥ u_{t−1},  u_0 ≥ 0
                y_t = Σ_{s ≤ t} r_s' w        (uncompounded equity curve)
                μ'w ≥ R*  (optional),  Σ w = 1,  w ∈ bounds

``u_t`` is the running peak: nothing in the objective rewards raising it, so
at the optimum it sits exactly on the high-water mark. The equity curve is
accumulated *uncompounded* (a sum of returns, not a product), which is what
keeps the constraint set linear in ``w``; over the horizons drawdown analysis
is usually run on, the difference from a compounded curve is small, and the
alternative is a non-convex problem.

Two limits worth stating plainly:

* The drawdown path is a single realized history, not a distribution of them.
  Where CVaR at α = 5% averages ~T/20 independent-ish tail scenarios, CDaR
  averages the worst 5% of a *path* whose points are highly dependent — so
  its effective sample size is far smaller than the observation count
  suggests. The optimizer reports the number of distinct drawdown episodes
  behind the answer for exactly this reason.
* Optimizing a path measure on one path is the most overfittable thing in
  this library. Run it through the walk-forward before believing it.

References:
    Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). "Drawdown Measure in
    Portfolio Optimization". *International Journal of Theoretical and
    Applied Finance* 8(1).
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from optimization_engine.optimizers._cvxpy_helpers import build_constraints, solve_problem
from optimization_engine.optimizers.base import BaseOptimizer

#: Below this many distinct underwater episodes the CDaR estimate is anecdote.
MIN_DRAWDOWN_EPISODES = 5


class CDaROptimizer(BaseOptimizer):
    """Mean-CDaR (conditional drawdown at risk) optimizer.

    Args:
        returns: Scenario matrix of *periodic* returns in chronological
            order. Unlike mean-CVaR, the row order matters here — the
            objective is a path statistic.
        alpha: Tail probability in ``(0, 1)``. ``0.05`` averages the worst 5%
            of the drawdown path. ``alpha → 0`` approaches the maximum
            drawdown; ``alpha = 1`` gives the average drawdown.
        target_return: Optional minimum *annualized* expected return.
        periods_per_year: Used to annualize reported figures and to convert
            historical means when no ``expected_returns`` is supplied.
    """

    name = "cdar"
    bounds_mode = "hard"

    def __init__(
        self,
        returns: pd.DataFrame,
        *args,
        alpha: float = 0.05,
        target_return: float | None = None,
        periods_per_year: int = 252,
        **kwargs,
    ) -> None:
        """Bind the return history the drawdown path is accumulated from.

        Args:
            returns: Periodic return history, one column per asset. Required, and
                it must have no gaps: CDaR accumulates an equity curve, so a
                missing value silently shifts every later drawdown.
            *args: Passed to :class:`~optimization_engine.optimizers.base.BaseOptimizer`.
            alpha: Drawdown tail probability, in ``(0, 1]``. Pass ``0.05`` for the
                worst 5% of the path; ``1`` gives the average drawdown.
            target_return: Optional minimum *annualized* expected return.
            periods_per_year: Annualization basis for the reported figures, and
                for converting historical means when no ``expected_returns`` is
                supplied.
            **kwargs: Passed to the base class.

        Raises:
            ValueError: If ``returns`` is missing, empty or contains NaNs, or
                ``alpha`` lies outside ``(0, 1]``.
        """
        super().__init__(*args, **kwargs)
        if returns is None or returns.empty:
            raise ValueError("CDaR optimizer requires a returns DataFrame")
        if not 0 < alpha <= 1:
            raise ValueError(
                f"alpha is the drawdown tail probability and must lie in "
                f"(0, 1]; got {alpha}. Pass 0.05 for the worst 5% of the path."
            )
        if returns.isna().any().any():
            raise ValueError(
                "The return history contains missing values. CDaR accumulates "
                "an equity curve, so a gap silently shifts every later "
                "drawdown — align the panel first."
            )
        self.returns = returns
        self.alpha = float(alpha)
        self.target_return = target_return
        self.periods_per_year = int(periods_per_year)

    @property
    def assets(self) -> list[str]:  # type: ignore[override]
        """The universe, taken from the return history's columns.

        Overrides the base class, which reads the covariance or the expected
        returns: here the return frame is the required input and the authority on
        what the universe is.
        """
        return list(self.returns.columns)

    def _solve(self) -> np.ndarray:
        T, n = self.returns.shape
        if T < 3:
            raise ValueError(
                f"CDaR needs a path to measure; got {T} observations."
            )

        w = cp.Variable(n)
        zeta = cp.Variable()
        z = cp.Variable(T, nonneg=True)
        peak = cp.Variable(T + 1)

        # Uncompounded equity curve: y_t = Σ_{s ≤ t} r_s'w. Building it as a
        # cumulative-sum matrix keeps the whole program a single LP.
        cumulative = np.tril(np.ones((T, T))) @ self.returns.values
        equity = cumulative @ w

        extras = [
            peak[0] == 0,
            peak[1:] >= equity,
            peak[1:] >= peak[:-1],
            z >= peak[1:] - equity - zeta,
        ]
        if self.target_return is not None:
            extras.append(self._target_mu_vector() @ w >= float(self.target_return))

        objective = zeta + cp.sum(z) / (self.alpha * T)
        cons = build_constraints(
            w, self.assets, self.constraints, extras,
            cov_matrix=self._sigma_matrix(),
        )
        problem = cp.Problem(cp.Minimize(objective), cons)
        info = solve_problem(problem)
        if w.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        self._diagnostics.update(info.as_dict())

        weights = np.asarray(w.value, dtype=float)
        self._record_drawdown_metrics(
            weights, float(zeta.value), float(objective.value)
        )
        return weights

    def _target_mu_vector(self) -> np.ndarray:
        """Annualized expected returns used for the optional return floor."""
        if self.expected_returns is not None:
            return self._mu_vector()
        warnings.warn(
            "No expected returns supplied; the CDaR return target is being "
            "compared against annualized historical means from the same "
            "history the drawdowns come from.",
            stacklevel=4,
        )
        # The package's one definition of μ — the arithmetic annualized mean,
        # as ``expected_returns_from_history("mean")`` gives every other
        # method. ``(1 + m)^ppy − 1`` sat above it, so a floor out of reach in
        # arithmetic terms "solved" here.
        from optimization_engine.data.covariance import expected_returns_from_history

        return expected_returns_from_history(
            self.returns, method="mean", periods_per_year=self.periods_per_year
        ).to_numpy(dtype=float)

    def _record_drawdown_metrics(
        self, weights: np.ndarray, zeta: float, solver_objective: float
    ) -> None:
        """Report the realized drawdown shape of the chosen allocation.

        The solver's objective is stated on the uncompounded curve; the
        figures here are recomputed on the compounded one, which is the
        drawdown an investor would actually have lived through. Reporting
        both is the honest way to show what the approximation cost.

        Args:
            weights: The chosen allocation.
            zeta: The solved ``ζ`` — the drawdown-at-risk threshold, the same
                quantity mean-CVaR reports as ``cvar_solver_zeta``. It is
                *not* the objective, which is why it is no longer called one.
            solver_objective: The solved ``ζ + Σz/(α·T)`` — the CDaR the
                program actually minimized, on the uncompounded curve.
        """
        from optimization_engine.analytics.risk import drawdown_series

        portfolio = pd.Series(
            self.returns.values @ weights, index=self.returns.index
        )
        realized = drawdown_series(portfolio)
        threshold = float(np.quantile(realized.values, self.alpha))
        tail = realized[realized <= threshold]
        cdar = float(-tail.mean()) if len(tail) else float("nan")

        underwater = (realized < -1e-12).values
        episodes = int(np.sum(underwater & ~np.concatenate(([False], underwater[:-1]))))
        if episodes < MIN_DRAWDOWN_EPISODES:
            warnings.warn(
                f"The chosen allocation went underwater in only {episodes} "
                "distinct episode(s) over this history, so its CDaR rests on "
                "very few independent events. Treat the number as descriptive "
                "of this path rather than as a forecast.",
                stacklevel=4,
            )

        self._diagnostics.update(
            {
                "cdar_alpha": self.alpha,
                "cdar_solver_zeta": zeta,
                "cdar_solver_objective": solver_objective,
                "cdar_realized": cdar,
                "dar_realized": float(-threshold),
                "max_drawdown": float(realized.min()),
                "average_drawdown": float(-realized.mean()),
                "drawdown_episodes": episodes,
                "n_scenarios": int(len(self.returns)),
                "cdar_note": (
                    f"Worst {self.alpha:.0%} of the drawdown path averages "
                    f"{cdar:.2%} against a maximum drawdown of "
                    f"{realized.min():.2%}, over {episodes} underwater "
                    "episode(s)."
                ),
            }
        )
