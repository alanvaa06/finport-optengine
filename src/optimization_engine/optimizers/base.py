"""Base classes shared by all optimizers."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from optimization_engine.constraints import (
    ConstraintLayer,
    coerce_layers,
    effective_layers,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle: audit imports this module
    from optimization_engine.optimizers.audit import AuditReport

_LOG = logging.getLogger(__name__)

#: How negative the smallest eigenvalue of a covariance may be, as a fraction
#: of its trace, before the matrix is refused as indefinite. Estimates on the
#: PSD boundary — singular, shrunk, detoned, or clipped by ``nearest_psd`` —
#: carry eigenvalues a few ulps below zero, many orders inside this.
PSD_RTOL = 1e-8


class NonPSDCovarianceError(ValueError):
    """The covariance handed to an optimizer has a materially negative eigenvalue.

    Every solver call wraps the covariance in ``cp.psd_wrap``, which tells
    CVXPY to trust it rather than check it. On an indefinite matrix that trust
    buys a quadratic form that can go negative — a long-short minimum-variance
    solve came back ``optimal`` at ``w'Σw = −0.0445`` — so the matrix is
    refused before any method sees it.

    Attributes:
        min_eigenvalue: The smallest eigenvalue of the symmetrized matrix.
        trace: Its trace, the scale the tolerance is measured against.
    """

    def __init__(self, min_eigenvalue: float, trace: float) -> None:
        """Build the error from the eigenvalue that failed the check.

        Args:
            min_eigenvalue: The smallest eigenvalue found.
            trace: The matrix's trace.
        """
        self.min_eigenvalue = float(min_eigenvalue)
        self.trace = float(trace)
        share = abs(self.min_eigenvalue) / self.trace if self.trace > 0 else float("inf")
        super().__init__(
            "The covariance matrix is not positive semi-definite: its smallest "
            f"eigenvalue is {self.min_eigenvalue:.4g} ({share:.2%} of its "
            "trace), so some portfolios would have negative variance and a "
            "solve could report one as optimal. Repair it with "
            "optimization_engine.nearest_psd — the eigenvalue clipping every "
            "estimator in data.covariance already applies — or check how it "
            "was built: pairwise-complete estimates and hand-edited "
            "correlations are the usual causes."
        )


def check_covariance_psd(sigma: np.ndarray | None) -> None:
    """Refuse a covariance whose smallest eigenvalue is materially negative.

    Args:
        sigma: The covariance, aligned to the solve's universe, or ``None``.
            A matrix with non-finite entries is left to the checks that own
            that failure, which name the asset rather than an eigenvalue.

    Raises:
        NonPSDCovarianceError: If the smallest eigenvalue is below
            ``−PSD_RTOL · trace``.
    """
    if sigma is None:
        return
    values = np.asarray(sigma, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        return
    symmetric = (values + values.T) / 2.0
    smallest = float(np.linalg.eigvalsh(symmetric)[0])
    trace = float(np.trace(symmetric))
    if smallest < -PSD_RTOL * max(abs(trace), np.finfo(float).tiny):
        raise NonPSDCovarianceError(smallest, trace)


@dataclass
class PortfolioConstraints:
    """Bounds, group constraints, exposure limits and a turnover budget.

    The constraints are applied uniformly by all CVXPY-based optimizers
    via the helper ``build_constraints``.

    Attributes:
        bounds: ``asset -> (min, max)`` weight.
        groups: ``asset -> group`` label (e.g. asset class).
        group_bounds: ``group -> (min, max)`` aggregate weight.
        fully_invested: Force ``sum(w) == 1``.
        long_only: Disallow negative weights. Tightens any bound whose
            minimum is negative up to ``0``.
        leverage: Cap on gross exposure ``Σ|w_i|``. ``None`` means uncapped
            (and, under ``long_only`` with a unit budget, gross is 1 anyway).
        target_return: Hard expected-return target ``μ'w == R*``.
        target_volatility: Hard volatility cap ``√(w'Σw) ≤ σ*``.
        previous_weights: The book being traded *from*. Required for a
            turnover budget and for reporting realized turnover.
        turnover_limit: Cap on ``Σ|w_i − w_prev,i|``. A one-way turnover of
            0.20 means at most 20% of the portfolio changes hands.
        benchmark_weights: The index the mandate is measured against. Carried
            on the constraints rather than on the objective because it is what
            the two limits below are expressed relative to.
        max_tracking_error: Cap on annualized active risk,
            ``√((w−b)'Σ(w−b)) ≤ TE*``. Imposed as the equivalent quadratic so
            the problem stays a QP. Needs both a benchmark and a covariance
            matrix; an optimizer that has neither reports the limit rather
            than silently dropping it.
        max_active_share: Cap on ``½·Σ|w_i − b_i|``. A positions-based limit
            that binds even in a calm market, where a tracking-error budget
            quietly permits a portfolio that shares nothing with its index.
        constraint_layers: Additional levels of the allocation policy, each
            slicing the universe its own way — sub-asset-class inside asset
            class, currency across all of them. ``groups``/``group_bounds``
            remain the first layer, so nothing that worked before changes;
            see :mod:`optimization_engine.constraints`.
    """

    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    group_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    fully_invested: bool = True
    long_only: bool = True
    leverage: float | None = None
    target_return: float | None = None
    target_volatility: float | None = None
    previous_weights: dict[str, float] | None = None
    turnover_limit: float | None = None
    benchmark_weights: dict[str, float] | None = None
    max_tracking_error: float | None = None
    max_active_share: float | None = None
    constraint_layers: tuple[ConstraintLayer, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Accept the mapping form a YAML round-trip produces, so a config
        # loaded from disk and one built in memory behave identically.
        """Coerce the constraint layers into their canonical form.

        A YAML round-trip produces mappings where the in-memory form has
        :class:`~optimization_engine.constraints.ConstraintLayer` objects. Coercing
        here is what makes a config loaded from disk and one built in memory
        behave identically.

        Raises:
            LayerConfigurationError: If any layer entry is malformed.
        """
        self.constraint_layers = coerce_layers(self.constraint_layers)

    @property
    def layers(self) -> tuple[ConstraintLayer, ...]:
        """Every level of the policy, the legacy grouping first.

        One accessor for the solver, the projection, the compliance check and
        the feasibility report, so none of them can be reading a different
        policy from the others.
        """
        return effective_layers(self)

    @property
    def has_layer_limits(self) -> bool:
        """Whether anything is constrained above the per-asset level."""
        return any(lyr.is_active for lyr in self.layers)

    @property
    def is_benchmark_relative(self) -> bool:
        """Whether any constraint is expressed against a benchmark."""
        return bool(self.benchmark_weights) and (
            self.max_tracking_error is not None or self.max_active_share is not None
        )

    def benchmark_vector(self, assets: list[str]) -> np.ndarray | None:
        """The benchmark's weights aligned to ``assets``, or None if unset.

        Args:
            assets: The universe, in the order the solve indexes it.

        Returns:
            A weight array aligned to ``assets``, or ``None`` when no benchmark
            weights are set. Assets the benchmark does not name are held at
            zero — for a benchmark that is a *subset* of the investable universe
            that is exactly right, and it is the only reading that lets a manager
            hold something the index does not.
        """
        if not self.benchmark_weights:
            return None
        return np.array(
            [float(self.benchmark_weights.get(a, 0.0)) for a in assets], dtype=float
        )

    def get_bounds(
        self, asset: str, default: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """Resolve the effective ``(min, max)`` weight for one asset.

        ``long_only`` is applied here rather than as a separate constraint so
        that every consumer — the CVXPY builder, the projection helpers and
        the post-solve checker — sees exactly the same box.

        Args:
            asset: The asset to resolve.
            default: Bounds to fall back on when the mandate names none.
                ``None`` uses the mandate's own global default.

        Returns:
            A ``(min, max)`` pair in weight units, with the long-only floor
            already applied.
        """
        if asset in self.bounds:
            lo, hi = self.bounds[asset]
            lo, hi = float(lo), float(hi)
        elif default is not None:
            lo, hi = default
        else:
            lo, hi = (0.0, 1.0) if self.long_only else (-1.0, 1.0)
        if self.long_only:
            lo = max(lo, 0.0)
            hi = max(hi, 0.0)
        return lo, hi


@dataclass
class OptimizationResult:
    """Output of an optimizer, with helpers for analytics.

    ``extras`` carries everything that is not the allocation itself: solver
    status, concentration diagnostics, constraint violations, and any
    method-specific output (the Black-Litterman posterior, the realized CVaR,
    the risk-contribution error of an ERC solve).

    Attributes:
        audit: The mandate audit for these weights, or ``None`` when the solve
            was asked not to run one. It carries the same breaches as
            ``extras["violations"]``, as
            :class:`~optimization_engine.optimizers.diagnostics.ConstraintViolation`
            objects rather than sentences — the structured form is the one a
            caller can act on.
    """

    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    extras: dict[str, Any] = field(default_factory=dict)
    # Appended last, and defaulted, because this is a plain dataclass built
    # positionally in places; a new field anywhere else would move the ones
    # after it.
    audit: AuditReport | None = None

    @property
    def solver_status(self) -> str:
        """The solver's own verdict, or ``"unknown"`` when none was recorded.

        Returns:
            Typically ``"optimal"`` or ``"optimal_inaccurate"``. An inaccurate
            status is a usable answer the solver is not confident in, and it is
            reported rather than hidden.
        """
        return str(self.extras.get("solver_status", "unknown"))

    @property
    def violations(self) -> list[str]:
        """Human-readable constraint breaches, empty when fully compliant."""
        return list(self.extras.get("violations", []))

    @property
    def is_compliant(self) -> bool:
        """Whether the solved weights breach no constraint.

        Returns:
            ``True`` when :attr:`violations` is empty. A solver can return an
            answer that violates a constraint within its own tolerance, which is
            why this is checked after the solve rather than assumed from it.
        """
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        """This result as a flat, JSON-serializable mapping.

        Returns:
            The weights as an ``asset -> weight`` dict, the three summary
            statistics as floats, and every diagnostic in :attr:`extras` merged in
            at the top level. This is what the CLI's ``--json`` payload is built
            from.
        """
        return {
            "weights": self.weights.to_dict(),
            "expected_return": float(self.expected_return),
            "expected_volatility": float(self.expected_volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
            **self.extras,
        }


class BaseOptimizer(ABC):
    """Abstract base for all optimizers in the engine.

    Subclasses implement ``_solve`` to return a 1-D weight vector. The base
    class handles input shaping, constraint plumbing, post-solve validation
    and summary statistics for the resulting allocation.
    """

    name: str = "base"

    #: How faithfully this optimizer honours per-asset and group bounds.
    #: ``"hard"``          — enforced inside the convex program.
    #: ``"soft_iterated"`` — solved unconstrained, then projected into the box.
    #: ``"constrained"``   — enforced in the program, up to solver tolerance.
    bounds_mode: str = "hard"

    def __init__(
        self,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        constraints: PortfolioConstraints | None = None,
        risk_free_rate: float = 0.0,
        accept_inaccurate: bool | None = None,
        strict_mandate: bool = False,
    ) -> None:
        """Store the inputs a solve will run against.

        None of them is validated here: the factory checks what each method
        requires, and the feasibility analysis checks whether the mandate can be
        satisfied at all.

        Args:
            expected_returns: Per-asset expected returns, in the same periodicity
                as the covariance. Optional for methods that do not use them.
            cov_matrix: Asset covariance, indexed and columned by asset name.
            constraints: The mandate. Defaults to an unconstrained long-only book
                summing to one.
            risk_free_rate: Per-period risk-free rate, in the same periodicity as
                the inputs. Used by the Sharpe-based objectives and reported in
                the result's summary statistics.
            accept_inaccurate: Whether to take an ``optimal_inaccurate``
                solution when no solver in the fallback chain converges
                exactly. ``False`` refuses it — the solve raises
                :class:`~optimization_engine.optimizers._cvxpy_helpers.SolverFailure`
                rather than report unverified weights as optimal. ``None``,
                the default, inherits from the surrounding solve if there is
                one and refuses otherwise, which is what lets NCO's
                per-cluster sub-optimizers run under the settings of the
                solve that built them. The methods that never reach a solver
                — HRP, HERC, and the naive weightings — ignore this entirely;
                NCO does *not*, because both of its layers are solved by real
                optimizers.
            strict_mandate: Whether a post-solve mandate breach raises rather
                than being reported. ``False`` — the default — keeps the
                long-standing contract: the audit is attached to the result and
                the caller decides. ``True`` raises
                :class:`~optimization_engine.optimizers.audit.MandateViolationError`
                on any violation past tolerance, which is what a caller who
                would rather have no book than a non-compliant one wants. It
                bites hardest on the projecting methods, which is the point —
                they are the ones that can return a book their mandate does not
                permit.
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.constraints = constraints or PortfolioConstraints()
        self.risk_free_rate = float(risk_free_rate)
        self.accept_inaccurate = accept_inaccurate
        self.strict_mandate = bool(strict_mandate)
        #: Populated by subclasses; surfaced through ``result.extras``.
        self._diagnostics: dict[str, Any] = {}

    @property
    def assets(self) -> list[str]:
        """The universe this optimizer will solve over, in column order.

        Returns:
            The covariance matrix's columns when there is one, and the expected
            returns' index otherwise.

        Raises:
            ValueError: If neither input was supplied, so there is no universe to
                infer.
        """
        if self.cov_matrix is not None:
            return list(self.cov_matrix.columns)
        if self.expected_returns is not None:
            return list(self.expected_returns.index)
        raise ValueError("Optimizer needs either cov_matrix or expected_returns")

    @abstractmethod
    def _solve(self) -> np.ndarray: ...

    def optimize(
        self, *, run_post_solve_diagnostics: bool = True
    ) -> OptimizationResult:
        """Solve, validate, and package the allocation with its diagnostics.

        Args:
            run_post_solve_diagnostics: Whether to run the post-solve pass —
                the concentration and exposure summary, the compliance check
                against :attr:`constraints`, and the mandate audit built from
                it. ``True`` is what a top-level solve wants and what every
                caller gets by default. ``False`` is for a solve whose
                constraints are deliberately *not* the mandate, where the check
                would measure the wrong quantity: NCO's per-cluster and
                inter-cluster layers run against budget-and-sign only, because
                a 10% cap on an asset means 10% of the *book* and not 10% of a
                cluster. Skipping it there also returns the cost, which was
                measured at 12.8% of an NCO solve over 15 assets and 12.3% over
                64. With it off,
                ``extras`` carries no ``diagnostics`` or ``violations`` key and
                ``result.audit`` is ``None``: nothing looked, which the result
                says rather than implying compliance.

        Returns:
            An :class:`OptimizationResult` carrying the weights, the three summary
            statistics, and every diagnostic the solve produced — the solver that
            answered, its status, and any constraint the answer breaches. A
            breach is reported rather than raised unless
            :attr:`strict_mandate` is set: by default the caller decides whether
            a violation within solver tolerance is acceptable.

        Raises:
            NonPSDCovarianceError: If the covariance has a materially negative
                eigenvalue. Checked before the solve, on the matrix as given —
                ``BaseOptimizer._sigma_matrix``, not a subclass's posterior.
            RuntimeError: If the solve produced non-finite weights, which means
                the problem is unbounded or numerically degenerate.
            SolverFailure: If no solver in the fallback chain returned a usable
                solution.
            InfeasibleBoundsError: If the projection step cannot reconcile the
                bounds with the budget.
            MandateViolationError: If ``strict_mandate`` is set and the audit
                found a breach past tolerance.
        """
        from optimization_engine.optimizers._cvxpy_helpers import accepting_inaccurate

        # Refused rather than repaired. The engine's estimators already pass
        # every estimate through ``nearest_psd``, so only a direct caller can
        # get here with an indefinite matrix — and repairing it silently would
        # solve a different matrix from the one passed, by an amount nobody
        # chose. The error names the repair instead.
        check_covariance_psd(BaseOptimizer._sigma_matrix(self))

        # The scope covers ``_solve`` and nothing else. Every CVXPY solve this
        # method makes happens in there -- including the ones a sub-optimizer
        # makes on the way down -- while the dust-cleanup projection below runs
        # outside it, on its own terms. Imported here rather than at module
        # scope because ``_cvxpy_helpers`` imports ``PortfolioConstraints``
        # from this module.
        with accepting_inaccurate(self.accept_inaccurate):
            weights = self._solve()
        weights = np.asarray(weights, dtype=float).flatten()
        if not np.isfinite(weights).all():
            raise RuntimeError(
                f"{self.name} produced non-finite weights — the problem is "
                "likely unbounded or numerically degenerate."
            )
        weights = self._clean_weights(weights)
        w = pd.Series(weights, index=self.assets, name="weight")

        mu = self._mu_vector()
        sigma = self._sigma_matrix()
        port_return = float(w.values @ mu) if mu is not None else float("nan")
        port_var = float(w.values @ sigma @ w.values) if sigma is not None else float("nan")
        port_vol = float(np.sqrt(max(port_var, 0.0))) if not np.isnan(port_var) else float("nan")
        if not np.isnan(port_vol) and port_vol > 0 and not np.isnan(port_return):
            sharpe = (port_return - self.risk_free_rate) / port_vol
        else:
            sharpe = float("nan")

        extras: dict[str, Any] = {
            "optimizer": self.name,
            "bounds_mode": self.bounds_mode,
            **self._diagnostics,
        }
        audit: AuditReport | None = None
        if run_post_solve_diagnostics:
            extras.update(self._post_solve_diagnostics(w))
            audit = self._audit(extras.get("diagnostics"))
            if self.strict_mandate and audit is not None and not audit.is_clean:
                from optimization_engine.optimizers.audit import MandateViolationError

                raise MandateViolationError(audit)

        return OptimizationResult(
            weights=w,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            extras=extras,
            audit=audit,
        )

    def _audit(self, diagnostics: Any) -> AuditReport | None:
        """Package the compliance check the diagnostics pass already ran.

        The audit is not a second check. ``portfolio_diagnostics`` calls
        :func:`~optimization_engine.optimizers.diagnostics.check_constraints`
        over the same weights, the same mandate and the same covariance, and
        that sweep is the expensive half of the post-solve pass — running it
        again to build a report would double the cost to produce a second copy
        of the same answer, and would leave two places for the two answers to
        disagree.

        Args:
            diagnostics: The
                :class:`~optimization_engine.optimizers.diagnostics.PortfolioDiagnostics`
                the post-solve pass produced, or ``None``.

        Returns:
            An :class:`~optimization_engine.optimizers.audit.AuditReport`, or
            ``None`` when no diagnostics were computed.
        """
        from optimization_engine.optimizers.audit import AuditReport
        from optimization_engine.optimizers.diagnostics import DEFAULT_TOLERANCE

        if diagnostics is None:
            return None
        return AuditReport(
            violations=tuple(diagnostics.violations), tolerance=DEFAULT_TOLERANCE
        )

    def _post_solve_diagnostics(self, weights: pd.Series) -> dict[str, Any]:
        """Concentration, diversification and constraint-compliance summary."""
        from optimization_engine.optimizers.diagnostics import portfolio_diagnostics

        diag = portfolio_diagnostics(
            weights, cov_matrix=self.cov_matrix, constraints=self.constraints
        )
        if diag.violations:
            _LOG.warning(
                "%s produced weights violating %d constraint(s): %s",
                self.name,
                len(diag.violations),
                "; ".join(v.describe() for v in diag.violations),
            )
        return {"diagnostics": diag, "violations": [v.describe() for v in diag.violations]}

    def _mu_vector(self) -> np.ndarray | None:
        """Expected returns aligned to ``self.assets``.

        Missing entries are treated as zero, which is a strong and usually
        wrong assumption — so it is warned about rather than done silently.
        """
        if self.expected_returns is None:
            return None
        aligned = self.expected_returns.reindex(self.assets)
        missing = [a for a, v in aligned.items() if pd.isna(v)]
        if missing:
            warnings.warn(
                f"{self.name}: no expected return for {len(missing)} asset(s) "
                f"({', '.join(map(str, missing[:5]))}"
                f"{' …' if len(missing) > 5 else ''}); assuming 0.0. "
                "A zero expected return is an active view, not a neutral one.",
                stacklevel=3,
            )
            self._diagnostics["missing_expected_returns"] = [str(m) for m in missing]
        return aligned.fillna(0.0).values

    def _sigma_matrix(self) -> np.ndarray | None:
        if self.cov_matrix is None:
            return None
        return self.cov_matrix.reindex(self.assets, axis=0).reindex(self.assets, axis=1).values

    def _clean_weights(self, w: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Zero out dust, restore the budget, and keep the result inside the box.

        Renormalizing after truncation can push a weight through its bound,
        so the result is re-projected onto the feasible set — group budgets
        included — whenever the naive rescale would breach it. Portfolios that
        are not fully invested (or that net to ~0, where rescaling is
        meaningless) are returned as-is apart from the dust removal.
        """
        from optimization_engine.optimizers._bounds import (
            InfeasibleBoundsError,
            project_to_constraints,
        )

        w = np.where(np.abs(w) < tol, 0.0, w)
        if not self.constraints.fully_invested:
            return w

        s = float(w.sum())
        if abs(s) < 1e-9:
            return w
        rescaled = w / s

        lb = np.array([self.constraints.get_bounds(a)[0] for a in self.assets])
        ub = np.array([self.constraints.get_bounds(a)[1] for a in self.assets])
        if (rescaled >= lb - tol).all() and (rescaled <= ub + tol).all():
            return rescaled
        try:
            return project_to_constraints(rescaled, self.assets, self.constraints)[0]
        except (InfeasibleBoundsError, RuntimeError):
            # Bounds and budget are mutually impossible; the feasibility
            # report is the right place to explain that, so keep the solver's
            # answer rather than silently mangling it.
            return rescaled
