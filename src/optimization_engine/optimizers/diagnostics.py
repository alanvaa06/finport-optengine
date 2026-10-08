"""Post-solve diagnostics: does the answer actually respect what was asked?

An optimizer that returns weights is not the same as an optimizer that
returned *valid* weights. Methods that solve unconstrained and project
(HRP, max-diversification, the naive baselines) can only approximate group
budgets; convex solvers can return ``optimal_inaccurate`` and drift outside
the box by more than a rounding error. These helpers make that visible
instead of leaving it for the analyst to discover in the weights table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optimization_engine.constraints import layer_breaches
from optimization_engine.optimizers.base import PortfolioConstraints

#: Weight drift below this is floating-point noise, not a real violation.
DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ConstraintViolation:
    """One breached constraint, with the size of the breach."""

    kind: str
    label: str
    limit: float
    actual: float

    @property
    def magnitude(self) -> float:
        """How far past the limit the allocation went, in weight terms."""
        return abs(self.actual - self.limit)

    def describe(self) -> str:
        """One line: what was breached, by how much, against which limit.

        Returns:
            Something like ``"Equity bucket: 62.4000% vs limit 60.0000% (off by
            2.4000%)"``.
        """
        return (
            f"{self.label}: {self.actual:.4%} vs limit {self.limit:.4%} "
            f"(off by {self.magnitude:.4%})"
        )


def _turnover(weights: pd.Series, previous: dict[str, float]) -> float:
    """``Σ|w − w_prev|`` over the union of both books.

    Reindexing the previous book onto ``weights`` dropped every holding the
    universe no longer contains, so selling it cost nothing. It is sold —
    the book cannot keep a name it cannot hold — and counts in full.
    """
    prev = pd.Series(previous, dtype=float)
    names = weights.index.union(prev.index)
    held = prev.reindex(names).fillna(0.0)
    return float((weights.reindex(names).fillna(0.0) - held).abs().sum())


def check_constraints(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    tolerance: float = DEFAULT_TOLERANCE,
    cov_matrix: pd.DataFrame | None = None,
) -> list[ConstraintViolation]:
    """List every constraint the solved ``weights`` breach beyond ``tolerance``.

    Returns an empty list when the allocation is fully compliant.

    Args:
        weights: The solved allocation.
        constraints: What was asked for.
        tolerance: Drift below this is floating-point noise.
        cov_matrix: Needed to check a tracking-error budget. Without it that
            one limit goes unchecked — which is why the methods that cannot
            impose it are the same ones that cannot verify it, and why the
            factory warns rather than letting it pass silently.
    """
    out: list[ConstraintViolation] = []

    for asset, w in weights.items():
        lo, hi = constraints.get_bounds(str(asset))
        if w < lo - tolerance:
            out.append(ConstraintViolation("bound", f"{asset} lower bound", lo, float(w)))
        if w > hi + tolerance:
            out.append(ConstraintViolation("bound", f"{asset} upper bound", hi, float(w)))

    if constraints.fully_invested:
        total = float(weights.sum())
        if abs(total - 1.0) > tolerance:
            out.append(ConstraintViolation("budget", "Sum of weights", 1.0, total))

    if constraints.long_only:
        negative = float(weights[weights < -tolerance].sum())
        if negative < 0:
            out.append(
                ConstraintViolation("long_only", "Short exposure", 0.0, negative)
            )

    if constraints.leverage is not None:
        gross = float(weights.abs().sum())
        if gross > constraints.leverage + tolerance:
            out.append(
                ConstraintViolation(
                    "leverage", "Gross exposure", float(constraints.leverage), gross
                )
            )

    # Every layer of the policy at once — asset class, sub-asset class,
    # currency — measured against the limit as a share of the book, so a
    # percent-of-parent cap is reported in the same units as the weights it
    # is being compared with.
    for label, _side, limit, actual in layer_breaches(weights, constraints.layers, tolerance):
        out.append(ConstraintViolation("group", label, limit, actual))

    if constraints.previous_weights and constraints.turnover_limit is not None:
        turnover = _turnover(weights, constraints.previous_weights)
        if turnover > float(constraints.turnover_limit) + tolerance:
            out.append(
                ConstraintViolation(
                    "turnover", "Turnover", float(constraints.turnover_limit), turnover
                )
            )

    out.extend(_benchmark_violations(weights, constraints, tolerance, cov_matrix))
    return out


def _benchmark_violations(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    tolerance: float,
    cov_matrix: pd.DataFrame | None,
) -> list[ConstraintViolation]:
    """Active-share and tracking-error breaches, when a benchmark is set."""
    if not constraints.benchmark_weights:
        return []
    out: list[ConstraintViolation] = []
    bench = (
        pd.Series(constraints.benchmark_weights).reindex(weights.index).fillna(0.0)
    )
    active = weights - bench

    if constraints.max_active_share is not None:
        share = float(active.abs().sum() / 2.0)
        limit = float(constraints.max_active_share)
        if share > limit + tolerance:
            out.append(
                ConstraintViolation("active_share", "Active share", limit, share)
            )

    if constraints.max_tracking_error is not None and cov_matrix is not None:
        aligned = cov_matrix.reindex(weights.index, axis=0).reindex(
            weights.index, axis=1
        )
        variance = float(active.values @ aligned.values @ active.values)
        te = float(np.sqrt(max(variance, 0.0)))
        limit = float(constraints.max_tracking_error)
        if te > limit + tolerance:
            out.append(
                ConstraintViolation("tracking_error", "Tracking error", limit, te)
            )

    return out


# ---------------------------------------------------------------------------
# Concentration / diversification
# ---------------------------------------------------------------------------


def herfindahl_index(weights: pd.Series | np.ndarray) -> float:
    """Sum of squared weights. ``1/N`` when equally weighted, ``1`` at a corner.

    Args:
        weights: Portfolio weights, as fractions of the book.

    Returns:
        The concentration index, in ``[1/N, 1]``.
    """
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w**2))


def effective_n(weights: pd.Series | np.ndarray) -> float:
    """Inverse Herfindahl: how many equally-weighted positions this is worth.

    A 10-asset portfolio with 90% in one name has an effective N near 1.2,
    not 10 — this is the number to look at before calling a book diversified.

    Args:
        weights: Portfolio weights, as fractions of the book.

    Returns:
        The effective number of positions, between ``1`` and the number of
        assets. It counts *capital*, not risk; for the risk-side answer see
        :func:`~optimization_engine.analytics.diversification.effective_number_of_bets`.
    """
    hhi = herfindahl_index(weights)
    return float(1.0 / hhi) if hhi > 0 else float("nan")


def diversification_ratio(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Weighted-average asset volatility divided by portfolio volatility.

    ``1.0`` means correlations bought you nothing; higher is better. This is
    the objective the max-diversification optimizer maximizes.

    Args:
        weights: Portfolio weights, as fractions of the book.
        cov_matrix: Asset covariance over the same universe.

    Returns:
        The ratio, at least ``1.0`` for a long-only book.
    """
    assets = list(weights.index)
    sigma = cov_matrix.reindex(assets, axis=0).reindex(assets, axis=1).values
    w = weights.values.astype(float)
    weighted_avg_vol = float(np.abs(w) @ np.sqrt(np.diag(sigma)))
    port_vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    return weighted_avg_vol / port_vol if port_vol > 0 else float("nan")


def effective_n_risk(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Effective number of *risk* bets: inverse Herfindahl of risk contributions.

    Weight diversification and risk diversification are different things — a
    60/40 book is diversified by weight and concentrated in equity risk.

    Args:
        weights: Portfolio weights, as fractions of the book.
        cov_matrix: Asset covariance over the same universe.

    Returns:
        A number between 1 and the asset count. It still measures assets, not
        independent factors — for that see
        :func:`~optimization_engine.analytics.diversification.effective_number_of_bets`.
    """
    rc = risk_contributions(weights, cov_matrix)
    return effective_n(rc.values)


def risk_contributions(
    weights: pd.Series, cov_matrix: pd.DataFrame
) -> pd.Series:
    """Per-asset share of total portfolio variance (sums to 1).

    Args:
        weights: Portfolio weights, as fractions of the book.
        cov_matrix: Asset covariance over the same universe.

    Returns:
        One share per asset, summing to 1. The Euler decomposition splits
        volatility and variance in the same proportions, so the share reads
        correctly either way.
    """
    assets = list(weights.index)
    sigma = cov_matrix.reindex(assets, axis=0).reindex(assets, axis=1).values
    w = weights.values.astype(float)
    total = float(w @ sigma @ w)
    if total <= 0:
        return pd.Series(np.zeros(len(assets)), index=assets)
    return pd.Series(w * (sigma @ w) / total, index=assets)


def risk_decomposition(
    weights: pd.Series, cov_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Full Euler risk decomposition of portfolio volatility.

    Reporting contributions in volatility units (not just shares) is what lets
    an analyst say "this sleeve costs me 4.2% of vol", which is the sentence
    a risk committee actually needs.

    Args:
        weights: Portfolio weights, as fractions of the book.
        cov_matrix: Asset covariance. Its periodicity sets the units of every
            volatility figure returned.

    Returns:
        A frame indexed by asset with:

        ``weight``            — the allocation.
        ``marginal_risk``     — ∂σ_p/∂w_i, the volatility added by one more unit.
        ``contribution``      — w_i · ∂σ_p/∂w_i, in annualized volatility units.
                                These sum exactly to portfolio volatility.
        ``share_of_risk``     — contribution as a fraction of portfolio volatility.
        ``standalone_vol``    — the asset's own volatility, for reference.
    """
    assets = list(weights.index)
    sigma = cov_matrix.reindex(assets, axis=0).reindex(assets, axis=1).values
    w = weights.values.astype(float)
    port_vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    if port_vol <= 0:
        zeros = np.zeros(len(assets))
        return pd.DataFrame(
            {
                "weight": w,
                "marginal_risk": zeros,
                "contribution": zeros,
                "share_of_risk": zeros,
                "standalone_vol": np.sqrt(np.diag(sigma)),
            },
            index=assets,
        )
    marginal = (sigma @ w) / port_vol
    contribution = w * marginal
    return pd.DataFrame(
        {
            "weight": w,
            "marginal_risk": marginal,
            "contribution": contribution,
            "share_of_risk": contribution / port_vol,
            "standalone_vol": np.sqrt(np.diag(sigma)),
        },
        index=assets,
    )


@dataclass(frozen=True)
class PortfolioDiagnostics:
    """Concentration, diversification and compliance summary for an allocation."""

    n_positions: int
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    max_weight: float
    herfindahl: float
    effective_n: float
    effective_n_risk: float
    diversification_ratio: float
    turnover: float | None = None
    violations: tuple[ConstraintViolation, ...] = field(default_factory=tuple)

    @property
    def is_compliant(self) -> bool:
        """Whether the allocation breaches no constraint.

        Returns:
            ``True`` when :attr:`violations` is empty.
        """
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        """These diagnostics as a flat, JSON-serializable mapping.

        Returns:
            Every concentration and exposure figure as a plain number, plus the
            violations rendered as human-readable strings. This is what reaches
            the CLI's ``--json`` payload and the report workbook.
        """
        return {
            "n_positions": self.n_positions,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "long_exposure": self.long_exposure,
            "short_exposure": self.short_exposure,
            "max_weight": self.max_weight,
            "herfindahl": self.herfindahl,
            "effective_n": self.effective_n,
            "effective_n_risk": self.effective_n_risk,
            "diversification_ratio": self.diversification_ratio,
            "turnover": self.turnover,
            "violations": [v.describe() for v in self.violations],
        }


def portfolio_diagnostics(
    weights: pd.Series,
    cov_matrix: pd.DataFrame | None = None,
    constraints: PortfolioConstraints | None = None,
    active_tolerance: float = 1e-4,
) -> PortfolioDiagnostics:
    """Summarize an allocation's concentration, exposure and compliance.

    Args:
        weights: The solved weights, as fractions of the book.
        cov_matrix: Asset covariance. Without it the risk-side figures —
            risk-effective N and the diversification ratio — cannot be
            computed.
        constraints: The mandate to check compliance against. Without it no
            violations are reported, which is not the same as compliance.
        active_tolerance: Weight below which a position is treated as flat
            rather than held, for counting positions.

    Returns:
        A :class:`PortfolioDiagnostics` with the concentration and exposure
        figures and any constraint violations found.
    """
    w = weights.astype(float)
    longs = float(w[w > 0].sum())
    shorts = float(w[w < 0].sum())
    turnover = None
    violations: tuple[ConstraintViolation, ...] = ()
    if constraints is not None:
        violations = tuple(check_constraints(w, constraints, cov_matrix=cov_matrix))
        if constraints.previous_weights:
            turnover = _turnover(w, constraints.previous_weights)

    return PortfolioDiagnostics(
        n_positions=int((w.abs() > active_tolerance).sum()),
        gross_exposure=float(w.abs().sum()),
        net_exposure=float(w.sum()),
        long_exposure=longs,
        short_exposure=shorts,
        max_weight=float(w.max()) if len(w) else float("nan"),
        herfindahl=herfindahl_index(w),
        effective_n=effective_n(w),
        effective_n_risk=(
            effective_n_risk(w, cov_matrix) if cov_matrix is not None else float("nan")
        ),
        diversification_ratio=(
            diversification_ratio(w, cov_matrix)
            if cov_matrix is not None
            else float("nan")
        ),
        turnover=turnover,
        violations=violations,
    )
