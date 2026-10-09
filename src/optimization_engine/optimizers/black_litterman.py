"""Black-Litterman expected-return blender.

Combines the implied equilibrium returns from a market-cap weighted
portfolio with subjective views to produce a posterior expected-return
vector. The optimizer plugs that posterior into a standard mean-variance
solve so all bound/group/turnover constraints continue to apply.

Two things this implementation does that a minimal one does not:

* **Relative views.** A view is a row of the pick matrix ``P``, so
  "European equity beats US equity by 2%" is expressible, not just
  "European equity returns 8%". Absolute views are the special case of a
  single ``+1`` in the row. Relative views are how practitioners actually
  hold opinions, and they leave the overall market level untouched.
* **He-Litterman Ω.** The default view uncertainty is
  ``Ω = diag(P · τΣ · Pᵀ)``, which scales each view's error with the prior
  variance of *that view's* portfolio. Using ``τ·σ_i²`` instead — as a
  per-asset default does — is only correct for single-asset views.

References:
    Black, F., & Litterman, R. (1992). Global portfolio optimization.
    He, G., & Litterman, R. (1999). The intuition behind Black-Litterman
    model portfolios. Goldman Sachs Quantitative Research.
    Idzorek, T. (2007). A step-by-step guide to the Black-Litterman model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimization_engine.optimizers.base import BaseOptimizer
from optimization_engine.optimizers.mean_variance import MeanVarianceOptimizer

# Below this, a view's pick portfolio has no prior variance at all: ``p'τΣp``
# is zero to the last bit rather than merely small. 1e-300 is denormal
# territory, so any view portfolio built from a real annualized covariance sits
# many orders of magnitude above it; the threshold catches exact degeneracy
# without passing judgement on a genuinely low-variance view.
_MIN_VIEW_VARIANCE = 1e-300


@dataclass(frozen=True)
class View:
    """One Black-Litterman view.

    Attributes:
        weights: ``asset -> coefficient`` in the view portfolio. An absolute
            view on a single asset is ``{"US_Equity": 1.0}``; a relative view
            is ``{"US_Equity": 1.0, "Intl_Equity": -1.0}``.
        expected_return: The return the view portfolio is expected to earn.
            For a relative view this is the *spread*, not a level.
        confidence: Optional variance of the view's error term (Ω diagonal).
            Smaller means more confident. ``None`` uses the He-Litterman
            default ``p' τΣ p``.
        label: Optional human-facing name, echoed back in diagnostics.
    """

    weights: dict[str, float]
    expected_return: float
    confidence: float | None = None
    label: str = ""

    @property
    def is_relative(self) -> bool:
        """True when the view's coefficients net to ~0 (a spread, not a level)."""
        return abs(sum(self.weights.values())) < 1e-9

    def describe(self) -> str:
        """The view as a sentence.

        Returns:
            A relative view reads as ``"A outperforms B by 2.00%"``; an absolute
            one as ``"A returns 6.00%"``. Which of the two it is follows from the
            coefficients, not from a flag.
        """
        longs = [a for a, v in self.weights.items() if v > 0]
        shorts = [a for a, v in self.weights.items() if v < 0]
        if shorts:
            return (
                f"{' + '.join(longs)} outperforms {' + '.join(shorts)} "
                f"by {self.expected_return:.2%}"
            )
        return f"{' + '.join(longs)} returns {self.expected_return:.2%}"


def normalize_views(
    views: dict[str, float] | list[View] | None,
    view_confidences: dict[str, float] | None = None,
) -> list[View]:
    """Accept either the simple ``asset -> return`` mapping or full ``View``s.

    The mapping form stays supported because it is what the config schema and
    the simple UI table produce; it is interpreted as one absolute view per
    entry.

    Args:
        views: ``{asset: annualized return}``, a list of :class:`View`
            objects, or ``None`` for no views.
        view_confidences: ``{asset: variance}``. Only applies to the mapping
            form; a :class:`View` carries its own confidence.

    Returns:
        One :class:`View` per input view, in the order given.
    """
    if not views:
        return []
    if isinstance(views, dict):
        confidences = view_confidences or {}
        return [
            View(
                weights={asset: 1.0},
                expected_return=float(q),
                confidence=(
                    float(confidences[asset]) if asset in confidences else None
                ),
                label=f"{asset} absolute",
            )
            for asset, q in views.items()
        ]
    return list(views)


def _identify_view(view: View, position: int) -> str:
    """Name one view for an error message.

    Args:
        view: The view to name.
        position: Its 1-based place in the list the caller passed.

    Returns:
        The view's label when it has one, otherwise its position and the assets
        it names — either way, enough for the caller to find the row they typed.
    """
    if view.label:
        return f"view {position} ({view.label!r})"
    return f"view {position} on {', '.join(view.weights)}"


def build_pick_matrix(
    views: list[View], assets: list[str]
) -> tuple[np.ndarray, np.ndarray, list[View]]:
    """Assemble ``(P, Q)`` from a list of views.

    Every asset a view names must be in the universe. Views used to be trimmed
    to fit it instead: a basket view's out-of-universe legs were zeroed, so
    "long A versus short B" quietly became "long A" — a different opinion, with
    a different effect on the posterior — and a view whose assets were *all*
    absent was dropped whole. Both changed what the caller said without saying
    so, and a view on a name the portfolio cannot hold is not a view, so both
    now raise.

    Args:
        views: The views to encode.
        assets: The universe, in the order the solve indexes it.

    Returns:
        ``(P, Q, kept)`` — the pick matrix, the view returns, and the views
        themselves in input order. Nothing is dropped, so ``kept`` is always
        every view passed; it stays in the signature because callers unpack
        three values.

    Raises:
        ValueError: If any view references an asset outside ``assets``. The
            message names every missing asset and every view that referenced
            one, not just the first hit.
    """
    idx = {a: i for i, a in enumerate(assets)}
    rows: list[np.ndarray] = []
    q: list[float] = []
    # An insertion-ordered set: the message lists missing names in the order
    # the views name them, so the same inputs always produce the same message.
    missing_assets: dict[str, None] = {}
    offenders: list[str] = []
    for position, view in enumerate(views, start=1):
        absent = [a for a in view.weights if a not in idx]
        if absent:
            for asset in absent:
                missing_assets[asset] = None
            named = _identify_view(view, position)
            # When every leg is missing the name already carries the assets;
            # "view 2 on ZZZZ references ZZZZ" is noise.
            offenders.append(
                named
                if len(absent) == len(view.weights)
                else f"{named} references {', '.join(absent)}"
            )
            continue
        row = np.zeros(len(assets))
        for asset, coeff in view.weights.items():
            row[idx[asset]] = float(coeff)
        rows.append(row)
        q.append(float(view.expected_return))
    if missing_assets:
        raise ValueError(
            "Black-Litterman views reference assets that are not in the "
            f"optimization universe: {', '.join(missing_assets)} ("
            + "; ".join(offenders)
            + "). A view on an asset the portfolio cannot hold says nothing "
            "about the assets being optimized, and a basket view reduced to "
            "the legs that happen to be held is a different view from the one "
            "written. Drop these views, or widen the universe to include the "
            "missing assets."
        )
    if not rows:
        return np.zeros((0, len(assets))), np.zeros(0), []
    return np.vstack(rows), np.array(q), list(views)


def implied_risk_aversion(
    market_return: float, market_variance: float, risk_free_rate: float = 0.0
) -> float:
    """Back out δ from the market portfolio: ``δ = (E[r_m] − rf) / σ_m²``.

    Picking δ by hand is the weakest link in most Black-Litterman
    implementations — it sets the entire level of the equilibrium returns.
    Calibrating it to an observed market Sharpe ratio makes the prior
    reproducible.

    Args:
        market_return: The market portfolio's expected return, annualized.
        market_variance: Its variance, on the same annual basis.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        The risk-aversion coefficient δ.

    Raises:
        ValueError: If ``market_variance`` is not positive.
    """
    if market_variance <= 0:
        raise ValueError(
            "Market variance must be positive to imply a risk-aversion "
            "coefficient."
        )
    return float((market_return - risk_free_rate) / market_variance)


def market_portfolio(
    market_weights: pd.Series | dict[str, float] | None, assets: list[str]
) -> pd.Series:
    """The equilibrium portfolio the prior is reverse-optimized from.

    One definition for the solve and for the pre-flight
    (:func:`~optimization_engine.optimizers.factory.effective_expected_returns`),
    so the posterior a return target is checked against is the posterior the
    solve optimizes.

    Args:
        market_weights: Market-capitalization weights, in any units, or
            ``None`` for equal weights — a real assumption, not a neutral one.
        assets: The universe, in the order the solve indexes it.

    Returns:
        Weights over ``assets`` summing to one. Assets the caps do not name
        are held at zero.

    Raises:
        ValueError: If the caps sum to zero or less over ``assets``, leaving
            no equilibrium portfolio to reverse-optimize from.
    """
    if market_weights is None:
        return pd.Series(np.ones(len(assets)) / len(assets), index=assets)
    mkt = pd.Series(market_weights, dtype=float).reindex(assets).fillna(0.0)
    total = float(mkt.sum())
    if total <= 0:
        raise ValueError(
            "Market-cap weights sum to zero, so there is no equilibrium "
            "portfolio to reverse-optimize from. Provide positive weights "
            "or leave them empty for equal weights."
        )
    return mkt / total


def resolve_risk_aversion(
    market: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float,
    *,
    calibrate: bool,
    market_return: float | None,
    risk_free_rate: float,
) -> float:
    """The δ the prior is built with: the configured one, or the implied one.

    Shared by the solve and the pre-flight for the same reason as
    :func:`market_portfolio`. The pre-flight used to read the configured δ
    regardless of ``calibrate``, so with a 12% market return it checked
    targets against a posterior built on δ = 2.5 while the solve used the
    implied 6.95 — and called a reachable target unreachable.

    Args:
        market: The equilibrium portfolio, from :func:`market_portfolio`.
        cov_matrix: The prior covariance Σ.
        risk_aversion: The configured δ, used when not calibrating.
        calibrate: Imply δ from the market's Sharpe ratio instead.
        market_return: The market's expected return. Required when
            ``calibrate`` is set.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        The risk-aversion coefficient δ.

    Raises:
        ConfigurationError: If ``calibrate`` is set with no ``market_return``.
        ValueError: If the market portfolio has no variance to imply δ from.
    """
    if not calibrate:
        return float(risk_aversion)
    _require_market_return(calibrate, market_return)
    weights = market.reindex(cov_matrix.columns).fillna(0.0).values
    market_var = float(weights @ cov_matrix.values @ weights)
    return implied_risk_aversion(float(market_return), market_var, risk_free_rate)


def _require_market_return(calibrate: bool, market_return: float | None) -> None:
    """Refuse a calibration that has nothing to calibrate against.

    It used to fall back to the hand-set δ the flag exists to replace, and
    say nothing.
    """
    from optimization_engine.optimizers import ConfigurationError

    if calibrate and market_return is None:
        raise ConfigurationError(
            "calibrate_risk_aversion is set but no market_return was given. "
            "Calibration implies delta from the market's Sharpe ratio, so it needs "
            "the market's expected return (bl_market_return in the config). "
            "Supply one, or turn calibration off to use risk_aversion as set."
        )


def implied_equilibrium_returns(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Reverse-optimize equilibrium returns from market weights.

    ``π = δ · Σ · w_mkt + rf``

    The ``rf`` term makes ``π`` a *total* expected return rather than an
    excess return, matching the convention used everywhere else in the
    engine (expected returns are totals, and Sharpe subtracts ``rf``).

    Args:
        market_weights: The market portfolio, as fractions.
        cov_matrix: Asset covariance, annualized.
        risk_aversion: The ``δ`` coefficient. See
            :func:`implied_risk_aversion` for deriving it rather than guessing.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Annualized total expected returns, one per asset.
    """
    w = market_weights.reindex(cov_matrix.columns).fillna(0.0).values
    pi = risk_aversion * cov_matrix.values @ w
    return pd.Series(pi + risk_free_rate, index=cov_matrix.columns)


def black_litterman_posterior(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    views: dict[str, float] | list[View] | None = None,
    view_confidences: dict[str, float] | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute the Black-Litterman posterior mean and covariance.

    Args:
        cov_matrix: The prior covariance Σ, annualized.
        market_weights: The market portfolio the equilibrium returns are
            reverse-optimized from.
        views: Either ``{asset: annualized return}`` (absolute views) or a
            list of :class:`View` objects (absolute *or* relative).
        view_confidences: Per-view variances. Only applies to the mapping
            form; ``View.confidence`` carries it otherwise.
        tau: Scalar on the prior covariance, in ``(0, 1]``. Small means the
            equilibrium is trusted and the views move the posterior little.
        risk_aversion: The δ in the reverse optimization.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        ``(posterior_mean, posterior_covariance)`` where the covariance is
        ``Σ + M`` — the prior plus the posterior parameter uncertainty, which
        is what should be fed to the downstream mean-variance solve.

    Raises:
        ValueError: If ``tau`` lies outside ``(0, 1]``, which makes the prior
            either degenerate or wider than the return distribution itself; if
            a view references an asset outside ``cov_matrix``'s columns (see
            :func:`build_pick_matrix`); if a view's pick portfolio has
            numerically zero prior variance, leaving the He-Litterman default
            uncertainty undefined; or if a view confidence is not strictly
            positive — a zero would assert a view held with perfect certainty.
    """
    if not 0 < tau <= 1:
        raise ValueError(
            f"tau must be in (0, 1]; got {tau}. It scales the prior "
            "covariance, so values outside that range make the prior either "
            "degenerate or wider than the return distribution itself."
        )
    pi = implied_equilibrium_returns(
        market_weights, cov_matrix, risk_aversion, risk_free_rate
    )
    view_list = normalize_views(views, view_confidences)
    if not view_list:
        return pi, cov_matrix

    assets = list(cov_matrix.columns)
    P, Q, kept = build_pick_matrix(view_list, assets)
    if P.shape[0] == 0:
        return pi, cov_matrix

    sigma = cov_matrix.values
    tau_sigma = tau * sigma

    # He-Litterman default: each view's uncertainty is the prior variance of
    # that view's own portfolio. Explicit confidences override per view.
    default_omega = np.diag(P @ tau_sigma @ P.T).copy()
    # A pick portfolio with no prior variance has no defensible uncertainty:
    # ``p'τΣp`` is the whole He-Litterman default. Flooring it, as this did,
    # handed such a view a variance of 1e-12 — near-perfect confidence — and
    # let it dominate the posterior it was meant to nudge. The comparison is
    # written as ``not >=`` so a NaN projection is caught too.
    degenerate = [
        j for j, var in enumerate(default_omega) if not var >= _MIN_VIEW_VARIANCE
    ]
    if degenerate:
        detail = "; ".join(
            f"{_identify_view(kept[j], j + 1)} projects onto prior variance "
            f"{float(default_omega[j]):.3g}"
            for j in degenerate
        )
        raise ValueError(
            "A Black-Litterman view has no prior variance to scale its "
            f"uncertainty by: {detail}, below {_MIN_VIEW_VARIANCE:g}. The "
            "default view uncertainty is the prior variance of the view's own "
            "portfolio, so a pick portfolio the covariance holds at zero "
            "variance - a spread between assets it treats as identical, or a "
            "row of zero coefficients - has no uncertainty to state, and the "
            "prior cannot be blended with it either: it is singular in that "
            "direction. Drop the view, or state it over assets the covariance "
            "tells apart."
        )
    omega_diag = np.array(
        [
            view.confidence if view.confidence is not None else default_omega[j]
            for j, view in enumerate(kept)
        ],
        dtype=float,
    )
    if (omega_diag <= 0).any():
        raise ValueError(
            "View confidences are variances and must be strictly positive; a "
            "zero would assert a view held with perfect certainty."
        )
    omega = np.diag(omega_diag)

    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.inv(omega)
    M = np.linalg.pinv(tau_sigma_inv + P.T @ omega_inv @ P)
    bl_mean = M @ (tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q)
    bl_cov = sigma + M

    return (
        pd.Series(bl_mean, index=assets),
        pd.DataFrame(bl_cov, index=assets, columns=assets),
    )


class BlackLittermanOptimizer(BaseOptimizer):
    """Black-Litterman + mean-variance optimizer.

    Pass ``market_weights`` (or omit for equal weights), ``views``, and
    optionally ``view_confidences``. The posterior is built and a downstream
    mean-variance solve respecting all the standard constraints is run.

    Set ``calibrate_risk_aversion=True`` with a ``market_return`` to imply δ
    from the market's own Sharpe ratio instead of hard-coding 2.5.
    """

    name = "black_litterman"
    bounds_mode = "hard"

    def __init__(
        self,
        *args,
        market_weights: pd.Series | dict[str, float] | None = None,
        views: dict[str, float] | list[View] | None = None,
        view_confidences: dict[str, float] | None = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        market_return: float | None = None,
        calibrate_risk_aversion: bool = False,
        **kwargs,
    ) -> None:
        """Set the equilibrium prior, the views, and how much to trust each.

        Args:
            *args: Passed to :class:`~optimization_engine.optimizers.base.BaseOptimizer`.
            market_weights: Market-capitalization weights to reverse-optimize the
                equilibrium returns from. Defaults to equal weights, which is a
                real assumption rather than a neutral one.
            views: Either the simple ``asset -> expected return`` mapping or full
                :class:`View` objects, which can express relative views.
            view_confidences: ``view -> confidence``. Absent views fall back to
                the standard proportional-to-variance uncertainty.
            tau: Scalar on the prior's uncertainty. Small means the equilibrium is
                trusted and the views move the posterior little.
            risk_aversion: The ``δ`` in the reverse optimization. Ignored when
                ``calibrate_risk_aversion`` is set.
            market_return: The market's own expected return, used for calibration.
            calibrate_risk_aversion: Derive ``δ`` from the market's Sharpe ratio
                instead of using the hard-coded default. Requires
                ``market_return``.
            **kwargs: Passed to the base class.

        Raises:
            ConfigurationError: If ``calibrate_risk_aversion`` is set without a
                ``market_return`` to calibrate against.
        """
        super().__init__(*args, **kwargs)
        _require_market_return(bool(calibrate_risk_aversion), market_return)
        self.market_weights = (
            pd.Series(market_weights) if isinstance(market_weights, dict) else market_weights
        )
        self.views = views or {}
        self.view_confidences = view_confidences or {}
        self.tau = float(tau)
        self.risk_aversion = float(risk_aversion)
        self.market_return = market_return
        self.calibrate_risk_aversion = bool(calibrate_risk_aversion)
        # The posterior the last solve optimized against. Kept apart from the
        # inputs so that a second ``optimize()`` starts from the same prior as
        # the first — writing the posterior back over ``cov_matrix`` made every
        # call reverse-optimize from the previous call's answer.
        self._posterior_mean: pd.Series | None = None
        self._posterior_cov: pd.DataFrame | None = None

    def _market_portfolio(self) -> pd.Series:
        return market_portfolio(self.market_weights, self.assets)

    def _solve(self) -> np.ndarray:
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix required for Black-Litterman")
        mkt = self._market_portfolio()

        delta = resolve_risk_aversion(
            mkt,
            self.cov_matrix,
            self.risk_aversion,
            calibrate=self.calibrate_risk_aversion,
            market_return=self.market_return,
            risk_free_rate=self.risk_free_rate,
        )
        if self.calibrate_risk_aversion:
            self._diagnostics["implied_risk_aversion"] = delta

        post_mean, post_cov = black_litterman_posterior(
            self.cov_matrix,
            mkt,
            self.views,
            self.view_confidences,
            tau=self.tau,
            risk_aversion=delta,
            risk_free_rate=self.risk_free_rate,
        )

        prior = implied_equilibrium_returns(
            mkt, self.cov_matrix, delta, self.risk_free_rate
        )
        view_list = normalize_views(self.views, self.view_confidences)
        self._diagnostics.update(
            {
                "bl_prior_returns": prior,
                "bl_posterior_returns": post_mean,
                "bl_view_impact": post_mean - prior,
                "bl_views": [v.describe() for v in view_list],
                "bl_tau": self.tau,
                "bl_risk_aversion": delta,
                "bl_market_weights": mkt,
            }
        )

        # ``π = δΣw`` is the first-order condition of ``μ'w − (δ/2)·w'Σw``;
        # the mean-variance utility is ``μ'w − λ·w'Σw`` with no half. Handing
        # it ``λ = δ/2`` is what makes the two agree — and what makes the
        # defining check of the model hold: with no views, the posterior is
        # the prior and the solve returns the market portfolio. With ``λ = δ``
        # the effective aversion is doubled and the no-view answer lands
        # halfway between the market and the minimum-variance portfolio.
        sub_optimizer = MeanVarianceOptimizer(
            expected_returns=post_mean,
            cov_matrix=post_cov,
            constraints=self.constraints,
            risk_free_rate=self.risk_free_rate,
            risk_aversion=delta / 2.0,
        )
        # Active risk is measured on the prior Σ, as every other method and
        # the audit of this result measure it — not on the posterior Σ + M the
        # sub-solve optimizes against, which made the budget tighter than the
        # one set. The sub-solve's own post-solve pass would audit against
        # Σ + M for the same reason, so it is skipped: this optimizer audits
        # the weights it returns against Σ.
        sub_optimizer._tracking_cov = self.cov_matrix
        result = sub_optimizer.optimize(run_post_solve_diagnostics=False)
        self._diagnostics.update(
            {
                k: v
                for k, v in result.extras.items()
                if k in ("solver", "solver_status", "solve_seconds", "mode")
            }
        )
        # The posterior is what the sub-solve optimized against, so the
        # result's return and risk are reported against it (see
        # ``_mu_vector`` / ``_sigma_matrix``); the prior inputs stay as given.
        self._posterior_mean = post_mean
        self._posterior_cov = post_cov
        return result.weights.reindex(self.assets).fillna(0.0).values

    def _mu_vector(self) -> np.ndarray | None:
        """The posterior mean once solved, so the reported return is the one optimized."""
        if self._posterior_mean is None:
            return super()._mu_vector()
        return self._posterior_mean.reindex(self.assets).fillna(0.0).to_numpy(dtype=float)

    def _sigma_matrix(self) -> np.ndarray | None:
        """The posterior covariance once solved; risk against the prior would understate it."""
        if self._posterior_cov is None:
            return super()._sigma_matrix()
        return (
            self._posterior_cov.reindex(index=self.assets, columns=self.assets)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
