"""Active management: the fundamental law, and what constraints cost you.

The rest of this library optimizes portfolios in absolute space. A manager
with a benchmark lives in a different one, where the quantities that matter
are the *active* weights, the active risk they generate, and whether the
forecasts driving them are any good.

Grinold and Kahn's fundamental law connects those:

    IR ≈ TC · IC · √BR

* **IC** (information coefficient) — the cross-sectional correlation between
  your forecasts and what actually happened. Skill. Realistic values are
  small: 0.05 is a good equity stock-picker, 0.10 is excellent.
* **BR** (breadth) — the number of genuinely independent bets per year. A
  manager who takes one macro view held for a year has BR = 1, however many
  positions the view produces.
* **TC** (transfer coefficient) — how much of the forecast actually reaches
  the portfolio, after long-only, position limits, group budgets and turnover
  have had their say. A number between 0 and 1, and typically between 0.3 and
  0.6 for a constrained long-only book (Clarke, de Silva & Thorley, 2002).

The third term is the one this engine is unusually well placed to compute,
because it already knows exactly which constraints were applied and what the
unconstrained answer would have been. A transfer coefficient of 0.35 says two
thirds of the manager's skill is being absorbed by the mandate — which is a
statement about the *constraints*, not the forecasts, and points at a
different fix than "get better signals".

The law also fixes the level of risk to take. Grinold and Kahn's value-added
objective ``IR·ψ − λ_A·ψ²`` in active risk ``ψ`` is maximized at
``ψ* = IR / (2·λ_A)``, which turns a tracking-error budget into an implied
risk-aversion coefficient and vice versa — see
:func:`risk_aversion_from_information_ratio`.

References:
    Grinold, R. (1989). "The Fundamental Law of Active Management".
    *The Journal of Portfolio Management* 15(3).

    Grinold, R. and Kahn, R. (1999). *Active Portfolio Management*, 2nd ed.
    McGraw-Hill.

    Clarke, R., de Silva, H. and Thorley, S. (2002). "Portfolio Constraints
    and the Fundamental Law of Active Management". *Financial Analysts
    Journal* 58(5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats

#: Ways of measuring how much of a forecast survives into the portfolio.
TRANSFER_METHODS = ("optimal", "risk_adjusted")


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InformationCoefficient:
    """Realized forecasting skill, period by period.

    Attributes:
        mean: Average cross-sectional correlation between forecast and
            outcome. This is the IC of the fundamental law.
        std: Its standard deviation across periods — the *variability* of
            skill, which is what makes a high average IC untrustworthy when
            it comes from a handful of periods.
        t_statistic: ``mean / (std / √n_periods)``. Below ~2 the skill is not
            distinguishable from zero, whatever the average says.
        hit_rate: Fraction of periods with a positive IC.
        n_periods: Periods with enough cross-section to compute an IC.
        n_assets: Median cross-sectional width.
        method: ``"spearman"`` or ``"pearson"``.
        series: The per-period IC, for plotting and for spotting whether the
            skill is concentrated in one regime.
    """

    mean: float
    std: float
    t_statistic: float
    hit_rate: float
    n_periods: int
    n_assets: float
    method: str
    series: pd.Series

    def describe(self) -> str:
        """The IC and its t-statistic, with the skill verdict spelled out.

        Returns:
            A sentence giving the mean IC, the cross-section size, the standard
            deviation, the t-statistic and the hit rate. A ``|t| < 2`` reads as
            "not distinguishable from no skill" rather than as a small number the
            reader has to interpret.
        """
        verdict = (
            "not distinguishable from no skill"
            if abs(self.t_statistic) < 2
            else "statistically distinguishable from zero"
        )
        return (
            f"Mean information coefficient {self.mean:.3f} over "
            f"{self.n_periods} periods (median cross-section "
            f"{self.n_assets:.0f} assets), standard deviation {self.std:.3f}, "
            f"t-statistic {self.t_statistic:.2f} — {verdict}. The forecast had "
            f"the right sign in {self.hit_rate:.0%} of periods."
        )


def information_coefficient(
    forecasts: pd.DataFrame,
    realized: pd.DataFrame,
    method: str = "spearman",
    min_assets: int = 3,
) -> InformationCoefficient:
    """Cross-sectional correlation between forecasts and outcomes.

    Both frames are aligned on their common dates *and* their common assets
    before anything is computed: an IC calculated across a forecast panel and
    a return panel that happen to be the same shape, but are not the same
    universe, describes nothing.

    Args:
        forecasts: Forecast (or score) per asset per period. The forecast for
            period ``t`` must be the one made *before* ``t`` — this function
            cannot check that, and it is the single easiest way to
            manufacture skill that is not there.
        realized: Realized returns over the same periods.
        method: ``"spearman"`` (rank correlation, the usual choice — it is
            robust to the outliers that dominate a Pearson IC) or
            ``"pearson"``.
        min_assets: Skip periods with fewer than this many paired
            observations.

    Raises:
        ValueError: On an unknown method, or when no period has enough
            overlap to compute an IC.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(
            f"Unknown IC method {method!r}. Use 'spearman' or 'pearson'."
        )
    dates = forecasts.index.intersection(realized.index)
    assets = forecasts.columns.intersection(realized.columns)
    if len(dates) == 0 or len(assets) == 0:
        raise ValueError(
            "Forecasts and realized returns share no dates or no assets, so "
            "no information coefficient can be computed."
        )
    f = forecasts.loc[dates, assets]
    r = realized.loc[dates, assets]

    values: dict[pd.Timestamp, float] = {}
    widths: list[int] = []
    for date in dates:
        pair = pd.concat([f.loc[date], r.loc[date]], axis=1).dropna()
        if len(pair) < min_assets or pair.iloc[:, 0].nunique() < 2:
            continue
        if method == "spearman":
            rho, _ = scipy.stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        else:
            rho, _ = scipy.stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        if np.isfinite(rho):
            values[date] = float(rho)
            widths.append(len(pair))

    if not values:
        raise ValueError(
            f"No period had at least {min_assets} paired forecast/return "
            "observations with any variation in the forecast."
        )

    series = pd.Series(values, name=f"ic_{method}").sort_index()
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    t_stat = (
        float(mean / (std / np.sqrt(len(series))))
        if std and np.isfinite(std) and std > 0
        else float("nan")
    )
    return InformationCoefficient(
        mean=mean,
        std=std,
        t_statistic=t_stat,
        hit_rate=float((series > 0).mean()),
        n_periods=len(series),
        n_assets=float(np.median(widths)),
        method=method,
        series=series,
    )


# ---------------------------------------------------------------------------
# Transfer: how much skill reaches the portfolio
# ---------------------------------------------------------------------------


def transfer_coefficient(
    alphas: pd.Series,
    active_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    method: str = "optimal",
) -> float:
    """How much of a forecast survives the mandate, in ``[-1, 1]``.

    Two definitions, both in the literature and both supported:

    ``"optimal"`` (Grinold & Kahn) — the correlation, measured in the risk
    metric, between the active weights actually held and the unconstrained
    optimal ones ``Δw* ∝ Σ⁻¹α``:

        ``TC = Δw'Σ Δw* / √((Δw'Σ Δw)(Δw*'Σ Δw*))``

    This is exact: it accounts for correlations, so two positions that
    express the same bet do not count twice.

    ``"risk_adjusted"`` (Clarke, de Silva & Thorley) — the plain
    cross-sectional correlation between ``α_i/σ_i`` and ``Δw_i·σ_i``. It needs
    no matrix inverse, which makes it usable when the covariance is singular
    (a detoned matrix, say), at the cost of ignoring the off-diagonals.

    Args:
        alphas: Expected *active* returns (benchmark-relative) per asset.
        active_weights: Portfolio weight minus benchmark weight per asset.
        cov_matrix: Covariance matrix over the same universe.
        method: See above.

    Returns:
        The transfer coefficient. ``1.0`` means the mandate cost nothing;
        ``0.0`` means the portfolio expresses none of the forecast.

    Raises:
        ValueError: On an unknown method, an empty overlap, a portfolio that
            *is* the benchmark, or an alpha vector with no cross-sectional
            dispersion. The last two are undefined rather than zero — there is
            nothing to transfer, or nothing to transfer it into — and saying
            so beats returning a NaN that surfaces three calls later.
    """
    if method not in TRANSFER_METHODS:
        raise ValueError(
            f"Unknown transfer-coefficient method {method!r}. "
            f"Available: {list(TRANSFER_METHODS)}"
        )
    assets = [a for a in cov_matrix.columns if a in alphas.index and a in active_weights.index]
    if len(assets) < 2:
        raise ValueError(
            "The alpha vector, the active weights and the covariance matrix "
            "must share at least 2 assets."
        )
    sigma = cov_matrix.loc[assets, assets].values
    alpha = alphas.reindex(assets).fillna(0.0).values.astype(float)
    active = active_weights.reindex(assets).fillna(0.0).values.astype(float)

    if np.allclose(active, 0.0):
        raise ValueError(
            "This portfolio holds the benchmark exactly, so it takes no active "
            "positions and the transfer coefficient is undefined rather than "
            "zero."
        )
    if np.allclose(alpha - alpha.mean(), 0.0):
        raise ValueError(
            "Every asset has the same alpha, so there is no cross-sectional "
            "forecast to transfer. A flat expected-return vector carries no "
            "view; the transfer coefficient is undefined."
        )

    if method == "risk_adjusted":
        std = np.sqrt(np.diag(sigma))
        if not (std > 0).all():
            raise ValueError(
                "The risk-adjusted transfer coefficient divides by each "
                "asset's volatility; at least one is zero."
            )
        return float(np.corrcoef(alpha / std, active * std)[0, 1])

    optimal = np.linalg.pinv(sigma) @ alpha
    numerator = float(active @ sigma @ optimal)
    denominator = float(
        np.sqrt(max(active @ sigma @ active, 0.0) * max(optimal @ sigma @ optimal, 0.0))
    )
    if denominator <= 0:
        return float("nan")
    return float(np.clip(numerator / denominator, -1.0, 1.0))


# ---------------------------------------------------------------------------
# The law itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FundamentalLawReport:
    """``IR ≈ TC · IC · √BR``, with the pieces named.

    Attributes:
        information_coefficient: Skill per bet.
        breadth: Independent bets per year.
        transfer_coefficient: Share of the forecast that reaches the book.
        information_ratio: The implied IR.
        unconstrained_information_ratio: What the same skill and breadth
            would produce with no constraints (``TC = 1``).
        constraint_cost: The IR given up to the mandate.
        expected_active_return: ``IR × active_risk``, when an active-risk
            budget was supplied.
        active_risk: The tracking-error budget the return was computed at.
    """

    information_coefficient: float
    breadth: float
    transfer_coefficient: float
    information_ratio: float
    unconstrained_information_ratio: float
    constraint_cost: float
    expected_active_return: float | None = None
    active_risk: float | None = None

    def describe(self) -> str:
        """The fundamental law, read as a sentence rather than a formula.

        Returns:
            IC and breadth, the unconstrained information ratio they imply, the
            transfer coefficient, and what the mandate's constraints cost in IR.
            When an active-risk budget was supplied, the expected active return
            that budget buys is appended.
        """
        line = (
            f"IC {self.information_coefficient:.3f} across "
            f"{self.breadth:.0f} independent bets a year implies an "
            f"information ratio of {self.unconstrained_information_ratio:.2f} "
            f"unconstrained. A transfer coefficient of "
            f"{self.transfer_coefficient:.2f} — the share of the forecast the "
            f"mandate lets through — cuts that to "
            f"{self.information_ratio:.2f}, giving up "
            f"{self.constraint_cost:.2f} of IR to the constraints."
        )
        if self.expected_active_return is not None:
            line += (
                f" At an active-risk budget of {self.active_risk:.2%}, that is "
                f"{self.expected_active_return:.2%} of expected active return "
                "a year."
            )
        return line


def fundamental_law(
    information_coefficient: float,
    breadth: float,
    transfer_coefficient: float = 1.0,
    active_risk: float | None = None,
) -> FundamentalLawReport:
    """Assemble ``IR = TC · IC · √BR`` and what it implies.

    Args:
        information_coefficient: Skill per bet, e.g. the ``mean`` from
            :func:`information_coefficient`.
        breadth: Independent bets per year. Count *independent* ones: 500
            stocks driven by one sector view is closer to breadth 1 than 500.
        transfer_coefficient: From :func:`transfer_coefficient`. Leave at 1
            to see the unconstrained ceiling.
        active_risk: Optional tracking-error budget, to convert the ratio
            into an expected active return.

    Raises:
        ValueError: If ``breadth < 1`` or the transfer coefficient is outside
            ``[-1, 1]``.
    """
    if breadth < 1:
        raise ValueError(
            f"Breadth is a count of independent bets per year and must be at "
            f"least 1; got {breadth}."
        )
    if not -1.0 <= transfer_coefficient <= 1.0:
        raise ValueError(
            f"The transfer coefficient is a correlation and must lie in "
            f"[-1, 1]; got {transfer_coefficient}."
        )
    unconstrained = float(information_coefficient * np.sqrt(breadth))
    achieved = float(transfer_coefficient * unconstrained)
    return FundamentalLawReport(
        information_coefficient=float(information_coefficient),
        breadth=float(breadth),
        transfer_coefficient=float(transfer_coefficient),
        information_ratio=achieved,
        unconstrained_information_ratio=unconstrained,
        constraint_cost=unconstrained - achieved,
        expected_active_return=(
            achieved * float(active_risk) if active_risk is not None else None
        ),
        active_risk=float(active_risk) if active_risk is not None else None,
    )


def implied_breadth(
    information_ratio: float,
    information_coefficient: float,
    transfer_coefficient: float = 1.0,
) -> float:
    """Independent bets a claimed IR would require, given the skill claimed.

    Run backwards, the law is a plausibility check: an IR of 1.0 from an IC
    of 0.03 needs 1,111 independent bets a year with no constraints at all.
    If the manager holds 40 positions and turns them over quarterly, the
    claim does not survive arithmetic.

    Args:
        information_ratio: The annualized information ratio being claimed.
        information_coefficient: Skill, as a correlation between forecast and
            realized return. Typically 0.02-0.10 for a real process.
        transfer_coefficient: The share of the forecast the mandate actually
            lets through, in ``[0, 1]``. ``1.0`` means unconstrained.

    Returns:
        Independent bets required per year, on the same annual basis as the
        information ratio.

    Raises:
        ValueError: If ``IC × TC`` is zero, where any IR would need infinite
            breadth.
    """
    denominator = information_coefficient * transfer_coefficient
    if abs(denominator) < 1e-12:
        raise ValueError(
            "With zero skill (or zero transfer) no amount of breadth produces "
            "a positive information ratio."
        )
    return float((information_ratio / denominator) ** 2)


def grinold_alpha(
    scores: pd.Series, volatility: pd.Series, information_coefficient: float
) -> pd.Series:
    """Turn scores into expected active returns: ``α = IC · σ · z``.

    Grinold's refinement, and the most useful single formula in *Active
    Portfolio Management*. A raw score — a rank, a z-score, an analyst's
    1-to-5 — is not an expected return and cannot be handed to an optimizer.
    This scales it into one, using the only two things that set the size of a
    defensible forecast: how much skill you have (``IC``), and how much the
    asset moves (``σ``).

    The discipline it imposes is the point. With an IC of 0.05, a two-standard
    -deviation score on a 20%-volatility asset earns an alpha of 2% — not the
    10% that gets typed into a spreadsheet, and the difference is exactly what
    stops mean-variance producing corner solutions.

    Args:
        scores: Standardized forecasts, one per asset. Cross-sectionally
            z-scored is the intended input; anything with a non-zero mean
            becomes an unintended market-direction bet.
        volatility: Annualized volatility per asset — residual (benchmark-
            relative) volatility if the alphas are to be active returns.
        information_coefficient: Realized or assumed skill.

    Returns:
        Expected active return per asset, on the same annual scale as
        ``volatility``.

    Raises:
        ValueError: If ``scores`` and ``volatility`` share no assets.
    """
    assets = scores.index.intersection(volatility.index)
    if len(assets) == 0:
        raise ValueError("Scores and volatilities share no assets.")
    alpha = (
        float(information_coefficient)
        * volatility.reindex(assets).astype(float)
        * scores.reindex(assets).astype(float)
    )
    return alpha.rename("grinold_alpha")


def risk_aversion_from_information_ratio(
    information_ratio: float, target_active_risk: float
) -> float:
    """Active risk aversion implied by an information ratio and a TE budget.

    Grinold and Kahn's value-added objective in active space is
    ``VA(ψ) = IR·ψ − λ_A·ψ²``, maximized at ``ψ* = IR / (2·λ_A)``. Read
    backwards, a manager who wants ``ψ*`` of tracking error and believes in an
    ``IR`` is implicitly using ``λ_A = IR / (2·ψ*)``.

    That is the number to put in a mean-variance utility, and deriving it
    beats guessing: "risk aversion = 2" means nothing on its own, while "a 4%
    tracking-error budget at an IR of 0.5" is a sentence a committee can
    argue with.

    Args:
        information_ratio: The annualized information ratio believed in.
        target_active_risk: The tracking-error budget, as an annualized
            fraction — ``0.04`` for 4%.

    Returns:
        The active risk-aversion coefficient ``λ_A``.

    Raises:
        ValueError: If ``target_active_risk`` is not positive.
    """
    if target_active_risk <= 0:
        raise ValueError(
            f"The active-risk budget must be positive; got {target_active_risk}."
        )
    return float(information_ratio / (2.0 * target_active_risk))


def optimal_active_risk(
    information_ratio: float, risk_aversion: float
) -> float:
    """The tracking error that maximizes value added: ``ψ* = IR / (2·λ_A)``.

    Args:
        information_ratio: The annualized information ratio believed in.
        risk_aversion: The active risk-aversion coefficient ``λ_A``.

    Returns:
        The optimal tracking error, as an annualized fraction.

    Raises:
        ValueError: If ``risk_aversion`` is not positive — with zero or
            negative active risk aversion the objective is unbounded.
    """
    if risk_aversion <= 0:
        raise ValueError(
            f"Active risk aversion must be positive; got {risk_aversion}. At "
            "or below zero the value-added objective has no maximum."
        )
    return float(information_ratio / (2.0 * risk_aversion))


def value_added(
    information_ratio: float, active_risk: float, risk_aversion: float
) -> float:
    """``IR·ψ − λ_A·ψ²`` — risk-adjusted value added at a given active risk.

    Args:
        information_ratio: The annualized information ratio.
        active_risk: The tracking error actually run, as an annualized
            fraction.
        risk_aversion: The active risk-aversion coefficient ``λ_A``.

    Returns:
        Value added, in the same annualized return units as the inputs. It is
        negative when the tracking error run is more than twice the optimum.
    """
    return float(
        information_ratio * active_risk - risk_aversion * active_risk**2
    )


# ---------------------------------------------------------------------------
# Where the active risk actually sits
# ---------------------------------------------------------------------------


def active_risk_decomposition(
    weights: pd.Series,
    benchmark_weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Euler decomposition of *tracking error* into per-asset contributions.

    The absolute-space version of this decomposition lives in
    :mod:`optimization_engine.optimizers.diagnostics`. This is its
    benchmark-relative twin, and it answers a different question: not "where
    is my risk" but "where is my risk *different from the benchmark's*".

    The two disagree in exactly the place it matters. A 40% position in the
    largest index constituent can be the single biggest source of absolute
    risk and contribute nothing at all to tracking error, because the
    benchmark holds it too. Managing one number while reporting the other is
    a standard way to be surprised.

    Args:
        weights: Portfolio weights, as fractions of the book. Assets it does
            not list count as zero — a name only the benchmark holds is an
            active underweight, and it is decomposed like any other.
        benchmark_weights: The benchmark's weights over the same universe.
            Assets it does not hold count as zero.
        cov_matrix: Asset covariance, on whatever periodicity the reported
            tracking error should carry.

    Returns:
        A frame indexed by asset with:

        ``weight`` / ``benchmark_weight`` / ``active_weight``
        ``marginal_tracking_error`` — ``∂TE/∂w_i``.
        ``contribution`` — ``Δw_i · ∂TE/∂w_i``, summing exactly to the
        tracking error.
        ``share_of_tracking_error`` — that contribution as a fraction.

    Raises:
        ValueError: If the inputs share no assets with the covariance matrix.
    """
    # Either side's names, not just the book's: a book listed by its holdings
    # used to drop every benchmark-only name, and with it that name's active
    # underweight — 4.71% tracking error reported against a true 8.16%.
    assets = [
        a for a in cov_matrix.columns
        if a in weights.index or a in benchmark_weights.index
    ]
    if not assets:
        raise ValueError(
            "Neither the weights nor the benchmark share an asset with the "
            "covariance matrix."
        )
    sigma = cov_matrix.loc[assets, assets].values
    w = weights.reindex(assets).fillna(0.0).values.astype(float)
    b = benchmark_weights.reindex(assets).fillna(0.0).values.astype(float)
    active = w - b

    tracking_error = float(np.sqrt(max(active @ sigma @ active, 0.0)))
    if tracking_error <= 0:
        zeros = np.zeros(len(assets))
        return pd.DataFrame(
            {
                "weight": w,
                "benchmark_weight": b,
                "active_weight": active,
                "marginal_tracking_error": zeros,
                "contribution": zeros,
                "share_of_tracking_error": zeros,
            },
            index=assets,
        )

    marginal = (sigma @ active) / tracking_error
    contribution = active * marginal
    return pd.DataFrame(
        {
            "weight": w,
            "benchmark_weight": b,
            "active_weight": active,
            "marginal_tracking_error": marginal,
            "contribution": contribution,
            "share_of_tracking_error": contribution / tracking_error,
        },
        index=assets,
    )
