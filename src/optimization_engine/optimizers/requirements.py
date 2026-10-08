"""Per-method input/support metadata for the optimizer registry.

Single source of truth shared by the engine (validation) and the Streamlit
UI (which fields to enable). Adding a new optimizer means: register it in
``factory._REGISTRY`` AND add a ``MethodRequirements`` entry here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ExtraKind = Literal["per_asset", "scalar", "choice", "view_table", "market_caps"]
BoundsMode = Literal["hard", "soft_iterated", "constrained", "hard_or_projected"]


@dataclass(frozen=True)
class ExtraInput:
    """One extra input a method needs beyond returns and a covariance matrix.

    Declarative so that the CLI, the MCP server and the Streamlit app can all
    render the same prompt without any of them hard-coding a method's
    particulars.

    Attributes:
        key: The config field this fills.
        label: Human-readable name, shown next to the input.
        kind: What sort of value it is, which decides the widget and the
            parsing.
        required: Whether the solve refuses without it.
        help: One line explaining what the value means.
        default: Pre-filled value, or ``None`` for no default.
        choices: The allowed values when the input is a closed set.
    """
    key: str
    label: str
    kind: ExtraKind
    required: bool
    help: str
    default: Any | None = None
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MethodRequirements:
    """What one optimizer needs, supports, and assumes.

    Beyond the machine-readable capability flags, each entry carries the
    prose an analyst needs to choose between methods: a one-line summary,
    the situation it suits, and the assumptions it is buying into. The UI
    renders these next to the method picker so the choice is informed rather
    than alphabetical.
    """

    name: str
    requires_mu: bool
    requires_cov: bool
    requires_returns: bool
    supports_target_return: bool
    supports_target_volatility: bool
    supports_risk_aversion: bool
    supports_risk_free_rate: bool
    supports_group_bounds: bool
    bounds_mode: BoundsMode
    supports_frontier: bool
    extras: tuple[ExtraInput, ...] = field(default_factory=tuple)
    supports_turnover: bool = False
    #: Whether a tracking-error or active-share budget is imposed *inside*
    #: the solve. False means the limit is reported but not enforced — the
    #: projection-based methods can neither impose nor verify it.
    supports_benchmark_limits: bool = False
    #: Whether the method is undefined without a benchmark.
    requires_benchmark: bool = False
    label: str = ""
    summary: str = ""
    when_to_use: str = ""
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    risk_measure: str = "volatility"

    @property
    def display_name(self) -> str:
        """The method's name as a person should read it.

        Returns:
            The declared label when there is one, and otherwise the registered
            name title-cased with its underscores removed.
        """
        return self.label or self.name.replace("_", " ").title()

    @property
    def bounds_note(self) -> str:
        """Plain-English description of how faithfully bounds are honoured."""
        return {
            "hard": "Weight bounds and group budgets are enforced inside the solve.",
            "constrained": (
                "Bounds are enforced inside the solve; tiny numerical drift is "
                "clipped afterwards."
            ),
            "soft_iterated": (
                "This method allocates first and applies the constraints "
                "afterwards, by projecting onto the closest feasible "
                "allocation. Bounds and group budgets do hold, but a binding "
                "one moves the result away from the method's own answer — the "
                "distance moved is reported with the result."
            ),
            "hard_or_projected": (
                "Bounds and group budgets are enforced inside the solve, "
                "which is where this method normally answers. If that solve "
                "fails numerically it falls back to an unconstrained solve "
                "plus a projection, and then the bounds hold only "
                "approximately — the result reports which of the two "
                "happened, and how far the projection moved it."
            ),
        }[self.bounds_mode]


_RISK_BUDGET = ExtraInput(
    key="risk_budget", label="Risk budget",
    kind="per_asset", required=False,
    help="Each asset's target share of total variance (sums to 1).",
)
_BL_VIEWS = ExtraInput(
    key="bl_views", label="Black-Litterman views",
    kind="view_table", required=False,
    help="Asset → annualized expected return.",
)
_BL_VIEW_CONFIDENCES = ExtraInput(
    key="bl_view_confidences", label="View confidences (Ω diagonal)",
    kind="view_table", required=False,
    help="Variance of each view's error term. Defaults to tau · σ_i².",
)
_BL_TAU = ExtraInput(
    key="bl_tau", label="Tau (prior uncertainty scale)",
    kind="scalar", required=False, default=0.05,
    help="Scales the prior covariance in the BL posterior.",
)
_BL_MARKET_CAPS = ExtraInput(
    key="bl_market_caps", label="Market caps / weights",
    kind="market_caps", required=False,
    help="Equilibrium market portfolio. Empty → equal weights.",
)
_CVAR_ALPHA = ExtraInput(
    key="cvar_alpha", label="CVaR tail probability α",
    kind="scalar", required=False, default=0.05,
    help="0.05 ⇒ 95% CVaR.",
)
_HRP_LINKAGE = ExtraInput(
    key="hrp_linkage", label="HRP linkage method",
    kind="choice", required=False, default="single",
    choices=("single", "average", "complete", "ward"),
    help="Hierarchical clustering linkage rule.",
)
_CLUSTER_LINKAGE = ExtraInput(
    key="cluster_linkage", label="Linkage method",
    kind="choice", required=False, default="ward",
    choices=("single", "average", "complete", "ward"),
    help=(
        "Linkage rule for the dendrogram. Ward, not single: these methods "
        "partition the tree rather than merely order it, and single linkage "
        "chains into one dominant cluster."
    ),
)
_N_CLUSTERS = ExtraInput(
    key="n_clusters", label="Number of clusters",
    kind="scalar", required=False, default=None,
    help=(
        "Force a cluster count. Empty selects it by maximizing the silhouette "
        "t-statistic — the ONC criterion."
    ),
)
_MAX_CLUSTERS = ExtraInput(
    key="max_clusters", label="Maximum clusters to consider",
    kind="scalar", required=False, default=None,
    help="Upper bound of the cluster-count search. Defaults to min(10, N-1).",
)
_HERC_RISK_MEASURE = ExtraInput(
    key="herc_risk_measure", label="Cluster risk measure",
    kind="choice", required=False, default="variance",
    choices=("variance", "std", "cvar", "cdar", "equal_weight"),
    help=(
        "How a cluster's risk is measured when splitting the budget between "
        "two branches. CVaR and CDaR need a return history."
    ),
)
_NCO_OBJECTIVE = ExtraInput(
    key="nco_objective", label="Objective at both layers",
    kind="choice", required=False, default="min_variance",
    choices=("min_variance", "max_sharpe"),
    help="Solved inside each cluster and again across clusters.",
)
_NCO_DETONE = ExtraInput(
    key="nco_detone_for_clustering", label="Detone before clustering",
    kind="scalar", required=False, default=True,
    help=(
        "Remove the market eigenvector before measuring correlation distance. "
        "Leaving it in makes every pair look alike and the partition degenerate."
    ),
)
_CDAR_ALPHA = ExtraInput(
    key="cdar_alpha", label="CDaR tail probability α",
    kind="scalar", required=False, default=0.05,
    help="0.05 averages the worst 5% of the drawdown path.",
)


REQUIREMENTS: dict[str, MethodRequirements] = {
    "mean_variance": MethodRequirements(
        name="mean_variance",
        supports_benchmark_limits=True,
        label="Mean-variance (Markowitz)",
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, supports_turnover=True, extras=(),
        summary=(
            "Trade expected return against variance, subject to every "
            "constraint you set."
        ),
        when_to_use=(
            "You have expected returns you are willing to defend, and you want "
            "an explicit return or volatility target."
        ),
        assumptions=(
            "Investors care only about the mean and variance of returns.",
            "The expected-return vector is accurate — results are famously "
            "sensitive to it, so small estimation errors produce very "
            "different portfolios.",
            "The covariance matrix is stable over the holding period.",
        ),
    ),
    "min_variance": MethodRequirements(
        name="min_variance",
        supports_benchmark_limits=True,
        label="Global minimum variance",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, supports_turnover=True, extras=(),
        summary="The lowest-variance portfolio the constraints allow.",
        when_to_use=(
            "You do not trust any expected-return estimate. This is the one "
            "mean-variance portfolio that does not need one."
        ),
        assumptions=(
            "Only risk matters; expected returns are ignored entirely.",
            "The covariance matrix is well estimated — with T/N below ~10 even "
            "this portfolio is unstable.",
        ),
    ),
    "max_sharpe": MethodRequirements(
        name="max_sharpe",
        supports_benchmark_limits=True,
        label="Maximum Sharpe (tangency)",
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, supports_turnover=False, extras=(),
        summary="The portfolio with the highest excess return per unit of risk.",
        when_to_use=(
            "You want the tangency portfolio, and you will size the risk "
            "separately by blending it with cash."
        ),
        assumptions=(
            "Expected returns are accurate — the tangency portfolio is the "
            "most estimation-sensitive point on the whole frontier.",
            "You can borrow and lend at the risk-free rate.",
            "Some allocation the mandate allows earns more than the "
            "risk-free rate — for a long-only book, at least one asset must.",
            "A turnover budget cannot be imposed on this solve.",
        ),
    ),
    "risk_parity": MethodRequirements(
        name="risk_parity",
        label="Risk parity / risk budgeting",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="constrained",
        supports_frontier=False, supports_turnover=False,
        extras=(_RISK_BUDGET,),
        summary=(
            "Size positions so each asset contributes its target share of "
            "total risk."
        ),
        when_to_use=(
            "You want risk, not capital, spread evenly — the standard answer "
            "when a 60/40 book's equity sleeve dominates its risk."
        ),
        assumptions=(
            "Expected returns are ignored; only the covariance matters.",
            "Equalizing risk contributions is a good proxy for a good "
            "portfolio — true when Sharpe ratios are similar across assets.",
            "Weights must be long-only for risk contributions to be well "
            "defined.",
        ),
    ),
    "hrp": MethodRequirements(
        name="hrp",
        label="Hierarchical Risk Parity",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="soft_iterated",
        supports_frontier=False, supports_turnover=False,
        extras=(_HRP_LINKAGE,),
        summary=(
            "Cluster assets by correlation, then split risk down the tree — "
            "no matrix inversion anywhere."
        ),
        when_to_use=(
            "Many assets relative to your history, or a covariance matrix that "
            "is close to singular. This is the robust choice when T/N is small."
        ),
        assumptions=(
            "The correlation hierarchy it finds is economically meaningful — "
            "check the reported clusters.",
            "Group budgets cannot be enforced; the method has its own "
            "hierarchy and will disagree with a hand-specified one.",
            "Weight bounds are applied by projection after allocation.",
        ),
    ),
    "herc": MethodRequirements(
        name="herc",
        label="Hierarchical Equal Risk Contribution",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="soft_iterated",
        supports_frontier=False, supports_turnover=False,
        extras=(_CLUSTER_LINKAGE, _N_CLUSTERS, _MAX_CLUSTERS, _HERC_RISK_MEASURE),
        summary=(
            "Split the budget at the dendrogram's own branch points, equalizing "
            "risk between siblings, and stop at the number of clusters the data "
            "supports."
        ),
        when_to_use=(
            "You want HRP's robustness but the clusters to be respected: HRP "
            "bisects a sorted list and can cut through a genuine group, HERC "
            "splits where the tree actually branches. Also the way in when "
            "drawdown or tail risk, not variance, is the measure that matters."
        ),
        assumptions=(
            "The correlation hierarchy is economically meaningful — check the "
            "reported clusters and the silhouette score.",
            "The chosen number of clusters is a real feature of the data, not "
            "of the linkage rule. A silhouette near zero means it is not.",
            "Group budgets cannot be enforced inside the allocation; bounds are "
            "applied by projection afterwards.",
        ),
    ),
    "nco": MethodRequirements(
        name="nco",
        label="Nested Clustered Optimization",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="soft_iterated",
        supports_frontier=False, supports_turnover=False,
        extras=(_NCO_OBJECTIVE, _CLUSTER_LINKAGE, _N_CLUSTERS, _MAX_CLUSTERS, _NCO_DETONE),
        summary=(
            "Optimize inside each correlation cluster, then across the "
            "clusters, so no single ill-conditioned matrix is ever inverted."
        ),
        when_to_use=(
            "Markowitz gives you corner solutions and you can see why: a "
            "condition number in the thousands, or clusters of near-identical "
            "assets. NCO keeps the mean-variance objective and removes the "
            "instability that comes from the block structure."
        ),
        assumptions=(
            "The universe genuinely clusters. Without block structure the "
            "two-layer solve adds the clustering's noise and little else.",
            "The same objective is appropriate at both layers.",
            "Per-asset and group limits are applied to the combined result by "
            "projection, not inside either solve.",
        ),
    ),
    "black_litterman": MethodRequirements(
        name="black_litterman",
        # Its sub-solve is mean-variance, which puts the budget in the convex
        # program — measured on the prior Σ, like the audit.
        supports_benchmark_limits=True,
        label="Black-Litterman",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, supports_turnover=True,
        extras=(_BL_VIEWS, _BL_VIEW_CONFIDENCES, _BL_TAU, _BL_MARKET_CAPS),
        summary=(
            "Start from the returns the market already implies, then tilt only "
            "where you hold a view."
        ),
        when_to_use=(
            "You have a few specific opinions and want the rest of the "
            "portfolio to stay at equilibrium instead of reacting to noisy "
            "historical means."
        ),
        assumptions=(
            "The market-cap portfolio you supply is in equilibrium.",
            "Views are expressed with an honest confidence — an over-confident "
            "view will dominate the posterior.",
            "The risk-aversion coefficient δ sets the level of the prior; "
            "calibrate it to an observed market Sharpe rather than guessing.",
        ),
    ),
    "cvar": MethodRequirements(
        name="cvar",
        supports_benchmark_limits=True,
        label="Mean-CVaR (expected shortfall)",
        requires_mu=False, requires_cov=False, requires_returns=True,
        supports_target_return=True, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, supports_turnover=True,
        extras=(_CVAR_ALPHA,), risk_measure="CVaR",
        summary=(
            "Minimize the average loss in the worst α of scenarios, straight "
            "from the return history."
        ),
        when_to_use=(
            "Returns are visibly skewed or fat-tailed and variance is the "
            "wrong risk measure — credit, options overlays, EM debt."
        ),
        assumptions=(
            "The historical scenarios represent the tail you actually face.",
            "Enough observations fall in the tail to estimate it: at α = 5% "
            "you are averaging roughly T/20 scenarios.",
            "Scenarios are equally likely and drawn from one regime.",
        ),
    ),
    "cdar": MethodRequirements(
        name="cdar",
        supports_benchmark_limits=True,
        label="Mean-CDaR (conditional drawdown)",
        requires_mu=False, requires_cov=False, requires_returns=True,
        supports_target_return=True, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, supports_turnover=True,
        extras=(_CDAR_ALPHA,), risk_measure="CDaR",
        summary=(
            "Minimize the average of the worst α of drawdowns along the "
            "realized path."
        ),
        when_to_use=(
            "The mandate is written in drawdown terms — a stop-loss, a "
            "high-water mark, a client who redeems at −20%. Variance and CVaR "
            "are both order-independent and cannot see how long the book "
            "stayed underwater."
        ),
        assumptions=(
            "The single realized return path is representative. This is the "
            "strongest assumption in the library: reorder the same returns and "
            "the objective changes.",
            "The equity curve is accumulated uncompounded, which is what keeps "
            "the problem linear.",
            "Enough distinct underwater episodes occurred to estimate a tail of "
            "drawdowns rather than describe one crisis.",
        ),
    ),
    "max_diversification": MethodRequirements(
        name="max_diversification",
        supports_benchmark_limits=True,
        label="Maximum diversification",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="hard_or_projected",
        supports_frontier=False, supports_turnover=False, extras=(),
        summary=(
            "Maximize weighted-average asset volatility divided by portfolio "
            "volatility."
        ),
        when_to_use=(
            "You want correlation benefit to be the objective rather than a "
            "by-product, without taking a view on returns."
        ),
        assumptions=(
            "Expected returns are proportional to volatility — the condition "
            "under which max-diversification is also mean-variance optimal.",
            "The correlation structure is stable; the objective is driven "
            "entirely by it.",
            "A turnover budget cannot be imposed on this solve.",
        ),
    ),
    "equal_weight": MethodRequirements(
        name="equal_weight",
        label="Equal weight (1/N)",
        requires_mu=False, requires_cov=False, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="soft_iterated",
        supports_frontier=False, supports_turnover=False, extras=(),
        summary="Allocate 1/N to every asset.",
        when_to_use=(
            "As the benchmark every other method has to beat. It estimates "
            "nothing, so it has no estimation error to be wrong about."
        ),
        assumptions=(
            "The universe is deliberately constructed — 1/N inherits whatever "
            "concentration is in the asset list itself.",
            "Bounds and group budgets are applied by projection afterwards, so "
            "a binding one makes the result something other than 1/N.",
        ),
    ),
    "active_mean_variance": MethodRequirements(
        name="active_mean_variance",
        supports_benchmark_limits=True,
        requires_benchmark=True,
        label="Active mean-variance (vs benchmark)",
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, supports_turnover=True, extras=(),
        summary=(
            "Trade expected return *in excess of the benchmark* against "
            "tracking error, instead of total return against total risk."
        ),
        when_to_use=(
            "The mandate is written relative to an index: a target excess "
            "return, an active-risk budget, or both. Set a tracking-error "
            "limit and this spends exactly that budget."
        ),
        assumptions=(
            "The benchmark is the right one — every number this produces is "
            "measured against it, so a badly chosen index makes a good "
            "portfolio look wrong and vice versa.",
            "Expected returns are accurate in *relative* terms; only the "
            "spread between assets and the benchmark drives the answer.",
            "The covariance matrix describes active risk as well as total "
            "risk, which is a stronger claim: tracking error depends on the "
            "off-diagonal terms far more than volatility does.",
        ),
    ),
    "inverse_vol": MethodRequirements(
        name="inverse_vol",
        label="Inverse volatility",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="soft_iterated",
        supports_frontier=False, supports_turnover=False, extras=(),
        summary="Weight each asset by the inverse of its own volatility.",
        when_to_use=(
            "A cheap risk-parity approximation when you do not trust the "
            "off-diagonal terms of the covariance matrix."
        ),
        assumptions=(
            "Correlations are ignored entirely — this equals true risk parity "
            "only when every pair is equally correlated.",
            "Bounds and group budgets are applied by projection afterwards.",
        ),
    ),
}


def requirements_for(name: str) -> MethodRequirements:
    """Return the :class:`MethodRequirements` for an optimizer name.

    Args:
        name: A registered optimizer name.

    Returns:
        What that method needs, supports and assumes. This is what
        ``optengine describe`` prints and what the UI renders next to the
        method picker.

    Raises:
        KeyError: If ``name`` is unknown. The message lists the known names.
    """
    try:
        return REQUIREMENTS[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown optimizer '{name}'. Available: {sorted(REQUIREMENTS.keys())}"
        ) from e
