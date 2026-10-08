"""High-level engine façade.

Glues together loading, covariance estimation, feasibility analysis and
optimizer dispatch so that callers can run the whole pipeline with a single
call — and get back not just an allocation but the evidence needed to decide
whether to trust it.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from optimization_engine.analytics.backtest import (
    BacktestResult,
    RebalanceFrequency,
    WalkForwardResult,
    backtest_weights,
    compare_in_and_out_of_sample,
    walk_forward_backtest,
)
from optimization_engine.analytics.performance import summary_stats
from optimization_engine.analytics.relative import summary_relative
from optimization_engine.analytics.report import PerformanceReport, performance_report
from optimization_engine.analytics.risk import (
    group_risk_contribution,
    risk_contribution,
)
from optimization_engine.backtest.results import RunResult
from optimization_engine.backtest.runner import run_backtest
from optimization_engine.backtest.spec import BacktestSpec
from optimization_engine.backtest.sweep import SweepResults, SweepSpec, run_sweep
from optimization_engine.backtest.tearsheet import Tearsheet, build_tearsheet
from optimization_engine.backtest.walkforward import WalkForwardRun, walk_forward_run
from optimization_engine.benchmark import (
    ResolvedBenchmark,
    resolve_benchmark,
)
from optimization_engine.config import EngineConfig
from optimization_engine.constraints import effective_layers, layer_exposures
from optimization_engine.data.covariance import (
    CovarianceDiagnostics,
    covariance_diagnostics,
    covariance_from_config,
)
from optimization_engine.frontier import FrontierResult, efficient_frontier
from optimization_engine.optimizers import ConfigurationError
from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
from optimization_engine.optimizers.base import OptimizationResult
from optimization_engine.optimizers.diagnostics import (
    PortfolioDiagnostics,
    risk_decomposition,
)
from optimization_engine.optimizers.factory import (
    constraints_from_config,
    effective_expected_returns,
    optimizer_factory,
    validate_benchmark_constraints,
)
from optimization_engine.optimizers.feasibility import (
    FeasibilityReport,
    InfeasibleConstraintsError,
    analyze_feasibility,
)
from optimization_engine.stress import (
    Shock,
    StressReport,
    shocks_from_dicts,
    stress_test,
)
from optimization_engine.universe import Eligibility

#: Pre-solve findings that ``strict_mandate`` turns from a warning into a
#: refusal: limits written against assets the panel does not hold.
_STRAY_MANDATE_CODES = frozenset(
    {"bounds_outside_universe", "layer_assets_outside_universe"}
)


def apply_fx_conversion(
    prices: pd.DataFrame,
    config: EngineConfig,
    fx_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert ``prices`` into ``config.base_currency`` if needed.

    No-op when every asset is already quoted in the base currency, or
    when ``config.currencies`` is empty.

    Args:
        prices: Price panel, dates down the index and one column per asset.
        config: Supplies ``currencies`` (``asset -> ISO code``) and
            ``base_currency``.
        fx_rates: Pre-fetched X→base rates. When ``None`` and a conversion is
            needed, they are fetched from FRED over the panel's range.

    Returns:
        The panel valued in the base currency, or ``prices`` unchanged when
        nothing needed converting.

    Raises:
        FXError: If a rate for one of the panel's currencies is unavailable, or
            the price index is not a ``DatetimeIndex``.
    """
    if not config.currencies:
        return prices
    base = config.base_currency.upper()
    needed = {config.currencies.get(a, base).upper() for a in prices.columns}
    if needed == {base}:
        return prices

    # Local import: keeps this module importable when urllib is restricted.
    from optimization_engine.data.fx import convert_prices_to_base

    return convert_prices_to_base(
        prices,
        asset_currency=config.currencies,
        base=base,
        fx_rates=fx_rates,
    )


@dataclass
class EngineRun:
    """Everything one optimization produced, plus the evidence behind it."""

    config: EngineConfig
    returns: pd.DataFrame
    cov_matrix: pd.DataFrame
    expected_returns: pd.Series
    result: OptimizationResult
    frontier: FrontierResult | None = None
    feasibility: FeasibilityReport | None = None
    covariance_diagnostics: CovarianceDiagnostics | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    #: The benchmark this run was measured — and possibly optimized — against,
    #: resolved once at solve time so every downstream view uses the same one.
    benchmark: ResolvedBenchmark | None = None
    #: What the configured shocks do to this allocation, when ``run_engine``
    #: was asked for them (``run_stress=True``) and the config named any.
    #: ``None`` means the question was not asked — never that nothing hurts.
    stress: StressReport | None = None

    # -- allocation views ---------------------------------------------------

    @property
    def diagnostics(self) -> PortfolioDiagnostics | None:
        """Concentration, exposure and compliance summary of the allocation."""
        return self.result.extras.get("diagnostics")

    def risk_contributions(self) -> pd.Series:
        """Per-asset share of total portfolio risk."""
        return risk_contribution(self.result.weights, self.cov_matrix)

    def risk_decomposition(self) -> pd.DataFrame:
        """Euler decomposition of portfolio volatility, in volatility units."""
        return risk_decomposition(self.result.weights, self.cov_matrix)

    def group_risk_contributions(self) -> pd.Series:
        """Risk shares aggregated to asset-class level."""
        return group_risk_contribution(
            self.result.weights, self.cov_matrix, self.config.groups
        )

    @property
    def constraint_layers(self):
        """Every layer of the allocation policy this run was solved under."""
        return effective_layers(self.config)

    def layer_exposures(self) -> pd.DataFrame:
        """Realized exposure of every bucket, on every layer, against its limit.

        The table an allocator reads to see *which* part of the policy is
        binding: a book that stops at 60% equity because the asset-class cap
        says so is a different portfolio from one that stops there because the
        EM sub-limit ran out, and only the headroom column tells them apart.
        """
        return layer_exposures(self.result.weights, self.constraint_layers)

    def layer_risk_contributions(self, layer: str | None = None) -> pd.Series:
        """Share of portfolio risk carried by each bucket of one layer.

        Weight limits are set on capital, but the thing they are trying to
        control is risk, and the two diverge sharply — a 30% fixed-income
        sleeve rarely carries 30% of the risk. Defaults to the first layer.

        Args:
            layer: Which layer to decompose, by name. ``None`` takes the first.

        Returns:
            One risk share per bucket, summing to 1.

        Raises:
            ValueError: When the run has no layers, or none by that name.
        """
        layers = self.constraint_layers
        if not layers:
            raise ValueError(
                "This run has no allocation layers, so there are no buckets "
                "to attribute risk to."
            )
        chosen = next(
            (lyr for lyr in layers if layer is None or lyr.name == layer), None
        )
        if chosen is None:
            names = ", ".join(lyr.name for lyr in layers)
            raise ValueError(f"No layer named {layer!r}. Available: {names}.")
        return group_risk_contribution(
            self.result.weights, self.cov_matrix, chosen.assignments
        )

    def diversification(self, model: str = "minimum_torsion"):
        """Effective number of bets on uncorrelated factors (Meucci).

        Complements :attr:`diagnostics`, whose ``effective_n`` and
        ``effective_n_risk`` are computed asset by asset and so cannot see
        that several positions are the same bet. Not computed on every solve
        because the minimum-torsion rotation is an iterative fixed point and
        the engine should not pay for it unless asked.

        Args:
            model: The rotation used to build uncorrelated factors —
                ``"minimum_torsion"`` (the default, Meucci's) or ``"pca"``.

        Returns:
            A :class:`~optimization_engine.analytics.diversification.DiversificationReport`
            with the effective number of bets, the variance share of each factor,
            and the largest single bet.
        """
        from optimization_engine.analytics.diversification import (
            diversification_distribution,
        )

        return diversification_distribution(
            self.result.weights, self.cov_matrix, model=model
        )

    def diversification_comparison(self) -> pd.DataFrame:
        """Effective bets under both rotations; the gap is the diagnostic."""
        from optimization_engine.analytics.diversification import (
            compare_diversification,
        )

        return compare_diversification(self.result.weights, self.cov_matrix)

    # -- benchmark-relative -------------------------------------------------

    @property
    def benchmark_returns(self) -> pd.Series | None:
        """The benchmark's return stream, or None when none was chosen."""
        return None if self.benchmark is None else self.benchmark.returns

    @property
    def benchmark_label(self) -> str | None:
        """What the benchmark is called, or ``None`` when none was chosen."""
        return None if self.benchmark is None else self.benchmark.label

    def _benchmark_weights(self) -> pd.Series:
        """The benchmark's weights over this run's universe.

        Raises:
            ValueError: When no benchmark is set, or when the one that is set
                has no positions in the investable universe. Active analytics
                are meaningless without a weight vector, and defaulting to
                equal weights would invent a benchmark nobody chose.
        """
        assets = list(self.result.weights.index)
        weights = self.config.benchmark_weight_map(assets)
        if weights is None and self.benchmark is not None:
            resolved = self.benchmark.weights
            weights = None if resolved is None else resolved.to_dict()
        if not weights:
            raise ValueError(
                "This run has no position-based benchmark, so there are no "
                "active positions to analyze. Choose a benchmark defined by "
                "weights (1/N, a single asset, or a custom vector) — an "
                "external index has no holdings in this universe."
            )
        return pd.Series(weights).reindex(assets).fillna(0.0)

    def active_risk_decomposition(self) -> pd.DataFrame:
        """Euler decomposition of *tracking error*, per asset.

        Where :meth:`risk_decomposition` says where the risk is, this says
        where the risk differs from the benchmark's — the two disagree
        precisely on the large index positions that carry absolute risk and
        no active risk at all.
        """
        from optimization_engine.analytics.active import active_risk_decomposition

        return active_risk_decomposition(
            self.result.weights, self._benchmark_weights(), self.cov_matrix
        )

    def transfer_coefficient(self, method: str = "optimal") -> float:
        """How much of this run's expected returns survived the mandate.

        The alphas are taken as the expected returns *in excess of the
        benchmark's*, and the active weights as the solved book minus the
        benchmark. A low number says the constraints, not the forecasts, are
        determining the portfolio — which is a fixable problem, and a
        different one from having poor forecasts.

        Args:
            method: Which definition to use. ``"optimal"`` (Grinold & Kahn) is
                the risk-metric correlation between the active weights held
                and the unconstrained optimal ones, so correlated positions
                expressing the same bet are not counted twice.
                ``"risk_adjusted"`` (Clarke, de Silva & Thorley) is the plain
                cross-sectional correlation of ``α_i/σ_i`` against
                ``Δw_i·σ_i``.

        Returns:
            The transfer coefficient, in ``[-1, 1]``. Around 0.3-0.5 is normal for
            a long-only mandate; below 0.2 means the constraints are writing the
            portfolio.

        Raises:
            ValueError: If no benchmark was set, so there are no active weights to
                measure.
        """
        from optimization_engine.analytics.active import transfer_coefficient

        benchmark = self._benchmark_weights()
        mu = self.expected_returns.reindex(self.result.weights.index).fillna(0.0)
        alphas = mu - float(mu @ benchmark)
        return transfer_coefficient(
            alphas, self.result.weights - benchmark, self.cov_matrix, method=method
        )

    # -- backtests ----------------------------------------------------------

    def backtest_returns(self, benchmark_returns: pd.Series | None = None) -> pd.DataFrame:
        """In-sample replay assuming costless rebalancing every period.

        Kept for backwards compatibility and for quick comparisons. It is
        **in-sample** — the weights were chosen knowing these returns — and it
        ignores drift and trading costs. Use :meth:`backtest` for an honest
        replay and :meth:`walk_forward` for an out-of-sample track record.

        Args:
            benchmark_returns: A benchmark stream to include as a second column.
                ``None`` returns the portfolio alone.

        Returns:
            A frame with a ``portfolio`` column, and a ``benchmark`` column when
            one was supplied.
        """
        port = (self.returns * self.result.weights.reindex(self.returns.columns).fillna(0.0)).sum(axis=1)
        out = pd.DataFrame({"portfolio": port})
        if benchmark_returns is None and self.benchmark is not None:
            benchmark_returns = self.benchmark.returns
        if benchmark_returns is not None:
            out["benchmark"] = benchmark_returns.reindex(port.index)
        return out

    def backtest(
        self,
        frequency: RebalanceFrequency = "monthly",
        transaction_cost_bps: float = 0.0,
    ) -> BacktestResult:
        """Replay the solved weights with realistic drift, rebalancing and costs.

        Still in-sample: the optimizer saw this history. What it adds over
        :meth:`backtest_returns` is that the weights actually drift between
        rebalances and trading is charged for.

        Args:
            frequency: How often the book is rebalanced back to target.
            transaction_cost_bps: Round-trip cost charged on the traded notional,
                in basis points.

        Returns:
            A :class:`~optimization_engine.analytics.backtest.BacktestResult` with
            the net return stream, the turnover and cost paths, and the rebalance
            dates.
        """
        return backtest_weights(
            self.returns,
            self.result.weights,
            frequency=frequency,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=self.config.periods_per_year,
            is_out_of_sample=False,
        )

    def walk_forward(
        self,
        lookback: int | None = None,
        rebalance_every: int | None = None,
        transaction_cost_bps: float = 0.0,
        expanding: bool = False,
        solve: Callable[[pd.DataFrame], pd.Series] | None = None,
        reestimate_expected_returns: bool = True,
        rebalance_frequency: RebalanceFrequency | None = None,
    ) -> WalkForwardResult:
        """Out-of-sample evaluation: re-estimate and re-solve on a rolling window.

        Defaults to a two-year lookback re-solved quarterly, scaled by
        ``periods_per_year`` so the same call works for daily, weekly or
        monthly data.

        Args:
            lookback: Estimation window in periods. Defaults to two years.
            rebalance_every: Periods between **re-solves** — how often the
                optimizer sees new data. Defaults to one quarter.
            transaction_cost_bps: One-way cost on traded notional.
            expanding: Grow the window from the start instead of rolling it.
            solve: Override the solver. Defaults to re-running this run's own
                config on each window — which is the point: the *process* is
                what gets evaluated, not one lucky weight vector.
            reestimate_expected_returns: Re-derive expected returns inside each
                window instead of reusing the ones on the config.

                This defaults to True because leaving it off is a look-ahead
                leak in the usual case. ``config.expected_returns`` is normally
                populated — the UI always fills that table, and it seeds it from
                the *full* history — so reusing it hands every "out-of-sample"
                window an estimate computed partly from its own future. On the
                sample panel that lifts walk-forward Sharpe from 0.46 to 0.89.

                Set it to False only when the expected returns are genuinely
                forward-looking capital-market assumptions rather than
                estimates from this history; then holding them fixed is right,
                and the engine cannot tell the two cases apart on its own.
            rebalance_frequency: How often the book is **traded back** to the
                current target *between* re-solves — the rebalancing cadence,
                which is a different decision from ``rebalance_every``. A
                committee that re-solves quarterly but rebalances monthly
                passes ``rebalance_every=63, rebalance_frequency="monthly"``
                on a daily panel. Defaults to ``"none"``: hold each solution
                untouched until the next one, letting the weights drift.
        """
        ppy = self.config.periods_per_year
        lookback = lookback or max(2 * ppy, 24)
        rebalance_every = rebalance_every or max(ppy // 4, 1)

        if solve is None:
            solve = self._window_solver(reestimate_expected_returns)

        result = walk_forward_backtest(
            self.returns,
            solve,
            lookback=lookback,
            rebalance_every=rebalance_every,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=ppy,
            expanding=expanding,
            rebalance_frequency=rebalance_frequency,
        )
        result.backtest.metadata["reestimated_expected_returns"] = bool(
            reestimate_expected_returns
        )
        return result

    # -- the full simulation stack -----------------------------------------

    def simulate(
        self,
        spec: BacktestSpec | None = None,
        *,
        weights: pd.Series | None = None,
        prices: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
    ) -> RunResult:
        """Replay the solved weights and return the full result bundle.

        Where :meth:`backtest` gives the compact result, this gives everything
        the simulation core produced: per-trade costs, NAV, the target
        schedule, and the spec and result hashes that let one run be compared
        with another. Take a :class:`~optimization_engine.backtest.spec.BacktestSpec`
        when you need an execution lag or a cost model with market impact.

        Still in-sample unless the spec says otherwise — the optimizer saw
        this history.

        Args:
            spec: The run description.
            weights: Targets to replay. Defaults to this run's solution.
            prices: Close prices, needed only to convert share volume into
                traded notional.
            volumes: Traded volume per asset. Optional: without it the impact
                model prices from a fixed participation rate, which is the
                only thing available for an index universe.
        """
        spec = spec or BacktestSpec(periods_per_year=self.config.periods_per_year)
        target = self.result.weights if weights is None else weights
        return run_backtest(
            self.returns, target, spec, prices=prices, volumes=volumes
        )

    def walk_forward_run(
        self,
        lookback: int | None = None,
        rebalance_every: int | None = None,
        spec: BacktestSpec | None = None,
        expanding: bool = False,
        solve: Callable[[pd.DataFrame], pd.Series] | None = None,
        reestimate_expected_returns: bool = True,
        rebalance_frequency: RebalanceFrequency | None = None,
        prices: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
        universe: Eligibility | None = None,
        universe_policy: str | None = None,
        delisting_grace: int | None = None,
    ) -> WalkForwardRun:
        """:meth:`walk_forward`, returning the full bundle instead of the digest.

        Same evaluation, same defaults; what differs is that the result
        carries the trade and cost frames and the provenance hashes, which is
        what :meth:`tearsheet` and the sweep need.

        ``rebalance_every`` is the re-optimization cadence and
        ``rebalance_frequency`` — or, if that is not given, the spec's own
        ``frequency`` — is the trading cadence. With no spec and no argument
        the book trades only when it re-solves.

        Args:
            prices: Close prices, needed only to convert share volume into
                traded notional.
            volumes: Traded volume per asset. Optional everywhere: without it
                the impact model prices from a fixed participation rate, which
                is the only thing available for an index universe.
            universe: Point-in-time membership — an
                :class:`~optimization_engine.universe.eligibility.Eligibility`,
                or ``None`` to take the universe from the panel's columns, which
                is the survivors-only reading every run had before this
                argument existed.
            universe_policy: How a cell the rules could not evaluate is read —
                ``"exclude"``, ``"include"`` or ``"raise"``. Required whenever
                ``universe`` is given, and deliberately without a default here
                too: passing the decision through unchanged is the point.
            delisting_grace: Bars of silence after which a name counts as
                delisted, or ``None`` to not diagnose delisting at all. A
                separate opt-in from ``universe`` because it answers a
                different question — the screen says what the mandate permits,
                this says what still trades.

        Returns:
            The bundle, forwarded from
            :func:`~optimization_engine.backtest.walkforward.walk_forward_run`.

        Raises:
            ValueError: If the history is too short, or ``universe`` was given
                with no ``universe_policy``.
            UniverseError: If the policy is unknown, or it is ``"raise"`` and
                some name was not evaluable on some bar.
        """
        ppy = self.config.periods_per_year
        # An explicit "none" rather than the spec default: a caller who never
        # mentioned a trading cadence has not asked to pay for one.
        spec = spec or BacktestSpec(periods_per_year=ppy, frequency="none")
        return walk_forward_run(
            self.returns,
            solve or self._window_solver(reestimate_expected_returns),
            lookback=lookback or max(2 * ppy, 24),
            rebalance_every=rebalance_every or max(ppy // 4, 1),
            spec=spec,
            expanding=expanding,
            rebalance_frequency=rebalance_frequency,
            prices=prices,
            volumes=volumes,
            universe=universe,
            universe_policy=universe_policy,
            delisting_grace=delisting_grace,
        )

    def tearsheet(
        self,
        run: RunResult | None = None,
        *,
        riskfree_rate: float | None = None,
        n_trials: int | None = None,
        trial_sharpes: pd.Series | None = None,
        overfitting: Any = None,
        shocks: Sequence[Shock] | None = None,
    ) -> Tearsheet:
        """The assembled reading of a run, caveats attached.

        Defaults to describing an in-sample replay of this run's own weights,
        which is the cheapest thing to produce and the least informative — the
        tearsheet says so in its caveats rather than leaving it to the reader.
        Pass a walk-forward run for a number worth quoting.

        Args:
            run: The run to describe. ``None`` builds an in-sample replay of this
                run's own weights.
            riskfree_rate: Per-period risk-free rate for the ratio metrics.
                Defaults to the config's.
            n_trials: How many configurations were tried before settling on this
                one, for the deflated Sharpe. Do not guess it — run a sweep and
                let it count itself.
            trial_sharpes: The Sharpe ratios of those trials, which sharpen the
                deflation.
            overfitting: A pre-computed overfitting report to attach.
            shocks: Stress scenarios for the book the run ends on. ``None``
                takes the ones this run's config carries, so a mandate that
                declares its scenarios gets the panel without asking twice;
                pass ``()`` to suppress it.

        Returns:
            A :class:`~optimization_engine.backtest.Tearsheet` carrying the run,
            its cost analysis, the selection-bias diagnostics, the stress panel
            when scenarios were configured, and the caveats that qualify them.

        Raises:
            StressError: If a configured shock names an asset outside the
                panel this run was solved on.
        """
        rf = self.config.optimizer.risk_free_rate if riskfree_rate is None else riskfree_rate
        applied = configured_shocks(self.config) if shocks is None else tuple(shocks)
        return build_tearsheet(
            run if run is not None else self.simulate(),
            self.returns,
            riskfree_rate=rf,
            n_trials=n_trials,
            trial_sharpes=trial_sharpes,
            overfitting=overfitting,
            shocks=applied,
            stress_cov_matrix=self.cov_matrix,
        )

    def sweep(
        self,
        sweep: SweepSpec,
        *,
        lookback: int | None = None,
        rebalance_every: int | None = None,
        spec: BacktestSpec | None = None,
        expanding: bool = False,
        progress: Callable[[int, int], None] | None = None,
        rebalance_frequency: RebalanceFrequency | None = None,
        prices: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
        universe: Eligibility | None = None,
        universe_policy: str | None = None,
        delisting_grace: int | None = None,
    ) -> SweepResults:
        """Walk-forward every cell of a grid, and count the trials.

        Each cell is evaluated out of sample, because a grid scored in sample
        measures how well each configuration memorized the history rather than
        how well it would have done. The results carry the trial count that
        the deflated Sharpe and the overfitting probability both need.

        Args:
            prices: Close prices, needed only to turn share volume into traded
                notional.
            volumes: Traded volume per asset. Every cell is priced the same
                way, so a grid run with a capacity-aware cost model must be
                handed the same panel the single run was — otherwise the grid
                is cheaper than the run it is supposed to contextualize.
            universe: Point-in-time membership, applied to every cell. Same
                argument as ``volumes``, and it bites harder: the cells'
                Sharpes are what the deflated Sharpe is deflated *against*, so
                a grid run on the survivors while the headline run was screened
                would deflate one universe's result by another universe's
                dispersion.
            universe_policy: How a *not evaluable* cell is read. Required
                whenever ``universe`` is given.
            delisting_grace: Bars of silence after which a name counts as
                delisted, applied to every cell for the same reason.
        """
        ppy = self.config.periods_per_year
        run_spec = spec or BacktestSpec(periods_per_year=ppy, frequency="none")
        window = lookback or max(2 * ppy, 24)
        step = rebalance_every or max(ppy // 4, 1)

        def evaluate(cell_config: EngineConfig) -> pd.Series:
            """Walk one grid cell forward and return its out-of-sample stream.

            Expected returns are cleared on the copied config so every cell is
            re-estimated inside each window rather than reading the run's own
            full-sample estimate — which would be the look-ahead the sweep exists to
            measure around.

            Args:
                cell_config: The configuration for this cell of the grid.

            Returns:
                The cell's walk-forward return stream.
            """
            import copy

            cell = copy.deepcopy(cell_config)
            cell.expected_returns = {}

            def solve(window_returns: pd.DataFrame) -> pd.Series:
                """Solve one window and return its target weights.

                Feasibility analysis is skipped: a cell whose mandate is infeasible fails
                as a solve, is recorded as a failed cell, and still counts as a trial.

                Args:
                    window_returns: The returns visible in this window.

                Returns:
                    The solved weights.
                """
                return run_engine(
                    window_returns, cell, check_feasibility=False
                ).result.weights

            return walk_forward_run(
                self.returns,
                solve,
                lookback=window,
                rebalance_every=step,
                spec=run_spec,
                expanding=expanding,
                rebalance_frequency=rebalance_frequency,
                prices=prices,
                volumes=volumes,
                universe=universe,
                universe_policy=universe_policy,
                delisting_grace=delisting_grace,
            ).returns

        return run_sweep(
            self.config,
            sweep,
            evaluate,
            periods_per_year=ppy,
            progress=progress,
        )

    def _window_solver(
        self, reestimate_expected_returns: bool
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """This run's own config, re-solved on whatever window it is handed."""
        import copy

        base_config = copy.deepcopy(self.config)
        if reestimate_expected_returns:
            # Emptying the vector makes run_engine derive it from the window
            # via expected_returns_method.
            base_config.expected_returns = {}

        def solve(window: pd.DataFrame) -> pd.Series:
            """Re-solve this run's config on one window.

            Feasibility analysis is skipped: inside a walk-forward, a window whose
            mandate is momentarily infeasible should fail as a solve rather than stop
            the run.

            Args:
                window: The returns visible in this window.

            Returns:
                The solved target weights.
            """
            return run_engine(window, base_config, check_feasibility=False).result.weights

        return solve

    def in_vs_out_of_sample(
        self, walk_forward_result: WalkForwardResult, riskfree_rate: float = 0.0
    ) -> pd.DataFrame:
        """Side-by-side fitted vs walk-forward statistics, with the gap.

        Args:
            walk_forward_result: The out-of-sample run to compare against this
                run's own in-sample replay.
            riskfree_rate: Per-period risk-free rate for the ratio metrics.

        Returns:
            One row per statistic, a column per sample, and a ``Degradation``
            column holding the in-sample figure minus the out-of-sample one.
        """
        oos = walk_forward_result.returns
        in_sample = self.backtest_returns()["portfolio"].reindex(oos.index)
        return compare_in_and_out_of_sample(
            in_sample, oos, self.config.periods_per_year, riskfree_rate
        )

    # -- summaries ----------------------------------------------------------

    def absolute_summary(
        self, riskfree_rate: float = 0.0, extended: bool = False
    ) -> pd.DataFrame:
        """Standard performance statistics for the run's own return stream.

        Args:
            riskfree_rate: Per-period risk-free rate used by the ratio metrics.
            extended: Include the fuller set — higher moments, drawdown detail,
                tail statistics — rather than the headline figures alone.

        Returns:
            A one-row-per-series summary frame, annualized on the config's
            ``periods_per_year``.
        """
        bt = self.backtest_returns()
        return summary_stats(
            bt,
            periods_per_year=self.config.periods_per_year,
            riskfree_rate=riskfree_rate,
            extended=extended,
        )

    def performance(
        self,
        riskfree_rate: float | None = None,
        frequency: RebalanceFrequency | None = "monthly",
        transaction_cost_bps: float = 0.0,
        benchmark_returns: pd.Series | None = None,
        rolling_window: int | None = None,
        period_freq: str = "yearly",
        returns_override: pd.Series | None = None,
    ) -> PerformanceReport:
        """Absolute and relative performance of this run, in one object.

        Args:
            riskfree_rate: Annual rate for the ratios. Defaults to the
                optimizer's own, so the report and the solve agree on cash.
            frequency: Rebalancing rule for the replay. ``None`` uses the
                costless constant-weight replay instead, which is the older
                and more optimistic convention.
            transaction_cost_bps: One-way cost charged at each rebalance.
            benchmark_returns: Override the run's own benchmark stream — used
                to report against something other than the one optimized
                against, and by the walk-forward view.
            rolling_window: Window for the rolling frames. Defaults to a year.
            period_freq: ``yearly``, ``quarterly`` or ``monthly`` table.
            returns_override: Use this return stream as the portfolio's
                instead of replaying the weights. This is how a walk-forward
                track record gets the same report as the fitted one.

        Note:
            Unless ``returns_override`` carries an out-of-sample stream, every
            number here is in-sample: the optimizer estimated its inputs from
            these same returns.
        """
        rf = (
            self.config.optimizer.risk_free_rate
            if riskfree_rate is None
            else float(riskfree_rate)
        )
        metadata: dict[str, Any] = {
            "optimizer": self.config.optimizer.name,
            "rebalancing": str(frequency or "none (constant weights)"),
            "transaction_cost_bps": float(transaction_cost_bps),
            "out_of_sample": returns_override is not None,
        }
        if returns_override is not None:
            portfolio = pd.Series(returns_override).dropna()
        elif frequency is None:
            portfolio = self.backtest_returns()["portfolio"]
        else:
            bt = self.backtest(
                frequency=frequency, transaction_cost_bps=transaction_cost_bps
            )
            portfolio = bt.returns
            metadata["annualized_turnover"] = float(bt.annualized_turnover)
            metadata["total_cost"] = float(bt.total_cost)

        bench = benchmark_returns
        if bench is None and self.benchmark is not None:
            bench = self.benchmark.returns
        label = self.benchmark_label if benchmark_returns is None else "Benchmark"

        benchmark_weights = None
        if self.benchmark is not None and self.benchmark.weights is not None:
            benchmark_weights = self.benchmark.weights
        elif self.config.benchmark_weight_map(list(self.result.weights.index)):
            benchmark_weights = pd.Series(
                self.config.benchmark_weight_map(list(self.result.weights.index))
            )

        return performance_report(
            portfolio,
            bench,
            periods_per_year=self.config.periods_per_year,
            riskfree_rate=rf,
            portfolio_weights=self.result.weights,
            benchmark_weights=benchmark_weights,
            benchmark_label=label,
            rolling_window=rolling_window,
            period_freq=period_freq,
            metadata=metadata,
        )

    def relative_summary(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        """Benchmark-relative statistics for the run's return stream.

        Args:
            benchmark_returns: The benchmark's return stream, aligned to the
                portfolio's own dates.

        Returns:
            A summary frame of the relative metrics — active return, tracking
            error, information ratio and the regression statistics — annualized on
            the config's ``periods_per_year``.

        Raises:
            MissingDependencyError: If the regression metrics are reached without
                statsmodels. Install it with ``finport-optengine[stats]``.
        """
        bt = self.backtest_returns(benchmark_returns)
        return summary_relative(
            bt[["portfolio"]],
            bt["benchmark"],
            periods_per_year=self.config.periods_per_year,
        )

    def assumptions(self) -> dict[str, Any]:
        """Every modelling choice this run rests on, in one place.

        A number without its assumptions is not a result. This is what the UI
        and the Excel report print alongside the weights so the reader can see
        what was assumed rather than infer it.
        """
        spec = self.config.optimizer
        start = self.returns.index.min()
        end = self.returns.index.max()
        return {
            "optimizer": spec.name,
            "objective_mode": self.result.extras.get("mode", "—"),
            "covariance_estimator": self.config.covariance_method,
            "ewma_lambda": (
                self.config.ewma_lambda
                if self.config.covariance_method == "ewma"
                else None
            ),
            "expected_returns_method": self.config.expected_returns_method,
            "risk_free_rate": spec.risk_free_rate,
            "risk_aversion": spec.risk_aversion,
            "target_return": spec.target_return,
            "target_volatility": spec.target_volatility,
            "periods_per_year": self.config.periods_per_year,
            "base_currency": self.config.base_currency,
            "sample_start": str(getattr(start, "date", lambda: start)()),
            "sample_end": str(getattr(end, "date", lambda: end)()),
            "n_observations": int(len(self.returns)),
            "n_assets": int(self.returns.shape[1]),
            "long_only": self.config.long_only,
            "fully_invested": self.config.fully_invested,
            "leverage_cap": self.config.leverage,
            "turnover_limit": self.config.turnover_limit,
            "benchmark": (self.benchmark_label or "—"),
            "benchmark_kind": (
                self.benchmark.spec.kind if self.benchmark is not None else "none"
            ),
            "max_tracking_error": self.config.max_tracking_error,
            "max_active_share": self.config.max_active_share,
            "constraint_layers": (
                "; ".join(
                    f"{lyr.name} ({len(lyr.limits)} buckets"
                    + (f", % of {lyr.parent}" if lyr.is_relative else "")
                    + ")"
                    for lyr in self.constraint_layers
                    if lyr.is_active
                )
                or "—"
            ),
            "solver": self.result.extras.get("solver"),
            "solver_status": self.result.extras.get("solver_status"),
        }


def resolve_expected_returns(
    config: EngineConfig,
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    expected_returns: pd.Series | None = None,
) -> pd.Series:
    """The expected-return vector a run will actually optimize against.

    Extracted because more than one caller needs it and they have to agree.
    A pre-flight check that validates a mandate against a different vector
    from the one the solve uses is worse than no check: on a config with no
    ``expected_returns`` block it would report the reachable return range as
    zero to zero, pronounce a return target unreachable, and then watch the
    solve succeed — or the reverse.

    Precedence: an explicit vector, then ``config.expected_returns``, then
    an estimate from the return history using the configured method.

    Args:
        config: Supplies ``expected_returns`` and, failing that, the estimator
            to derive them with.
        returns: The return history to estimate from, and the authority on the
            universe.
        cov: The covariance, needed by the CAPM and shrinkage estimators.
        expected_returns: An explicit vector, which wins over everything else.

    Returns:
        Annualized expected returns, always reindexed onto
        ``returns.columns``. An asset the vector does not cover is filled with
        zero rather than a NaN that would propagate through the objective — and
        a ``UserWarning`` names it, because a zero expected return is an active
        view, not a neutral one. Names the vector carries that the panel does
        not hold are dropped. :func:`run_engine` records both in
        ``run.warnings`` and ``result.extras``.

    Raises:
        ConfigurationError: If the vector both misses panel assets *and* names
            assets the panel does not hold — the signature of a misspelt key,
            which would otherwise zero the asset it was meant for. See
            :func:`expected_return_gaps`.
    """
    resolved, _, _ = _resolve_expected_returns(config, returns, cov, expected_returns)
    return resolved


def expected_return_gaps(
    expected_returns: pd.Series | dict[str, float], assets: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Where an expected-return vector and a universe disagree.

    Args:
        expected_returns: The vector, as a Series or a mapping.
        assets: The universe it is meant to cover.

    Returns:
        ``(missing, unknown)``: the assets with no expected return — absent, or
        NaN — in universe order, and the vector's names the universe does not
        hold, in the vector's order.
    """
    vector = pd.Series(expected_returns, dtype=float)
    universe = {str(a) for a in assets}
    covered = {str(k) for k, v in vector.items() if pd.notna(v)}
    missing = [str(a) for a in assets if str(a) not in covered]
    unknown = [str(k) for k in vector.index if str(k) not in universe]
    return missing, unknown


def _resolve_expected_returns(
    config: EngineConfig,
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    expected_returns: pd.Series | None,
) -> tuple[pd.Series, list[str], list[str]]:
    """:func:`resolve_expected_returns`, also returning what it filled and dropped.

    The rule, applied to an explicit or configured vector (an estimate from the
    history covers the panel by construction):

    * Missing *and* unknown names together are refused. That is what a
      misspelling looks like — ``EM_Equityy`` given, ``EM_Equity`` therefore
      absent — and filling the real asset with zero would build the book on a
      bearish view nobody stated.
    * Missing names alone are filled with zero and warned about.
    * Unknown names alone are dropped. A walk-forward hands a screened window
      a subset of the columns the config was written for; that is not a
      mistake, and refusing it would fail every such window.

    Returns:
        ``(vector, missing, unknown)``.

    Raises:
        ConfigurationError: As :func:`resolve_expected_returns` documents.
    """
    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns)
    if expected_returns is None:
        from optimization_engine.data.covariance import expected_returns_from_history

        market_w = pd.Series(config.market_weights) if config.market_weights else None
        expected_returns = expected_returns_from_history(
            returns,
            method=(
                "mean"
                if config.expected_returns_method == "historical_mean"
                else config.expected_returns_method
            ),
            periods_per_year=config.periods_per_year,
            span=config.ema_span,
            market_return=config.market_return,
            risk_free_rate=config.optimizer.risk_free_rate,
            market_weights=market_w,
            cov_matrix=cov,
        )
    missing, unknown = expected_return_gaps(expected_returns, list(returns.columns))
    if missing and unknown:
        raise ConfigurationError(
            f"The expected returns name {', '.join(unknown)}, which the panel does "
            f"not hold, and give none for {', '.join(missing)}, which it does. That "
            "is what a misspelt key looks like, and filling the real asset with "
            "0.0 would optimize on a bearish view nobody stated. Fix the names."
        )
    if missing:
        warnings.warn(
            f"resolve_expected_returns: no expected return for {len(missing)} asset(s) "
            f"({', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}); "
            "assuming 0.0. A zero expected return is an active view, not a "
            "neutral one.",
            UserWarning,
            stacklevel=3,
        )
    return expected_returns.reindex(returns.columns).fillna(0.0), missing, unknown


def configured_shocks(config: EngineConfig) -> tuple[Shock, ...]:
    """The stress scenarios a configuration carries, if it carries any.

    Args:
        config: The configuration to read ``stress`` from.

    Returns:
        The shocks as a tuple, empty when none are configured. Read through
        ``getattr`` for the same reason
        :func:`~optimization_engine.data.covariance.covariance_from_config`
        reads the denoising settings that way — a config object built by an
        older release, or duck-typed by a caller, still runs.

    Raises:
        StressError: If ``config.stress`` is set to something that is neither a
            sequence of :class:`~optimization_engine.stress.Shock` nor of the
            mappings one serializes to, or names the same scenario twice.
    """
    return shocks_from_dicts(getattr(config, "stress", ()) or ())


def run_engine(
    returns: pd.DataFrame,
    config: EngineConfig,
    expected_returns: pd.Series | None = None,
    build_frontier: bool = False,
    n_frontier_points: int = 25,
    return_range: tuple[float, float] | None = None,
    check_feasibility: bool = True,
    raise_on_infeasible: bool = False,
    external_returns: pd.DataFrame | pd.Series | None = None,
    run_stress: bool = False,
) -> EngineRun:
    """Run the engine end-to-end.

    Args:
        returns: A DataFrame of asset returns (rows = periods, cols = assets).
        config: An :class:`EngineConfig` describing the optimizer + constraints.
        expected_returns: Override for expected returns. Defaults to
            ``config.expected_returns``.
        build_frontier: If True, also computes the efficient frontier.
        n_frontier_points: Resolution of the frontier sweep.
        return_range: Optional (lo, hi) range to sweep. Defaults to the range
            the constraints can actually reach.
        check_feasibility: Analyze the constraint set before solving and
            attach the report to the run. Cheap relative to the solve, and it
            turns ``status=infeasible`` into an actionable message.
        raise_on_infeasible: Raise :class:`InfeasibleConstraintsError` instead
            of letting the solver fail with a less informative error.
        external_returns: Return series from outside the investable universe,
            needed only when ``config.benchmark`` names an external index.
        run_stress: Apply ``config.stress`` to the solved book and attach the
            report as ``run.stress``. Off by default, and deliberately: the
            walk-forward and sweep solvers call this function once per window
            and throw everything but the weights away, so nothing they do not
            read should be computed. Setting it with no shocks configured is a
            no-op, not an error.

    Returns:
        An :class:`EngineRun` carrying the allocation and the evidence behind
        it — the covariance, the expected returns, the feasibility report when
        one was asked for, the frontier when one was built, and the stress
        report when ``run_stress`` was set and shocks were configured.

    Raises:
        ValueError: If ``returns`` is empty or has no columns.
        InfeasibleConstraintsError: When ``raise_on_infeasible`` is set and the
            constraints cannot be satisfied.
        StressError: When ``run_stress`` is set and a configured shock names an
            asset outside the panel, which is the same defect as a view on one.
        ConfigurationError: When the expected returns both miss panel assets
            and name assets the panel does not hold — a misspelt key (see
            :func:`resolve_expected_returns`). Also when
            ``config.strict_mandate`` is set and a bound or a layer assignment
            names an asset the panel does not hold; that one is found by the
            pre-solve analysis, so not checked under ``check_feasibility=False``.
        MandateViolationError: When ``config.strict_mandate`` is set and the
            solved book breaches the mandate past tolerance. Unlike the other
            two this is raised *after* a successful solve — the answer arrived
            and does not comply — so it is not a solver failure and the
            ``except SolverFailure`` below deliberately does not intercept it
            (``SolverFailure`` is a ``RuntimeError``, this is a ``ValueError``,
            so the two never overlap). Two consequences worth stating rather
            than leaving to be discovered:

            * A caller guarding a solve with ``except SolverFailure`` alone
              will not catch it. Catch it by name, or leave
              ``strict_mandate`` off and read ``run.result.audit``.
            * Inside a walk-forward it does not propagate.
              :func:`~optimization_engine.backtest.walkforward.walk_forward_run`
              catches every exception a window's solve raises and records the
              window as ``failed: <message>``, carrying the previous book
              forward — or holding cash when there is no previous book. That is
              coherent (a window whose only compliant answer is "no book" is a
              window the desk could not trade) but surprising: turning
              ``strict_mandate`` on does not stop a backtest, it converts
              non-compliant windows into carried-forward ones. Read
              ``walk.failures`` and ``walk.n_failures``, which name every one.
              The single exception is a method that can *never* satisfy the
              mandate, where every window refuses: the walk-forward then raises
              a plain ``ValueError`` quoting the first window's message, so
              that case is loud — but it arrives as a ``ValueError`` with no
              ``.report`` on it, not as the ``MandateViolationError`` a caller
              would be catching for.
    """
    if returns is None or returns.empty:
        raise ValueError("run_engine received an empty returns frame.")
    if returns.shape[1] == 0:
        raise ValueError("run_engine received returns with no asset columns.")

    cov = covariance_from_config(returns, config)
    cov_diag = covariance_diagnostics(
        cov,
        n_observations=len(returns),
        method=config.covariance_method,
        ewma_lambda=config.ewma_lambda,
    )

    expected_returns, missing_mu, ignored_mu = _resolve_expected_returns(
        config, returns, cov, expected_returns
    )

    benchmark = resolve_benchmark(config.benchmark, returns, external_returns)
    constraints = constraints_from_config(config, list(returns.columns))
    # Before the feasibility LP, not after: a budget with no benchmark to
    # measure it against is a configuration error, and the LP would otherwise
    # be the first thing to trip over it and report it as a solver problem.
    validate_benchmark_constraints(config.optimizer, constraints)

    feasibility: FeasibilityReport | None = None
    if check_feasibility:
        # Black-Litterman optimizes against its equilibrium posterior, not the
        # configured vector, so a return target has to be checked against the
        # returns the solver will really see.
        feasibility = analyze_feasibility(
            list(returns.columns),
            constraints,
            expected_returns=effective_expected_returns(config, cov, expected_returns),
            cov_matrix=cov,
        )
        # A bound or layer assignment on an asset the panel does not hold is a
        # warning by default — it constrains nothing, so the mandate is still
        # solvable. Under strict_mandate it is refused: the author believes the
        # limit binds, and the usual cause is a misspelt name. Tied to this
        # analysis rather than done unconditionally because a walk-forward
        # window skips it, and a screened universe narrowing for a window is not
        # a misspelling.
        stray = [i.message for i in feasibility.issues if i.code in _STRAY_MANDATE_CODES]
        if stray and getattr(config, "strict_mandate", False):
            raise ConfigurationError(
                "strict_mandate refuses a mandate that names assets the panel does "
                f"not hold: {'; '.join(stray)}. Fix the names, or drop the limits."
            )
        if raise_on_infeasible and not feasibility.is_feasible:
            raise InfeasibleConstraintsError(feasibility)

    optimizer = optimizer_factory(
        config, cov, expected_returns=expected_returns, returns=returns
    )
    # The gate lives on the optimizer, not here, so that *every* entry into a
    # solve honours it and not just this one: the frontier sweep, the
    # walk-forward's per-window solver and any caller holding the instance all
    # go through ``optimize()``, and a check written into ``run_engine`` would
    # bind on exactly one of those paths. Read through ``getattr`` for the same
    # reason ``configured_shocks`` does — a duck-typed config from an older
    # release still runs.
    optimizer.strict_mandate = bool(getattr(config, "strict_mandate", False))
    try:
        result = optimizer.optimize()
    except SolverFailure as exc:
        # Deliberately narrow. ``optimize()`` also raises
        # ``MandateViolationError`` under the ``strict_mandate`` set above, and
        # that must pass straight through: the solve succeeded and the *answer*
        # is non-compliant, which is not a solver failure and must not be
        # re-dressed as one. The types keep them apart on their own —
        # ``SolverFailure`` is a ``RuntimeError``, ``MandateViolationError`` a
        # ``ValueError`` — so this is a statement of intent, not a guard.
        #
        # A solver that reports "infeasible" has found the same thing the
        # pre-solve analysis did, but says it in solver terms. When the
        # analysis named the culprit, attach its findings rather than leaving
        # the caller with "no allocation satisfies every constraint at once" —
        # the useful sentence is "the 7% return target is above the 6.8% these
        # constraints reach". The exception type is unchanged, so callers
        # catching SolverFailure keep working.
        if feasibility is not None and not feasibility.is_feasible:
            raise SolverFailure(
                exc.status, exc.attempts, detail=feasibility.describe()
            ) from exc
        raise

    frontier = None
    if build_frontier:
        frontier = efficient_frontier(
            config,
            cov,
            expected_returns=expected_returns,
            returns=returns,
            n_points=n_frontier_points,
            return_range=return_range,
        )

    stress: StressReport | None = None
    if run_stress:
        shocks = configured_shocks(config)
        if shocks:
            stress = stress_test(result.weights, shocks, cov_matrix=cov)

    run_warnings: list[str] = list(cov_diag.warnings)
    # The vector reaching the optimizer is already complete, so the
    # optimizer's own missing-return warning cannot fire on this path. The
    # gaps are recorded here instead, where the caller reads them.
    if missing_mu:
        result.extras["missing_expected_returns"] = list(missing_mu)
        run_warnings.append(
            f"No expected return for {', '.join(missing_mu)}; optimized with 0.0, "
            "which is an active view, not a neutral one."
        )
    if ignored_mu:
        result.extras["ignored_expected_returns"] = list(ignored_mu)
        run_warnings.append(
            f"Expected returns given for {', '.join(ignored_mu)}, which the panel "
            "does not hold; they were ignored."
        )
    if feasibility is not None:
        run_warnings.extend(i.message for i in feasibility.warnings)
    run_warnings.extend(result.violations)

    return EngineRun(
        config=config,
        returns=returns,
        cov_matrix=cov,
        expected_returns=expected_returns,
        result=result,
        frontier=frontier,
        feasibility=feasibility,
        covariance_diagnostics=cov_diag,
        warnings=tuple(run_warnings),
        benchmark=benchmark,
        stress=stress,
    )
