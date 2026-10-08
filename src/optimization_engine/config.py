"""Configuration objects for the optimization engine.

The engine is intentionally driven by data. All asset metadata —
expected returns, group mappings, weight bounds, optimizer choice —
lives in a `EngineConfig` object that can be loaded from YAML/JSON,
mutated programmatically, or built from a UI.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from optimization_engine.benchmark import BenchmarkSpec
from optimization_engine.constraints import ConstraintLayer, coerce_layers
from optimization_engine.stress import Shock, shocks_from_dicts, shocks_to_dicts


@dataclass
class OptimizerSpec:
    """Specification of which optimizer to run and its hyperparameters.

    Attributes:
        name: Registered optimizer name (see ``available_optimizers()``).
        target_return: Hard annualized expected-return target.
        target_volatility: Hard annualized volatility cap.
        risk_free_rate: Annual risk-free rate used for Sharpe and for the
            tangency and CAPM/Black-Litterman calculations.
        risk_aversion: Utility coefficient λ in ``μ'w − λ·w'Σw``.
        cvar_alpha: Tail probability for mean-CVaR (``0.05`` ⇒ 95% CVaR).
        risk_budget: ``asset -> target share of total risk`` for risk parity.
        bl_views: Black-Litterman views. Either ``{asset: return}`` for
            absolute views, or a list of ``{"weights": {...},
            "expected_return": x, "confidence": y, "label": z}`` mappings for
            relative (spread) views.
        bl_view_confidences: Ω diagonal for the mapping form of ``bl_views``.
        bl_tau: Prior-uncertainty scale in the Black-Litterman posterior.
        bl_market_caps: Equilibrium market portfolio for reverse optimization.
        bl_market_return: Observed market return, used to imply δ.
        bl_calibrate_risk_aversion: Imply δ from the market Sharpe ratio
            instead of using ``risk_aversion`` verbatim.
        hrp_linkage: Hierarchical clustering linkage rule for HRP.
        cdar_alpha: Tail probability for mean-CDaR (``0.05`` ⇒ the worst 5%
            of the drawdown path).
        cluster_linkage: Linkage rule for the clustering methods that
            *partition* the tree (HERC, NCO) rather than merely order it.
        n_clusters: Force a cluster count for HERC/NCO. ``None`` selects it
            by maximizing the silhouette t-statistic.
        max_clusters: Upper bound for that search.
        herc_risk_measure: Cluster-level risk measure for HERC.
        nco_objective: Objective solved at both NCO layers.
        nco_detone_for_clustering: Strip the market eigenvector before
            clustering in NCO.
        accept_inaccurate: Take an ``optimal_inaccurate`` answer when no
            solver in the fallback chain converges exactly. ``False`` — the
            default — refuses it: the run raises rather than report weights
            no solver would vouch for as though they were optimal. Set it to
            ``True`` only having decided that an approximate book is more
            use than no book, and read ``solver_status`` on the result, which
            says which one you got. It is a no-op for the methods that never
            call a solver (HRP, HERC and the naive weightings); it *does*
            reach NCO, whose two layers are solved by real optimizers.
    """

    name: str = "mean_variance"
    target_return: float | None = None
    target_volatility: float | None = None
    risk_free_rate: float = 0.0
    risk_aversion: float = 1.0
    cvar_alpha: float = 0.05
    risk_budget: dict[str, float] | None = None
    bl_views: dict[str, float] | None = None
    bl_view_confidences: dict[str, float] | None = None
    bl_tau: float = 0.05
    bl_market_caps: dict[str, float] | None = None
    hrp_linkage: Literal["single", "average", "complete", "ward"] = "single"
    cdar_alpha: float = 0.05
    cluster_linkage: Literal["single", "average", "complete", "ward"] = "ward"
    n_clusters: int | None = None
    max_clusters: int | None = None
    herc_risk_measure: Literal[
        "variance", "std", "cvar", "cdar", "equal_weight"
    ] = "variance"
    nco_objective: Literal["min_variance", "max_sharpe"] = "min_variance"
    nco_detone_for_clustering: bool = True
    bl_market_return: float | None = None
    bl_calibrate_risk_aversion: bool = False
    accept_inaccurate: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The spec as a plain dict, with unset fields omitted.

        Returns:
            Every field that is not ``None``, so a serialized config carries only
            what was actually chosen rather than every knob's default. Copied,
            not shared: a caller editing the dump — ``extra``, ``risk_budget``,
            the views — must not reach back into the spec.
        """
        return {k: copy.deepcopy(v) for k, v in self.__dict__.items() if v is not None}


#: Every key :meth:`EngineConfig.from_dict` reads. A key outside this set is
#: a typo or a field from another schema, and either way the config would
#: load while silently not doing what its author wrote.
_CONFIG_KEYS = frozenset(
    {
        "expected_returns", "bounds", "groups", "group_bounds", "constraint_layers",
        "currencies", "base_currency", "periods_per_year", "covariance_method",
        "ewma_lambda", "denoise", "denoise_method", "denoise_alpha", "detone",
        "expected_returns_method", "ema_span", "market_return", "market_weights",
        "optimizer", "benchmark", "benchmark_weights", "max_tracking_error",
        "max_active_share", "long_only", "fully_invested", "leverage",
        "previous_weights", "turnover_limit", "strict_mandate", "stress",
    }
)
_OPTIMIZER_KEYS = frozenset(OptimizerSpec.__dataclass_fields__)

#: Where a config's expected-return vocabulary differs from the estimator's.
#: ``EngineConfig`` says ``historical_mean``; the estimator in
#: :mod:`optimization_engine.data.covariance` calls the same thing ``mean``.
#: Every other name is shared. One entry, but it was written out inline in
#: four separate places, so a name added on one side quietly failed to reach
#: the other.
_EXPECTED_RETURN_METHOD_ALIASES: dict[str, str] = {"historical_mean": "mean"}


#: The switches on :class:`EngineConfig`, read by :func:`_as_bool` rather than
#: by ``bool()``.
_BOOLEAN_KEYS = ("long_only", "fully_invested", "strict_mandate", "denoise")


def _as_bool(value: Any, key: str) -> bool:
    """A config switch, read strictly: a real boolean, or the words true and false.

    ``bool()`` is the wrong reader for a value that came from a file. It turns
    the JSON string ``"false"`` into ``True`` — so a file that quoted its
    booleans switched every one of them on, ``long_only`` included — and an
    empty YAML value into ``False``. YAML and JSON both have real booleans; the
    two words are accepted, in any case, for the file that quoted them, and so
    are ``0`` and ``1``. Anything else is a question with no safe answer.

    Args:
        value: The raw value.
        key: The field it was read for, named in the error.

    Returns:
        The boolean.

    Raises:
        ConfigurationError: If ``value`` is anything else — ``"no"``, ``2``,
            ``None``.
    """
    from optimization_engine.optimizers import ConfigurationError

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ConfigurationError(f"{key} must be true or false; got {value!r}.")


def expected_return_method_for_estimator(method: str) -> str:
    """Translate a config's expected-return method into an estimator name.

    Args:
        method: The value of ``EngineConfig.expected_returns_method``.

    Returns:
        The corresponding
        :data:`~optimization_engine.data.covariance.ExpectedReturnMethod`.
        Unknown names pass through untouched so
        :func:`~optimization_engine.data.covariance.expected_returns_from_history`
        raises with its own message rather than this one guessing.
    """
    return _EXPECTED_RETURN_METHOD_ALIASES.get(method, method)


#: How ``EngineConfig`` seeds expected returns. Named rather than written
#: inline on the field, so a caller can read the vocabulary without resolving
#: the class's annotations: ``typing.get_type_hints(EngineConfig)`` evaluates
#: every annotation on the class, and several use PEP 604 unions, which are a
#: runtime ``TypeError`` on the 3.9 this package still supports.
ExpectedReturnsMethod = Literal[
    "historical_mean", "geometric_mean", "ema", "capm", "shrunk_mean"
]


@dataclass
class EngineConfig:
    """Complete engine configuration.

    Attributes:
        expected_returns: Expected (annual) return per asset.
        bounds: Min/max weight per asset, as ``[lo, hi]`` pairs.
        groups: Optional ``asset -> group`` mapping (e.g. asset class).
            This is the first layer of the allocation policy.
        group_bounds: Optional ``group -> [lo, hi]`` pairs.
        constraint_layers: Further layers of the policy, applied on top of
            ``groups``/``group_bounds`` and simultaneously with it — a
            sub-asset-class breakdown inside each asset class, a currency
            split cutting across all of them, a regional overlay. Each layer
            maps assets to buckets and caps each bucket, either as a share of
            the whole book or as a share of its parent layer's bucket. See
            :mod:`optimization_engine.constraints`.
        periods_per_year: Number of return observations per year. Must be
            positive: it annualizes the covariance, and zero made it the zero
            matrix — which then solved to an "optimal" equal-weight book with
            no risk at all.
        covariance_method: ``sample``, ``ledoit_wolf``, ``oas``,
            ``shrink`` (an alias for ``ledoit_wolf``),
            ``ewma``, ``semi``, or ``denoised`` (sample covariance filtered
            through the Marchenko-Pastur eigenvalue cutoff).
        ewma_lambda: Decay used when ``covariance_method == "ewma"``. Strictly
            between 0 and 1, and checked whatever the method, since a value
            outside that interval cannot mean anything.
        denoise: Apply the Marchenko-Pastur eigenvalue filter to the
            covariance estimate (López de Prado, 2020). Composable with any
            estimator; implied by ``covariance_method="denoised"``.
        denoise_method: ``constant_residual`` or ``targeted_shrinkage``.
        denoise_alpha: Noise-block shrinkage retained under
            ``targeted_shrinkage``.
        detone: Number of leading eigenvectors (the market component) to
            remove after denoising. Non-zero makes the covariance singular —
            appropriate for the clustering methods, not for solves that
            invert it.
        expected_returns_method: How to seed expected returns when
            ``expected_returns`` is empty: ``historical_mean`` (the
            annualized *arithmetic* mean, which is the single-period
            expectation mean-variance is defined against),
            ``geometric_mean`` (the compound growth rate — a different
            question, and inconsistent with an arithmetic covariance),
            ``ema``, ``shrunk_mean`` or ``capm``. The names map onto
            :data:`~optimization_engine.data.covariance.EXPECTED_RETURN_DESCRIPTIONS`
            via :func:`expected_return_method_for_estimator`.
        ema_span: Span for the ``ema`` method.
        market_return: Optional CAPM market return (defaults to estimated).
        market_weights: Optional CAPM market portfolio (defaults to equal).
        optimizer: ``OptimizerSpec`` describing the run.
        benchmark: How the benchmark is defined — 1/N, a single asset, an
            explicit policy vector, or an external index. It drives both the
            relative performance report and the benchmark-relative
            constraints below, so the report and the solve cannot disagree
            about what the benchmark is.
        benchmark_weights: Explicit benchmark weight vector. Overrides
            whatever ``benchmark`` would resolve to; kept because a caller
            may have the vector without wanting to describe how it was built.
        max_tracking_error: Cap on annualized active risk versus the
            benchmark, ``√((w−b)'Σ(w−b))``. Imposed inside the solve by the
            mean-variance family, mean-CVaR/CDaR and active mean-variance;
            reported but not enforced by the projection-based methods.
        max_active_share: Cap on ``½·Σ|w_i − b_i|``. Binds on positions
            rather than on realized risk, so it holds in a calm market where
            a tracking-error budget silently permits an unrecognizable
            portfolio.
        long_only: Forbid short positions. When False, per-asset minimum
            weights may go negative.
        fully_invested: Require ``sum(w) == 1``.
        leverage: Cap on gross exposure ``Σ|w_i|``. Only meaningful once
            ``long_only`` is off — a long-only, fully-invested book always
            has gross exposure of exactly 1.
        previous_weights: The portfolio being traded from. Needed for the
            turnover budget and for turnover reporting.
        turnover_limit: Cap on ``Σ|w_i − w_prev,i|``, summed over both books:
            a previous holding the universe no longer contains is sold in
            full and counts against the budget. Honoured by the
            mean-variance family and mean-CVaR; the homogeneous solves
            (max-Sharpe, max-diversification, risk parity) warn instead.
        strict_mandate: Refuse a book that breaches the mandate instead of
            reporting the breach. Off by default, which is the engine's
            long-standing contract: the post-solve audit is attached to the
            result and to ``run.warnings``, and the caller decides. Turning it
            on raises
            :class:`~optimization_engine.optimizers.audit.MandateViolationError`
            on any violation past tolerance. It matters most for the methods
            that apply bounds by projection — HRP, HERC, NCO, the naive
            weightings — which can return a book their mandate does not permit;
            a turnover budget or a tracking-error cap is dropped by the
            projection entirely, and this is what turns that from a warning
            into a stop. It also refuses, before the solve, a bound or a layer
            assignment naming an asset the panel does not hold — otherwise a
            warning — because a limit the author believes is binding and that
            constrains nothing is a mandate breach nobody can see. That check
            rides on the pre-solve analysis, so ``run_engine(...,
            check_feasibility=False)`` — the walk-forward's per-window solve,
            where a screened universe legitimately narrows — skips it.
        stress: Named one-period shock scenarios for the pre-trade stress
            report — see :mod:`optimization_engine.stress`. Applied by
            ``run_engine(..., run_stress=True)`` and by
            :meth:`~optimization_engine.engine.EngineRun.tearsheet`; empty runs
            no stress test. Entries may be given as
            :class:`~optimization_engine.stress.Shock` objects or as the
            mappings they serialize to, and are normalized to the former.
    """

    expected_returns: dict[str, float] = field(default_factory=dict)
    bounds: dict[str, list[float]] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    group_bounds: dict[str, list[float]] = field(default_factory=dict)
    constraint_layers: list[ConstraintLayer] = field(default_factory=list)
    currencies: dict[str, str] = field(default_factory=dict)
    base_currency: str = "USD"
    periods_per_year: int = 252
    covariance_method: str = "ledoit_wolf"
    ewma_lambda: float = 0.94
    denoise: bool = False
    denoise_method: str = "constant_residual"
    denoise_alpha: float = 0.0
    detone: int = 0
    expected_returns_method: ExpectedReturnsMethod = "historical_mean"
    ema_span: int = 180
    market_return: float | None = None
    market_weights: dict[str, float] | None = None
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    benchmark: BenchmarkSpec = field(default_factory=BenchmarkSpec)
    benchmark_weights: dict[str, float] | None = None
    max_tracking_error: float | None = None
    max_active_share: float | None = None
    long_only: bool = True
    fully_invested: bool = True
    leverage: float | None = None
    previous_weights: dict[str, float] | None = None
    turnover_limit: float | None = None
    strict_mandate: bool = False
    stress: tuple[Shock, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the nested blocks into canonical form, and refuse impossible values.

        A config loaded from YAML has mappings where one built in memory has
        :class:`~optimization_engine.constraints.ConstraintLayer`,
        :class:`~optimization_engine.stress.Shock` and
        :class:`~optimization_engine.benchmark.BenchmarkSpec` objects; after
        this they behave identically. It runs on direct construction too, so
        ``EngineConfig(stress=[{"name": ..., "returns": ...}])`` is as valid as
        the loaded form, and a malformed scenario is refused where it was
        written rather than at the solve.

        The switches are read with :func:`_as_bool` rather than ``bool()``,
        and the two numbers that cannot take any value are checked here, where
        they were written: each of them used to load, and then either solve to
        a meaningless book or fail deep inside an estimator.

        Raises:
            LayerConfigurationError: If any layer entry is malformed.
            StressError: If any stress entry is malformed, or two scenarios
                share a name.
            BenchmarkError: If the benchmark block is malformed.
            ConfigurationError: If a switch is neither true nor false,
                ``periods_per_year`` is not positive, or ``ewma_lambda`` is not
                strictly between 0 and 1.
        """
        from optimization_engine.optimizers import ConfigurationError

        self.constraint_layers = list(coerce_layers(self.constraint_layers))
        self.stress = shocks_from_dicts(self.stress)
        self.benchmark = BenchmarkSpec.from_dict(self.benchmark)
        for key in _BOOLEAN_KEYS:
            setattr(self, key, _as_bool(getattr(self, key), key))
        periods = float(self.periods_per_year)
        if not math.isfinite(periods) or periods <= 0:
            raise ConfigurationError(
                f"periods_per_year must be a positive number of return "
                f"observations per year — 252 daily, 52 weekly, 12 monthly; got "
                f"{self.periods_per_year!r}. It annualizes the covariance, and a "
                "zero or negative one makes every risk figure meaningless."
            )
        decay = float(self.ewma_lambda)
        if not 0.0 < decay < 1.0:
            raise ConfigurationError(
                f"ewma_lambda must lie strictly between 0 and 1; got "
                f"{self.ewma_lambda!r}. The estimator weights the observation k "
                "periods back by (1 − λ)·λᵏ: at 0 the covariance rests on a single "
                "day, at 1 every weight is zero, and above 1 they turn negative."
            )

    @property
    def assets(self) -> list[str]:
        """The universe, taken from the expected-returns keys, in insertion order."""
        return list(self.expected_returns.keys())

    def benchmark_weight_map(
        self, assets: list[str] | None = None
    ) -> dict[str, float] | None:
        """The benchmark's weights, or ``None`` when it has none.

        An explicit ``benchmark_weights`` vector wins over the spec: a caller
        that supplied the numbers directly meant those numbers. Otherwise the
        spec is expanded over ``assets`` — which matters for the rule-based
        kinds, since 1/N over ten assets is a different portfolio from 1/N
        over twelve.

        Args:
            assets: The universe to expand the spec over. ``None`` uses the
                config's own.

        Returns:
            ``asset -> weight``, or ``None``. An external-index benchmark has no
            weights in the investable universe and correctly returns ``None``.

        Raises:
            BenchmarkError: If the spec names an asset or weights the universe
                does not contain.
        """
        if self.benchmark_weights:
            return {str(k): float(v) for k, v in self.benchmark_weights.items()}
        universe = list(assets) if assets else self.assets
        if not universe or not self.benchmark.has_weights:
            return None
        weights = self.benchmark.weight_vector(universe)
        return None if weights is None else {str(k): float(v) for k, v in weights.items()}

    def get_bounds(self, asset: str, default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
        """The weight bounds for one asset.

        Args:
            asset: Asset name.
            default: Returned when the config sets no bounds for this asset.
                Defaults to long-only and unlevered, ``(0.0, 1.0)``.

        Returns:
            A ``(min, max)`` pair of weights, as fractions of the book.
        """
        if asset in self.bounds:
            lo, hi = self.bounds[asset]
            return float(lo), float(hi)
        return default

    def to_dict(self) -> dict[str, Any]:
        """The whole configuration as a plain, JSON- and YAML-serializable dict.

        Nested objects — the optimizer spec, the benchmark spec, the constraint
        layers — are expanded to their own dict form, so the result round-trips
        through :meth:`from_dict` without losing anything.

        Returns:
            A mapping with one key per configuration field.
        """
        return {
            "expected_returns": dict(self.expected_returns),
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "groups": dict(self.groups),
            "group_bounds": {k: list(v) for k, v in self.group_bounds.items()},
            "constraint_layers": [lyr.to_dict() for lyr in self.constraint_layers],
            "currencies": dict(self.currencies),
            "base_currency": self.base_currency,
            "periods_per_year": self.periods_per_year,
            "covariance_method": self.covariance_method,
            "ewma_lambda": self.ewma_lambda,
            "denoise": self.denoise,
            "denoise_method": self.denoise_method,
            "denoise_alpha": self.denoise_alpha,
            "detone": self.detone,
            "expected_returns_method": self.expected_returns_method,
            "ema_span": self.ema_span,
            "market_return": self.market_return,
            "market_weights": (dict(self.market_weights) if self.market_weights else None),
            "optimizer": self.optimizer.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "benchmark_weights": (
                dict(self.benchmark_weights)
                if self.benchmark_weights is not None else None
            ),
            "max_tracking_error": self.max_tracking_error,
            "max_active_share": self.max_active_share,
            "long_only": self.long_only,
            "fully_invested": self.fully_invested,
            "leverage": self.leverage,
            "previous_weights": (
                dict(self.previous_weights) if self.previous_weights else None
            ),
            "turnover_limit": self.turnover_limit,
            "strict_mandate": self.strict_mandate,
            # Plain mappings, not ``Shock`` objects: this dict is
            # ``yaml.safe_dump``-ed by ``save_config`` and by the app's config
            # panel, and ``json.dumps``-ed by ``config_signature``. A dataclass
            # here would make every one of those raise a representer error.
            "stress": shocks_to_dicts(self.stress),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        """Build a configuration from a mapping, applying every default.

        Args:
            data: The mapping to read, typically parsed from YAML or JSON.
                ``optimizer`` may be given as a bare method name instead of a
                mapping, which is the common shorthand in a config file.

        Returns:
            A validated :class:`EngineConfig`.

        Raises:
            ConfigurationError: If the mapping carries a key this class does
                not read, or the optimizer block carries one
                :class:`OptimizerSpec` does not. A misspelt ``max_tracking_eror``
                used to load cleanly and simply not constrain anything. Also
                for a value :meth:`__post_init__` refuses — a switch that is
                neither true nor false, a non-positive ``periods_per_year``, an
                ``ewma_lambda`` outside ``(0, 1)``.
            LayerConfigurationError: If a constraint layer is malformed.
            BenchmarkError: If the benchmark block is malformed.
            StressError: If the ``stress`` block is not a list of scenario
                mappings, one of them is malformed, or two share a name.
        """
        from optimization_engine.optimizers import ConfigurationError

        if data is None:
            data = {}
        unknown = sorted(k for k in data if k not in _CONFIG_KEYS)
        if unknown:
            raise ConfigurationError(
                f"Unknown config key(s): {', '.join(unknown)}. Known keys: "
                f"{', '.join(sorted(_CONFIG_KEYS))}."
            )
        opt_raw = data.get("optimizer") or {}
        if isinstance(opt_raw, str):
            opt_raw = {"name": opt_raw}
        unknown_opt = sorted(k for k in opt_raw if k not in _OPTIMIZER_KEYS)
        if unknown_opt:
            raise ConfigurationError(
                f"Unknown optimizer key(s): {', '.join(unknown_opt)}. Known keys: "
                f"{', '.join(sorted(_OPTIMIZER_KEYS))}."
            )
        return cls(
            expected_returns=dict(data.get("expected_returns") or {}),
            bounds={k: list(v) for k, v in (data.get("bounds") or {}).items()},
            groups=dict(data.get("groups") or {}),
            group_bounds={k: list(v) for k, v in (data.get("group_bounds") or {}).items()},
            constraint_layers=list(coerce_layers(data.get("constraint_layers"))),
            currencies=dict(data.get("currencies") or {}),
            base_currency=str(data.get("base_currency", "USD")).upper(),
            periods_per_year=int(data.get("periods_per_year", 252)),
            covariance_method=str(data.get("covariance_method", "ledoit_wolf")),
            ewma_lambda=float(data.get("ewma_lambda", 0.94)),
            # The switches go in raw: ``__post_init__`` reads them with
            # ``_as_bool``, because ``bool("false")`` is ``True``.
            denoise=data.get("denoise", False),
            denoise_method=str(data.get("denoise_method", "constant_residual")),
            denoise_alpha=float(data.get("denoise_alpha", 0.0)),
            detone=int(data.get("detone", 0) or 0),
            expected_returns_method=str(
                data.get("expected_returns_method", "historical_mean")
            ),
            ema_span=int(data.get("ema_span", 180)),
            market_return=(
                float(data["market_return"])
                if data.get("market_return") is not None else None
            ),
            market_weights=(
                dict(data["market_weights"])
                if data.get("market_weights") else None
            ),
            # Copied, so the config does not share risk_budget, the views or
            # ``extra`` with the mapping it was read from — editing that
            # mapping afterwards used to edit the config too.
            optimizer=OptimizerSpec(**copy.deepcopy(dict(opt_raw))),
            benchmark=BenchmarkSpec.from_dict(data.get("benchmark")),
            benchmark_weights=(
                dict(data["benchmark_weights"])
                if data.get("benchmark_weights") is not None else None
            ),
            max_tracking_error=(
                float(data["max_tracking_error"])
                if data.get("max_tracking_error") is not None
                else None
            ),
            max_active_share=(
                float(data["max_active_share"])
                if data.get("max_active_share") is not None
                else None
            ),
            long_only=data.get("long_only", True),
            fully_invested=data.get("fully_invested", True),
            leverage=(
                float(data["leverage"]) if data.get("leverage") is not None else None
            ),
            previous_weights=(
                dict(data["previous_weights"]) if data.get("previous_weights") else None
            ),
            turnover_limit=(
                float(data["turnover_limit"])
                if data.get("turnover_limit") is not None
                else None
            ),
            strict_mandate=data.get("strict_mandate", False),
            stress=shocks_from_dicts(data.get("stress")),
        )


def load_config(path: str | Path) -> EngineConfig:
    """Load an :class:`EngineConfig` from a YAML or JSON file.

    Args:
        path: The file to read. The format follows the extension —
            ``.yaml``/``.yml`` or ``.json``.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If the extension is neither YAML nor JSON.
        FileNotFoundError: If the path does not exist.
        LayerConfigurationError: If a constraint layer is malformed.
        BenchmarkError: If the benchmark block is malformed.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")
    return EngineConfig.from_dict(data)


def save_config(config: EngineConfig, path: str | Path) -> None:
    """Persist an :class:`EngineConfig` to YAML or JSON depending on extension.

    Args:
        config: The configuration to write.
        path: Destination. ``.json`` writes JSON; anything else writes YAML.
            Parent directories are not created.
    """
    p = Path(path)
    data = config.to_dict()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".yaml", ".yml"}:
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
