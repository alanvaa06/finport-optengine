"""The declarative description of a backtest, and its provenance hash.

Everything a run needs to be reproduced lives in one frozen object. That is
the point of separating the spec from the runner: a :class:`BacktestSpec` is
data, so it can be serialized, diffed, put in a grid, and hashed. Two runs
that carry the same ``spec_hash`` were asked the same question; two that do
not were not, however similar their charts look.

The spec is validated on construction rather than at use. A rebalance
cadence the runner does not know, a negative cost, an execution lag longer
than the sample — those are configuration errors, and the cheapest place to
find them is before any simulation has run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

RebalanceFrequency = Literal[
    "none", "daily", "weekly", "monthly", "quarterly", "annual"
]

#: Pandas offset aliases for each supported rebalance cadence. ``none`` and
#: ``daily`` are handled without resampling: one trade ever, or every period.
FREQUENCY_ALIASES: dict[str, str | None] = {
    "none": None,
    "daily": None,
    "weekly": "W",
    "monthly": "ME",
    "quarterly": "QE",
    "annual": "YE",
}

REBALANCE_DESCRIPTIONS: dict[str, str] = {
    "none": (
        "Buy and hold. Weights drift with performance; winners compound into "
        "a larger share of the book."
    ),
    "daily": "Rebalance every period. Zero drift, maximum turnover and cost.",
    "weekly": "Rebalance weekly.",
    "monthly": "Rebalance monthly - the common institutional default.",
    "quarterly": "Rebalance quarterly.",
    "annual": "Rebalance annually. Low cost, large intra-year drift.",
}


#: The smallest NAV at which an ADV-based impact charge means anything. Below
#: it a book of a few currency units cannot move any real market, so a run
#: configured that way is a mistake rather than a scenario. Inclusive: this
#: exact value is allowed, so a control offering it as its minimum cannot
#: produce a spec the validator rejects.
MIN_ADV_CAPITAL = 1_000.0

#: Where the impact model's participation rate comes from. ``"fixed"`` needs
#: no market data at all; ``"adv"`` needs a traded-volume panel and degrades
#: gracefully to the fixed rate for any asset that has none.
PARTICIPATION_SOURCES: frozenset[str] = frozenset({"fixed", "adv"})


class SpecValidationError(ValueError):
    """A spec that cannot describe a runnable backtest."""


@dataclass(frozen=True)
class CostSpec:
    """What trading costs, in three separable pieces.

    Costs are quoted as fractions of the traded notional and charged against
    NAV, so every figure here is comparable across books of any size.

    The split matters because the pieces behave differently. Commission is
    linear in what you trade and is the same whether you trade a lot or a
    little. Slippage — the half-spread you cross — is also linear, but it is
    a market property rather than a broker one. Impact is neither: it grows
    with the *square root* of participation, which is why doubling the size
    of a trade costs more than twice as much, and why an allocation that
    looks costless at one fund size stops being so at ten times that.

    Attributes:
        commission_bps: One-way broker commission, in basis points of the
            traded notional.
        slippage_bps: One-way spread cost, in basis points of the traded
            notional.
        impact_coefficient: The ``eta`` of the square-root impact law
            ``eta · sigma · sqrt(q / participation)``, where ``q`` is the
            fraction of the book traded in the name and ``sigma`` is that
            asset's per-period volatility. Zero disables impact entirely.
        impact_participation: The fraction of the book that can be traded in
            one name, in one period, without impact — the weight-space
            analogue of average daily volume. Must be positive when
            ``impact_coefficient`` is. Used directly when
            ``impact_participation_source`` is ``"fixed"``, and as the
            fallback when it is ``"adv"`` and an asset has no volume.
        impact_participation_source: Where participation comes from.
            ``"fixed"`` (the default) uses ``impact_participation`` for every
            trade and needs no volume data at all — which is what lets an
            index universe, which has no volume by construction, be
            backtested. ``"adv"`` derives it from a trailing traded-volume
            panel, so capacity is measured rather than assumed; assets with no
            volume fall back to the fixed rate and the run log says so.
        impact_adv_lookback: Trailing periods averaged when computing ADV.
        impact_adv_share: The fraction of an asset's average daily traded
            notional this book is willing to be. 0.10 means "we will not be
            more than a tenth of the day's volume in one name".
        min_adv_observations: Below this many volume observations in the
            window the ADV estimate is not used.
        impact_volatility_lookback: Trailing periods used to estimate the
            per-asset ``sigma`` that scales impact.
        min_impact_observations: Below this many trailing returns the
            volatility estimate is not trustworthy; impact degrades to zero
            for that trade and the run log says so, rather than silently
            inventing a number.
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_coefficient: float = 0.0
    impact_participation: float = 0.05
    impact_volatility_lookback: int = 63
    min_impact_observations: int = 21
    impact_participation_source: str = "fixed"
    impact_adv_lookback: int = 21
    impact_adv_share: float = 0.10
    min_adv_observations: int = 5

    def __post_init__(self) -> None:
        # Normalize on construction so that ``CostSpec(commission_bps=10)`` and
        # ``CostSpec(commission_bps=10.0)`` are the same spec, and therefore
        # carry the same hash. An int that survives into the canonical JSON
        # would make two identical runs look like different ones.
        """Coerce every rate to ``float`` and validate the combination.

        Normalizing on construction is what makes ``CostSpec(commission_bps=10)``
        and ``CostSpec(commission_bps=10.0)`` the same spec, and therefore carry
        the same hash: an ``int`` surviving into the canonical JSON would make two
        identical runs look like different ones.

        Raises:
            SpecValidationError: On a negative rate, impact enabled with a
                non-positive participation, a lookback or observation floor below
                its minimum, an unknown participation source, or an
                ``impact_adv_share`` outside ``(0, 1]``.
        """
        for name in ("commission_bps", "slippage_bps", "impact_coefficient",
                     "impact_participation", "impact_adv_share"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("impact_volatility_lookback", "min_impact_observations",
                     "impact_adv_lookback", "min_adv_observations"):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(
            self, "impact_participation_source",
            str(self.impact_participation_source).strip().lower(),
        )
        for name in ("commission_bps", "slippage_bps", "impact_coefficient"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise SpecValidationError(f"{name} must be non-negative; got {value}.")
        if self.impact_coefficient > 0.0 and self.impact_participation <= 0.0:
            raise SpecValidationError(
                "impact_participation must be positive when impact is enabled; "
                f"got {self.impact_participation}."
            )
        if self.impact_volatility_lookback < 2:
            raise SpecValidationError(
                "impact_volatility_lookback must be at least 2 periods; "
                f"got {self.impact_volatility_lookback}."
            )
        if self.min_impact_observations < 2:
            raise SpecValidationError(
                "min_impact_observations must be at least 2; "
                f"got {self.min_impact_observations}."
            )
        if self.impact_participation_source not in PARTICIPATION_SOURCES:
            raise SpecValidationError(
                "impact_participation_source must be one of "
                f"{', '.join(sorted(PARTICIPATION_SOURCES))}; "
                f"got {self.impact_participation_source!r}."
            )
        if self.impact_adv_lookback < 1:
            raise SpecValidationError(
                "impact_adv_lookback must be at least 1 period; "
                f"got {self.impact_adv_lookback}."
            )
        if self.min_adv_observations < 1:
            raise SpecValidationError(
                "min_adv_observations must be at least 1; "
                f"got {self.min_adv_observations}."
            )
        if not 0.0 < self.impact_adv_share <= 1.0:
            raise SpecValidationError(
                "impact_adv_share is the share of daily volume this book is "
                f"willing to be, so it must lie in (0, 1]; got {self.impact_adv_share}."
            )

    @property
    def total_linear_bps(self) -> float:
        """Commission plus slippage: the part of the cost that is size-blind."""
        return float(self.commission_bps) + float(self.slippage_bps)

    @property
    def has_impact(self) -> bool:
        """Whether a square-root impact term is charged at all.

        Returns:
            ``True`` when :attr:`impact_coefficient` is positive.
        """
        return float(self.impact_coefficient) > 0.0

    @property
    def uses_volume(self) -> bool:
        """Whether this cost model reads a traded-volume panel.

        False for every default, which is the point: the engine backtests
        indices, funds and anything else with no volume out of the box, and
        only asks for a volume panel when explicitly told to price capacity
        from one.
        """
        return self.has_impact and self.impact_participation_source == "adv"

    @property
    def is_free(self) -> bool:
        """Whether trading is costless under this spec.

        Returns:
            ``True`` when both linear rates are zero and impact is off. Free
            trading is honest only as a deliberate upper bound on performance.
        """
        return self.total_linear_bps == 0.0 and not self.has_impact

    def to_dict(self) -> dict[str, Any]:
        """This spec as a plain dict, one key per field.

        Returns:
            A JSON-serializable mapping suitable for a config file or a hash.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CostSpec:
        """Build a spec from a mapping, filling every absent key with its default.

        Args:
            data: Field values, or ``None`` for an all-defaults spec. Unknown keys
                are ignored, so a config carrying extra fields still loads.

        Returns:
            A validated :class:`CostSpec`.

        Raises:
            SpecValidationError: If the resulting combination is invalid.
        """
        data = dict(data or {})
        return cls(
            commission_bps=float(data.get("commission_bps", 0.0)),
            slippage_bps=float(data.get("slippage_bps", 0.0)),
            impact_coefficient=float(data.get("impact_coefficient", 0.0)),
            impact_participation=float(data.get("impact_participation", 0.05)),
            impact_volatility_lookback=int(data.get("impact_volatility_lookback", 63)),
            min_impact_observations=int(data.get("min_impact_observations", 21)),
            impact_participation_source=str(
                data.get("impact_participation_source", "fixed")
            ),
            impact_adv_lookback=int(data.get("impact_adv_lookback", 21)),
            impact_adv_share=float(data.get("impact_adv_share", 0.10)),
            min_adv_observations=int(data.get("min_adv_observations", 5)),
        )

    @classmethod
    def from_bps(cls, transaction_cost_bps: float) -> CostSpec:
        """The one-number cost model, kept for callers that only have one number.

        Args:
            transaction_cost_bps: Cost per side in basis points, charged on
                every unit of traded notional, buys and sells alike.

        Returns:
            A spec charging it all as commission, with no spread and no impact.
        """
        return cls(commission_bps=float(transaction_cost_bps))


@dataclass(frozen=True)
class BacktestSpec:
    """A complete, reproducible description of one simulation.

    Attributes:
        frequency: How often the book is traded back to its target weights.
        costs: The cost model. See :class:`CostSpec`.
        execution_lag: Bars between the date a target is stamped with and
            the bar it is first held over. Zero means a target dated ``t`` is
            traded at the close before ``t`` and earns ``t``'s return, so
            ``t`` is the first holding bar and the target may use nothing
            from it — which is how the walk-forward runner dates its
            schedule. A schedule dated by the close it was computed on needs
            one: traded on the next bar, at a price it had not seen.
        periods_per_year: Observations per year, for annualizing.
        initial_capital: Starting NAV, in the currency the prices are quoted
            in. Cosmetic for returns — they are fractions either way — but
            **not** cosmetic once impact is priced from traded volume: capacity
            is a currency amount, so the fund's size is what decides whether a
            name's average daily volume is deep or thin for this book. A run
            with ``impact_participation_source="adv"`` must therefore state a
            real one; the default of 1.0 describes a one-unit fund, for which
            every market on earth is infinitely deep.
        is_out_of_sample: Whether the weights were chosen without seeing the
            returns they are replayed on. The single most important caveat
            attached to any backtest number, so it travels with the spec and
            ends up stamped on the result.
        name: A label carried into the result and the audit log.
    """

    frequency: RebalanceFrequency = "monthly"
    costs: CostSpec = field(default_factory=CostSpec)
    execution_lag: int = 0
    periods_per_year: int = 252
    initial_capital: float = 1.0
    is_out_of_sample: bool = False
    name: str = "backtest"

    def __post_init__(self) -> None:
        """Coerce the integral fields and validate the spec.

        Raises:
            SpecValidationError: On an unknown rebalance frequency, a negative
                ``execution_lag`` (which would trade on a decision not yet taken),
                ``periods_per_year`` below 1, a non-positive ``initial_capital``,
                or volume-priced impact on a book too small for capacity to mean
                anything — a run that reported a cost near zero for that reason
                would look like evidence of infinite capacity.
        """
        object.__setattr__(self, "execution_lag", int(self.execution_lag))
        object.__setattr__(self, "periods_per_year", int(self.periods_per_year))
        object.__setattr__(self, "initial_capital", float(self.initial_capital))
        if self.frequency not in FREQUENCY_ALIASES:
            raise SpecValidationError(
                f"Unknown rebalance frequency {self.frequency!r}. "
                f"Available: {sorted(FREQUENCY_ALIASES)}"
            )
        if self.execution_lag < 0:
            raise SpecValidationError(
                f"execution_lag cannot be negative; got {self.execution_lag}. "
                "A negative lag would trade on a decision not yet taken."
            )
        if self.periods_per_year < 1:
            raise SpecValidationError(
                f"periods_per_year must be at least 1; got {self.periods_per_year}."
            )
        if self.initial_capital <= 0.0:
            raise SpecValidationError(
                f"initial_capital must be positive; got {self.initial_capital}."
            )
        if not isinstance(self.costs, CostSpec):
            object.__setattr__(self, "costs", CostSpec.from_dict(dict(self.costs)))
        if self.costs.uses_volume and self.initial_capital < MIN_ADV_CAPITAL:
            # Silently allowing this is the worst outcome available: the run
            # completes, reports a cost near zero, and looks like evidence
            # that the strategy has no capacity problem. It has no capacity
            # problem because it is a one-currency-unit fund.
            raise SpecValidationError(
                "Pricing impact from traded volume needs a real fund size: "
                f"initial_capital is {self.initial_capital:g}, which describes "
                "a book so small that every market is infinitely deep and the "
                "impact charge rounds to zero. Set initial_capital to the "
                "capital being deployed, or use "
                'impact_participation_source="fixed".'
            )

    def with_(self, **changes: Any) -> BacktestSpec:
        """A copy with fields replaced — the way a sweep builds its cells.

        Args:
            **changes: Field values to override.

        Returns:
            A new, revalidated :class:`BacktestSpec`. The original is unchanged.

        Raises:
            SpecValidationError: If the result is invalid.
        """
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """This spec as a plain dict, with the cost spec nested under ``costs``.

        Returns:
            A JSON-serializable mapping. This is what :meth:`canonical_json` and
            :attr:`spec_hash` are computed from.
        """
        return {
            "frequency": self.frequency,
            "costs": self.costs.to_dict(),
            "execution_lag": int(self.execution_lag),
            "periods_per_year": int(self.periods_per_year),
            "initial_capital": float(self.initial_capital),
            "is_out_of_sample": bool(self.is_out_of_sample),
            "name": str(self.name),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> BacktestSpec:
        """Build a spec from a mapping, filling every absent key with its default.

        Args:
            data: Field values, or ``None`` for an all-defaults spec. ``costs`` is
                handed to :meth:`CostSpec.from_dict`.

        Returns:
            A validated :class:`BacktestSpec`.

        Raises:
            SpecValidationError: If the resulting combination is invalid.
        """
        data = dict(data or {})
        return cls(
            frequency=str(data.get("frequency", "monthly")),  # type: ignore[arg-type]
            costs=CostSpec.from_dict(data.get("costs")),
            execution_lag=int(data.get("execution_lag", 0)),
            periods_per_year=int(data.get("periods_per_year", 252)),
            initial_capital=float(data.get("initial_capital", 1.0)),
            is_out_of_sample=bool(data.get("is_out_of_sample", False)),
            name=str(data.get("name", "backtest")),
        )

    def canonical_json(self) -> str:
        """Key-sorted JSON — the thing that actually gets hashed."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        """SHA-256 of the canonical form, excluding the cosmetic ``name``.

        Two specs share a hash exactly when they would produce the same
        simulation from the same data. The label is deliberately outside the
        hash: renaming a run does not make it a different run.
        """
        payload = dict(self.to_dict())
        payload.pop("name", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def describe(self) -> str:
        """One line naming the rebalance rule, the cost model and the fill lag.

        Returns:
            Something like ``"backtest: monthly rebalance, 12.0 bps linear,
            1-period execution lag (in-sample)"`` — the header a tearsheet and a
            run log both carry.
        """
        cost = (
            "costless"
            if self.costs.is_free
            else f"{self.costs.total_linear_bps:.1f} bps linear"
            + (f" + sqrt impact (eta={self.costs.impact_coefficient:g})" if self.costs.has_impact else "")
        )
        lag = "same-period fills" if self.execution_lag == 0 else f"{self.execution_lag}-period execution lag"
        sample = "out-of-sample" if self.is_out_of_sample else "in-sample"
        return f"{self.name}: {self.frequency} rebalance, {cost}, {lag} ({sample})"


__all__ = [
    "FREQUENCY_ALIASES",
    "MIN_ADV_CAPITAL",
    "PARTICIPATION_SOURCES",
    "REBALANCE_DESCRIPTIONS",
    "BacktestSpec",
    "CostSpec",
    "RebalanceFrequency",
    "SpecValidationError",
]
