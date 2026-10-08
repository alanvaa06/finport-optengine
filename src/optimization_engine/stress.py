"""What a named shock does to a book, and which position did it.

A stress test here is deliberately the simplest thing that is honest: a
one-period return shock per asset, applied to a fixed set of weights, with the
answer decomposed back onto the positions that produced it. No simulation, no
distributional assumption, no horizon — just ``P&L = Σ wᵢ·rᵢ`` and the terms
of that sum, which is why the contributions add up to the scenario's P&L
*exactly* rather than approximately.

Optionally a shock also carries a stressed covariance — a scalar multiplier on
the base matrix (every volatility scaled, correlations unchanged), a shift of
every correlation toward one, or a full replacement matrix — so the report can
say what the book's volatility becomes under the scenario, not only what it
loses on the day.

Three things this module refuses to do quietly:

* **A shock naming an asset the book cannot hold raises.** See
  :func:`stress_test`; the reasoning is there, and ``unknown_assets="ignore"``
  is the explicit way out, which records what it dropped.
* **A stress test with no shocks raises.** An empty report reads exactly like
  a passing one.
* **A covariance that does not cover the book raises.** A missing row is
  missing risk, and treating it as zero understates the very number the
  scenario exists to produce.

The units are the units of the inputs. Weights are fractions of book value,
shock returns are one-period simple returns, so a P&L of ``-0.18`` is "the
book loses 18% of its value". Volatilities come back in the units of the
covariance handed in — annualized when the matrix is annualized, which is what
:attr:`~optimization_engine.engine.EngineRun.cov_matrix` carries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, cast

import numpy as np
import pandas as pd

#: What :func:`stress_test` may be told to do about a shock naming an asset
#: that is not in the book.
UNKNOWN_ASSET_POLICIES = ("raise", "ignore")

#: Negative variance this small is solver dust and is clamped to zero; anything
#: more negative means the matrix is not a covariance and is reported as such.
_VARIANCE_FLOOR = -1e-12

#: The shock as it may be written: a scalar multiple of the base covariance, a
#: full matrix, or nothing.
CovarianceScale = Union[float, pd.DataFrame, Mapping[str, Mapping[str, float]], None]


class StressError(ValueError):
    """A stress scenario, or a book, that cannot be stressed as written.

    Subclasses :class:`ValueError`, so callers already catching that keep
    working. Raised for an unnamed shock, a non-finite shock return, a
    negative covariance multiplier, a covariance that does not cover the book,
    an asymmetric or non-PSD stressed covariance, a correlation shift outside
    ``[0, 1]``, a duplicate scenario name, an empty shock list, and — by
    default — a shock on an asset the book does not hold.
    """


# ---------------------------------------------------------------------------
# The scenario
# ---------------------------------------------------------------------------


@dataclass
class Shock:
    """One named scenario: a return shock per asset, over a single period.

    Attributes:
        name: What the scenario is called. Non-empty; it is the label every
            report line is keyed by.
        returns: ``asset -> one-period simple return``, as a fraction
            (``-0.32`` is a 32% fall). Assets the book holds that this mapping
            does not name are shocked by ``0.0`` — a scenario says what it
            says, and an unmentioned name is unmoved, not undefined.
        covariance_scale: The scenario's effect on risk, if it has one. A
            scalar multiplies the base covariance (``4.0`` doubles every
            volatility, correlations unchanged — so every book's volatility
            moves by the same ``√4``); a mapping or frame replaces it outright,
            in the same units as the base matrix, and must be positive
            semi-definite. ``None`` leaves risk at its unstressed level.
        notes: Free text — where the numbers came from, which historical
            episode they are calibrated to.
        correlation_shift: Move every pairwise correlation this fraction of
            the way to +1, keeping every volatility:
            ``ρᵢⱼ' = ρᵢⱼ + s·(1 − ρᵢⱼ)``, which is
            ``Σ' = (1 − s)·Σ + s·σσ'``. ``0`` changes nothing, ``1`` makes
            every asset the same bet. It is what a scalar cannot say —
            diversification failing — and it hurts the books that relied on
            it while leaving a single-asset book alone. A blend of two valid
            covariances, so it is always one. Applied before a scalar
            ``covariance_scale``; refused next to a replacement matrix, which
            already states its correlations. ``None`` leaves correlations
            alone.

    Raises:
        StressError: If the name is empty, a shock return is not finite, a
            scalar ``covariance_scale`` is negative or not finite, a matrix
            ``covariance_scale`` is not square, symmetric and positive
            semi-definite, or ``correlation_shift`` is outside ``[0, 1]`` or
            set next to a matrix.
    """

    name: str
    returns: Mapping[str, float]
    covariance_scale: CovarianceScale = None
    notes: str = ""
    correlation_shift: float | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the scenario as written.

        Coerces ``returns`` to a plain ``{str: float}`` dict and a matrix
        ``covariance_scale`` to a :class:`pandas.DataFrame`, so every consumer
        downstream sees one shape.

        Raises:
            StressError: As documented on the class.
        """
        self.name = str(self.name).strip()
        if not self.name:
            raise StressError("A shock needs a name; the report is keyed by it.")
        if not isinstance(self.returns, Mapping):
            raise StressError(
                f"Shock {self.name!r}: 'returns' is an asset -> return mapping; "
                f"got {type(self.returns).__name__}."
            )
        shocked: dict[str, float] = {}
        for asset, value in self.returns.items():
            r = float(value)
            if not np.isfinite(r):
                raise StressError(
                    f"Shock {self.name!r} sets {str(asset)!r} to {value!r}. A "
                    "shock is a finite one-period return."
                )
            shocked[str(asset)] = r
        self.returns = shocked
        self.notes = str(self.notes or "")
        self.covariance_scale = _normalize_covariance_scale(
            self.name, self.covariance_scale
        )
        if self.correlation_shift is not None:
            shift = float(self.correlation_shift)
            if not 0.0 <= shift <= 1.0:
                raise StressError(
                    f"Shock {self.name!r}: correlation_shift is the fraction of the "
                    f"way every correlation moves to +1, so it lies in [0, 1]; got "
                    f"{self.correlation_shift!r}."
                )
            if isinstance(self.covariance_scale, pd.DataFrame):
                raise StressError(
                    f"Shock {self.name!r}: a replacement covariance already states "
                    "its correlations, so a correlation_shift on top of it would "
                    "contradict it. Use one or the other."
                )
            self.correlation_shift = shift

    @property
    def assets(self) -> tuple[str, ...]:
        """The assets this scenario names, in the order it names them."""
        return tuple(self.returns)

    def unknown_assets(self, assets: Sequence[str]) -> tuple[str, ...]:
        """The assets this shock names that ``assets`` does not contain.

        Args:
            assets: The book's universe.

        Returns:
            The shocked names absent from it, in the shock's own order.
        """
        held = set(map(str, assets))
        return tuple(a for a in self.returns if a not in held)

    def return_vector(self, assets: Sequence[str]) -> pd.Series:
        """This shock as a dense return vector over ``assets``.

        Args:
            assets: The book's universe, in the order the answer should carry.

        Returns:
            One-period simple returns as a float :class:`pandas.Series`
            indexed by ``assets``. Names the shock does not mention are
            ``0.0``. Names the shock mentions but ``assets`` omits are
            **dropped silently here** — :func:`stress_test` is where that is
            adjudicated, so this stays a pure projection.
        """
        labels = [str(a) for a in assets]
        return pd.Series(
            [float(self.returns.get(a, 0.0)) for a in labels],
            index=pd.Index(labels),
            dtype=float,
        )

    def stressed_covariance(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """The covariance this scenario implies, over ``cov_matrix``'s assets.

        Args:
            cov_matrix: The base covariance, in whatever annualization the
                caller is working in.

        Returns:
            The base matrix when the shock carries no risk clause, the shock's
            own matrix reindexed onto the base matrix's assets, or the base
            with its correlations shifted (``correlation_shift``) and then
            multiplied by a scalar ``covariance_scale``. Same units as
            ``cov_matrix``.

        Raises:
            StressError: If a matrix ``covariance_scale`` does not cover every
                asset in ``cov_matrix``.
        """
        scale = self.covariance_scale
        if scale is None and self.correlation_shift is None:
            return cov_matrix
        if isinstance(scale, pd.DataFrame):
            assets = list(cov_matrix.columns)
            missing = [a for a in assets if a not in scale.columns or a not in scale.index]
            if missing:
                raise StressError(
                    f"Shock {self.name!r} carries a stressed covariance that does "
                    f"not cover {', '.join(map(str, missing))}. A missing row is "
                    "missing risk, not zero risk."
                )
            return scale.loc[assets, assets].astype(float)
        stressed = cov_matrix
        if self.correlation_shift is not None:
            # (1 − s)·ρσσ + s·σσ keeps the diagonal and moves each correlation
            # s of the way to one; no division, so a zero-volatility asset
            # (cash) stays at zero rather than becoming a NaN.
            vols = np.sqrt(np.clip(np.diag(cov_matrix.to_numpy(dtype=float)), 0.0, None))
            s = float(self.correlation_shift)
            stressed = (1.0 - s) * cov_matrix + s * pd.DataFrame(
                np.outer(vols, vols), index=cov_matrix.index, columns=cov_matrix.columns
            )
        # Everything that is neither None nor a frame was normalized to a
        # scalar by ``__post_init__``; a mapping never survives it.
        return stressed if scale is None else stressed * cast(float, scale)

    def to_dict(self) -> dict[str, Any]:
        """This scenario as a plain, YAML- and JSON-serializable mapping.

        Returns:
            ``name``, ``returns`` (``asset -> one-period return``), ``notes``,
            and ``covariance_scale`` — a float, a nested ``asset -> asset ->
            covariance`` mapping, or ``None`` — plus ``correlation_shift`` when
            one is set, so a shock without one serializes exactly as it always
            did. Round-trips through :meth:`from_dict`.
        """
        scale: Any = self.covariance_scale
        if isinstance(scale, pd.DataFrame):
            scale = {
                str(row): {str(col): float(scale.loc[row, col]) for col in scale.columns}
                for row in scale.index
            }
        elif scale is not None:
            scale = float(scale)
        out: dict[str, Any] = {
            "name": self.name,
            "returns": {str(k): float(v) for k, v in self.returns.items()},
            "covariance_scale": scale,
            "notes": self.notes,
        }
        if self.correlation_shift is not None:
            out["correlation_shift"] = float(self.correlation_shift)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Shock:
        """Rebuild a scenario from its serialized form.

        Args:
            data: A mapping as produced by :meth:`to_dict`. ``covariance_scale``,
                ``correlation_shift`` and ``notes`` are optional; ``name`` and
                ``returns`` are not.

        Returns:
            The :class:`Shock`.

        Raises:
            StressError: If ``name`` or ``returns`` is missing, the entry is
                not a mapping, or the scenario fails the validation
                :meth:`__post_init__` performs.
        """
        if not isinstance(data, Mapping):
            raise StressError(
                f"A shock is written as a mapping; got {type(data).__name__}."
            )
        unknown = sorted(
            set(data)
            - {"name", "returns", "covariance_scale", "correlation_shift", "notes"}
        )
        if unknown:
            raise StressError(
                f"Unknown shock key(s): {', '.join(unknown)}. Known keys: "
                "correlation_shift, covariance_scale, name, notes, returns."
            )
        if "name" not in data:
            raise StressError("A shock entry is missing required key 'name'.")
        if "returns" not in data:
            raise StressError(
                f"Shock {data['name']!r} is missing required key 'returns'."
            )
        return cls(
            name=str(data["name"]),
            returns=dict(data["returns"] or {}),
            covariance_scale=data.get("covariance_scale"),
            notes=str(data.get("notes") or ""),
            correlation_shift=data.get("correlation_shift"),
        )

    def describe(self) -> str:
        """One line: the scenario, its biggest moves, and its risk multiplier.

        Returns:
            A sentence naming up to three of the largest absolute shocks. The
            scenario's own notes are appended when it has any.
        """
        ordered = sorted(self.returns.items(), key=lambda kv: -abs(kv[1]))
        moves = ", ".join(f"{a} {r:+.1%}" for a, r in ordered[:3])
        if len(ordered) > 3:
            moves += f", +{len(ordered) - 3} more"
        risk = ""
        if isinstance(self.covariance_scale, pd.DataFrame):
            risk = "; covariance replaced"
        elif self.covariance_scale is not None:
            risk = f"; covariance ×{cast(float, self.covariance_scale):.2f}"
        if self.correlation_shift is not None:
            risk += f"; correlations {self.correlation_shift:.0%} of the way to 1"
        tail = f" — {self.notes}" if self.notes else ""
        return f"{self.name}: {moves or 'no shocks'}{risk}{tail}"


# ---------------------------------------------------------------------------
# The results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioStress:
    """What one shock did to one book.

    Attributes:
        name: The shock's name.
        pnl: The book's one-period return under the shock, as a fraction of
            book value — ``-0.18`` is an 18% loss. Equal to
            ``contributions.sum()`` by construction, not by approximation.
        contributions: ``asset -> wᵢ·rᵢ``, in the same fraction-of-book units,
            indexed like the weight vector.
        stressed_volatility: ``√(w'Σₛw)`` under the scenario's covariance, in
            the units of the covariance handed to :func:`stress_test`, or
            ``None`` when no covariance was given.
        base_volatility: The same quantity under the unstressed covariance, or
            ``None``.
        volatility_ratio: ``stressed / base``, or ``None`` when either is
            unavailable or the base volatility is zero.
        ignored_assets: Names the shock moved that the book does not hold, and
            which were therefore dropped. Only ever non-empty under
            ``unknown_assets="ignore"``; the default raises instead.
        notes: The shock's notes, carried through.
    """

    name: str
    pnl: float
    contributions: pd.Series
    stressed_volatility: float | None = None
    base_volatility: float | None = None
    volatility_ratio: float | None = None
    ignored_assets: tuple[str, ...] = ()
    notes: str = ""

    @property
    def largest_contributor(self) -> str | None:
        """The asset whose P&L contribution is largest in magnitude.

        Returns:
            The asset name, or ``None`` when the book is empty. Magnitude, not
            sign: in a loss scenario this is normally the biggest loser, but a
            position that gained more than anything else lost is genuinely the
            line that moved the number most. Ties break toward the earlier
            asset in the weight vector, so the answer is stable across runs.
        """
        if self.contributions.empty:
            return None
        return str(self.contributions.abs().idxmax())

    @property
    def largest_contribution(self) -> float | None:
        """The signed P&L of :attr:`largest_contributor`, as a fraction of book.

        Returns:
            The contribution, or ``None`` when the book is empty.
        """
        top = self.largest_contributor
        return None if top is None else float(self.contributions.loc[top])

    def describe(self) -> str:
        """One line: the scenario's P&L, its risk, and the position that drove it.

        Returns:
            A sentence in percent-of-book units, naming the largest
            contributor and — when a covariance was supplied — the stressed
            volatility and its ratio to the base.
        """
        parts = [f"{self.name}: {self.pnl:+.2%} of book value"]
        if self.stressed_volatility is not None:
            vol = f"vol {self.stressed_volatility:.2%}"
            if self.volatility_ratio is not None:
                vol += f" (×{self.volatility_ratio:.2f})"
            parts.append(vol)
        top = self.largest_contributor
        if top is not None:
            parts.append(f"largest contributor {top} {self.largest_contribution:+.2%}")
        if self.ignored_assets:
            parts.append(
                f"{len(self.ignored_assets)} shocked name(s) not held and dropped: "
                + ", ".join(self.ignored_assets)
            )
        if self.notes:
            parts.append(self.notes)
        return " | ".join(parts)


@dataclass(frozen=True)
class StressReport:
    """Every scenario applied to one book, and the worst of them.

    Attributes:
        weights: The book that was stressed, as fractions of book value.
        scenarios: One :class:`ScenarioStress` per shock, in the order the
            shocks were given.
        base_volatility: ``√(w'Σw)`` under the unstressed covariance, in that
            matrix's units, or ``None`` when no covariance was given.
        metadata: Free-form provenance — the policy the report was run under,
            and anything a caller attaches.
    """

    weights: pd.Series
    scenarios: tuple[ScenarioStress, ...]
    base_volatility: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def worst(self) -> ScenarioStress:
        """The scenario with the lowest P&L — the one the meeting is about.

        Returns:
            Its :class:`ScenarioStress`, carrying the scenario ``name``, its
            ``pnl`` and its ``largest_contributor``. Ties break toward the
            scenario given first, so the answer is stable. Never ``None``:
            :func:`stress_test` refuses an empty shock list.
        """
        return min(self.scenarios, key=lambda s: s.pnl)

    def by_severity(self) -> tuple[ScenarioStress, ...]:
        """The scenarios worst-first.

        Returns:
            The same objects ordered by ascending P&L (most negative first),
            ties in the order they were given.
        """
        return tuple(sorted(self.scenarios, key=lambda s: s.pnl))

    def to_frame(self) -> pd.DataFrame:
        """The scenario summary as a frame, worst-first.

        Returns:
            One row per scenario indexed by name, with ``pnl`` (fraction of
            book value), ``stressed_volatility`` and ``base_volatility`` (in
            the covariance's units), ``volatility_ratio``,
            ``largest_contributor``, ``largest_contribution`` and
            ``ignored_assets`` (comma-joined, empty when none).
        """
        rows = [
            {
                "pnl": s.pnl,
                "stressed_volatility": s.stressed_volatility,
                "base_volatility": s.base_volatility,
                "volatility_ratio": s.volatility_ratio,
                "largest_contributor": s.largest_contributor,
                "largest_contribution": s.largest_contribution,
                "ignored_assets": ", ".join(s.ignored_assets),
                "notes": s.notes,
            }
            for s in self.by_severity()
        ]
        index = pd.Index([s.name for s in self.by_severity()], name="scenario")
        return pd.DataFrame(rows, index=index)

    def contributions_frame(self) -> pd.DataFrame:
        """Per-asset P&L contributions, one row per scenario, worst-first.

        Returns:
            A ``scenario × asset`` frame in fraction-of-book units. Each row
            sums to that scenario's ``pnl``.
        """
        ordered = self.by_severity()
        frame = pd.DataFrame(
            [s.contributions for s in ordered],
            index=pd.Index([s.name for s in ordered], name="scenario"),
        )
        return frame.reindex(columns=list(self.weights.index))

    def describe(self) -> str:
        """The whole report as text, worst scenario first.

        Returns:
            A headline naming the worst scenario, its loss and the position
            that drove it, then one line per scenario in ascending order of
            P&L. Newline-joined.
        """
        ordered = self.by_severity()
        worst = ordered[0]
        top = worst.largest_contributor
        driver = (
            "" if top is None else f", driven by {top} ({worst.largest_contribution:+.2%})"
        )
        head = (
            f"Stressed {len(self.weights)} position(s) against "
            f"{len(ordered)} scenario(s). Worst: {worst.name} at "
            f"{worst.pnl:+.2%} of book value{driver}."
        )
        return "\n".join([head, *(f"  {s.describe()}" for s in ordered)])


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def stress_test(
    weights: pd.Series | Mapping[str, float],
    shocks: Sequence[Shock],
    cov_matrix: pd.DataFrame | None = None,
    *,
    unknown_assets: str = "raise",
) -> StressReport:
    """Apply each shock to the book and decompose what it did.

    The P&L of a scenario is ``Σ wᵢ·rᵢ`` over the book's assets, and the
    contributions reported are the terms of that sum — so they add to the P&L
    exactly, to floating-point summation error and nothing else. When a
    covariance is supplied, each scenario also reports the book's volatility
    under its stressed covariance alongside the unstressed one.

    **A shock naming an asset the book does not hold raises by default.** A
    scenario library written for a wide universe, applied to a narrow book,
    produces a loss that is quietly smaller than the scenario describes, and
    nothing in the output says so — the same defect as a Black-Litterman view
    on an asset outside the universe, which this engine also refuses. Pass
    ``unknown_assets="ignore"`` to apply it anyway; the dropped names are then
    recorded on every scenario that dropped one, reported by ``describe()``
    and carried in ``to_frame()``, so the narrowing is visible rather than
    assumed.

    Args:
        weights: The book, as ``asset -> fraction of book value``. A mapping
            is accepted and coerced. Need not sum to 1 — a levered or partly
            invested book stresses exactly as written.
        shocks: The scenarios to apply, in the order to report them. Must be
            non-empty and carry no duplicate names.
        cov_matrix: The unstressed covariance, indexed and columned by asset,
            in any consistent annualization. Supply it to get volatilities;
            omit it and every volatility field is ``None``.
        unknown_assets: ``"raise"`` (default) or ``"ignore"`` — what to do
            with a shocked name the book does not hold.

    Returns:
        A :class:`StressReport` whose ``scenarios`` are in the given order and
        whose ``worst`` is the lowest-P&L one.

    Raises:
        StressError: If ``shocks`` is empty, two shocks share a name,
            ``weights`` is empty or carries a non-finite or duplicated entry,
            ``unknown_assets`` is not one of :data:`UNKNOWN_ASSET_POLICIES`, a
            shock names an asset the book does not hold under the default
            policy, the covariance does not cover the book, or a stressed
            covariance implies a negative variance.
    """
    if unknown_assets not in UNKNOWN_ASSET_POLICIES:
        raise StressError(
            f"unknown_assets must be one of {UNKNOWN_ASSET_POLICIES}; "
            f"got {unknown_assets!r}."
        )
    book = _as_weight_series(weights)
    shock_list = list(shocks)
    if not shock_list:
        raise StressError(
            "A stress test with no shocks tests nothing, and an empty report "
            "reads exactly like a passing one. Supply at least one Shock."
        )
    seen: set[str] = set()
    for shock in shock_list:
        if not isinstance(shock, Shock):
            raise StressError(
                f"Every entry of 'shocks' must be a Shock; got {type(shock).__name__}."
            )
        if shock.name in seen:
            raise StressError(f"Duplicate shock name: {shock.name!r}")
        seen.add(shock.name)

    assets = list(book.index)
    base_cov = None if cov_matrix is None else _aligned_covariance(cov_matrix, assets)
    base_vol = None if base_cov is None else _portfolio_volatility(book, base_cov, "base")

    results: list[ScenarioStress] = []
    for shock in shock_list:
        unheld = shock.unknown_assets(assets)
        if unheld and unknown_assets == "raise":
            raise StressError(
                f"Shock {shock.name!r} moves {', '.join(unheld)}, which this book "
                "does not hold. The scenario's loss on those names cannot reach a "
                "portfolio that cannot hold them, so the P&L reported would be "
                "quietly smaller than the scenario describes. Drop them from the "
                "shock, widen the book, or pass unknown_assets='ignore' to apply "
                "it anyway and have the dropped names recorded."
            )
        contributions = book * shock.return_vector(assets)
        pnl = float(contributions.sum())

        stressed_vol: float | None = None
        ratio: float | None = None
        if base_cov is not None:
            stressed_cov = shock.stressed_covariance(base_cov)
            stressed_vol = _portfolio_volatility(book, stressed_cov, shock.name)
            if base_vol is not None and base_vol > 0.0:
                ratio = float(stressed_vol / base_vol)

        results.append(
            ScenarioStress(
                name=shock.name,
                pnl=pnl,
                contributions=contributions,
                stressed_volatility=stressed_vol,
                base_volatility=base_vol,
                volatility_ratio=ratio,
                ignored_assets=unheld if unknown_assets == "ignore" else (),
                notes=shock.notes,
            )
        )

    return StressReport(
        weights=book,
        scenarios=tuple(results),
        base_volatility=base_vol,
        metadata={
            "unknown_assets": unknown_assets,
            "has_covariance": base_cov is not None,
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_weight_series(weights: pd.Series | Mapping[str, float]) -> pd.Series:
    """Coerce a book into a float Series indexed by asset name.

    Args:
        weights: A Series or an ``asset -> weight`` mapping.

    Returns:
        A float :class:`pandas.Series` with string labels, in the given order.

    Raises:
        StressError: If it is neither, is empty, names an asset twice, or
            carries a non-finite weight.
    """
    series: pd.Series
    if isinstance(weights, pd.Series):
        series = weights.astype(float)
    elif isinstance(weights, Mapping):
        series = pd.Series({str(k): float(v) for k, v in weights.items()}, dtype=float)
    else:
        raise StressError(
            "'weights' is a Series or an asset -> weight mapping; got "
            f"{type(weights).__name__}."
        )
    if series.empty:
        raise StressError("Cannot stress an empty book.")
    series.index = pd.Index([str(a) for a in series.index])
    if series.index.has_duplicates:
        dupes = sorted({str(a) for a in series.index[series.index.duplicated()]})
        raise StressError(
            f"'weights' names the same asset twice: {', '.join(dupes)}. Which "
            "weight the shock should hit has no answer."
        )
    if not np.isfinite(series.to_numpy()).all():
        bad = sorted(str(a) for a in series.index[~np.isfinite(series.to_numpy())])
        raise StressError(f"'weights' is not finite for: {', '.join(bad)}.")
    return series


def _normalize_covariance_scale(name: str, scale: CovarianceScale) -> CovarianceScale:
    """Validate a shock's covariance clause and put it in one shape.

    Args:
        name: The shock's name, for the error messages.
        scale: ``None``, a scalar multiplier, or a matrix in either mapping or
            frame form.

    Returns:
        ``None``, a ``float``, or a square :class:`pandas.DataFrame` with
        matching index and columns.

    Raises:
        StressError: If a scalar is negative or not finite, or a matrix is not
            square, has mismatched labels, carries a non-finite entry, or is
            not symmetric and positive semi-definite.
    """
    if scale is None:
        return None
    if isinstance(scale, (int, float, np.floating, np.integer)) and not isinstance(
        scale, bool
    ):
        value = float(scale)
        if not np.isfinite(value):
            raise StressError(
                f"Shock {name!r}: covariance_scale must be finite; got {scale!r}."
            )
        if value < 0.0:
            raise StressError(
                f"Shock {name!r}: covariance_scale is {value}. A negative multiple "
                "of a covariance matrix is not a covariance matrix — variances "
                "would come out negative."
            )
        return value
    if isinstance(scale, pd.DataFrame):
        matrix = scale.astype(float)
    elif isinstance(scale, Mapping):
        matrix = pd.DataFrame.from_dict(
            {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in scale.items()},
            orient="index",
        ).astype(float)
        matrix = matrix.reindex(index=matrix.index, columns=matrix.index)
    else:
        raise StressError(
            f"Shock {name!r}: covariance_scale is a number, a matrix, or None; "
            f"got {type(scale).__name__}."
        )
    matrix.index = pd.Index([str(a) for a in matrix.index])
    matrix.columns = pd.Index([str(a) for a in matrix.columns])
    if list(matrix.index) != list(matrix.columns):
        raise StressError(
            f"Shock {name!r}: a stressed covariance is square and labelled the "
            "same way on both axes; this one has rows "
            f"{list(matrix.index)} and columns {list(matrix.columns)}."
        )
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise StressError(
            f"Shock {name!r}: the stressed covariance carries a non-finite entry."
        )
    if not np.allclose(values, values.T, rtol=1e-8, atol=1e-12):
        raise StressError(
            f"Shock {name!r}: the stressed covariance is not symmetric. "
            "Cov(a,b) and Cov(b,a) are the same number."
        )
    # Checked here, where the matrix is written, and not only per book in
    # ``_portfolio_volatility``: a book that happens not to load on the
    # negative eigenvector gets a plausible volatility out of a matrix that is
    # not a covariance. A hand-typed crisis correlation is the usual culprit.
    eigenvalues = np.linalg.eigvalsh(values)
    tolerance = 1e-10 * max(1.0, float(np.abs(eigenvalues).max()))
    if float(eigenvalues.min()) < -tolerance:
        raise StressError(
            f"Shock {name!r}: the stressed covariance is not positive "
            f"semi-definite — its smallest eigenvalue is {eigenvalues.min():.3g}, "
            "so some book would have a negative variance. Repair it (for "
            "instance with nearest_psd), or express a correlation breakdown "
            "as a correlation_shift, which is valid by construction."
        )
    return matrix


def _aligned_covariance(cov_matrix: pd.DataFrame, assets: Sequence[str]) -> pd.DataFrame:
    """Restrict a covariance to the book's assets, refusing to invent rows.

    Args:
        cov_matrix: The covariance, indexed and columned by asset.
        assets: The book's assets, in report order.

    Returns:
        The ``assets × assets`` sub-matrix, as floats.

    Raises:
        StressError: If it is not a frame, or does not cover every asset in
            the book — a missing row is missing risk, not zero risk.
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        raise StressError(
            f"'cov_matrix' is a DataFrame; got {type(cov_matrix).__name__}."
        )
    frame = cov_matrix.copy()
    frame.index = pd.Index([str(a) for a in frame.index])
    frame.columns = pd.Index([str(a) for a in frame.columns])
    missing = [a for a in assets if a not in frame.index or a not in frame.columns]
    if missing:
        raise StressError(
            f"The covariance does not cover {', '.join(missing)}. Stressing a "
            "book against a matrix that omits some of its positions reports "
            "less risk than the book carries."
        )
    return frame.loc[list(assets), list(assets)].astype(float)


def _portfolio_volatility(
    weights: pd.Series, cov_matrix: pd.DataFrame, label: str
) -> float:
    """``√(w'Σw)``, in the covariance's own units.

    Args:
        weights: The book, aligned to ``cov_matrix``.
        cov_matrix: The covariance to measure against.
        label: What to call the matrix in an error message.

    Returns:
        The volatility. Variance within :data:`_VARIANCE_FLOOR` of zero is
        clamped to zero rather than producing a NaN from ``sqrt``.

    Raises:
        StressError: If the implied variance is meaningfully negative, which
            means the matrix is not positive semi-definite.
    """
    w = weights.to_numpy(dtype=float)
    variance = float(w @ cov_matrix.to_numpy(dtype=float) @ w)
    if variance < _VARIANCE_FLOOR:
        raise StressError(
            f"The {label} covariance implies a variance of {variance:.3g} for this "
            "book, which is negative: the matrix is not positive semi-definite, "
            "so the volatility it reports would be imaginary."
        )
    return float(np.sqrt(max(variance, 0.0)))




# ---------------------------------------------------------------------------
# Serialization — what `EngineConfig` and the CLI need to carry scenarios
# ---------------------------------------------------------------------------

#: Version of the on-disk shock payload, and the key its entries live under.
SHOCKS_SCHEMA_VERSION = 1
_SHOCKS_KEY = "shocks"


def shocks_to_dicts(shocks: Sequence[Shock]) -> list[dict[str, Any]]:
    """Serialize scenarios for a config's ``to_dict``.

    Args:
        shocks: The scenarios, in order.

    Returns:
        One plain mapping per scenario, in the same order. A list rather than
        a tuple because that is what YAML and JSON round-trip.
    """
    return [shock.to_dict() for shock in shocks]


def shocks_from_dicts(raw: Any) -> tuple[Shock, ...]:
    """Rebuild scenarios from a config's ``from_dict``, or from a file.

    Args:
        raw: A sequence of mappings as produced by :func:`shocks_to_dicts`, a
            single such mapping, ``None``, or an already-built sequence of
            :class:`Shock` objects — which passes through untouched, so a
            config assembled in memory need not be serialized first.

    Returns:
        The scenarios as a tuple, empty for ``None`` or an empty sequence.

    Raises:
        StressError: If ``raw`` is not a sequence of mappings, an entry is
            malformed, or two entries share a name — a duplicate would shadow
            a scenario the author meant to run.
    """
    if raw is None:
        return ()
    if isinstance(raw, Shock):
        return (raw,)
    if isinstance(raw, Mapping):
        raw = [raw]
    if isinstance(raw, (str, bytes)):
        raise StressError(
            "Stress scenarios are a list of mappings, not a string. To load "
            "them from a file, use load_shocks(path)."
        )
    try:
        entries = list(raw)
    except TypeError as exc:
        raise StressError(
            f"Stress scenarios are a list of mappings; got {type(raw).__name__}."
        ) from exc

    shocks: list[Shock] = []
    seen: set[str] = set()
    for entry in entries:
        shock = entry if isinstance(entry, Shock) else Shock.from_dict(entry)
        if shock.name in seen:
            raise StressError(f"Duplicate shock name: {shock.name!r}")
        seen.add(shock.name)
        shocks.append(shock)
    return tuple(shocks)


def dump_shocks_yaml(shocks: Sequence[Shock]) -> str:
    """Serialize scenarios into the YAML the ``--stress`` flag reads.

    Args:
        shocks: The scenarios to write, in order.

    Returns:
        YAML carrying a ``schema_version`` and a ``shocks`` list.
    """
    import yaml

    payload = {
        "schema_version": SHOCKS_SCHEMA_VERSION,
        _SHOCKS_KEY: shocks_to_dicts(shocks),
    }
    return yaml.safe_dump(payload, sort_keys=False)


def load_shocks_yaml(text: str) -> tuple[Shock, ...]:
    """Parse a shocks document into scenarios.

    Two shapes are accepted, because both get hand-written: a mapping with an
    optional ``schema_version`` and a ``shocks`` list, which is what
    :func:`dump_shocks_yaml` writes, or a bare top-level list of scenarios.

    Args:
        text: The YAML (or JSON — YAML is a superset) document.

    Returns:
        The scenarios, in the document's order.

    Raises:
        StressError: If the payload is neither shape, carries an unsupported
            ``schema_version``, or holds a malformed or duplicated scenario.
    """
    import yaml

    data = yaml.safe_load(text)
    if data is None:
        raise StressError("The shocks document is empty.")
    if isinstance(data, list):
        return shocks_from_dicts(data)
    if not isinstance(data, dict):
        raise StressError(
            "A shocks document is a mapping with a 'shocks' list, or a bare "
            f"list of shocks; got {type(data).__name__}."
        )
    version = data.get("schema_version", SHOCKS_SCHEMA_VERSION)
    if version != SHOCKS_SCHEMA_VERSION:
        raise StressError(
            f"Unsupported shocks schema_version={version!r}; expected "
            f"{SHOCKS_SCHEMA_VERSION}."
        )
    if _SHOCKS_KEY not in data:
        raise StressError(
            "A shocks document needs a 'shocks' key. Known keys: schema_version, "
            f"shocks; got {', '.join(sorted(map(str, data))) or 'nothing'}."
        )
    entries = data.get(_SHOCKS_KEY) or []
    if not isinstance(entries, list):
        raise StressError(f"'{_SHOCKS_KEY}' must be a list.")
    return shocks_from_dicts(entries)


def load_shocks(path: str | Path) -> tuple[Shock, ...]:
    """Load stress scenarios from a YAML or JSON file.

    Args:
        path: The file to read. Both formats parse through the YAML loader,
            which reads JSON as well.

    Returns:
        The scenarios, in the file's order.

    Raises:
        StressError: If the document is malformed, as :func:`load_shocks_yaml`
            documents.
        FileNotFoundError: If the path does not exist.
    """
    return load_shocks_yaml(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "CovarianceScale",
    "SHOCKS_SCHEMA_VERSION",
    "ScenarioStress",
    "Shock",
    "StressError",
    "StressReport",
    "UNKNOWN_ASSET_POLICIES",
    "dump_shocks_yaml",
    "load_shocks",
    "load_shocks_yaml",
    "shocks_from_dicts",
    "shocks_to_dicts",
    "stress_test",
]
