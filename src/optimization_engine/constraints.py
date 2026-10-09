"""Layered (multi-level) allocation constraints.

A real multi-asset mandate is not one flat list of caps. It reads:

    no more than 60% equity, 30% fixed income, 10% commodities;
    inside equity, at most 40% developed and 20% emerging;
    at most 30% in foreign currency.

Those are three *layers* of the same portfolio, each slicing the universe a
different way. The engine's original ``groups`` / ``group_bounds`` pair could
express exactly one of them — a single partition with one budget per group —
so the second and third had to be approximated by per-asset bounds, which do
not bind on the aggregate at all.

This module generalizes that into an ordered list of
:class:`ConstraintLayer` objects. Each layer maps assets to buckets and caps
each bucket, and every layer is applied simultaneously. Nothing about the
optimization changes character: a bucket budget is a linear inequality in the
weights, so the problems stay convex and every solver in the engine keeps its
guarantees.

Two ways to express a nested limit are supported, because mandates are
written both ways and they are *different constraints*:

``basis="portfolio"``
    ``40% DM`` means 40% of the whole book. The constraint is
    ``Σ_{i∈DM} w_i ≤ 0.40``.

``basis="parent"``
    ``40% DM`` means 40% *of the equity sleeve*. The constraint is
    ``Σ_{i∈DM} w_i ≤ 0.40 · Σ_{i∈Equity} w_i`` — still linear, so it is a
    hard constraint rather than a post-hoc check, and it moves with the
    equity allocation the optimizer chooses.

The parent bucket of a child bucket is *derived* from the layers rather than
typed twice: the assets in "DM" all live in "Equity" one layer up, so the
mapping is read off the assignments. When they do not agree, that is a
configuration error and the feasibility report names it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from optimization_engine.universe.classification import Classification

#: Bucket limits are read as a share of the total portfolio.
BASIS_PORTFOLIO = "portfolio"
#: Bucket limits are read as a share of the parent layer's bucket.
BASIS_PARENT = "parent"
BASES = (BASIS_PORTFOLIO, BASIS_PARENT)

#: Name given to the layer synthesized from the legacy ``groups`` mapping.
LEGACY_LAYER_NAME = "Asset class"

#: Weight drift below this is floating-point noise, not a breach.
DEFAULT_TOLERANCE = 1e-6

#: A bucket this close to a limit is being held there by it. Wider than
#: DEFAULT_TOLERANCE on purpose: a conic solver lands a few basis points
#: inside an active constraint, and reporting that bucket as unconstrained
#: hides the very limit that produced the allocation.
BINDING_TOLERANCE = 5e-4


class LayerConfigurationError(ValueError):
    """A layer is malformed in a way no solve can recover from."""


@dataclass
class ConstraintLayer:
    """One level of a hierarchical allocation policy.

    Attributes:
        name: Human-readable label, unique within a layer list. It is what the
            feasibility report, the compliance panel and the exposure table
            call this level of the policy.
        assignments: ``asset -> bucket``. Assets left out are simply not
            covered by this layer, which is deliberate: an FX layer has
            nothing to say about a cash line quoted in the base currency.
        limits: ``bucket -> (min, max)``. Only buckets named here are
            constrained; a bucket that exists in ``assignments`` but not here
            is reported in the exposure table and left free.
        basis: ``"portfolio"`` (limits are shares of the whole book) or
            ``"parent"`` (shares of the parent layer's bucket).
        parent: Name of the layer this one nests inside. Required when
            ``basis == "parent"``; optional otherwise, where it only enriches
            the exposure report.
    """

    name: str
    assignments: dict[str, str] = field(default_factory=dict)
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    basis: str = BASIS_PORTFOLIO
    parent: str | None = None

    def __post_init__(self) -> None:
        """Normalize the layer and reject one no solve could use.

        Assignments are stringified and pruned of the empty placeholders a
        spreadsheet or a UI leaves behind — a blank cell, or an em dash — so an
        asset nobody assigned is genuinely uncovered rather than assigned to a
        bucket named ``"—"``.

        Raises:
            LayerConfigurationError: On an unknown ``basis``, or a layer that
                expresses its limits as a share of its parent while naming no
                parent layer.
        """
        self.name = str(self.name).strip()
        self.basis = str(self.basis).lower().strip()
        if self.basis not in BASES:
            raise LayerConfigurationError(
                f"Layer {self.name!r}: basis must be one of {BASES}; "
                f"got {self.basis!r}."
            )
        self.assignments = {
            str(a): str(b)
            for a, b in dict(self.assignments).items()
            if b is not None and str(b).strip() != "" and str(b) != "—"
        }
        self.limits = {
            str(b): (float(v[0]), float(v[1]))
            for b, v in dict(self.limits).items()
        }
        self.parent = None if not self.parent else str(self.parent).strip()
        if self.basis == BASIS_PARENT and not self.parent:
            raise LayerConfigurationError(
                f"Layer {self.name!r} expresses its limits as a share of its "
                "parent but names no parent layer."
            )

    # -- structure ----------------------------------------------------------

    @property
    def is_relative(self) -> bool:
        """Whether the limits are shares of the parent rather than of the book."""
        return self.basis == BASIS_PARENT

    @property
    def is_active(self) -> bool:
        """Whether this layer constrains anything at all."""
        return bool(self.limits) and bool(self.assignments)

    def buckets(self) -> list[str]:
        """Every bucket this layer knows about, limits first then assignments."""
        out = list(self.limits.keys())
        for bucket in self.assignments.values():
            if bucket not in out:
                out.append(bucket)
        return out

    def members(self, assets: Sequence[str]) -> dict[str, list[str]]:
        """``bucket -> member assets``, restricted to ``assets``.

        Args:
            assets: The universe to restrict to.

        Returns:
            One entry per bucket that has at least one member in ``assets``.
        """
        out: dict[str, list[str]] = {}
        for asset in assets:
            bucket = self.assignments.get(str(asset))
            if bucket is not None:
                out.setdefault(bucket, []).append(str(asset))
        return out

    def member_indices(self, assets: Sequence[str]) -> dict[str, list[int]]:
        """``bucket -> positions in ``assets``, for the constrained buckets only.

        Args:
            assets: The universe, in the order the solve indexes it.

        Returns:
            Positions into ``assets``, one entry per bucket that carries a limit.
            Buckets without limits are left out: they generate no constraint.
        """
        out: dict[str, list[int]] = {}
        for i, asset in enumerate(assets):
            bucket = self.assignments.get(str(asset))
            if bucket is not None and bucket in self.limits:
                out.setdefault(bucket, []).append(i)
        return out

    def covers_all(self, assets: Sequence[str]) -> bool:
        """Whether every asset is assigned to a *constrained* bucket.

        Args:
            assets: The universe to check.

        Returns:
            ``True`` when no asset escapes this layer's limits. A layer that does
            not cover everything is still valid — the uncovered assets are simply
            unconstrained by it.
        """
        return all(
            self.assignments.get(str(a)) in self.limits for a in assets
        )

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The layer as a plain, YAML-serializable dict.

        Returns:
            Its name, assignments, limits as ``[lo, hi]`` lists, basis and parent.
            Round-trips through :meth:`from_dict`.
        """
        return {
            "name": self.name,
            "assignments": dict(self.assignments),
            "limits": {b: [float(lo), float(hi)] for b, (lo, hi) in self.limits.items()},
            "basis": self.basis,
            "parent": self.parent,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstraintLayer:
        """Build a layer from a mapping, accepting the older spellings.

        ``limits`` may also arrive as ``bounds``, and ``assignments`` as
        ``groups``, so a config written against an earlier version still loads.

        Args:
            data: The mapping to read.

        Returns:
            A validated :class:`ConstraintLayer`.

        Raises:
            LayerConfigurationError: If the result is malformed.
        """
        limits_raw = data.get("limits") or data.get("bounds") or {}
        limits = {
            str(b): (float(v[0]), float(v[1])) for b, v in dict(limits_raw).items()
        }
        return cls(
            name=str(data.get("name") or "Layer"),
            assignments=dict(data.get("assignments") or data.get("groups") or {}),
            limits=limits,
            basis=str(data.get("basis") or BASIS_PORTFOLIO),
            parent=data.get("parent"),
        )

    @classmethod
    def from_classification(
        cls,
        classification: Classification,
        as_of: Any,
        limits: Mapping[str, Any],
        *,
        name: str | None = None,
        basis: str = BASIS_PORTFOLIO,
        parent: str | None = None,
    ) -> ConstraintLayer:
        """Build a layer from a classification **as it stood on a date**.

        This is the point-in-time sibling of :func:`layer_from_mapping`: it
        resolves the assignments through
        :meth:`~optimization_engine.universe.classification.Classification.assignments`,
        so a sector cap applied in 2019 uses the 2019 sector map rather than
        today's. A name reclassified since — or not yet classified at all on
        that date — lands in the bucket it was actually in, or in none.

        The layer that comes back is a **snapshot**, not a moving object: it
        holds the assignments of one date. A backtest that wants the mapping
        to move builds one layer per decision date.

        Args:
            classification: Any object exposing ``assignments(as_of)``, which
                is what :class:`~optimization_engine.universe.classification.Classification`
                provides.
            as_of: The date to read the classification as of. Required when
                the classification is dated; ``None`` is accepted for a static
                one.
            limits: ``bucket -> (min, max)`` or ``bucket -> max``, as fractions
                of the book (or of the parent bucket under
                ``basis="parent"``). A bare number is read as a cap with a zero
                floor, exactly as in :func:`layer_from_mapping`.
            name: The layer's label. Defaults to the classification's own
                ``name``, or ``"Classification"``.
            basis: ``"portfolio"`` or ``"parent"``.
            parent: The layer these limits are relative to, required under
                ``basis="parent"``.

        Returns:
            The :class:`ConstraintLayer` for that date.

        Raises:
            LayerConfigurationError: On an unknown ``basis``, or a
                parent-relative layer naming no parent.
            UniverseError: If the classification is dated and ``as_of`` is
                ``None`` — see
                :meth:`~optimization_engine.universe.classification.Classification.label`.
        """
        label = name or str(getattr(classification, "name", "") or "Classification")
        return layer_from_mapping(
            label,
            classification.assignments(as_of),
            limits,
            basis=basis,
            parent=parent,
        )


# ---------------------------------------------------------------------------
# Layer lists
# ---------------------------------------------------------------------------


def coerce_layers(raw: Iterable[Any] | None) -> tuple[ConstraintLayer, ...]:
    """Accept layers as objects or as the mappings a YAML config round-trips.

    Args:
        raw: Layers as :class:`ConstraintLayer` objects, as mappings, or a
            mix. ``None`` yields an empty tuple.

    Returns:
        The layers, in order.

    Raises:
        LayerConfigurationError: If an entry is neither a mapping nor a
            :class:`ConstraintLayer`, or a mapping entry is malformed.
    """
    if not raw:
        return ()
    out: list[ConstraintLayer] = []
    for entry in raw:
        if isinstance(entry, ConstraintLayer):
            out.append(entry)
        elif isinstance(entry, Mapping):
            out.append(ConstraintLayer.from_dict(entry))
        else:
            raise LayerConfigurationError(
                f"A constraint layer must be a mapping or a ConstraintLayer; "
                f"got {type(entry).__name__}."
            )
    return tuple(out)


def legacy_group_layer(
    groups: Mapping[str, str] | None,
    group_bounds: Mapping[str, Any] | None,
    name: str = LEGACY_LAYER_NAME,
) -> ConstraintLayer | None:
    """Wrap the flat ``groups`` / ``group_bounds`` pair as a layer.

    Keeps every existing config, script and saved scenario working: the single
    grouping they already express becomes the first layer, and anything the
    user adds stacks on top of it.

    Args:
        groups: ``asset -> group`` mapping, or ``None``.
        group_bounds: ``group -> (min, max)`` or ``group -> max``, as
            fractions of the book, or ``None``.
        name: What to call the resulting layer.

    Returns:
        The layer, or ``None`` when there is nothing to wrap.
    """
    if not groups or not group_bounds:
        return None
    limits = {
        str(g): (float(v[0]), float(v[1])) for g, v in dict(group_bounds).items()
    }
    return ConstraintLayer(
        name=name, assignments=dict(groups), limits=limits, basis=BASIS_PORTFOLIO
    )


def resolve_parent(
    layer: ConstraintLayer, layers: Sequence[ConstraintLayer]
) -> ConstraintLayer | None:
    """The layer ``layer.parent`` names, or ``None`` when it names nothing real.

    Args:
        layer: The layer whose parent to find.
        layers: The whole policy to look in.

    Returns:
        The named parent layer, or ``None`` if no layer carries that name.
    """
    if not layer.parent:
        return None
    for candidate in layers:
        if candidate.name == layer.parent and candidate is not layer:
            return candidate
    return None


def parent_bucket_map(
    layer: ConstraintLayer,
    parent: ConstraintLayer,
    assets: Sequence[str] | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Derive ``child bucket -> parent bucket`` from the two layers' assignments.

    Args:
        layer: The child layer, whose buckets are being mapped.
        parent: The layer its limits are expressed as a share of.
        assets: Restrict the mapping to this universe. ``None`` uses every
            asset the two layers assign.

    Returns:
        A ``(mapping, ambiguous)`` pair. ``ambiguous`` lists any child bucket
        whose members straddle more than one parent bucket, with the parents
        they straddle. Ambiguity is a real modelling error — "40% of the
        parent" has no meaning when the members sit in two different parents —
        so it is returned rather than silently resolved by majority.
    """
    universe = list(assets) if assets is not None else sorted(layer.assignments)
    seen: dict[str, list[str]] = {}
    for asset in universe:
        child = layer.assignments.get(str(asset))
        if child is None:
            continue
        up = parent.assignments.get(str(asset))
        if up is None:
            continue
        bucket_parents = seen.setdefault(child, [])
        if up not in bucket_parents:
            bucket_parents.append(up)
    mapping = {c: ps[0] for c, ps in seen.items() if len(ps) == 1}
    ambiguous = {c: ps for c, ps in seen.items() if len(ps) > 1}
    return mapping, ambiguous


def effective_layers(constraints_or_config: Any) -> tuple[ConstraintLayer, ...]:
    """Every layer that applies, legacy grouping included and first.

    Accepts anything carrying ``groups``/``group_bounds`` and optionally
    ``constraint_layers`` — both :class:`~optimization_engine.config.EngineConfig`
    and :class:`~optimization_engine.optimizers.base.PortfolioConstraints`
    qualify — so the solver, the projection, the diagnostics and the
    feasibility report all read the policy from one place and cannot disagree
    about what it says.

    Args:
        constraints_or_config: Any object with ``groups`` and ``group_bounds``,
            and optionally ``constraint_layers``.

    Returns:
        Every active layer, with the legacy ``groups``/``group_bounds`` pair
        rendered as a layer and placed first.

    Raises:
        LayerConfigurationError: If a layer names a parent that does not
            exist, or has buckets spanning more than one parent bucket.
    """
    obj = constraints_or_config
    layers: list[ConstraintLayer] = []
    legacy = legacy_group_layer(
        getattr(obj, "groups", None), getattr(obj, "group_bounds", None)
    )
    explicit = coerce_layers(getattr(obj, "constraint_layers", None))
    if legacy is not None and not any(lyr.name == legacy.name for lyr in explicit):
        layers.append(legacy)
    layers.extend(explicit)
    return tuple(layers)


def has_layer_constraints(constraints_or_config: Any) -> bool:
    """Whether anything at all is constrained above the per-asset level.

    Args:
        constraints_or_config: Any object with ``groups`` and ``group_bounds``,
            and optionally ``constraint_layers``.

    Returns:
        ``True`` when at least one layer carries an active limit.
    """
    return any(lyr.is_active for lyr in effective_layers(constraints_or_config))


# ---------------------------------------------------------------------------
# CVXPY translation
# ---------------------------------------------------------------------------


def _membership_matrix(
    assets: Sequence[str],
    assignments: Mapping[str, str],
    buckets: Sequence[str],
) -> np.ndarray:
    """0/1 rows selecting each bucket's members out of ``assets``.

    Rows follow ``buckets`` and may repeat: a relative layer asks for one
    parent row per *child*, and two children of the same parent legitimately
    produce the same row.
    """
    row_of: dict[str, list[int]] = {}
    for row, bucket in enumerate(buckets):
        row_of.setdefault(bucket, []).append(row)
    matrix = np.zeros((len(buckets), len(assets)))
    for col, asset in enumerate(assets):
        for row in row_of.get(assignments.get(str(asset), ""), ()):
            matrix[row, col] = 1.0
    return matrix


def layer_cvxpy_constraints(
    weights,
    assets: Sequence[str],
    layers: Sequence[ConstraintLayer],
    scale=None,
):
    """Translate every layer into CVXPY inequalities on ``weights``.

    Emitted as one matrix inequality per layer per side rather than one scalar
    inequality per bucket. Canonicalization cost in CVXPY scales with the
    *number* of constraint objects, and a three-layer policy over a dozen
    buckets would otherwise add fifty of them to every solve — which is paid
    again on each point of a frontier sweep and each window of a walk-forward.

    Args:
        weights: The decision variable. Either the weights themselves or the
            unnormalized ray ``y`` of the homogeneous ``w = y/κ``
            reformulations used by max-Sharpe and max-diversification.
        assets: Column order ``weights`` is indexed by.
        layers: The policy, typically from :func:`effective_layers`.
        scale: ``None`` in weight space; the ``κ`` variable in ray space.
            Portfolio-basis limits need it (``Σ w ≤ hi`` becomes
            ``Σ y ≤ hi·κ``). Parent-basis limits do not: both sides are
            homogeneous of degree one, so ``Σ_child y ≤ hi·Σ_parent y`` is the
            same statement in either space — which is why a nested mandate
            survives the tangency portfolio's change of variables intact.

    Returns:
        A list of CVXPY constraints, empty when nothing is constrained.

    Raises:
        LayerConfigurationError: When a relative layer names a parent that
            does not exist, or one of its buckets straddles two parents.
    """
    cons: list[Any] = []
    layer_list = list(layers)

    for layer in layer_list:
        members = layer.member_indices(assets)
        buckets = [b for b in layer.limits if members.get(b)]
        if not buckets:
            continue

        if not layer.is_relative:
            lo = np.array([float(layer.limits[b][0]) for b in buckets])
            hi = np.array([float(layer.limits[b][1]) for b in buckets])
            held = _membership_matrix(assets, layer.assignments, buckets) @ weights
            reference = 1.0 if scale is None else scale
            cons.append(held <= hi * reference)
            if (lo > 0).any():
                cons.append(held >= lo * reference)
            continue

        parent = resolve_parent(layer, layer_list)
        if parent is None:
            raise LayerConfigurationError(
                f"Layer {layer.name!r} is expressed as a share of "
                f"{layer.parent!r}, but no layer by that name exists."
            )
        child_to_parent, ambiguous = parent_bucket_map(layer, parent, assets)
        if ambiguous:
            bad = ", ".join(
                f"{c} (spans {', '.join(ps)})" for c, ps in ambiguous.items()
            )
            raise LayerConfigurationError(
                f"Layer {layer.name!r} is expressed as a share of "
                f"{parent.name!r}, but these buckets sit in more than one "
                f"parent: {bad}. Split them, or switch the layer to "
                "percent-of-portfolio."
            )
        # Parent membership is *not* restricted to the parent's constrained
        # buckets: "40% of equity" is 40% of everything in equity, whether or
        # not the equity bucket itself carries a cap. A child whose parent
        # holds nothing in this universe is dropped — the membership already
        # forces it to zero, and a positive floor on it is caught by the
        # feasibility report rather than posed here as ``0 ≥ lo``.
        parent_members = parent.members(assets)
        buckets = [
            b for b in buckets if parent_members.get(child_to_parent.get(b, ""))
        ]
        if not buckets:
            continue
        parents = [child_to_parent[b] for b in buckets]
        lo = np.array([float(layer.limits[b][0]) for b in buckets])
        hi = np.array([float(layer.limits[b][1]) for b in buckets])
        child = _membership_matrix(assets, layer.assignments, buckets)
        above = _membership_matrix(assets, parent.assignments, parents)

        # (child − hi·parent)·w ≤ 0 and (lo·parent − child)·w ≤ 0. Both sides
        # are homogeneous of degree one, so no ``scale`` appears and the same
        # rows are correct in weight space and in ray space alike.
        cons.append((child - hi[:, None] * above) @ weights <= 0)
        if (lo > 0).any():
            rows = np.flatnonzero(lo > 0)
            cons.append(
                (lo[rows, None] * above[rows, :] - child[rows, :]) @ weights <= 0
            )
    return cons

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _bucket_totals(
    weights: pd.Series, layer: ConstraintLayer
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for asset, w in weights.items():
        bucket = layer.assignments.get(str(asset))
        if bucket is not None:
            totals[bucket] = totals.get(bucket, 0.0) + float(w)
    return totals


def layer_exposures(
    weights: pd.Series, layers: Sequence[ConstraintLayer]
) -> pd.DataFrame:
    """Realized exposure of every bucket against its limits.

    One row per bucket, with the limits restated as portfolio shares even for
    a relative layer — "40% of a 55% equity sleeve" is a 22% cap on the book,
    and that is the number an allocator compares against the weights.

    Args:
        weights: Portfolio weights, as fractions of the book.
        layers: The policy to measure against, normally from
            :func:`effective_layers`.

    Returns:
        A frame with one row per bucket and the columns ``layer``, ``bucket``,
        ``basis``, ``parent``, ``weight``, ``min`` and ``max`` (as specified),
        ``effective_min`` and ``effective_max`` (as portfolio shares),
        ``headroom`` (distance to the cap, negative when breached) and
        ``binding``.
    """
    rows: list[dict[str, Any]] = []
    layer_list = list(layers)
    assets = [str(a) for a in weights.index]
    for layer in layer_list:
        totals = _bucket_totals(weights, layer)
        parent = resolve_parent(layer, layer_list) if layer.parent else None
        mapping: dict[str, str] = {}
        parent_totals: dict[str, float] = {}
        if parent is not None:
            mapping, _ = parent_bucket_map(layer, parent, assets)
            parent_totals = _bucket_totals(weights, parent)
        for bucket in layer.buckets():
            weight = float(totals.get(bucket, 0.0))
            lo, hi = layer.limits.get(bucket, (float("nan"), float("nan")))
            up = mapping.get(bucket)
            if layer.is_relative and up is not None:
                base = float(parent_totals.get(up, 0.0))
                eff_lo, eff_hi = lo * base, hi * base
            else:
                eff_lo, eff_hi = lo, hi
            rows.append(
                {
                    "layer": layer.name,
                    "bucket": bucket,
                    "basis": layer.basis,
                    "parent": up or (parent.name if parent is not None else None),
                    "weight": weight,
                    "min": lo,
                    "max": hi,
                    "effective_min": eff_lo,
                    "effective_max": eff_hi,
                    "headroom": (
                        float(eff_hi - weight) if np.isfinite(eff_hi) else float("nan")
                    ),
                    # A zero floor that a zero weight "meets" is not the
                    # policy shaping the portfolio, so only a positive floor
                    # counts as binding from below.
                    "binding": bool(
                        np.isfinite(eff_hi)
                        and abs(eff_hi - weight) <= BINDING_TOLERANCE
                    )
                    or bool(
                        np.isfinite(eff_lo)
                        and eff_lo > 1e-9
                        and abs(weight - eff_lo) <= BINDING_TOLERANCE
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "layer", "bucket", "basis", "parent", "weight", "min", "max",
            "effective_min", "effective_max", "headroom", "binding",
        ],
    )


def layer_breaches(
    weights: pd.Series,
    layers: Sequence[ConstraintLayer],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[tuple[str, str, float, float]]:
    """Breached bucket limits as ``(label, side, limit, actual)`` tuples.

    Kept free of the diagnostics module's types so this module stays importable
    on its own; :mod:`optimization_engine.optimizers.diagnostics` wraps the
    tuples into :class:`ConstraintViolation` objects.

    Args:
        weights: The solved weights, as fractions of the book.
        layers: The policy to check, normally from :func:`effective_layers`.
        tolerance: How far past a limit an allocation may sit before it counts
            as a breach, in weight units. Absorbs solver noise.

    Returns:
        One tuple per breach: the bucket's label, which side was breached, the
        limit as a portfolio share, and the realized exposure. Empty when the
        allocation is compliant.
    """
    out: list[tuple[str, str, float, float]] = []
    exposures = layer_exposures(weights, layers)
    for _, row in exposures.iterrows():
        lo, hi = row["effective_min"], row["effective_max"]
        actual = float(row["weight"])
        label = f"{row['layer']} | {row['bucket']}"
        if pd.notna(lo) and actual < float(lo) - tolerance:
            out.append((f"{label} lower bound", "min", float(lo), actual))
        if pd.notna(hi) and actual > float(hi) + tolerance:
            out.append((f"{label} upper bound", "max", float(hi), actual))
    return out


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def layer_from_mapping(
    name: str,
    assignments: Mapping[str, str],
    limits: Mapping[str, Any],
    basis: str = BASIS_PORTFOLIO,
    parent: str | None = None,
) -> ConstraintLayer:
    """Build a layer from plain dicts, accepting ``(lo, hi)`` or ``hi`` limits.

    A bare number is read as a cap with a zero floor, which is how allocation
    policies are usually written ("no more than 60% equity") and saves the
    caller from typing the floor they did not mean to set.

    Args:
        name: The layer's label, unique within a policy.
        assignments: ``asset -> bucket``. Assets left out are not covered by
            this layer, which is deliberate rather than an error.
        limits: ``bucket -> (min, max)`` or ``bucket -> max``, as fractions.
        basis: Whether the limits are shares of the whole book or of the
            parent bucket.
        parent: The layer the limits are relative to. Required when ``basis``
            is the parent one.

    Returns:
        The :class:`ConstraintLayer`.

    Raises:
        LayerConfigurationError: On an unknown ``basis``, or a parent-relative
            layer naming no parent.
    """
    parsed: dict[str, tuple[float, float]] = {}
    for bucket, value in dict(limits).items():
        if isinstance(value, (int, float)):
            parsed[str(bucket)] = (0.0, float(value))
        else:
            lo, hi = value
            parsed[str(bucket)] = (float(lo), float(hi))
    return ConstraintLayer(
        name=name,
        assignments=dict(assignments),
        limits=parsed,
        basis=basis,
        parent=parent,
    )


def currency_layer(
    name: str,
    currencies: Mapping[str, str],
    base_currency: str,
    local_max: float | None = None,
    foreign_max: float | None = None,
    local_min: float = 0.0,
    foreign_min: float = 0.0,
    local_label: str | None = None,
    foreign_label: str = "Foreign FX",
) -> ConstraintLayer:
    """A two-bucket local/foreign FX layer derived from the currency map.

    The split is by the currency each series is *quoted in*, which is the
    exposure an allocator with a local liability actually cares about — it is
    unaffected by the base currency the engine happens to report in.

    Args:
        name: The layer's label.
        currencies: ``asset -> ISO currency code``.
        base_currency: The ISO code that counts as local.
        local_max: Cap on the local bucket, as a fraction of the book.
            ``None`` leaves it uncapped.
        foreign_max: Cap on everything else, as a fraction of the book.
        local_min: Floor on the local bucket, as a fraction of the book.
        foreign_min: Floor on the foreign bucket.
        local_label: What to call the local bucket. Defaults to the base
            currency's own code.
        foreign_label: What to call the foreign bucket.

    Returns:
        A two-bucket :class:`ConstraintLayer` over the whole book.
    """
    base = str(base_currency).upper()
    local_label = local_label or f"Local FX ({base})"
    assignments = {
        str(asset): (local_label if str(ccy).upper() == base else foreign_label)
        for asset, ccy in dict(currencies).items()
    }
    limits = {
        local_label: (float(local_min), float(1.0 if local_max is None else local_max)),
        foreign_label: (
            float(foreign_min),
            float(1.0 if foreign_max is None else foreign_max),
        ),
    }
    return ConstraintLayer(name=name, assignments=assignments, limits=limits)
