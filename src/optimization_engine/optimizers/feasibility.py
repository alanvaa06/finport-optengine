"""Pre-solve feasibility analysis.

``status=infeasible`` is the least useful thing an optimizer can tell an
analyst: it names no constraint, suggests no fix, and looks identical whether
the cause is a typo in one bound or a genuinely impossible mandate.

This module answers the question the solver won't: *which* constraint makes
the problem impossible, and what has to change. It runs in two stages, and
:attr:`FeasibilityReport.stage_reached` says how far it got.

**Stage 1 — structural arithmetic, no solver.** Box capacity against the
budget (``Σ lb ≤ 1 ≤ Σ ub`` when the book must be fully invested; the minimum
gross exposure the bounds force when it must not), every layer's bucket
budgets against its members' bounds, the coherence of a nested layer against
the layer it sits inside, and bounds or layers naming assets the universe does
not hold. These answer a question no LP can: *which* constraint is impossible.
Any fatal finding stops the analysis here — a reachable-return range computed
over an empty feasible set means nothing.

**Stage 2 — the reachable-return range.** When the mandate is box-and-budget
only, the range is the fractional-knapsack closed form: sort by ``μ``, fill
from the caps down for the maximum and from the floors up for the minimum. No
solver is involved, so no solver can fail. When layers or benchmark-relative
budgets are in play the range comes from two LPs (minimise and maximise
``μ'w``) over the *same* constraint translation the optimizer uses, run
through the solver fallback chain.

The distinction stage 2 exists to protect is between *impossible* and
*unanswered*: a missing solver, an unbounded program or a numerical failure is
reported as ``solver_error`` and leaves the mandate feasible-as-far-as-we-know,
while a genuinely infeasible LP is reported as ``jointly_infeasible`` and names
the component whose removal restores a solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from optimization_engine.constraints import (
    ConstraintLayer,
    LayerConfigurationError,
    parent_bucket_map,
    resolve_parent,
)
from optimization_engine.optimizers.base import OptimizationResult, PortfolioConstraints

_TOL = 1e-9

#: :attr:`FeasibilityReport.stage_reached` when the structural stage is as far
#: as the analysis got: either a fatal finding stopped it, no expected-return
#: vector was supplied to measure a range against, or the range itself could
#: not be computed (an accompanying issue says which).
STAGE_STRUCTURAL = "structural"

#: :attr:`FeasibilityReport.stage_reached` when both stages completed and the
#: report carries a reachable-return range.
STAGE_REACHABLE_RETURN = "reachable_return"


@dataclass(frozen=True)
class FeasibilityIssue:
    """One reason the problem cannot be solved as specified."""

    code: str
    message: str
    #: Concrete, imperative suggestion the analyst can act on.
    suggestion: str
    fatal: bool = True

    @property
    def severity(self) -> str:
        """This finding's severity as a word, for printing and export.

        Returns:
            ``"fatal"`` when the finding makes the mandate impossible,
            ``"warning"`` when it leaves it solvable. The same distinction
            :attr:`fatal` carries, in the vocabulary the CLI, the workbook
            export and the ``--json`` payload use.
        """
        return "fatal" if self.fatal else "warning"


@dataclass(frozen=True)
class FeasibilityReport:
    """Result of a pre-solve feasibility analysis.

    Attributes:
        issues: Problems found, fatal ones first.
        min_return: Lowest expected return reachable under the constraints,
            in the periodicity of the expected-return vector it was measured
            against (annualized, for every caller in the engine). ``None``
            when stage 2 did not run or could not answer.
        max_return: Highest expected return reachable under the constraints,
            same units as ``min_return``.
        min_variance_return: Expected return of the minimum-variance portfolio,
            same units again; the point below which the frontier stops being
            efficient. ``None`` when no covariance was supplied or the
            minimum-variance solve failed.
        stage_reached: How far the analysis got — :data:`STAGE_STRUCTURAL` or
            :data:`STAGE_REACHABLE_RETURN`. It names the last stage that
            produced an answer, so a stage-2 attempt that ended in
            ``solver_error`` still reports ``"structural"``; the issue list
            says why.
    """

    issues: tuple[FeasibilityIssue, ...] = field(default_factory=tuple)
    min_return: float | None = None
    max_return: float | None = None
    min_variance_return: float | None = None
    stage_reached: str = STAGE_STRUCTURAL

    @property
    def reachable_return(self) -> tuple[float, float] | None:
        """The reachable expected-return range as one ``(min, max)`` pair.

        Derived from :attr:`min_return` and :attr:`max_return` rather than
        stored beside them, so the pair and the two scalars cannot disagree.

        Returns:
            ``(min_return, max_return)`` in the expected-return vector's
            units, or ``None`` when stage 2 produced no range.
        """
        if self.min_return is None or self.max_return is None:
            return None
        return (self.min_return, self.max_return)

    @property
    def is_feasible(self) -> bool:
        """Whether any allocation satisfies every constraint at once.

        Returns:
            ``True`` when no issue is fatal. Warnings do not make a mandate
            infeasible — they make it worth reading about first.
        """
        return not any(i.fatal for i in self.issues)

    @property
    def fatal_issues(self) -> tuple[FeasibilityIssue, ...]:
        """The issues that make the mandate impossible, in the order found."""
        return tuple(i for i in self.issues if i.fatal)

    @property
    def warnings(self) -> tuple[FeasibilityIssue, ...]:
        """The issues worth knowing about that still leave the mandate solvable."""
        return tuple(i for i in self.issues if not i.fatal)

    def describe(self) -> str:
        """One multi-line, human-readable explanation of everything found."""
        if not self.issues:
            return "Constraints are feasible."
        return "\n".join(
            f"* {i.message}\n  -> {i.suggestion}" for i in self.issues
        )


def _universe_issues(
    assets: list[str], constraints: PortfolioConstraints
) -> list[FeasibilityIssue]:
    """Bounds and layer assignments naming assets the universe does not hold.

    Never fatal: an instruction about an asset that is not there constrains
    nothing, so the mandate remains solvable. It is reported because the
    analyst who wrote it believes it is binding — and because the usual cause
    is a name that the data pipeline dropped for want of history, which is the
    one explanation the weights table will never offer.
    """
    known = {str(a) for a in assets}
    issues: list[FeasibilityIssue] = []

    stray_bounds = sorted(a for a in map(str, constraints.bounds) if a not in known)
    if stray_bounds:
        issues.append(
            FeasibilityIssue(
                code="bounds_outside_universe",
                message=(
                    f"{len(stray_bounds)} per-asset bound(s) name assets that are "
                    f"not in the universe: {', '.join(stray_bounds[:5])}"
                    + (" ..." if len(stray_bounds) > 5 else "")
                ),
                suggestion=(
                    "Those bounds constrain nothing. Drop them, or check "
                    "whether the assets were removed from the panel for want "
                    "of history."
                ),
                fatal=False,
            )
        )

    for layer in constraints.layers:
        stray = sorted(a for a in map(str, layer.assignments) if a not in known)
        if not stray:
            continue
        issues.append(
            FeasibilityIssue(
                code="layer_assets_outside_universe",
                message=(
                    f"Layer {layer.name!r} assigns {len(stray)} asset(s) that are "
                    f"not in the universe: {', '.join(stray[:5])}"
                    + (" ..." if len(stray) > 5 else "")
                ),
                suggestion=(
                    f"They add nothing to {layer.name!r}'s buckets. Remove them "
                    "from the layer, or check whether the panel lost them."
                ),
                fatal=False,
            )
        )
    return issues


def _budget_issues(
    constraints: PortfolioConstraints, lb: np.ndarray, ub: np.ndarray
) -> list[FeasibilityIssue]:
    """Whether the box can span whatever budget the mandate imposes.

    Under ``fully_invested`` the budget is ``Σw = 1``, so the box has to
    straddle it: ``Σ lb ≤ 1 ≤ Σ ub``. Without it there is no budget row in the
    program at all — ``Σw`` is free — and the only remaining capacity
    statement is the gross-exposure cap, which the box can violate from below:
    the smallest ``|w_i|`` the box allows is ``max(lb_i, −ub_i, 0)``, and the
    sum of those cannot exceed the cap.
    """
    issues: list[FeasibilityIssue] = []

    if constraints.fully_invested:
        if lb.sum() > 1.0 + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="min_weights_exceed_budget",
                    message=(
                        f"Minimum weights sum to {lb.sum():.2%}, which is more "
                        "than the 100% budget."
                    ),
                    suggestion=(
                        f"Lower the minimums by at least {lb.sum() - 1.0:.2%} in "
                        "total, or allow leverage."
                    ),
                )
            )
        if ub.sum() < 1.0 - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="max_weights_below_budget",
                    message=(
                        f"Maximum weights sum to only {ub.sum():.2%}, so the "
                        "portfolio cannot reach 100% invested."
                    ),
                    suggestion=(
                        f"Raise the caps by at least {1.0 - ub.sum():.2%} in total, "
                        "or add more assets to the universe."
                    ),
                )
            )

    if constraints.leverage is None:
        return issues

    cap = float(constraints.leverage)
    if constraints.fully_invested and cap < 1.0 - _TOL:
        issues.append(
            FeasibilityIssue(
                code="leverage_below_budget",
                message=(
                    f"Gross exposure is capped at {cap:.2f}x while "
                    "the portfolio must be 100% invested."
                ),
                suggestion="Set the gross-exposure cap to at least 1.0.",
            )
        )
    floor_gross = float(np.maximum(np.maximum(lb, -ub), 0.0).sum())
    if floor_gross > cap + 1e-8:
        issues.append(
            FeasibilityIssue(
                code="leverage_below_box_minimum",
                message=(
                    f"The per-asset bounds force a gross exposure of at least "
                    f"{floor_gross:.2f}x, above the {cap:.2f}x cap."
                ),
                suggestion=(
                    f"Raise the gross-exposure cap to at least {floor_gross:.2f}x, "
                    "or widen the per-asset bounds towards zero."
                ),
            )
        )
    return issues


def _structural_issues(
    assets: list[str], constraints: PortfolioConstraints
) -> list[FeasibilityIssue]:
    """Checks that need no solver: budget arithmetic on the bounds themselves."""
    issues: list[FeasibilityIssue] = []
    lb = np.array([constraints.get_bounds(a)[0] for a in assets], dtype=float)
    ub = np.array([constraints.get_bounds(a)[1] for a in assets], dtype=float)

    inverted = [a for a, lo, hi in zip(assets, lb, ub) if lo > hi + _TOL]
    if inverted:
        issues.append(
            FeasibilityIssue(
                code="inverted_bounds",
                message=(
                    f"{len(inverted)} asset(s) have a minimum weight above their "
                    f"maximum: {', '.join(inverted[:5])}"
                    + (" ..." if len(inverted) > 5 else "")
                ),
                suggestion="Fix the Min/Max columns so Min <= Max for every asset.",
            )
        )

    issues.extend(_budget_issues(constraints, lb, ub))

    if constraints.long_only and (lb < -_TOL).any():
        shorts = [a for a, lo in zip(assets, lb) if lo < -_TOL]
        issues.append(
            FeasibilityIssue(
                code="negative_bound_long_only",
                message=(
                    "Long-only is on but these assets have a negative minimum "
                    f"weight: {', '.join(shorts[:5])}"
                ),
                suggestion="Set those minimums to 0, or turn long-only off.",
            )
        )

    for layer in constraints.layers:
        issues.extend(_layer_issues(layer, constraints, assets, lb, ub))

    issues.extend(_universe_issues(assets, constraints))

    return issues


def _layer_issues(
    layer: ConstraintLayer,
    constraints: PortfolioConstraints,
    assets: list[str],
    lb: np.ndarray,
    ub: np.ndarray,
) -> list[FeasibilityIssue]:
    """Structural checks for one layer of the allocation policy.

    Each layer is a partition of (part of) the universe with a budget per
    bucket, so the same arithmetic applies whether the layer is asset class,
    sub-asset class or currency. The messages name the layer, because "Group
    'EM' is capped at 20%" is ambiguous once three layers can own a bucket
    called EM.
    """
    issues: list[FeasibilityIssue] = []
    members: dict[str, list[int]] = {}
    for i, asset in enumerate(assets):
        bucket = layer.assignments.get(str(asset))
        if bucket is not None:
            members.setdefault(bucket, []).append(i)

    where = f"{layer.name} | " if layer.name else ""
    parent_layer = resolve_parent(layer, constraints.layers) if layer.parent else None
    parent_map: dict[str, str] = {}
    if layer.is_relative and parent_layer is None:
        return [
            FeasibilityIssue(
                code="missing_parent_layer",
                message=(
                    f"Layer {layer.name!r} states its limits as a share of "
                    f"{layer.parent!r}, but there is no layer by that name."
                ),
                suggestion=(
                    "Point the layer at an existing one, or switch its "
                    "limits to percent-of-portfolio."
                ),
            )
        ]
    if parent_layer is not None:
        parent_map, ambiguous = parent_bucket_map(layer, parent_layer, assets)
        # A straddling bucket is only a modelling error when the limits are
        # *stated* as a share of the parent: "40% of the parent" has no
        # meaning across two parents, while "40% of the book" has the same
        # meaning wherever its members sit. Either way it drops out of
        # ``parent_map``, so no coherence claim is made about it below.
        if layer.is_relative:
            for bucket, parents in ambiguous.items():
                issues.append(
                    FeasibilityIssue(
                        code="ambiguous_parent_bucket",
                        message=(
                            f"{where}{bucket!r} is capped as a share of "
                            f"{parent_layer.name!r}, but its assets sit in more "
                            f"than one: {', '.join(parents)}."
                        ),
                        suggestion=(
                            f"Split {bucket!r} so all its assets share a parent, "
                            "or switch this layer to percent-of-portfolio."
                        ),
                    )
                )

    total_min = 0.0
    total_max = 0.0
    covered: set[int] = set()
    for bucket, (blo, bhi) in layer.limits.items():
        blo, bhi = float(blo), float(bhi)
        idx = members.get(bucket, [])
        if not idx:
            issues.append(
                FeasibilityIssue(
                    code="empty_group",
                    message=(
                        f"{where}{bucket!r} has limits but no assets assigned "
                        "to it."
                    ),
                    suggestion=(
                        f"Assign assets to {bucket!r}, or remove its limits."
                    ),
                    fatal=blo > _TOL,
                )
            )
            continue
        covered.update(idx)
        if blo > bhi + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="inverted_group_bounds",
                    message=(
                        f"{where}{bucket!r} has a minimum ({blo:.2%}) above its "
                        f"maximum ({bhi:.2%})."
                    ),
                    suggestion="Swap or correct that bucket's Min/Max weights.",
                )
            )

        if layer.is_relative:
            if bhi > 1.0 + 1e-8:
                issues.append(
                    FeasibilityIssue(
                        code="relative_cap_above_parent",
                        message=(
                            f"{where}{bucket!r} is capped at {bhi:.0%} of its "
                            "parent, which is more than the whole parent."
                        ),
                        suggestion=(
                            "A share-of-parent cap above 100% never binds; "
                            "lower it or switch the layer to "
                            "percent-of-portfolio."
                        ),
                        fatal=False,
                    )
                )
            continue

        member_max = float(ub[idx].sum())
        member_min = float(lb[idx].sum())
        if member_max < blo - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="group_min_unreachable",
                    message=(
                        f"{where}{bucket!r} needs at least {blo:.2%} but its "
                        f"members' caps only add up to {member_max:.2%}."
                    ),
                    suggestion=(
                        f"Raise the per-asset maximums inside {bucket!r}, or "
                        f"lower its minimum below {member_max:.2%}."
                    ),
                )
            )
        if member_min > bhi + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="group_max_unreachable",
                    message=(
                        f"{where}{bucket!r} is capped at {bhi:.2%} but its "
                        f"members' minimums already force {member_min:.2%}."
                    ),
                    suggestion=(
                        f"Lower the per-asset minimums inside {bucket!r}, or "
                        f"raise its maximum above {member_min:.2%}."
                    ),
                )
            )
        total_min += blo
        total_max += bhi

    if layer.is_relative:
        issues.extend(_relative_layer_issues(layer, parent_map, constraints, assets))
        return issues

    if parent_layer is not None:
        issues.extend(_parent_coherence_issues(layer, parent_layer, parent_map, assets))

    if constraints.fully_invested and covered and len(covered) == len(assets):
        if total_min > 1.0 + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="group_mins_exceed_budget",
                    message=(
                        f"{where}bucket minimums sum to {total_min:.2%}, more "
                        "than the 100% budget."
                    ),
                    suggestion=(
                        "Lower the minimums on this layer so they sum to at "
                        "most 100%."
                    ),
                )
            )
        if total_max < 1.0 - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="group_maxes_below_budget",
                    message=(
                        f"{where}bucket maximums sum to only {total_max:.2%}, "
                        "so the portfolio cannot be fully invested."
                    ),
                    suggestion=(
                        "Raise the maximums on this layer so they sum to at "
                        "least 100%, or leave some assets out of it."
                    ),
                )
            )
    return issues


def _parent_coherence_issues(
    layer: ConstraintLayer,
    parent_layer: ConstraintLayer,
    parent_map: dict[str, str],
    assets: list[str],
) -> list[FeasibilityIssue]:
    """Whether a nested percent-of-portfolio layer fits inside the layer above it.

    A child bucket's members are a subset of its parent bucket's, so
    ``w_child ≤ w_parent`` holds by construction. Two statements follow, and
    both are arithmetic on the limits alone:

    * the child floors inside one parent cannot exceed that parent's cap —
      "at least 25% DM and at least 20% EM" is impossible inside an equity
      sleeve capped at 30%;
    * when the children cover the parent bucket entirely, their caps cannot
      fall below the parent's floor — the sleeve could not be filled to its
      own minimum.

    Layers stating their limits as a share of the parent are handled by
    :func:`_relative_layer_issues` instead, where the same two statements are
    expressed in the parent's own units.

    Args:
        layer: The child layer, whose limits are shares of the whole book.
        parent_layer: The layer it names as its parent.
        parent_map: ``child bucket -> parent bucket``, from
            :func:`~optimization_engine.constraints.parent_bucket_map`.
        assets: The universe, in the order the solve indexes it.

    Returns:
        One issue per incoherent parent bucket, fatal in both directions.
    """
    issues: list[FeasibilityIssue] = []
    where = f"{layer.name} | " if layer.name else ""
    children_of: dict[str, list[str]] = {}
    for bucket, up in parent_map.items():
        if bucket in layer.limits:
            children_of.setdefault(up, []).append(bucket)

    parent_members = parent_layer.members(assets)
    for up, children in children_of.items():
        limits = parent_layer.limits.get(up)
        if limits is None:
            continue
        parent_floor, parent_cap = float(limits[0]), float(limits[1])
        floor_total = sum(float(layer.limits[c][0]) for c in children)
        if floor_total > parent_cap + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="child_floors_exceed_parent_cap",
                    message=(
                        f"{where}{', '.join(sorted(children))} sit inside "
                        f"{parent_layer.name} | {up!r}, which is capped at "
                        f"{parent_cap:.2%}, but their minimums already require "
                        f"{floor_total:.2%}."
                    ),
                    suggestion=(
                        f"Lower those minimums to at most {parent_cap:.2%} in "
                        f"total, or raise {up!r}'s cap above {floor_total:.2%}."
                    ),
                )
            )
        covers_parent = all(
            layer.assignments.get(str(a)) in layer.limits
            for a in parent_members.get(up, [])
        )
        cap_total = sum(float(layer.limits[c][1]) for c in children)
        if covers_parent and parent_floor > _TOL and cap_total < parent_floor - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="child_caps_below_parent_floor",
                    message=(
                        f"{where}{', '.join(sorted(children))} cover "
                        f"{parent_layer.name} | {up!r} entirely, but their caps "
                        f"add up to {cap_total:.2%} while {up!r} must hold at "
                        f"least {parent_floor:.2%}."
                    ),
                    suggestion=(
                        f"Raise those caps to at least {parent_floor:.2%} in "
                        f"total, or lower {up!r}'s minimum to {cap_total:.2%}."
                    ),
                )
            )
    return issues


def _relative_layer_issues(
    layer: ConstraintLayer,
    parent_map: dict[str, str],
    constraints: PortfolioConstraints,
    assets: list[str],
) -> list[FeasibilityIssue]:
    """Whether a percent-of-parent layer can fill its parent at all.

    Sub-buckets that cap out below 100% of the parent while covering all of it
    do not, on their own, make the problem infeasible — the parent simply
    cannot be filled — but that is almost always a typo (30/30 inside a sleeve
    the allocator means to fill), so it is reported as a warning with the
    arithmetic shown.

    It stops being a warning the moment the parent bucket carries a floor:
    caps summing to less than the whole sleeve force the sleeve to zero, and a
    sleeve that must hold at least something cannot be zero. That pair is
    fatal, and each half of it is satisfiable alone — which is exactly the kind
    of finding the LP would report as a bare ``infeasible``.
    """
    issues: list[FeasibilityIssue] = []
    if constraints.long_only is False:
        issues.append(
            FeasibilityIssue(
                code="relative_layer_shorts",
                message=(
                    f"Layer {layer.name!r} is a share of its parent while short "
                    "positions are allowed."
                ),
                suggestion=(
                    "A share-of-parent limit assumes the parent sleeve is "
                    "positive; with shorts it can invert. Use "
                    "percent-of-portfolio limits instead."
                ),
                fatal=False,
            )
        )
    by_parent: dict[str, list[str]] = {}
    for bucket, up in parent_map.items():
        if bucket in layer.limits:
            by_parent.setdefault(up, []).append(bucket)
    parent_layer = resolve_parent(layer, constraints.layers)
    for up, children in by_parent.items():
        covers_parent = parent_layer is not None and all(
            layer.assignments.get(str(a)) in layer.limits
            for a in assets
            if parent_layer.assignments.get(str(a)) == up
        )
        if not covers_parent:
            continue
        cap_total = sum(float(layer.limits[c][1]) for c in children)
        floor_total = sum(float(layer.limits[c][0]) for c in children)
        parent_floor = 0.0
        if parent_layer is not None and up in parent_layer.limits:
            parent_floor = float(parent_layer.limits[up][0])
        if cap_total < 1.0 - 1e-8 and parent_floor > _TOL:
            # w_parent = Σ_children ≤ cap_total·w_parent with cap_total < 1
            # forces w_parent = 0, and the parent's own floor forbids that.
            # Two limits that are each satisfiable alone; together, nothing.
            issues.append(
                FeasibilityIssue(
                    code="relative_caps_starve_parent_floor",
                    message=(
                        f"Inside {up!r}, the {layer.name} caps sum to "
                        f"{cap_total:.0%} of the sleeve, so the sleeve can only "
                        f"be held at zero - but {up!r} must hold at least "
                        f"{parent_floor:.2%} of the book."
                    ),
                    suggestion=(
                        f"Raise the {layer.name} caps inside {up!r} so they sum "
                        f"to at least 100%, or drop {up!r}'s minimum to zero."
                    ),
                )
            )
        elif cap_total < 1.0 - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="relative_caps_below_parent",
                    message=(
                        f"Inside {up!r}, the {layer.name} caps sum to "
                        f"{cap_total:.0%} of the sleeve, so at most that much "
                        "of it can be filled."
                    ),
                    suggestion=(
                        f"Raise the {layer.name} caps inside {up!r} so they sum "
                        "to at least 100%, or lower the cap on "
                        f"{up!r} itself to match."
                    ),
                    fatal=False,
                )
            )
        if floor_total > 1.0 + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="relative_floors_exceed_parent",
                    message=(
                        f"Inside {up!r}, the {layer.name} minimums sum to "
                        f"{floor_total:.0%} of the sleeve - more than the "
                        "sleeve itself."
                    ),
                    suggestion=(
                        f"Lower the {layer.name} minimums inside {up!r} so they "
                        "sum to at most 100%."
                    ),
                )
            )
    return issues


def reduces_to_box_and_budget(constraints: PortfolioConstraints) -> bool:
    """Whether the mandate is nothing but per-asset bounds and the budget.

    That is the case the reachable-return range has a closed form for, so it
    is worth naming: no layer carries a limit, no benchmark-relative budget is
    set, no turnover budget binds, and any gross-exposure cap is one that
    cannot bind. The last is the only non-obvious test — under ``long_only``
    every weight is non-negative, so a fully-invested book has gross exposure
    of exactly ``1``, and a cap at or above ``1`` constrains nothing.

    Args:
        constraints: The mandate to classify.

    Returns:
        ``True`` when the feasible set is exactly the box intersected with the
        budget, so :func:`knapsack_return_range` answers exactly.
    """
    if constraints.has_layer_limits:
        return False
    if (
        constraints.max_tracking_error is not None
        or constraints.max_active_share is not None
    ):
        return False
    if constraints.turnover_limit is not None and constraints.previous_weights:
        return False
    if constraints.leverage is not None:
        return bool(
            constraints.long_only
            and constraints.fully_invested
            and float(constraints.leverage) >= 1.0 - _TOL
        )
    return True


def knapsack_return_range(
    expected_returns: pd.Series,
    constraints: PortfolioConstraints,
    assets: list[str] | None = None,
) -> tuple[float, float] | None:
    """The reachable expected-return range under bounds and a budget, in closed form.

    Maximising ``μ'w`` subject to ``lb ≤ w ≤ ub`` and ``Σw = 1`` is the
    fractional knapsack: start every asset at its floor, then spend the
    remaining ``1 − Σlb`` of budget on the highest-``μ`` assets until each is
    at its cap. Reversing the sort gives the minimum. No solver is involved,
    which is the point — a range this cheap cannot fail for want of one, and
    it is exact rather than accurate to a tolerance.

    Without a budget (``fully_invested=False``) the two optima are simply the
    per-asset corners: each weight goes to whichever end of its box the sign
    of its ``μ`` prefers.

    Args:
        expected_returns: Expected returns, one per asset. The range comes
            back in these units — annualized, for every caller in the engine.
        constraints: The mandate whose bounds and budget to read. Only
            meaningful when :func:`reduces_to_box_and_budget` accepts it.
        assets: Universe order. Defaults to the returns' index.

    Returns:
        ``(min, max)`` expected return, or ``None`` when the universe is empty
        or the box cannot span the budget at all — the infeasibility the
        structural stage reports as ``min_weights_exceed_budget`` or
        ``max_weights_below_budget``.
    """
    order = list(assets) if assets is not None else list(expected_returns.index)
    if not order:
        return None
    mu = expected_returns.reindex(order).fillna(0.0).to_numpy(dtype=float)
    lb = np.array([constraints.get_bounds(a)[0] for a in order], dtype=float)
    ub = np.array([constraints.get_bounds(a)[1] for a in order], dtype=float)

    if not constraints.fully_invested:
        corners = np.stack([mu * lb, mu * ub])
        return (float(corners.min(axis=0).sum()), float(corners.max(axis=0).sum()))

    room = np.maximum(ub - lb, 0.0)
    budget = 1.0 - float(lb.sum())
    if budget < -1e-8 or room.sum() < budget - 1e-8:
        return None
    budget = max(budget, 0.0)
    base = float(mu @ lb)
    ascending = np.argsort(mu, kind="stable")
    lo = base + _fill(mu, room, ascending, budget)
    hi = base + _fill(mu, room, ascending[::-1], budget)
    return (min(lo, hi), max(lo, hi))


def _fill(
    mu: np.ndarray, room: np.ndarray, order: np.ndarray, budget: float
) -> float:
    """Spend ``budget`` of weight along ``order``, and report what it buys."""
    left = budget
    bought = 0.0
    for i in order:
        if left <= _TOL:
            break
        take = min(left, float(room[i]))
        bought += take * float(mu[i])
        left -= take
    return bought


def _lp_return_range(
    expected_returns: pd.Series,
    constraints: PortfolioConstraints,
    assets: list[str],
    cov_matrix: pd.DataFrame | None = None,
) -> tuple[tuple[float, float] | None, list[FeasibilityIssue]]:
    """Two LPs for the reachable-return range, and what to say when they fail.

    Runs through :func:`~optimization_engine.optimizers._cvxpy_helpers.solve_problem`
    so the range is computed on the same solver chain, and the same constraint
    translation, as the solve it is meant to pre-empt.

    Returns:
        A ``(range, issues)`` pair. Exactly one of the two is populated: a
        range, or the findings explaining its absence. A failure that is the
        solver's — a crash, a missing solver, a status nobody can act on — is
        reported as a non-fatal ``solver_error``, because "we could not tell"
        is not the same claim as "your mandate is impossible".
    """
    import cvxpy as cp

    from optimization_engine.optimizers._cvxpy_helpers import (
        SolverFailure,
        build_constraints,
        solve_problem,
    )

    mu = expected_returns.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    sigma = None
    if cov_matrix is not None:
        sigma = cov_matrix.reindex(assets, axis=0).reindex(assets, axis=1).values

    bounds: list[float] = []
    for sense in (cp.Minimize, cp.Maximize):
        w = cp.Variable(len(assets))
        try:
            cons = build_constraints(w, assets, constraints, cov_matrix=sigma)
        except (ValueError, LayerConfigurationError) as exc:
            return None, [
                FeasibilityIssue(
                    code="constraint_translation_error",
                    message=(
                        f"The mandate could not be written down as a program: {exc}"
                    ),
                    suggestion=(
                        "Fix the constraint the message names; the optimizer "
                        "will refuse the same mandate for the same reason."
                    ),
                )
            ]
        problem = cp.Problem(sense(mu @ w), cons)
        try:
            solve_problem(problem)
        except SolverFailure as exc:
            if exc.status == "infeasible":
                return None, [_jointly_infeasible_issue(assets, constraints, sigma)]
            if exc.status == "unbounded":
                return None, [
                    FeasibilityIssue(
                        code="reachable_range_unbounded",
                        message=(
                            "The expected return these constraints allow has no "
                            "bound, so there is no range to check a target "
                            "against."
                        ),
                        suggestion=(
                            "Add per-asset caps or a budget; the mandate itself "
                            "is satisfiable, only unbounded."
                        ),
                        fatal=False,
                    )
                ]
            return None, [_solver_error_issue(exc)]
        except Exception as exc:  # a crash inside CVXPY or a solver binding
            return None, [_solver_error_issue(exc)]
        if w.value is None:
            return None, [
                _solver_error_issue(
                    RuntimeError(f"solver returned no weights (status={problem.status})")
                )
            ]
        bounds.append(float(mu @ np.asarray(w.value).flatten()))

    lo, hi = bounds
    return (min(lo, hi), max(lo, hi)), []


def _solver_error_issue(exc: BaseException) -> FeasibilityIssue:
    """The finding for "the solver could not answer", which is not "no answer exists"."""
    return FeasibilityIssue(
        code="solver_error",
        message=(
            "The reachable-return range could not be computed: the solver "
            f"failed ({exc}). This says nothing about whether the constraints "
            "have a solution."
        ),
        suggestion=(
            "Install or repair a solver (CLARABEL, ECOS, SCS or OSQP) and run "
            "the check again. Until then the return target has not been "
            "validated against what the constraints can reach."
        ),
        fatal=False,
    )


def _jointly_infeasible_issue(
    assets: list[str],
    constraints: PortfolioConstraints,
    sigma: np.ndarray | None,
) -> FeasibilityIssue:
    """The finding for an LP that is infeasible after every structural check passed.

    Each piece of the mandate is satisfiable on its own — the structural stage
    has already proved that — so the contradiction is between them. Which pair
    is found by dropping one component at a time and re-testing, which is one
    small LP per component and only on the path that has already failed.
    """
    culprit = _binding_component(assets, constraints, sigma)
    if culprit is None:
        return FeasibilityIssue(
            code="jointly_infeasible",
            message=(
                "Every part of this mandate is satisfiable on its own, but no "
                "allocation satisfies them all at once - and dropping any "
                "single one of them does not make it solvable either."
            ),
            suggestion=(
                "Relax the mandate a piece at a time: the contradiction "
                "involves at least three of its parts, so widening any one of "
                "them alone will not clear it."
            ),
        )
    return FeasibilityIssue(
        code="jointly_infeasible",
        message=(
            f"The bounds, the budget and {culprit} are each satisfiable on their "
            "own, but no allocation satisfies them all at once."
        ),
        suggestion=(
            f"Relax {culprit} - widen its caps or lower its floors - or widen "
            "the per-asset bounds it has to work with. Removing it is what "
            "makes the rest of the mandate solvable."
        ),
    )


def _relaxations(
    constraints: PortfolioConstraints,
) -> list[tuple[str, PortfolioConstraints]]:
    """Each removable component of the mandate, paired with the mandate without it.

    The box and the budget are not on the list: they are what everything else
    is tested *against*, and the structural stage has already cleared them.
    """
    out: list[tuple[str, PortfolioConstraints]] = []
    layers = list(constraints.layers)
    for target in layers:
        kept = tuple(lyr for lyr in layers if lyr is not target)
        out.append(
            (
                f"the {target.name!r} layer",
                # ``groups``/``group_bounds`` are the first effective layer, so
                # they are cleared and every layer that survives is passed
                # through explicitly. Anything else on the mandate carries over.
                replace(constraints, groups={}, group_bounds={}, constraint_layers=kept),
            )
        )
    if constraints.turnover_limit is not None and constraints.previous_weights:
        out.append(
            ("the turnover budget", replace(constraints, turnover_limit=None))
        )
    if constraints.max_tracking_error is not None:
        out.append(
            ("the tracking-error budget", replace(constraints, max_tracking_error=None))
        )
    if constraints.max_active_share is not None:
        out.append(
            ("the active-share budget", replace(constraints, max_active_share=None))
        )
    if constraints.leverage is not None:
        out.append(("the gross-exposure cap", replace(constraints, leverage=None)))
    return out


def _binding_component(
    assets: list[str],
    constraints: PortfolioConstraints,
    sigma: np.ndarray | None,
) -> str | None:
    """Name the one component whose removal makes the mandate solvable, if there is one."""
    for label, relaxed in _relaxations(constraints):
        if _is_lp_feasible(assets, relaxed, sigma) is True:
            return label
    return None


def _is_lp_feasible(
    assets: list[str],
    constraints: PortfolioConstraints,
    sigma: np.ndarray | None,
) -> bool | None:
    """Whether any weight vector satisfies ``constraints``; ``None`` if unanswerable."""
    import cvxpy as cp

    from optimization_engine.optimizers._cvxpy_helpers import (
        SolverFailure,
        build_constraints,
        solve_problem,
    )

    w = cp.Variable(len(assets))
    try:
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(w)),
            build_constraints(w, assets, constraints, cov_matrix=sigma),
        )
        solve_problem(problem)
    except SolverFailure as exc:
        return False if exc.status == "infeasible" else None
    except Exception:
        # The probe is a diagnosis, not a verdict: when it cannot be answered
        # the component is simply not named as the culprit.
        return None
    return w.value is not None


def reachable_return_range(
    expected_returns: pd.Series,
    constraints: PortfolioConstraints,
    assets: list[str] | None = None,
    cov_matrix: pd.DataFrame | None = None,
    *,
    use_closed_form: bool = True,
) -> tuple[float, float] | None:
    """The lowest and highest expected return the constraints allow.

    This is the range a target return must fall inside. Under bounds and a
    budget alone it is the fractional-knapsack closed form
    (:func:`knapsack_return_range`); with layers or benchmark-relative budgets
    it is two LPs over the shared constraint translation, run through the
    solver fallback chain.

    Args:
        expected_returns: The vector the target is expressed in. The range
            comes back in the same units.
        constraints: The constraint set about to be solved.
        assets: Universe order. Defaults to the returns' index.
        cov_matrix: Needed only when a tracking-error budget is set — that
            budget genuinely narrows the reachable range, and omitting it
            would report a range the solve cannot deliver.
        use_closed_form: Take the closed form when the mandate allows it. Set
            ``False`` to force the LP path; the two agree to solver tolerance,
            and the test suite pins that they do.

    Returns:
        ``(min, max)`` expected return, or ``None`` when no range could be
        established — whether because the constraints are infeasible or
        because no solver could answer. :func:`analyze_feasibility`
        distinguishes those two; this function, kept for callers that only
        want the range, does not.
    """
    order = list(assets) if assets is not None else list(expected_returns.index)
    if not order:
        return None
    if use_closed_form and reduces_to_box_and_budget(constraints):
        return knapsack_return_range(expected_returns, constraints, order)
    found, _ = _lp_return_range(expected_returns, constraints, order, cov_matrix)
    return found


def min_variance_return(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> float | None:
    """Expected return of the constrained global minimum-variance portfolio.

    Frontier targets below this point are *inefficient*: the same volatility
    buys a higher return above the GMV, so plotting them as "efficient" is
    misleading.

    Args:
        expected_returns: Annualized expected returns, one per asset.
        cov_matrix: Asset covariance over the same universe.
        constraints: The mandate the GMV is solved under.

    Returns:
        The GMV portfolio's expected return, in the units of
        ``expected_returns``, or ``None`` when the solve did not produce a
        usable answer. :func:`analyze_feasibility` reports the reason as a
        ``gmv_solver_error`` warning rather than dropping it.
    """
    result, _ = _solve_gmv(expected_returns, cov_matrix, constraints)
    return None if result is None else float(result.expected_return)


def _solve_gmv(
    expected_returns: pd.Series | None,
    cov_matrix: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> tuple[OptimizationResult | None, str | None]:
    """The constrained minimum-variance portfolio, or why there isn't one.

    One solve answers two questions the report asks — where the frontier stops
    being efficient, and the lowest volatility the mandate can reach — so it
    is done once and the result shared.

    Returns:
        An ``(result, reason)`` pair. Exactly one is populated; ``reason`` is
        the failure rendered as text, never swallowed.
    """
    from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer

    try:
        return (
            MinVarianceOptimizer(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                constraints=constraints,
            ).optimize(),
            None,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def analyze_feasibility(
    assets: list[str],
    constraints: PortfolioConstraints,
    expected_returns: pd.Series | None = None,
    cov_matrix: pd.DataFrame | None = None,
    check_targets: bool = True,
) -> FeasibilityReport:
    """Full pre-solve feasibility analysis, in two stages.

    Stage 1 is arithmetic on the mandate itself and needs no solver; any fatal
    finding stops the analysis there, because a reachable-return range over an
    empty feasible set means nothing. Stage 2 measures that range — in closed
    form when the mandate is box-and-budget only, and otherwise with two LPs
    through the solver fallback chain — and checks the return and volatility
    targets against it.

    Args:
        assets: The universe, in the order the optimizer will see it.
        constraints: The constraint object about to be solved.
        expected_returns: Needed to check a return target and report the
            reachable range. Its units set the report's; annualized, for every
            caller in the engine. Without it, stage 2 does not run.
        cov_matrix: Needed to locate the minimum-variance return and to check a
            volatility target.
        check_targets: Validate ``target_return`` / ``target_volatility``
            against what the constraints can actually reach.

    Returns:
        A :class:`FeasibilityReport`. It reports a mandate as infeasible only
        when something proved it so: a structural contradiction, or an LP that
        came back ``infeasible``. A solver that crashed or was missing leaves
        the verdict open and says so in a non-fatal ``solver_error`` finding.
    """
    issues = _structural_issues(assets, constraints)
    stage = STAGE_STRUCTURAL
    lo = hi = gmv = None

    if not any(i.fatal for i in issues) and expected_returns is not None and assets:
        rng = None
        found: list[FeasibilityIssue] = []
        if reduces_to_box_and_budget(constraints):
            rng = knapsack_return_range(expected_returns, constraints, assets)
        if rng is None:
            # Either the mandate is more than box-and-budget, or the closed
            # form declined it — which can only mean the box cannot span the
            # budget, and the LP is then the thing that says so out loud
            # rather than leaving the absent range unexplained.
            rng, found = _lp_return_range(
                expected_returns, constraints, assets, cov_matrix
            )
        issues.extend(found)
        if rng is not None:
            stage = STAGE_REACHABLE_RETURN
            lo, hi = rng

    wants_vol_floor = check_targets and constraints.target_volatility is not None
    gmv_result = None
    if (
        cov_matrix is not None
        and not any(i.fatal for i in issues)
        and (stage == STAGE_REACHABLE_RETURN or wants_vol_floor)
    ):
        gmv_result, gmv_reason = _solve_gmv(expected_returns, cov_matrix, constraints)
        if gmv_result is not None and expected_returns is not None:
            gmv = float(gmv_result.expected_return)
        elif gmv_reason is not None:
            issues.append(
                FeasibilityIssue(
                    code="gmv_solver_error",
                    message=(
                        "The minimum-variance portfolio could not be solved "
                        f"({gmv_reason}), so this report cannot say where the "
                        "frontier stops being efficient, nor how low a "
                        "volatility target the mandate can reach."
                    ),
                    suggestion=(
                        "Check the covariance estimate and the solver install; "
                        "the constraints themselves have not been shown to be "
                        "at fault."
                    ),
                    fatal=False,
                )
            )

    target = constraints.target_return
    if check_targets and target is not None and lo is not None and hi is not None:
        if target > hi + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="target_return_too_high",
                    message=(
                        f"Target return of {target:.2%} is above the "
                        f"{hi:.2%} maximum these constraints can reach."
                    ),
                    suggestion=(
                        f"Lower the target to at most {hi:.2%}, or raise the "
                        "caps on the highest-return assets."
                        + (
                            " Note that Black-Litterman optimizes "
                            "against its equilibrium posterior, which "
                            "usually sits well below historical means "
                            "- so a target that suits mean-variance "
                            "can be unreachable here."
                            if expected_returns is not None
                            and getattr(expected_returns, "name", "")
                            == "black_litterman_posterior"
                            else ""
                        )
                    ),
                )
            )
        elif target < lo - 1e-8:
            # A warning, not a fatal: the target is a floor (``μ'w ≥ R*``)
            # since 0.7.0, and every allocation the constraints allow clears
            # one set below their minimum. This branch predates that change
            # and kept calling a solvable mandate impossible, which is what
            # ``raise_on_infeasible`` then refused.
            issues.append(
                FeasibilityIssue(
                    code="target_return_too_low",
                    message=(
                        f"Target return of {target:.2%} is below the "
                        f"{lo:.2%} minimum these constraints allow, so every "
                        "allocation clears it and it will not bind."
                    ),
                    suggestion=(
                        "The return target is a floor, so you will get the "
                        "portfolio the method picks without it. Raise the "
                        f"target above {lo:.2%} to have it change the answer."
                    ),
                    fatal=False,
                )
            )
        elif gmv is not None and target < gmv - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="target_return_inefficient",
                    message=(
                        f"Target return of {target:.2%} sits below the "
                        f"minimum-variance portfolio's {gmv:.2%}, so it "
                        "will not bind."
                    ),
                    suggestion=(
                        f"You will get the minimum-variance portfolio "
                        f"and its {gmv:.2%} return, not a portfolio "
                        f"built to {target:.2%}: the return target is a "
                        "floor, and minimum variance already clears it. "
                        "Raise the target above the minimum-variance "
                        "return to have it change the answer."
                    ),
                    fatal=False,
                )
            )

    target_vol = constraints.target_volatility
    if wants_vol_floor and gmv_result is not None and target_vol is not None:
        floor_vol = float(gmv_result.expected_volatility)
        if target_vol < floor_vol - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="target_vol_below_gmv",
                    message=(
                        f"Target volatility of {target_vol:.2%} "
                        f"is below the {floor_vol:.2%} floor set by the "
                        "minimum-variance portfolio."
                    ),
                    suggestion=(
                        f"Raise the target to at least {floor_vol:.2%}, or add a "
                        "lower-risk asset (e.g. cash) to the universe."
                    ),
                )
            )

    issues.sort(key=lambda i: not i.fatal)
    return FeasibilityReport(
        issues=tuple(issues),
        min_return=lo,
        max_return=hi,
        min_variance_return=gmv,
        stage_reached=stage,
    )


class InfeasibleConstraintsError(ValueError):
    """Raised with a full :class:`FeasibilityReport` attached."""

    def __init__(self, report: FeasibilityReport) -> None:
        """Build the error and attach the report that explains it.

        Args:
            report: The analysis that found the mandate impossible. It stays
                reachable as ``exc.report``, and its :meth:`~FeasibilityReport.describe`
                output — which names the binding constraint and the fix — becomes
                the exception's message.
        """
        self.report = report
        super().__init__(
            "The constraints cannot be satisfied:\n" + report.describe()
        )
