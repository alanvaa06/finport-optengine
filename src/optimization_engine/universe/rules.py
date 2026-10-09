"""A universe written down: rules on a page, and the data they are asked of.

:mod:`~optimization_engine.universe.eligibility` can express a membership rule
but not *store* one. Everything there is built in Python from a frame already
in memory, which is fine for a notebook and useless for the two callers that
matter most — the CLI, which has no Python to run, and a committee, which wants
the mandate in a file it can read, diff and sign off. This module is that file
format and its loader.

**The document.** One mapping, four blocks, all but ``rules`` optional::

    schema_version: 1
    combine: all                 # or "any" — how the rules below join
    panels:                      # the data the rules are asked of
      adv:   {path: adv.csv}
      mcap:  {path: market_cap.parquet}
    rules:
      - {kind: rolling,   panel: adv,  window: 63, agg: mean, op: ">=", value: 1.0e7}
      - {kind: rank,      panel: mcap, top_n: 500}
      - {kind: threshold, panel: returns, op: ">", value: -0.5, name: not in freefall}
    hysteresis:                  # optional: easier to keep than to gain
      exit: [{kind: rolling, panel: adv, window: 63, agg: mean, op: "<", value: 5.0e6}]
      initial: null
    hold_through:                # optional: a reconstitution calendar
      dates: ["2020-03-20", "2020-06-19"]

Composition is in that order and only that order: the ``rules`` list joins
under ``combine`` into the **entry** rule, ``hysteresis`` wraps it so that
staying in needs only the exit rule *not* to fire, and ``hold_through`` freezes
the finished verdict onto a review calendar. Writing it any other way — holding
a rule through reconstitutions and *then* applying hysteresis to the held
series — would apply the churn brake to a series that no longer churns.

**Unknown keys are refused, never ignored.**
:meth:`~optimization_engine.stress.Shock.from_dict` sets the precedent and the
reason is the same one: a misspelt ``windwo`` that loads cleanly gives you a
rule with a default window and no warning, which is a *different mandate*
silently substituted for the one that was signed off.

**Where the characteristic panel comes from, and why here.** A liquidity or
size screen needs data that is not prices — ADV, market capitalisation, an
index-membership flag — and the engine has no route for one: the CLI's
``_prepare_inputs`` loads a price panel and nothing else, and
:func:`~optimization_engine.backtest.walkforward.walk_forward_run` takes an
:class:`~optimization_engine.universe.eligibility.Eligibility` that is already
built. So the rules file carries the paths itself, in the ``panels`` block, and
this loader reads them.

The alternative — a second ``--characteristics`` flag beside ``--universe`` —
was rejected because it lets the two drift. A rule and the panel it is
evaluated against are one statement ("eligible if *this* series clears *that*
number"), and splitting them across two arguments makes it possible, and
eventually routine, to run rule set A against panel B and get a universe
nobody wrote. One file, one universe definition, data included by reference.
Relative paths resolve against the rules file's own directory, so the document
and its panels move together.

Two panel names are supplied by the *run* rather than by the file, because they
are the run's own data and naming a path for them would be a second copy:
``returns`` and ``prices``. A rules file that uses only those needs no
``panels`` block at all and no data alongside it, which is the whole first cut
of a point-in-time universe — "it printed all week", "it is not in freefall",
"its trailing volatility is under 40%" — available with nothing but the panel
already on the command line.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from optimization_engine.universe.eligibility import (
    COMPARISONS,
    ROLLING_AGGS,
    Eligibility,
    point_in_time_mask,
)
from optimization_engine.universe.signal import UniverseError

#: The document schema this loader reads. Bumped only for a change that would
#: make an older file mean something different, never for a new optional key.
RULES_SCHEMA_VERSION = 1

#: The rule kinds ``rules:`` entries may declare, mapped to the
#: :class:`~optimization_engine.universe.eligibility.Eligibility` constructor
#: each one calls.
RULE_KINDS = ("threshold", "rank", "rolling")

#: Panel names the *run* supplies rather than the file. A rules file using only
#: these needs no ``panels`` block and no data files beside it.
RUN_PANELS = ("returns", "prices")

#: How several rules join. ``"all"`` is the conjunction, ``"any"`` the
#: disjunction — both under Kleene logic, so an unknown that cannot change the
#: answer does not make the answer unknown.
COMBINERS = ("all", "any")

#: The extensions :func:`load_universe_rules` reads. Both parse through the
#: YAML loader, which reads JSON as well.
RULES_SUFFIXES = (".yaml", ".yml", ".json")

#: Largest rules document read, in bytes. A real one is a few kilobytes; the
#: limit exists so that a path pointing somewhere else is refused unread.
MAX_RULES_BYTES = 1024 * 1024

_DOCUMENT_KEYS = frozenset(
    {"schema_version", "combine", "panels", "rules", "hysteresis", "hold_through"}
)
_PANEL_KEYS = frozenset({"path", "sheet", "index_col"})
_HYSTERESIS_KEYS = frozenset({"exit", "combine", "initial"})
_HOLD_THROUGH_KEYS = frozenset({"dates"})
_RULE_KEYS: dict[str, frozenset[str]] = {
    "threshold": frozenset({"kind", "panel", "op", "value", "name"}),
    "rank": frozenset({"kind", "panel", "top_n", "name"}),
    "rolling": frozenset({"kind", "panel", "window", "agg", "op", "value", "name"}),
}


def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    """``value`` as a mapping, or a :class:`UniverseError` naming what it was.

    Args:
        value: The parsed fragment.
        what: How to name it in the message.

    Returns:
        The mapping.

    Raises:
        UniverseError: If it is not one.
    """
    if not isinstance(value, Mapping):
        raise UniverseError(f"{what} is written as a mapping; got {type(value).__name__}.")
    return value


def _reject_unknown(data: Mapping[str, Any], known: frozenset[str], what: str) -> None:
    """Refuse a key nothing reads, rather than dropping it in silence.

    Args:
        data: The mapping to check.
        known: Every key this level understands.
        what: How to name the level in the message.

    Raises:
        UniverseError: On any key outside ``known``.
    """
    unknown = sorted(str(k) for k in data if k not in known)
    if unknown:
        raise UniverseError(
            f"Unknown {what} key(s): {', '.join(unknown)}. Known keys: "
            f"{', '.join(sorted(known))}."
        )


def _required(data: Mapping[str, Any], key: str, what: str) -> Any:
    """One required key, or a message saying which one is missing from what.

    Args:
        data: The mapping to read.
        key: The key that has to be there.
        what: How to name the mapping in the message.

    Returns:
        The value.

    Raises:
        UniverseError: If the key is absent.
    """
    if key not in data:
        raise UniverseError(f"{what} is missing required key {key!r}.")
    return data[key]


def _as_float(value: Any, what: str) -> float:
    """A number, or a message naming the field that was not one."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise UniverseError(f"{what} must be a number; got {value!r}.") from exc


def _as_int(value: Any, what: str) -> int:
    """A whole number, or a message naming the field that was not one."""
    if isinstance(value, bool):
        raise UniverseError(f"{what} must be a whole number; got {value!r}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise UniverseError(f"{what} must be a whole number; got {value!r}.") from exc


@dataclass(frozen=True)
class PanelSpec:
    """One characteristic panel, named in the rules file and read from disk.

    Attributes:
        name: How the ``rules`` entries refer to it.
        path: The file, as written. Relative paths resolve against the rules
            file's directory, not the working directory: the document and its
            data are one artifact and have to survive being moved together.
        sheet: Worksheet for an Excel workbook, by name or position. Ignored
            for CSV and Parquet. Defaults to the first sheet — *not* to the
            price loader's ``"Precios"``, which is a convention of this
            project's price workbooks and has nothing to do with a
            characteristic panel.
        index_col: Which column carries the dates.
    """

    name: str
    path: str
    sheet: str | int = 0
    index_col: str | int = 0

    @classmethod
    def from_entry(cls, name: str, entry: Any) -> PanelSpec:
        """Read one ``panels:`` entry, in either of its two spellings.

        Args:
            name: The panel's name, from the block's key.
            entry: Either a bare path — the common case, and the only thing
                needed for CSV and Parquet — or a mapping carrying ``path``
                and, for a workbook, ``sheet`` and ``index_col``.

        Returns:
            The :class:`PanelSpec`.

        Raises:
            UniverseError: If the entry is neither shape, carries a key this
                class does not read, or omits ``path``.
        """
        if isinstance(entry, (str, Path)):
            return cls(name=name, path=str(entry))
        data = _require_mapping(entry, f"Panel {name!r}")
        _reject_unknown(data, _PANEL_KEYS, f"panel ({name!r})")
        return cls(
            name=name,
            path=str(_required(data, "path", f"Panel {name!r}")),
            sheet=data.get("sheet", 0),
            index_col=data.get("index_col", 0),
        )

    def load(self, base_dir: Path | None) -> pd.DataFrame:
        """Read the panel off disk.

        Args:
            base_dir: The rules file's directory, or ``None`` when the document
                came from memory — in which case a relative path is resolved
                against the working directory, because there is nothing better
                to resolve it against.

        Returns:
            The panel as a ``date × asset`` frame, dates ascending.

        Raises:
            UniverseError: If the file is missing, its extension is not one the
                loader reads, or pandas cannot parse it. The underlying error
                is quoted — a bad panel is a fixable input, and the fix is
                usually in the message.
        """
        from optimization_engine.data.loader import load_prices

        path = Path(self.path)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        try:
            return load_prices(path, sheet_name=self.sheet, index_col=self.index_col)
        except FileNotFoundError as exc:
            raise UniverseError(
                f"Panel {self.name!r} names a file that does not exist: {path}."
            ) from exc
        except Exception as exc:  # noqa: BLE001 — every reader failure is one input error
            raise UniverseError(f"Panel {self.name!r} could not be read from {path}: {exc}") from exc


@dataclass(frozen=True)
class RuleSpec:
    """One ``rules:`` entry, parsed but not yet evaluated.

    Kept separate from the :class:`Eligibility` it builds so that a document
    can be validated — every key known, every panel named, every operator
    real — before any data is read. A typo in the third rule should not cost
    you the load of the first two panels.

    Attributes:
        kind: One of :data:`RULE_KINDS`.
        panel: The panel this rule reads, by name.
        name: The label :meth:`Eligibility.explain` will quote. ``None`` lets
            the eligibility constructor generate one.
        op: The comparison, for ``threshold`` and ``rolling``.
        value: The threshold, in the panel's own units.
        top_n: How many names ``rank`` admits per date.
        window: How many *prior* periods ``rolling`` aggregates.
        agg: Which aggregation ``rolling`` applies.
    """

    kind: str
    panel: str
    name: str | None = None
    op: str | None = None
    value: float | None = None
    top_n: int | None = None
    window: int | None = None
    agg: str | None = None

    @classmethod
    def from_dict(cls, data: Any, where: str) -> RuleSpec:
        """Parse one rule entry.

        Args:
            data: The mapping to read.
            where: How to name this entry in a message — the caller knows
                whether it came from ``rules`` or from ``hysteresis.exit``.

        Returns:
            The :class:`RuleSpec`.

        Raises:
            UniverseError: On an unknown ``kind``, a key that kind does not
                read, a missing required key, or an operator or aggregation
                that is not one of the sets
                :mod:`~optimization_engine.universe.eligibility` publishes.
        """
        entry = _require_mapping(data, where)
        kind = str(_required(entry, "kind", where))
        if kind not in RULE_KINDS:
            raise UniverseError(
                f"{where} has kind={kind!r}, which is not a rule this loader "
                f"builds. Available: {list(RULE_KINDS)}."
            )
        _reject_unknown(entry, _RULE_KEYS[kind], f"{kind} rule")
        panel = str(_required(entry, "panel", where))
        label = entry.get("name")
        name = None if label is None else str(label)

        if kind == "rank":
            return cls(
                kind=kind,
                panel=panel,
                name=name,
                top_n=_as_int(_required(entry, "top_n", where), f"{where} top_n"),
            )

        op = str(_required(entry, "op", where))
        if op not in COMPARISONS:
            raise UniverseError(
                f"{where} has op={op!r}, which is not a comparison. "
                f"Available: {sorted(COMPARISONS)}."
            )
        value = _as_float(_required(entry, "value", where), f"{where} value")
        if kind == "threshold":
            return cls(kind=kind, panel=panel, name=name, op=op, value=value)

        agg = str(_required(entry, "agg", where))
        if agg not in ROLLING_AGGS:
            raise UniverseError(
                f"{where} has agg={agg!r}, which is not a rolling aggregation. "
                f"Available: {list(ROLLING_AGGS)}."
            )
        return cls(
            kind=kind,
            panel=panel,
            name=name,
            op=op,
            value=value,
            window=_as_int(_required(entry, "window", where), f"{where} window"),
            agg=agg,
        )

    def build(self, panels: Mapping[str, pd.DataFrame]) -> Eligibility:
        """Evaluate this rule against the panels.

        Args:
            panels: Every panel available, by name — the run's own and the
                file's, already merged.

        Returns:
            The :class:`Eligibility` this one rule defines.

        Raises:
            UniverseError: If the panel this rule names is not among them, or
                the eligibility constructor refuses the frame.
        """
        if self.panel not in panels:
            raise UniverseError(
                f"Rule {self.describe()!r} reads panel {self.panel!r}, which is "
                f"not available. Declare it under 'panels:', or use one the run "
                f"supplies: {', '.join(sorted(panels)) or 'none'}."
            )
        frame = panels[self.panel]
        # The asserts narrow types mypy cannot see through; ``from_dict`` is
        # the only constructor and it fills exactly the fields each kind needs.
        if self.kind == "threshold":
            assert self.op is not None and self.value is not None
            return Eligibility.from_threshold(frame, self.op, self.value, name=self.label())
        if self.kind == "rank":
            assert self.top_n is not None
            return Eligibility.from_rank(frame, self.top_n, name=self.label())
        assert self.window is not None and self.agg is not None
        assert self.op is not None and self.value is not None
        return Eligibility.from_rolling(
            frame, self.window, self.agg, self.op, self.value, name=self.label()
        )

    def label(self) -> str:
        """The rule's own label, or a generated one naming its panel."""
        return self.name or self.describe()

    def describe(self) -> str:
        """One line naming the panel, so a generated label says which data it read.

        Returns:
            A phrase such as ``"adv rolling mean over 63 prior periods >= 1e+07"``.
            The eligibility constructors generate labels that name the
            comparison but not the series, which reads fine with one rule and
            not at all with four.
        """
        if self.kind == "rank":
            return f"top {self.top_n} by {self.panel}"
        if self.kind == "threshold":
            return f"{self.panel} {self.op} {self.value:g}"
        return (
            f"{self.panel} rolling {self.agg} over {self.window} prior periods "
            f"{self.op} {self.value:g}"
        )


@dataclass(frozen=True)
class UniverseRules:
    """A parsed rules document, and the panels it still needs.

    Parsing and evaluation are separate steps on purpose: :meth:`from_dict`
    validates the whole document — every key, every operator, every rule's
    shape — without touching the disk, and :meth:`build` is the step that reads
    files and evaluates. A caller that only wants to know whether a file is
    well-formed (``optengine`` before a long run, an app validating a text
    box) can stop after the first.

    Attributes:
        combine: How the ``rules`` list joins — one of :data:`COMBINERS`.
        rules: The entry rules, in the document's order.
        panels: The characteristic panels the file declares, by name.
        exit_rules: The ``hysteresis.exit`` rules, empty when the file declares
            no hysteresis.
        exit_combine: How those join.
        initial: ``hysteresis.initial`` — membership before the first date.
            ``None`` means *unknown*, which propagates rather than defaulting
            a name in or out, and is the default for exactly that reason.
        hold_through: The reconstitution dates, empty when there are none.
        base_dir: What relative panel paths resolve against.
    """

    combine: str = "all"
    rules: tuple[RuleSpec, ...] = ()
    panels: tuple[PanelSpec, ...] = ()
    exit_rules: tuple[RuleSpec, ...] = ()
    exit_combine: str = "all"
    initial: bool | None = None
    has_hysteresis: bool = False
    hold_through: tuple[str, ...] = ()
    base_dir: Path | None = None

    @classmethod
    def from_dict(cls, data: Any, *, base_dir: Path | None = None) -> UniverseRules:
        """Parse a rules document, refusing anything it does not understand.

        Args:
            data: The parsed mapping.
            base_dir: Directory relative panel paths resolve against, normally
                the rules file's own.

        Returns:
            The :class:`UniverseRules`.

        Raises:
            UniverseError: On an unsupported ``schema_version``, an unknown key
                at any level, an empty or missing ``rules`` list, a malformed
                rule, or a ``hysteresis`` block with no ``exit``.
        """
        document = _require_mapping(data, "A universe rules document")
        _reject_unknown(document, _DOCUMENT_KEYS, "rules document")

        version = document.get("schema_version", RULES_SCHEMA_VERSION)
        if version != RULES_SCHEMA_VERSION:
            raise UniverseError(
                f"Unsupported universe rules schema_version={version!r}; expected "
                f"{RULES_SCHEMA_VERSION}."
            )

        combine = cls._combiner(document.get("combine", "all"), "combine")

        raw_rules = document.get("rules")
        if not raw_rules:
            raise UniverseError(
                "A universe rules document needs a non-empty 'rules' list. A "
                "document with no rules does not describe a universe: it would "
                "admit or exclude every name depending only on the collapse "
                "policy, which is not a screen."
            )
        if isinstance(raw_rules, Mapping) or not isinstance(raw_rules, Sequence):
            raise UniverseError(
                f"'rules' is a list of rule mappings; got {type(raw_rules).__name__}."
            )
        rules = tuple(
            RuleSpec.from_dict(entry, f"rules[{i}]") for i, entry in enumerate(raw_rules)
        )

        panels_block = document.get("panels") or {}
        panels = tuple(
            PanelSpec.from_entry(str(name), entry)
            for name, entry in _require_mapping(panels_block, "'panels'").items()
        )
        clashing = sorted({p.name for p in panels} & set(RUN_PANELS))
        if clashing:
            raise UniverseError(
                f"Panel name(s) {', '.join(clashing)} are supplied by the run "
                f"itself and cannot be redeclared. Reserved: {', '.join(RUN_PANELS)}."
            )

        exit_rules: tuple[RuleSpec, ...] = ()
        exit_combine = "all"
        initial: bool | None = None
        has_hysteresis = "hysteresis" in document and document["hysteresis"] is not None
        if has_hysteresis:
            block = _require_mapping(document["hysteresis"], "'hysteresis'")
            _reject_unknown(block, _HYSTERESIS_KEYS, "hysteresis")
            raw_exit = _required(block, "exit", "'hysteresis'")
            entries = raw_exit if isinstance(raw_exit, list) else [raw_exit]
            if not entries:
                raise UniverseError(
                    "'hysteresis.exit' is empty. Hysteresis without an exit rule "
                    "is a universe nothing can leave."
                )
            exit_rules = tuple(
                RuleSpec.from_dict(entry, f"hysteresis.exit[{i}]")
                for i, entry in enumerate(entries)
            )
            exit_combine = cls._combiner(block.get("combine", "all"), "hysteresis.combine")
            raw_initial = block.get("initial")
            initial = None if raw_initial is None else bool(raw_initial)

        hold: tuple[str, ...] = ()
        if document.get("hold_through") is not None:
            block = _require_mapping(document["hold_through"], "'hold_through'")
            _reject_unknown(block, _HOLD_THROUGH_KEYS, "hold_through")
            dates = _required(block, "dates", "'hold_through'")
            if isinstance(dates, (str, bytes)) or not isinstance(dates, Sequence):
                raise UniverseError(
                    f"'hold_through.dates' is a list of dates; got {type(dates).__name__}."
                )
            if not dates:
                raise UniverseError(
                    "'hold_through.dates' is empty. With no reconstitution the "
                    "verdict is never reviewed and every row comes back unknown."
                )
            hold = tuple(str(d) for d in dates)

        return cls(
            combine=combine,
            rules=rules,
            panels=panels,
            exit_rules=exit_rules,
            exit_combine=exit_combine,
            initial=initial,
            has_hysteresis=has_hysteresis,
            hold_through=hold,
            base_dir=base_dir,
        )

    @staticmethod
    def _combiner(value: Any, where: str) -> str:
        """One of :data:`COMBINERS`, or a message naming both."""
        combine = str(value).lower()
        if combine not in COMBINERS:
            raise UniverseError(
                f"{where}={value!r} is not a combiner. Available: {list(COMBINERS)}."
            )
        return combine

    @property
    def panel_names(self) -> tuple[str, ...]:
        """Every panel the rules read, whether the run supplies it or the file does."""
        names = {rule.panel for rule in self.rules} | {rule.panel for rule in self.exit_rules}
        return tuple(sorted(names))

    def build(self, **run_panels: pd.DataFrame | None) -> Eligibility:
        """Read the declared panels and evaluate the rules against them.

        Args:
            **run_panels: The panels the run supplies rather than the file —
                ``returns`` and ``prices``, named in :data:`RUN_PANELS`. A
                ``None`` is treated as absent, so a caller with no price frame
                may pass ``prices=None`` rather than branching.

        Returns:
            The :class:`Eligibility` the document describes: the ``rules``
            joined under ``combine``, wrapped in hysteresis when the file asked
            for it, and held through the reconstitution calendar when it named
            one — in that order.

        Raises:
            UniverseError: If a run panel is offered under a name that is not
                one of :data:`RUN_PANELS`, a declared panel cannot be read, a
                rule names a panel that is not available, or a rule's frame is
                one its constructor refuses.
        """
        unexpected = sorted(set(run_panels) - set(RUN_PANELS))
        if unexpected:
            raise UniverseError(
                f"Unknown run panel(s): {', '.join(unexpected)}. The run supplies "
                f"{', '.join(RUN_PANELS)}; everything else is declared under "
                "'panels:' in the rules file."
            )
        available: dict[str, pd.DataFrame] = {
            name: frame for name, frame in run_panels.items() if frame is not None
        }
        # Only the panels some rule actually reads. A document may declare more
        # than it uses -- a shared file edited down for one run -- and paying
        # for a Parquet read nothing consults is a cost with no answer attached.
        wanted = set(self.panel_names)
        for spec in self.panels:
            if spec.name in wanted:
                available[spec.name] = spec.load(self.base_dir)

        eligibility = self._combine(self.rules, self.combine, available)
        if self.has_hysteresis:
            eligibility = Eligibility.with_hysteresis(
                eligibility,
                self._combine(self.exit_rules, self.exit_combine, available),
                self.initial,
            )
        if self.hold_through:
            eligibility = eligibility.hold_through(self.hold_through)
        return eligibility

    @staticmethod
    def _combine(
        rules: Sequence[RuleSpec], how: str, panels: Mapping[str, pd.DataFrame]
    ) -> Eligibility:
        """Build every rule and join them, left to right, under Kleene logic."""
        built = [rule.build(panels) for rule in rules]
        joined = built[0]
        for other in built[1:]:
            joined = joined & other if how == "all" else joined | other
        return joined

    def describe(self) -> str:
        """The document in one paragraph, for a CLI that has just loaded it.

        Returns:
            A few lines naming the rules, the panels they read, and the two
            optional blocks when the file uses them.
        """
        joiner = " and " if self.combine == "all" else " or "
        lines = [f"  Universe: {joiner.join(r.label() for r in self.rules)}"]
        if self.has_hysteresis:
            exit_joiner = " and " if self.exit_combine == "all" else " or "
            lines.append(
                f"    Hysteresis: a name leaves only when "
                f"{exit_joiner.join(r.label() for r in self.exit_rules)}; "
                f"membership before the first date is "
                f"{'unknown' if self.initial is None else str(self.initial).lower()}."
            )
        if self.hold_through:
            lines.append(
                f"    Held through {len(self.hold_through)} reconstitution(s), "
                f"{self.hold_through[0]} ... {self.hold_through[-1]}."
            )
        declared = [p.name for p in self.panels if p.name in set(self.panel_names)]
        if declared:
            lines.append(f"    Panels read from disk: {', '.join(sorted(declared))}.")
        return "\n".join(lines)


def load_universe_rules(path: str | Path) -> UniverseRules:
    """Parse a rules file without reading any of the panels it names.

    Args:
        path: A ``.yaml``, ``.yml`` or ``.json`` document. Both parse through
            the YAML loader, which reads JSON as well.

    Returns:
        The :class:`UniverseRules`, with ``base_dir`` set to the file's own
        directory so its relative panel paths resolve.

    Raises:
        UniverseError: If the extension is not one of the three above, the
            file is larger than :data:`MAX_RULES_BYTES`, the document is
            malformed, as :meth:`UniverseRules.from_dict` describes, or it is
            not valid YAML.
        FileNotFoundError: If the path does not exist.
    """
    import yaml

    p = Path(path)
    # Extension and size before the read. The suffix used to choose a parser
    # only after the whole file was in memory, and everything that was not
    # .json went to YAML — so an extensionless credentials file was read and
    # its second line came back inside the parse error.
    suffix = p.suffix.lower()
    if suffix not in RULES_SUFFIXES:
        raise UniverseError(
            f"{p}: a rules document ends in {', '.join(RULES_SUFFIXES)}; "
            f"got {suffix or 'no extension'}."
        )
    size = p.stat().st_size
    if size > MAX_RULES_BYTES:
        raise UniverseError(
            f"{p} is {size:,} bytes; a rules document is read up to "
            f"{MAX_RULES_BYTES:,} bytes."
        )
    text = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text) if suffix != ".json" else json.loads(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        # The position, not the parser's message: that quotes the offending
        # line, which in a file that is not a rules document is its content.
        mark = getattr(exc, "problem_mark", None)
        if isinstance(exc, json.JSONDecodeError):
            where = f" (line {exc.lineno}, column {exc.colno})"
        elif mark is not None:
            where = f" (line {mark.line + 1}, column {mark.column + 1})"
        else:
            where = ""
        raise UniverseError(
            f"{p} is not a readable YAML/JSON document{where}."
        ) from exc
    if data is None:
        raise UniverseError(f"{p} is empty.")
    return UniverseRules.from_dict(data, base_dir=p.parent)


def load_universe(
    path: str | Path,
    *,
    returns: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> Eligibility:
    """A rules file and the run's panels, in, an :class:`Eligibility` out.

    The one call the CLI and the app both make. Parsing and evaluation are
    still separate underneath — :func:`load_universe_rules` then
    :meth:`UniverseRules.build` — for a caller that wants to validate a
    document without reading its data.

    Args:
        path: The rules file.
        returns: The run's return panel, available to rules as ``returns``.
        prices: The run's price panel, available as ``prices``. Optional: a
            rules file that never names it does not need one.

    Returns:
        The point-in-time membership the document describes.

    Raises:
        UniverseError: On a malformed document, an unreadable panel, or a rule
            naming a panel nothing supplied.
        FileNotFoundError: If the rules file does not exist.
    """
    return load_universe_rules(path).build(returns=returns, prices=prices)


def count_unresolved(
    universe: Eligibility,
    index: Any,
    assets: Sequence[str],
    *,
    execution_lag: int | None = None,
) -> tuple[int, int, tuple[str, ...]]:
    """How much of the run the collapse policy — not the rules — decides.

    Every rule with a warm-up leaves cells nothing evaluated, and so does every
    asset in the panel that the rules never mention. Under ``"exclude"`` those
    cells silently shrink the book and under ``"include"`` they silently widen
    it, and in neither case did a screen say so. This counts them, so the CLI
    can report the size of the decision it made on the caller's behalf instead
    of making it invisibly.

    It is computed as the disagreement between the two collapses: a cell that
    ``"include"`` reads ``True`` and ``"exclude"`` reads ``False`` is exactly a
    cell no rule reached.

    Args:
        universe: The membership definition.
        index: The run's bars.
        assets: The return frame's columns, in order.
        execution_lag: The run's lag, so the count reads the universe the way
            the run will; see
            :func:`~optimization_engine.universe.eligibility.point_in_time_mask`.

    Returns:
        ``(cells, bars, assets)`` — how many ``(bar, asset)`` cells the policy
        decides, over how many bars they are spread, and the names any of them
        touch, sorted.

    Raises:
        UniverseError: If the universe's own axes cannot be read as dates.
    """
    admitted = point_in_time_mask(
        universe, "include", index, assets, execution_lag=execution_lag
    )
    refused = point_in_time_mask(
        universe, "exclude", index, assets, execution_lag=execution_lag
    )
    disputed = admitted & (~refused)
    cells = int(disputed.to_numpy(dtype=bool).sum())
    bars = int(disputed.any(axis=1).sum())
    names = tuple(str(c) for c in disputed.columns[disputed.any(axis=0)])
    return cells, bars, names


__all__ = [
    "COMBINERS",
    "RULES_SCHEMA_VERSION",
    "RULE_KINDS",
    "RUN_PANELS",
    "PanelSpec",
    "RuleSpec",
    "UniverseRules",
    "count_unresolved",
    "load_universe",
    "load_universe_rules",
]
