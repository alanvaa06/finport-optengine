"""Every string the package can print is ASCII.

A piped run on Windows encodes stdout with the ANSI code page, and an em-dash
or an arrow in a message either raises ``UnicodeEncodeError`` or, behind the
``backslashreplace`` guard in ``cli``, reaches the reader as ``\\u2014``. The
CLI's own ``print`` calls are only half of what it prints: data-quality
findings, feasibility diagnoses, optimizer warnings, method descriptions and
exception messages are all built in the library and echoed by ``optengine``
and ``scripts/``. Tracing which literal reaches which stream is not something a
test can do reliably, so this one checks the stronger property instead: no
string literal under ``src/optimization_engine`` or ``scripts/`` contains a
non-ASCII character, docstrings excepted.

Docstrings are exempt because they are read through ``help()`` and the
rendered API reference, never printed by a command. Comments are not string
literals and are never seen here. ``app/`` is out of scope: Streamlit renders
in a browser, where the typography is deliberate.

The allowlist is keyed by file, optionally narrowed to exact literal values
(a sentinel the code matches rather than prints), with a reason per entry. It
is exact in both directions: an entry whose file or literal no longer exists
fails too, so an exemption cannot outlive its cause.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "optimization_engine"
SCRIPTS = ROOT / "scripts"


class Allowed(NamedTuple):
    """One file whose string literals may carry non-ASCII characters.

    Attributes:
        path: Path relative to the repository root, with forward slashes.
        reason: Why nothing exempted here reaches a console. Required.
        literals: The exact values exempted, or ``None`` to exempt the whole
            file. A value is data the code matches against, not text it shows.
    """

    path: str
    reason: str
    literals: tuple[str, ...] | None = None


ALLOWED = (
    Allowed(
        "src/optimization_engine/constraints.py",
        "The em-dash is ui_state.UNASSIGNED, the app's 'no bucket' choice; "
        "ConstraintLayer drops assignments equal to it. Matched, never printed.",
        literals=("—",),
    ),
    Allowed(
        "src/optimization_engine/reporting/plots.py",
        "Plotly axis titles and hover templates; rendered in a browser or "
        "an image, never written to a console.",
    ),
    Allowed(
        "src/optimization_engine/ui_state.py",
        "Captions and labels for the Streamlit app; imported by app/ only.",
    ),
)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Return the ids of every docstring constant in ``tree``."""
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, owners) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def _non_ascii_literals(path: Path, exempt: tuple[str, ...] = ()) -> list[str]:
    """Return ``line: literal`` for each non-docstring literal that is not ASCII.

    Args:
        path: The Python file to scan.
        exempt: Exact literal values to skip.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = _docstring_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and not node.value.isascii()
            and node.value not in exempt
        ):
            hits.append(f"{node.lineno}: {node.value!a}")
    return hits


def _sources() -> list[Path]:
    """Every Python file whose literals can reach a console."""
    return sorted(PACKAGE.rglob("*.py")) + sorted(SCRIPTS.rglob("*.py"))


def _relative(path: Path) -> str:
    """``path`` relative to the repository root, with forward slashes."""
    return path.relative_to(ROOT).as_posix()


def test_every_allowlist_entry_has_a_reason():
    for entry in ALLOWED:
        assert entry.reason.strip(), entry.path


def test_no_console_facing_literal_is_non_ascii():
    entries = {entry.path: entry for entry in ALLOWED}
    offenders = {}
    for path in _sources():
        entry = entries.get(_relative(path))
        if entry is not None and entry.literals is None:
            continue
        exempt = entry.literals if entry is not None else ()
        if hits := _non_ascii_literals(path, exempt):
            offenders[_relative(path)] = hits
    assert not offenders, (
        "Non-ASCII characters in string literals that can reach a console. "
        "Use '-' for a dash, '->' for an arrow, '...' for an ellipsis, 'x' "
        "for a times sign and spelled-out Greek. If the file never prints, "
        "add it to ALLOWED with the reason.\n"
        + "\n".join(f"{p}:{hit}" for p, hits in offenders.items() for hit in hits)
    )


def test_the_allowlist_has_no_stale_entries():
    stale = []
    for entry in ALLOWED:
        path = ROOT / entry.path
        found = (
            {hit.split(": ", 1)[1] for hit in _non_ascii_literals(path)}
            if path.exists()
            else set()
        )
        wanted = {ascii(v) for v in entry.literals} if entry.literals else None
        if not found or (wanted is not None and not wanted <= found):
            stale.append(entry.path)
    assert not stale, f"These files no longer need an exemption: {stale}"
