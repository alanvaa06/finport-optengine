"""Render the API reference from the package's own docstrings.

The README explains what the engine is for and ``docs/RESEARCH.md`` explains
where its methods come from. Neither one lists what a given function takes and
returns, and a library with 142 exported names needs that written down
somewhere other than the source.

pdoc is used rather than Sphinx because there is nothing to author: the
reference is a projection of the docstrings, so the only file that can go stale
is one that does not exist. Google-style sections (``Args:``, ``Returns:``,
``Raises:``) are what the package writes and what ``--docformat google`` reads.

Modules are discovered by walking the package rather than enumerated, so a new
one is documented the day it lands. Private modules -- any whose path has a
component starting with an underscore -- are skipped: they are implementation,
and pdoc would otherwise publish ``_cvxpy_helpers`` next to ``optimizers``.

A module whose optional extra is missing is skipped with a warning rather than
failing the build, so a core checkout can still render most of the reference.
``--strict`` turns that warning into an error, which is what CI runs: the
published reference should never quietly lose ``mcp_server`` because an extra
was left out of the install.

Usage::

    python scripts/build_api_docs.py [--output site/api] [--strict]
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PACKAGE = "optimization_engine"

#: Source links on every documented object. pdoc renders these as "View Source"
#: anchors, which is the difference between a reference you read and one you
#: can check.
EDIT_URL = "https://github.com/alanvaa06/finport-optengine/blob/main/src/optimization_engine/"


def _is_private(module: str) -> bool:
    """Whether any component below the package root is underscore-prefixed."""
    return any(part.startswith("_") for part in module.split(".")[1:])


def discover(package: str) -> list[str]:
    """Return every importable public module in ``package``, root first.

    Args:
        package: Import path of the package to walk, e.g. ``"optimization_engine"``.

    Returns:
        Module names in the order pdoc should receive them: the package itself,
        then its public submodules sorted alphabetically. Private modules are
        excluded.
    """
    root = importlib.import_module(package)
    # pdoc discovers submodules a package re-exports in ``__all__`` on its own.
    # Passing those again as explicit specs is not an error, but it warns about
    # a duplicate on every build, so leave them to pdoc.
    reexported = {f"{package}.{name}" for name in getattr(root, "__all__", ())}
    found = [package]
    for info in pkgutil.walk_packages(root.__path__, f"{package}."):
        if not _is_private(info.name) and info.name not in reexported:
            found.append(info.name)
    return found[:1] + sorted(found[1:])


def importable(modules: list[str], strict: bool) -> list[str]:
    """Drop modules that cannot be imported in this environment.

    A module gated behind an optional extra raises :class:`ImportError` on
    import; that is a missing install, not a broken module, so it is reported
    and skipped unless ``strict`` is set.

    Args:
        modules: Candidate module names, as returned by :func:`discover`.
        strict: Fail the build on the first unimportable module instead of
            skipping it.

    Returns:
        The subset of ``modules`` that imported cleanly.

    Raises:
        SystemExit: If ``strict`` is set and any module fails to import.
    """
    usable: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            if strict:
                raise SystemExit(f"{name} could not be imported: {exc}") from exc
            print(f"  skipping {name}: {exc}", file=sys.stderr)
            continue
        usable.append(name)
    return usable


def main(argv: list[str] | None = None) -> int:
    """Build the reference and report where it landed.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "site" / "api",
        help="Directory to write the HTML into (default: site/api).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any public module cannot be imported, rather than skipping it.",
    )
    args = parser.parse_args(argv)

    try:
        import pdoc
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SystemExit(
            "pdoc is not installed. Install it with: pip install 'finport-optengine[docs]'"
        ) from exc

    modules = importable(discover(PACKAGE), args.strict)

    pdoc.render.configure(docformat="google", edit_url_map={PACKAGE: EDIT_URL}, search=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pdoc.pdoc(*modules, output_directory=args.output)

    print(f"wrote {len(modules)} modules to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
