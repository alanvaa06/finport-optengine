"""What the distribution declares about itself, held to what is true.

Packaging metadata is read by people and tools that never open the source —
pip resolving a floor, PyPI rendering the project links — so a wrong value
there is wrong for every installer at once and is invisible to every other
test. These read ``pyproject.toml`` as text rather than through ``tomllib``,
which only arrived in 3.11 and this package still supports 3.9.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_no_pyarrow_floor_admits_the_parquet_code_execution_release():
    """pyarrow 14.0.0 executes arbitrary code when reading untrusted Parquet.

    CVE-2023-47248, fixed in 14.0.1. The CLI, the MCP server and the app all
    accept a ``.parquet`` price file from whoever supplies one, and
    ``pyarrow>=14.0`` let a resolver pick exactly the vulnerable release.
    """
    floors = re.findall(r'"pyarrow>=([0-9.]+)"', PYPROJECT)
    assert floors, "pyarrow is no longer declared; update this test"
    assert all(_version(f) >= (14, 0, 1) for f in floors), floors
