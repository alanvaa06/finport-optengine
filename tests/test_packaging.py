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


#: The repository's address before it became finport-optengine, in either of
#: the spellings it was linked with. GitHub redirects it for now; PyPI's project
#: links, the HTTP User-Agent and the docs should not depend on that lasting.
_OLD_ADDRESS = "alanvaa06/optimization_engine"


def _published_text_files() -> list[Path]:
    files = [
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "llms.txt",
        ROOT / "CHANGELOG.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "src" / "optimization_engine").rglob("*.py")),
    ]
    return [f for f in files if f.exists()]


def test_every_link_names_the_renamed_repository():
    hits = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in _published_text_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _OLD_ADDRESS in line.lower()
    ]
    assert hits == [], hits


def test_the_trusted_publisher_is_documented_under_the_new_name():
    """PyPI compares the repository name in the OIDC claim literally.

    A publisher still registered as ``Optimization_Engine`` rejects the next
    tag's upload, so the setup table has to name the repository as it is now.
    """
    releasing = (ROOT / "docs" / "RELEASING.md").read_text(encoding="utf-8")
    assert "| Repository name | `finport-optengine` |" in releasing


def test_the_project_urls_point_at_finport_optengine():
    block = PYPROJECT.split("[project.urls]", 1)[1].split("\n[", 1)[0]
    urls = re.findall(r'"(https://[^"]+)"', block)
    assert urls
    assert all(u.startswith("https://github.com/alanvaa06/finport-optengine") for u in urls)


def test_the_readme_banner_does_not_announce_an_old_release():
    version = re.search(r'^version = "([^"]+)"', PYPROJECT, re.M)[1]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for announced in re.findall(r"New in (\d+\.\d+\.\d+)", readme):
        assert announced == version, f"README says 'New in {announced}' on {version}"


def test_every_changelog_release_has_its_compare_link():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[([^\]]+)\](?! and earlier)", changelog, re.M)
    links = set(re.findall(r"^\[([^\]]+)\]: https://", changelog, re.M))
    missing = [h for h in headings if h not in links]
    assert missing == [], missing


# ---------------------------------------------------------------------------
# The release workflow
# ---------------------------------------------------------------------------

RELEASE = ROOT / ".github" / "workflows" / "release.yml"


def _release_jobs() -> dict:
    import yaml

    return yaml.safe_load(RELEASE.read_text(encoding="utf-8"))["jobs"]


def _upstream(jobs: dict, name: str) -> set[str]:
    """Every job ``name`` waits on, directly or through another job."""
    seen: set[str] = set()
    pending = [name]
    while pending:
        needs = jobs[pending.pop()].get("needs", [])
        for job in [needs] if isinstance(needs, str) else needs:
            if job not in seen:
                seen.add(job)
                pending.append(job)
    return seen


def _runs(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job.get("steps", []))


def test_a_tag_publishes_only_from_main():
    """A ``v*`` tag pushed from any branch published that branch to PyPI."""
    jobs = _release_jobs()
    guards = {
        name
        for name, job in jobs.items()
        if "merge-base --is-ancestor" in _runs(job) and "origin/main" in _runs(job)
    }
    assert guards & _upstream(jobs, "pypi"), "pypi does not wait on a main-ancestry check"


def test_pypi_waits_for_the_test_suite():
    """The release built and smoke-tested a wheel but never ran the tests."""
    jobs = _release_jobs()
    testing = {name for name, job in jobs.items() if "pytest" in _runs(job)}
    assert testing & _upstream(jobs, "pypi"), "pypi does not wait on a pytest job"


def test_actions_are_pinned_by_commit():
    """A tag is a pointer its owner can move; a commit SHA is not.

    The one exception is PyPA's publish action, documented in
    docs/RELEASING.md: it pulls a container image named after the ref, and
    images exist only for release tags.
    """
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)", RELEASE.read_text(encoding="utf-8"), re.M)
    assert uses
    unpinned = [
        u
        for u in uses
        if not re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u)
        and not u.startswith("pypa/gh-action-pypi-publish@v")
    ]
    assert unpinned == [], unpinned


def test_the_build_tools_are_pinned():
    text = RELEASE.read_text(encoding="utf-8")
    assert re.search(r"\bbuild==[0-9.]+", text)
    assert re.search(r"\btwine==[0-9.]+", text)
