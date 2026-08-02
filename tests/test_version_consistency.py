"""The version must agree everywhere it is declared.

Before 0.2.0 the version was hardcoded in both ``pyproject.toml`` and
``spatioev/__init__.py`` with nothing checking that they matched, and
``CITATION.cff`` was a third manual copy. ``pyproject.toml`` now derives the
version from ``spatioev.__version__``, so only the citation metadata can
drift — which is what this guards.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import spatioev

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)


def _pyproject() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_is_a_valid_semver_string():
    assert re.fullmatch(r"\d+\.\d+\.\d+([.-].+)?", spatioev.__version__), (
        f"__version__ = {spatioev.__version__!r} is not a version string"
    )


def test_pyproject_derives_the_version_from_the_package():
    """pyproject must not hardcode a second copy of the version."""
    metadata = _pyproject()
    assert "version" in metadata["project"].get("dynamic", []), (
        "project.version should be dynamic, not a literal"
    )
    attr = metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "spatioev.__version__"
    assert "version" not in metadata["project"], (
        "a literal project.version would shadow the dynamic one"
    )


def test_citation_matches_the_package_version():
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", citation, flags=re.M)
    assert match, "CITATION.cff has no version field"

    cited = match.group(1).strip("\"'")
    assert cited == spatioev.__version__, (
        f"CITATION.cff says {cited}, package says {spatioev.__version__}. "
        "Anyone citing the package would name the wrong release."
    )


def test_changelog_documents_the_current_version():
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{spatioev.__version__}]" in changelog, (
        f"CHANGELOG.md has no section for {spatioev.__version__}"
    )


@pytest.mark.parametrize("field", ["name", "description", "requires-python"])
def test_core_metadata_is_present(field):
    assert _pyproject()["project"].get(field)
