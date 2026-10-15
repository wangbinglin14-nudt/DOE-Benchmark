"""tests.test_metadata

Cross-file version-consistency checks.  A reviewer running
``pip show doe-testbed`` versus ``import benchmark; benchmark.__version__``
must not see drift.  This test catches the common forgetting-one-of-four
mistake that bit us between v0.3 and v0.4.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_pyproject_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', _read(ROOT / "pyproject.toml"), flags=re.M)
    assert m, "version = '…' line missing from pyproject.toml"
    return m.group(1)


def _extract_init_version(relpath: str) -> str:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', _read(ROOT / relpath))
    assert m, f"__version__ missing in {relpath}"
    return m.group(1)


def _extract_changelog_top_version() -> str:
    # Match the first '## [x.y.z] — date' header (skip '## [Unreleased]').
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        pytest.skip("CHANGELOG.md not found")
    for line in _read(changelog).splitlines():
        m = re.match(r"^## \[([^\]]+)\]", line)
        if m and m.group(1).lower() != "unreleased":
            return m.group(1)
    raise AssertionError("No tagged version header in CHANGELOG.md")


def _normalise(v: str) -> str:
    """Convert both ``0.4.0-dev`` and ``0.4.0.dev0`` to the same tuple.

    PEP 440 prefers ``0.4.0.dev0``, CalVer/CFF commonly use ``0.4.0-dev``;
    we accept either but expose a single canonical form to compare.
    """
    return re.sub(r"[.\-]?dev\d*$", ".dev", v.strip())


def test_all_version_files_agree():
    versions = {
        "pyproject.toml": _extract_pyproject_version(),
        "doe_testbed/__init__.py": _extract_init_version("doe_testbed/__init__.py"),
        "benchmark/__init__.py": _extract_init_version("benchmark/__init__.py"),
    }
    changelog = ROOT / "CHANGELOG.md"
    if changelog.exists():
        versions["CHANGELOG.md"] = _extract_changelog_top_version()
    normalised = {k: _normalise(v) for k, v in versions.items()}
    unique = set(normalised.values())
    assert len(unique) == 1, (
        "Version drift detected:\n"
        + "\n".join(f"  {k:30s}  {v}  (normalised: {normalised[k]})"
                    for k, v in versions.items())
    )


def test_no_stale_placeholder_orcid_in_shipped_metadata():
    """ORCID placeholders are allowed in the paper drafts (annotated in
    REVIEWER_CRITIQUE.md §D2) but must not slip into package metadata.
    """
    # CITATION.cff *may* carry a placeholder pre-release; skip with an
    # explicit TODO that the release workflow checks.
    pyproject = _read(ROOT / "pyproject.toml")
    assert "PLACEHOLDER" not in pyproject, (
        "pyproject.toml contains PLACEHOLDER — refuse to ship"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
