"""CI guard: every relative Markdown link in the repo resolves.

Wraps ``scripts/check_doc_links.py`` so a broken cross-reference fails
CI before a reviewer encounters it. This is the safety net for public technical
documentation and the ``manuscripts/`` research records. Private plans are not
required public artifacts.

The test:
1. Imports ``scripts/check_doc_links.py`` as a module (no subprocess so
   the test runs fast and shows tracebacks inline on failure).
2. Calls its ``main()`` with ``--quiet`` and asserts exit 0.
3. Additionally pins a small allow-list of "documents that MUST exist"
   so a future ``rm`` cannot quietly delete a key publication-track
   document and have the link checker pass on a vacuous repo.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _import_check_doc_links():
    """Load ``scripts/check_doc_links.py`` as a module without sys.path tricks."""
    path = ROOT / "scripts" / "check_doc_links.py"
    spec = importlib.util.spec_from_file_location("check_doc_links", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_doc_links"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── 1. Every relative link resolves ────────────────────────────────────
def test_all_relative_links_resolve(capsys):
    """Run ``check_doc_links.py --quiet`` and assert exit 0.

    Captures stdout via the pytest ``capsys`` fixture so a regression
    surfaces the offending file(s) inline rather than dumping nothing.
    """
    mod = _import_check_doc_links()
    rc = mod.main(["--quiet"])
    out = capsys.readouterr().out
    assert rc == 0, "scripts/check_doc_links.py reported broken Markdown links:\n" + out


# ─── 2. Pin the publication-track documents ─────────────────────────────
REQUIRED_DOCS = (
    "docs/README.md",
    "docs/CASE_EVIDENCE_INDEX.md",
    "docs/STATISTICAL_ANALYSIS_PLAN.md",
    "docs/REPRODUCIBILITY.md",
    "../manuscripts/protocol/preregistration.md",
    "../manuscripts/protocol/preregistration_amendments.md",
    "../manuscripts/protocol/preregistration_decisions.md",
    "../manuscripts/numbers.tex",
    "../manuscripts/numbers.json",
    "scripts/README.md",
    "scripts/check_doc_links.py",
    "scripts/generate_run_manifest.py",
    "scripts/launch_sweep_local_16gb.ps1",
    "benchmark/README.md",
    "benchmark/tests/test_experiment_contracts.py",
    "../README.md",
)


def test_required_publication_track_docs_exist():
    """A future cleanup should not silently delete a key document.

    The link checker would not catch deletion of, say,
    ``docs/STATISTICAL_ANALYSIS_PLAN.md`` if no other document linked
    to it.  This test pins the canonical set so accidental ``rm``
    fails CI even before any link goes red.
    """
    missing = [p for p in REQUIRED_DOCS if not (ROOT / p).is_file()]
    assert not missing, (
        f"Missing required publication-track documents: {missing}.  "
        "If a removal is intentional, update REQUIRED_DOCS in this test "
        "and update README cross-references in the same commit."
    )


# ─── 3. Every doc has a non-empty body (no accidental zero-byte files) ──
def test_required_docs_have_nonempty_body():
    """Empty .md files would pass the link check trivially; refuse them."""
    empty = []
    for p in REQUIRED_DOCS:
        path = ROOT / p
        if path.is_file() and path.stat().st_size < 200:
            # 200 bytes is the minimum for "front-matter + one paragraph".
            empty.append(f"{p} ({path.stat().st_size} B)")
    assert not empty, (
        "Publication-track documents are unexpectedly small (likely a "
        "truncation accident):\n  " + "\n  ".join(empty)
    )


# ─── 4. Quick smoke on the link extractor itself ────────────────────────
def test_link_extractor_handles_inline_and_reference_styles():
    mod = _import_check_doc_links()
    sample = (
        "[a](docs/x.md)\n"
        "[b][1]\n"
        "[1]: docs/y.md\n"
        "[c](https://example.com)\n"
        "[d](#anchor)\n"
        "[e]: <docs/z.md>\n"
    )
    found = mod._extract_links(sample)
    # Inline + reference link defs should both surface; pure-anchor and
    # external links are surfaced too — the broken-check filters them
    # out at a later stage, not the extraction stage.
    assert "docs/x.md" in found
    assert "docs/y.md" in found
    assert "docs/z.md" in found
    assert "https://example.com" in found
    assert "#anchor" in found


def test_external_links_are_skipped():
    mod = _import_check_doc_links()
    assert mod._is_external("https://example.com")
    assert mod._is_external("http://example.com/x")
    assert mod._is_external("mailto:foo@bar")
    assert mod._is_external("ftp://example.com")
    assert not mod._is_external("docs/x.md")
    assert not mod._is_external("./docs/x.md")
    assert not mod._is_external("../paper/preregistration.md")


def test_anchor_only_link_does_not_trigger_file_check(tmp_path):
    """``[Section](#anchor)`` must be treated as in-doc, not a file ref."""
    mod = _import_check_doc_links()
    md = tmp_path / "x.md"
    md.write_text("# Title\n[jump](#title)\n", encoding="utf-8")
    assert mod.check_file(md) == []


def test_relative_link_to_missing_file_is_flagged(tmp_path):
    mod = _import_check_doc_links()
    md = tmp_path / "x.md"
    md.write_text("[gone](does/not/exist.md)\n", encoding="utf-8")
    bad = mod.check_file(md)
    assert len(bad) == 1
    assert "does/not/exist.md" in bad[0][0]


def test_default_scan_includes_the_public_repository_root():
    mod = _import_check_doc_links()
    assert mod.ROOT == ROOT.parent


def test_temporary_audit_copies_are_not_public_documentation(tmp_path, monkeypatch):
    mod = _import_check_doc_links()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "_tmp").mkdir()
    snapshot = tmp_path / "_tmp" / "README.md"
    snapshot.write_text("[old](missing.md)\n", encoding="utf-8")
    assert list(mod._iter_markdown_files([tmp_path], mod.DEFAULT_EXCLUDE_DIRS)) == []
