"""
scripts/check_doc_links.py
============================
Static check: every relative Markdown link inside the repository
resolves to an existing file or anchor.

Why this exists
---------------
After the 2026-05-23 root consolidation, `C:/DOE-benchmark` carries a
non-trivial `docs/` + `paper/` documentation tree (`PROJECT_STATUS.md`,
`STATISTICAL_ANALYSIS_PLAN.md`, etc.).  These files cross-link each other,
the README files cross-link them, and `scripts/README.md` points at the
one-click pipeline.  A typo or a renamed file silently turns into a dead
link without this guard.

Coverage
--------
- All live-documentation `*.md` files in the repo, except generated or
  auxiliary trees inside ``.git``, ``.agents``, ``__pycache__``,
  ``.pytest_cache``, ``.pytest_tmp``, ``logs/``, ``paper-reviews/``,
  ``review_nature_style/``, ``tmp/`` and ``benchmark/results/``.
- Every relative link of the form ``[text](path)`` and every reference
  link target (``[label]: path``).
- Anchor fragments (``... #section``) are stripped before file-existence
  is checked; anchor *contents* are not validated (Markdown anchor
  generation is renderer-specific).
- External URLs (``http://``, ``https://``, ``mailto:``, ``ftp:``) are
  ignored.

Exit codes
----------
- 0 — no broken links.
- 1 — at least one broken link; the offenders are printed to stdout.
- 2 — CLI / runtime error.

Usage
-----
    python scripts/check_doc_links.py            # scan the whole repo
    python scripts/check_doc_links.py --root docs ../manuscripts
    python scripts/check_doc_links.py --quiet    # only print broken links

This script is exercised by ``benchmark/tests/test_doc_links.py`` so
regressions fail CI before they reach a reviewer.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[2]

# Inline link:    [text](target "optional title")
# Reference link target: [label]: target "optional title"
INLINE_LINK_RE = re.compile(r"\[[^\]]*?\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REF_LINK_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s*<?([^>\s]+)>?", re.MULTILINE)

# Directories whose .md files are *not* part of the live documentation
# (archived results, build caches, generated reports).  Prune these so
# we do not flag stale links inside historical artefacts.
DEFAULT_EXCLUDE_DIRS = (
    ".git",
    ".agents",
    "__pycache__",
    ".pytest_cache",
    ".pytest_tmp",
    "logs",
    "node_modules",
    "paper-reviews",
    "submission",  # Generated delivery snapshots are checked by their own manifests.
    "review_nature_style",
    "tmp",
    "_tmp",
    "code/_tmp",
    "code/logs",
    "_archive",
    # benchmark/results/<run>/REPORT.md files reference results CSVs by
    # relative path that may move with the run; treat them as data, not
    # documentation.
    "code/benchmark/results",
)


def _is_external(target: str) -> bool:
    return target.startswith(
        ("http://", "https://", "mailto:", "ftp://", "ftps://", "tel:", "//")
    )


def _iter_markdown_files(roots: Sequence[Path], exclude: Sequence[str]) -> Iterable[Path]:
    """Walk every ``*.md`` under each root, pruning excluded directories."""
    excl_resolved = {(ROOT / e).resolve() for e in exclude}
    for root in roots:
        for path in sorted(root.rglob("*.md")):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            # Skip if any ancestor of ``path`` is in the exclude set.
            if any(resolved == ex or ex in resolved.parents for ex in excl_resolved):
                continue
            yield path


def _extract_links(text: str) -> List[str]:
    """Extract every (target) string from inline + reference links."""
    raw: List[str] = []
    raw += INLINE_LINK_RE.findall(text)
    raw += REF_LINK_DEF_RE.findall(text)
    return raw


def check_file(md: Path) -> List[tuple[str, str]]:
    """Return list of (link, reason) for broken links in this file."""
    try:
        text = md.read_text(encoding="utf-8")
    except OSError as exc:
        return [("<file>", f"cannot read: {exc}")]

    bad: List[tuple[str, str]] = []
    for raw in _extract_links(text):
        link = raw.strip()
        if not link or link.startswith("#"):
            # Empty or pure anchor link — not a file reference.
            continue
        if _is_external(link):
            continue
        # Strip anchor fragment (#section) and query (?query).
        target_str = link.split("#", 1)[0].split("?", 1)[0]
        if not target_str:
            continue
        # A Markdown file may be reached through the workspace's paper junction.
        # Relative links belong to the actual file's directory, not the alias.
        target = (md.resolve().parent / target_str).resolve()
        if not target.exists():
            bad.append((link, "target does not exist"))
    return bad


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Verify every relative Markdown link in the repo resolves "
            "to an existing file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--root",
        type=Path,
        nargs="+",
        default=[ROOT],
        help="One or more directories to scan (default: repo root).",
    )
    p.add_argument(
        "--exclude",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_DIRS),
        help="Directories (relative to repo root) to skip.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print broken links, not the running counter.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [r.resolve() for r in args.root]
    n_files = 0
    n_broken = 0
    n_links = 0

    for md in _iter_markdown_files(roots, args.exclude):
        n_files += 1
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        n_links += len(_extract_links(text))
        bad = check_file(md)
        if bad:
            n_broken += len(bad)
            print(f"\n[BROKEN] {md.relative_to(ROOT).as_posix()}")
            for link, reason in bad:
                print(f"    -> {link}  ({reason})")

    if not args.quiet:
        print(
            f"\nScanned {n_files} markdown file(s); "
            f"{n_links} link(s) total; {n_broken} broken."
        )
    if n_broken:
        print("\nresult: FAIL — fix the broken links above.")
        return 1
    if not args.quiet:
        print("result: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
