"""check_paper_draft_ready.py
================================

Positive gate for the pre-results manuscript package.

``check_submission_ready.py`` is intentionally strict and must fail while the
full sweep is incomplete.  This companion script answers a different question:
is the Nature-family draft package complete enough to keep writing safely before
the final ``results.csv`` and signed decision ledger exist?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.project_layout import paper_file, metadata_file
PAPER = ROOT / "paper"


def _load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _filesystem_error(exc: OSError) -> str:
    target = getattr(exc, "filename", None) or getattr(exc, "filename2", None) or "<unknown>"
    return (
        f"filesystem access failed while checking {target}: "
        f"{type(exc).__name__}: {exc}. Fix the file/directory ACL or rebuild "
        "the generated paper artifact package, then rerun this draft gate."
    )


def _result_pending_count(text: str) -> int:
    return len(re.findall(r"\bRESULT(?:\\_|\_)?PENDING\b", text))


STATISTICAL_EXPLANATION_PATTERNS = (
    (
        r"Spearman\s+\$\\rho\$\s+is\s+interpreted\s+as\s+the\s+D1\s+effect-size",
        "D1 Spearman rho effect-size interpretation",
    ),
    (
        r"Clopper--Pearson\s+(?:one-sided\s+)?upper\s+(?:confidence\s+)?bound",
        "D2 Clopper-Pearson upper-bound interpretation",
    ),
    (
        r"limited\s+power",
        "D2 fixed-topology limited-power caveat",
    ),
    (
        r"Bonferroni\s+and\s+BH-FDR\s+are\s+retained\s+only\s+as\s+sensitivity",
        "multiplicity sensitivity policy",
    ),
    (
        r"Test-set\s+Monte-Carlo\s+noise\s+is\s+reported",
        "test-set MC-noise reporting hook",
    ),
    (
        r"cluster-vs-i\.i\.d\.\s+CI-width\s+ratio",
        "D3 cluster-vs-iid sensitivity hook",
    ),
    (
        r"Supplementary\s+tables\s+describe.*multiple-testing\s+policy",
        "Supplementary Information multiple-testing index",
    ),
    (
        r"Supplementary\s+tables\s+describe.*D2\s+exploratory-topology\s+denominator\s+policy",
        "Supplementary Information D2 policy index",
    ),
    (
        r"Supplementary\s+tables\s+describe.*test-set\s+Monte-Carlo\s+noise\s+floor",
        "Supplementary Information MC-noise index",
    ),
    (
        r"evidence\s+tiers.*P2\s+plausibility-only\s+cases.*not\s+strong\s+external\s+case-level\s+validation\s+claims",
        "case-validation evidence-tier caveat",
    ),
)

SIGNIFICANCE_WORD_MIN = 100
SIGNIFICANCE_WORD_MAX = 200
COVER_LETTER_WORD_MAX = 450
COVER_LETTER_REQUIRED_PATTERNS = (
    (r"\bDear\s+Editor\b", "editor salutation"),
    (r"\bYours\s+sincerely\b", "closing signature line"),
    (r"DOE-Benchmark", "manuscript title"),
    (r"physics-grounded", "physics-grounded positioning"),
    (r"20\s+design-of-experiments", "20-method benchmark scope"),
    (r"34\s+physics-grounded", "34-case benchmark scope"),
    (r"immutable\s+archive", "archive/data availability hook"),
)
PREREG_DECISION_REQUIRED_PATTERNS = (
    (r"##\s+Provenance", "provenance section"),
    (
        r"Run\s+manifest.*benchmark/results/full_benchmark/RUN_MANIFEST\.json",
        "run-manifest provenance hook",
    ),
    (r"##\s+D1\b.*Spearman", "D1 decision section"),
    (r"\|\s*dimension\s*\|.*\|\s*topology\s*\|.*\|\s*random\s*\|", "D1 pairing rows"),
    (r"##\s+D2\b.*10d\s+rule", "D2 decision section"),
    (r"pass_count\s*/\s*7", "D2 pass-count field"),
    (r"##\s+D3\b.*failure", "D3 decision section"),
    (r"Mean\s+signed\s+bias", "D3 signed-bias field"),
    (r"##\s+Holm-Bonferroni\s+Ledger", "Holm-Bonferroni ledger"),
    (r"\|\s*D1\s*\|.*\|\s*D2\s*\|.*\|\s*D3\s*\|", "Holm D1/D2/D3 rows"),
    (r"##\s+Falsification\s+Log", "falsification log"),
    (r"##\s+Sign-off", "sign-off section"),
    (r"attest.*RUN_MANIFEST\.json", "archive-pinned attestation"),
    (
        r"no\s+D1/D2/D3\s+manuscript\s+text\s+was\s+rewritten\s+before\s+this\s+file\s+was\s+signed",
        "pre-manuscript-rewrite attestation",
    ),
)


def _markdown_body(text: str) -> str:
    """Return prose body text, excluding headings and status blockquotes."""
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith(">"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def _check_draft_package_impl(paper_dir: Path = PAPER) -> list[str]:
    """Return draft-readiness errors for the materialized paper package."""
    errors: list[str] = []
    submission = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_ready_for_draft",
    )
    audit_macros = _load_module("scripts/audit_macros.py", "audit_macros_for_draft")

    required = [
        *submission.REQUIRED_SUBMISSION_SOURCES,
        *submission.REQUIRED_SI_FILES,
        "README.md",
        "figure_manifest.json",
        "numbers.json",
        "numbers.tex",
    ]
    for rel in required:
        path = paper_file(paper_dir, rel)
        if not path.exists():
            errors.append(f"draft package missing required file: {rel}")
        elif path.stat().st_size == 0:
            errors.append(f"draft package required file is empty: {rel}")
    for err in submission.check_required_si_files(paper_dir):
        errors.append(f"draft SI package drift: {err}")

    for rel in submission.REQUIRED_SUBMISSION_SOURCES:
        path = paper_file(paper_dir, rel)
        if not path.exists():
            continue
        text = _read(path)
        pending_count = _result_pending_count(text)
        if pending_count == 0:
            errors.append(
                f"{rel}: draft source has no RESULT_PENDING marker; "
                "either mark unresolved claims or run submission-ready for final mode"
            )
        for lineno, kind, line in submission.scan_file(path):
            if kind != "RESULT_PENDING":
                errors.append(f"{rel}:{lineno}: unresolved {kind}: {line}")

    main = paper_dir / "nature_cs_main.tex"
    if main.exists():
        text = _read(main)
        required_sections = [
            r"\\begin\{abstract\}",
            r"\\section\*\{Results\}",
            r"\\section\*\{Discussion\}",
            r"\\section\*\{(?:Methods|Online Methods)\}",
            r"\\section\*\{Data availability\}",
            r"\\section\*\{Code availability\}",
            r"\\section\*\{Author contributions\}",
            r"\\section\*\{Competing interests\}",
        ]
        for pattern in required_sections:
            if not re.search(pattern, text):
                errors.append(f"nature_cs_main.tex missing section pattern: {pattern}")
        for pattern, label in STATISTICAL_EXPLANATION_PATTERNS:
            if not re.search(pattern, text, flags=re.S):
                errors.append(
                    f"nature_cs_main.tex missing statistical explanation: {label}"
                )
        for i in range(1, 7):
            rel = f"figures/pdf/fig{i}.pdf"
            if rel not in text:
                errors.append(f"nature_cs_main.tex does not reference {rel}")
            elif not (paper_file(paper_dir, rel)).exists():
                errors.append(f"nature_cs_main.tex references missing figure: {rel}")

        numbers = paper_dir / "numbers.tex"
        missing_macros = (
            audit_macros.macros_used(main)
            - (audit_macros.macros_defined(numbers) if numbers.exists() else set())
            - audit_macros.macros_defined(main)
            - audit_macros.HELPERS
        )
        if missing_macros:
            errors.append(
                "nature_cs_main.tex has undefined number macros: "
                f"{sorted(missing_macros)}"
            )

    significance = paper_file(paper_dir, "significance_statement.md")
    if significance.exists():
        n_words = _word_count(_markdown_body(_read(significance)))
        if not (SIGNIFICANCE_WORD_MIN <= n_words <= SIGNIFICANCE_WORD_MAX):
            errors.append(
                "significance_statement.md body word count outside Nature-family "
                f"draft range {SIGNIFICANCE_WORD_MIN}-{SIGNIFICANCE_WORD_MAX}: "
                f"observed {n_words}"
            )

    cover_letter = paper_file(paper_dir, "cover_letter.md")
    if cover_letter.exists():
        text = _read(cover_letter)
        n_words = _word_count(_markdown_body(text))
        if n_words > COVER_LETTER_WORD_MAX:
            errors.append(
                "cover_letter.md body word count exceeds Nature-family "
                f"draft limit {COVER_LETTER_WORD_MAX}: observed {n_words}"
            )
        for pattern, label in COVER_LETTER_REQUIRED_PATTERNS:
            if not re.search(pattern, text, flags=re.I | re.S):
                errors.append(f"cover_letter.md missing required element: {label}")
        number_macros = sorted(set(re.findall(r"\\Num[A-Za-z0-9]+", text)))
        if number_macros:
            errors.append(
                "cover_letter.md contains LaTeX number macros; write plain text "
                f"for editors: {number_macros}"
            )

    decisions = paper_file(paper_dir, "preregistration_decisions.md")
    if decisions.exists():
        text = _read(decisions)
        normalized = re.sub(r"\s+", " ", text)
        for pattern, label in PREREG_DECISION_REQUIRED_PATTERNS:
            if not re.search(pattern, normalized, flags=re.I | re.S):
                errors.append(
                    "preregistration_decisions.md missing required decision-ledger "
                    f"element: {label}"
                )

    numbers_json = paper_dir / "numbers.json"
    numbers_tex = paper_dir / "numbers.tex"
    if numbers_json.exists() or numbers_tex.exists():
        for err in submission.check_numbers_tex_consistency(numbers_tex, numbers_json):
            errors.append(f"draft numbers package drift: {err}")

    figure_manifest = paper_dir / "figure_manifest.json"
    if figure_manifest.exists():
        for err in submission.check_figure_manifest(
            figure_manifest,
            paper_dir,
            allow_draft_figdata=True,
        ):
            errors.append(f"draft figure package drift: {err}")

    if numbers_json.exists():
        try:
            numbers = json.loads(_read(numbers_json))
        except Exception as exc:
            errors.append(f"numbers.json is not parseable JSON: {type(exc).__name__}: {exc}")
        else:
            expected = {
                "nCases": 34,
                "nMethods": 20,
                "nRepeats": 30,
                "nBudgets": 4,
                "budgetList": "5, 10, 20, 50",
                "totalCells": 81600,
                "primaryFailurePolicy": "penalized_failed_in_denominator",
                "dOneRankPolicy": "complete_case",
            }
            for key, value in expected.items():
                if numbers.get(key) != value:
                    errors.append(
                        f"numbers.json draft protocol value {key} mismatch: "
                        f"expected {value!r}, observed {numbers.get(key)!r}"
                    )
    return errors


def check_draft_package(paper_dir: Path = PAPER) -> list[str]:
    try:
        return _check_draft_package_impl(paper_dir)
    except PermissionError as exc:
        return [_filesystem_error(exc)]
    except OSError as exc:
        if getattr(exc, "winerror", None) == 5:
            return [_filesystem_error(exc)]
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the Nature-family paper draft package is coherent.",
    )
    parser.add_argument("paper_dir", nargs="?", type=Path, default=PAPER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors = check_draft_package(args.paper_dir.resolve())
    for err in errors:
        print(f"[draft] {err}")
    if errors:
        print(f"[result] FAIL - draft package has {len(errors)} issue(s).")
        return 1
    print("[result] PASS - paper draft package is coherent and marked pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
