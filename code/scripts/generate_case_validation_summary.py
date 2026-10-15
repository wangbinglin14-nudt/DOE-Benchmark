"""Generate a human-readable case-validation audit summary.

The machine-readable source of truth is ``docs/case_validation_status.csv``.
This script joins that ledger to the actual ``案例正确性校验`` folder so a
reviewer can see which cases had external notes, how many reference files were
present, which targeted physics tests pin the fixes, and what remains to do.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "docs" / "case_validation_status.csv"
DEFAULT_NOTE_SOURCES = ROOT / "docs" / "case_validation_note_sources.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "CASE_VALIDATION_SUMMARY.md"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--note-sources-csv", type=Path, default=DEFAULT_NOTE_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail before writing if note folders, targeted tests, or "
            "unreferenced validation-note folders make the summary incomplete."
        ),
    )
    return parser.parse_args(argv)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _files_for_note_path(note_path: str) -> tuple[list[Path], list[Path]]:
    if note_path == "NONE":
        return [], []
    raw = Path(note_path)
    root = raw if raw.is_absolute() else ROOT / raw
    if not root.exists():
        return [], []
    files = [path for path in root.iterdir() if path.is_file()]
    docx = [
        path for path in files
        if path.suffix.lower() == ".docx" and not path.name.startswith("~$")
    ]
    refs = [
        path for path in files
        if not (path.suffix.lower() == ".docx" or path.name.startswith("~$"))
    ]
    return docx, refs


def _validation_note_dirs() -> set[Path]:
    root = ROOT / "案例正确性校验"
    if not root.exists():
        return set()
    return {
        path.parent.resolve()
        for path in root.rglob("*.docx")
        if not path.name.startswith("~$")
    }


def _registered_case_metadata() -> dict[str, tuple[str, int]]:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import doe_testbed as db

    return {
        meta.case_id: (str(meta.domain), int(meta.dim_input))
        for meta in db.list_cases()
    }


def validation_errors(rows: list[dict[str, str]]) -> list[str]:
    """Return evidence-chain errors that would make the summary misleading."""
    errors: list[str] = []
    documented_dirs: set[Path] = set()
    registered = _registered_case_metadata()
    documented_cases = [row.get("case_id", "") for row in rows]
    duplicate_cases = sorted(
        case_id for case_id, count in Counter(documented_cases).items() if count > 1
    )
    if duplicate_cases:
        errors.append(f"case-validation ledger has duplicate case rows: {duplicate_cases}")
    missing_cases = sorted(set(registered) - set(documented_cases))
    extra_cases = sorted(set(documented_cases) - set(registered))
    if missing_cases:
        errors.append(f"case-validation ledger is missing registered cases: {missing_cases}")
    if extra_cases:
        errors.append(f"case-validation ledger contains unregistered cases: {extra_cases}")

    for row in rows:
        case_id = row.get("case_id", "<missing case_id>")
        risk = row.get("risk_level", "")
        disposition = row.get("final_disposition", "").lower()
        if case_id in registered:
            expected_domain, expected_dim = registered[case_id]
            if row.get("domain") != expected_domain:
                errors.append(
                    f"{case_id}: ledger domain {row.get('domain')!r} "
                    f"does not match registered domain {expected_domain!r}"
                )
            try:
                observed_dim = int(row.get("dim", ""))
            except ValueError:
                errors.append(f"{case_id}: ledger dim {row.get('dim')!r} is not an integer")
            else:
                if observed_dim != expected_dim:
                    errors.append(
                        f"{case_id}: ledger dim {observed_dim} "
                        f"does not match registered dim {expected_dim}"
                    )

        note_path = row.get("validation_note_path", "NONE")
        if note_path != "NONE":
            raw_note_dir = Path(note_path)
            note_dir = (
                raw_note_dir if raw_note_dir.is_absolute() else ROOT / raw_note_dir
            ).resolve()
            documented_dirs.add(note_dir)
            if not note_dir.exists():
                errors.append(f"{case_id}: validation note path missing: {note_path}")
            else:
                docx, refs = _files_for_note_path(note_path)
                if not docx:
                    errors.append(
                        f"{case_id}: validation note folder has no real docx: {note_path}"
                    )
                if not refs:
                    errors.append(
                        f"{case_id}: validation note folder has no reference attachments: {note_path}"
                    )
        elif risk == "P2" and not (
            "plausibility only" in disposition
            or "analytic" in disposition
            or "control" in disposition
        ):
            errors.append(
                f"{case_id}: P2 row without a prior validation-note folder must remain "
                "plausibility-only or explicitly labelled as an analytical/control "
                "case; external-validation claims require a registered validation-note folder"
            )

        test_path = row.get("targeted_physics_test", "NONE")
        if test_path != "NONE" and not (ROOT / test_path).exists():
            errors.append(f"{case_id}: targeted physics test missing: {test_path}")

    missing_from_ledger = sorted(_validation_note_dirs() - documented_dirs)
    for path in missing_from_ledger:
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(path)
        errors.append(f"validation note folder is not referenced by ledger: {rel}")
    return errors


def note_source_errors(note_sources_csv: Path) -> list[str]:
    try:
        generator_path = ROOT / "scripts" / "update_case_validation_note_sources.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_case_validation_note_sources_for_summary",
            generator_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {generator_path}")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        expected = generator.csv_text(generator.current_rows())
    except Exception as exc:
        return [f"could not compute validation note-source fingerprints: {type(exc).__name__}: {exc}"]
    if not note_sources_csv.exists():
        return [f"validation note-source fingerprint CSV missing: {note_sources_csv}"]
    observed = note_sources_csv.read_text(encoding="utf-8")
    if observed != expected:
        return [
            "validation note-source fingerprint CSV is stale; run "
            "python scripts/update_case_validation_note_sources.py --write"
        ]
    return []


def _md_path(path_text: str) -> str:
    if path_text == "NONE":
        return "NONE"
    if " " in path_text:
        return f"`{path_text}`"
    return f"[{path_text}](../{path_text.replace(' ', '%20')})"


def _md_file(path: Path) -> str:
    return f"`{path.name}`"


def _cross_validation_inventory(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    """Return per-case checked-in cross-validation write-up assets."""
    inventory: list[tuple[str, str, str]] = []
    root = ROOT / "tests" / "_cross_validation"
    for row in sorted(rows, key=lambda item: item["case_id"]):
        case_id = row["case_id"]
        py_path = root / f"{case_id}.py"
        md_path = root / f"{case_id}.md"
        py_rel = py_path.relative_to(ROOT).as_posix() if py_path.exists() else "NONE"
        md_rel = md_path.relative_to(ROOT).as_posix() if md_path.exists() else "NONE"
        inventory.append((case_id, py_rel, md_rel))
    return inventory


def _risk_counts(rows: list[dict[str, str]]) -> str:
    counts = Counter(row["risk_level"] for row in rows)
    return ", ".join(f"{risk}: {counts.get(risk, 0)}" for risk in ["P0", "P1", "P2"])


def evidence_tier(row: dict[str, str]) -> str:
    """Classify how strong the current case-validation evidence is."""
    has_note = row.get("validation_note_path") != "NONE"
    disposition = row.get("final_disposition", "").lower()
    scope = row.get("external_validation_scope", "").lower()
    if has_note and "plausibility only" not in disposition:
        if row.get("risk_level") == "P1":
            return "external_note_targeted_regression"
        return "external_note_analytic_control"
    if has_note:
        return "external_note_plausibility"
    if "analytic" in disposition or "control" in disposition:
        return "targeted_analytical_control"
    if "plausibility only" in disposition or "no prior validation-note" in scope:
        return "targeted_plausibility_only"
    return "targeted_regression_only"


def _evidence_tier_rows(rows: list[dict[str, str]]) -> list[tuple[str, int, str]]:
    descriptions = {
        "external_note_targeted_regression": (
            "Prior validation-note/reference folder plus targeted physics tests; "
            "documentary and regression evidence only; neither source-page verification "
            "nor experimental validation is established by this label. See adjacent case evidence notes."
        ),
        "external_note_analytic_control": (
            "Prior validation-note/reference folder plus targeted tests, but retained "
            "as lower-priority analytic/plausibility control."
        ),
        "external_note_plausibility": (
            "Prior validation-note/reference folder exists, but disposition is still "
            "plausibility-only."
        ),
        "targeted_regression_only": (
            "Targeted tests and cross-validation exist, but no prior validation-note "
            "folder is registered."
        ),
        "targeted_analytical_control": (
            "Targeted tests pin closed-form or table-form equations against public "
            "analytical references, but no prior validation-note folder is registered; "
            "safe for analytical-control claims, not external case-validation claims."
        ),
        "targeted_plausibility_only": (
            "Targeted tests pin local physics trends, but no prior validation-note "
            "folder exists; do not claim strong external case validation."
        ),
    }
    counts = Counter(evidence_tier(row) for row in rows)
    return [
        (tier, counts.get(tier, 0), descriptions[tier])
        for tier in descriptions
        if counts.get(tier, 0)
    ]


def _domain_coverage_rows(rows: list[dict[str, str]]) -> list[tuple[str, int, int, int, int]]:
    domains = sorted({row["domain"] for row in rows})
    coverage = []
    for domain in domains:
        domain_rows = [row for row in rows if row["domain"] == domain]
        note_rows = [row for row in domain_rows if row["validation_note_path"] != "NONE"]
        targeted_rows = [row for row in domain_rows if row["targeted_physics_test"] != "NONE"]
        p2_rows = [row for row in domain_rows if row["risk_level"] == "P2"]
        coverage.append((domain, len(domain_rows), len(note_rows), len(targeted_rows), len(p2_rows)))
    return coverage


def render_markdown(rows: list[dict[str, str]], *, note_source_rows: int | None = None) -> str:
    rows = sorted(rows, key=lambda row: (row["risk_level"], row["domain"], row["case_id"]))
    note_rows = [row for row in rows if row["validation_note_path"] != "NONE"]
    targeted_rows = [row for row in rows if row["targeted_physics_test"] != "NONE"]
    resweep_rows = [
        row for row in rows
        if "rerun" in row["next_action"].lower() or "resweep" in row["next_action"].lower()
    ]

    lines = [
        "# Case Validation Summary",
        "",
        "This file is auto-generated by `scripts/generate_case_validation_summary.py`",
        "from `docs/case_validation_status.csv` and the `案例正确性校验/`",
        "review folders. Regenerate it after case-code, case-ledger, or",
        "validation-folder changes.",
        "",
        "## Summary",
        "",
        f"- Cases in ledger: {len(rows)}",
        f"- Cases with prior validation-note folders: {len(note_rows)}",
        f"- Validation note/reference files fingerprinted: {note_source_rows if note_source_rows is not None else 'not checked'}",
        f"- Cases with targeted physics tests: {len(targeted_rows)}",
        f"- Risk counts: {_risk_counts(rows)}",
        f"- Cases whose next action mentions rerun/resweep: {len(resweep_rows)}",
        "",
        "## Evidence Tiers",
        "",
        "These tiers prevent a passing ledger from being read as a claim that every",
        "case has the same external validation strength. P2/plausibility-only rows",
        "are suitable for benchmark stress tests, but need additional literature or",
        "reference checks before strong case-level validation claims.",
        "",
        "| Evidence tier | Cases | Interpretation |",
        "|---|---:|---|",
    ]
    for tier, count, description in _evidence_tier_rows(rows):
        lines.append(f"| `{tier}` | {count} | {description} |")

    lines.extend([
        "",
        "## Domain Evidence Coverage",
        "",
        "| Domain | Cases | Prior-note cases | Targeted-test cases | P2/plausibility-only risk rows |",
        "|---|---:|---:|---:|---:|",
    ])
    for domain, n_cases, n_notes, n_targeted, n_p2 in _domain_coverage_rows(rows):
        lines.append(
            f"| {domain} | {n_cases} | {n_notes} | {n_targeted} | {n_p2} |"
        )

    lines.extend([
        "",
        "## Case Ledger",
        "",
        "| Case | Risk | Evidence tier | Note folder | Note docx | Reference files | Targeted test | Disposition | Next action |",
        "|---|---:|---|---|---:|---:|---|---|---|",
    ])

    for row in rows:
        docx, refs = _files_for_note_path(row["validation_note_path"])
        lines.append(
            "| "
            f"`{row['case_id']}` | "
            f"{row['risk_level']} | "
            f"`{evidence_tier(row)}` | "
            f"{_md_path(row['validation_note_path'])} | "
            f"{len(docx)} | "
            f"{len(refs)} | "
            f"{_md_path(row['targeted_physics_test'])} | "
            f"{row['final_disposition']} | "
            f"{row['next_action']} |"
        )

    lines.extend([
        "",
        "## Prior Validation-Note Inventory",
        "",
        "This inventory is generated from the referenced note folders so reviewers",
        "can trace each manual validation note to its supporting files.",
        "",
    ])
    for note_path in sorted({row["validation_note_path"] for row in note_rows}):
        docx, refs = _files_for_note_path(note_path)
        lines.extend([
            f"### {_md_path(note_path)}",
            "",
            f"- Note docx: {', '.join(_md_file(path) for path in sorted(docx)) or 'NONE'}",
            f"- Reference files: {', '.join(_md_file(path) for path in sorted(refs)) or 'NONE'}",
            "",
        ])

    cross_validation_refs = ROOT / "tests" / "_cross_validation" / "references.bib"
    refs_rel = (
        cross_validation_refs.relative_to(ROOT).as_posix()
        if cross_validation_refs.exists()
        else "NONE"
    )
    lines.extend([
        "## Supporting Cross-Validation Inventory",
        "",
        "These checked-in materials document independent formulas, reference",
        "points, and qualitative physics trends used when authoring targeted",
        "physics tests. They are supporting evidence only: cases without a",
        "registered validation-note folder remain in their declared evidence",
        "tier and must not be described as externally validated solely from",
        "these files.",
        "",
        f"- Shared bibliography: {_md_path(refs_rel)}",
        "",
        "| Case | Python write-up | Markdown note |",
        "|---|---|---|",
    ])
    for case_id, py_rel, md_rel in _cross_validation_inventory(rows):
        lines.append(f"| `{case_id}` | {_md_path(py_rel)} | {_md_path(md_rel)} |")

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_rows(args.status_csv.resolve())
    errors = validation_errors(rows)
    note_sources_csv = args.note_sources_csv.resolve()
    errors.extend(note_source_errors(note_sources_csv))
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        if args.strict:
            print("[ERROR] strict case-validation summary check failed; not writing output.")
            return 1
    note_source_rows = len(_read_rows(note_sources_csv)) if note_sources_csv.exists() else None
    markdown = render_markdown(rows, note_source_rows=note_source_rows)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=str(output.parent)))
    tmp_output = tmp_parent / output.name
    try:
        tmp_output.write_text(markdown, encoding="utf-8")
        shutil.move(str(tmp_output), str(output))
    finally:
        if tmp_parent.exists():
            shutil.rmtree(tmp_parent)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
