from __future__ import annotations

import ast
import csv
from pathlib import Path

import doe_testbed as db
import pytest


ROOT = Path(__file__).resolve().parents[1]
STATUS_CSV = ROOT / "docs" / "case_validation_status.csv"
SOURCES_CSV = ROOT / "docs" / "case_validation_sources.csv"
NOTE_SOURCES_CSV = ROOT / "docs" / "case_validation_note_sources.csv"
VALIDATION_ROOT = ROOT / "\u6848\u4f8b\u6b63\u786e\u6027\u6821\u9a8c"
HAS_VALIDATION_NOTES = VALIDATION_ROOT.exists() and any(
    not path.name.startswith("~$") for path in VALIDATION_ROOT.rglob("*.docx")
)
requires_validation_notes = pytest.mark.skipif(
    not HAS_VALIDATION_NOTES,
    reason="git-ignored local validation-note files are not present in this checkout",
)
REQUIRED_COLUMNS = [
    "case_id",
    "domain",
    "dim",
    "validation_note_path",
    "external_validation_scope",
    "formula_status",
    "code_status",
    "boundary_test_status",
    "targeted_physics_test",
    "risk_level",
    "final_disposition",
    "next_action",
]
VALID_RISKS = {"P0", "P1", "P2"}
VALID_FORMULA_STATUS = {
    "fixed_targeted_tested",
    "targeted_tested_no_code_change",
    "needs_manual_formula_review",
    "cross_validation_only",
}
SOURCE_COLUMNS = [
    "case_id",
    "case_source_path",
    "case_source_sha256",
]
NOTE_SOURCE_COLUMNS = [
    "path",
    "size_bytes",
    "sha256",
]
MIN_P2_PLAUSIBILITY_TEST_FUNCTIONS = 3
MIN_P2_PLAUSIBILITY_PHYSICS_TEST_FUNCTIONS = 2
GENERIC_TEST_NAME_FRAGMENTS = (
    "declared_output_ranges",
    "backend_is_used",
)
LOW_ORDER_SURROGATE_SCOPE_CASES = {
    "battery_thermal_7d",
    "casting_solidification_5d",
    "fixed_bed_reactor_6d",
    "membrane_reactor_5d",
}
VARIABLE_BEAM_SCOPE_CASES = {
    "variable_beam_5d",
    "variable_beam_10d",
    "variable_beam_20d",
}
NUMERICAL_CONTROL_SCOPE_CASES = {
    "soil_consolidation_5d",
    "tuned_mass_damper_3d",
}
TRUSS_FEM_SCOPE_CASES = {
    "ten_bar_truss_10d",
}
ANALYTICAL_CONTROL_SCOPE_CASES = {
    "cantilever_beam_5d",
    "fin_heat_sink_4d",
    "hetero_conduction_2d",
    "multilayer_wall_4d",
    "multilayer_wall_8d",
    "multilayer_wall_16d",
    "plate_with_hole_5d",
    "pressure_vessel_4d",
    "supersonic_nozzle_5d",
    "thin_plate_buckling_5d",
}
PLAUSIBILITY_SCOPE_CASES = {
    "dam_seepage_6d",
}


def _rows() -> list[dict[str, str]]:
    with STATUS_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _validation_note_dirs() -> set[Path]:
    """Return directories that contain real validation-note docx files."""
    if not VALIDATION_ROOT.exists():
        return set()
    return {
        path.parent
        for path in VALIDATION_ROOT.rglob("*.docx")
        if not path.name.startswith("~$")
    }


def _test_function_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
    return sorted(names)


def _cross_validation_file(case_id: str) -> Path:
    return ROOT / "tests" / "_cross_validation" / f"{case_id}.py"


def test_case_validation_status_schema_is_stable():
    with STATUS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REQUIRED_COLUMNS


def test_case_validation_status_covers_registered_cases_once():
    rows = _rows()
    documented = [row["case_id"] for row in rows]
    registered = sorted(meta.case_id for meta in db.list_cases())

    assert sorted(documented) == registered
    assert len(documented) == len(set(documented)) == 34


def test_case_validation_source_fingerprints_match_registered_cases():
    import importlib
    import hashlib

    with SOURCES_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == SOURCE_COLUMNS
        rows = list(reader)

    by_case = {row["case_id"]: row for row in rows}
    registered = sorted(meta.case_id for meta in db.list_cases())
    assert sorted(by_case) == registered

    for case_id in registered:
        case = db.get_case(case_id)
        module = importlib.import_module(type(case).__module__)
        source = Path(str(module.__file__)).resolve()
        rel = source.relative_to(ROOT).as_posix()
        sha = hashlib.sha256(source.read_bytes()).hexdigest()

        row = by_case[case_id]
        assert row["case_source_path"] == rel
        assert row["case_source_sha256"] == sha


def test_update_case_validation_sources_cli_check_and_write(tmp_path):
    from scripts.update_case_validation_sources import main

    out = tmp_path / "case_validation_sources.csv"

    assert main(["--output", str(out)]) == 1
    assert main(["--output", str(out), "--write"]) == 0
    assert main(["--output", str(out)]) == 0

    text = out.read_text(encoding="utf-8")
    out.write_text(text.replace(",", ",stale-", 1), encoding="utf-8")

    assert main(["--output", str(out)]) == 1


@requires_validation_notes
def test_case_validation_note_source_fingerprints_are_current():
    from scripts.update_case_validation_note_sources import current_rows

    with NOTE_SOURCES_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == NOTE_SOURCE_COLUMNS
        observed = list(reader)

    assert observed == current_rows()
    assert any(row["path"].endswith(".docx") for row in observed)


def test_update_case_validation_note_sources_cli_check_and_write(tmp_path):
    from scripts.update_case_validation_note_sources import main

    out = tmp_path / "case_validation_note_sources.csv"

    assert main(["--output", str(out)]) == 1
    assert main(["--output", str(out), "--write"]) == 0
    assert main(["--output", str(out)]) == 0

    text = out.read_text(encoding="utf-8")
    out.write_text(text.replace(",", ",stale-", 1), encoding="utf-8")

    assert main(["--output", str(out)]) == 1


@requires_validation_notes
def test_generate_case_validation_summary_from_ledger(tmp_path):
    from scripts.generate_case_validation_summary import main

    out = tmp_path / "summary.md"

    assert main(["--output", str(out), "--strict"]) == 0
    text = out.read_text(encoding="utf-8")

    assert "# Case Validation Summary" in text
    assert "- Cases in ledger: 34" in text
    assert "- Cases with targeted physics tests: 34" in text
    with NOTE_SOURCES_CSV.open("r", encoding="utf-8", newline="") as handle:
        note_source_count = sum(1 for _ in csv.DictReader(handle))
    assert f"- Validation note/reference files fingerprinted: {note_source_count}" in text
    assert "- Risk counts: P0: 0, P1: 17, P2: 17" in text
    assert "## Evidence Tiers" in text
    assert "| `external_note_targeted_regression` | 17 |" in text
    assert "| `external_note_analytic_control` | 3 |" in text
    assert "| `targeted_analytical_control` | 2 |" in text
    assert "| `targeted_plausibility_only` | 12 |" in text
    assert "do not claim strong external case validation" in text
    assert "| Case | Risk | Evidence tier | Note folder |" in text
    assert "| `cantilever_beam_5d` | P1 | `external_note_targeted_regression` |" in text
    assert "| `aircraft_conceptual_8d` | P2 | `targeted_plausibility_only` |" in text
    assert "| `multilayer_wall_4d` | P2 | `external_note_analytic_control` |" in text
    assert "| `pressure_vessel_4d` | P2 | `targeted_analytical_control` |" in text
    assert "| `soil_consolidation_5d` | P2 | `targeted_analytical_control` |" in text
    assert "`cantilever_beam_5d`" in text
    assert "tests/test_cantilever_beam_physics.py" in text
    assert "Note docx" in text
    assert "Reference files" in text
    assert "## Domain Evidence Coverage" in text
    assert "| aerodynamics | 6 | 0 | 6 | 6 |" in text
    assert "| geomechanics | 4 | 0 | 4 | 4 |" in text
    assert "| heat_transfer | 7 | 7 | 7 | 3 |" in text
    assert "## Prior Validation-Note Inventory" in text
    assert "1.1 cantilever_beam_5d.docx" in text
    cantilever_cross_validation = (
        ROOT / "tests" / "_cross_validation" / "cantilever_beam_5d.md"
    ).read_text(encoding="utf-8")
    assert "Roark2002" in cantilever_cross_validation
    assert "## Supporting Cross-Validation Inventory" in text
    assert "tests/_cross_validation/references.bib" in text
    assert "tests/_cross_validation/airfoil_stall_2d.py" in text
    assert "tests/_cross_validation/pressure_vessel_4d.md" in text
    assert "must not be described as externally validated solely from" in text


def test_low_order_surrogate_cross_validation_scope_is_not_external_accuracy_claim():
    forbidden_fragments = (
        "Target accuracy",
        "target accuracy",
        "≤ 5 %",
        "<= 5 %",
    )

    for case_id in sorted(LOW_ORDER_SURROGATE_SCOPE_CASES):
        text = _cross_validation_file(case_id).read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        assert "Validation scope" in text
        assert "do not establish a 5 %" in normalized_text
        for fragment in forbidden_fragments:
            assert fragment not in text


def test_variable_beam_cross_validation_scope_matches_cantilever_fem_model():
    for case_id in sorted(VARIABLE_BEAM_SCOPE_CASES):
        text = _cross_validation_file(case_id).read_text(encoding="utf-8")
        assert "clamped-free uniform-beam" in text
        assert "uniform cantilever beam" in text
        assert "convergent" in text
        assert "simply-supported" not in text
        assert "FEM exact" not in text
        assert "Target accuracy" not in text


def test_numerical_control_cross_validation_scope_is_not_global_external_accuracy_claim():
    for case_id in sorted(NUMERICAL_CONTROL_SCOPE_CASES):
        text = _cross_validation_file(case_id).read_text(encoding="utf-8")
        assert "Validation scope" in text
        assert "Target accuracy" not in text
        assert "≤ 5 %" not in text
        assert "<= 5 %" not in text
        assert "global 5 %" not in text
        assert "external validation" not in text

    tmd_md = ROOT / "tests" / "_cross_validation" / "tuned_mass_damper_3d.md"
    tmd_text = tmd_md.read_text(encoding="utf-8")
    assert "500-point" not in tmd_text
    assert "Target accuracy" not in tmd_text


def test_truss_fem_cross_validation_scope_is_idealized_model_control():
    for case_id in sorted(TRUSS_FEM_SCOPE_CASES):
        text = _cross_validation_file(case_id).read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        assert "Validation scope" in text
        assert "ideal pin-jointed" in text
        assert "linear-elastic axial-bar truss model" in text
        assert "constrained-DOF handling" in text
        assert "do not establish a calibrated 1 % external validation" in normalized_text
        assert "Target accuracy" not in text
        assert "≤ 1 %" not in text
        assert "<= 1 %" not in text
        assert "exact for truss elements" not in text


def test_analytical_control_cross_validation_scope_is_not_external_accuracy_claim():
    for case_id in sorted(ANALYTICAL_CONTROL_SCOPE_CASES):
        text = _cross_validation_file(case_id).read_text(encoding="utf-8")
        assert "Validation scope" in text
        assert "Target accuracy" not in text
        assert "target accuracy" not in text
        assert "≤ 1 %" not in text
        assert "<= 1 %" not in text

    for rel_path in [
        "tests/_cross_validation/cantilever_beam_5d.md",
        "tests/_cross_validation/plate_with_hole_5d.md",
        "tests/_cross_validation/pressure_vessel_4d.md",
    ]:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "Validation scope" in text
        assert "Target accuracy" not in text
        assert "target accuracy" not in text
        assert "≤ 1 %" not in text
        assert "<= 1 %" not in text


def test_plausibility_cross_validation_scope_is_not_accuracy_claim():
    for case_id in sorted(PLAUSIBILITY_SCOPE_CASES):
        text = _cross_validation_file(case_id).read_text(encoding="utf-8")
        assert "Validation scope" in text
        assert "plausibility" in text.lower()
        assert "Target accuracy" not in text
        assert "target accuracy" not in text


def test_generate_case_validation_summary_non_strict_allows_missing_note_sources(tmp_path):
    from scripts.generate_case_validation_summary import main

    out = tmp_path / "summary.md"

    assert main([
        "--note-sources-csv", str(tmp_path / "missing_note_sources.csv"),
        "--output", str(out),
    ]) == 0
    text = out.read_text(encoding="utf-8")

    assert "- Validation note/reference files fingerprinted: not checked" in text


def test_generate_case_validation_summary_strict_requires_note_sources(tmp_path):
    from scripts.generate_case_validation_summary import main

    out = tmp_path / "summary.md"
    out.write_text("old summary\n", encoding="utf-8")

    assert main([
        "--note-sources-csv", str(tmp_path / "missing_note_sources.csv"),
        "--output", str(out),
        "--strict",
    ]) == 1
    assert out.read_text(encoding="utf-8") == "old summary\n"


def test_generate_case_validation_summary_strict_failure_preserves_output(tmp_path):
    from scripts.generate_case_validation_summary import main

    status = tmp_path / "case_validation_status.csv"
    out = tmp_path / "summary.md"
    out.write_text("old summary\n", encoding="utf-8")
    with STATUS_CSV.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["validation_note_path"] = "案例正确性校验/missing_note_folder"
    with status.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assert main(["--status-csv", str(status), "--output", str(out), "--strict"]) == 1
    assert out.read_text(encoding="utf-8") == "old summary\n"
    assert not list(tmp_path.glob(".summary.md.tmp.*"))


def test_generate_case_validation_summary_strict_requires_reference_attachments(tmp_path):
    from scripts.generate_case_validation_summary import main

    note_dir = tmp_path / "tmp_no_reference_case"
    note_dir.mkdir(parents=True, exist_ok=True)
    note_docx = note_dir / "tmp_no_reference_case.docx"
    note_docx.write_bytes(b"docx placeholder")
    status = tmp_path / "case_validation_status.csv"
    out = tmp_path / "summary.md"
    out.write_text("old summary\n", encoding="utf-8")
    try:
        with STATUS_CSV.open("r", encoding="utf-8", newline="") as src:
            reader = csv.DictReader(src)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows[0]["validation_note_path"] = note_dir.as_posix()
        with status.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        assert main(["--status-csv", str(status), "--output", str(out), "--strict"]) == 1
        assert out.read_text(encoding="utf-8") == "old summary\n"
    finally:
        note_docx.unlink(missing_ok=True)
        note_dir.rmdir()


def test_generate_case_validation_summary_strict_rejects_metadata_drift(tmp_path):
    from scripts.generate_case_validation_summary import main

    status = tmp_path / "case_validation_status.csv"
    out = tmp_path / "summary.md"
    out.write_text("old summary\n", encoding="utf-8")
    with STATUS_CSV.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["domain"] = "wrong_domain"
    rows[1]["dim"] = "999"
    with status.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assert main(["--status-csv", str(status), "--output", str(out), "--strict"]) == 1
    assert out.read_text(encoding="utf-8") == "old summary\n"


@requires_validation_notes
def test_generate_case_validation_summary_strict_accepts_no_note_p2_analytic_control(tmp_path):
    from scripts.generate_case_validation_summary import main

    status = tmp_path / "case_validation_status.csv"
    out = tmp_path / "summary.md"
    with STATUS_CSV.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if row["case_id"] == "aircraft_conceptual_8d":
            row["final_disposition"] = "accepted analytic control"
    with status.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assert main(["--status-csv", str(status), "--output", str(out), "--strict"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "| `aircraft_conceptual_8d` | P2 | `targeted_analytical_control` |" in text
    assert "not external case-validation claims" in text


def test_generate_case_validation_summary_strict_rejects_no_note_p2_external_claim(tmp_path):
    from scripts.generate_case_validation_summary import main

    status = tmp_path / "case_validation_status.csv"
    out = tmp_path / "summary.md"
    out.write_text("old summary\n", encoding="utf-8")
    with STATUS_CSV.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if row["case_id"] == "aircraft_conceptual_8d":
            row["final_disposition"] = "externally validated"
    with status.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assert main(["--status-csv", str(status), "--output", str(out), "--strict"]) == 1
    assert out.read_text(encoding="utf-8") == "old summary\n"


def test_case_validation_status_matches_registered_metadata():
    rows = {row["case_id"]: row for row in _rows()}
    metas = {meta.case_id: meta for meta in db.list_cases()}

    for case_id, meta in metas.items():
        row = rows[case_id]
        assert row["domain"] == meta.domain
        assert int(row["dim"]) == meta.dim_input


def test_case_validation_status_paths_and_status_values_are_valid():
    rows = _rows()
    for row in rows:
        assert row["risk_level"] in VALID_RISKS
        assert row["formula_status"] in VALID_FORMULA_STATUS
        assert row["code_status"]
        assert row["next_action"]

        note_path = row["validation_note_path"]
        if note_path != "NONE" and HAS_VALIDATION_NOTES:
            note_dir = ROOT / note_path
            assert note_dir.exists(), (
                f"{row['case_id']} validation note path missing: {note_path}"
            )
            assert any(
                path.suffix.lower() == ".docx" and not path.name.startswith("~$")
                for path in note_dir.iterdir()
            ), f"{row['case_id']} validation note folder has no real docx"

        test_path = row["targeted_physics_test"]
        if test_path != "NONE":
            assert (ROOT / test_path).exists(), (
                f"{row['case_id']} targeted physics test missing: {test_path}"
            )

        cross_validation = ROOT / "tests" / "_cross_validation" / f"{row['case_id']}.py"
        assert cross_validation.exists(), (
            f"{row['case_id']} missing cross-validation module"
        )


def test_prior_validation_note_folders_are_all_accounted_for():
    rows = _rows()
    documented_dirs = {
        (ROOT / row["validation_note_path"]).resolve()
        for row in rows
        if row["validation_note_path"] != "NONE"
    }
    note_dirs = {path.resolve() for path in _validation_note_dirs()}

    missing = sorted(note_dirs - documented_dirs)
    assert missing == []


def test_fixed_case_rows_have_targeted_physics_tests():
    rows = _rows()
    fixed = [
        row for row in rows
        if row["formula_status"] in {"fixed_targeted_tested", "targeted_tested_no_code_change"}
    ]

    assert fixed, "expected at least one fixed case row"
    for row in fixed:
        assert row["targeted_physics_test"] != "NONE", row["case_id"]


def test_case_validation_evidence_tiers_match_declared_support():
    rows = _rows()
    for row in rows:
        has_note = row["validation_note_path"] != "NONE"
        disposition = row["final_disposition"].lower()
        if row["risk_level"] == "P1":
            assert has_note, row["case_id"]
            assert "accepted" in disposition, row["case_id"]
        if row["risk_level"] == "P2":
            assert (
                "plausibility only" in disposition
                or "analytic" in disposition
                or "control" in disposition
            ), row["case_id"]
            if not has_note:
                assert (
                    "plausibility only" in disposition
                    or "analytic" in disposition
                    or "control" in disposition
                ), row["case_id"]
                assert "external" not in disposition or "not external" in disposition, row["case_id"]


def test_p2_cases_have_explicit_web_audit_evidence_tier():
    rows = _rows()
    audit_text = (ROOT / "docs" / "CASE_WEB_VALIDATION_AUDIT.md").read_text(
        encoding="utf-8"
    )
    audit_rows = [line for line in audit_text.splitlines() if line.startswith("| `")]

    missing = []
    for row in rows:
        if row["risk_level"] != "P2":
            continue
        case_id = row["case_id"]
        matches = [line for line in audit_rows if f"`{case_id}`" in line]
        if not matches:
            missing.append(f"{case_id}: no web-audit row")
            continue
        if not any(
            "Evidence tier remains P2" in line
            or "Evidence tier remains P2 analytic-control" in line
            for line in matches
        ):
            missing.append(f"{case_id}: missing explicit P2 evidence-tier phrase")

    assert missing == []


def test_p2_plausibility_only_cases_have_nontrivial_physics_tests():
    rows = _rows()
    p2_plausibility_only = [
        row
        for row in rows
        if row["risk_level"] == "P2"
        and row["validation_note_path"] == "NONE"
        and "plausibility only" in row["final_disposition"].lower()
    ]

    assert p2_plausibility_only, "expected no-note P2 plausibility-only rows"
    for row in p2_plausibility_only:
        path = ROOT / row["targeted_physics_test"]
        names = _test_function_names(path)
        physics_names = [
            name
            for name in names
            if not any(fragment in name for fragment in GENERIC_TEST_NAME_FRAGMENTS)
        ]
        assert len(names) >= MIN_P2_PLAUSIBILITY_TEST_FUNCTIONS, row["case_id"]
        assert len(physics_names) >= MIN_P2_PLAUSIBILITY_PHYSICS_TEST_FUNCTIONS, row["case_id"]


def test_no_note_p2_cases_have_reviewer_facing_markdown_evidence_cards():
    rows = _rows()
    p2_no_note = [
        row
        for row in rows
        if row["risk_level"] == "P2" and row["validation_note_path"] == "NONE"
    ]

    assert p2_no_note, "expected no-note P2 rows"
    for row in p2_no_note:
        case_id = row["case_id"]
        md_path = ROOT / "tests" / "_cross_validation" / f"{case_id}.md"
        py_path = ROOT / "tests" / "_cross_validation" / f"{case_id}.py"
        assert md_path.exists(), f"{case_id} missing reviewer-facing evidence card"
        assert py_path.exists(), f"{case_id} missing executable cross-validation module"

        text = md_path.read_text(encoding="utf-8")
        normalized = " ".join(text.split()).lower()
        assert "evidence tier" in normalized, case_id
        assert "p2" in normalized, case_id
        assert (
            "plausibility-only" in normalized
            or "plausibility only" in normalized
            or "analytical-control" in normalized
            or "analytic-control" in normalized
        ), case_id
        assert "http://" in text or "https://" in text, case_id
        has_negative_validation_boundary = any(
            prefix in normalized
            and any(word in normalized for word in ("validation", "validated", "validate", "calibration", "calibrated"))
            for prefix in ("not ", "does not ", "do not ")
        )
        assert has_negative_validation_boundary, case_id


def test_case_physics_runner_covers_ledger_tests_and_expected_ranges():
    from scripts.run_case_physics_tests import (
        build_pytest_command,
        discover_case_physics_tests,
        validate_ledger_test_coverage,
    )

    discovered = {
        path.relative_to(ROOT).as_posix()
        for path in discover_case_physics_tests(ROOT)
    }
    ledger_tests = {
        row["targeted_physics_test"]
        for row in _rows()
        if row["targeted_physics_test"] != "NONE"
    }

    assert ledger_tests <= discovered
    assert "tests/test_expected_range_envelopes.py" in discovered
    assert validate_ledger_test_coverage(discover_case_physics_tests(ROOT), ROOT) == []
    command = build_pytest_command(["-q"], ROOT)
    assert command[1:3] == ["-m", "pytest"]
    assert command[-1] == "-q"
