"""Run the targeted per-case physics tests and check ledger coverage.

Discovers ``tests/test_*_physics.py`` (plus the expected-range envelope
tests), verifies that every case marked as physics-tested in
``docs/case_validation_status.csv`` has a matching test file, and runs the
collected files through pytest.  Exits non-zero on coverage gaps or test
failures.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "tests"
CASE_VALIDATION_STATUS = ROOT / "docs" / "case_validation_status.csv"
DEFAULT_PYTEST_ARGS = ("-q",)


def discover_case_physics_tests(root: Path = ROOT) -> list[Path]:
    """Return the targeted case-physics test files in deterministic order."""
    tests_root = root / "tests"
    test_files = sorted(tests_root.glob("test_*_physics.py"))
    expected_ranges = tests_root / "test_expected_range_envelopes.py"
    if expected_ranges.exists():
        test_files.append(expected_ranges)
    return test_files


def validate_ledger_test_coverage(
    test_files: Sequence[Path],
    root: Path = ROOT,
    ledger: Path | None = None,
) -> list[str]:
    """Return coverage errors between the validation ledger and pytest files."""
    ledger_path = ledger or root / "docs" / "case_validation_status.csv"
    errors: list[str] = []
    if not ledger_path.exists():
        return [f"case validation ledger missing: {ledger_path}"]

    discovered = {path.relative_to(root).as_posix() for path in test_files}
    expected_range_test = "tests/test_expected_range_envelopes.py"
    if expected_range_test not in discovered:
        errors.append(f"case physics runner missing envelope test: {expected_range_test}")

    with ledger_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ledger_tests = sorted(
        {
            (row.get("targeted_physics_test") or "").strip()
            for row in rows
            if (row.get("targeted_physics_test") or "").strip()
            and (row.get("targeted_physics_test") or "").strip() != "NONE"
        }
    )
    missing = sorted(set(ledger_tests) - discovered)
    for rel in missing:
        errors.append(
            "case physics runner does not discover targeted ledger test: "
            f"{rel}"
        )
    return errors


def build_pytest_command(
    pytest_args: Sequence[str] | None = None,
    root: Path = ROOT,
) -> list[str]:
    test_files = discover_case_physics_tests(root)
    args = list(pytest_args) if pytest_args else list(DEFAULT_PYTEST_ARGS)
    rel_tests = [path.relative_to(root).as_posix() for path in test_files]
    return [sys.executable, "-m", "pytest", *rel_tests, *args]


def main(argv: Sequence[str] | None = None) -> int:
    root = ROOT
    test_files = discover_case_physics_tests(root)
    if not test_files:
        print("No targeted case-physics tests were found.", file=sys.stderr)
        return 2
    coverage_errors = validate_ledger_test_coverage(
        test_files,
        root,
        CASE_VALIDATION_STATUS,
    )
    if coverage_errors:
        for err in coverage_errors:
            print(err, file=sys.stderr)
        return 2

    pytest_args = list(argv if argv is not None else sys.argv[1:])
    command = build_pytest_command(pytest_args, root)
    print(
        "Running targeted case-physics tests: "
        f"{len(test_files)} files under {TEST_ROOT.relative_to(root).as_posix()}"
    )
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
