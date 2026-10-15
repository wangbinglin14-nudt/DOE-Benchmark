#!/usr/bin/env python
"""Run the no-sweep local gates before handing work to the Linux server.

This is the review-workstation companion to ``check_server_launch_ready.py``.
It intentionally runs only documentation, ledger, unit-test, physics-test, and
draft-paper checks.  It never calls the full benchmark runner and never passes
large-run confirmation flags.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.server_operator_contracts import server_operator_contract_nodeids  # noqa: E402


@dataclass(frozen=True)
class LocalCheck:
    label: str
    command: list[str]


def _script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def command_to_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def build_local_checks(
    *,
    python: str = sys.executable,
    include_doc_links: bool = True,
    include_case_physics: bool = True,
    include_paper_draft: bool = True,
) -> list[LocalCheck]:
    checks: list[LocalCheck] = []
    if include_doc_links:
        checks.append(
            LocalCheck(
                "doc-links",
                [
                    python,
                    _script("check_doc_links.py"),
                    "--root",
                    "docs",
                    "paper",
                    "scripts",
                    "--quiet",
                ],
            )
        )
    checks.extend(
        [
            LocalCheck(
                "case-source-fingerprints",
                [python, _script("update_case_validation_sources.py")],
            ),
            LocalCheck(
                "case-note-fingerprints",
                [python, _script("update_case_validation_note_sources.py")],
            ),
            LocalCheck(
                "case-validation-and-prereg",
                [
                    python,
                    "-m",
                    "pytest",
                    "tests/test_case_validation_status.py",
                    "benchmark/tests/test_prereg_compliance.py",
                    "-q",
                ],
            ),
            LocalCheck(
                "pipeline-orchestrator",
                [
                    python,
                    "-m",
                    "pytest",
                    "benchmark/tests/test_pipeline_orchestrator.py",
                    "-q",
                ],
            ),
            LocalCheck(
                "sweep-integrity-audit-contract",
                [
                    python,
                    "-m",
                    "pytest",
                    "benchmark/tests/test_sweep_integrity_audit.py",
                    "-q",
                ],
            ),
            LocalCheck(
                "runtime-row-contracts",
                [
                    python,
                    "-m",
                    "pytest",
                    "benchmark/tests/test_runtime_contracts.py",
                    "-q",
                ],
            ),
            LocalCheck(
                "server-operator-contracts",
                [
                    python,
                    "-m",
                    "pytest",
                    *server_operator_contract_nodeids(),
                    "-q",
                ],
            ),
            LocalCheck(
                "d1-pairing-modes",
                [
                    python,
                    "-m",
                    "pytest",
                    "benchmark/tests/test_new_methods.py::TestD1PairingModes",
                    "-q",
                ],
            ),
        ]
    )
    if include_case_physics:
        checks.append(
            LocalCheck(
                "case-physics-tests",
                [python, _script("run_case_physics_tests.py"), "-q"],
            )
        )
    if include_paper_draft:
        checks.append(
            LocalCheck(
                "paper-draft-ready",
                [python, _script("check_paper_draft_ready.py"), "paper"],
            )
        )
    return checks


def validate_no_sweep_commands(checks: Sequence[LocalCheck]) -> list[str]:
    """Return safety errors if a local pre-server check can launch Stage 2."""
    errors: list[str] = []
    for check in checks:
        command = [str(part) for part in check.command]
        normalized = [part.replace("\\", "/") for part in command]
        text = command_to_text(command)
        if any(part.endswith("benchmark/scripts/run_full_benchmark.py") for part in normalized):
            errors.append(
                f"{check.label}: local pre-server checks must not call "
                f"run_full_benchmark.py ({text})"
            )
        if "--confirm-large-run" in command or "--confirm-full-run" in command:
            errors.append(
                f"{check.label}: local pre-server checks must not pass large/full "
                f"run confirmation flags ({text})"
            )
    return errors


def run_checks(checks: Sequence[LocalCheck], *, cwd: Path = ROOT) -> int:
    safety_errors = validate_no_sweep_commands(checks)
    if safety_errors:
        for error in safety_errors:
            print(f"[safety:error] {error}", file=sys.stderr)
        print("[result] FAIL - local pre-server command graph is unsafe.", file=sys.stderr)
        return 2

    for check in checks:
        print(f"[{check.label}] {command_to_text(check.command)}", flush=True)
        completed = subprocess.run(check.command, cwd=cwd)
        if completed.returncode != 0:
            print(
                f"[{check.label}] FAIL exit={completed.returncode}",
                file=sys.stderr,
                flush=True,
            )
            print("[result] FAIL - local pre-server readiness checks failed.", file=sys.stderr)
            return int(completed.returncode)
        print(f"[{check.label}] PASS", flush=True)

    print("[result] PASS - local workstation is ready for server handoff.")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for subprocess checks.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the local no-sweep command list without executing it.",
    )
    parser.add_argument(
        "--skip-doc-links",
        action="store_true",
        help="Skip the documentation link checker.",
    )
    parser.add_argument(
        "--skip-case-physics",
        action="store_true",
        help="Skip targeted case-physics tests.",
    )
    parser.add_argument(
        "--skip-paper-draft",
        action="store_true",
        help="Skip the pre-results paper draft gate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checks = build_local_checks(
        python=str(args.python),
        include_doc_links=not bool(args.skip_doc_links),
        include_case_physics=not bool(args.skip_case_physics),
        include_paper_draft=not bool(args.skip_paper_draft),
    )
    safety_errors = validate_no_sweep_commands(checks)
    if safety_errors:
        for error in safety_errors:
            print(f"[safety:error] {error}", file=sys.stderr)
        return 2
    if args.list:
        for check in checks:
            print(f"{check.label}\t{command_to_text(check.command)}")
        return 0
    return run_checks(checks)


if __name__ == "__main__":
    raise SystemExit(main())
