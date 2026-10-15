"""Publication-readiness roll-up audit for DOE-Benchmark.

This script does not replace the strict stage gates.  It gives maintainers a
single handoff report that answers: which parts of the case, experiment, and
paper pipeline are proven today, and which concrete artifacts still block a
Nature-family submission?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.project_layout import MANUSCRIPTS_DIR  # noqa: E402

PAPER = MANUSCRIPTS_DIR
FULL_SWEEP = ROOT / "benchmark" / "results" / "full_benchmark"
RESULTS_ROOT = ROOT / "benchmark" / "results"
DEFAULT_OUTPUT = ROOT / "docs" / "PUBLICATION_READINESS_AUDIT.md"
DEFAULT_LAUNCH_OUTPUT = ROOT / "docs" / "PUBLICATION_LAUNCH_PREFLIGHT.md"
CASE_READINESS = ROOT / "docs" / "case_experiment_readiness.csv"
CASE_VALIDATION_STATUS = ROOT / "docs" / "case_validation_status.csv"
CASE_VALIDATION_SOURCES = ROOT / "docs" / "case_validation_sources.csv"
CASE_VALIDATION_NOTE_SOURCES = ROOT / "docs" / "case_validation_note_sources.csv"
CASE_VALIDATION_SUMMARY = ROOT / "docs" / "CASE_VALIDATION_SUMMARY.md"
CASE_WEB_VALIDATION_AUDIT = ROOT / "docs" / "CASE_WEB_VALIDATION_AUDIT.md"
EXPERIMENT_BENCHMARK_AUDIT = ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md"
SWEEP_PASS_VERDICT = "PASS: sweep satisfies the declared integrity gate."
EXPECTED_POOL = "nature"
EXPECTED_BUDGETS = [5, 10, 20, 50]
EXPECTED_N_REPEATS = 30
EXPECTED_N_TEST = 1000
EXPECTED_SEED_STRATEGY = "case_repeat"
EXPECTED_SEED_START = 1000
EXPECTED_N_CASES = 34
EXPECTED_N_METHODS = 20
CASE_READINESS_REQUIRED_COLUMNS = {
    "case_id",
    "domain",
    "pde_type",
    "response_topology",
    "n_topology_tags",
    "governing_equation",
    "n_physics_references",
    "status",
    "n_nan",
    "n_inf",
    "input_mutation_max_abs_delta",
    "input_mutation_n_changed",
    "determinism_max_abs_delta",
    "determinism_n_mismatch",
    "n_expected_range_checked",
    "n_expected_range_violations",
    "expected_range_max_abs_violation",
}


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    evidence: str
    next_action: str


def _run_short_gate(cmd: list[str], *, timeout_s: int = 60) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        return 124, f"{out}\nTIMEOUT after {timeout_s}s".strip()
    return proc.returncode, (proc.stdout or "").strip()


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _count_csv_rows_checked(path: Path) -> tuple[int | None, str | None]:
    try:
        return _count_csv_rows(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _nonportable_command_file_requests(items: list[object]) -> list[str]:
    """Return command-history paths that cannot be restored repo-relatively."""
    bad: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        normalized = text.replace("\\", "/")
        posix = PurePosixPath(normalized)
        windows = PureWindowsPath(text)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
            bad.append(item)
    return bad


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _has_flag_value(tokens: list[str], flag: str, value: str) -> bool:
    return any(
        token == f"{flag}={value}"
        or (token == flag and idx + 1 < len(tokens) and tokens[idx + 1] == value)
        for idx, token in enumerate(tokens)
    )


def _formal_pipeline_invocation_error(command: str) -> str | None:
    tokens = _split_command(command)
    normalized = command.replace("\\", "/")
    uses_python_orchestrator = "scripts/run_pipeline.py" in normalized
    uses_shell_wrapper = "scripts/run_pipeline.sh" in normalized
    if not uses_python_orchestrator and not uses_shell_wrapper:
        return (
            "pipeline_invocation must use the formal full-pipeline entrypoint "
            "`scripts/run_pipeline.py` or `scripts/run_pipeline.sh`"
        )
    if "--dry-run" in tokens:
        return "pipeline_invocation must not be a dry-run command"
    if uses_python_orchestrator:
        if not _has_flag_value(tokens, "--profile", "full"):
            return "pipeline_invocation must use `--profile full`"
        if "--confirm-full-run" not in tokens:
            return "pipeline_invocation must include `--confirm-full-run`"
    elif _has_flag_value(tokens, "--profile", "smoke"):
        return "pipeline_invocation must not use the smoke profile"
    return None


def _registered_case_ids() -> tuple[list[str] | None, str | None]:
    try:
        import doe_testbed as db
    except Exception as exc:
        return None, f"could not import the current case registry: {type(exc).__name__}: {exc}"
    try:
        return sorted(str(case.case_id) for case in db.list_cases()), None
    except Exception as exc:
        return None, f"could not enumerate the current case registry: {type(exc).__name__}: {exc}"


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())


def _filesystem_error_gate(name: str, exc: OSError) -> GateResult:
    target = getattr(exc, "filename", None) or getattr(exc, "filename2", None) or "<unknown>"
    return GateResult(
        name,
        "BLOCKED",
        (
            f"Filesystem access failed while checking `{target}`: "
            f"{type(exc).__name__}: {exc}"
        ),
        (
            "Fix the file/directory ACL or rebuild the generated artifact package, "
            "then rerun the readiness audit."
        ),
    )


def _safe_gate(name: str, fn) -> GateResult:
    try:
        return fn()
    except PermissionError as exc:
        return _filesystem_error_gate(name, exc)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 5:
            return _filesystem_error_gate(name, exc)
        raise


def _registered_case_metadata() -> tuple[dict[str, dict[str, str]] | None, str | None]:
    try:
        import doe_testbed as db
    except Exception as exc:
        return None, f"could not import the current case registry: {type(exc).__name__}: {exc}"
    try:
        out: dict[str, dict[str, str]] = {}
        for meta in db.list_cases():
            topology = list(getattr(meta, "response_topology", []) or [])
            physics_references = list(getattr(meta, "physics_references", []) or [])
            out[str(meta.case_id)] = {
                "domain": str(getattr(meta, "domain", "")),
                "pde_type": str(getattr(meta, "pde_type", "")),
                "response_topology": ";".join(str(item) for item in topology),
                "n_topology_tags": str(len(topology)),
                "governing_equation": _one_line(getattr(meta, "governing_equation", "")),
                "n_physics_references": str(len(physics_references)),
            }
        return out, None
    except Exception as exc:
        return None, f"could not enumerate current case metadata: {type(exc).__name__}: {exc}"


def _registered_method_names(pool: str = EXPECTED_POOL) -> tuple[list[str] | None, str | None]:
    try:
        from benchmark.scripts.run_full_benchmark import POOLS
    except Exception as exc:
        return None, f"could not import the current method registry: {type(exc).__name__}: {exc}"
    try:
        factory = POOLS[str(pool)]
    except KeyError:
        return None, f"method pool {pool!r} is not registered"
    try:
        return sorted(str(method.name) for method in factory()), None
    except Exception as exc:
        return None, f"could not enumerate method pool {pool!r}: {type(exc).__name__}: {exc}"


def _registered_method_family_pairs(pool: str = EXPECTED_POOL) -> tuple[list[str] | None, str | None]:
    try:
        from benchmark.scripts.run_full_benchmark import POOLS
    except Exception as exc:
        return None, f"could not import the current method registry: {type(exc).__name__}: {exc}"
    try:
        factory = POOLS[str(pool)]
    except KeyError:
        return None, f"method pool {pool!r} is not registered"
    try:
        return sorted(f"{method.name}:{method.family}" for method in factory()), None
    except Exception as exc:
        return None, f"could not enumerate method pool {pool!r}: {type(exc).__name__}: {exc}"


def _rerun_sweep_integrity(results_csv: Path) -> tuple[int, str]:
    """Re-run the sweep-integrity gate against the current restored CSV.

    The archived ``sweep_integrity_report.md`` proves what the server claimed
    at archive time.  This subprocess proves the local file we are about to
    trust still satisfies the same protocol after restore/copy/edit steps.
    """
    expected_cases, case_error = _registered_case_ids()
    if case_error:
        return 1, case_error
    expected_methods, method_error = _registered_method_names(EXPECTED_POOL)
    if method_error:
        return 1, method_error
    expected_method_families, family_error = _registered_method_family_pairs(EXPECTED_POOL)
    if family_error:
        return 1, family_error
    cmd = [
        sys.executable,
        "scripts/audit_sweep_integrity.py",
        str(results_csv),
        "--expected-cases", ",".join(expected_cases or []),
        "--expected-methods", ",".join(expected_methods or []),
        "--expected-method-families", ",".join(expected_method_families or []),
        "--expected-budgets", *(str(item) for item in EXPECTED_BUDGETS),
        "--expected-repeats", str(EXPECTED_N_REPEATS),
        "--expected-n-test", str(EXPECTED_N_TEST),
        "--expected-test-seed", "0",
        "--seed-strategy", EXPECTED_SEED_STRATEGY,
        "--seed-start", str(EXPECTED_SEED_START),
        "--min-repeats-per-cell", "28",
        "--min-ok-fraction", "0.95",
        "--max-timeout-fraction", "0.05",
    ]
    return _run_short_gate(cmd, timeout_s=300)


def _rerun_smoke_integrity(results_csv: Path) -> tuple[int, str]:
    """Re-run sweep-integrity using the current smoke-profile protocol."""
    try:
        from scripts import run_pipeline
    except Exception as exc:
        return 1, f"could not import smoke-profile constants: {type(exc).__name__}: {exc}"
    expected_methods, method_error = _registered_method_names(run_pipeline.SMOKE_POOL)
    if method_error:
        return 1, method_error
    expected_method_families, family_error = _registered_method_family_pairs(run_pipeline.SMOKE_POOL)
    if family_error:
        return 1, family_error

    cmd = [
        sys.executable,
        "scripts/audit_sweep_integrity.py",
        str(results_csv),
        "--expected-cases", ",".join(run_pipeline.SMOKE_CASES),
        "--expected-methods", ",".join(expected_methods or []),
        "--expected-method-families", ",".join(expected_method_families or []),
        "--expected-budgets", *(str(item) for item in run_pipeline.SMOKE_BUDGETS),
        "--expected-repeats", str(run_pipeline.SMOKE_N_REPEATS),
        "--expected-n-test", str(run_pipeline.SMOKE_N_TEST),
        "--expected-test-seed", "0",
        "--seed-strategy", EXPECTED_SEED_STRATEGY,
        "--seed-start", str(EXPECTED_SEED_START),
        "--min-repeats-per-cell", "1",
        "--min-ok-fraction", "1.0",
        "--max-timeout-fraction", "0.0",
    ]
    return _run_short_gate(cmd, timeout_s=120)


def _set_drift_errors(
    *,
    label: str,
    declared: object,
    observed: object,
    count_label: str,
    declared_count: object,
    observed_count: object,
) -> list[str]:
    errors: list[str] = []
    if isinstance(declared, list) and isinstance(observed, list):
        declared_set = set(map(str, declared))
        observed_values = [str(item) for item in observed]
        observed_set = set(observed_values)
        missing = sorted(declared_set - observed_set)
        extra = sorted(observed_set - declared_set)
        duplicates = sorted({item for item in observed_values if observed_values.count(item) > 1})
        if missing or extra:
            errors.append(
                f"`RUN_MANIFEST.json` results_summary {label} set drifts from experiment declaration: "
                f"missing={missing}, extra={extra}."
            )
        if duplicates:
            errors.append(
                f"`RUN_MANIFEST.json` results_summary {label} observed set contains duplicates: {duplicates}."
            )
        if isinstance(observed_count, int):
            if observed_count != len(observed_set):
                errors.append(
                    f"`RUN_MANIFEST.json` results_summary {label} observed count does not match the unique observed set size."
                )
            if isinstance(declared_count, int) and observed_count != declared_count:
                errors.append(
                    f"`RUN_MANIFEST.json` results_summary {count_label} count drifts from experiment declaration: "
                    f"declared {declared_count}, observed {observed_count}."
                )
    elif isinstance(declared, list):
        errors.append(f"`RUN_MANIFEST.json` results_summary {label} observed set is missing or malformed.")
    elif isinstance(declared_count, int) and observed_count != declared_count:
        errors.append(
            f"`RUN_MANIFEST.json` results_summary {count_label} count drifts from experiment declaration: "
            f"declared {declared_count}, observed {observed_count}."
        )
    return errors


def _nonnegative_int_from_summary(
    summary: dict[str, object],
    key: str,
    errors: list[str],
) -> int:
    raw = summary.get(key, 0)
    if raw in (None, ""):
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"`RUN_MANIFEST.json` results_summary.{key} must be a non-negative integer.")
        return 0
    if value < 0:
        errors.append(f"`RUN_MANIFEST.json` results_summary.{key} must be a non-negative integer.")
        return 0
    return value


def _nonnegative_int_value(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _manifest_structure_errors(
    path: Path,
    *,
    row_count: int | None,
    results_csv: Path,
    sweep_report: Path,
) -> list[str]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse `RUN_MANIFEST.json`: {type(exc).__name__}: {exc}"]
    if not isinstance(manifest, dict):
        return ["`RUN_MANIFEST.json` must contain a JSON object."]

    errors: list[str] = []
    manifest_command = manifest.get("manifest_command")
    if not isinstance(manifest_command, str) or not manifest_command.strip():
        errors.append("`RUN_MANIFEST.json` manifest_command is missing.")
    git = manifest.get("git")
    if not isinstance(git, dict):
        errors.append("`RUN_MANIFEST.json` git section is missing or malformed.")
    else:
        commit = git.get("commit")
        if not isinstance(commit, str) or not commit.strip():
            errors.append("`RUN_MANIFEST.json` git.commit is missing.")
        if git.get("source_dirty"):
            entries = git.get("source_dirty_entries")
            detail = f" ({len(entries)} entries)" if isinstance(entries, list) else ""
            errors.append(f"`RUN_MANIFEST.json` was generated from dirty source/config state{detail}.")
        elif git.get("dirty") and "source_dirty" not in git:
            errors.append("`RUN_MANIFEST.json` was generated from a dirty git worktree.")
    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        errors.append("`RUN_MANIFEST.json` experiment section is missing or malformed.")
    else:
        case_resolution_error = experiment.get("case_resolution_error")
        if "case_resolution_error" not in experiment:
            errors.append("`RUN_MANIFEST.json` experiment.case_resolution_error is missing or malformed.")
        elif case_resolution_error not in (None, ""):
            errors.append("`RUN_MANIFEST.json` experiment.case_resolution_error is non-empty.")
        commands = experiment.get("commands")
        if not isinstance(commands, list) or not any(
            isinstance(item, str) and item.strip() for item in commands
        ):
            errors.append("`RUN_MANIFEST.json` experiment.commands is empty.")
        elif any(
            isinstance(item, str) and "--dry-run" in item.split()
            for item in commands
        ):
            errors.append("`RUN_MANIFEST.json` experiment.commands contains `--dry-run` entries.")
        requested_command_files = experiment.get("command_files_requested")
        if not isinstance(requested_command_files, list):
            errors.append("`RUN_MANIFEST.json` experiment.command_files_requested is missing or malformed.")
        elif not requested_command_files:
            errors.append("`RUN_MANIFEST.json` experiment.command_files_requested is empty.")
        else:
            if any(not isinstance(item, str) or not item.strip() for item in requested_command_files):
                errors.append("`RUN_MANIFEST.json` experiment.command_files_requested contains blank or non-string entries.")
            nonportable_requested = _nonportable_command_file_requests(requested_command_files)
            if nonportable_requested:
                errors.append(
                    "`RUN_MANIFEST.json` experiment.command_files_requested contains non-portable paths; "
                    f"use repo-relative paths without parent traversal: {nonportable_requested}."
                )
            if any(isinstance(item, str) and ".dry_run" in item for item in requested_command_files):
                errors.append("`RUN_MANIFEST.json` experiment.command_files_requested contains dry-run command history files.")
        missing_command_files = experiment.get("command_files_missing")
        if not isinstance(missing_command_files, list):
            errors.append("`RUN_MANIFEST.json` experiment.command_files_missing is missing or malformed.")
        elif missing_command_files:
            errors.append("`RUN_MANIFEST.json` experiment.command_files_missing is non-empty.")
        command_file_parse_errors = experiment.get("command_file_parse_errors")
        if not isinstance(command_file_parse_errors, list):
            errors.append("`RUN_MANIFEST.json` experiment.command_file_parse_errors is missing or malformed.")
        elif command_file_parse_errors:
            errors.append("`RUN_MANIFEST.json` experiment.command_file_parse_errors is non-empty.")
        command_entries = experiment.get("command_entries")
        if not isinstance(command_entries, list) or not command_entries:
            errors.append("`RUN_MANIFEST.json` experiment.command_entries is missing or malformed.")
        else:
            bad_entries = [
                item for item in command_entries
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("label"), str)
                    or not item.get("label", "").strip()
                    or not isinstance(item.get("command"), str)
                    or not item.get("command", "").strip()
                )
            ]
            if bad_entries:
                errors.append("`RUN_MANIFEST.json` experiment.command_entries contains malformed entries.")
            entry_commands = [
                item.get("command")
                for item in command_entries
                if isinstance(item, dict) and isinstance(item.get("command"), str)
            ]
            if isinstance(commands, list) and entry_commands != commands:
                errors.append("`RUN_MANIFEST.json` experiment.command_entries commands do not match experiment.commands.")
            pipeline_invocations = [
                item for item in command_entries
                if (
                    isinstance(item, dict)
                    and item.get("label") == "pipeline_invocation"
                    and isinstance(item.get("command"), str)
                    and item.get("command", "").strip()
                )
            ]
            if not pipeline_invocations:
                errors.append("`RUN_MANIFEST.json` experiment.command_entries does not include pipeline_invocation.")
            elif len(pipeline_invocations) > 1:
                errors.append("`RUN_MANIFEST.json` experiment.command_entries contains multiple pipeline_invocation entries.")
            else:
                invocation_error = _formal_pipeline_invocation_error(
                    str(pipeline_invocations[0].get("command", ""))
                )
                if invocation_error:
                    errors.append(
                        f"`RUN_MANIFEST.json` experiment.command_entries {invocation_error}."
                    )
        prereg_checks = {
            "pool": EXPECTED_POOL,
            "budgets": EXPECTED_BUDGETS,
            "n_repeats": EXPECTED_N_REPEATS,
            "n_test": EXPECTED_N_TEST,
            "seed_strategy": EXPECTED_SEED_STRATEGY,
            "seed_start": EXPECTED_SEED_START,
            "n_cases_declared": EXPECTED_N_CASES,
            "n_methods_declared": EXPECTED_N_METHODS,
        }
        for key, expected in prereg_checks.items():
            if experiment.get(key) != expected:
                errors.append(
                    f"`RUN_MANIFEST.json` experiment.{key} deviates from the preregistered full-sweep protocol."
                )
        declared_cases = experiment.get("case_ids_declared")
        if not isinstance(declared_cases, list):
            errors.append("`RUN_MANIFEST.json` experiment.case_ids_declared is missing or malformed.")
        elif len(declared_cases) != EXPECTED_N_CASES:
            errors.append(
                "`RUN_MANIFEST.json` experiment.case_ids_declared count deviates from the preregistered full-sweep protocol."
            )
        else:
            expected_cases, registry_error = _registered_case_ids()
            if registry_error:
                errors.append(f"`RUN_MANIFEST.json` case registry check failed: {registry_error}.")
            elif expected_cases is not None:
                observed = [str(item) for item in declared_cases]
                observed_set = set(observed)
                expected_set = set(expected_cases)
                duplicates = sorted({case_id for case_id in observed if observed.count(case_id) > 1})
                missing = sorted(expected_set - observed_set)
                extra = sorted(observed_set - expected_set)
                if missing or extra or duplicates:
                    detail = []
                    if missing:
                        detail.append(f"missing={missing}")
                    if extra:
                        detail.append(f"extra={extra}")
                    if duplicates:
                        detail.append(f"duplicates={duplicates}")
                    errors.append(
                        "`RUN_MANIFEST.json` experiment.case_ids_declared does not match the current registered case library "
                        + "; ".join(detail)
                        + "."
                    )
        declared_methods = experiment.get("methods_declared")
        if not isinstance(declared_methods, list):
            errors.append("`RUN_MANIFEST.json` experiment.methods_declared is missing or malformed.")
        else:
            expected_methods, registry_error = _registered_method_names(str(experiment.get("pool") or EXPECTED_POOL))
            if registry_error:
                errors.append(f"`RUN_MANIFEST.json` method registry check failed: {registry_error}.")
            elif expected_methods is not None:
                observed = [str(item) for item in declared_methods]
                observed_set = set(observed)
                expected_set = set(expected_methods)
                duplicates = sorted({method for method in observed if observed.count(method) > 1})
                missing = sorted(expected_set - observed_set)
                extra = sorted(observed_set - expected_set)
                if missing or extra or duplicates:
                    detail = []
                    if missing:
                        detail.append(f"missing={missing}")
                    if extra:
                        detail.append(f"extra={extra}")
                    if duplicates:
                        detail.append(f"duplicates={duplicates}")
                    errors.append(
                        "`RUN_MANIFEST.json` experiment.methods_declared does not match the current registered method pool "
                        + "; ".join(detail)
                        + "."
                    )
    summary = manifest.get("results_summary")
    if not isinstance(summary, dict):
        errors.append("`RUN_MANIFEST.json` results_summary section is missing or malformed.")
    else:
        if summary.get("exists") is not True:
            errors.append("`RUN_MANIFEST.json` results_summary.exists is not true.")
        if summary.get("n_rows") != row_count:
            errors.append(
                "`RUN_MANIFEST.json` results_summary.n_rows does not match the current results.csv row count."
            )
        if isinstance(experiment, dict):
            budgets = experiment.get("budgets")
            n_cases = experiment.get("n_cases_declared")
            n_methods = experiment.get("n_methods_declared")
            n_repeats = experiment.get("n_repeats")
            if (
                isinstance(n_cases, int)
                and isinstance(n_methods, int)
                and isinstance(n_repeats, int)
                and isinstance(budgets, list)
            ):
                expected_total = n_cases * n_methods * len(budgets) * n_repeats
                if summary.get("n_rows") != expected_total:
                    errors.append(
                        "`RUN_MANIFEST.json` results_summary.n_rows does not match the declared experiment grid size."
                    )
            errors.extend(_set_drift_errors(
                label="case",
                declared=experiment.get("case_ids_declared"),
                observed=summary.get("case_ids_observed"),
                count_label="case",
                declared_count=n_cases,
                observed_count=summary.get("n_cases_observed"),
            ))
            errors.extend(_set_drift_errors(
                label="method",
                declared=experiment.get("methods_declared"),
                observed=summary.get("methods_observed"),
                count_label="method",
                declared_count=n_methods,
                observed_count=summary.get("n_methods_observed"),
            ))
            observed_budgets = summary.get("budgets_observed")
            if isinstance(budgets, list) and not isinstance(observed_budgets, list):
                errors.append("`RUN_MANIFEST.json` results_summary budget observed set is missing or malformed.")
            elif isinstance(budgets, list) and isinstance(observed_budgets, list):
                try:
                    expected_budgets = sorted(int(item) for item in budgets)
                    actual_budgets = sorted(int(item) for item in observed_budgets)
                except (TypeError, ValueError):
                    expected_budgets = budgets
                    actual_budgets = observed_budgets
                if actual_budgets != expected_budgets:
                    errors.append(
                        "`RUN_MANIFEST.json` results_summary budget set drifts from experiment declaration."
                    )
            observed_repeats = summary.get("repeats_observed")
            if isinstance(n_repeats, int) and not isinstance(observed_repeats, list):
                errors.append("`RUN_MANIFEST.json` results_summary repeat observed set is missing or malformed.")
            elif isinstance(n_repeats, int) and isinstance(observed_repeats, list):
                expected_repeats = list(range(n_repeats))
                try:
                    actual_repeats = sorted(int(item) for item in observed_repeats)
                except (TypeError, ValueError):
                    actual_repeats = observed_repeats
                if actual_repeats != expected_repeats:
                    errors.append(
                        "`RUN_MANIFEST.json` results_summary repeat set drifts from experiment declaration."
                    )
        status_counts = summary.get("status_counts")
        if not isinstance(status_counts, dict):
            errors.append("`RUN_MANIFEST.json` results_summary.status_counts is missing or malformed.")
        else:
            status_total = 0
            status_total_valid = True
            unknown_statuses: list[str] = []
            for key, value in status_counts.items():
                if str(key) not in {"ok", "timeout", "failed"}:
                    unknown_statuses.append(str(key))
                parsed = _nonnegative_int_value(value)
                if parsed is None:
                    errors.append(
                        f"`RUN_MANIFEST.json` results_summary.status_counts has non-negative integer count required for {key!r}."
                    )
                    status_total_valid = False
                else:
                    status_total += parsed
            if unknown_statuses:
                errors.append(
                    f"`RUN_MANIFEST.json` results_summary.status_counts contains unknown statuses: {sorted(unknown_statuses)}."
                )
            if status_total_valid and summary.get("n_rows") is not None and status_total != summary.get("n_rows"):
                errors.append(
                    "`RUN_MANIFEST.json` results_summary.status_counts total does not match results_summary.n_rows."
                )
        if _nonnegative_int_from_summary(summary, "n_invalid_budget_mult_values", errors):
            errors.append("`RUN_MANIFEST.json` results_summary records invalid budget_mult values.")
        if _nonnegative_int_from_summary(summary, "n_invalid_repeat_values", errors):
            errors.append("`RUN_MANIFEST.json` results_summary records invalid repeat values.")
        row_contract_fields = {
            "n_missing_protocol_key_values": "rows with missing protocol key values",
            "n_invalid_status_values": "invalid status values",
            "n_invalid_ok_nrmse_values": "invalid ok-row nrmse values",
            "n_invalid_t_total_s_values": "invalid t_total_s values",
            "n_missing_failure_error_values": "timeout/failed rows without error reasons",
            "n_ok_rows_with_error_values": "ok rows with non-empty errors",
        }
        for key, label in row_contract_fields.items():
            if _nonnegative_int_from_summary(summary, key, errors):
                errors.append(f"`RUN_MANIFEST.json` results_summary records {label}.")
        try:
            import pandas as pd
            from scripts.prepare_figdata import validate_results_contract

            validate_results_contract(pd.read_csv(results_csv, low_memory=False))
        except AssertionError as exc:
            errors.append(f"`results.csv` row-level contract failed: {exc}.")
        except Exception as exc:
            errors.append(
                "`results.csv` row-level contract could not be validated: "
                f"{type(exc).__name__}: {exc}."
            )
    try:
        from scripts.check_submission_ready import (
            check_case_source_fingerprints,
            check_method_source_fingerprints,
        )
    except Exception as exc:
        errors.append(
            f"`RUN_MANIFEST.json` source fingerprint checks could not be loaded: {type(exc).__name__}: {exc}."
        )
    else:
        for error in check_case_source_fingerprints(path):
            errors.append(f"`RUN_MANIFEST.json` case source provenance failed: {error}.")
        for error in check_method_source_fingerprints(path):
            errors.append(f"`RUN_MANIFEST.json` method source provenance failed: {error}.")
    artifacts = manifest.get("artifacts")
    files = artifacts.get("files") if isinstance(artifacts, dict) else None
    if not isinstance(files, list) or not files:
        errors.append("`RUN_MANIFEST.json` artifacts.files is missing or empty.")
    else:
        results_entry = None
        sweep_report_entry = None
        artifact_paths: list[str] = []
        for item in files:
            if not isinstance(item, dict):
                errors.append("`RUN_MANIFEST.json` artifacts.files contains a non-object entry.")
                continue
            rel = str(item.get("path") or "").replace("\\", "/")
            if not rel:
                errors.append("`RUN_MANIFEST.json` artifacts.files contains an entry without path.")
                continue
            artifact_paths.append(rel)
            if rel == "results.csv":
                results_entry = item
            if rel == "sweep_integrity_report.md":
                sweep_report_entry = item
        duplicates = sorted({rel for rel in artifact_paths if artifact_paths.count(rel) > 1})
        if duplicates:
            errors.append(f"`RUN_MANIFEST.json` artifacts.files contains duplicate paths: {duplicates}.")
        if results_entry is None:
            errors.append("`RUN_MANIFEST.json` artifacts.files does not archive results.csv.")
        else:
            expected_size = results_entry.get("size_bytes")
            if expected_size is not None:
                try:
                    size_matches = int(expected_size) == results_csv.stat().st_size
                except (TypeError, ValueError):
                    size_matches = False
                if not size_matches:
                    errors.append(
                        "`RUN_MANIFEST.json` results.csv artifact size_bytes does not match the current file."
                    )
            expected_sha = results_entry.get("sha256")
            if expected_sha and str(expected_sha).lower() != _sha256(results_csv).lower():
                errors.append(
                    "`RUN_MANIFEST.json` results.csv artifact sha256 does not match the current file."
                )
        if sweep_report_entry is None:
            errors.append("`RUN_MANIFEST.json` artifacts.files does not archive sweep_integrity_report.md.")
        else:
            expected_size = sweep_report_entry.get("size_bytes")
            if expected_size is not None:
                try:
                    size_matches = int(expected_size) == sweep_report.stat().st_size
                except (TypeError, ValueError):
                    size_matches = False
                if not size_matches:
                    errors.append(
                        "`RUN_MANIFEST.json` sweep_integrity_report.md artifact size_bytes does not match the current file."
                    )
            expected_sha = sweep_report_entry.get("sha256")
            if expected_sha and str(expected_sha).lower() != _sha256(sweep_report).lower():
                errors.append(
                    "`RUN_MANIFEST.json` sweep_integrity_report.md artifact sha256 does not match the current file."
                )
    return errors


def summarize_case_readiness(path: Path = CASE_READINESS) -> GateResult:
    if not path.exists():
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"`{path.relative_to(ROOT).as_posix()}` is missing.",
            "Run `python scripts/audit_case_experiment_readiness.py --output docs/case_experiment_readiness.csv --fail-on-warning`.",
        )

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            raw_columns = list(reader.fieldnames or [])
            rows = list(reader)
    except UnicodeDecodeError as exc:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            (
                f"`{_display_path(path)}` could not be read as UTF-8 CSV: "
                f"{type(exc).__name__}: {exc}"
            ),
            (
                "Restore or regenerate a valid UTF-8 "
                "`docs/case_experiment_readiness.csv` with "
                "`python scripts/audit_case_experiment_readiness.py --output "
                "docs/case_experiment_readiness.csv --fail-on-warning`."
            ),
        )
    except csv.Error as exc:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"`{_display_path(path)}` is not a valid CSV audit: {type(exc).__name__}: {exc}",
            (
                "Regenerate `docs/case_experiment_readiness.csv` with the current "
                "`scripts/audit_case_experiment_readiness.py` before launching the "
                "full sweep."
            ),
        )

    if not raw_columns:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"`{_display_path(path)}` is missing a CSV header row.",
            (
                "Regenerate `docs/case_experiment_readiness.csv` with the current "
                "`scripts/audit_case_experiment_readiness.py`."
            ),
        )
    blank_columns = [repr(column) for column in raw_columns if not str(column or "").strip()]
    duplicate_columns = sorted({
        str(column).strip()
        for column in raw_columns
        if str(column or "").strip() and raw_columns.count(column) > 1
    })
    if blank_columns or duplicate_columns:
        details: list[str] = []
        if blank_columns:
            details.append(f"blank header(s)={blank_columns}")
        if duplicate_columns:
            details.append(f"duplicate header(s)={duplicate_columns}")
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"`{_display_path(path)}` has an invalid CSV header: {'; '.join(details)}.",
            (
                "Regenerate `docs/case_experiment_readiness.csv` with the current "
                "`scripts/audit_case_experiment_readiness.py`."
            ),
        )

    ragged_rows = [
        idx + 2
        for idx, row in enumerate(rows)
        if None in row
    ]
    if ragged_rows:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            (
                f"`{_display_path(path)}` has ragged CSV row(s) with extra cells at "
                f"line(s) {ragged_rows[:5]}"
                + (f" (+{len(ragged_rows) - 5} more)" if len(ragged_rows) > 5 else "")
                + "."
            ),
            (
                "Regenerate `docs/case_experiment_readiness.csv` with the current "
                "`scripts/audit_case_experiment_readiness.py`."
            ),
        )

    columns = set(raw_columns)
    statuses: dict[str, int] = {}
    for row in rows:
        status = (row.get("status") or "missing").strip()
        statuses[status] = statuses.get(status, 0) + 1

    fatal = {key: value for key, value in statuses.items() if key != "ok"}
    evidence = f"{len(rows)} case rows; statuses={statuses}."
    missing_columns = sorted(CASE_READINESS_REQUIRED_COLUMNS - columns)
    if missing_columns:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"{evidence} Missing required columns: {missing_columns}.",
            "Regenerate the audit with the current `scripts/audit_case_experiment_readiness.py`.",
        )
    quality_errors: list[str] = []
    metadata_errors: list[str] = []
    for row in rows:
        case_id = row.get("case_id", "<unknown>")
        for column in [
            "n_nan",
            "n_inf",
            "input_mutation_n_changed",
            "determinism_n_mismatch",
            "n_expected_range_violations",
        ]:
            raw = str(row.get(column, "")).strip()
            try:
                value = int(raw)
            except ValueError:
                quality_errors.append(f"{case_id}: {column} is not an integer ({raw!r})")
                continue
            if value != 0:
                quality_errors.append(f"{case_id}: {column}={value}")
        for column in [
            "domain",
            "pde_type",
            "response_topology",
            "governing_equation",
        ]:
            if not str(row.get(column, "")).strip():
                metadata_errors.append(f"{case_id}: {column} is empty")
        try:
            n_topology_tags = int(str(row.get("n_topology_tags", "")).strip())
        except ValueError:
            metadata_errors.append(
                f"{case_id}: n_topology_tags is not an integer "
                f"({row.get('n_topology_tags', '')!r})"
            )
            n_topology_tags = -1
        if n_topology_tags < 1:
            metadata_errors.append(f"{case_id}: n_topology_tags={n_topology_tags}")
        topology_tags = [
            item.strip()
            for item in str(row.get("response_topology", "")).split(";")
            if item.strip()
        ]
        if n_topology_tags >= 0 and len(topology_tags) != n_topology_tags:
            metadata_errors.append(
                f"{case_id}: n_topology_tags={n_topology_tags} but "
                f"response_topology has {len(topology_tags)} tag(s)"
            )
        try:
            n_refs = int(str(row.get("n_physics_references", "")).strip())
        except ValueError:
            metadata_errors.append(
                f"{case_id}: n_physics_references is not an integer "
                f"({row.get('n_physics_references', '')!r})"
            )
            n_refs = -1
        if n_refs < 1:
            metadata_errors.append(f"{case_id}: n_physics_references={n_refs}")
    if len(rows) != 34:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            evidence,
            "Regenerate the audit and confirm all 34 preregistered cases are represented.",
        )
    if fatal:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            evidence,
            (
                "Fix non-finite outputs, interface errors, missing failure-threshold "
                "metadata, or timing metadata drift before the server full sweep."
            ),
        )
    if quality_errors:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"{evidence} Determinism/finite-output/range errors: {quality_errors[:5]}.",
            (
                "Fix case evaluate determinism, non-finite outputs, or expected_range metadata drift, then rerun "
                "`python scripts/audit_case_experiment_readiness.py --output docs/case_experiment_readiness.csv --fail-on-warning`."
            ),
        )
    if metadata_errors:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"{evidence} Case metadata readiness errors: {metadata_errors[:5]}.",
            (
                "Fix response_topology/PDE/governing-equation/reference metadata drift, then rerun "
                "`python scripts/audit_case_experiment_readiness.py --output docs/case_experiment_readiness.csv --fail-on-warning`."
            ),
        )
    expected_cases, registry_error = _registered_case_ids()
    if registry_error:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"{evidence} {registry_error}.",
            "Fix the current case registry import before trusting case-readiness evidence.",
        )
    observed_cases = [str(row.get("case_id", "")).strip() for row in rows]
    observed_set = {case_id for case_id in observed_cases if case_id}
    expected_set = set(expected_cases or [])
    duplicates = sorted({case_id for case_id in observed_cases if observed_cases.count(case_id) > 1})
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    blanks = sum(1 for case_id in observed_cases if not case_id)
    if missing or extra or duplicates or blanks:
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"extra={extra}")
        if duplicates:
            detail.append(f"duplicates={duplicates}")
        if blanks:
            detail.append(f"blank_case_id_rows={blanks}")
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"{evidence} Case coverage drift: {'; '.join(detail)}.",
            "Regenerate the audit against the current registered case library before launching the full sweep.",
        )
    expected_metadata, metadata_error = _registered_case_metadata()
    if metadata_error:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"{evidence} {metadata_error}.",
            "Fix the current case registry metadata import before trusting case-readiness evidence.",
        )
    metadata_drift: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        expected = (expected_metadata or {}).get(case_id, {})
        for column in [
            "domain",
            "pde_type",
            "response_topology",
            "n_topology_tags",
            "governing_equation",
            "n_physics_references",
        ]:
            observed = str(row.get(column, "")).strip()
            if column == "governing_equation":
                observed = _one_line(observed)
            want = str(expected.get(column, "")).strip()
            if observed != want:
                metadata_drift.append(
                    f"{case_id}: {column} observed={observed!r} expected={want!r}"
                )
    if metadata_drift:
        return GateResult(
            "Case experiment readiness",
            "BLOCKED",
            f"{evidence} Case metadata drift vs registry: {metadata_drift[:5]}.",
            (
                "Regenerate `docs/case_experiment_readiness.csv` against the current "
                "case registry before launching the full sweep."
            ),
        )
    return GateResult(
        "Case experiment readiness",
        "PASS",
        evidence,
        "No action.",
    )


def summarize_case_validation(
    status_csv: Path = CASE_VALIDATION_STATUS,
    sources_csv: Path = CASE_VALIDATION_SOURCES,
    note_sources_csv: Path = CASE_VALIDATION_NOTE_SOURCES,
    summary_md: Path = CASE_VALIDATION_SUMMARY,
) -> GateResult:
    status_rows = _count_csv_rows(status_csv)
    missing = [
        rel
        for path, rel in [
            (status_csv, _display_path(status_csv)),
            (sources_csv, _display_path(sources_csv)),
            (note_sources_csv, _display_path(note_sources_csv)),
            (summary_md, _display_path(summary_md)),
        ]
        if not path.exists()
    ]
    if missing:
        return GateResult(
            "Case physics validation ledger",
            "BLOCKED",
            f"Missing required case-validation artifact(s): {missing}.",
            (
                "Regenerate the ledger/summary, then rerun "
                "`python -m pytest tests/test_case_validation_status.py -q` and "
                "`python scripts/run_case_physics_tests.py -q`."
            ),
        )
    try:
        from scripts.check_submission_ready import (
            check_case_validation_sources,
            check_case_validation_note_sources,
            check_case_validation_status,
            check_case_validation_summary,
        )
    except Exception as exc:
        return GateResult(
            "Case physics validation ledger",
            "BLOCKED",
            f"Could not load strict case-validation checks: {type(exc).__name__}: {exc}.",
            "Fix the readiness helper import and rerun `python scripts/audit_publication_readiness.py`.",
        )

    errors = check_case_validation_status(status_csv.resolve())
    errors.extend(check_case_validation_sources(status_csv.resolve(), sources_csv.resolve()))
    errors.extend(check_case_validation_note_sources(note_sources_csv.resolve()))
    errors.extend(
        check_case_validation_summary(
            status_csv.resolve(),
            summary_md.resolve(),
            note_sources_csv.resolve(),
        )
    )
    if errors:
        shown = errors[:8]
        suffix = "" if len(errors) <= len(shown) else f" (+{len(errors) - len(shown)} more)."
        return GateResult(
            "Case physics validation ledger",
            "BLOCKED",
            f"{status_rows} case rows. Strict case-validation errors: {' '.join(shown)}{suffix}",
            (
                "Repair the ledger, source fingerprints, or generated summary; then rerun "
                "`python -m pytest tests/test_case_validation_status.py -q` and "
                "`python scripts/run_case_physics_tests.py -q`."
            ),
        )
    if status_rows != 34:
        return GateResult(
            "Case physics validation ledger",
            "BLOCKED",
            f"`docs/case_validation_status.csv` has {status_rows} rows, expected 34.",
            "Repair the ledger so every preregistered case is covered exactly once.",
        )
    return GateResult(
        "Case physics validation ledger",
        "PASS",
        "Ledger, source/note fingerprints, and generated summary are current; 34 registered case rows passed strict submission-readiness checks.",
        (
            "Keep source and validation-note fingerprints fresh after any case-code "
            "or review-material edit, then rerun `python scripts/run_case_physics_tests.py -q`."
        ),
    )


def _read_case_validation_rows(path: Path) -> tuple[list[dict[str, str]], str | None]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f)), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _format_counts(counts: Counter[str], keys: Iterable[str]) -> str:
    return ", ".join(f"{key}: {counts.get(key, 0)}" for key in keys)


def summarize_case_evidence_tiers(
    status_csv: Path = CASE_VALIDATION_STATUS,
    summary_md: Path = CASE_VALIDATION_SUMMARY,
    audit_md: Path = CASE_WEB_VALIDATION_AUDIT,
    *,
    expected_n_cases: int = EXPECTED_N_CASES,
) -> GateResult:
    """Make evidence-tier honesty an explicit pre-launch publication gate."""
    missing = [
        rel
        for path, rel in [
            (status_csv, _display_path(status_csv)),
            (summary_md, _display_path(summary_md)),
            (audit_md, _display_path(audit_md)),
        ]
        if not path.exists()
    ]
    if missing:
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            f"Missing required evidence-tier artifact(s): {missing}.",
            (
                "Regenerate the case-validation summary and independent P2 "
                "web/literature audit before launch."
            ),
        )

    rows, row_error = _read_case_validation_rows(status_csv)
    if row_error:
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            f"Could not read `{_display_path(status_csv)}`: {row_error}.",
            "Repair the case-validation ledger CSV before launch.",
        )
    if len(rows) != expected_n_cases:
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            (
                f"`{_display_path(status_csv)}` has {len(rows)} rows, "
                f"expected {expected_n_cases}."
            ),
            "Repair the ledger so every preregistered case is covered exactly once.",
        )

    try:
        from scripts.generate_case_validation_summary import evidence_tier
    except Exception as exc:
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            f"Could not load evidence-tier classifier: {type(exc).__name__}: {exc}.",
            "Fix `scripts/generate_case_validation_summary.py` before launch.",
        )

    risk_counts = Counter(str(row.get("risk_level", "")).strip() for row in rows)
    if risk_counts.get("P0", 0):
        p0_cases = [
            str(row.get("case_id", "")).strip()
            for row in rows
            if str(row.get("risk_level", "")).strip() == "P0"
        ]
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            f"P0/unaccepted case rows remain in the launch ledger: {p0_cases}.",
            "Resolve or explicitly remove P0 status before launching the headline sweep.",
        )

    missing_tests = [
        str(row.get("case_id", "")).strip() or "<missing case_id>"
        for row in rows
        if str(row.get("targeted_physics_test", "")).strip() in {"", "NONE"}
    ]
    if missing_tests:
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            f"Case row(s) missing targeted physics tests: {missing_tests}.",
            "Add or restore targeted physics tests before launching the headline sweep.",
        )

    tier_counts = Counter(evidence_tier(row) for row in rows)
    try:
        from scripts.check_server_launch_ready import (
            validate_case_web_validation_audit_coverage,
            validate_p2_source_verification_snapshots,
        )
    except Exception as exc:
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            f"Could not load P2 web-validation audit checker: {type(exc).__name__}: {exc}.",
            "Fix `scripts/check_server_launch_ready.py` before launch.",
        )

    audit_errors = validate_case_web_validation_audit_coverage(
        status_csv=status_csv,
        audit_md=audit_md,
    )
    audit_errors.extend(
        validate_p2_source_verification_snapshots(
            status_csv=status_csv,
            audit_md=audit_md,
        )
    )
    if audit_errors:
        shown = audit_errors[:5]
        suffix = "" if len(audit_errors) <= len(shown) else f" (+{len(audit_errors) - len(shown)} more)"
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            f"P2 evidence-tier audit coverage failed: {' '.join(shown)}{suffix}.",
            (
                "Update `docs/CASE_WEB_VALIDATION_AUDIT.md` so every P2 case has "
                "an explicit independent web/literature row, `Evidence tier remains P2`, "
                "and a keep/fix/resweep judgment."
            ),
        )

    audit_text = audit_md.read_text(encoding="utf-8", errors="replace")
    required_audit_markers = [
        "Evidence tier remains P2",
    ]
    missing_audit_markers = [
        marker for marker in required_audit_markers if marker not in audit_text
    ]
    if missing_audit_markers:
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            (
                f"`{_display_path(audit_md)}` is missing independent-source "
                f"snapshot marker(s): {missing_audit_markers}."
            ),
            (
                "Restore the P2 source-verification snapshot markers so public "
                "sources are recorded as boundary/context evidence rather than "
                "external validation claims."
            ),
        )

    summary_text = summary_md.read_text(encoding="utf-8", errors="replace")
    required_summary_markers = [
        "## Evidence Tiers",
        "case has the same external validation strength",
        "P2/plausibility-only rows",
        "do not claim strong external case validation",
    ]
    missing_summary_markers = [
        marker for marker in required_summary_markers if marker not in summary_text
    ]
    if missing_summary_markers:
        return GateResult(
            "Case evidence tiers",
            "BLOCKED",
            (
                f"`{_display_path(summary_md)}` is missing evidence-tier honesty "
                f"marker(s): {missing_summary_markers}."
            ),
            (
                "Regenerate or repair the summary so P2/plausibility-only cases are "
                "clearly separated from externally validated cases."
            ),
        )

    p2_cases = [
        str(row.get("case_id", "")).strip()
        for row in rows
        if str(row.get("risk_level", "")).strip() == "P2"
    ]
    return GateResult(
        "Case evidence tiers",
        "PASS",
        (
            f"{len(rows)} case rows; risk counts "
            f"{_format_counts(risk_counts, ['P0', 'P1', 'P2'])}; evidence tiers "
            f"{_format_counts(tier_counts, sorted(tier_counts))}; "
            f"{len(p2_cases)} P2 case(s) have explicit web/literature audit rows "
            "and remain labelled as P2 where appropriate."
        ),
        (
            "Keep P2 cases in the benchmark only with explicit P2 wording; upgrade "
            "tiers only after adding stronger validation evidence and regenerating "
            "the summary."
        ),
    )


def summarize_sweep_artifacts(results_dir: Path = FULL_SWEEP) -> GateResult:
    results_csv = results_dir / "results.csv"
    manifest = results_dir / "RUN_MANIFEST.json"
    audit_report = results_dir / "sweep_integrity_report.md"
    if not results_csv.exists():
        return GateResult(
            "Full sweep results",
            "BLOCKED",
            f"`{_display_path(results_csv)}` is missing.",
            (
                "Run or restore the complete Stage 2 sweep on the designated "
                "server; see `docs/SERVER_RUN_HANDOFF.md`, then rerun "
                "`sweep_audit`."
            ),
        )
    row_count, row_error = _count_csv_rows_checked(results_csv)
    if row_error:
        return GateResult(
            "Full sweep results",
            "BLOCKED",
            f"could not read `{_display_path(results_csv)}`: {row_error}",
            (
                "Restore a valid UTF-8 CSV from the server archive or rerun "
                "Stage 2 before trusting downstream paper artifacts."
            ),
        )
    expected_rows = 34 * 20 * 4 * 30
    evidence_parts = [f"`results.csv` has {row_count} rows."]
    if manifest.exists():
        evidence_parts.append("`RUN_MANIFEST.json` exists.")
    else:
        evidence_parts.append("`RUN_MANIFEST.json` is missing.")
    if audit_report.exists():
        evidence_parts.append("`sweep_integrity_report.md` exists.")
    else:
        evidence_parts.append("`sweep_integrity_report.md` is missing.")
    if row_count != expected_rows:
        return GateResult(
            "Full sweep results",
            "BLOCKED",
            " ".join(evidence_parts),
            (
                "Complete/resume the server sweep until the preregistered "
                "81,600-row grid is present."
            ),
        )
    if not audit_report.exists():
        return GateResult(
            "Full sweep results",
            "WARN",
            " ".join(evidence_parts),
            "Run `python scripts/run_pipeline.py --only sweep_audit --confirm-full-run` before Stage 3-10 artifacts are trusted.",
        )
    if not _audit_report_passed(audit_report):
        return GateResult(
            "Full sweep results",
            "BLOCKED",
            " ".join(evidence_parts) + " Sweep audit did not contain a PASS verdict.",
            (
                "Inspect `sweep_integrity_report.md`, fix the reported grid, "
                "schema, provenance, or failure-rate issue, and rerun `sweep_audit`."
            ),
        )
    if not manifest.exists():
        return GateResult(
            "Full sweep results",
            "BLOCKED",
            " ".join(evidence_parts),
            (
                "Run Stage 10 manifest generation after the passed sweep audit so "
                "the archive has command history, source fingerprints, and file hashes."
            ),
        )
    if manifest.stat().st_size == 0:
        return GateResult(
            "Full sweep results",
            "BLOCKED",
            " ".join(evidence_parts) + " `RUN_MANIFEST.json` is empty.",
            "Regenerate `RUN_MANIFEST.json` with `--require-complete-archive`.",
        )
    manifest_errors = _manifest_structure_errors(
        manifest,
        row_count=row_count,
        results_csv=results_csv,
        sweep_report=audit_report,
    )
    if manifest_errors:
        return GateResult(
            "Full sweep results",
            "BLOCKED",
            " ".join(evidence_parts) + " " + " ".join(manifest_errors),
            "Regenerate `RUN_MANIFEST.json` with `--require-complete-archive`.",
        )
    rc, fresh_audit = _rerun_sweep_integrity(results_csv)
    if rc != 0:
        excerpt = " ".join(fresh_audit.split())[:1000]
        return GateResult(
            "Full sweep results",
            "BLOCKED",
            " ".join(evidence_parts)
            + f" Fresh local sweep-integrity audit failed with exit code {rc}: {excerpt}",
            (
                "Rerun `python scripts/audit_sweep_integrity.py "
                "benchmark/results/full_benchmark/results.csv --expected-cases ...` "
                "or restore a clean server archive before trusting paper stages."
            ),
        )
    evidence_parts.append("fresh local sweep-integrity audit PASS.")
    return GateResult(
        "Full sweep results",
        "PASS",
        " ".join(evidence_parts),
        "No action.",
    )


def summarize_server_return_package(
    sweep_gate: GateResult,
    *,
    results_root: Path = RESULTS_ROOT,
    results_dir: Path = FULL_SWEEP,
    paper_dir: Path = PAPER,
    logs_dir: Path = ROOT / "logs",
) -> GateResult:
    if sweep_gate.status != "PASS":
        return GateResult(
            "Server return package",
            "DEFERRED",
            (
                "Post-return package checks depend on a passed full-sweep gate; "
                f"current `Full sweep results` status is `{sweep_gate.status}`."
            ),
            (
                "Restore or complete the server full-sweep archive, then run "
                "`python scripts/check_server_return_package.py`."
            ),
        )
    try:
        from scripts.check_server_return_package import check_server_return_package
    except Exception as exc:
        return GateResult(
            "Server return package",
            "BLOCKED",
            f"Could not load post-return package checks: {type(exc).__name__}: {exc}.",
            "Fix `scripts/check_server_return_package.py` import errors and rerun the audit.",
        )

    groups = check_server_return_package(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        logs_dir=logs_dir,
    )
    errors = [
        f"{group}: {error}"
        for group, group_errors in groups.items()
        for error in group_errors
    ]
    if errors:
        shown = errors[:8]
        suffix = f" (+{len(errors) - len(shown)} more)" if len(errors) > len(shown) else ""
        return GateResult(
            "Server return package",
            "BLOCKED",
            " ".join(shown) + suffix,
            (
                "Restore missing archive pieces or regenerate downstream artifacts, "
                "then rerun `python scripts/check_server_return_package.py`."
            ),
        )
    return GateResult(
        "Server return package",
        "PASS",
        (
            "Returned full-sweep archive, manifest, command history, D1/D2/D3, "
            "figdata, figures, and numbers passed the post-return package checks."
        ),
        "No action.",
    )


def summarize_smoke_pipeline(results_root: Path = RESULTS_ROOT) -> GateResult:
    candidates = sorted(
        (
            path
            for path in results_root.glob("_smoke*")
            if path.is_dir() and (path / "results.csv").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return GateResult(
            "Smoke pipeline",
            "WARN",
            "No `_smoke*` results.csv was found.",
            "Run `python scripts/run_pipeline.py --profile smoke --to-stage 2` after pipeline edits.",
        )

    passed: list[Path] = []
    fresh_failures: list[str] = []
    for path in candidates:
        if not _smoke_audit_passed(path):
            continue
        rc, fresh_audit = _rerun_smoke_integrity(path / "results.csv")
        if rc == 0:
            passed.append(path)
        else:
            excerpt = " ".join(fresh_audit.split())[:500]
            fresh_failures.append(
                f"`{_display_path(path)}` fresh smoke audit failed with exit code {rc}: {excerpt}"
            )
    if passed:
        latest = passed[0]
        row_count, row_error = _count_csv_rows_checked(latest / "results.csv")
        if row_error:
            return GateResult(
                "Smoke pipeline",
                "WARN",
                f"Latest passing smoke directory `{_display_path(latest)}` has an unreadable results.csv: {row_error}",
                "Rerun `python scripts/run_pipeline.py --profile smoke` so smoke results and audit are regenerated together.",
            )
        report = latest / "report" / "REPORT.md"
        figs = latest / "report" / "figs"
        n_figs = len(list(figs.glob("*.png"))) if figs.exists() else 0
        if not report.exists() or n_figs == 0:
            missing = []
            if not report.exists():
                missing.append("report/REPORT.md")
            if n_figs == 0:
                missing.append("report/figs/*.png")
            return GateResult(
                "Smoke pipeline",
                "WARN",
                (
                    f"Latest passing smoke directory `{_display_path(latest)}` has "
                    f"{row_count} result rows and a passing sweep audit, but missing "
                    f"analysis artifact(s): {missing}."
                ),
                "Run `python scripts/run_pipeline.py --profile smoke --only report --sweep-dir <smoke-dir>`.",
            )
        return GateResult(
            "Smoke pipeline",
            "PASS",
            (
                f"Latest passing smoke directory `{_display_path(latest)}` has "
                f"{row_count} result rows, a passing archived and fresh sweep audit, "
                f"REPORT.md, and {n_figs} figure(s)."
            ),
            "No action.",
        )

    latest = candidates[0]
    results_csv = latest / "results.csv"
    audit_report = latest / "sweep_integrity_report.md"
    row_count, row_error = _count_csv_rows_checked(results_csv)
    if row_error:
        return GateResult(
            "Smoke pipeline",
            "WARN",
            f"Latest smoke directory `{_display_path(latest)}` has an unreadable results.csv: {row_error}",
            "Rerun `python scripts/run_pipeline.py --profile smoke` so smoke results and audit are regenerated together.",
        )
    evidence = f"Latest smoke directory `{_display_path(latest)}` has {row_count} result rows."
    if fresh_failures:
        evidence += " " + " ".join(fresh_failures[:2])
    if not audit_report.exists():
        return GateResult(
            "Smoke pipeline",
            "WARN",
            evidence + " `sweep_integrity_report.md` is missing.",
            "Rerun smoke through `--to-stage 2` so the sweep audit executes.",
        )
    if not _smoke_audit_passed(latest):
        return GateResult(
            "Smoke pipeline",
            "BLOCKED",
            evidence + " Sweep audit did not contain a PASS verdict.",
            "Inspect the latest smoke `sweep_integrity_report.md` and fix runner/audit drift.",
        )
    return GateResult(
        "Smoke pipeline",
        "WARN",
        evidence + " Archived smoke audit contains PASS, but no smoke directory passed a fresh local audit.",
        "Rerun `python scripts/run_pipeline.py --profile smoke` so smoke results, audit, report, and figures are regenerated together.",
    )


def _smoke_audit_passed(path: Path) -> bool:
    return _audit_report_passed(path / "sweep_integrity_report.md")


def _audit_report_passed(audit_report: Path) -> bool:
    if not audit_report.exists():
        return False
    text = audit_report.read_text(encoding="utf-8", errors="replace")
    return SWEEP_PASS_VERDICT in text


def summarize_benchmark_protocol_audit(
    audit_path: Path = EXPERIMENT_BENCHMARK_AUDIT,
) -> GateResult:
    """Check the first-principles benchmark/protocol audit before launch."""
    if not audit_path.exists():
        return GateResult(
            "Benchmark protocol audit",
            "BLOCKED",
            f"`{_display_path(audit_path)}` is missing.",
            "Restore `docs/EXPERIMENT_BENCHMARK_AUDIT.md` before launching the server sweep.",
        )

    text = audit_path.read_text(encoding="utf-8", errors="replace")
    required_markers = [
        "Current check date: 2026-06-09",
        "External Source Verification Matrix",
        "Nature Computational Science Fit and Reporting Controls",
        "Reviewer-Facing Pre-Launch Acceptance Checklist",
        "Reviewer Smoke-Reproduction Drill",
        "Nature Portfolio reporting standards",
        "Nature Computational Science submission guidance",
        "development and use of computational techniques and mathematical models",
        "cross-domain engineering/physics DOE resource",
        "Code Availability section",
        "permanent identifiers",
        "reviewer rerun path",
        "GO FAIR FAIR Principles",
        "ACM artifact review badging",
        "NeurIPS 2026 Evaluations & Datasets benchmark/reusable-artifact guidance",
        "OpenML benchmark suites and run records",
        "What is a reusable run record?",
        "Independently re-evaluable runs",
        "without access to the live server or hidden notebooks",
        "MLCommons/MLPerf benchmark practice",
        "COCO/BBOB, HPOBench, YAHPO Gym, and Bayesmark",
        "34 cases × 20 `nature_pool()` methods × budgets `{5,10,20,50}` × 30 repeats",
        "`results.csv`",
        "`run_config.json`",
        "`RUN_MANIFEST.json`",
        "`protocol_digest`",
        "runtime trace",
        "failure_category",
        "Server-launch proof obligation",
        "Physics-benchmark honesty",
        "run-config/report mismatch",
        "dead while the headline grid is missing",
        "representative exercisability check",
        "python scripts/run_pipeline.py --profile smoke --dry-run",
        "python scripts/run_pipeline.py --profile smoke",
        "Server return package",
        "immutable DOI",
        "Do not launch the full server run until",
        "python scripts/check_server_launch_ready.py --require-clean-sweep-dir",
        "bash scripts/run_pipeline.sh",
        "strict clean-sweep",
        "DOE_BENCHMARK_ALLOW_STALE_DERIVED=1",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        return GateResult(
            "Benchmark protocol audit",
            "BLOCKED",
            (
                f"`{_display_path(audit_path)}` is missing required benchmark-control "
                f"marker(s): {missing[:8]}"
                + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
            ),
            (
                "Update the experiment benchmark audit so external-source, protocol, "
                "runtime, failure, FAIR/data-package, and launch-rule controls are explicit."
            ),
        )

    return GateResult(
        "Benchmark protocol audit",
        "PASS",
        (
            "External-source matrix, frozen protocol, row-level provenance, "
            "runtime/failure recovery, FAIR data package, and launch-rule "
            "controls are documented in `docs/EXPERIMENT_BENCHMARK_AUDIT.md`."
        ),
        "Keep this audit current after protocol, result-contract, monitoring, recovery, or analysis-pipeline edits.",
    )


def summarize_draft_gate(paper_dir: Path = PAPER) -> GateResult:
    cmd = [sys.executable, "scripts/check_paper_draft_ready.py", str(paper_dir)]
    rc, out = _run_short_gate(cmd)
    if rc == 0:
        return GateResult(
            "Pre-results paper draft",
            "PASS",
            _last_lines(out),
            "Keep RESULT_PENDING markers until the full sweep and signed decisions are final.",
        )
    return GateResult(
        "Pre-results paper draft",
        "BLOCKED",
        _last_lines(out),
        "Fix the draft package before continuing manuscript writing.",
    )


def summarize_manuscript_stage(
    paper_dir: Path = PAPER,
    *,
    results_dir: Path = FULL_SWEEP,
    sweep_gate: GateResult | None = None,
) -> GateResult:
    """Report the manuscript stage without applying the wrong paper gate.

    The draft gate is intentionally for the pre-results state.  Once a
    headline ``results.csv`` exists, RESULT_PENDING and draft-only structure
    checks are no longer the right diagnostic; the strict submission gate is
    authoritative after the sweep gate passes.  Keep the draft fallback for
    launch-preflight and older/missing-result trees.
    """
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        return summarize_draft_gate(paper_dir)

    rows, read_error = _count_csv_rows_checked(results_csv)
    row_text = "headline results.csv is present"
    if read_error:
        row_text += f" but could not be counted ({read_error})"
    elif rows is not None:
        row_text += f" ({rows:,} data rows)"

    if sweep_gate is not None and sweep_gate.status == "PASS":
        return GateResult(
            "Results-filled manuscript",
            "PASS",
            (
                f"{row_text}; the pre-results draft gate is not applicable. "
                "The strict final submission gate is the authoritative paper check."
            ),
            "Run the final submission gate and complete human/archive sign-off.",
        )

    return GateResult(
        "Results-filled manuscript",
        "WARN",
        (
            f"{row_text}; the pre-results draft gate is not applicable while "
            "the full-sweep gate is unresolved."
        ),
        "Resolve the full-sweep/provenance gate, then run check_submission_ready.py.",
    )


def summarize_submission_gate(paper_dir: Path = PAPER, results_dir: Path = FULL_SWEEP) -> GateResult:
    cmd = [
        sys.executable,
        "scripts/check_submission_ready.py",
        str(paper_dir),
        "--manifest",
        str(results_dir / "RUN_MANIFEST.json"),
        "--numbers",
        str(paper_dir / "numbers.json"),
        "--numbers-tex",
        str(paper_dir / "numbers.tex"),
        "--figure-manifest",
        str(paper_dir / "figure_manifest.json"),
        "--figdata-dir",
        str(paper_dir / "figdata"),
        "--case-validation-summary",
        str(CASE_VALIDATION_SUMMARY),
    ]
    rc, out = _run_short_gate(cmd)
    if rc == 0:
        return GateResult(
            "Final submission gate",
            "PASS",
            _last_lines(out),
            "Ready for human sign-off and archive DOI insertion.",
        )
    return GateResult(
        "Final submission gate",
        "BLOCKED",
        _last_lines(out),
        "Resolve all reported pending manuscript tokens, stale numbers, manifest, figure, and lineage errors.",
    )


def defer_submission_gate(sweep_gate: GateResult) -> GateResult:
    return GateResult(
        "Final submission gate",
        "DEFERRED",
        (
            "Strict submission-ready checks depend on a passed full-sweep gate; "
            f"current `Full sweep results` status is `{sweep_gate.status}`."
        ),
        (
            "Restore or complete the server full-sweep archive, rerun "
            "`sweep_audit`, then run `check_submission_ready.py` through the "
            "pipeline final stage."
        ),
    )


def _last_lines(text: str, *, n: int = 8) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return "<no output>"
    return " / ".join(lines[-n:])


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def render_report(
    results: Iterable[GateResult],
    *,
    title: str = "Publication Readiness Audit",
) -> str:
    items = list(results)
    status_order = {"BLOCKED": 0, "WARN": 1, "DEFERRED": 2, "PASS": 3}
    if any(x.status == "BLOCKED" for x in items):
        overall = "BLOCKED"
    elif any(x.status == "WARN" for x in items):
        overall = "WARN"
    else:
        overall = "PASS"
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        f"# {title}",
        "",
        f"Generated UTC: `{generated}`",
        "",
        f"Overall status: **{overall}**",
        "",
        "| Gate | Status | Evidence | Next action |",
        "|---|---|---|---|",
    ]
    for item in sorted(items, key=lambda x: (status_order.get(x.status, -1), x.name)):
        lines.append(
            "| {name} | {status} | {evidence} | {next_action} |".format(
                name=item.name,
                status=item.status,
                evidence=_md_cell(item.evidence),
                next_action=_md_cell(item.next_action),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `PASS` means the current artifact is internally consistent for this gate.",
        "- `WARN` means the artifact is usable but deserves review before the next long run.",
        "- `DEFERRED` means the gate is intentionally postponed until an upstream artifact is present.",
        "- `BLOCKED` means the current worktree cannot yet support final paper claims for that gate.",
        "- Case experiment `timing_warning` rows are `BLOCKED` here because the formal Stage 1 preflight uses `--fail-on-warning`.",
        "",
    ])
    return "\n".join(lines)


def _md_cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


def write_report_atomic(report: str, output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=str(output.parent))
    )
    tmp_output = tmp_dir / output.name
    try:
        tmp_output.write_text(report, encoding="utf-8")
        shutil.move(str(tmp_output), str(output))
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a Markdown roll-up of publication-readiness gates.",
    )
    parser.add_argument("--paper-dir", type=Path, default=PAPER)
    parser.add_argument("--results-dir", type=Path, default=FULL_SWEEP)
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-submission-gate",
        action="store_true",
        help="Skip the strict submission-ready subprocess check.",
    )
    parser.add_argument(
        "--launch-preflight",
        action="store_true",
        help=(
            "Only run pre-full-sweep gates that should pass before launching "
            "the server headline sweep."
        ),
    )
    return parser.parse_args(argv)


def _launch_preflight_results(args: argparse.Namespace) -> list[GateResult]:
    results_root = args.results_dir.resolve().parent
    paper_dir = args.paper_dir.resolve()
    return [
        _safe_gate("Case physics validation ledger", summarize_case_validation),
        _safe_gate("Case evidence tiers", summarize_case_evidence_tiers),
        _safe_gate("Case experiment readiness", summarize_case_readiness),
        _safe_gate("Benchmark protocol audit", summarize_benchmark_protocol_audit),
        _safe_gate("Smoke pipeline", lambda: summarize_smoke_pipeline(results_root)),
        _safe_gate("Pre-results paper draft", lambda: summarize_draft_gate(paper_dir)),
    ]


def _publication_readiness_results(args: argparse.Namespace) -> list[GateResult]:
    results_dir = args.results_dir
    results_root = results_dir.parent
    paper_dir = args.paper_dir
    logs_dir = args.logs_dir
    sweep_gate = _safe_gate("Full sweep results", lambda: summarize_sweep_artifacts(results_dir))
    results = [
        _safe_gate("Case physics validation ledger", summarize_case_validation),
        _safe_gate("Case evidence tiers", summarize_case_evidence_tiers),
        _safe_gate("Case experiment readiness", summarize_case_readiness),
        _safe_gate("Smoke pipeline", lambda: summarize_smoke_pipeline(results_root)),
        sweep_gate,
        _safe_gate(
            "Server return package",
            lambda: summarize_server_return_package(
                sweep_gate,
                results_root=results_root,
                results_dir=results_dir,
                paper_dir=paper_dir,
                logs_dir=logs_dir,
            ),
        ),
        _safe_gate(
            "Manuscript stage",
            lambda: summarize_manuscript_stage(
                paper_dir,
                results_dir=results_dir,
                sweep_gate=sweep_gate,
            ),
        ),
    ]
    if not args.no_submission_gate:
        if sweep_gate.status == "PASS":
            results.append(
                _safe_gate(
                    "Final submission gate",
                    lambda: summarize_submission_gate(paper_dir, results_dir),
                )
            )
        else:
            results.append(defer_submission_gate(sweep_gate))
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.launch_preflight:
        results = _launch_preflight_results(args)
        title = "Publication Launch Preflight"
        output = (args.output or DEFAULT_LAUNCH_OUTPUT).resolve()
    else:
        results = _publication_readiness_results(args)
        title = "Publication Readiness Audit"
        output = (args.output or DEFAULT_OUTPUT).resolve()
    report = render_report(results, title=title)
    write_report_atomic(report, output)
    print(f"[audit] wrote {output}")
    for item in results:
        print(f"[audit] {item.status:7s} {item.name}")
    if args.launch_preflight:
        return 0 if all(item.status == "PASS" for item in results) else 1
    return 1 if any(item.status == "BLOCKED" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
