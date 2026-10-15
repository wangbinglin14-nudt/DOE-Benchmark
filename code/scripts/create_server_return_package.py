#!/usr/bin/env python
"""Create the server-return tarball after the formal full pipeline completes."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.project_layout import project_file
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FILES = (
    Path(".zenodo.json"),
    Path("CITATION.cff"),
    Path("codemeta.json"),
    Path("croissant-metadata.json"),
    Path("ro-crate-metadata.json"),
    Path("benchmark/results/full_benchmark/results.csv"),
    Path("benchmark/results/full_benchmark/sweep_integrity_report.md"),
    Path("benchmark/results/full_benchmark/runtime_health_report.txt"),
    Path("benchmark/results/full_benchmark/run_config.json"),
    Path("benchmark/results/full_benchmark/RUN_MANIFEST.json"),
    Path("benchmark/results/full_benchmark/python_environment_pip_freeze.txt"),
    Path("benchmark/results/full_benchmark/analysis/test_set_sufficiency.csv"),
    Path("docs/case_validation_status.csv"),
    Path("docs/case_validation_sources.csv"),
    Path("docs/case_validation_note_sources.csv"),
    Path("docs/case_experiment_readiness.csv"),
    Path("docs/CASE_VALIDATION_SUMMARY.md"),
    Path("docs/CASE_WEB_VALIDATION_AUDIT.md"),
    Path("docs/EXPERIMENT_BENCHMARK_AUDIT.md"),
    Path("docs/PUBLICATION_LAUNCH_PREFLIGHT.md"),
    Path("docs/STATISTICAL_ANALYSIS_PLAN.md"),
    Path("paper/numbers.json"),
    Path("paper/numbers.tex"),
    Path("paper/figure_manifest.json"),
    Path("paper/figdata/figdata_manifest.json"),
    Path("logs/pipeline_command_history.txt"),
)

REQUIRED_DIRS = (
    Path("benchmark/results/ranking_reversal"),
    Path("benchmark/results/sample_complexity_0.20"),
    Path("benchmark/results/reliability_recovery"),
    Path("paper/figdata"),
    Path("paper/figures"),
)

SUPPORTING_FILES = (
    Path("docs/SERVER_RUN_HANDOFF.md"),
    Path("docs/REPRODUCIBILITY.md"),
    Path("docs/RESEARCH_REUSE.md"),
    Path("paper/preregistration.md"),
    Path("paper/preregistration_decisions.md"),
)
SWEEP_PASS_VERDICT = "PASS: sweep satisfies the declared integrity gate."
RUNTIME_HEALTH_PASS_VERDICT = "[result] PASS - runtime sweep health is acceptable."
RUNTIME_HEALTH_RUN_CONFIG_VERDICT = "run_config.json contract matched partial results"
RUNTIME_HEALTH_RUNTIME_CONTROLS_VERDICT = (
    "run_config.json runtime controls recorded and valid"
)
RUNTIME_HEALTH_ENVIRONMENT_VERDICT = (
    "run_config.json runtime environment recorded and valid"
)
RUNTIME_HEALTH_TORCH_REPRO_VERDICT = (
    "run_config.json torch reproducibility policy recorded and valid"
)
RUNTIME_HEALTH_CURRENT_ENVIRONMENT_VERDICT = (
    "current runtime environment matches run_config.json snapshot"
)
RUNTIME_HEALTH_FAILURE_CATEGORY_VERDICT = (
    "runtime failure-category telemetry summarized"
)
RUNTIME_HEALTH_RESOURCE_VERDICT = "runtime resource telemetry summarized"
RUNTIME_HEALTH_WALL_TIME_VERDICT = "runtime wall-time telemetry summarized"
RUNTIME_HEALTH_RESULTS_RELATIVE_PATH = "benchmark/results/full_benchmark/results.csv"
RUNTIME_HEALTH_ARTIFACT_RE = re.compile(
    r"^\[artifact\]\s+results_csv=(?P<path>.*?)\s+"
    r"size_bytes=(?P<size>[0-9]+)\s+sha256=(?P<sha256>[0-9a-f]{64})\s*$",
    re.MULTILINE,
)
RUNTIME_HEALTH_RUN_CONFIG_ARTIFACT_RE = re.compile(
    r"^\[artifact\]\s+run_config_json=(?P<path>.*?)\s+"
    r"size_bytes=(?P<size>[0-9]+)\s+sha256=(?P<sha256>[0-9a-f]{64})\s*$",
    re.MULTILINE,
)


def _default_output(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return root / "benchmark" / "results" / f"server_return_package_{stamp}.tar.gz"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_reported_results_csv_path(path_text: str) -> str:
    raw = str(path_text).strip()
    if not raw:
        return ""
    windows = PureWindowsPath(raw)
    if windows.drive or windows.root == "\\":
        return PurePosixPath(*windows.parts).as_posix()
    return PurePosixPath(raw.replace("\\", "/")).as_posix()


def _runtime_health_binding_errors(root: Path) -> list[str]:
    """Ensure runtime_health_report.txt fingerprints this run configuration."""
    results_csv = root / "benchmark/results/full_benchmark/results.csv"
    run_config_json = root / "benchmark/results/full_benchmark/run_config.json"
    report = root / "benchmark/results/full_benchmark/runtime_health_report.txt"
    if not results_csv.is_file() or results_csv.is_symlink():
        return []
    if not report.is_file() or report.is_symlink():
        return []
    text = report.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []
    if RUNTIME_HEALTH_RUN_CONFIG_VERDICT not in text:
        errors.append(
            "required runtime_health_report.txt does not prove that "
            "results.csv was checked against run_config.json; rerun "
            "`scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` before packaging"
        )
    if RUNTIME_HEALTH_RUNTIME_CONTROLS_VERDICT not in text:
        errors.append(
            "required runtime_health_report.txt does not prove that "
            "run_config.json runtime controls were recorded and validated; "
            "rerun `scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before packaging"
        )
    if RUNTIME_HEALTH_ENVIRONMENT_VERDICT not in text:
        errors.append(
            "required runtime_health_report.txt does not prove that "
            "run_config.json runtime environment was recorded and validated; "
            "rerun `scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before packaging"
        )
    if RUNTIME_HEALTH_TORCH_REPRO_VERDICT not in text:
        errors.append(
            "required runtime_health_report.txt does not prove that "
            "run_config.json torch reproducibility policy was recorded and "
            "validated; rerun `scripts/check_sweep_runtime_health.py "
            "--run-config-json benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before packaging"
        )
    if RUNTIME_HEALTH_CURRENT_ENVIRONMENT_VERDICT not in text:
        errors.append(
            "required runtime_health_report.txt does not prove that the "
            "current packaging environment matched the run_config.json startup "
            "snapshot; rerun `scripts/check_sweep_runtime_health.py "
            "--run-config-json benchmark/results/full_benchmark/run_config.json "
            "--require-run-config --validate-current-environment` on the same "
            "server environment before packaging"
        )
    if RUNTIME_HEALTH_FAILURE_CATEGORY_VERDICT not in text:
        errors.append(
            "required runtime_health_report.txt does not summarize "
            "failure-category telemetry; rerun "
            "`scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before packaging"
        )
    if RUNTIME_HEALTH_RESOURCE_VERDICT not in text:
        errors.append(
            "required runtime_health_report.txt does not summarize runtime "
            "resource telemetry; rerun "
            "`scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before packaging"
        )
    if RUNTIME_HEALTH_WALL_TIME_VERDICT not in text:
        errors.append(
            "required runtime_health_report.txt does not summarize runtime "
            "wall-time telemetry; rerun "
            "`scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before packaging"
        )
    match = RUNTIME_HEALTH_ARTIFACT_RE.search(text)
    if match is None:
        errors.append(
            "required runtime_health_report.txt does not record a results.csv "
            "size_bytes/sha256 artifact fingerprint"
        )
        return errors
    observed_path = str(match.group("path")).strip()
    normalized_observed_path = _normalize_reported_results_csv_path(observed_path)
    if not normalized_observed_path.endswith(RUNTIME_HEALTH_RESULTS_RELATIVE_PATH):
        errors.append(
            "required runtime_health_report.txt results.csv path does not "
            "point at the canonical headline archive location: "
            f"report={observed_path}, expected_suffix={RUNTIME_HEALTH_RESULTS_RELATIVE_PATH}"
        )
    try:
        observed_resolved = Path(observed_path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(
            "required runtime_health_report.txt results.csv path could not be "
            f"resolved: report={observed_path}, error={type(exc).__name__}: {exc}"
        )
    else:
        if observed_resolved != results_csv.resolve():
            errors.append(
                "required runtime_health_report.txt results.csv path does not "
                "match current results.csv: "
                f"report={observed_resolved}, actual={results_csv.resolve()}"
            )
    expected_size = results_csv.stat().st_size
    expected_sha = _sha256_file(results_csv)
    observed_size = int(match.group("size"))
    observed_sha = str(match.group("sha256"))
    if observed_size != expected_size:
        errors.append(
            "required runtime_health_report.txt results.csv size_bytes does "
            f"not match current results.csv: report={observed_size}, "
            f"actual={expected_size}"
        )
    if observed_sha != expected_sha:
        errors.append(
            "required runtime_health_report.txt results.csv sha256 does not "
            f"match current results.csv: report={observed_sha}, actual={expected_sha}"
        )

    if not run_config_json.is_file() or run_config_json.is_symlink():
        return errors
    run_config_match = RUNTIME_HEALTH_RUN_CONFIG_ARTIFACT_RE.search(text)
    if run_config_match is None:
        errors.append(
            "required runtime_health_report.txt does not record a run_config.json "
            "size_bytes/sha256 artifact fingerprint"
        )
        return errors
    observed_config_path = str(run_config_match.group("path")).strip()
    normalized_observed_config_path = _normalize_reported_results_csv_path(
        observed_config_path
    )
    expected_config_suffix = "benchmark/results/full_benchmark/run_config.json"
    if not normalized_observed_config_path.endswith(expected_config_suffix):
        errors.append(
            "required runtime_health_report.txt run_config.json path does not "
            "point at the canonical headline startup snapshot location: "
            f"report={observed_config_path}, expected_suffix={expected_config_suffix}"
        )
    try:
        observed_config_resolved = Path(observed_config_path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(
            "required runtime_health_report.txt run_config.json path could not be "
            f"resolved: report={observed_config_path}, error={type(exc).__name__}: {exc}"
        )
    else:
        if observed_config_resolved != run_config_json.resolve():
            errors.append(
                "required runtime_health_report.txt run_config.json path does not "
                "match current run_config.json: "
                f"report={observed_config_resolved}, actual={run_config_json.resolve()}"
            )
    expected_config_size = run_config_json.stat().st_size
    expected_config_sha = _sha256_file(run_config_json)
    observed_config_size = int(run_config_match.group("size"))
    observed_config_sha = str(run_config_match.group("sha256"))
    if observed_config_size != expected_config_size:
        errors.append(
            "required runtime_health_report.txt run_config.json size_bytes does "
            f"not match current run_config.json: report={observed_config_size}, "
            f"actual={expected_config_size}"
        )
    if observed_config_sha != expected_config_sha:
        errors.append(
            "required runtime_health_report.txt run_config.json sha256 does not "
            f"match current run_config.json: report={observed_config_sha}, "
            f"actual={expected_config_sha}"
        )
    return errors


def _directory_has_content(path: Path) -> bool:
    return any(path.iterdir())


def _unsafe_package_member_errors(root: Path) -> list[str]:
    """Return paths that would create unsafe tar members if packaged."""
    errors: list[str] = []
    candidate_paths: list[Path] = []
    candidate_paths.extend(project_file(root, rel_path) for rel_path in REQUIRED_FILES)
    for rel_path in REQUIRED_DIRS:
        path = project_file(root, rel_path)
        candidate_paths.append(path)
        if path.is_dir():
            candidate_paths.extend(path.rglob("*"))
    for rel_path in SUPPORTING_FILES:
        path = project_file(root, rel_path)
        if path.exists() or path.is_symlink():
            candidate_paths.append(path)

    for path in candidate_paths:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.as_posix()
        if path.is_symlink():
            errors.append(
                f"package member is a symlink and cannot be safely archived: {rel}"
            )
        elif path.exists() and not (path.is_file() or path.is_dir()):
            errors.append(
                "package member is not a regular file or directory and cannot "
                f"be safely archived: {rel}"
            )
    return errors


def _unsafe_output_path_errors(root: Path, output: Path) -> list[str]:
    """Return output paths that would overwrite or self-include artifacts."""
    errors: list[str] = []
    if not str(output.name).endswith(".tar.gz"):
        errors.append(
            "return package output filename must end with .tar.gz: "
            f"{output}"
        )
    if output.is_symlink():
        errors.append(f"return package output is a symlink and cannot be trusted: {output}")
    for parent in (output.parent, *output.parent.parents):
        if parent.is_symlink():
            errors.append(
                "return package output parent directory is a symlink and "
                f"cannot be trusted: {parent}"
            )
    packaged_files = set(REQUIRED_FILES)
    packaged_files.update(SUPPORTING_FILES)
    for rel_path in packaged_files:
        if output == (project_file(root, rel_path)).resolve():
            errors.append(
                "return package output would overwrite packaged artifact: "
                f"{rel_path.as_posix()}"
            )

    for rel_dir in REQUIRED_DIRS:
        package_dir = (root / rel_dir).resolve()
        try:
            output.relative_to(package_dir)
        except ValueError:
            continue
        errors.append(
            "return package output must not be written inside packaged "
            f"directory: {rel_dir.as_posix()}"
        )
    return errors


def check_required_artifacts(root: Path) -> list[str]:
    """Return missing/empty required artifact errors."""
    errors: list[str] = []
    for rel_path in REQUIRED_FILES:
        path = project_file(root, rel_path)
        if path.is_symlink():
            errors.append(f"required file is a symlink: {rel_path.as_posix()}")
        elif not path.exists():
            errors.append(f"required file missing: {rel_path.as_posix()}")
        elif not path.is_file():
            errors.append(f"required path is not a file: {rel_path.as_posix()}")
        elif path.stat().st_size == 0:
            errors.append(f"required file is empty: {rel_path.as_posix()}")
        elif (
            rel_path == Path("benchmark/results/full_benchmark/sweep_integrity_report.md")
            and SWEEP_PASS_VERDICT
            not in path.read_text(encoding="utf-8", errors="replace")
        ):
            errors.append(
                "required sweep_integrity_report.md does not contain the "
                "declared PASS verdict; rerun `sweep_audit` before packaging"
            )
        elif (
            rel_path == Path("benchmark/results/full_benchmark/runtime_health_report.txt")
            and RUNTIME_HEALTH_PASS_VERDICT
            not in path.read_text(encoding="utf-8", errors="replace")
        ):
            errors.append(
                "required runtime_health_report.txt does not contain the "
                "runtime health PASS verdict; rerun "
                "`scripts/check_sweep_runtime_health.py` before packaging"
            )
        elif rel_path == Path("benchmark/results/full_benchmark/runtime_health_report.txt"):
            errors.extend(_runtime_health_binding_errors(root))

    try:
        from scripts.check_server_return_package import check_machine_readable_metadata
    except Exception as exc:
        errors.append(
            "could not import machine-readable metadata package gate: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        errors.extend(check_machine_readable_metadata(root))

    run_config_path = root / "benchmark/results/full_benchmark/run_config.json"
    pip_freeze_path = root / "benchmark/results/full_benchmark/python_environment_pip_freeze.txt"
    manifest_path = root / "benchmark/results/full_benchmark/RUN_MANIFEST.json"
    if run_config_path.exists() and pip_freeze_path.exists():
        try:
            from scripts.check_submission_ready import (
                check_manifest_temp_storage_matches_run_config,
                check_python_environment_lock_matches_run_config,
            )
        except Exception as exc:
            errors.append(
                "could not import manifest/run-config consistency gate: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            errors.extend(
                check_python_environment_lock_matches_run_config(
                    run_config_path,
                    pip_freeze_path,
                )
            )
            if manifest_path.exists():
                errors.extend(
                    check_manifest_temp_storage_matches_run_config(
                        manifest_path,
                        run_config_path,
                    )
                )

    for rel_path in REQUIRED_DIRS:
        path = project_file(root, rel_path)
        if path.is_symlink():
            errors.append(f"required directory is a symlink: {rel_path.as_posix()}")
        elif not path.exists():
            errors.append(f"required directory missing: {rel_path.as_posix()}")
        elif not path.is_dir():
            errors.append(f"required path is not a directory: {rel_path.as_posix()}")
        elif not _directory_has_content(path):
            errors.append(f"required directory is empty: {rel_path.as_posix()}")
    if not errors:
        try:
            from scripts.check_server_return_package import (
                check_case_web_validation_audit_content,
            )
        except Exception as exc:
            errors.append(
                "could not import case web-validation audit package gate: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            errors.extend(
                f"case web-validation audit content: {error}"
                for error in check_case_web_validation_audit_content(
                    docs_dir=root / "docs"
                )
            )
    errors.extend(_unsafe_package_member_errors(root))
    return errors


def _iter_package_paths(root: Path) -> list[Path]:
    paths = set(REQUIRED_FILES)
    paths.update(REQUIRED_DIRS)
    for rel_path in SUPPORTING_FILES:
        path = project_file(root, rel_path)
        if not path.is_symlink() and path.is_file() and path.stat().st_size > 0:
            paths.add(rel_path)
    return sorted(paths, key=lambda path: path.as_posix())


def _add_member(
    tar: tarfile.TarFile,
    root: Path,
    rel_path: Path,
    *,
    seen: set[str],
) -> None:
    arcname = rel_path.as_posix()
    if arcname in seen:
        return
    tar.add(project_file(root, rel_path), arcname=arcname, recursive=False)
    seen.add(arcname)


def _add_path(tar: tarfile.TarFile, root: Path, rel_path: Path, *, seen: set[str]) -> None:
    path = project_file(root, rel_path)
    if path.is_dir():
        _add_member(tar, root, rel_path, seen=seen)
        children = sorted(path.rglob("*"), key=lambda child: child.relative_to(root).as_posix())
        for child in children:
            _add_member(tar, root, child.relative_to(root), seen=seen)
    else:
        _add_member(tar, root, rel_path, seen=seen)


def create_package(root: Path, output: Path, *, overwrite: bool = False) -> Path:
    """Create a tar.gz return package and sha256 sidecar."""
    root = root.resolve()
    raw_output = output
    output_errors = _unsafe_output_path_errors(root, raw_output)
    output = raw_output.resolve()
    sidecar = raw_output.with_name(f"{raw_output.name}.sha256")
    if sidecar.is_symlink():
        output_errors.append(
            f"return package sha256 sidecar is a symlink and cannot be trusted: {sidecar}"
        )
    output_errors.extend(_unsafe_output_path_errors(root, output))
    output_errors = list(dict.fromkeys(output_errors))
    if output_errors:
        for error in output_errors:
            print(f"[error] {error}", file=sys.stderr)
        print(
            f"[result] FAIL - unsafe return package output path; "
            f"{len(output_errors)} issue(s).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    errors = check_required_artifacts(root)
    if errors:
        for error in errors:
            print(f"[error] {error}", file=sys.stderr)
        print(
            f"[result] FAIL - cannot create return package; {len(errors)} "
            "required artifact issue(s).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    env_overwrite = os.environ.get("OVERWRITE_RETURN_PACKAGE") == "1"
    if output.exists() and not (overwrite or env_overwrite):
        print(f"[error] output already exists: {output}", file=sys.stderr)
        print("        Pass --overwrite or set OVERWRITE_RETURN_PACKAGE=1 to replace it.", file=sys.stderr)
        raise SystemExit(2)
    sidecar = output.with_name(f"{output.name}.sha256")
    if sidecar.is_symlink():
        print(
            f"[error] return package sha256 sidecar is a symlink and cannot be trusted: {sidecar}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if sidecar.exists() and not (overwrite or env_overwrite):
        print(f"[error] sha256 sidecar already exists: {sidecar}", file=sys.stderr)
        print("        Pass --overwrite or set OVERWRITE_RETURN_PACKAGE=1 to replace it.", file=sys.stderr)
        raise SystemExit(2)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=str(output.parent)))
    tmp_output = tmp_dir / output.name
    tmp_sidecar = tmp_dir / f"{output.name}.sha256"
    try:
        with tarfile.open(tmp_output, "w:gz") as tar:
            seen: set[str] = set()
            for rel_path in _iter_package_paths(root):
                _add_path(tar, root, rel_path, seen=seen)

        digest = _sha256_file(tmp_output)
        tmp_sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
        from scripts.check_server_return_package import check_return_tarball

        validation_errors = check_return_tarball(tmp_output, sha256_path=tmp_sidecar)
        if validation_errors:
            for error in validation_errors:
                print(f"[error] {error}", file=sys.stderr)
            print(
                f"[result] FAIL - wrote an invalid return package; {len(validation_errors)} "
                "tarball issue(s) found.  Removing the invalid package.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        os.replace(tmp_output, output)
        os.replace(tmp_sidecar, sidecar)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create benchmark/results/server_return_package_<UTC>.tar.gz after "
            "the formal full pipeline has completed."
        )
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Optional tar.gz output path. Defaults under benchmark/results/.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root containing benchmark/, paper/, docs/, and logs/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output tarball.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output = args.output or _default_output(root)
    package = create_package(root, output, overwrite=args.overwrite)
    print("[result] PASS - server return package written:")
    print(f"  {package}")
    sidecar = package.with_name(f"{package.name}.sha256")
    if sidecar.is_file():
        print(f"  {sidecar}")
    print()
    print("After transfer to the review workstation, extract at the repo root and run:")
    print("  python scripts/check_server_return_package.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
