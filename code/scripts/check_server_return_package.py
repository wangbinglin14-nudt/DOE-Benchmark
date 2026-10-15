"""Validate the server-returned DOE-Benchmark result package.

This is the first local check to run after the full-sweep archive is copied
back from the server.  It intentionally stops short of manuscript sign-off:
``check_submission_ready.py`` still owns placeholder scans, preregistration
decisions, and final paper-source checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shlex
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.project_layout import MANUSCRIPTS_DIR  # noqa: E402

RESULTS_ROOT_DEFAULT = ROOT / "benchmark" / "results"
FULL_SWEEP_DEFAULT = RESULTS_ROOT_DEFAULT / "full_benchmark"
PAPER_DIR_DEFAULT = MANUSCRIPTS_DIR
DOCS_DIR_DEFAULT = ROOT / "docs"
LOGS_DIR_DEFAULT = ROOT / "logs"


CORE_FULL_SWEEP_FILES = (
    "full-sweep results",
    "full-sweep integrity report",
    "full-sweep run configuration",
    "full-sweep run manifest",
)
DISALLOWED_PACKAGE_PREFIXES = (
    "benchmark/results/sota_robustness",
    "benchmark/results/mf_supplementary",
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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CROISSANT_REQUIRED_HEADLINE_FIELDS: dict[str, str] = {
    "case_id": "sc:Text",
    "case_source_path": "sc:Text",
    "case_source_sha256": "sc:Text",
    "domain": "sc:Text",
    "dim": "sc:Integer",
    "n_output": "sc:Integer",
    "method": "sc:Text",
    "method_family": "sc:Text",
    "method_source_path": "sc:Text",
    "method_source_sha256": "sc:Text",
    "budget_mult": "sc:Integer",
    "n_train": "sc:Integer",
    "n_test": "sc:Integer",
    "repeat": "sc:Integer",
    "seed": "sc:Integer",
    "test_seed": "sc:Integer",
    "seed_strategy": "sc:Text",
    "seed_start": "sc:Integer",
    "result_schema_version": "sc:Text",
    "protocol_digest": "sc:Text",
    "status": "sc:Text",
    "failure_category": "sc:Text",
    "error": "sc:Text",
    "nrmse": "sc:Float",
    "rmse": "sc:Float",
    "mae": "sc:Float",
    "maxae": "sc:Float",
    "q95ae": "sc:Float",
    "r2": "sc:Float",
    "t_total_s": "sc:Float",
    "worker_host": "sc:Text",
    "worker_pid": "sc:Integer",
    "cell_started_utc": "sc:DateTime",
    "cell_finished_utc": "sc:DateTime",
    "rss_gb_start": "sc:Float",
    "rss_gb_end": "sc:Float",
    "rss_gb_peak": "sc:Float",
}


def _filesystem_error(path: Path, exc: OSError) -> str:
    target = getattr(exc, "filename", None) or getattr(exc, "filename2", None) or str(path)
    return (
        f"permission/access error while inspecting {target}: "
        f"{type(exc).__name__}: {exc}. Fix the file/directory ACL or rebuild "
        "the returned artifact package, then rerun this check."
    )


def _expected_package_paths() -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Load the server packaging contract without duplicating path lists."""
    from scripts.create_server_return_package import REQUIRED_DIRS, REQUIRED_FILES

    return tuple(REQUIRED_FILES), tuple(REQUIRED_DIRS)


def _parse_sha256_sidecar(
    path: Path,
    *,
    expected_filename: str,
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    if path.is_symlink():
        return None, [f"sha256 sidecar is a symlink and cannot be trusted: {path}"]
    if not path.is_file() or path.stat().st_size == 0:
        return None, [f"sha256 sidecar missing or empty: {path}"]
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    parts = text.split()
    if len(parts) != 2 or len(parts[0]) != 64:
        return None, [f"sha256 sidecar is malformed: {path}"]
    digest = parts[0].lower()
    if any(ch not in "0123456789abcdef" for ch in digest):
        errors.append(f"sha256 sidecar digest is not hexadecimal: {path}")
        return None, errors
    observed_filename = Path(parts[1]).name
    if observed_filename != expected_filename:
        errors.append(
            "sha256 sidecar filename does not match package: "
            f"expected {expected_filename}, observed {parts[1]}"
        )
    return digest, errors


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_metadata(label: str, text: str) -> tuple[dict[str, object] | None, list[str]]:
    try:
        payload = json.loads(text)
    except Exception as exc:
        return None, [f"{label} is not valid JSON: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"{label} JSON root must be an object"]
    return payload, []


def _non_empty_string(payload: dict[str, object], key: str) -> bool:
    return isinstance(payload.get(key), str) and bool(str(payload.get(key)).strip())


# A repository/dataset link may legitimately be unresolved while a manuscript is
# under review: the deposit is published only on acceptance.  The readiness gate
# must not push authors towards inventing a URL to satisfy it, so these fields
# accept either a real link or an explicit statement of the current state.
PENDING_PUBLICATION_MARKERS = (
    "not yet public",
    "not yet published",
    "on acceptance",
)


def _publication_link_ok(payload: dict[str, object], key: str) -> bool:
    """True for a resolved http(s) link, or an explicit pending-publication note."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if re.match(r"https?://\S+$", text):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in PENDING_PUBLICATION_MARKERS)


def _machine_readable_metadata_text_errors(
    *,
    zenodo_text: str,
    codemeta_text: str,
    croissant_text: str,
    ro_crate_text: str,
) -> list[str]:
    """Validate root FAIR/Zenodo/CodeMeta/Croissant/RO-Crate metadata contracts."""
    errors: list[str] = []

    zenodo, json_errors = _load_json_metadata(".zenodo.json", zenodo_text)
    errors.extend(json_errors)
    if zenodo is not None:
        for key in ("title", "description", "version", "license", "upload_type"):
            if not _non_empty_string(zenodo, key):
                errors.append(f".zenodo.json missing non-empty {key}")
        creators = zenodo.get("creators")
        if not isinstance(creators, list) or not creators:
            errors.append(".zenodo.json creators must be a non-empty list")
        elif not all(
            isinstance(item, dict) and _non_empty_string(item, "name")
            for item in creators
        ):
            errors.append(".zenodo.json creators entries must include non-empty name")

    codemeta, json_errors = _load_json_metadata("codemeta.json", codemeta_text)
    errors.extend(json_errors)
    if codemeta is not None:
        context = codemeta.get("@context")
        if "codemeta" not in str(context).lower():
            errors.append("codemeta.json @context must reference CodeMeta")
        for key in ("@type", "name", "description", "version", "license"):
            if not _non_empty_string(codemeta, key):
                errors.append(f"codemeta.json missing non-empty {key}")
        if not _publication_link_ok(codemeta, "codeRepository"):
            errors.append(
                "codemeta.json codeRepository must be an https link or an "
                "explicit pending-publication statement"
            )
        authors = codemeta.get("author")
        if not isinstance(authors, list) or not authors:
            errors.append("codemeta.json author must be a non-empty list")

    croissant, json_errors = _load_json_metadata("croissant-metadata.json", croissant_text)
    errors.extend(json_errors)
    if croissant is not None:
        context = croissant.get("@context")
        if "croissant" not in str(context).lower() and "mlcommons" not in str(context).lower():
            errors.append("croissant-metadata.json @context must reference Croissant")
        if str(croissant.get("@type")) != "sc:Dataset":
            errors.append("croissant-metadata.json @type must be 'sc:Dataset'")
        for key in ("name", "description", "license", "version", "conformsTo"):
            if not _non_empty_string(croissant, key):
                errors.append(f"croissant-metadata.json missing non-empty {key}")
        if not _publication_link_ok(croissant, "url"):
            errors.append(
                "croissant-metadata.json url must be an https link or an "
                "explicit pending-publication statement"
            )
        if "croissant/1.1" not in str(croissant.get("conformsTo", "")).lower():
            errors.append("croissant-metadata.json conformsTo must reference Croissant 1.1")

        distribution = croissant.get("distribution")
        if not isinstance(distribution, list) or not distribution:
            errors.append("croissant-metadata.json distribution must be a non-empty list")
        else:
            content_urls = {
                str(item.get("contentUrl"))
                for item in distribution
                if isinstance(item, dict)
            }
            for required_url in (
                "benchmark/results/full_benchmark/results.csv",
                "benchmark/results/full_benchmark/RUN_MANIFEST.json",
                "paper/figdata/figdata_manifest.json",
            ):
                if required_url not in content_urls:
                    errors.append(
                        "croissant-metadata.json distribution missing "
                        f"{required_url}"
                    )
            for item in distribution:
                if not isinstance(item, dict):
                    errors.append("croissant-metadata.json distribution entries must be objects")
                    continue
                if str(item.get("@type")) != "cr:FileObject":
                    errors.append(
                        "croissant-metadata.json distribution entry @type must be 'cr:FileObject'"
                    )
                for key in ("@id", "name", "contentUrl", "encodingFormat", "description"):
                    if not _non_empty_string(item, key):
                        errors.append(
                            f"croissant-metadata.json distribution entry missing non-empty {key}"
                        )

        record_sets = croissant.get("recordSet")
        if not isinstance(record_sets, list) or not record_sets:
            errors.append("croissant-metadata.json recordSet must be a non-empty list")
        else:
            field_names: set[str] = set()
            field_types: dict[str, str] = {}
            for item in record_sets:
                if not isinstance(item, dict):
                    errors.append("croissant-metadata.json recordSet entries must be objects")
                    continue
                if str(item.get("@type")) != "cr:RecordSet":
                    errors.append(
                        "croissant-metadata.json recordSet entry @type must be 'cr:RecordSet'"
                    )
                for key in ("@id", "name", "description"):
                    if not _non_empty_string(item, key):
                        errors.append(
                            f"croissant-metadata.json recordSet entry missing non-empty {key}"
                        )
                fields = item.get("field")
                if not isinstance(fields, list) or not fields:
                    errors.append("croissant-metadata.json recordSet field must be non-empty")
                    continue
                for field in fields:
                    if not isinstance(field, dict):
                        errors.append("croissant-metadata.json field entries must be objects")
                        continue
                    name = field.get("name")
                    if isinstance(name, str):
                        field_names.add(name)
                        data_type = field.get("dataType")
                        if isinstance(data_type, str) and data_type.strip():
                            field_types[name] = data_type.strip()
                    if not _non_empty_string(field, "@id"):
                        errors.append("croissant-metadata.json field missing non-empty @id")
                    if not _non_empty_string(field, "dataType"):
                        errors.append("croissant-metadata.json field missing non-empty dataType")
                    source = field.get("source")
                    if not isinstance(source, dict):
                        errors.append("croissant-metadata.json field source must be an object")
                    else:
                        file_object = source.get("fileObject")
                        extract = source.get("extract")
                        if (
                            not isinstance(file_object, dict)
                            or file_object.get("@id") != "headline_results_csv"
                        ):
                            errors.append(
                                "croissant-metadata.json field source.fileObject "
                                "must reference headline_results_csv"
                            )
                        if not isinstance(extract, dict) or extract.get("column") != name:
                            errors.append(
                                "croissant-metadata.json field source.extract.column "
                                "must match field name"
                            )
            for required_field, expected_type in CROISSANT_REQUIRED_HEADLINE_FIELDS.items():
                if required_field not in field_names:
                    errors.append(
                        "croissant-metadata.json headline_results recordSet missing field "
                        f"{required_field}"
                    )
                    continue
                observed_type = field_types.get(required_field)
                if observed_type != expected_type:
                    errors.append(
                        "croissant-metadata.json headline_results field "
                        f"{required_field} dataType must be {expected_type}"
                    )

        for key in (
            "rai:dataLimitations",
            "rai:dataBiases",
            "rai:personalSensitiveInformation",
            "rai:dataUseCases",
            "rai:dataSocialImpact",
        ):
            if not _non_empty_string(croissant, key):
                errors.append(f"croissant-metadata.json missing non-empty {key}")
        if not isinstance(croissant.get("rai:hasSyntheticData"), bool):
            errors.append("croissant-metadata.json rai:hasSyntheticData must be boolean")

    ro_crate, json_errors = _load_json_metadata("ro-crate-metadata.json", ro_crate_text)
    errors.extend(json_errors)
    if ro_crate is not None:
        context = ro_crate.get("@context")
        if "ro/crate" not in str(context).lower():
            errors.append("ro-crate-metadata.json @context must reference RO-Crate")
        graph = ro_crate.get("@graph")
        if not isinstance(graph, list) or not graph:
            errors.append("ro-crate-metadata.json @graph must be a non-empty list")
        else:
            by_id = {
                item.get("@id"): item
                for item in graph
                if isinstance(item, dict) and isinstance(item.get("@id"), str)
            }
            if "ro-crate-metadata.json" not in by_id:
                errors.append("ro-crate-metadata.json missing metadata descriptor node")
            root = by_id.get("./")
            if not isinstance(root, dict):
                errors.append("ro-crate-metadata.json missing root Dataset node './'")
            else:
                has_part = root.get("hasPart")
                if not isinstance(has_part, list) or not has_part:
                    errors.append("ro-crate-metadata.json root Dataset hasPart must be non-empty")
                else:
                    part_ids = {
                        item.get("@id")
                        for item in has_part
                        if isinstance(item, dict) and isinstance(item.get("@id"), str)
                    }
                    for required in (
                        "CITATION.cff",
                        "codemeta.json",
                        "croissant-metadata.json",
                        "benchmark/results/full_benchmark/results.csv",
                        "benchmark/results/full_benchmark/RUN_MANIFEST.json",
                        "paper/figdata/figdata_manifest.json",
                    ):
                        if required not in part_ids:
                            errors.append(
                                "ro-crate-metadata.json root Dataset hasPart missing "
                                f"{required}"
                            )
    return errors


def check_machine_readable_metadata(return_root: Path) -> list[str]:
    """Check root metadata files are parseable and contain key FAIR fields."""
    paths = {
        "zenodo_text": return_root / ".zenodo.json",
        "codemeta_text": return_root / "codemeta.json",
        "croissant_text": return_root / "croissant-metadata.json",
        "ro_crate_text": return_root / "ro-crate-metadata.json",
    }
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        return []
    try:
        texts = {
            key: path.read_text(encoding="utf-8", errors="replace")
            for key, path in paths.items()
        }
    except OSError as exc:
        return [f"could not read machine-readable metadata: {type(exc).__name__}: {exc}"]
    return _machine_readable_metadata_text_errors(**texts)


def _normalize_reported_results_csv_path(path_text: str) -> str:
    raw = str(path_text).strip()
    if not raw:
        return ""
    windows = PureWindowsPath(raw)
    if windows.drive or windows.root == "\\":
        return PurePosixPath(*windows.parts).as_posix()
    return PurePosixPath(raw.replace("\\", "/")).as_posix()


def _runtime_health_binding_errors(
    report_text: str,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_run_config_size_bytes: int | None = None,
    expected_run_config_sha256: str | None = None,
    artifact_name: str,
) -> list[str]:
    """Validate that runtime-health output fingerprints this result/config pair."""
    errors: list[str] = []
    if RUNTIME_HEALTH_RUN_CONFIG_VERDICT not in report_text:
        errors.append(
            f"{artifact_name} does not prove that results.csv was checked "
            "against run_config.json; rerun "
            "`scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` before trusting the returned archive"
        )
    if RUNTIME_HEALTH_RUNTIME_CONTROLS_VERDICT not in report_text:
        errors.append(
            f"{artifact_name} does not prove that run_config.json runtime "
            "controls were recorded and validated; rerun "
            "`scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before "
            "trusting the returned archive"
        )
    if RUNTIME_HEALTH_ENVIRONMENT_VERDICT not in report_text:
        errors.append(
            f"{artifact_name} does not prove that run_config.json runtime "
            "environment was recorded and validated; rerun "
            "`scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before "
            "trusting the returned archive"
        )
    if RUNTIME_HEALTH_TORCH_REPRO_VERDICT not in report_text:
        errors.append(
            f"{artifact_name} does not prove that run_config.json torch "
            "reproducibility policy was recorded and validated; rerun "
            "`scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before "
            "trusting the returned archive"
        )
    if RUNTIME_HEALTH_CURRENT_ENVIRONMENT_VERDICT not in report_text:
        errors.append(
            f"{artifact_name} does not prove that the current packaging "
            "environment matched run_config.json; rerun "
            "`scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config --validate-current-environment` on the same "
            "server environment before trusting the returned archive"
        )
    if RUNTIME_HEALTH_FAILURE_CATEGORY_VERDICT not in report_text:
        errors.append(
            f"{artifact_name} does not summarize failure-category telemetry; "
            "rerun `scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before "
            "trusting the returned archive"
        )
    if RUNTIME_HEALTH_RESOURCE_VERDICT not in report_text:
        errors.append(
            f"{artifact_name} does not summarize runtime resource telemetry; "
            "rerun `scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before "
            "trusting the returned archive"
        )
    if RUNTIME_HEALTH_WALL_TIME_VERDICT not in report_text:
        errors.append(
            f"{artifact_name} does not summarize runtime wall-time telemetry; "
            "rerun `scripts/check_sweep_runtime_health.py --run-config-json "
            "benchmark/results/full_benchmark/run_config.json "
            "--require-run-config` with the current runtime-health gate before "
            "trusting the returned archive"
        )
    match = RUNTIME_HEALTH_ARTIFACT_RE.search(report_text)
    if match is None:
        errors.append(
            f"{artifact_name} does not record a results.csv size_bytes/sha256 "
            "artifact fingerprint"
        )
        return errors
    observed_path = str(match.group("path")).strip()
    normalized_observed_path = _normalize_reported_results_csv_path(observed_path)
    if not normalized_observed_path.endswith(RUNTIME_HEALTH_RESULTS_RELATIVE_PATH):
        errors.append(
            f"{artifact_name} results.csv path does not point at the canonical "
            "headline archive location: "
            f"report={observed_path}, expected_suffix={RUNTIME_HEALTH_RESULTS_RELATIVE_PATH}"
        )
    observed_size = int(match.group("size"))
    observed_sha = match.group("sha256")
    if observed_size != int(expected_size_bytes):
        errors.append(
            f"{artifact_name} results.csv size_bytes does not match the "
            f"returned results.csv: report={observed_size}, "
            f"actual={int(expected_size_bytes)}"
        )
    if observed_sha != expected_sha256:
        errors.append(
            f"{artifact_name} results.csv sha256 does not match the returned "
            f"results.csv: report={observed_sha}, actual={expected_sha256}"
        )
    if expected_run_config_size_bytes is None or expected_run_config_sha256 is None:
        return errors
    run_config_match = RUNTIME_HEALTH_RUN_CONFIG_ARTIFACT_RE.search(report_text)
    if run_config_match is None:
        errors.append(
            f"{artifact_name} does not record a run_config.json size_bytes/sha256 "
            "artifact fingerprint"
        )
        return errors
    observed_config_path = str(run_config_match.group("path")).strip()
    normalized_config_path = _normalize_reported_results_csv_path(observed_config_path)
    expected_suffix = "benchmark/results/full_benchmark/run_config.json"
    if not normalized_config_path.endswith(expected_suffix):
        errors.append(
            f"{artifact_name} run_config.json path does not point at the "
            "canonical headline startup snapshot location: "
            f"report={observed_config_path}, expected_suffix={expected_suffix}"
        )
    observed_config_size = int(run_config_match.group("size"))
    observed_config_sha = run_config_match.group("sha256")
    if observed_config_size != int(expected_run_config_size_bytes):
        errors.append(
            f"{artifact_name} run_config.json size_bytes does not match the "
            f"returned run_config.json: report={observed_config_size}, "
            f"actual={int(expected_run_config_size_bytes)}"
        )
    if observed_config_sha != expected_run_config_sha256:
        errors.append(
            f"{artifact_name} run_config.json sha256 does not match the "
            f"returned run_config.json: report={observed_config_sha}, "
            f"actual={expected_run_config_sha256}"
        )
    return errors


def _tar_text_member_contains(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    needle: str,
) -> bool:
    extracted = tar.extractfile(member)
    if extracted is None:
        return False
    with extracted:
        text = extracted.read().decode("utf-8", errors="replace")
    return needle in text


def _tar_member_bytes(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes | None:
    extracted = tar.extractfile(member)
    if extracted is None:
        return None
    with extracted:
        return extracted.read()


def _summary_from_results_csv_bytes(raw: bytes) -> dict[str, object]:
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    summary: dict[str, object] = {
        "exists": True,
        "n_rows": len(rows),
    }
    fieldnames = set(reader.fieldnames or [])
    if "case_id" in fieldnames:
        summary["n_cases"] = len({
            str(row.get("case_id", "")).strip()
            for row in rows
            if str(row.get("case_id", "")).strip()
        })
    if "method" in fieldnames:
        summary["n_methods"] = len({
            str(row.get("method", "")).strip()
            for row in rows
            if str(row.get("method", "")).strip()
        })
    for column, key in (("budget_mult", "budgets"), ("repeat", "repeats")):
        if column not in fieldnames:
            continue
        values: set[int] = set()
        for row in rows:
            raw_value = str(row.get(column, "")).strip()
            if not raw_value:
                continue
            try:
                values.add(int(float(raw_value)))
            except ValueError:
                continue
        summary[key] = sorted(values)
    for column, key in (
        ("status", "status_counts"),
        ("failure_category", "failure_category_counts"),
    ):
        if column not in fieldnames:
            continue
        counts: Counter[str] = Counter(
            str(row.get(column, "")).strip()
            for row in rows
        )
        summary[key] = {
            label: int(counts[label])
            for label in sorted(counts)
        }
    return summary


def _tar_figdata_manifest_summary_errors(
    manifest_bytes: bytes,
    results_bytes: bytes,
) -> list[str]:
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        return [
            "paper/figdata/figdata_manifest.json is not valid JSON: "
            f"{type(exc).__name__}: {exc}"
        ]
    if not isinstance(manifest, dict):
        return ["paper/figdata/figdata_manifest.json JSON root must be an object"]
    summary = manifest.get("results_summary")
    if not isinstance(summary, dict):
        return [
            "paper/figdata/figdata_manifest.json missing results_summary; "
            "regenerate figdata before packaging the server return archive"
        ]
    observed = _summary_from_results_csv_bytes(results_bytes)
    errors: list[str] = []
    for key, expected_value in observed.items():
        current_value = summary.get(key)
        if current_value != expected_value:
            errors.append(
                "paper/figdata/figdata_manifest.json results_summary "
                f"{key} mismatch against packaged results.csv: "
                f"expected {expected_value!r}, observed {current_value!r}"
            )
    return errors


def _json_object_from_bytes(label: str, raw: bytes) -> tuple[dict[str, object] | None, list[str]]:
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        return None, [f"{label} is not valid JSON: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"{label} JSON root must be an object"]
    return payload, []


def _single_results_provenance_sha(
    payload: dict[str, object],
    *,
    inputs_path: tuple[str, ...],
    context: str,
) -> tuple[str | None, list[str]]:
    current: object = payload
    for key in inputs_path:
        if not isinstance(current, dict):
            return None, [f"{context}: missing {'.'.join(inputs_path)} block"]
        current = current.get(key)
    if not isinstance(current, list):
        return None, [f"{context}: {'.'.join(inputs_path)} is missing or malformed"]
    matches = [
        item for item in current
        if isinstance(item, dict) and str(item.get("label", "")).strip() == "results_csv"
    ]
    if len(matches) != 1:
        return None, [
            f"{context}: expected exactly one provenance input labelled "
            f"'results_csv', observed {len(matches)}"
        ]
    sha = matches[0].get("sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha.strip().lower()):
        return None, [
            f"{context}: provenance input 'results_csv' lacks a valid sha256"
        ]
    return sha.strip().lower(), []


def _manifest_results_sha_from_payload(
    payload: dict[str, object],
) -> tuple[str | None, list[str]]:
    files = payload.get("artifacts", {})
    if isinstance(files, dict):
        files = files.get("files")
    if not isinstance(files, list):
        return None, ["RUN_MANIFEST.json artifacts.files is missing or malformed"]
    matches = [
        item for item in files
        if isinstance(item, dict)
        and str(item.get("path", "")).replace("\\", "/").split("/")[-1] == "results.csv"
    ]
    if len(matches) != 1:
        return None, [
            "RUN_MANIFEST.json must archive exactly one results.csv artifact, "
            f"observed {len(matches)}"
        ]
    sha = matches[0].get("sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha.strip().lower()):
        return None, ["RUN_MANIFEST.json results.csv artifact lacks a valid sha256"]
    return sha.strip().lower(), []


def _tar_paper_artifact_lineage_errors(
    *,
    results_sha256: str,
    run_manifest_bytes: bytes | None,
    numbers_bytes: bytes | None,
    figdata_manifest_bytes: bytes | None,
) -> list[str]:
    """Bind packaged paper artifacts to the packaged results.csv before extraction."""
    errors: list[str] = []
    manifest_sha: str | None = None
    if run_manifest_bytes is not None:
        manifest, manifest_errors = _json_object_from_bytes(
            "benchmark/results/full_benchmark/RUN_MANIFEST.json",
            run_manifest_bytes,
        )
        errors.extend(manifest_errors)
        if manifest is not None:
            manifest_sha, sha_errors = _manifest_results_sha_from_payload(manifest)
            errors.extend(sha_errors)
            if manifest_sha is not None and manifest_sha != results_sha256:
                errors.append(
                    "RUN_MANIFEST.json results.csv artifact sha256 does not "
                    "match packaged results.csv: "
                    f"manifest={manifest_sha}, packaged={results_sha256}"
                )

    if numbers_bytes is not None:
        numbers, numbers_errors = _json_object_from_bytes(
            "paper/numbers.json",
            numbers_bytes,
        )
        errors.extend(numbers_errors)
        if numbers is not None:
            numbers_sha, sha_errors = _single_results_provenance_sha(
                numbers,
                inputs_path=("_provenance", "inputs"),
                context="paper/numbers.json",
            )
            errors.extend(sha_errors)
            if numbers_sha is not None and numbers_sha != results_sha256:
                errors.append(
                    "paper/numbers.json results_csv provenance sha256 does "
                    "not match packaged results.csv: "
                    f"numbers={numbers_sha}, packaged={results_sha256}"
                )
            if (
                numbers_sha is not None
                and manifest_sha is not None
                and numbers_sha != manifest_sha
            ):
                errors.append(
                    "paper/numbers.json results_csv provenance sha256 does "
                    "not match RUN_MANIFEST archived results.csv: "
                    f"numbers={numbers_sha}, manifest={manifest_sha}"
                )

    if figdata_manifest_bytes is not None:
        figdata, figdata_errors = _json_object_from_bytes(
            "paper/figdata/figdata_manifest.json",
            figdata_manifest_bytes,
        )
        errors.extend(figdata_errors)
        if figdata is not None:
            figdata_sha, sha_errors = _single_results_provenance_sha(
                figdata,
                inputs_path=("inputs",),
                context="paper/figdata/figdata_manifest.json",
            )
            errors.extend(sha_errors)
            if figdata_sha is not None and figdata_sha != results_sha256:
                errors.append(
                    "paper/figdata/figdata_manifest.json results_csv "
                    "provenance sha256 does not match packaged results.csv: "
                    f"figdata={figdata_sha}, packaged={results_sha256}"
                )
            if (
                figdata_sha is not None
                and manifest_sha is not None
                and figdata_sha != manifest_sha
            ):
                errors.append(
                    "paper/figdata/figdata_manifest.json results_csv "
                    "provenance sha256 does not match RUN_MANIFEST archived "
                    f"results.csv: figdata={figdata_sha}, manifest={manifest_sha}"
                )
    return errors


def _tar_command_history_manifest_errors(
    *,
    run_manifest_bytes: bytes | None,
    command_history_bytes: bytes | None,
) -> list[str]:
    """Validate command-history provenance before trusting extracted files."""
    if run_manifest_bytes is None or command_history_bytes is None:
        return []
    manifest, manifest_errors = _json_object_from_bytes(
        "benchmark/results/full_benchmark/RUN_MANIFEST.json",
        run_manifest_bytes,
    )
    errors = list(manifest_errors)
    history_text = command_history_bytes.decode("utf-8", errors="replace")
    observed_entries, history_errors = _parse_command_history_entries(history_text)
    errors.extend(history_errors)
    if manifest is None or history_errors:
        return errors
    if not observed_entries:
        errors.append("pipeline_command_history.txt has no command entries")
        return errors
    errors.extend(_command_history_manifest_contract_errors(manifest, observed_entries))
    return errors


def check_return_tarball(
    package_path: Path,
    *,
    sha256_path: Path | None = None,
) -> list[str]:
    """Validate the transferred tarball before or after extraction."""
    errors: list[str] = []
    if package_path.is_symlink():
        return [f"server return package is a symlink and cannot be trusted: {package_path}"]
    if not package_path.is_file() or package_path.stat().st_size == 0:
        return [f"server return package missing or empty: {package_path}"]

    sidecar = sha256_path or package_path.with_name(f"{package_path.name}.sha256")
    expected_digest, sidecar_errors = _parse_sha256_sidecar(
        sidecar,
        expected_filename=package_path.name,
    )
    errors.extend(sidecar_errors)
    if expected_digest is not None:
        actual_digest = _sha256_file(package_path)
        if actual_digest != expected_digest:
            errors.append(
                f"server return package sha256 mismatch: expected {expected_digest}, "
                f"observed {actual_digest}"
            )

    try:
        with tarfile.open(package_path, "r:gz") as tar:
            members = tar.getmembers()
            member_by_name = {
                member.name.replace("\\", "/"): member
                for member in members
            }
            results_member = member_by_name.get(
                "benchmark/results/full_benchmark/results.csv"
            )
            results_bytes = (
                _tar_member_bytes(tar, results_member)
                if results_member is not None and results_member.isfile()
                else None
            )
            results_sha256 = (
                hashlib.sha256(results_bytes).hexdigest()
                if results_bytes is not None
                else None
            )
            run_config_member = member_by_name.get(
                "benchmark/results/full_benchmark/run_config.json"
            )
            run_config_bytes = (
                _tar_member_bytes(tar, run_config_member)
                if run_config_member is not None and run_config_member.isfile()
                else None
            )
            run_config_sha256 = (
                hashlib.sha256(run_config_bytes).hexdigest()
                if run_config_bytes is not None
                else None
            )
            sweep_member = member_by_name.get(
                "benchmark/results/full_benchmark/sweep_integrity_report.md"
            )
            sweep_report_has_pass = (
                sweep_member is not None
                and sweep_member.isfile()
                and _tar_text_member_contains(tar, sweep_member, SWEEP_PASS_VERDICT)
            )
            runtime_health_member = member_by_name.get(
                "benchmark/results/full_benchmark/runtime_health_report.txt"
            )
            runtime_health_has_pass = (
                runtime_health_member is not None
                and runtime_health_member.isfile()
                and _tar_text_member_contains(
                    tar,
                    runtime_health_member,
                    RUNTIME_HEALTH_PASS_VERDICT,
                )
            )
            runtime_health_binding_errors: list[str] = []
            metadata_text_errors: list[str] = []
            figdata_manifest_summary_errors: list[str] = []
            paper_artifact_lineage_errors: list[str] = []
            command_history_manifest_errors: list[str] = []
            if (
                runtime_health_member is not None
                and runtime_health_member.isfile()
                and results_bytes is not None
                and results_sha256 is not None
            ):
                runtime_health_bytes = _tar_member_bytes(tar, runtime_health_member)
                if runtime_health_bytes is not None:
                    runtime_health_binding_errors = _runtime_health_binding_errors(
                        runtime_health_bytes.decode("utf-8", errors="replace"),
                        expected_size_bytes=len(results_bytes),
                        expected_sha256=results_sha256,
                        expected_run_config_size_bytes=(
                            len(run_config_bytes)
                            if run_config_bytes is not None
                            else None
                        ),
                        expected_run_config_sha256=run_config_sha256,
                        artifact_name=(
                            "server return package runtime_health_report.txt"
                        ),
                    )
            metadata_members = {
                "zenodo_text": member_by_name.get(".zenodo.json"),
                "codemeta_text": member_by_name.get("codemeta.json"),
                "croissant_text": member_by_name.get("croissant-metadata.json"),
                "ro_crate_text": member_by_name.get("ro-crate-metadata.json"),
            }
            if all(
                member is not None and member.isfile()
                for member in metadata_members.values()
            ):
                metadata_texts: dict[str, str] = {}
                for key, member in metadata_members.items():
                    assert member is not None
                    raw = _tar_member_bytes(tar, member)
                    metadata_texts[key] = (
                        raw.decode("utf-8", errors="replace")
                        if raw is not None
                        else ""
                    )
                metadata_text_errors = _machine_readable_metadata_text_errors(
                    **metadata_texts
                )
            figdata_manifest_member = member_by_name.get(
                "paper/figdata/figdata_manifest.json"
            )
            figdata_manifest_bytes: bytes | None = None
            if (
                figdata_manifest_member is not None
                and figdata_manifest_member.isfile()
                and results_bytes is not None
            ):
                figdata_manifest_bytes = _tar_member_bytes(tar, figdata_manifest_member)
                if figdata_manifest_bytes is not None:
                    figdata_manifest_summary_errors = (
                        _tar_figdata_manifest_summary_errors(
                            figdata_manifest_bytes,
                            results_bytes,
                        )
                    )
            run_manifest_member = member_by_name.get(
                "benchmark/results/full_benchmark/RUN_MANIFEST.json"
            )
            numbers_member = member_by_name.get("paper/numbers.json")
            command_history_member = member_by_name.get("logs/pipeline_command_history.txt")
            command_history_manifest_errors = _tar_command_history_manifest_errors(
                run_manifest_bytes=(
                    _tar_member_bytes(tar, run_manifest_member)
                    if run_manifest_member is not None and run_manifest_member.isfile()
                    else None
                ),
                command_history_bytes=(
                    _tar_member_bytes(tar, command_history_member)
                    if command_history_member is not None and command_history_member.isfile()
                    else None
                ),
            )
            if results_sha256 is not None:
                paper_artifact_lineage_errors = _tar_paper_artifact_lineage_errors(
                    results_sha256=results_sha256,
                    run_manifest_bytes=(
                        _tar_member_bytes(tar, run_manifest_member)
                        if run_manifest_member is not None and run_manifest_member.isfile()
                        else None
                    ),
                    numbers_bytes=(
                        _tar_member_bytes(tar, numbers_member)
                        if numbers_member is not None and numbers_member.isfile()
                        else None
                    ),
                    figdata_manifest_bytes=figdata_manifest_bytes,
                )
    except tarfile.TarError as exc:
        errors.append(f"could not read server return package tar.gz: {type(exc).__name__}: {exc}")
        return errors

    if not members:
        errors.append("server return package tarball has no members")
        return errors

    normalized_members = [(member, member.name.replace("\\", "/")) for member in members]
    duplicate_names = sorted(
        name for name, count in Counter(name for _, name in normalized_members).items()
        if count > 1
    )
    for name in duplicate_names:
        errors.append(f"server return package contains duplicate member path: {name}")

    names = {name for _, name in normalized_members}
    for member, name in normalized_members:
        posix = PurePosixPath(name)
        windows = PureWindowsPath(member.name)
        parts = posix.parts
        if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in parts:
            errors.append(f"server return package contains unsafe member path: {member.name}")
        if member.issym() or member.islnk():
            errors.append(f"server return package contains link member: {member.name}")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            errors.append(
                "server return package contains unsupported special member: "
                f"{member.name}"
            )
        for prefix in DISALLOWED_PACKAGE_PREFIXES:
            if name == prefix or name.startswith(f"{prefix}/"):
                errors.append(
                    "server return package must not include optional robustness "
                    f"directory {prefix}; package supplementary results separately"
                )

    required_files, required_dirs = _expected_package_paths()
    for rel_path in required_files:
        name = rel_path.as_posix()
        member = next(
            (item for item, normalized in normalized_members if normalized == name),
            None,
        )
        if member is None:
            errors.append(f"server return package missing required file: {name}")
        elif not member.isfile():
            errors.append(f"server return package required path is not a file: {name}")
        elif member.size <= 0:
            errors.append(f"server return package required file is empty: {name}")
        elif (
            name == "benchmark/results/full_benchmark/sweep_integrity_report.md"
            and not sweep_report_has_pass
        ):
            errors.append(
                "server return package sweep_integrity_report.md does not "
                "contain the declared PASS verdict"
            )
        elif (
            name == "benchmark/results/full_benchmark/runtime_health_report.txt"
            and not runtime_health_has_pass
        ):
            errors.append(
                "server return package runtime_health_report.txt does not "
                "contain the runtime health PASS verdict"
            )
        elif name == "benchmark/results/full_benchmark/runtime_health_report.txt":
            errors.extend(runtime_health_binding_errors)

    errors.extend(f"server return package {error}" for error in metadata_text_errors)
    errors.extend(f"server return package {error}" for error in figdata_manifest_summary_errors)
    errors.extend(f"server return package {error}" for error in paper_artifact_lineage_errors)
    errors.extend(f"server return package {error}" for error in command_history_manifest_errors)

    for rel_path in required_dirs:
        dir_name = rel_path.as_posix()
        prefix = dir_name.rstrip("/") + "/"
        member = next(
            (item for item, normalized in normalized_members if normalized == dir_name),
            None,
        )
        if member is not None and not member.isdir():
            errors.append(
                "server return package required path is not a directory: "
                f"{dir_name}"
            )
            continue
        has_child = any(name.startswith(prefix) and name != prefix for name in names)
        if member is None and not has_child:
            errors.append(f"server return package missing required directory: {dir_name}")
        elif not has_child:
            errors.append(f"server return package required directory is empty: {dir_name}")
    return errors


def check_required_return_paths(
    *,
    results_root: Path = RESULTS_ROOT_DEFAULT,
    results_dir: Path = FULL_SWEEP_DEFAULT,
    paper_dir: Path = PAPER_DIR_DEFAULT,
    docs_dir: Path = DOCS_DIR_DEFAULT,
    logs_dir: Path = LOGS_DIR_DEFAULT,
) -> list[str]:
    """Check that the copied archive contains the expected top-level pieces."""
    return_root = _infer_return_root(
        results_root=results_root,
        paper_dir=paper_dir,
        docs_dir=docs_dir,
        logs_dir=logs_dir,
    )
    required_files = {
        "Zenodo metadata": return_root / ".zenodo.json",
        "citation metadata": return_root / "CITATION.cff",
        "CodeMeta metadata": return_root / "codemeta.json",
        "Croissant metadata": return_root / "croissant-metadata.json",
        "RO-Crate metadata": return_root / "ro-crate-metadata.json",
        "full-sweep results": results_dir / "results.csv",
        "full-sweep integrity report": results_dir / "sweep_integrity_report.md",
        "full-sweep runtime health report": results_dir / "runtime_health_report.txt",
        "full-sweep run configuration": results_dir / "run_config.json",
        "full-sweep run manifest": results_dir / "RUN_MANIFEST.json",
        "python environment lock": results_dir / "python_environment_pip_freeze.txt",
        "test-set sufficiency": results_dir / "analysis" / "test_set_sufficiency.csv",
        "case-validation ledger": docs_dir / "case_validation_status.csv",
        "case source fingerprints": docs_dir / "case_validation_sources.csv",
        "case validation-note fingerprints": docs_dir / "case_validation_note_sources.csv",
        "case experiment readiness": docs_dir / "case_experiment_readiness.csv",
        "case-validation summary": docs_dir / "CASE_VALIDATION_SUMMARY.md",
        "case web-validation audit": docs_dir / "CASE_WEB_VALIDATION_AUDIT.md",
        "experiment benchmark audit": docs_dir / "EXPERIMENT_BENCHMARK_AUDIT.md",
        "publication launch preflight": docs_dir / "PUBLICATION_LAUNCH_PREFLIGHT.md",
        "statistical analysis plan": docs_dir / "STATISTICAL_ANALYSIS_PLAN.md",
        "paper numbers JSON": paper_dir / "numbers.json",
        "paper numbers TeX": paper_dir / "numbers.tex",
        "figure manifest": paper_dir / "figure_manifest.json",
        "figdata manifest": paper_dir / "figdata" / "figdata_manifest.json",
        "pipeline command history": logs_dir / "pipeline_command_history.txt",
    }
    required_dirs = {
        "D1 ranking-reversal results": results_root / "ranking_reversal",
        "D2 sample-complexity results": results_root / "sample_complexity_0.20",
        "D3 reliability-recovery results": results_root / "reliability_recovery",
        "figdata directory": paper_dir / "figdata",
        "figure directory": paper_dir / "figures",
    }

    errors: list[str] = []
    for label, path in {
        "results root": results_root,
        "full-sweep result directory": results_dir,
        "paper directory": paper_dir,
        "docs directory": docs_dir,
        "logs directory": logs_dir,
    }.items():
        try:
            if path.is_symlink():
                errors.append(f"{label} is a symlink and cannot be trusted: {path}")
        except PermissionError as exc:
            errors.append(f"{label}: {_filesystem_error(path, exc)}")
        except OSError as exc:
            if getattr(exc, "winerror", None) == 5:
                errors.append(f"{label}: {_filesystem_error(path, exc)}")
            else:
                raise

    for label, path in required_files.items():
        try:
            if path.is_symlink():
                errors.append(f"{label} is a symlink and cannot be trusted: {path}")
            elif not path.exists():
                errors.append(f"{label} missing: {path}")
            elif not path.is_file():
                errors.append(f"{label} is not a file: {path}")
            elif path.stat().st_size == 0:
                errors.append(f"{label} is empty: {path}")
            elif (
                label == "full-sweep integrity report"
                and SWEEP_PASS_VERDICT
                not in path.read_text(encoding="utf-8", errors="replace")
            ):
                errors.append(
                    "full-sweep integrity report does not contain the declared "
                    f"PASS verdict: {path}"
                )
            elif (
                label == "full-sweep runtime health report"
                and RUNTIME_HEALTH_PASS_VERDICT
                not in path.read_text(encoding="utf-8", errors="replace")
            ):
                errors.append(
                    "full-sweep runtime health report does not contain the "
                    f"runtime health PASS verdict: {path}"
                )
            elif label == "full-sweep runtime health report":
                results_csv = results_dir / "results.csv"
                run_config_json = results_dir / "run_config.json"
                if results_csv.is_file() and not results_csv.is_symlink():
                    run_config_kwargs = {}
                    if run_config_json.is_file() and not run_config_json.is_symlink():
                        run_config_kwargs = {
                            "expected_run_config_size_bytes": run_config_json.stat().st_size,
                            "expected_run_config_sha256": _sha256_file(run_config_json),
                        }
                    errors.extend(
                        _runtime_health_binding_errors(
                            path.read_text(encoding="utf-8", errors="replace"),
                            expected_size_bytes=results_csv.stat().st_size,
                            expected_sha256=_sha256_file(results_csv),
                            artifact_name="full-sweep runtime health report",
                            **run_config_kwargs,
                        )
                    )
        except PermissionError as exc:
            errors.append(f"{label}: {_filesystem_error(path, exc)}")
        except OSError as exc:
            if getattr(exc, "winerror", None) == 5:
                errors.append(f"{label}: {_filesystem_error(path, exc)}")
            else:
                raise

    errors.extend(check_machine_readable_metadata(return_root))

    run_config_path = results_dir / "run_config.json"
    pip_freeze_path = results_dir / "python_environment_pip_freeze.txt"
    if run_config_path.exists() and pip_freeze_path.exists():
        try:
            from scripts.check_submission_ready import (
                check_python_environment_lock_matches_run_config,
            )
        except Exception as exc:
            errors.append(
                "could not import python-environment/run-config consistency gate: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            errors.extend(
                check_python_environment_lock_matches_run_config(
                    run_config_path,
                    pip_freeze_path,
                )
            )

    for label, path in required_dirs.items():
        try:
            if path.is_symlink():
                errors.append(f"{label} is a symlink and cannot be trusted: {path}")
            elif not path.exists():
                errors.append(f"{label} missing: {path}")
            elif not path.is_dir():
                errors.append(f"{label} is not a directory: {path}")
            elif not any(path.iterdir()):
                errors.append(f"{label} is empty: {path}")
        except PermissionError as exc:
            errors.append(f"{label}: {_filesystem_error(path, exc)}")
        except OSError as exc:
            if getattr(exc, "winerror", None) == 5:
                errors.append(f"{label}: {_filesystem_error(path, exc)}")
            else:
                raise
    return errors


def _infer_return_root(
    *,
    results_root: Path,
    paper_dir: Path,
    docs_dir: Path,
    logs_dir: Path,
) -> Path:
    """Infer the extracted package root from the standard top-level dirs."""
    candidates: list[Path] = []
    if results_root.name == "results" and results_root.parent.name == "benchmark":
        candidates.append(results_root.parent.parent)
    else:
        candidates.append(results_root.parent)
    candidates.extend(path.parent for path in (paper_dir, docs_dir, logs_dir))
    try:
        return Path(os.path.commonpath([str(path.resolve()) for path in candidates]))
    except ValueError:
        return ROOT


def check_case_web_validation_audit_content(
    *,
    docs_dir: Path = DOCS_DIR_DEFAULT,
) -> list[str]:
    """Validate returned P2 web-audit content, not only file presence."""
    try:
        from scripts.check_server_launch_ready import (
            validate_case_web_validation_audit_coverage,
            validate_p2_source_verification_snapshots,
        )
    except Exception as exc:
        return [
            "could not import case web-validation audit coverage gate: "
            f"{type(exc).__name__}: {exc}"
        ]

    errors = validate_case_web_validation_audit_coverage(
        status_csv=docs_dir / "case_validation_status.csv",
        audit_md=docs_dir / "CASE_WEB_VALIDATION_AUDIT.md",
    )
    errors.extend(
        validate_p2_source_verification_snapshots(
            status_csv=docs_dir / "case_validation_status.csv",
            audit_md=docs_dir / "CASE_WEB_VALIDATION_AUDIT.md",
        )
    )
    status_csv = docs_dir / "case_validation_status.csv"
    audit_md = docs_dir / "CASE_WEB_VALIDATION_AUDIT.md"
    return errors


def _core_full_sweep_missing(
    *,
    results_dir: Path,
) -> bool:
    required = [
        results_dir / "results.csv",
        results_dir / "sweep_integrity_report.md",
        results_dir / "RUN_MANIFEST.json",
    ]
    try:
        return any(not _path.is_file() or _path.stat().st_size == 0 for _path in required)
    except PermissionError:
        return True
    except OSError as exc:
        if getattr(exc, "winerror", None) == 5:
            return True
        raise


def check_full_sweep_gate(results_dir: Path = FULL_SWEEP_DEFAULT) -> list[str]:
    """Run the publication roll-up's full-sweep gate against the restored CSV."""
    from scripts.audit_publication_readiness import summarize_sweep_artifacts

    gate = summarize_sweep_artifacts(results_dir)
    if gate.status == "PASS":
        return []
    return [
        (
            f"Full sweep gate is {gate.status}: {gate.evidence} "
            f"Next action: {gate.next_action}"
        )
    ]


def _parse_command_history_entries(
    text: str,
    *,
    context: str = "pipeline_command_history.txt",
) -> tuple[list[dict[str, str]], list[str]]:
    entries: list[dict[str, str]] = []
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "\t" not in line:
            errors.append(
                f"{context} line {line_number} is not label<TAB>command"
            )
            continue
        label, command = line.split("\t", 1)
        label = label.strip()
        command = command.strip()
        if not label or not command:
            errors.append(
                f"{context} line {line_number} has blank label or command"
            )
            continue
        entries.append({"label": label, "command": command})
    return entries, errors


def _read_command_history_entries(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    return _parse_command_history_entries(
        path.read_text(encoding="utf-8", errors="replace")
    )


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


def _command_history_manifest_contract_errors(
    manifest: dict[str, object],
    observed_entries: list[dict[str, str]],
    *,
    history_context: str = "pipeline_command_history.txt",
) -> list[str]:
    experiment = manifest.get("experiment", {})
    if not isinstance(experiment, dict):
        return ["RUN_MANIFEST experiment section is missing or malformed"]
    requested = experiment.get("command_files_requested")
    manifest_entries = experiment.get("command_entries")
    if not isinstance(requested, list) or not requested:
        return ["RUN_MANIFEST experiment.command_files_requested is missing or empty"]
    malformed_requested = [
        item
        for item in requested
        if not isinstance(item, str) or not item.strip()
    ]
    if malformed_requested:
        return [
            "RUN_MANIFEST experiment.command_files_requested contains "
            "non-string or blank entries"
        ]
    nonportable_requested = _nonportable_command_file_requests(requested)
    if nonportable_requested:
        return [
            "RUN_MANIFEST experiment.command_files_requested contains "
            "non-portable paths; use repo-relative paths without parent "
            f"traversal: {nonportable_requested}"
        ]
    if not isinstance(manifest_entries, list) or not manifest_entries:
        return ["RUN_MANIFEST experiment.command_entries is missing or empty"]

    bad_entries = [
        item for item in manifest_entries
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not item.get("label", "").strip()
            or not isinstance(item.get("command"), str)
            or not item.get("command", "").strip()
        )
    ]
    if bad_entries:
        return ["RUN_MANIFEST experiment.command_entries contains malformed entries"]

    pipeline_manifest_entries = [
        entry for entry in manifest_entries
        if entry.get("label") == "pipeline_invocation"
    ]
    if len(pipeline_manifest_entries) != 1:
        return [
            "RUN_MANIFEST experiment.command_entries must contain exactly one "
            "pipeline_invocation entry"
        ]
    if "--dry-run" in pipeline_manifest_entries[0].get("command", "").split():
        return ["RUN_MANIFEST experiment.command_entries records a dry-run pipeline_invocation"]
    manifest_invocation_error = _formal_pipeline_invocation_error(
        str(pipeline_manifest_entries[0].get("command", ""))
    )
    if manifest_invocation_error:
        return [f"RUN_MANIFEST experiment.command_entries {manifest_invocation_error}"]

    requested_set = {
        str(item).replace("\\", "/") for item in requested if isinstance(item, str)
    }
    expected_rel = "logs/pipeline_command_history.txt"
    errors: list[str] = []
    if not any(item == expected_rel or item.endswith(f"/{expected_rel}") for item in requested_set):
        errors.append(
            "RUN_MANIFEST experiment.command_files_requested does not include "
            f"{expected_rel}"
        )

    history_only = [
        entry for entry in manifest_entries
        if isinstance(entry, dict)
        and entry.get("label") != "manual_command"
    ]
    if observed_entries != history_only:
        errors.append(
            f"{history_context} entries do not match RUN_MANIFEST "
            "experiment.command_entries"
        )

    pipeline_entries = [
        entry for entry in observed_entries
        if entry.get("label") == "pipeline_invocation"
    ]
    if len(pipeline_entries) != 1:
        errors.append(
            f"{history_context} must contain exactly one "
            "pipeline_invocation entry"
        )
    elif "--dry-run" in pipeline_entries[0].get("command", "").split():
        errors.append(f"{history_context} records a dry-run pipeline_invocation")
    else:
        history_invocation_error = _formal_pipeline_invocation_error(
            str(pipeline_entries[0].get("command", ""))
        )
        if history_invocation_error:
            errors.append(f"{history_context} {history_invocation_error}")

    return errors


def check_command_history_matches_manifest(
    manifest_path: Path,
    *,
    logs_dir: Path = LOGS_DIR_DEFAULT,
) -> list[str]:
    """Ensure restored command history matches RUN_MANIFEST provenance."""
    if not manifest_path.exists():
        return [f"RUN_MANIFEST.json does not exist: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"]

    history_path = logs_dir / "pipeline_command_history.txt"
    if not history_path.is_file() or history_path.stat().st_size == 0:
        return [f"pipeline command history missing or empty: {history_path}"]

    observed_entries, history_errors = _read_command_history_entries(history_path)
    if history_errors:
        return history_errors
    if not observed_entries:
        return [f"pipeline command history has no command entries: {history_path}"]

    return _command_history_manifest_contract_errors(manifest, observed_entries)


def check_manifest_package(
    manifest_path: Path,
    *,
    logs_dir: Path = LOGS_DIR_DEFAULT,
) -> list[str]:
    """Validate run-manifest provenance and archived artifact hashes."""
    from scripts.check_submission_ready import (
        check_case_source_fingerprints,
        check_manifest_artifacts,
        check_manifest_provenance,
        check_manifest_result_contract,
        check_method_source_fingerprints,
        check_prereg_manifest_config,
    )

    errors: list[str] = []
    errors.extend(check_manifest_artifacts(manifest_path))
    errors.extend(check_manifest_provenance(manifest_path))
    errors.extend(check_manifest_result_contract(manifest_path))
    errors.extend(check_prereg_manifest_config(manifest_path))
    errors.extend(check_case_source_fingerprints(manifest_path))
    errors.extend(check_method_source_fingerprints(manifest_path))
    errors.extend(check_command_history_matches_manifest(manifest_path, logs_dir=logs_dir))
    return errors


def check_downstream_analysis_package(
    *,
    results_root: Path = RESULTS_ROOT_DEFAULT,
    results_dir: Path = FULL_SWEEP_DEFAULT,
    paper_dir: Path = PAPER_DIR_DEFAULT,
) -> list[str]:
    """Validate D1/D2/D3, figdata, figures, and numbers without scanning text."""
    from scripts.check_submission_ready import (
        check_figdata_manifest,
        check_figdata_methods,
        check_figdata_schema,
        check_figure_manifest,
        check_numbers_consistency,
        check_numbers_provenance,
        check_numbers_submission_completeness,
        check_numbers_tex_consistency,
        check_paper_artifact_lineage,
        check_required_analysis_artifacts,
    )

    manifest_path = results_dir / "RUN_MANIFEST.json"
    figdata_dir = paper_dir / "figdata"
    errors: list[str] = []
    errors.extend(check_numbers_consistency(paper_dir / "numbers.json", manifest_path))
    errors.extend(check_numbers_submission_completeness(paper_dir / "numbers.json"))
    errors.extend(check_numbers_provenance(paper_dir / "numbers.json", paper_dir))
    errors.extend(
        check_numbers_tex_consistency(
            paper_dir / "numbers.tex",
            paper_dir / "numbers.json",
        )
    )
    errors.extend(check_figure_manifest(paper_dir / "figure_manifest.json", paper_dir))
    errors.extend(check_figdata_schema(figdata_dir))
    errors.extend(check_figdata_manifest(figdata_dir, paper_dir))
    errors.extend(check_figdata_methods(figdata_dir, manifest_path))
    errors.extend(
        check_paper_artifact_lineage(
            manifest_path,
            paper_dir / "numbers.json",
            figdata_dir,
        )
    )
    errors.extend(
        check_required_analysis_artifacts(
            results_root / "ranking_reversal",
            results_root / "sample_complexity_0.20",
            results_root / "reliability_recovery",
            results_dir / "analysis",
        )
    )
    return errors


def check_server_return_package(
    *,
    results_root: Path = RESULTS_ROOT_DEFAULT,
    results_dir: Path = FULL_SWEEP_DEFAULT,
    paper_dir: Path = PAPER_DIR_DEFAULT,
    docs_dir: Path = DOCS_DIR_DEFAULT,
    logs_dir: Path = LOGS_DIR_DEFAULT,
    package_path: Path | None = None,
    sha256_path: Path | None = None,
) -> dict[str, list[str]]:
    manifest_path = results_dir / "RUN_MANIFEST.json"
    groups: dict[str, list[str]] = {}
    if package_path is not None:
        groups["package"] = check_return_tarball(package_path, sha256_path=sha256_path)
    try:
        required_path_errors = check_required_return_paths(
            results_root=results_root,
            results_dir=results_dir,
            paper_dir=paper_dir,
            docs_dir=docs_dir,
            logs_dir=logs_dir,
        )
    except PermissionError as exc:
        required_path_errors = [_filesystem_error(results_dir, exc)]
    except OSError as exc:
        if getattr(exc, "winerror", None) == 5:
            required_path_errors = [_filesystem_error(results_dir, exc)]
        else:
            raise
    groups["required-paths"] = required_path_errors
    groups["case-web-validation-audit"] = check_case_web_validation_audit_content(
        docs_dir=docs_dir
    )
    if _core_full_sweep_missing(results_dir=results_dir):
        groups["full-sweep"] = check_full_sweep_gate(results_dir)
        groups["manifest"] = [
            "Skipped until results.csv, sweep_integrity_report.md, and RUN_MANIFEST.json "
            "are all present and non-empty."
        ]
        groups["downstream-analysis"] = [
            "Skipped until the core full-sweep archive is restored."
        ]
        return groups

    full_sweep_errors = check_full_sweep_gate(results_dir)
    groups["full-sweep"] = full_sweep_errors
    if full_sweep_errors:
        groups["manifest"] = [
            "Skipped until the restored full-sweep gate passes; fix the "
            "full-sweep findings first so manifest diagnostics are tied to a "
            "trustworthy results.csv."
        ]
        groups["downstream-analysis"] = [
            "Skipped until the restored full-sweep gate passes."
        ]
        return groups

    groups["manifest"] = check_manifest_package(manifest_path, logs_dir=logs_dir)
    try:
        groups["downstream-analysis"] = check_downstream_analysis_package(
            results_root=results_root,
            results_dir=results_dir,
            paper_dir=paper_dir,
        )
    except PermissionError as exc:
        groups["downstream-analysis"] = [_filesystem_error(paper_dir, exc)]
    except OSError as exc:
        if getattr(exc, "winerror", None) == 5:
            groups["downstream-analysis"] = [_filesystem_error(paper_dir, exc)]
        else:
            raise
    return groups


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=FULL_SWEEP_DEFAULT)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR_DEFAULT)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR_DEFAULT)
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR_DEFAULT)
    parser.add_argument(
        "--package",
        type=Path,
        help="Optional server_return_package_<UTC>.tar.gz to verify before trusting extracted files.",
    )
    parser.add_argument(
        "--sha256",
        type=Path,
        help="Optional sha256 sidecar path. Defaults to <package>.sha256 when --package is set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    groups = check_server_return_package(
        results_root=args.results_root,
        results_dir=args.results_dir,
        paper_dir=args.paper_dir,
        docs_dir=args.docs_dir,
        logs_dir=args.logs_dir,
        package_path=args.package if args.package else None,
        sha256_path=args.sha256 if args.sha256 else None,
    )

    total = 0
    for group, errors in groups.items():
        if not errors:
            print(f"[{group}] PASS")
            continue
        print(f"[{group}] FAIL ({len(errors)})")
        for error in errors:
            print(f"  - {error}")
        total += len(errors)

    if total:
        print(f"[result] FAIL - server return package has {total} issue(s).")
        return 1
    print("[result] PASS - server return package is ready for local paper finalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
