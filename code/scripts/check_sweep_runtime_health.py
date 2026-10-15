"""Read-only runtime health gate for the full_benchmark sweep.

Run this periodically while the server sweep is active.  It never launches or
resumes experiments; it only parses the current ``results.csv`` and reports
whether the partial archive still looks compatible with the preregistered
headline protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import doe_testbed as db  # noqa: E402
from scripts._check_sweep_progress import (  # noqa: E402
    GRID_COLUMNS,
    PREREG_BUDGETS,
    PREREG_CASES,
    PREREG_METHOD_FAMILIES,
    PREREG_METHODS,
    PREREG_REPS,
    PREREG_TARGET,
    VALID_STATUSES,
    summarize_progress,
)
from benchmark.core import (  # noqa: E402
    _runtime_environment_snapshot,
    _torch_reproducibility_snapshot,
    case_source_fingerprint,
    derive_seed,
    method_source_fingerprint,
)
from benchmark.runtime_contracts import (  # noqa: E402
    ALLOWED_FAILURE_CATEGORIES,
    RESULT_SCHEMA_VERSION,
    csv_header_contract_errors,
    failure_category_is_consistent,
    invalid_protocol_digest_mask,
    invalid_runtime_duration_mask,
    strict_integer_series,
)

DEFAULT_RESULTS_CSV = ROOT / "benchmark" / "results" / "full_benchmark" / "results.csv"
PREREG_N_TEST = 1000
PREREG_TEST_SEED = 0
PREREG_SEED_STRATEGY = "case_repeat"
PREREG_SEED_START = 1000
_PREREG_METHOD_FINGERPRINTS_CACHE: dict[str, dict[str, str]] | None = None
RUNTIME_CONTRACT_COLUMNS = {
    "case_id",
    "case_source_path",
    "case_source_sha256",
    "domain",
    "dim",
    "n_output",
    "method",
    "method_family",
    "method_source_path",
    "method_source_sha256",
    "budget_mult",
    "n_train",
    "n_test",
    "repeat",
    "seed",
    "test_seed",
    "seed_strategy",
    "seed_start",
    "result_schema_version",
    "protocol_digest",
    "status",
    "failure_category",
    "t_total_s",
    "error",
    "nrmse",
    "worker_host",
    "worker_pid",
    "cell_started_utc",
    "cell_finished_utc",
    "rss_gb_start",
    "rss_gb_end",
    "rss_gb_peak",
}
RUNTIME_TRACE_COLUMNS = {
    "worker_host",
    "worker_pid",
    "cell_started_utc",
    "cell_finished_utc",
    "rss_gb_start",
    "rss_gb_end",
    "rss_gb_peak",
}
OPTIONAL_OK_NUMERIC_METRICS = ("rmse", "mae", "maxae", "q95ae", "r2")
RUN_CONFIG_RUNTIME_CONTROLS_VERDICT = (
    "run_config.json runtime controls recorded and valid"
)
RUN_CONFIG_ENVIRONMENT_VERDICT = (
    "run_config.json runtime environment recorded and valid"
)
RUN_CONFIG_TORCH_REPRO_VERDICT = (
    "run_config.json torch reproducibility policy recorded and valid"
)
CURRENT_ENVIRONMENT_VERDICT = (
    "current runtime environment matches run_config.json snapshot"
)
FAILURE_CATEGORY_SUMMARY_VERDICT = "runtime failure-category telemetry summarized"
RESOURCE_SUMMARY_VERDICT = "runtime resource telemetry summarized"
WALL_TIME_SUMMARY_VERDICT = "runtime wall-time telemetry summarized"
REQUIRED_RUN_CONFIG_ENV_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "gpytorch",
    "botorch",
    "neuralfoil",
    "aerosandbox",
    "psutil",
)
TEMP_STORAGE_ENV_VARS = (
    "TMPDIR",
    "TMP",
    "TEMP",
    "TEMPDIR",
    "SLURM_TMPDIR",
    "SCRATCH",
    "LOCAL_SCRATCH",
    "JOBSCRATCH",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prereg_method_fingerprints() -> dict[str, dict[str, str]]:
    """Compute method source fingerprints only when results.csv needs them."""
    global _PREREG_METHOD_FINGERPRINTS_CACHE
    if _PREREG_METHOD_FINGERPRINTS_CACHE is None:
        from benchmark.methods import nature_pool

        _PREREG_METHOD_FINGERPRINTS_CACHE = {
            method.name: method_source_fingerprint(method)
            for method in nature_pool()
        }
    return _PREREG_METHOD_FINGERPRINTS_CACHE


class _LazyMethodFingerprintMap:
    def __getitem__(self, key: str) -> dict[str, str]:
        return _prereg_method_fingerprints()[key]

    def get(self, key: str, default: object = None) -> object:
        return _prereg_method_fingerprints().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in _prereg_method_fingerprints()

    def __iter__(self):
        return iter(_prereg_method_fingerprints())

    def __len__(self) -> int:
        return len(_prereg_method_fingerprints())

    def items(self):
        return _prereg_method_fingerprints().items()


PREREG_METHOD_FINGERPRINTS = _LazyMethodFingerprintMap()


def _read_results_csv(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if path.is_symlink():
        return None, [f"results.csv is a symlink and cannot be trusted: {path}"]
    if not path.exists():
        return None, [f"results.csv is missing: {path}"]
    if path.stat().st_size == 0:
        return None, [f"results.csv is empty: {path}"]
    header_errors = csv_header_contract_errors(path)
    if header_errors:
        return None, header_errors
    try:
        # The archived error column intentionally mixes empty cells and
        # failure messages; parse in one pass so the health report is not
        # polluted by chunk-inference DtypeWarnings on the full CSV.
        return pd.read_csv(path, low_memory=False), []
    except Exception as exc:
        return None, [f"could not parse results.csv: {type(exc).__name__}: {exc}"]


def _finite_non_negative(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & numeric.map(math.isfinite) & (numeric >= 0)


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_json_number(value: object) -> float | None:
    """Parse runner-authored numeric JSON fields without accepting strings.

    ``run_config.json`` is a provenance snapshot written by Python's JSON
    encoder.  Runtime controls should therefore be real JSON numbers or null,
    not hand-edited strings that happen to parse as numbers.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _temp_storage_candidates() -> list[tuple[str, Path]]:
    """Return unique temp/scratch paths that the current job may rely on."""
    candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def add(label: str, path_like: str | Path | None) -> None:
        if path_like in (None, ""):
            return
        try:
            path = Path(str(path_like)).expanduser()
        except Exception:
            return
        key = path.as_posix()
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, path))

    add("tempfile.gettempdir()", tempfile.gettempdir())
    for name in TEMP_STORAGE_ENV_VARS:
        add(name, os.environ.get(name))
    return candidates


def _positive_int_value(value: object) -> int | None:
    """Parse strict JSON positive-integer config fields.

    ``run_config.json`` is generated by the benchmark runner, so protocol
    counts should be real JSON integers.  Accepting strings or floats here
    would make a hand-edited startup snapshot easier to misread during resume
    or return-package validation.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def _non_negative_int_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _runtime_summary_lines(df: pd.DataFrame) -> list[str]:
    """Return human-readable runtime telemetry summaries for archival reports."""
    lines: list[str] = []
    latest = df.drop_duplicates(GRID_COLUMNS, keep="last")

    if "failure_category" in latest.columns:
        category_counts = {
            str(key): int(value)
            for key, value in sorted(
                latest["failure_category"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", "<blank>")
                .value_counts()
                .items()
            )
        }
        non_ok = {
            key: value
            for key, value in category_counts.items()
            if key not in {"ok", "<blank>"}
        }
        lines.append(
            "[summary] "
            f"{FAILURE_CATEGORY_SUMMARY_VERDICT}: "
            f"unique_cells={len(latest)} "
            f"failure_category_counts={json.dumps(category_counts, sort_keys=True)} "
            f"non_ok_failure_category_counts={json.dumps(non_ok, sort_keys=True)}"
        )

    if RUNTIME_TRACE_COLUMNS.issubset(latest.columns):
        rss_peak = pd.to_numeric(latest["rss_gb_peak"], errors="coerce")
        valid_peak = rss_peak[rss_peak.notna() & rss_peak.map(math.isfinite)]
        if valid_peak.empty:
            rss_payload = {"n_valid": 0}
        else:
            by_method = (
                latest.assign(_rss_peak=rss_peak)
                .dropna(subset=["_rss_peak"])
                .groupby("method")["_rss_peak"]
                .max()
                .sort_values(ascending=False)
                .head(5)
            )
            rss_payload = {
                "n_valid": int(valid_peak.shape[0]),
                "min": round(float(valid_peak.min()), 6),
                "p50": round(float(valid_peak.quantile(0.50)), 6),
                "p95": round(float(valid_peak.quantile(0.95)), 6),
                "max": round(float(valid_peak.max()), 6),
                "top_methods": {
                    str(method): round(float(value), 6)
                    for method, value in by_method.items()
                },
            }
        lines.append(
            "[summary] "
            f"{RESOURCE_SUMMARY_VERDICT}: "
            f"rss_gb_peak={json.dumps(rss_payload, sort_keys=True)}"
        )
    if "t_total_s" in latest.columns:
        wall_time = pd.to_numeric(latest["t_total_s"], errors="coerce")
        valid_time = wall_time[
            wall_time.notna() & wall_time.map(math.isfinite) & (wall_time >= 0)
        ]
        if valid_time.empty:
            wall_payload = {"n_valid": 0}
        else:
            by_method = (
                latest.assign(_wall_time_s=wall_time)
                .dropna(subset=["_wall_time_s"])
                .groupby("method")["_wall_time_s"]
                .max()
                .sort_values(ascending=False)
                .head(5)
            )
            wall_payload = {
                "n_valid": int(valid_time.shape[0]),
                "min": round(float(valid_time.min()), 6),
                "p50": round(float(valid_time.quantile(0.50)), 6),
                "p95": round(float(valid_time.quantile(0.95)), 6),
                "max": round(float(valid_time.max()), 6),
                "top_methods": {
                    str(method): round(float(value), 6)
                    for method, value in by_method.items()
                },
            }
        lines.append(
            "[summary] "
            f"{WALL_TIME_SUMMARY_VERDICT}: "
            f"t_total_s={json.dumps(wall_payload, sort_keys=True)}"
        )
    return lines


def _integer_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    return strict_integer_series(series)


def _valid_utc_timestamp_series(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    return text.ne("") & parsed.notna()


def _load_run_config(path: Path) -> tuple[dict[str, object] | None, list[str]]:
    if path.is_symlink():
        return None, [f"run_config.json is a symlink and cannot be trusted: {path}"]
    if not path.exists():
        return None, [f"run_config.json is missing: {path}"]
    if not path.is_file() or path.stat().st_size == 0:
        return None, [f"run_config.json is not a non-empty file: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"could not parse run_config.json: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"run_config.json root must be an object: {path}"]
    return payload, []


def _string_set(values: object, *, key: str, field: str) -> set[str] | None:
    if not isinstance(values, list):
        return None
    out: set[str] = set()
    for item in values:
        if not isinstance(item, dict) or not str(item.get(key, "")).strip():
            return None
        value = str(item.get(key, "")).strip()
        if not value:
            return None
        if value in out:
            return None
        out.add(value)
        if field and not str(item.get(field, "")).strip():
            return None
    return out


def _run_config_environment_errors(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return ["run_config.json environment must be an object"]

    if environment.get("schema_version") != 1:
        errors.append(
            "run_config.json environment.schema_version must be 1; "
            f"observed {environment.get('schema_version')!r}"
        )
    for key in (
        "python_version",
        "python_executable",
        "platform",
        "machine",
        "node",
        "cwd",
    ):
        if not str(environment.get(key, "")).strip():
            errors.append(f"run_config.json environment missing non-empty {key}")

    packages = environment.get("packages")
    if not isinstance(packages, dict):
        errors.append("run_config.json environment.packages must be an object")
        return errors

    for package in REQUIRED_RUN_CONFIG_ENV_PACKAGES:
        record = packages.get(package)
        if not isinstance(record, dict):
            errors.append(
                f"run_config.json environment.packages missing object for {package}"
            )
            continue
        status = str(record.get("status", "")).strip()
        if status not in {"installed", "missing", "error"}:
            errors.append(
                "run_config.json environment.packages "
                f"{package}.status must be installed/missing/error; observed {status!r}"
            )
        elif status != "installed":
            errors.append(
                "run_config.json environment.packages "
                f"{package}.status must be 'installed' for the formal headline run; "
                f"observed {status!r}"
            )
        version = str(record.get("version", "")).strip()
        if status == "installed" and not version:
            errors.append(
                "run_config.json environment.packages "
                f"{package}.version must be non-empty when installed"
            )
    return errors


def _run_config_temp_storage_errors(payload: dict[str, object]) -> list[str]:
    """Validate Stage-2 startup temp/scratch storage provenance."""
    errors: list[str] = []
    temp_storage = payload.get("temp_storage")
    if not isinstance(temp_storage, dict):
        return ["run_config.json temp_storage must be an object"]

    if temp_storage.get("schema_version") != 1:
        errors.append(
            "run_config.json temp_storage.schema_version must be 1; "
            f"observed {temp_storage.get('schema_version')!r}"
        )
    resolved_path = temp_storage.get("resolved_path")
    if not isinstance(resolved_path, str) or not resolved_path.strip():
        errors.append("run_config.json temp_storage.resolved_path must be non-empty")
    for key in ("exists", "writable"):
        if not isinstance(temp_storage.get(key), bool):
            errors.append(f"run_config.json temp_storage.{key} must be boolean")
    free_gb = temp_storage.get("free_gb")
    if free_gb is not None:
        parsed = _finite_float(free_gb)
        if parsed is None or parsed < 0:
            errors.append(
                "run_config.json temp_storage.free_gb must be null or a "
                f"finite non-negative number; observed {free_gb!r}"
            )
    env = temp_storage.get("env")
    if not isinstance(env, dict):
        errors.append("run_config.json temp_storage.env must be an object")
    else:
        for name in TEMP_STORAGE_ENV_VARS:
            if name not in env:
                errors.append(f"run_config.json temp_storage.env missing {name}")
    return errors


def _torch_reproducibility_errors(
    payload: dict[str, object],
    *,
    context: str = "run_config.json",
) -> list[str]:
    """Validate recorded PyTorch backend reproducibility state."""
    errors: list[str] = []
    torch_repro = payload.get("torch_reproducibility")
    if not isinstance(torch_repro, dict):
        return [f"{context} torch_reproducibility must be an object"]
    if torch_repro.get("schema_version") != 1:
        errors.append(
            f"{context} torch_reproducibility.schema_version must be 1; "
            f"observed {torch_repro.get('schema_version')!r}"
        )
    status = str(torch_repro.get("status", "")).strip()
    if status not in {"ok", "missing"}:
        errors.append(
            f"{context} torch_reproducibility.status must be ok/missing; "
            f"observed {status!r}"
        )
    policy = torch_repro.get("policy")
    if not isinstance(policy, dict):
        errors.append(f"{context} torch_reproducibility.policy must be an object")
    else:
        expected_policy = {
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
        for key, expected in expected_policy.items():
            if policy.get(key) != expected:
                errors.append(
                    f"{context} torch_reproducibility.policy.{key} must be "
                    f"{expected!r}; observed {policy.get(key)!r}"
                )
        workspace = policy.get("cublas_workspace_config")
        if not isinstance(workspace, str) or not workspace.strip():
            errors.append(
                f"{context} torch_reproducibility.policy.cublas_workspace_config "
                "must be a non-empty string"
            )

    if status == "ok":
        state = torch_repro.get("state")
        if not isinstance(state, dict):
            errors.append(f"{context} torch_reproducibility.state must be an object")
        else:
            if state.get("deterministic_algorithms") is not True:
                errors.append(
                    f"{context} torch deterministic_algorithms policy is not enabled"
                )
            if state.get("cudnn_benchmark") is True:
                errors.append(f"{context} torch cudnn benchmark is enabled")
            if state.get("cuda_matmul_allow_tf32") is True:
                errors.append(f"{context} torch CUDA matmul TF32 is enabled")
            if state.get("cudnn_allow_tf32") is True:
                errors.append(f"{context} torch cuDNN TF32 is enabled")
            if state.get("float32_matmul_precision") not in {
                "highest",
                "high",
                None,
            }:
                errors.append(
                    f"{context} torch float32_matmul_precision is unexpected: "
                    f"{state.get('float32_matmul_precision')!r}"
                )
    return errors


def _current_environment_drift_errors(payload: dict[str, object]) -> list[str]:
    """Compare the active interpreter environment with run_config.json."""
    errors: list[str] = []
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return ["run_config.json environment must be an object"]

    _torch_reproducibility_snapshot(apply_policy=True)
    current = _runtime_environment_snapshot()
    for key in ("python_version", "python_executable", "machine"):
        expected = str(environment.get(key, "")).strip()
        observed = str(current.get(key, "")).strip()
        if expected and observed and observed != expected:
            errors.append(
                f"current environment {key} drifted from run_config.json: "
                f"expected={expected!r}, observed={observed!r}"
            )

    recorded_packages = environment.get("packages")
    current_packages = current.get("packages")
    if isinstance(recorded_packages, dict) and isinstance(current_packages, dict):
        for package in REQUIRED_RUN_CONFIG_ENV_PACKAGES:
            expected_record = recorded_packages.get(package)
            observed_record = current_packages.get(package)
            if not isinstance(expected_record, dict) or not isinstance(
                observed_record, dict
            ):
                continue
            expected_status = str(expected_record.get("status", "")).strip()
            observed_status = str(observed_record.get("status", "")).strip()
            if expected_status != observed_status:
                errors.append(
                    f"current environment package status drifted for {package}: "
                    f"expected={expected_status!r}, observed={observed_status!r}"
                )
                continue
            if expected_status == "installed":
                expected_version = str(expected_record.get("version", "")).strip()
                observed_version = str(observed_record.get("version", "")).strip()
                if (
                    expected_version
                    and observed_version
                    and observed_version != expected_version
                ):
                    errors.append(
                        f"current environment package version drifted for {package}: "
                        f"expected={expected_version!r}, observed={observed_version!r}"
                    )

    thread_env = payload.get("thread_env")
    if isinstance(thread_env, dict):
        for name, expected in thread_env.items():
            observed = os.environ.get(str(name))
            if observed != expected:
                errors.append(
                    f"current thread env drifted for {name}: "
                    f"expected={expected!r}, observed={observed!r}"
                )

    recorded_torch = payload.get("torch_reproducibility")
    current_torch = current.get("torch_reproducibility")
    if isinstance(recorded_torch, dict) and isinstance(current_torch, dict):
        recorded_state = recorded_torch.get("state")
        current_state = current_torch.get("state")
        if isinstance(recorded_state, dict) and isinstance(current_state, dict):
            for key in (
                "deterministic_algorithms",
                "cudnn_benchmark",
                "cuda_matmul_allow_tf32",
                "cudnn_allow_tf32",
                "float32_matmul_precision",
                "cublas_workspace_config",
            ):
                expected = recorded_state.get(key)
                observed = current_state.get(key)
                if expected != observed:
                    errors.append(
                        "current torch reproducibility state drifted for "
                        f"{key}: expected={expected!r}, observed={observed!r}"
                    )
    return errors


def _run_config_errors(
    df: pd.DataFrame,
    run_config_json: Path,
    *,
    validate_current_environment: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate partial results against the Stage-2 startup snapshot."""
    payload, read_errors = _load_run_config(run_config_json)
    if read_errors:
        return read_errors, []
    assert payload is not None
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("artifact") != "doe-benchmark-stage2-run-config":
        errors.append(
            "run_config.json artifact marker is not "
            "'doe-benchmark-stage2-run-config'"
        )

    for key in ("created_utc", "output_dir"):
        if not str(payload.get(key, "")).strip():
            errors.append(f"run_config.json missing non-empty {key}")

    for key in ("n_jobs", "bo_n_jobs"):
        value = _positive_int_value(payload.get(key))
        if value is None:
            errors.append(
                f"run_config.json {key} must be a JSON positive integer; "
                f"observed {payload.get(key)!r}"
            )

    if "skip_existing" not in payload or not isinstance(payload.get("skip_existing"), bool):
        errors.append("run_config.json skip_existing must be recorded as a boolean")

    for key in ("timeout_s", "bo_timeout_s", "ref_timeout_s", "watchdog_s"):
        if key not in payload:
            errors.append(f"run_config.json missing {key}")
            continue
        value = payload.get(key)
        if value is None:
            continue
        parsed = _finite_json_number(value)
        if parsed is None or parsed <= 0:
            errors.append(
                f"run_config.json {key} must be null or a finite positive JSON "
                f"number; observed {value!r}"
            )

    for key in ("heartbeat_s", "max_wall_clock_s"):
        if key not in payload:
            errors.append(f"run_config.json missing {key}")
            continue
        value = payload.get(key)
        if value is None:
            continue
        parsed = _finite_json_number(value)
        if parsed is None or parsed < 0:
            errors.append(
                f"run_config.json {key} must be null or a finite non-negative JSON "
                f"number; observed {value!r}"
            )

    if "mem_cap_gb" not in payload:
        errors.append("run_config.json missing mem_cap_gb")
    else:
        mem_cap = payload.get("mem_cap_gb")
        if mem_cap is not None:
            parsed = _finite_json_number(mem_cap)
            if parsed is None or parsed <= 0:
                errors.append(
                    "run_config.json mem_cap_gb must be null or a finite "
                    f"positive JSON number; observed {mem_cap!r}"
                )

    thread_env = payload.get("thread_env")
    if not isinstance(thread_env, dict):
        errors.append("run_config.json thread_env must be an object")
    else:
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            if name not in thread_env:
                errors.append(f"run_config.json thread_env missing {name}")

    errors.extend(_run_config_environment_errors(payload))
    errors.extend(_run_config_temp_storage_errors(payload))
    errors.extend(_torch_reproducibility_errors(payload))

    latest = df.drop_duplicates(GRID_COLUMNS, keep="last")

    result_schema_version = payload.get("result_schema_version")
    if result_schema_version != RESULT_SCHEMA_VERSION:
        errors.append(
            "run_config.json result_schema_version mismatch: "
            f"expected {RESULT_SCHEMA_VERSION}, observed {result_schema_version!r}"
        )

    for key, observed_values, allow_zero in (
        ("n_test", sorted(pd.to_numeric(latest.get("n_test"), errors="coerce").dropna().astype(int).unique()) if "n_test" in latest else [], False),
        ("test_seed", sorted(pd.to_numeric(latest.get("test_seed"), errors="coerce").dropna().astype(int).unique()) if "test_seed" in latest else [], True),
        ("seed_start", sorted(pd.to_numeric(latest.get("seed_start"), errors="coerce").dropna().astype(int).unique()) if "seed_start" in latest else [], True),
    ):
        configured = payload.get(key)
        if configured is None:
            errors.append(f"run_config.json missing {key}")
            continue
        parsed_configured = (
            _non_negative_int_value(configured)
            if allow_zero
            else _positive_int_value(configured)
        )
        if parsed_configured is None:
            requirement = "a non-negative integer" if allow_zero else "a positive integer"
            errors.append(
                f"run_config.json {key} must be {requirement}; "
                f"observed {configured!r}"
            )
            continue
        if observed_values and observed_values != [parsed_configured]:
            errors.append(
                f"results.csv {key} values {observed_values} do not match "
                f"run_config.json {key}={configured!r}"
            )

    configured_seed_strategy = str(payload.get("seed_strategy", "")).strip()
    if not configured_seed_strategy:
        errors.append("run_config.json missing seed_strategy")
    elif "seed_strategy" in latest.columns:
        observed_seed_strategy = sorted(
            value
            for value in latest["seed_strategy"].dropna().astype(str).str.strip().unique()
            if value
        )
        if observed_seed_strategy and observed_seed_strategy != [configured_seed_strategy]:
            errors.append(
                "results.csv seed_strategy values "
                f"{observed_seed_strategy} do not match run_config.json "
                f"seed_strategy={configured_seed_strategy!r}"
            )

    configured_budgets: list[int] = []
    budgets = payload.get("budgets")
    if not isinstance(budgets, list) or not budgets:
        errors.append("run_config.json missing non-empty budgets")
    else:
        parsed_budgets = [_positive_int_value(value) for value in budgets]
        if any(value is None for value in parsed_budgets):
            errors.append(
                "run_config.json budgets must be positive integers; "
                f"observed {budgets!r}"
            )
            configured_budgets = []
        else:
            configured_budgets = sorted(int(value) for value in parsed_budgets)
            if len(configured_budgets) != len(set(configured_budgets)):
                errors.append(
                    "run_config.json budgets must not contain duplicates; "
                    f"observed {budgets!r}"
                )
        if "budget_mult" in latest.columns:
            observed_budgets = sorted(
                pd.to_numeric(latest["budget_mult"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            extra = sorted(set(observed_budgets) - set(configured_budgets))
            if extra:
                errors.append(
                    "results.csv contains budget_mult values outside "
                    f"run_config.json: {extra}"
                )

    configured_repeats = payload.get("n_repeats")
    parsed_repeats = _positive_int_value(configured_repeats)
    if parsed_repeats is None:
        errors.append(
            "run_config.json n_repeats must be a positive integer; "
            f"observed {configured_repeats!r}"
        )
    elif "repeat" in latest.columns:
        repeat_values = (
            pd.to_numeric(latest["repeat"], errors="coerce").dropna().astype(int)
        )
        bad_repeats = sorted(
            int(value)
            for value in repeat_values.unique()
            if int(value) < 0 or int(value) >= parsed_repeats
        )
        if bad_repeats:
            errors.append(
                "results.csv contains repeat values outside run_config.json "
                f"n_repeats={configured_repeats!r}: {bad_repeats}"
            )

    cases = payload.get("cases")
    configured_cases = _string_set(cases, key="case_id", field="case_source_sha256")
    if configured_cases is None:
        errors.append("run_config.json has malformed cases entries")
    elif "case_id" in latest.columns:
        observed_cases = {
            value
            for value in latest["case_id"].dropna().astype(str).str.strip().unique()
            if value
        }
        extra_cases = sorted(observed_cases - configured_cases)
        if extra_cases:
            errors.append(
                "results.csv contains case_id values outside run_config.json: "
                f"{extra_cases}"
            )
        case_by_id = {
            str(item["case_id"]): item
            for item in cases or []
            if isinstance(item, dict) and str(item.get("case_id", "")).strip()
        }
        for case_id in sorted(observed_cases & configured_cases):
            sub = latest[latest["case_id"].astype(str) == case_id]
            configured = case_by_id[case_id]
            for col, config_key in (
                ("case_source_path", "case_source_path"),
                ("case_source_sha256", "case_source_sha256"),
                ("domain", "domain"),
                ("dim", "dim"),
                ("n_output", "n_output"),
            ):
                if col not in sub.columns:
                    continue
                observed = {
                    str(value).strip()
                    for value in sub[col].dropna().unique()
                    if str(value).strip()
                }
                expected = str(configured.get(config_key, "")).strip()
                if observed and observed != {expected}:
                    errors.append(
                        f"results.csv {col} for {case_id} does not match "
                        f"run_config.json: observed={sorted(observed)}, "
                        f"expected={expected!r}"
                    )

    methods = payload.get("methods")
    configured_methods = _string_set(methods, key="method", field="method_source_sha256")
    if configured_methods is None:
        errors.append("run_config.json has malformed methods entries")
    elif "method" in latest.columns:
        observed_methods = {
            value
            for value in latest["method"].dropna().astype(str).str.strip().unique()
            if value
        }
        extra_methods = sorted(observed_methods - configured_methods)
        if extra_methods:
            errors.append(
                "results.csv contains method values outside run_config.json: "
                f"{extra_methods}"
            )
        method_by_name = {
            str(item["method"]): item
            for item in methods or []
            if isinstance(item, dict) and str(item.get("method", "")).strip()
        }
        for method in sorted(observed_methods & configured_methods):
            sub = latest[latest["method"].astype(str) == method]
            configured = method_by_name[method]
            for col, config_key in (
                ("method_family", "method_family"),
                ("method_source_path", "method_source_path"),
                ("method_source_sha256", "method_source_sha256"),
            ):
                if col not in sub.columns:
                    continue
                observed = {
                    str(value).strip()
                    for value in sub[col].dropna().unique()
                    if str(value).strip()
                }
                expected = str(configured.get(config_key, "")).strip()
                if observed and observed != {expected}:
                    errors.append(
                        f"results.csv {col} for {method} does not match "
                        f"run_config.json: observed={sorted(observed)}, "
                        f"expected={expected!r}"
                    )

    total_cells = payload.get("total_cells")
    parsed_total_cells = _positive_int_value(total_cells)
    if parsed_total_cells is None:
        errors.append(
            "run_config.json total_cells must be a positive integer; "
            f"observed {total_cells!r}"
        )
    elif (
        isinstance(budgets, list)
        and budgets
        and all(value is not None for value in [_positive_int_value(item) for item in budgets])
        and configured_cases is not None
        and configured_methods is not None
        and parsed_repeats is not None
    ):
        parsed_budget_values = [
            int(_positive_int_value(value))
            for value in budgets
            if _positive_int_value(value) is not None
        ]
        expected_total = (
            len(configured_cases)
            * len(configured_methods)
            * len(parsed_budget_values)
            * parsed_repeats
        )
        if parsed_total_cells != expected_total:
            errors.append(
                "run_config.json total_cells does not match cases×methods×"
                f"budgets×repeats: observed={total_cells!r}, expected={expected_total}"
            )
        unique_cells = int(latest[GRID_COLUMNS].drop_duplicates().shape[0])
        if unique_cells > parsed_total_cells:
            errors.append(
                f"results.csv has {unique_cells} unique cells, exceeding "
                f"run_config.json total_cells={total_cells!r}"
            )

    if validate_current_environment:
        errors.extend(_current_environment_drift_errors(payload))

    if not errors:
        warnings.append(
            f"run_config.json contract matched partial results: {run_config_json}"
        )
        warnings.append(
            f"{RUN_CONFIG_RUNTIME_CONTROLS_VERDICT}: {run_config_json}"
        )
        warnings.append(
            f"{RUN_CONFIG_ENVIRONMENT_VERDICT}: {run_config_json}"
        )
        warnings.append(
            f"{RUN_CONFIG_TORCH_REPRO_VERDICT}: {run_config_json}"
        )
        if validate_current_environment:
            warnings.append(
                f"{CURRENT_ENVIRONMENT_VERDICT}: {run_config_json}"
            )
    return errors, warnings


def _empty_runtime_contract_frame() -> pd.DataFrame:
    """Return an empty frame with enough columns for run-config-only checks."""
    columns = sorted(RUNTIME_CONTRACT_COLUMNS | set(GRID_COLUMNS))
    return pd.DataFrame(columns=columns)


def _resolved_run_config_path(
    results_csv: Path,
    run_config_json: Path | None,
) -> Path | None:
    if run_config_json is not None:
        return run_config_json
    candidate = results_csv.with_name("run_config.json")
    return candidate if candidate.exists() else None


def _startup_run_config_check(
    results_csv: Path,
    run_config_json: Path | None,
    *,
    require_run_config: bool,
    validate_current_environment: bool,
) -> tuple[list[str], list[str]]:
    """Validate Stage-2 startup metadata before the first streamed row exists."""
    if run_config_json is not None:
        resolved = run_config_json
    elif require_run_config:
        resolved = results_csv.with_name("run_config.json")
    else:
        resolved = None
    if resolved is None:
        return [], []
    if not resolved.exists():
        return [
            "run_config.json is required for this runtime-health check but "
            f"was not found next to results.csv: {results_csv.with_name('run_config.json')}"
        ], []
    return _run_config_errors(
        _empty_runtime_contract_frame(),
        resolved,
        validate_current_environment=validate_current_environment,
    )


def _case_metadata_mismatch_count(df: pd.DataFrame) -> int:
    checks = {
        "domain": lambda meta: str(meta.domain),
        "dim": lambda meta: int(meta.dim_input),
        "n_output": lambda meta: int(meta.dim_output),
    }
    n_mismatches = 0
    for case_id in sorted(df["case_id"].dropna().astype(str).unique()):
        try:
            meta = db.get_case_meta(case_id)
        except Exception:
            continue
        sub = df[df["case_id"].astype(str) == case_id]
        for col, expected_fn in checks.items():
            if col not in sub.columns:
                continue
            expected = expected_fn(meta)
            if col in {"dim", "n_output"}:
                numeric = pd.to_numeric(sub[col], errors="coerce")
                mismatch = bool(numeric.isna().any()) or bool(
                    numeric.dropna().astype(float).ne(float(expected)).any()
                )
            else:
                observed = {
                    str(value)
                    for value in sub[col].dropna().unique()
                    if str(value).strip()
                }
                mismatch = observed != {str(expected)}
            if mismatch:
                n_mismatches += 1
    return n_mismatches


def _source_field_mismatch_count(
    sub: pd.DataFrame,
    *,
    expected: dict[str, str],
    path_col: str,
    sha_col: str,
) -> int:
    def _normalize_path(value: object) -> str:
        text = str(value).strip().replace("\\", "/")
        if not text:
            return ""
        try:
            path = Path(text)
            if path.is_absolute():
                try:
                    return path.resolve().relative_to(ROOT).as_posix()
                except (OSError, ValueError):
                    return path.as_posix()
        except (OSError, ValueError):
            pass
        return text

    observed_paths = {
        _normalize_path(value)
        for value in sub[path_col].dropna().unique()
        if str(value).strip()
    }
    observed_hashes = {
        str(value)
        for value in sub[sha_col].dropna().unique()
        if str(value).strip()
    }
    expected_path = _normalize_path(expected[path_col])
    expected_hash = str(expected[sha_col])
    path_matches = bool(observed_paths) and all(
        observed == expected_path or observed.endswith(f"/{expected_path}")
        for observed in observed_paths
    )
    return int(not path_matches or observed_hashes != {expected_hash})


def _case_source_mismatch_count(df: pd.DataFrame) -> int:
    n_mismatches = 0
    for case_id in sorted(df["case_id"].dropna().astype(str).unique()):
        try:
            expected = case_source_fingerprint(db.get_case(case_id))
        except Exception:
            continue
        sub = df[df["case_id"].astype(str) == case_id]
        n_mismatches += _source_field_mismatch_count(
            sub,
            expected=expected,
            path_col="case_source_path",
            sha_col="case_source_sha256",
        )
    return n_mismatches


def _method_source_mismatch_count(df: pd.DataFrame) -> int:
    n_mismatches = 0
    expected_by_method = _prereg_method_fingerprints()
    for method in sorted(df["method"].dropna().astype(str).unique()):
        expected = expected_by_method.get(method)
        if expected is None:
            continue
        sub = df[df["method"].astype(str) == method]
        n_mismatches += _source_field_mismatch_count(
            sub,
            expected=expected,
            path_col="method_source_path",
            sha_col="method_source_sha256",
        )
    return n_mismatches


def _method_family_mismatch_count(df: pd.DataFrame) -> int:
    n_mismatches = 0
    for method in sorted(df["method"].dropna().astype(str).unique()):
        expected = PREREG_METHOD_FAMILIES.get(method)
        if expected is None:
            continue
        sub = df[df["method"].astype(str) == method]
        observed = {
            str(value)
            for value in sub["method_family"].dropna().unique()
            if str(value).strip()
        }
        if observed != {str(expected)}:
            n_mismatches += 1
    return n_mismatches


def _runtime_option_errors(
    *,
    stale_after_min: float,
    max_timeout_fraction: float,
    max_failed_fraction: float,
    min_cells_for_fraction_gate: int,
    expected_n_test: int,
    expected_test_seed: int,
    seed_strategy: str,
    seed_start: int,
    mem_cap_gb: float | None,
    mem_warning_fraction: float,
    disk_warning_gb: float | None,
) -> list[str]:
    errors: list[str] = []

    stale = _finite_float(stale_after_min)
    if stale is None or stale < 0:
        errors.append(
            "stale_after_min must be a finite non-negative value; "
            f"got {stale_after_min!r}"
        )

    for name, value in (
        ("max_timeout_fraction", max_timeout_fraction),
        ("max_failed_fraction", max_failed_fraction),
    ):
        fraction = _finite_float(value)
        if fraction is None or fraction < 0 or fraction > 1:
            errors.append(
                f"{name} must be a finite fraction between 0 and 1; got {value!r}"
            )

    if isinstance(min_cells_for_fraction_gate, bool):
        parsed_cells = None
    else:
        try:
            parsed_cells = int(min_cells_for_fraction_gate)
        except (TypeError, ValueError):
            parsed_cells = None
    if parsed_cells is None or parsed_cells < 1:
        errors.append(
            "min_cells_for_fraction_gate must be a positive integer; "
            f"got {min_cells_for_fraction_gate!r}"
        )

    for name, value in (
        ("expected_n_test", expected_n_test),
        ("expected_test_seed", expected_test_seed),
        ("seed_start", seed_start),
    ):
        if isinstance(value, bool):
            parsed = None
        else:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = None
        if parsed is None or parsed < 0:
            errors.append(f"{name} must be a non-negative integer; got {value!r}")

    if seed_strategy not in {"case_repeat", "legacy"}:
        errors.append(
            "seed_strategy must be 'case_repeat' or 'legacy'; "
            f"got {seed_strategy!r}"
        )

    if mem_cap_gb is not None:
        cap = _finite_float(mem_cap_gb)
        if cap is None or cap <= 0:
            errors.append(
                "mem_cap_gb must be a finite positive value when set; "
                f"got {mem_cap_gb!r}"
            )

    warning_fraction = _finite_float(mem_warning_fraction)
    if warning_fraction is None or warning_fraction <= 0 or warning_fraction > 1:
        errors.append(
            "mem_warning_fraction must be a finite fraction in (0, 1]; "
            f"got {mem_warning_fraction!r}"
        )

    if disk_warning_gb is not None:
        disk_threshold = _finite_float(disk_warning_gb)
        if disk_threshold is None or disk_threshold < 0:
            errors.append(
                "disk_warning_gb must be a finite non-negative value when set; "
                f"got {disk_warning_gb!r}"
            )

    return errors


def _disk_space_warnings(results_csv: Path, disk_warning_gb: float | None) -> list[str]:
    if disk_warning_gb is None:
        return []
    threshold = _finite_float(disk_warning_gb)
    if threshold is None:
        return []
    warnings: list[str] = []
    seen_paths: set[str] = set()

    def nearest_existing(path: Path) -> Path:
        current = path
        while not current.exists() and current != current.parent:
            current = current.parent
        return current

    def check(label: str, path: Path) -> None:
        check_path = nearest_existing(path)
        key = check_path.as_posix()
        if key in seen_paths:
            return
        seen_paths.add(key)
        try:
            usage = shutil.disk_usage(check_path)
        except Exception as exc:
            warnings.append(
                "could not determine free disk space while checking runtime health "
                f"for {label} at {check_path}: {type(exc).__name__}: {exc}"
            )
            return
        free_gb = float(usage.free) / 1024**3
        if free_gb < threshold:
            warnings.append(
                f"free disk space near {label} is below the runtime warning "
                f"threshold: path={check_path}, free={free_gb:.2f} GB, "
                f"threshold={threshold:.2f} GB. Pause/resume on a larger filesystem "
                "or clean non-headline artifacts before storage exhaustion corrupts "
                "the next streamed row."
            )

    check("results.csv", results_csv.parent)
    for label, path in _temp_storage_candidates():
        check(f"temp/scratch ({label})", path)
    return warnings


def _protocol_field_errors(df: pd.DataFrame) -> list[str]:
    """Return field-specific protocol-grid errors for operator diagnostics."""
    errors: list[str] = []
    if not set(GRID_COLUMNS).issubset(df.columns):
        return errors

    case_text = df["case_id"].fillna("").astype(str).str.strip()
    blank_cases = int(case_text.eq("").sum())
    extra_cases = sorted(set(case_text) - PREREG_CASES - {""})
    if blank_cases or extra_cases:
        detail = []
        if blank_cases:
            detail.append(f"blank={blank_cases}")
        if extra_cases:
            detail.append(f"outside={extra_cases[:20]}")
        errors.append(
            "results.csv has case_id values outside the preregistered suite: "
            + ", ".join(detail)
        )

    method_text = df["method"].fillna("").astype(str).str.strip()
    blank_methods = int(method_text.eq("").sum())
    extra_methods = sorted(set(method_text) - PREREG_METHODS - {""})
    if blank_methods or extra_methods:
        detail = []
        if blank_methods:
            detail.append(f"blank={blank_methods}")
        if extra_methods:
            detail.append(f"outside={extra_methods[:20]}")
        errors.append(
            "results.csv has method values outside the frozen nature_pool() protocol: "
            + ", ".join(detail)
        )

    budget, valid_budget = _integer_series(df["budget_mult"])
    invalid_budget_shape = int((~valid_budget).sum())
    observed_budgets = {
        int(value)
        for value in budget.loc[valid_budget].dropna().unique()
    }
    extra_budgets = sorted(observed_budgets - PREREG_BUDGETS)
    if invalid_budget_shape or extra_budgets:
        detail = []
        if invalid_budget_shape:
            detail.append(f"non_integer_or_blank={invalid_budget_shape}")
        if extra_budgets:
            detail.append(f"outside={extra_budgets}")
        errors.append(
            "results.csv has budget_mult values outside the preregistered set "
            f"{sorted(PREREG_BUDGETS)}: "
            + ", ".join(detail)
        )

    repeat, valid_repeat = _integer_series(df["repeat"])
    invalid_repeat_shape = int((~valid_repeat).sum())
    out_of_range_repeat = (
        valid_repeat
        & (repeat.notna())
        & ((repeat < 0) | (repeat >= PREREG_REPS))
    )
    extra_repeats = sorted(
        int(value)
        for value in repeat.loc[out_of_range_repeat].dropna().unique()
    )
    if invalid_repeat_shape or extra_repeats:
        detail = []
        if invalid_repeat_shape:
            detail.append(f"non_integer_or_blank={invalid_repeat_shape}")
        if extra_repeats:
            detail.append(f"outside={extra_repeats}")
        errors.append(
            "results.csv has repeat values outside the preregistered range "
            f"[0, {PREREG_REPS - 1}]: "
            + ", ".join(detail)
        )

    return errors


def check_runtime_health(
    results_csv: Path = DEFAULT_RESULTS_CSV,
    *,
    run_config_json: Path | None = None,
    require_run_config: bool = False,
    validate_current_environment: bool = False,
    stale_after_min: float = 30.0,
    max_timeout_fraction: float = 0.05,
    max_failed_fraction: float = 0.01,
    min_cells_for_fraction_gate: int = 100,
    expected_n_test: int = PREREG_N_TEST,
    expected_test_seed: int = PREREG_TEST_SEED,
    seed_strategy: str = PREREG_SEED_STRATEGY,
    seed_start: int = PREREG_SEED_START,
    mem_cap_gb: float | None = None,
    mem_warning_fraction: float = 0.90,
    disk_warning_gb: float | None = None,
    allow_missing: bool = False,
    fail_on_stale: bool = False,
) -> tuple[int, list[str], list[str]]:
    """Return ``(exit_code, errors, warnings)`` for the current partial sweep."""
    option_errors = _runtime_option_errors(
        stale_after_min=stale_after_min,
        max_timeout_fraction=max_timeout_fraction,
        max_failed_fraction=max_failed_fraction,
        min_cells_for_fraction_gate=min_cells_for_fraction_gate,
        expected_n_test=expected_n_test,
        expected_test_seed=expected_test_seed,
        seed_strategy=str(seed_strategy),
        seed_start=seed_start,
        mem_cap_gb=mem_cap_gb,
        mem_warning_fraction=mem_warning_fraction,
        disk_warning_gb=disk_warning_gb,
    )
    if option_errors:
        return 1, option_errors, []

    df, read_errors = _read_results_csv(results_csv)
    if read_errors:
        if allow_missing and not results_csv.exists():
            config_errors, config_warnings = _startup_run_config_check(
                results_csv,
                run_config_json,
                require_run_config=require_run_config,
                validate_current_environment=validate_current_environment,
            )
            warnings = list(read_errors)
            warnings.extend(config_warnings)
            return (1 if config_errors else 0), config_errors, warnings
        return 1, read_errors, []
    assert df is not None
    if df.empty:
        message = (
            "results.csv has a header but no data rows yet; wait for the first "
            "streamed cell or inspect the writer before treating the sweep as healthy"
        )
        mtime = datetime.fromtimestamp(results_csv.stat().st_mtime)
        age_min = (datetime.now() - mtime).total_seconds() / 60.0
        if age_min > stale_after_min:
            stale_message = (
                f"{message}; last update was {age_min:.1f} min ago "
                f"(threshold {stale_after_min:.1f} min)"
            )
            return 1, [stale_message], []
        if allow_missing:
            config_errors, config_warnings = _startup_run_config_check(
                results_csv,
                run_config_json,
                require_run_config=require_run_config,
                validate_current_environment=validate_current_environment,
            )
            warnings = [message]
            warnings.extend(config_warnings)
            return (1 if config_errors else 0), config_errors, warnings
        return 1, [message], []

    errors: list[str] = []
    warnings: list[str] = []
    now = datetime.now()
    mtime = datetime.fromtimestamp(results_csv.stat().st_mtime)
    age_min = (now - mtime).total_seconds() / 60.0
    if age_min > stale_after_min:
        stale_message = (
            f"results.csv has not been updated for {age_min:.1f} min "
            f"(threshold {stale_after_min:.1f} min)"
        )
        if fail_on_stale:
            errors.append(stale_message)
        else:
            warnings.append(stale_message)
    warnings.extend(_disk_space_warnings(results_csv, disk_warning_gb))

    missing_columns = sorted(set(GRID_COLUMNS + ["status"]) - set(df.columns))
    if missing_columns:
        errors.append(f"results.csv missing required runtime columns: {missing_columns}")
        return 1, errors, warnings

    missing_contract_columns = sorted(RUNTIME_CONTRACT_COLUMNS - set(df.columns))
    if missing_contract_columns:
        errors.append(
            "results.csv missing required runtime contract columns for safe "
            f"resume/monitoring: {missing_contract_columns}"
        )

    summary = summarize_progress(df)
    if int(summary["out_of_protocol_rows"]) > 0:
        errors.append(
            f"results.csv has {summary['out_of_protocol_rows']} row(s) outside "
            "the preregistered case/method/budget/repeat grid"
        )
        errors.extend(_protocol_field_errors(df))
    if int(summary["duplicate_protocol_rows"]) > 0:
        errors.append(
            f"results.csv has {summary['duplicate_protocol_rows']} duplicate "
            "protocol row(s); stop and repair the streamed CSV before resuming"
        )
    status_text = df["status"].fillna("").astype(str).str.strip()
    missing_status = status_text.eq("")
    if bool(missing_status.any()):
        errors.append(
            f"results.csv has missing/blank status values: {int(missing_status.sum())}"
        )
    raw_statuses = set(status_text)
    invalid_statuses = sorted(raw_statuses - VALID_STATUSES - {""})
    if invalid_statuses:
        errors.append(f"results.csv has invalid status values: {invalid_statuses}")

    status_counts = dict(summary["status_counts_unique"])
    unique_cells = int(summary["protocol_unique_cells"])
    if unique_cells > PREREG_TARGET:
        errors.append(
            f"protocol unique cells {unique_cells} exceeds target {PREREG_TARGET}"
        )
    if unique_cells >= int(min_cells_for_fraction_gate):
        timeout_fraction = status_counts.get("timeout", 0) / unique_cells
        failed_fraction = status_counts.get("failed", 0) / unique_cells
        if timeout_fraction > max_timeout_fraction:
            errors.append(
                f"timeout fraction {timeout_fraction:.3%} exceeds "
                f"{max_timeout_fraction:.3%}"
            )
        if failed_fraction > max_failed_fraction:
            errors.append(
                f"failed fraction {failed_fraction:.3%} exceeds "
                f"{max_failed_fraction:.3%}"
            )
    elif unique_cells and (
        status_counts.get("timeout", 0) > 0 or status_counts.get("failed", 0) > 0
    ):
        warnings.append(
            "timeout/failed fraction gate deferred until at least "
            f"{int(min_cells_for_fraction_gate)} protocol cells are observed "
            f"(current={unique_cells}, timeout={status_counts.get('timeout', 0)}, "
            f"failed={status_counts.get('failed', 0)})"
        )

    if "t_total_s" in df.columns:
        bad_time = ~_finite_non_negative(df["t_total_s"])
        if bool(bad_time.any()):
            errors.append(f"results.csv has {int(bad_time.sum())} invalid t_total_s value(s)")
    else:
        warnings.append("results.csv has no t_total_s column yet")

    if "nrmse" in df.columns:
        latest = df.drop_duplicates(GRID_COLUMNS, keep="last")
        status_text = latest["status"].fillna("").astype(str)
        nrmse = pd.to_numeric(latest["nrmse"], errors="coerce")
        bad_ok_nrmse = (
            status_text.eq("ok")
            & (nrmse.isna() | ~nrmse.map(math.isfinite) | (nrmse < 0))
        )
        if bool(bad_ok_nrmse.any()):
            errors.append(
                f"ok latest cells with invalid nrmse={int(bad_ok_nrmse.sum())}"
            )
        nrmse_text = latest["nrmse"].fillna("").astype(str).str.strip()
        non_ok_nrmse = ~status_text.eq("ok") & nrmse_text.ne("")
        if bool(non_ok_nrmse.any()):
            errors.append(
                "timeout/failed latest cells with non-empty nrmse="
                f"{int(non_ok_nrmse.sum())}; failure penalties must be applied "
                "only by the preregistered analysis"
            )
        for metric in OPTIONAL_OK_NUMERIC_METRICS:
            if metric not in latest.columns:
                continue
            values = pd.to_numeric(latest[metric], errors="coerce")
            bad_metric = status_text.eq("ok") & (
                values.isna() | ~values.map(math.isfinite)
            )
            if metric != "r2":
                bad_metric = bad_metric | (status_text.eq("ok") & (values < 0))
            if bool(bad_metric.any()):
                errors.append(
                    f"ok latest cells with invalid {metric}="
                    f"{int(bad_metric.sum())}"
                )
            metric_text = latest[metric].fillna("").astype(str).str.strip()
            non_ok_metric = ~status_text.eq("ok") & metric_text.ne("")
            if bool(non_ok_metric.any()):
                errors.append(
                    f"timeout/failed latest cells with non-empty {metric}="
                    f"{int(non_ok_metric.sum())}; failure penalties must be "
                    "applied only by the preregistered analysis"
                )
    else:
        warnings.append("results.csv has no nrmse column yet")

    if "error" in df.columns:
        latest = df.drop_duplicates(GRID_COLUMNS, keep="last")
        status_text = latest["status"].fillna("").astype(str)
        error_text = latest["error"].fillna("").astype(str).str.strip()
        missing_error = status_text.isin({"timeout", "failed"}) & error_text.eq("")
        ok_with_error = status_text.eq("ok") & error_text.ne("")
        if bool(missing_error.any()):
            errors.append(
                f"timeout/failed latest cells with empty error={int(missing_error.sum())}"
            )
        if bool(ok_with_error.any()):
            errors.append(f"ok latest cells with non-empty error={int(ok_with_error.sum())}")
    else:
        warnings.append("results.csv has no error column yet")

    if {"status", "error", "failure_category"}.issubset(df.columns):
        latest = df.drop_duplicates(GRID_COLUMNS, keep="last")
        category_text = latest["failure_category"].fillna("").astype(str).str.strip()
        invalid_category = category_text.eq("") | ~category_text.isin(
            ALLOWED_FAILURE_CATEGORIES
        )
        if bool(invalid_category.any()):
            errors.append(
                "latest cells with missing/invalid failure_category: "
                f"{int(invalid_category.sum())}"
            )
        category_mismatch = pd.Series(False, index=latest.index)
        for idx in latest.index:
            if not failure_category_is_consistent(
                latest.at[idx, "status"],
                latest.at[idx, "error"],
                latest.at[idx, "failure_category"],
            ):
                category_mismatch.loc[idx] = True
        if bool(category_mismatch.any()):
            errors.append(
                "latest cells with failure_category inconsistent with status/error: "
                f"{int(category_mismatch.sum())}"
            )
    else:
        missing_category_cols = sorted(
            {"status", "error", "failure_category"} - set(df.columns)
        )
        warnings.append(
            "results.csv cannot validate failure-category taxonomy; missing "
            f"{missing_category_cols}"
        )

    latest = df.drop_duplicates(GRID_COLUMNS, keep="last")
    protocol_columns = {
        "result_schema_version",
        "seed_strategy",
        "seed_start",
        "protocol_digest",
    }
    if protocol_columns.issubset(latest.columns):
        schema_version, valid_schema_version = _integer_series(
            latest["result_schema_version"]
        )
        bad_schema_version = (
            ~valid_schema_version
            | schema_version.ne(int(RESULT_SCHEMA_VERSION))
        )
        if bool(bad_schema_version.any()):
            errors.append(
                "latest cells with result_schema_version != "
                f"{int(RESULT_SCHEMA_VERSION)}: {int(bad_schema_version.sum())}"
            )
        seed_strategy_text = (
            latest["seed_strategy"].fillna("").astype(str).str.strip()
        )
        bad_seed_strategy = seed_strategy_text.ne(str(seed_strategy))
        if bool(bad_seed_strategy.any()):
            errors.append(
                "latest cells with seed_strategy != "
                f"{str(seed_strategy)!r}: {int(bad_seed_strategy.sum())}"
            )
        seed_start_value, valid_seed_start = _integer_series(latest["seed_start"])
        bad_seed_start = (
            ~valid_seed_start | seed_start_value.ne(int(seed_start))
        )
        if bool(bad_seed_start.any()):
            errors.append(
                "latest cells with seed_start != "
                f"{int(seed_start)}: {int(bad_seed_start.sum())}"
            )
        bad_digest = invalid_protocol_digest_mask(latest)
        if bool(bad_digest.any()):
            errors.append(
                "latest cells with protocol_digest inconsistent with row-level "
                f"protocol fields: {int(bad_digest.sum())}"
            )
    else:
        warnings.append(
            "results.csv cannot validate row-level protocol digests; missing "
            f"{sorted(protocol_columns - set(latest.columns))}"
        )

    case_source_columns = {"case_source_path", "case_source_sha256"}
    if case_source_columns.issubset(latest.columns):
        n_case_source_mismatches = _case_source_mismatch_count(latest)
        if n_case_source_mismatches:
            errors.append(
                "latest cells with case source fingerprints mismatching the current "
                f"registry: {n_case_source_mismatches}"
            )
    else:
        warnings.append(
            "results.csv cannot validate case source fingerprint drift; missing "
            f"{sorted(case_source_columns - set(latest.columns))}"
        )

    method_source_columns = {"method_source_path", "method_source_sha256"}
    if method_source_columns.issubset(latest.columns):
        n_method_source_mismatches = _method_source_mismatch_count(latest)
        if n_method_source_mismatches:
            errors.append(
                "latest cells with method source fingerprints mismatching the current "
                f"nature_pool registry: {n_method_source_mismatches}"
            )
    else:
        warnings.append(
            "results.csv cannot validate method source fingerprint drift; missing "
            f"{sorted(method_source_columns - set(latest.columns))}"
        )

    if "method_family" in latest.columns:
        n_family_mismatches = _method_family_mismatch_count(latest)
        if n_family_mismatches:
            errors.append(
                "latest cells with method_family values mismatching the current "
                f"nature_pool registry: {n_family_mismatches}"
            )
    else:
        warnings.append("results.csv cannot validate method_family drift; missing ['method_family']")

    metadata_columns = {"domain", "dim", "n_output"}
    if metadata_columns.issubset(latest.columns):
        n_metadata_mismatches = _case_metadata_mismatch_count(latest)
        if n_metadata_mismatches:
            errors.append(
                "latest cells with case metadata columns mismatching the current "
                f"registry: {n_metadata_mismatches}"
            )
    else:
        missing_metadata_cols = sorted(metadata_columns - set(latest.columns))
        warnings.append(
            "results.csv cannot validate case metadata drift; missing "
            f"{missing_metadata_cols}"
        )

    if {"budget_mult", "dim", "n_train"}.issubset(latest.columns):
        budget_mult, valid_budget = _integer_series(latest["budget_mult"])
        dim, valid_dim = _integer_series(latest["dim"])
        n_train, valid_n_train = _integer_series(latest["n_train"])
        bad_n_train = (
            ~valid_budget
            | ~valid_dim
            | ~valid_n_train
            | budget_mult.le(0)
            | dim.le(0)
            | n_train.ne(budget_mult * dim)
        )
        if bool(bad_n_train.any()):
            errors.append(
                "latest cells with n_train != budget_mult * dim or non-positive "
                f"integer budget/dim/train fields: {int(bad_n_train.sum())}"
            )
    else:
        missing_train_cols = sorted(
            set(["budget_mult", "dim", "n_train"]) - set(latest.columns)
        )
        warnings.append(
            "results.csv cannot validate n_train budget contract; missing "
            f"{missing_train_cols}"
        )

    if "n_test" in latest.columns:
        n_test, valid_n_test = _integer_series(latest["n_test"])
        bad_n_test = ~valid_n_test | n_test.ne(int(expected_n_test))
        if bool(bad_n_test.any()):
            errors.append(
                f"latest cells with n_test != {int(expected_n_test)}: {int(bad_n_test.sum())}"
            )
    else:
        warnings.append("results.csv has no n_test column yet")

    if "test_seed" in latest.columns:
        test_seed, valid_test_seed = _integer_series(latest["test_seed"])
        bad_test_seed = ~valid_test_seed | test_seed.ne(int(expected_test_seed))
        if bool(bad_test_seed.any()):
            errors.append(
                f"latest cells with test_seed != {int(expected_test_seed)}: "
                f"{int(bad_test_seed.sum())}"
            )
    else:
        warnings.append("results.csv has no test_seed column yet")

    if "seed" in latest.columns:
        seed, valid_seed = _integer_series(latest["seed"])
        repeat, valid_repeat = _integer_series(latest["repeat"])
        bad_seed = ~valid_seed | ~valid_repeat
        for idx in latest.index[valid_seed & valid_repeat]:
            expected = derive_seed(
                str(latest.at[idx, "case_id"]),
                int(repeat.at[idx]),
                strategy=str(seed_strategy),
                seed_start=int(seed_start),
            )
            if int(seed.at[idx]) != expected:
                bad_seed.loc[idx] = True
        if bool(bad_seed.any()):
            errors.append(
                "latest cells with seed not matching "
                f"{str(seed_strategy)} seed_start={int(seed_start)}: "
                f"{int(bad_seed.sum())}"
            )
    else:
        warnings.append("results.csv has no seed column yet")

    if RUNTIME_TRACE_COLUMNS.issubset(latest.columns):
        host_text = latest["worker_host"].fillna("").astype(str).str.strip()
        bad_host = host_text.eq("")
        if bool(bad_host.any()):
            errors.append(
                f"latest cells with empty worker_host: {int(bad_host.sum())}"
            )

        worker_pid, valid_worker_pid = _integer_series(latest["worker_pid"])
        bad_pid = ~valid_worker_pid | worker_pid.le(0)
        if bool(bad_pid.any()):
            errors.append(
                "latest cells with invalid positive-integer worker_pid: "
                f"{int(bad_pid.sum())}"
            )

        bad_started = ~_valid_utc_timestamp_series(latest["cell_started_utc"])
        bad_finished = ~_valid_utc_timestamp_series(latest["cell_finished_utc"])
        if bool(bad_started.any()):
            errors.append(
                "latest cells with invalid cell_started_utc timestamps: "
                f"{int(bad_started.sum())}"
            )
        if bool(bad_finished.any()):
            errors.append(
                "latest cells with invalid cell_finished_utc timestamps: "
                f"{int(bad_finished.sum())}"
            )
        parsed_started = pd.to_datetime(
            latest["cell_started_utc"].fillna("").astype(str).str.strip(),
            errors="coerce",
            utc=True,
        )
        parsed_finished = pd.to_datetime(
            latest["cell_finished_utc"].fillna("").astype(str).str.strip(),
            errors="coerce",
            utc=True,
        )
        bad_time_order = parsed_finished < parsed_started
        if bool(bad_time_order.any()):
            errors.append(
                "latest cells with cell_finished_utc earlier than "
                f"cell_started_utc: {int(bad_time_order.sum())}"
            )
        if "t_total_s" in latest.columns:
            bad_duration = invalid_runtime_duration_mask(
                latest["t_total_s"],
                parsed_started,
                parsed_finished,
            )
            if bool(bad_duration.any()):
                errors.append(
                    "latest cells with t_total_s inconsistent with "
                    "cell_finished_utc - cell_started_utc: "
                    f"{int(bad_duration.sum())}"
                )

        rss_start_ok = _finite_non_negative(latest["rss_gb_start"])
        rss_end_ok = _finite_non_negative(latest["rss_gb_end"])
        rss_peak_ok = _finite_non_negative(latest["rss_gb_peak"])
        if bool((~rss_start_ok).any()):
            errors.append(
                "latest cells with invalid finite non-negative rss_gb_start: "
                f"{int((~rss_start_ok).sum())}"
            )
        if bool((~rss_end_ok).any()):
            errors.append(
                "latest cells with invalid finite non-negative rss_gb_end: "
                f"{int((~rss_end_ok).sum())}"
            )
        if bool((~rss_peak_ok).any()):
            errors.append(
                "latest cells with invalid finite non-negative rss_gb_peak: "
                f"{int((~rss_peak_ok).sum())}"
            )
        rss_start = pd.to_numeric(latest["rss_gb_start"], errors="coerce")
        rss_end = pd.to_numeric(latest["rss_gb_end"], errors="coerce")
        rss_peak = pd.to_numeric(latest["rss_gb_peak"], errors="coerce")
        bad_peak_order = rss_peak_ok & (
            rss_peak.lt(rss_start.fillna(-math.inf))
            | rss_peak.lt(rss_end.fillna(-math.inf))
        )
        if bool(bad_peak_order.any()):
            errors.append(
                "latest cells with rss_gb_peak below start/end RSS: "
                f"{int(bad_peak_order.sum())}"
            )

        cap = _finite_float(mem_cap_gb)
        warn_fraction = _finite_float(mem_warning_fraction)
        if cap is not None and warn_fraction is not None and bool(rss_peak_ok.any()):
            observed_peak = float(rss_peak[rss_peak_ok].max())
            warning_threshold = cap * warn_fraction
            if observed_peak >= warning_threshold:
                warnings.append(
                    "observed rss_gb_peak is close to the configured soft memory "
                    f"cap: max={observed_peak:.2f} GB, threshold="
                    f"{warning_threshold:.2f} GB ({warn_fraction:.0%} of "
                    f"mem_cap_gb={cap:.2f} GB). Consider reducing n_jobs/bo_n_jobs "
                    "or resuming with more memory before an OOM kill corrupts the "
                    "next unfinished cell."
                )

    run_config_json = _resolved_run_config_path(results_csv, run_config_json)
    if run_config_json is not None:
        config_errors, config_warnings = _run_config_errors(
            df,
            run_config_json,
            validate_current_environment=validate_current_environment,
        )
        errors.extend(config_errors)
        warnings.extend(config_warnings)
    elif require_run_config:
        errors.append(
            "run_config.json is required for this runtime-health check but "
            f"was not found next to results.csv: {results_csv.with_name('run_config.json')}"
        )

    return (1 if errors else 0), errors, warnings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_csv", type=Path, nargs="?", default=DEFAULT_RESULTS_CSV)
    parser.add_argument(
        "--run-config-json",
        type=Path,
        default=None,
        help=(
            "Stage-2 run_config.json startup snapshot to validate against "
            "results.csv. Defaults to a sibling run_config.json when present."
        ),
    )
    parser.add_argument(
        "--require-run-config",
        action="store_true",
        help="Fail if no run_config.json startup snapshot can be validated.",
    )
    parser.add_argument(
        "--validate-current-environment",
        action="store_true",
        help=(
            "Also compare the active interpreter, required package versions, "
            "and BLAS thread caps against the recorded run_config.json startup "
            "environment. Recommended on the server before resume/packaging."
        ),
    )
    parser.add_argument("--stale-after-min", type=float, default=30.0)
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help=(
            "Treat a non-empty results.csv whose mtime is older than "
            "--stale-after-min as a failure.  Useful for cron/automation; "
            "default keeps it as a warning for long-running cells."
        ),
    )
    parser.add_argument("--max-timeout-fraction", type=float, default=0.05)
    parser.add_argument("--max-failed-fraction", type=float, default=0.01)
    parser.add_argument("--min-cells-for-fraction-gate", type=int, default=100)
    parser.add_argument(
        "--expected-n-test",
        type=int,
        default=PREREG_N_TEST,
        help=(
            "Expected test-set size recorded in each row. Defaults to the "
            "headline preregistered value; pass the smoke value only when "
            "checking an explicit smoke/diagnostic CSV."
        ),
    )
    parser.add_argument(
        "--expected-test-seed",
        type=int,
        default=PREREG_TEST_SEED,
        help="Expected held-out test-set seed recorded in each row.",
    )
    parser.add_argument(
        "--seed-strategy",
        default=PREREG_SEED_STRATEGY,
        help="Per-cell repeat-seed strategy expected in results.csv.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=PREREG_SEED_START,
        help="Seed-start salt expected for the per-cell repeat seeds.",
    )
    parser.add_argument(
        "--mem-cap-gb",
        type=float,
        default=None,
        help=(
            "Configured soft memory cap for the sweep. When provided, the "
            "health gate warns if observed rss_gb_peak approaches the cap."
        ),
    )
    parser.add_argument(
        "--mem-warning-fraction",
        type=float,
        default=0.90,
        help=(
            "Warn when max rss_gb_peak is at or above this fraction of "
            "--mem-cap-gb."
        ),
    )
    parser.add_argument(
        "--disk-warning-gb",
        type=float,
        default=None,
        help=(
            "Warn when the filesystem containing results.csv has less than "
            "this many GiB free. Useful for long server runs where logs or "
            "partial artifacts can fill the target filesystem."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Return 0 if results.csv has not appeared yet.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rc, errors, warnings = check_runtime_health(
        args.results_csv,
        run_config_json=args.run_config_json,
        require_run_config=bool(args.require_run_config),
        validate_current_environment=bool(args.validate_current_environment),
        stale_after_min=float(args.stale_after_min),
        max_timeout_fraction=float(args.max_timeout_fraction),
        max_failed_fraction=float(args.max_failed_fraction),
        min_cells_for_fraction_gate=int(args.min_cells_for_fraction_gate),
        expected_n_test=int(args.expected_n_test),
        expected_test_seed=int(args.expected_test_seed),
        seed_strategy=str(args.seed_strategy),
        seed_start=int(args.seed_start),
        mem_cap_gb=args.mem_cap_gb if args.mem_cap_gb is not None else None,
        mem_warning_fraction=float(args.mem_warning_fraction),
        disk_warning_gb=(
            float(args.disk_warning_gb)
            if args.disk_warning_gb is not None
            else None
        ),
        allow_missing=bool(args.allow_missing),
        fail_on_stale=bool(args.fail_on_stale),
    )
    for warning in warnings:
        print(f"[warn] {warning}")
    for error in errors:
        print(f"[error] {error}")
    if args.results_csv.exists() and not args.results_csv.is_symlink():
        summary_df, summary_read_errors = _read_results_csv(args.results_csv)
        if summary_df is not None and not summary_df.empty:
            for line in _runtime_summary_lines(summary_df):
                print(line)
        elif summary_read_errors:
            print(
                "[warn] runtime telemetry summaries unavailable because "
                "results.csv could not be parsed"
            )
    if args.results_csv.exists() and not args.results_csv.is_symlink():
        try:
            print(
                "[artifact] "
                f"results_csv={args.results_csv.resolve()} "
                f"size_bytes={args.results_csv.stat().st_size} "
                f"sha256={_sha256_file(args.results_csv)}"
            )
        except OSError as exc:
            print(f"[warn] could not fingerprint results.csv: {type(exc).__name__}: {exc}")
    run_config_artifact = args.run_config_json
    if run_config_artifact is None:
        candidate = args.results_csv.with_name("run_config.json")
        run_config_artifact = candidate if candidate.exists() else None
    if (
        run_config_artifact is not None
        and run_config_artifact.exists()
        and not run_config_artifact.is_symlink()
    ):
        try:
            print(
                "[artifact] "
                f"run_config_json={run_config_artifact.resolve()} "
                f"size_bytes={run_config_artifact.stat().st_size} "
                f"sha256={_sha256_file(run_config_artifact)}"
            )
        except OSError as exc:
            print(
                "[warn] could not fingerprint run_config.json: "
                f"{type(exc).__name__}: {exc}"
            )
    if rc == 0:
        print("[result] PASS - runtime sweep health is acceptable.")
    else:
        print("[result] FAIL - runtime sweep health needs attention.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
