from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import doe_testbed as db
import numpy as np
import pytest
from benchmark.core import case_source_fingerprint, method_source_components, method_source_fingerprint
from benchmark.methods import nature_pool
from benchmark.runtime_contracts import RESULT_SCHEMA_VERSION, protocol_digest_for_row

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_NOTES_ROOT = ROOT / "案例正确性校验"
HAS_VALIDATION_NOTES = VALIDATION_NOTES_ROOT.exists() and any(
    not path.name.startswith("~$") for path in VALIDATION_NOTES_ROOT.rglob("*.docx")
)
requires_validation_notes = pytest.mark.skipif(
    not HAS_VALIDATION_NOTES,
    reason="git-ignored local validation-note files are not present in this checkout",
)
PREREG_CASE_IDS = sorted(case.case_id for case in db.list_cases())
PREREG_METHOD_NAMES = sorted(method.name for method in nature_pool())
CASE_READINESS_TEST_COLUMNS = [
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
]


def _runtime_trace_fields() -> dict[str, object]:
    return {
        "worker_host": "server-a",
        "worker_pid": 12345,
        "cell_started_utc": "2026-06-01T00:00:00+00:00",
        "cell_finished_utc": "2026-06-01T00:00:01+00:00",
        "rss_gb_start": 1.25,
        "rss_gb_end": 1.50,
        "rss_gb_peak": 1.75,
    }


def _runtime_timestamp_summary(n_rows: int = 1) -> dict[str, object]:
    return {
        "timestamp_trace_columns_present": True,
        "missing_timestamp_trace_columns": [],
        "n_rows": n_rows,
        "n_invalid_runtime_timestamp_values": 0,
        "n_invalid_runtime_time_order_values": 0,
        "n_invalid_runtime_duration_values": 0,
    }


def _publication_clean_git() -> dict[str, object]:
    return {
        "commit": "abcdef123456",
        "dirty": False,
        "source_dirty": False,
        "status_porcelain": [],
        "source_dirty_entries": [],
        "derived_dirty_entries": [],
    }


def _publication_machine() -> dict[str, object]:
    return {
        "node": "server-a",
        "platform": "Linux-6.8-x86_64",
        "cpu_count": 64,
        "ram_gb": 256.0,
        "effective_ram_gb": 128.0,
        "effective_ram_source": "cgroup v2 memory.max",
        "cgroup_memory": {
            "schema_version": 1,
            "detected": True,
            "source": "cgroup v2 memory.max",
            "raw": str(128 * 1024**3),
            "limit_gb": 128.0,
            "unlimited": False,
        },
        "resource_limits": {
            "schema_version": 1,
            "status": "ok",
            "limits": {
                "RLIMIT_NOFILE": {"soft": 1048576, "hard": 1048576, "status": "ok"},
                "RLIMIT_AS": {"soft": "unlimited", "hard": "unlimited", "status": "ok"},
            },
        },
        "temp_storage": {
            "schema_version": 1,
            "resolved_path": "/scratch/job_12345/tmp",
            "exists": True,
            "writable": True,
            "free_gb": 512.0,
            "env": {
                "TMPDIR": "/scratch/job_12345/tmp",
                "TMP": None,
                "TEMP": None,
                "TEMPDIR": None,
                "SLURM_TMPDIR": "/scratch/job_12345/tmp",
                "SCRATCH": "/scratch/job_12345",
                "LOCAL_SCRATCH": None,
                "JOBSCRATCH": None,
            },
        },
    }


def _with_result_protocol(
    row: dict[str, object],
    *,
    seed_strategy: str = "case_repeat",
    seed_start: int = 1000,
) -> dict[str, object]:
    out = dict(row)
    out.setdefault("result_schema_version", RESULT_SCHEMA_VERSION)
    out.setdefault("seed_strategy", seed_strategy)
    out.setdefault("seed_start", seed_start)
    out["protocol_digest"] = protocol_digest_for_row(out)
    return out


def _results_frame(rows, *, seed_strategy: str = "case_repeat", seed_start: int = 1000):
    import pandas as pd

    return pd.DataFrame([
        _with_result_protocol(
            dict(row),
            seed_strategy=seed_strategy,
            seed_start=seed_start,
        )
        for row in rows
    ])


def _publication_scheduler() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "slurm",
        "detected": True,
        "env": {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NAME": "doe-benchmark",
            "SLURM_CPUS_PER_TASK": "16",
            "SLURM_MEM_PER_NODE": "256G",
        },
    }


def _publication_accelerator() -> dict[str, object]:
    return {
        "schema_version": 1,
        "cuda_visible_devices": "0",
        "nvidia_smi": {
            "status": "ok",
            "command": (
                "nvidia-smi --query-gpu=name,driver_version,memory.total "
                "--format=csv,noheader"
            ),
            "returncode": 0,
            "stdout": "NVIDIA A100-SXM4-40GB, 550.54.14, 40960 MiB",
            "stderr": "",
        },
        "torch_cuda": {
            "status": "ok",
            "torch_version": "2.5.0",
            "cuda_available": True,
            "torch_cuda_version": "12.4",
            "device_count": 1,
            "devices": [{
                "index": 0,
                "name": "NVIDIA A100-SXM4-40GB",
                "total_memory_gb": 40.0,
                "capability": [8, 0],
            }],
        },
    }


def _publication_container() -> dict[str, object]:
    return {
        "schema_version": 1,
        "detected": False,
        "env": {
            "APPTAINER_CONTAINER": None,
            "SINGULARITY_CONTAINER": None,
            "CONTAINER_IMAGE": None,
            "DOCKER_IMAGE": None,
            "KUBERNETES_SERVICE_HOST": None,
        },
        "marker_files": {
            "/.dockerenv": False,
            "/run/.containerenv": False,
        },
    }


def _publication_python_environment() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "ok",
        "command": "/usr/bin/python -m pip freeze --all",
        "returncode": 0,
        "package_count": 128,
        "pip_freeze_sha256": "b" * 64,
        "pip_freeze_path": "python_environment_pip_freeze.txt",
    }


def _publication_run_config_environment_packages() -> dict[str, dict[str, object]]:
    return {
        package: {"status": "installed", "version": "1.0.0"}
        for package in (
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
    }


def _publication_pip_freeze_contents(
    *,
    overrides: dict[str, str] | None = None,
) -> str:
    packages = {
        name: str(payload["version"])
        for name, payload in _publication_run_config_environment_packages().items()
    }
    if overrides:
        packages.update(overrides)
    return "".join(f"{name}=={version}\n" for name, version in packages.items())


def _write_minimal_publication_run_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    machine = _publication_machine()
    path.write_text(
        json.dumps(
            {
                "environment": {
                    "packages": _publication_run_config_environment_packages(),
                },
                "temp_storage": machine["temp_storage"],
                "torch_reproducibility": _torch_reproducibility_snapshot(),
            }
        ),
        encoding="utf-8",
    )


def _torch_reproducibility_snapshot() -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
        },
        "applied": True,
        "status": "ok",
        "errors": [],
        "state": {
            "torch_version": "2.11.0",
            "cuda_available": False,
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }


def _materialize_publication_python_environment(
    base_dir: Path,
    *,
    contents: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if contents is None:
        contents = _publication_pip_freeze_contents()
    pip_freeze_artifact = base_dir / "python_environment_pip_freeze.txt"
    pip_freeze_artifact.parent.mkdir(parents=True, exist_ok=True)
    pip_freeze_artifact.write_text(contents, encoding="utf-8")
    pip_freeze_sha = _file_sha256(pip_freeze_artifact)
    python_environment = {
        **_publication_python_environment(),
        "pip_freeze_sha256": pip_freeze_sha,
    }
    artifact = {
        "path": "python_environment_pip_freeze.txt",
        "size_bytes": pip_freeze_artifact.stat().st_size,
        "sha256": pip_freeze_sha,
    }
    return python_environment, artifact


def _attach_publication_python_environment_artifact(
    payload: dict[str, object],
    base_dir: Path,
    *,
    contents: str | None = None,
) -> dict[str, object]:
    python_environment, artifact = _materialize_publication_python_environment(
        base_dir,
        contents=contents,
    )
    payload["python_environment"] = python_environment
    artifacts = payload.setdefault("artifacts", {})
    files = artifacts.setdefault("files", [])
    if not any(
        isinstance(item, dict)
        and str(item.get("path", "")).replace("\\", "/") == artifact["path"]
        for item in files
    ):
        files.append(artifact)
    return payload


def _write_manifest_with_publication_python_environment(
    path: Path,
    payload: dict[str, object],
    *,
    contents: str | None = None,
) -> None:
    enriched = _attach_publication_python_environment_artifact(
        dict(payload),
        path.parent,
        contents=contents,
    )
    path.write_text(json.dumps(enriched), encoding="utf-8")


def _publication_run_window() -> dict[str, object]:
    return {
        "source": "artifact_mtime_utc",
        "started_at_utc": "2026-06-01T00:00:00+00:00",
        "finished_at_utc": "2026-06-02T00:00:00+00:00",
        "n_artifacts": 2,
    }


def _decision_manifest_payload(results_sha: str = "a" * 64) -> dict[str, object]:
    return {
        "machine": _publication_machine(),
        "run_window": _publication_run_window(),
        "artifacts": {"files": [{"path": "results.csv", "sha256": results_sha}]},
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _publication_experiment(commands: list[str] | None = None) -> dict[str, object]:
    if commands is None:
        commands = ["python scripts/run_pipeline.py --profile full --confirm-full-run"]
    return {
        "pool": "nature",
        "budgets": [5, 10, 20, 50],
        "n_repeats": 30,
        "n_test": 1000,
        "seed_strategy": "case_repeat",
        "seed_start": 1000,
        "case_resolution_error": None,
        "n_cases_declared": 34,
        "case_ids_declared": PREREG_CASE_IDS,
        "n_methods_declared": 20,
        "methods_declared": PREREG_METHOD_NAMES,
        "command_files_requested": ["logs/pipeline_command_history.txt"],
        "command_files_missing": [],
        "command_file_parse_errors": [],
        "command_entries": [{
            "label": "pipeline_invocation",
            "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
        }],
        "commands": commands,
    }


def _publication_case_sources() -> dict[str, object]:
    entries = []
    for case_id in PREREG_CASE_IDS:
        case = db.get_case(case_id)
        fingerprint = case_source_fingerprint(case)
        entries.append({
            "case_id": case_id,
            "status": "ok",
            "source_path": fingerprint["case_source_path"],
            "sha256": fingerprint["case_source_sha256"],
        })
    return {"schema_version": 1, "status": "ok", "cases": entries}


def _publication_method_sources() -> dict[str, object]:
    entries = []
    for method in nature_pool():
        fingerprint = method_source_fingerprint(method)
        entries.append({
            "method": method.name,
            "family": method.family,
            "status": "ok",
            "components": method_source_components(method),
            "method_source_path": fingerprint["method_source_path"],
            "method_source_sha256": fingerprint["method_source_sha256"],
        })
    return {"schema_version": 1, "status": "ok", "methods": entries}


def _write_minimal_publication_manifest(path: Path, results_csv: Path | None = None) -> None:
    artifact = {"path": "results.csv"}
    if results_csv is not None:
        artifact.update({
            "size_bytes": results_csv.stat().st_size,
            "sha256": _file_sha256(results_csv),
        })
    sweep_artifact = {"path": "sweep_integrity_report.md"}
    python_environment, pip_freeze_artifact = _materialize_publication_python_environment(path.parent)
    results_summary = {
        "exists": True,
        "n_rows": 34 * 20 * 4 * 30,
        "status_counts": {"ok": 34 * 20 * 4 * 30},
        "failure_category_counts": {"ok": 34 * 20 * 4 * 30},
        "n_cases_observed": 34,
        "case_ids_observed": PREREG_CASE_IDS,
        "n_methods_observed": 20,
        "methods_observed": PREREG_METHOD_NAMES,
        "budgets_observed": [5, 10, 20, 50],
        "repeats_observed": list(range(30)),
        "n_invalid_budget_mult_values": 0,
        "n_invalid_repeat_values": 0,
        "n_missing_failure_category_values": 0,
        "n_invalid_failure_category_values": 0,
        "n_failure_category_status_mismatch_values": 0,
    }
    path.write_text(
        json.dumps({
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py --require-complete-archive",
            "git": _publication_clean_git(),
            "machine": _publication_machine(),
            "python_environment": python_environment,
            "torch_reproducibility": _torch_reproducibility_snapshot(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "experiment": _publication_experiment(),
            "case_sources": _publication_case_sources(),
            "method_sources": _publication_method_sources(),
            "results_summary": results_summary,
            "artifacts": {
                "files": [
                    artifact,
                    sweep_artifact,
                    pip_freeze_artifact,
                ]
            },
        }),
        encoding="utf-8",
    )


def _write_results_rows(path: Path, rows: int) -> None:
    from benchmark.core import derive_seed

    case_meta_by_id = {case_id: db.get_case(case_id).meta for case_id in PREREG_CASE_IDS}
    case_fingerprint_by_id = {
        case_id: case_source_fingerprint(db.get_case(case_id))
        for case_id in PREREG_CASE_IDS
    }
    method_by_name = {method.name: method for method in nature_pool()}
    method_fingerprint_by_name = {
        name: method_source_fingerprint(method)
        for name, method in method_by_name.items()
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
                "error",
                "nrmse",
                "t_total_s",
                "rmse",
                "mae",
                "maxae",
                "q95ae",
                "r2",
                "worker_host",
                "worker_pid",
                "cell_started_utc",
                "cell_finished_utc",
                "rss_gb_start",
                "rss_gb_end",
                "rss_gb_peak",
            ],
        )
        writer.writeheader()
        budgets = [5, 10, 20, 50]
        for i in range(rows):
            repeat = i % 30
            budget = budgets[(i // 30) % len(budgets)]
            method = PREREG_METHOD_NAMES[(i // (30 * len(budgets))) % len(PREREG_METHOD_NAMES)]
            case_id = PREREG_CASE_IDS[
                (i // (30 * len(budgets) * len(PREREG_METHOD_NAMES))) % len(PREREG_CASE_IDS)
            ]
            meta = case_meta_by_id[case_id]
            method_obj = method_by_name[method]
            row = {
                "case_id": case_id,
                **case_fingerprint_by_id[case_id],
                "domain": meta.domain,
                "dim": meta.dim_input,
                "n_output": meta.dim_output,
                "method": method,
                "method_family": method_obj.family,
                **method_fingerprint_by_name[method],
                "budget_mult": budget,
                "n_train": int(budget) * int(meta.dim_input),
                "n_test": 1000,
                "repeat": repeat,
                "seed": derive_seed(case_id, repeat, strategy="case_repeat", seed_start=1000),
                "test_seed": 0,
                "seed_strategy": "case_repeat",
                "seed_start": 1000,
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": "0.10",
                "t_total_s": "0.01",
                "rmse": "0.10",
                "mae": "0.08",
                "maxae": "0.20",
                "q95ae": "0.15",
                "r2": "0.90",
                **_runtime_trace_fields(),
            }
            row["protocol_digest"] = protocol_digest_for_row(row)
            writer.writerow(row)


def _runtime_results_csv(rows: list[str]) -> str:
    header = [
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
        "error",
        "nrmse",
        "t_total_s",
        "worker_host",
        "worker_pid",
        "cell_started_utc",
        "cell_finished_utc",
        "rss_gb_start",
        "rss_gb_end",
        "rss_gb_peak",
    ]
    trace = _runtime_trace_fields()
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=header)
    writer.writeheader()
    for row in rows:
        cells = next(csv.reader([row.rstrip("\n")]))
        case_id = cells[0].strip()
        method = cells[1].strip()
        budget_mult = int(cells[2])
        repeat = int(cells[3])
        status = cells[4].strip()
        error = cells[5].strip() if len(cells) > 5 else ""
        nrmse = cells[6].strip() if len(cells) > 6 else ""
        t_total_s = cells[7].strip() if len(cells) > 7 else "1.0"
        if status == "ok":
            category = "ok"
        elif status == "timeout":
            category = "timeout"
        elif "timeout" in error.lower():
            category = "timeout"
        else:
            category = "exception"
        compact_row = {
            "case_id": case_id,
            "case_source_path": "doe_testbed/cases/test_case.py",
            "case_source_sha256": "a" * 64,
            "domain": "test",
            "dim": 2,
            "n_output": 1,
            "method": method,
            "method_family": "test_family",
            "method_source_path": "benchmark/methods.py",
            "method_source_sha256": "b" * 64,
            "budget_mult": budget_mult,
            "n_train": budget_mult * 2,
            "n_test": 1000,
            "repeat": repeat,
            "seed": 1000 + repeat,
            "test_seed": 0,
            "seed_strategy": "legacy",
            "seed_start": 1000,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "status": status,
            "failure_category": category,
            "error": error,
            "nrmse": nrmse,
            "t_total_s": t_total_s,
            **trace,
        }
        compact_row["protocol_digest"] = protocol_digest_for_row(compact_row)
        writer.writerow(compact_row)
    return out.getvalue()


def _case_readiness_row(case_id: str, **overrides: object) -> dict[str, object]:
    try:
        meta = db.get_case_meta(case_id)
        domain = meta.domain
        pde_type = meta.pde_type
        response_topology = ";".join(str(item) for item in meta.response_topology)
        n_topology_tags = len(meta.response_topology)
        governing_equation = " ".join(str(meta.governing_equation or "").split())
        n_physics_references = len(meta.physics_references)
    except KeyError:
        domain = "structures"
        pde_type = "algebraic"
        response_topology = "smooth_monotone"
        n_topology_tags = 1
        governing_equation = "test equation"
        n_physics_references = 1
    row: dict[str, object] = {
        "case_id": case_id,
        "domain": domain,
        "pde_type": pde_type,
        "response_topology": response_topology,
        "n_topology_tags": n_topology_tags,
        "governing_equation": governing_equation,
        "n_physics_references": n_physics_references,
        "status": "ok",
        "n_nan": 0,
        "n_inf": 0,
        "input_mutation_max_abs_delta": 0,
        "input_mutation_n_changed": 0,
        "determinism_max_abs_delta": 0,
        "determinism_n_mismatch": 0,
        "n_expected_range_checked": 1,
        "n_expected_range_violations": 0,
        "expected_range_max_abs_violation": 0,
    }
    row.update(overrides)
    return row


def _write_case_readiness_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_READINESS_TEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gap10_ranking_reversal_cli_pool_and_seed_contract():
    rr = _load_module(
        "benchmark/scripts/run_ranking_reversal.py",
        "run_ranking_reversal_contract",
    )
    full = _load_module(
        "benchmark/scripts/run_full_benchmark.py",
        "run_full_benchmark_contract",
    )

    args = rr.parse_args([])
    assert args.pool == "nature"
    assert args.seed_start == 1000
    assert set(rr.POOLS) == set(full.POOLS)
    assert set(rr.POOLS) == {"default", "extended", "full", "bo", "nature", "sota", "mf"}


def test_gap10_ranking_reversal_reuse_csv_rejects_duplicate_cells(tmp_path):
    rr = _load_module(
        "benchmark/scripts/run_ranking_reversal.py",
        "run_ranking_reversal_reuse_contract",
    )
    results = tmp_path / "results.csv"
    results.write_text(
        "case_id,method,budget_mult,repeat,status,failure_category,error,nrmse,t_total_s\n"
        "branin_2d,lhs+gp,10,0,ok,ok,,0.10,1.0\n"
        "branin_2d,lhs+gp,10,0,ok,ok,,0.20,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="duplicate .*cells"):
        rr._read_reusable_results_csv(results)


def test_gap10_pipeline_stage4_command_carries_pool_and_seed_start():
    run_pipeline = _load_module("scripts/run_pipeline.py", "run_pipeline_contract")
    cfg = run_pipeline.resolve_config(
        run_pipeline.parse_args(["--profile", "full", "--dry-run"])
    )
    cmd = run_pipeline._cmd_d1(cfg)

    assert cmd[cmd.index("--pool") + 1] == "nature"
    assert cmd[cmd.index("--seed-start") + 1] == "1000"
    assert cmd[cmd.index("--pairing-mode") + 1] == "all"


def test_submission_ready_bare_cli_uses_full_archive_defaults():
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_ready_cli_defaults")

    args = mod.parse_args(["paper"])

    assert args.manifest == mod.MANIFEST_DEFAULT
    assert args.numbers == mod.NUMBERS_DEFAULT
    assert args.numbers_tex == mod.NUMBERS_TEX_DEFAULT
    assert args.figure_manifest == mod.FIGURE_MANIFEST_DEFAULT
    assert args.figdata_dir is None
    assert args.d1_dir == mod.D1_DIR_DEFAULT
    assert args.d2_dir == mod.D2_DIR_DEFAULT
    assert args.d3_dir == mod.D3_DIR_DEFAULT
    assert args.test_protocol_dir == mod.TEST_PROTOCOL_DIR_DEFAULT
    assert args.case_validation_status == mod.CASE_VALIDATION_STATUS_DEFAULT
    assert args.case_validation_sources == mod.CASE_VALIDATION_SOURCES_DEFAULT
    assert args.case_validation_note_sources == mod.CASE_VALIDATION_NOTE_SOURCES_DEFAULT
    assert args.case_validation_summary == mod.CASE_VALIDATION_SUMMARY_DEFAULT
    assert args.case_experiment_readiness == mod.CASE_EXPERIMENT_READINESS_DEFAULT
    assert args.skip_manifest_check is False
    assert args.skip_paper_artifact_check is False
    assert args.skip_source_check is False
    assert args.skip_case_validation_check is False


def test_full_benchmark_large_grid_requires_explicit_confirmation(tmp_path, capsys):
    full = _load_module(
        "benchmark/scripts/run_full_benchmark.py",
        "run_full_benchmark_large_guard",
    )

    out_dir = tmp_path / "full_benchmark"
    rc = full.main([
        "--pool", "nature",
        "--budgets", "5", "10", "20", "50",
        "--n-repeats", "30",
        "--output-dir", str(out_dir),
    ])

    assert rc == 2
    assert "--confirm-large-run" in capsys.readouterr().err
    assert not out_dir.exists()


def test_full_benchmark_require_botorch_rejects_formal_nature_without_dispatch(tmp_path, monkeypatch, capsys):
    full = _load_module(
        "benchmark/scripts/run_full_benchmark.py",
        "run_full_benchmark_require_botorch",
    )
    optional = _load_module("scripts/check_optional_dependencies.py", "run_full_benchmark_optional_deps")
    sys.modules["scripts.check_optional_dependencies"] = optional
    monkeypatch.setattr(
        optional,
        "_installed_modules",
        lambda: {"torch": True, "gpytorch": False, "botorch": False},
    )
    monkeypatch.setattr(full, "BenchmarkRunner", lambda **kwargs: (_ for _ in ()).throw(AssertionError("no dispatch")))

    rc = full.main([
        "--cases", "cantilever_beam_5d",
        "--pool", "nature",
        "--budgets", "5",
        "--n-repeats", "1",
        "--require-botorch",
        "--output-dir", str(tmp_path / "out"),
    ])

    assert rc == 2
    assert "requires BoTorch-compatible methods" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_full_benchmark_cases_all_alias_matches_registry():
    full = _load_module(
        "benchmark/scripts/run_full_benchmark.py",
        "run_full_benchmark_cases_all",
    )
    expected = sorted(c.case_id for c in db.list_cases())

    assert full._resolve_case_ids(None) == expected
    assert full._resolve_case_ids(["ALL"]) == expected


def test_full_benchmark_cases_are_deduplicated_and_validated():
    full = _load_module(
        "benchmark/scripts/run_full_benchmark.py",
        "run_full_benchmark_cases_validate",
    )

    assert full._resolve_case_ids([
        "cantilever_beam_5d",
        "cantilever_beam_5d",
        "wing_vlm_6d",
    ]) == ["cantilever_beam_5d", "wing_vlm_6d"]

    try:
        full._resolve_case_ids(["not_a_case"])
    except SystemExit as exc:
        assert "Unknown case_id" in str(exc)
    else:
        raise AssertionError("unknown case should fail")

    try:
        full._resolve_case_ids(["ALL", "cantilever_beam_5d"])
    except SystemExit as exc:
        assert "ALL must be used by itself" in str(exc)
    else:
        raise AssertionError("ALL mixed with explicit cases should fail")


def test_full_benchmark_empty_case_filter_fails_before_output(tmp_path):
    full = _load_module(
        "benchmark/scripts/run_full_benchmark.py",
        "run_full_benchmark_empty_filter",
    )
    out = tmp_path / "empty_filter"

    try:
        full.main([
            "--cases", "cantilever_beam_5d",
            "--cases-filter", "mf",
            "--pool", "default",
            "--budgets", "5",
            "--n-repeats", "1",
            "--output-dir", str(out),
        ])
    except SystemExit as exc:
        assert "produced an empty case list" in str(exc)
    else:
        raise AssertionError("empty cases-filter should fail")
    assert not out.exists()


def test_full_benchmark_smoke_grid_does_not_require_large_confirmation(tmp_path, monkeypatch):
    full = _load_module(
        "benchmark/scripts/run_full_benchmark.py",
        "run_full_benchmark_smoke_guard",
    )
    calls = []

    class DummyRunner:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self):
            import pandas as pd

            return type("Result", (), {"df": pd.DataFrame({"case_id": ["cantilever_beam_5d"]})})()

    monkeypatch.setattr(full, "BenchmarkRunner", DummyRunner)

    rc = full.main([
        "--cases", "cantilever_beam_5d",
        "--pool", "default",
        "--budgets", "5",
        "--n-repeats", "1",
        "--output-dir", str(tmp_path / "smoke"),
        "--quiet",
    ])

    assert rc == 0
    assert calls


def test_full_benchmark_completion_message_uses_pipeline_for_formal_stage2():
    text = (ROOT / "benchmark" / "scripts" / "run_full_benchmark.py").read_text(encoding="utf-8")

    assert "Nature-grade Stage-2 sweep runner" in text
    assert "Full publication archive (preferred)" in text
    assert "bash scripts/run_pipeline.sh" in text
    assert "Next steps for a standalone formal Stage 2 run" in text
    assert "--only sweep_audit" in text
    assert "--from-stage 3" in text
    assert "single entry-point" not in text
    assert "Next step: python benchmark/scripts/analyze_full_benchmark.py" not in text


def test_benchmark_readme_marks_analyze_full_benchmark_as_legacy_posthoc():
    text = (ROOT / "benchmark" / "README.md").read_text(encoding="utf-8")

    assert "analyze_full_benchmark.py" in text
    assert "Legacy/post-hoc" in text
    assert "Nature CS headline artifacts" in text
    assert "--from-stage 3 --confirm-full-run" in text


def test_gap10_correlation_json_metadata_is_written():
    text = (ROOT / "benchmark/scripts/run_ranking_reversal.py").read_text(encoding="utf-8")
    assert '"method_pool": args.pool' in text
    assert '"n_methods": len(methods)' in text
    assert '"seed_start": int(args.seed_start)' in text


def test_gap13_generate_run_manifest_writes_reproducibility_snapshot(tmp_path):
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_contract")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status\n"
        "case_a,m1,10,0,ok\n"
        "case_a,m2,20,1,timeout\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "10", "20",
        "--n-repeats", "2",
        "--n-test", "5",
        "--seed-strategy", "case_repeat",
        "--seed-start", "123",
        "--cases", "case_a", "case_b",
        "--command", "python benchmark/scripts/run_full_benchmark.py",
    ])

    assert rc == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["manifest_script"] == "scripts/generate_run_manifest.py"
    assert data["machine"]["node"]
    assert "effective_ram_gb" in data["machine"]
    assert isinstance(data["machine"]["effective_ram_source"], str)
    assert data["machine"]["cgroup_memory"]["schema_version"] == 1
    assert isinstance(data["machine"]["cgroup_memory"]["detected"], bool)
    assert data["machine"]["resource_limits"]["schema_version"] == 1
    assert isinstance(data["machine"]["resource_limits"]["limits"], dict)
    assert data["machine"]["temp_storage"]["schema_version"] == 1
    assert isinstance(data["machine"]["temp_storage"]["resolved_path"], str)
    assert isinstance(data["machine"]["temp_storage"]["exists"], bool)
    assert isinstance(data["machine"]["temp_storage"]["writable"], bool)
    assert isinstance(data["machine"]["temp_storage"]["env"], dict)
    assert data["scheduler"]["schema_version"] == 1
    assert isinstance(data["scheduler"]["detected"], bool)
    assert isinstance(data["scheduler"]["env"], dict)
    assert data["accelerator"]["schema_version"] == 1
    assert isinstance(data["accelerator"]["nvidia_smi"], dict)
    assert isinstance(data["accelerator"]["torch_cuda"], dict)
    assert data["container"]["schema_version"] == 1
    assert isinstance(data["container"]["detected"], bool)
    assert isinstance(data["container"]["marker_files"], dict)
    assert data["run_window"]["source"] == "artifact_mtime_utc"
    assert data["run_window"]["started_at_utc"]
    assert data["run_window"]["finished_at_utc"]
    assert data["run_window"]["n_artifacts"] >= 1
    assert data["experiment"]["pool"] == "default"
    assert data["experiment"]["budgets"] == [10, 20]
    assert data["experiment"]["seed_start"] == 123
    assert data["experiment"]["case_ids_declared"] == ["case_a", "case_b"]
    assert data["results_summary"]["n_rows"] == 2
    assert data["results_summary"]["status_counts"] == {"ok": 1, "timeout": 1}
    assert any(f["path"] == "results.csv" for f in data["artifacts"]["files"])
    assert not (results_dir / "python_environment_pip_freeze.txt").exists()
    assert "pip_freeze_path" not in data["python_environment"]
    assert data["case_sources"]["schema_version"] == 1
    assert {item["case_id"] for item in data["case_sources"]["cases"]} == {"case_a", "case_b"}


def test_generate_run_manifest_captures_cgroup_memory_limit(tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_cgroup_memory",
    )
    (tmp_path / "memory.max").write_text(str(96 * 1024**3), encoding="utf-8")

    cgroup = mod._cgroup_memory_limit_gb(tmp_path)
    effective, source = mod._effective_ram_gb(256.0, cgroup)

    assert cgroup["detected"] is True
    assert cgroup["source"] == "cgroup v2 memory.max"
    assert cgroup["limit_gb"] == 96.0
    assert cgroup["unlimited"] is False
    assert effective == 96.0
    assert source == "cgroup v2 memory.max"


def test_generate_run_manifest_temp_storage_probe_cleans_up(monkeypatch, tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_temp_storage_probe_cleanup",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(scratch))

    info = mod._temp_storage_info()

    assert info["resolved_path"] == scratch.resolve().as_posix()
    assert info["exists"] is True
    assert info["writable"] is True
    assert not list(scratch.glob(".doe_benchmark_manifest_write_probe_*.tmp"))


def test_generate_run_manifest_temp_storage_probe_marks_unwritable(monkeypatch, tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_temp_storage_probe_unwritable",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(scratch))
    original_named_temporary_file = mod.tempfile.NamedTemporaryFile

    def fake_named_temporary_file(*args, **kwargs):
        target_dir = Path(kwargs.get("dir", ""))
        if target_dir == scratch:
            raise PermissionError("quota exceeded")
        return original_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(mod.tempfile, "NamedTemporaryFile", fake_named_temporary_file)

    info = mod._temp_storage_info()

    assert info["resolved_path"] == scratch.resolve().as_posix()
    assert info["exists"] is True
    assert info["writable"] is False


def test_generate_run_manifest_prefers_current_cgroup_memory_limit(tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_current_cgroup_memory",
    )
    proc_cgroup = tmp_path / "proc_self_cgroup"
    proc_cgroup.write_text("0::/slurm/uid_1000/job_42/step_0\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text(str(256 * 1024**3), encoding="utf-8")
    job_dir = tmp_path / "slurm" / "uid_1000" / "job_42" / "step_0"
    job_dir.mkdir(parents=True)
    (job_dir / "memory.max").write_text(str(48 * 1024**3), encoding="utf-8")

    cgroup = mod._cgroup_memory_limit_gb(tmp_path, proc_cgroup)

    assert cgroup["source"] == "cgroup v2 memory.max"
    assert cgroup["path"].endswith("slurm/uid_1000/job_42/step_0/memory.max")
    assert cgroup["limit_gb"] == 48.0
    assert cgroup["unlimited"] is False


def test_generate_run_manifest_write_failure_preserves_existing_file(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_atomic_write",
    )
    output = tmp_path / "RUN_MANIFEST.json"
    output.write_text("previous manifest\n", encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("json serialization failed")

    monkeypatch.setattr(mod.json, "dumps", boom)

    with pytest.raises(RuntimeError, match="json serialization failed"):
        mod.write_manifest_atomic({"manifest": "new"}, output)

    assert output.read_text(encoding="utf-8") == "previous manifest\n"
    assert not list(tmp_path.glob(".RUN_MANIFEST.json.tmp.*"))


def test_generate_run_manifest_records_and_rejects_symlink_artifacts(
    tmp_path, monkeypatch
):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_symlink_artifacts",
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    linked = results_dir / "results.csv"
    linked.write_text("case_id,method\ncase_a,m1\n", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(self):
        if Path(self) == linked:
            return True
        return original_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

    files = mod._file_manifest(results_dir)
    entry = next(item for item in files if item["path"] == "results.csv")
    assert entry["is_symlink"] is True
    assert entry["sha256"] is None

    errors = mod._strict_archive_errors({
        "git": {"source_dirty": False},
        "python_environment": {
            "status": "ok",
            "command": "python -m pip freeze --all",
            "pip_freeze_sha256": "0" * 64,
            "package_count": 1,
        },
        "run_window": {
            "started_at_utc": "2026-06-01T00:00:00+00:00",
            "finished_at_utc": "2026-06-01T00:00:01+00:00",
            "n_artifacts": 1,
        },
        "experiment": {
            "case_resolution_error": None,
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "command_entries": [{
                "label": "pipeline_invocation",
                "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
            }],
        },
        "results_summary": {"exists": False},
        "artifacts": {"files": files},
        "case_sources": {"status": "ok", "cases": [{"status": "ok", "sha256": "0" * 64}]},
        "method_sources": {"status": "ok", "methods": [{"status": "ok", "method_source_sha256": "0" * 64}]},
    })

    assert any("artifacts.files contains symlink entries" in err for err in errors)


def test_generate_run_manifest_records_and_rejects_csv_header_contract_errors(tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_csv_header_contract",
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_csv = results_dir / "results.csv"
    results_csv.write_text(
        "case_id,method,budget_mult,repeat,status,status\n"
        "pressure_vessel_4d,random+linear,5,0,ok,failed\n",
        encoding="utf-8",
    )

    summary = mod._results_summary(results_csv)
    errors = mod._strict_archive_errors({
        "git": {"source_dirty": False},
        "python_environment": {
            "status": "ok",
            "command": "python -m pip freeze --all",
            "pip_freeze_sha256": "0" * 64,
            "package_count": 1,
        },
        "run_window": {
            "started_at_utc": "2026-06-01T00:00:00+00:00",
            "finished_at_utc": "2026-06-01T00:00:01+00:00",
            "n_artifacts": 1,
        },
        "experiment": {
            "case_resolution_error": None,
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "command_entries": [{
                "label": "pipeline_invocation",
                "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
            }],
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [5],
            "n_repeats": 1,
        },
        "results_summary": summary,
        "artifacts": {
            "files": [
                {
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                    "is_symlink": False,
                },
                {
                    "path": "sweep_integrity_report.md",
                    "size_bytes": 10,
                    "sha256": "1" * 64,
                    "is_symlink": False,
                },
            ],
        },
        "case_sources": {
            "status": "ok",
            "cases": [{"status": "ok", "sha256": "0" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"status": "ok", "method_source_sha256": "0" * 64}],
        },
    })

    assert summary["n_csv_contract_errors"] == 1
    assert "duplicate header column" in summary["csv_contract_errors"][0]
    assert any("results_summary.n_csv_contract_errors is non-zero" in err for err in errors)


def test_generate_run_manifest_records_registered_case_source_fingerprints(tmp_path):
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_case_sources")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status\n"
        "pressure_vessel_4d,m1,10,0,ok\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "10",
        "--n-repeats", "1",
        "--n-test", "5",
        "--cases", "pressure_vessel_4d",
    ])

    assert rc == 0
    sources = json.loads(output.read_text(encoding="utf-8"))["case_sources"]["cases"]
    entry = sources[0]
    assert entry["case_id"] == "pressure_vessel_4d"
    assert entry["status"] == "ok"
    assert entry["source_path"].endswith("pressure_vessel_4d.py")
    assert len(entry["sha256"]) == 64


def test_generate_run_manifest_records_method_source_fingerprints(tmp_path):
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_method_sources")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status\n"
        "pressure_vessel_4d,random+linear,10,0,ok\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "10",
        "--n-repeats", "1",
        "--n-test", "5",
        "--cases", "pressure_vessel_4d",
    ])

    assert rc == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    entry = data["method_sources"]["methods"][0]
    assert entry["method"] == data["experiment"]["methods_declared"][0]
    assert entry["status"] == "ok"
    assert "method_source_path" in entry
    assert len(entry["method_source_sha256"]) == 64
    assert isinstance(entry.get("components"), list) and entry["components"]
    for component in entry["components"]:
        assert {"class", "path", "sha256", "config"}.issubset(component)


def test_generate_run_manifest_resolves_all_cases_with_error_status(monkeypatch):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_case_resolution",
    )

    resolved = mod._resolve_cases(["ALL"])

    assert len(resolved.case_ids) == 34
    assert resolved.error is None

    monkeypatch.setitem(sys.modules, "doe_testbed", None)
    broken = mod._resolve_cases(["ALL"])

    assert broken.case_ids == ["ALL"]
    assert broken.error is not None


def test_generate_run_manifest_reads_command_file_line_by_line(tmp_path):
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_command_file")
    command_file = tmp_path / "commands.txt"
    command_file.write_text(
        "# resolved commands\n"
        "pipeline_invocation\tpython scripts/run_pipeline.py --profile full --from-stage 3 --confirm-full-run\n"
        "validate\tpython scripts/validate_run_config.py --pool nature\n"
        "\n"
        "sweep\tpython benchmark/scripts/run_full_benchmark.py --pool nature\n",
        encoding="utf-8",
    )
    args = mod.parse_args([
        "--command", "python scripts/run_pipeline.py --profile full --confirm-full-run",
        "--command-file", str(command_file),
    ])

    assert mod._read_commands(args) == [
        "python scripts/run_pipeline.py --profile full --confirm-full-run",
        "python scripts/run_pipeline.py --profile full --from-stage 3 --confirm-full-run",
        "python scripts/validate_run_config.py --pool nature",
        "python benchmark/scripts/run_full_benchmark.py --pool nature",
    ]
    assert mod._read_command_entries(args) == [
        {
            "label": "manual_command",
            "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
        },
        {
            "label": "pipeline_invocation",
            "command": "python scripts/run_pipeline.py --profile full --from-stage 3 --confirm-full-run",
        },
        {
            "label": "validate",
            "command": "python scripts/validate_run_config.py --pool nature",
        },
        {
            "label": "sweep",
            "command": "python benchmark/scripts/run_full_benchmark.py --pool nature",
        },
    ]


def test_generate_run_manifest_records_missing_command_files(tmp_path):
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_missing_command_file")
    missing = ROOT / "logs" / "missing_commands_for_test.txt"
    assert not missing.exists()
    args = mod.parse_args([
        "--command", "python scripts/run_pipeline.py --profile full --confirm-full-run",
        "--command-file", str(missing),
    ])

    assert mod._command_file_status(args) == {
        "requested": ["logs/missing_commands_for_test.txt"],
        "missing": ["logs/missing_commands_for_test.txt"],
        "parse_errors": [],
    }
    assert mod._read_commands(args) == [
        "python scripts/run_pipeline.py --profile full --confirm-full-run",
    ]


def test_generate_run_manifest_rejects_malformed_command_file_lines(tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_bad_command_file_lines",
    )
    command_file = tmp_path / "commands.txt"
    command_file.write_text(
        "# resolved commands\n"
        "pipeline_invocation python scripts/run_pipeline.py --profile full --confirm-full-run\n"
        "\tpython scripts/validate_run_config.py --pool nature\n"
        "validate\t\n"
        "case_readiness\tpython scripts/audit_case_experiment_readiness.py\n",
        encoding="utf-8",
    )
    args = mod.parse_args(["--command-file", str(command_file)])

    status = mod._command_file_status(args)
    entries = mod._read_command_entries(args)

    assert len(status["parse_errors"]) == 3
    assert any("line 2 is not label<TAB>command" in err for err in status["parse_errors"])
    assert any("line 3 has blank label or command" in err for err in status["parse_errors"])
    assert any("line 4 has blank label or command" in err for err in status["parse_errors"])
    assert entries == [{
        "label": "case_readiness",
        "command": "python scripts/audit_case_experiment_readiness.py",
    }]

    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": [entry["command"] for entry in entries],
            "command_files_requested": status["requested"],
            "command_files_missing": status["missing"],
            "command_file_parse_errors": status["parse_errors"],
            "command_entries": entries,
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {"exists": False},
        "artifacts": {"files": []},
        "case_sources": {"status": "ok", "cases": []},
        "method_sources": {"status": "ok", "methods": []},
    }

    assert any("command_file_parse_errors" in err for err in mod._strict_archive_errors(manifest))


def test_generate_run_manifest_records_external_command_files_as_absolute(tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_external_command_file",
    )
    missing = Path(tempfile.gettempdir()) / "doe_benchmark_outside_commands_missing.txt"
    if missing.exists():
        missing.unlink()
    args = mod.parse_args(["--command-file", str(missing)])

    status = mod._command_file_status(args)

    assert status == {
        "requested": [missing.resolve().as_posix()],
        "missing": [missing.resolve().as_posix()],
        "parse_errors": [],
    }


def test_generate_run_manifest_records_invalid_numeric_grid_values(tmp_path):
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_invalid_grid")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status\n"
        "case_a,m1,5.5,0,ok\n"
        "case_a,m1,10,0.5,ok\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "5", "10",
        "--n-repeats", "2",
        "--n-test", "5",
        "--seed-strategy", "case_repeat",
        "--seed-start", "123",
        "--cases", "case_a",
    ])

    assert rc == 0
    summary = json.loads(output.read_text(encoding="utf-8"))["results_summary"]
    assert summary["budgets_observed"] == [10]
    assert summary["repeats_observed"] == [0]
    assert summary["n_invalid_budget_mult_values"] == 1
    assert summary["invalid_budget_mult_values"][0]["value"] == "5.5"
    assert summary["n_invalid_repeat_values"] == 1
    assert summary["invalid_repeat_values"][0]["value"] == "0.5"


def test_generate_run_manifest_records_missing_protocol_key_values(tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_missing_protocol_keys",
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1,,0.1\n"
        ",m1,5,1,ok,1,,0.1\n"
        "case_a,,5,2,ok,1,,0.1\n"
        "case_a,m1,,3,ok,1,,0.1\n"
        "case_a,m1,5,,ok,1,,0.1\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "5",
        "--n-repeats", "5",
        "--n-test", "5",
        "--cases", "case_a",
    ])

    assert rc == 0
    summary = json.loads(output.read_text(encoding="utf-8"))["results_summary"]
    assert summary["n_missing_protocol_key_values"] == 4
    assert {
        item["field"] for item in summary["missing_protocol_key_values"]
    } == {"case_id", "method", "budget_mult", "repeat"}


def test_generate_run_manifest_records_row_level_result_contract_counts(tmp_path):
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_row_contract")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1,,0.1\n"
        "case_a,m1,5,1,ok,1,stale error,0.1\n"
        "case_a,m1,5,2,ok,1,,nan\n"
        "case_a,m1,5,3,timeout,2,,nan\n"
        "case_a,m1,5,4,failed,-1,failure,nan\n"
        "case_a,m1,5,5,cancelled,1,,nan\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "5",
        "--n-repeats", "6",
        "--n-test", "5",
        "--cases", "case_a",
        "--command", "python scripts/run_pipeline.py --profile full --confirm-full-run",
    ])

    assert rc == 0
    summary = json.loads(output.read_text(encoding="utf-8"))["results_summary"]
    assert summary["n_invalid_status_values"] == 1
    assert summary["invalid_status_values"][0]["value"] == "cancelled"
    assert summary["n_invalid_ok_nrmse_values"] == 1
    assert summary["n_invalid_t_total_s_values"] == 1
    assert summary["n_missing_failure_error_values"] == 1
    assert summary["n_ok_rows_with_error_values"] == 1


def test_generate_run_manifest_records_runtime_resource_summary(tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_runtime_resource_summary",
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse,"
        "rss_gb_start,rss_gb_end,rss_gb_peak\n"
        "case_a,m1,5,0,ok,1,,0.1,1.0,1.2,1.5\n"
        "case_a,m1,5,1,ok,1,,0.1,1.1,1.4,1.7\n"
        "case_a,m2,5,0,ok,1,,0.1,2.0,2.4,2.8\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "5",
        "--n-repeats", "3",
        "--n-test", "5",
        "--cases", "case_a",
    ])

    assert rc == 0
    summary = json.loads(output.read_text(encoding="utf-8"))["results_summary"]
    resource = summary["runtime_resource_summary"]
    assert summary["n_invalid_rss_gb_values"] == 0
    assert summary["n_invalid_rss_gb_peak_order_values"] == 0
    assert resource["rss_trace_columns_present"] is True
    assert resource["n_rows_with_valid_rss_gb_peak"] == 3
    assert resource["rss_gb_peak_max"] == 2.8
    assert resource["rss_gb_peak_max_by_method"] == {"m1": 1.7, "m2": 2.8}


def test_generate_run_manifest_records_invalid_runtime_resource_values(tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_bad_runtime_resource_summary",
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse,"
        "rss_gb_start,rss_gb_end,rss_gb_peak\n"
        "case_a,m1,5,0,ok,1,,0.1,1.0,1.2,0.9\n"
        "case_a,m1,5,1,ok,1,,0.1,1.0,1.2,bad\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "5",
        "--n-repeats", "2",
        "--n-test", "5",
        "--cases", "case_a",
    ])

    assert rc == 0
    summary = json.loads(output.read_text(encoding="utf-8"))["results_summary"]
    assert summary["n_invalid_rss_gb_values"] == 1
    assert summary["n_invalid_rss_gb_peak_order_values"] == 1
    assert summary["runtime_resource_summary"]["n_rows_with_valid_rss_gb_peak"] == 1


def test_generate_run_manifest_records_runtime_timestamp_summary(tmp_path):
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_runtime_timestamp_summary",
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse,"
        "cell_started_utc,cell_finished_utc\n"
        "case_a,m1,5,0,ok,1,,0.1,2026-06-01T00:00:00+00:00,2026-06-01T00:00:01+00:00\n"
        "case_a,m1,5,1,ok,1,,0.1,bad,2026-06-01T00:00:01+00:00\n"
        "case_a,m1,5,2,ok,1,,0.1,2026-06-01T00:00:02+00:00,2026-06-01T00:00:01+00:00\n"
        "case_a,m1,5,3,ok,100,,0.1,2026-06-01T00:00:03+00:00,2026-06-01T00:00:04+00:00\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "5",
        "--n-repeats", "4",
        "--n-test", "5",
        "--cases", "case_a",
    ])

    assert rc == 0
    summary = json.loads(output.read_text(encoding="utf-8"))["results_summary"]
    timestamp = summary["runtime_timestamp_summary"]
    assert summary["n_invalid_runtime_timestamp_values"] == 1
    assert summary["n_invalid_runtime_time_order_values"] == 1
    assert summary["n_invalid_runtime_duration_values"] == 1
    assert timestamp["timestamp_trace_columns_present"] is True
    assert timestamp["n_rows"] == 4
    assert timestamp["n_invalid_runtime_timestamp_values"] == 1
    assert timestamp["n_invalid_runtime_time_order_values"] == 1
    assert timestamp["n_invalid_runtime_duration_values"] == 1


def test_generate_run_manifest_strict_archive_rejects_method_drift_before_write(tmp_path):
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_strict_bad")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results.csv").write_text(
        "case_id,method,budget_mult,repeat,status\n"
        "pressure_vessel_4d,random+linear,10,0,ok\n"
        "pressure_vessel_4d,unexpected_method,10,0,ok\n",
        encoding="utf-8",
    )
    output = tmp_path / "RUN_MANIFEST.json"
    output.write_text("previous manifest\n", encoding="utf-8")

    rc = mod.main([
        "--results-dir", str(results_dir),
        "--output", str(output),
        "--pool", "default",
        "--budgets", "10",
        "--n-repeats", "1",
        "--n-test", "5",
        "--cases", "pressure_vessel_4d",
        "--command", "python scripts/run_pipeline.py --profile full --confirm-full-run",
        "--require-complete-archive",
    ])

    assert rc == 3
    assert output.read_text(encoding="utf-8") == "previous manifest\n"


def test_generate_run_manifest_strict_archive_allows_derived_dirty_only():
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_strict_dirty_split")
    python_environment, pip_freeze_artifact = _materialize_publication_python_environment(ROOT)
    manifest = {
        "git": {
            "dirty": True,
            "source_dirty": False,
            "derived_dirty_entries": [
                " M paper/numbers.json",
                " M docs/CASE_VALIDATION_SUMMARY.md",
                " M docs/case_validation_note_sources.csv",
            ],
        },
        "python_environment": python_environment,
        "torch_reproducibility": _torch_reproducibility_snapshot(),
        "machine": _publication_machine(),
        "scheduler": _publication_scheduler(),
        "accelerator": _publication_accelerator(),
        "container": _publication_container(),
        "run_window": _publication_run_window(),
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "command_entries": [{
                "label": "pipeline_invocation",
                "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
            }],
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "n_cases_observed": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "n_methods_observed": 1,
            "methods_observed": ["random+linear"],
            "failure_category_counts": {"ok": 1},
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_invalid_runtime_timestamp_values": 0,
            "n_invalid_runtime_time_order_values": 0,
            "n_invalid_runtime_duration_values": 0,
            "n_invalid_rss_gb_values": 0,
            "n_invalid_rss_gb_peak_order_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
            "runtime_timestamp_summary": _runtime_timestamp_summary(),
            "runtime_resource_summary": {
                "rss_trace_columns_present": True,
                "missing_rss_trace_columns": [],
                "n_rows": 1,
                "n_rows_with_valid_rss_gb_peak": 1,
                "rss_gb_peak_min": 1.0,
                "rss_gb_peak_p50": 1.0,
                "rss_gb_peak_p95": 1.0,
                "rss_gb_peak_max": 1.0,
                "rss_gb_peak_max_by_method": {"random+linear": 1.0},
            },
        },
        "artifacts": {
            "files": [
                {"path": "results.csv"},
                {"path": "sweep_integrity_report.md"},
                pip_freeze_artifact,
            ]
        },
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    assert mod._strict_archive_errors(manifest) == []

    manifest["git"]["source_dirty"] = True
    manifest["git"]["source_dirty_entries"] = [" M scripts/run_pipeline.py"]

    assert any("source/config" in err for err in mod._strict_archive_errors(manifest))


def test_generate_run_manifest_strict_archive_rejects_row_contract_counts():
    mod = _load_module("scripts/generate_run_manifest.py", "generate_run_manifest_strict_row_contract")
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_missing_protocol_key_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 1,
            "n_invalid_runtime_duration_values": 1,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("n_invalid_t_total_s_values" in err for err in errors)
    assert any("n_invalid_runtime_duration_values" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_missing_protocol_key_count():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_missing_protocol_keys",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_missing_protocol_key_values": 1,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("n_missing_protocol_key_values" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_observed_set_count_drift():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_observed_set_count_drift",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "case_ids_declared": ["pressure_vessel_4d", "cantilever_beam_5d"],
            "methods_declared": ["random+linear", "lhs+gp"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 4,
            "n_cases_observed": 3,
            "case_ids_observed": ["pressure_vessel_4d", "pressure_vessel_4d"],
            "n_methods_observed": 3,
            "methods_observed": ["random+linear", "random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_missing_protocol_key_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("case_ids_observed contains duplicates" in err for err in errors)
    assert any("methods_observed contains duplicates" in err for err in errors)
    assert any("case count drift" in err for err in errors)
    assert any("method count drift" in err for err in errors)
    assert any("case set drift" in err and "cantilever_beam_5d" in err for err in errors)
    assert any("method set drift" in err and "lhs+gp" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_case_resolution_error():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_case_resolution_error",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": "ImportError: no doe_testbed",
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "command_entries": [{
                "label": "pipeline_invocation",
                "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
            }],
            "case_ids_declared": ["ALL"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {"exists": False},
        "artifacts": {"files": []},
        "case_sources": {"status": "unavailable", "cases": []},
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("case_resolution_error" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_dry_run_command_history():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_dry_run_commands",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": [
                "python scripts/run_pipeline.py --profile full --confirm-full-run --dry-run",
            ],
            "case_resolution_error": None,
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("--dry-run" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_missing_command_file():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_missing_command_file",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "command_files_missing": ["logs/pipeline_command_history.txt"],
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("command_files_missing" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_malformed_command_file_fields():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_command_file_fields",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "command_files_requested": "logs/pipeline_command_history.txt",
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("command_files_requested" in err for err in errors)
    assert any("command_files_missing" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_empty_command_file_request():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_empty_command_file_request",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "command_files_requested": [],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("command_files_requested is empty" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_bad_command_file_entries():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_bad_command_file_entries",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "command_files_requested": [
                "logs/pipeline_command_history.dry_run.txt",
                "../logs/pipeline_command_history.txt",
                r"C:\DOE-benchmark\logs\pipeline_command_history.txt",
                "",
                123,
            ],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("blank or non-string" in err for err in errors)
    assert any("non-portable paths" in err for err in errors)
    assert any("dry-run command history" in err for err in errors)


def test_generate_run_manifest_strict_archive_requires_pipeline_invocation_entry():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_pipeline_invocation_entry",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "command_entries": [{
                "label": "sweep",
                "command": "python benchmark/scripts/run_full_benchmark.py --pool nature",
            }],
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("pipeline_invocation" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_low_level_pipeline_invocation():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_low_level_pipeline_invocation",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python benchmark/scripts/run_full_benchmark.py --pool nature"],
            "case_resolution_error": None,
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "command_entries": [{
                "label": "pipeline_invocation",
                "command": "python benchmark/scripts/run_full_benchmark.py --pool nature",
            }],
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("formal full-pipeline entrypoint" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_malformed_command_entries():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_malformed_command_entries",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "command_entries": [
                {"label": "pipeline_invocation", "command": ""},
                {"label": "", "command": "python scripts/run_pipeline.py --profile full --confirm-full-run"},
            ],
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("malformed entries" in err for err in errors)
    assert any("do not match experiment.commands" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_duplicate_pipeline_invocation_entries():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_duplicate_pipeline_invocation_entries",
    )
    command = "python scripts/run_pipeline.py --profile full --confirm-full-run"
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": [command, command],
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_files_missing": [],
            "command_file_parse_errors": [],
            "command_entries": [
                {"label": "pipeline_invocation", "command": command},
                {"label": "pipeline_invocation", "command": command},
            ],
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("multiple pipeline_invocation" in err for err in errors)


def test_generate_run_manifest_strict_archive_requires_sweep_audit_artifact():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_requires_sweep_audit",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("sweep_integrity_report.md" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_duplicate_artifact_paths():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_duplicate_artifacts",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {
            "files": [
                {"path": "results.csv"},
                {"path": "results.csv"},
                {"path": "sweep_integrity_report.md"},
            ]
        },
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("duplicate paths" in err and "results.csv" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_malformed_row_contract_counts():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_row_contract_malformed",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": [10],
            "n_repeats": 1,
        },
        "results_summary": {
            "exists": True,
            "n_rows": 1,
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": [10],
            "repeats_observed": [0],
            "n_invalid_budget_mult_values": "bad",
            "n_invalid_repeat_values": -1,
            "n_missing_protocol_key_values": "bad",
            "n_invalid_status_values": "bad",
            "n_missing_failure_category_values": "bad",
            "n_invalid_failure_category_values": -1,
            "n_failure_category_status_mismatch_values": "bad",
            "n_invalid_ok_nrmse_values": -1,
            "n_invalid_t_total_s_values": "bad",
            "n_missing_failure_error_values": -1,
            "n_ok_rows_with_error_values": "bad",
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("n_invalid_budget_mult_values must be a non-negative integer" in err for err in errors)
    assert any("n_invalid_repeat_values must be a non-negative integer" in err for err in errors)
    assert any("n_missing_protocol_key_values must be a non-negative integer" in err for err in errors)
    assert any("n_invalid_status_values must be a non-negative integer" in err for err in errors)
    assert any("n_invalid_ok_nrmse_values must be a non-negative integer" in err for err in errors)
    assert any("n_invalid_t_total_s_values must be a non-negative integer" in err for err in errors)
    assert any("n_missing_failure_error_values must be a non-negative integer" in err for err in errors)
    assert any("n_ok_rows_with_error_values must be a non-negative integer" in err for err in errors)


def test_generate_run_manifest_strict_archive_rejects_malformed_grid_summary():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_strict_grid_malformed",
    )
    manifest = {
        "git": {"source_dirty": False},
        "experiment": {
            "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
            "case_ids_declared": ["pressure_vessel_4d"],
            "methods_declared": ["random+linear"],
            "budgets": ["10", "bad"],
            "n_repeats": "bad",
        },
        "results_summary": {
            "exists": True,
            "n_rows": "bad",
            "case_ids_observed": ["pressure_vessel_4d"],
            "methods_observed": ["random+linear"],
            "budgets_observed": ["10", "bad"],
            "repeats_observed": ["0", "bad"],
            "n_invalid_budget_mult_values": 0,
            "n_invalid_repeat_values": 0,
            "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
            "n_invalid_ok_nrmse_values": 0,
            "n_invalid_t_total_s_values": 0,
            "n_missing_failure_error_values": 0,
            "n_ok_rows_with_error_values": 0,
        },
        "artifacts": {"files": [{"path": "results.csv"}, {"path": "sweep_integrity_report.md"}]},
        "case_sources": {
            "status": "ok",
            "cases": [{"case_id": "pressure_vessel_4d", "status": "ok", "sha256": "a" * 64}],
        },
        "method_sources": {
            "status": "ok",
            "methods": [{"method": "random+linear", "status": "ok", "method_source_sha256": "b" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("experiment.budgets must be a list of integers" in err for err in errors)
    assert any("results_summary.budgets_observed must be a list of integers" in err for err in errors)
    assert any("results_summary.repeats_observed must be a list of integers" in err for err in errors)
    assert any("experiment.n_repeats must be a non-negative integer" in err for err in errors)
    assert any("results_summary.n_rows must be a non-negative integer" in err for err in errors)


def test_gap14_windows_16gb_launcher_contract():
    path = ROOT / "scripts" / "launch_sweep_local_16gb.ps1"
    text = path.read_text(encoding="utf-8")

    assert "Local full-sweep launch is disabled" in text
    assert "Do not launch the 34 x 20 x 4 x 30 Nature CS sweep from the local workstation" in text
    assert "exit 2" in text
    assert "Formal Linux server launch:" in text
    assert "bash scripts/run_pipeline.sh" in text
    assert "validate_run_config.py" in text
    assert "check_optional_dependencies.py" in text
    assert "--pool nature" in text
    assert "--require-botorch" in text
    assert "--expected-methods" in text
    assert "--require-runtime-monitoring" in text
    assert '"20"' in text
    assert "--budgets 5 10 20 50" in text
    assert "--n-repeats 30" in text
    assert "--n-test 1000" in text
    assert "--seed-strategy case_repeat" in text
    assert "--seed-start 1000" in text
    assert "--skip-existing" in text
    assert "--timeout-per-cell 600" in text
    assert "--bo-timeout-per-cell 1800" in text
    assert "--ref-timeout-s 600" in text
    assert "Start-Process" in text
    assert "[int]$NJobs      = 8" in text
    assert "[int]$BoNJobs    = 1" in text
    assert "[double]$MemCapGb = 12.0" in text

    match = re.search(r"\$Cases\s*=\s*@\((.*?)\)", text, flags=re.S)
    assert match is not None
    cases = re.findall(r'"([a-z0-9_]+)"', match.group(1))
    assert len(cases) == 34
    assert len(set(cases)) == 34


def test_gap14_windows_pipeline_wrapper_is_smoke_preflight_only():
    path = ROOT / "scripts" / "run_pipeline.ps1"
    text = path.read_text(encoding="utf-8")

    assert "[string]$Profile = 'smoke'" in text
    assert "Windows run_pipeline.ps1 is smoke/preflight only" in text
    assert "Do not launch the 34 x 20 x 4 x 30 Nature CS full sweep locally" in text
    assert "exit 2" in text
    assert "[switch]$DryRun" in text
    assert "[int]$ToStage" in text
    assert "Test-IsStage1Only" in text
    assert "-ToStage 1" in text
    assert "--dry-run" in text
    assert "bash scripts/run_pipeline.sh" in text


def test_server_return_package_requires_core_archive_paths(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_paths",
    )
    results_root = tmp_path / "benchmark" / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = tmp_path / "paper"
    docs_dir = tmp_path / "docs"
    logs_dir = tmp_path / "logs"
    for path in [
        results_dir / "analysis",
        results_root / "ranking_reversal",
        results_root / "sample_complexity_0.20",
        results_root / "reliability_recovery",
        paper_dir / "figdata",
        paper_dir / "figures",
        docs_dir,
        logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [
        tmp_path / ".zenodo.json",
        tmp_path / "CITATION.cff",
        tmp_path / "codemeta.json",
        tmp_path / "croissant-metadata.json",
        tmp_path / "ro-crate-metadata.json",
        results_dir / "results.csv",
        results_dir / "runtime_health_report.txt",
        results_dir / "RUN_MANIFEST.json",
        results_dir / "analysis" / "test_set_sufficiency.csv",
        docs_dir / "case_validation_status.csv",
        docs_dir / "case_validation_sources.csv",
        docs_dir / "case_validation_note_sources.csv",
        docs_dir / "case_experiment_readiness.csv",
        docs_dir / "CASE_VALIDATION_SUMMARY.md",
        docs_dir / "CASE_WEB_VALIDATION_AUDIT.md",
        docs_dir / "EXPERIMENT_BENCHMARK_AUDIT.md",
        docs_dir / "PUBLICATION_LAUNCH_PREFLIGHT.md",
        docs_dir / "STATISTICAL_ANALYSIS_PLAN.md",
        paper_dir / "numbers.json",
        paper_dir / "numbers.tex",
        paper_dir / "figure_manifest.json",
        paper_dir / "figdata" / "figdata_manifest.json",
        logs_dir / "pipeline_command_history.txt",
    ]:
        path.write_text("x\n", encoding="utf-8")
    _write_minimal_machine_readable_metadata(tmp_path)
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )
    results_csv = results_dir / "results.csv"
    _materialize_publication_python_environment(results_dir)
    run_config_json = results_dir / "run_config.json"
    _write_minimal_publication_run_config(run_config_json)
    (results_dir / "runtime_health_report.txt").write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{results_dir / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{results_dir / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{results_dir / 'run_config.json'}\n"
        "[warn] run_config.json torch reproducibility policy recorded and valid: "
        f"{results_dir / 'run_config.json'}\n"
        "[warn] current runtime environment matches run_config.json snapshot: "
        f"{results_dir / 'run_config.json'}\n"
        "[summary] runtime failure-category telemetry summarized: "
        'unique_cells=1 failure_category_counts={"ok": 1} '
        "non_ok_failure_category_counts={}\n"
        "[summary] runtime resource telemetry summarized: "
        'rss_gb_peak={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        "[summary] runtime wall-time telemetry summarized: "
        't_total_s={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[artifact] "
        f"run_config_json={run_config_json.resolve()} "
        f"size_bytes={run_config_json.stat().st_size} "
        f"sha256={_file_sha256(run_config_json)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )
    (results_root / "ranking_reversal" / "sentinel.txt").write_text("x\n", encoding="utf-8")
    (results_root / "sample_complexity_0.20" / "sentinel.txt").write_text("x\n", encoding="utf-8")
    (results_root / "reliability_recovery" / "sentinel.txt").write_text("x\n", encoding="utf-8")
    (paper_dir / "figdata" / "sentinel.csv").write_text("x\n", encoding="utf-8")
    (paper_dir / "figures" / "sentinel.png").write_text("x\n", encoding="utf-8")

    errors = mod.check_required_return_paths(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        docs_dir=docs_dir,
        logs_dir=logs_dir,
    )

    assert errors == []

    (results_root / "ranking_reversal" / "sentinel.txt").unlink()
    (paper_dir / "figdata" / "figdata_manifest.json").write_text("", encoding="utf-8")

    errors = mod.check_required_return_paths(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        docs_dir=docs_dir,
        logs_dir=logs_dir,
    )

    assert any("D1 ranking-reversal results is empty" in err for err in errors)
    assert any("figdata manifest is empty" in err for err in errors)


def _write_minimal_machine_readable_metadata(root: Path) -> None:
    def croissant_headline_fields() -> list[dict[str, object]]:
        fields = {
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
        return [
            {
                "@id": f"headline_results/{name}",
                "name": name,
                "dataType": data_type,
                "source": {
                    "fileObject": {"@id": "headline_results_csv"},
                    "extract": {"column": name},
                },
            }
            for name, data_type in fields.items()
        ]

    (root / ".zenodo.json").write_text(
        json.dumps({
            "title": "DOE-Benchmark",
            "description": "Return package metadata for DOE-Benchmark.",
            "version": "0.4.0-dev",
            "license": "MIT",
            "upload_type": "software",
            "creators": [{"name": "Wang, Binglin"}],
        }),
        encoding="utf-8",
    )
    (root / "codemeta.json").write_text(
        json.dumps({
            "@context": "https://w3id.org/codemeta/3.0",
            "@type": "SoftwareSourceCode",
            "name": "doe-testbed",
            "description": "Physics-grounded benchmark suite.",
            "version": "0.4.0-dev",
            "codeRepository": "https://github.com/wangbinglin14/doe-testbed",
            "license": "https://spdx.org/licenses/MIT",
            "author": [{"@type": "Person", "familyName": "Wang"}],
        }),
        encoding="utf-8",
    )
    (root / "croissant-metadata.json").write_text(
        json.dumps({
            "@context": {
                "sc": "https://schema.org/",
                "cr": "http://mlcommons.org/croissant/",
                "rai": "http://mlcommons.org/croissant/RAI/",
            },
            "@type": "sc:Dataset",
            "name": "DOE-Benchmark headline benchmark archive",
            "description": "Croissant metadata for DOE-Benchmark.",
            "url": "https://github.com/wangbinglin14/doe-testbed",
            "license": "https://spdx.org/licenses/MIT",
            "version": "0.4.0-dev",
            "conformsTo": "http://mlcommons.org/croissant/1.1",
            "distribution": [
                {
                    "@type": "cr:FileObject",
                    "@id": "headline_results_csv",
                    "name": "results.csv",
                    "contentUrl": "benchmark/results/full_benchmark/results.csv",
                    "encodingFormat": "text/csv",
                    "description": "Canonical row-level benchmark result table.",
                },
                {
                    "@type": "cr:FileObject",
                    "@id": "run_manifest_json",
                    "name": "RUN_MANIFEST.json",
                    "contentUrl": "benchmark/results/full_benchmark/RUN_MANIFEST.json",
                    "encodingFormat": "application/json",
                    "description": "Run manifest and provenance.",
                },
                {
                    "@type": "cr:FileObject",
                    "@id": "figdata_manifest_json",
                    "name": "figdata_manifest.json",
                    "contentUrl": "paper/figdata/figdata_manifest.json",
                    "encodingFormat": "application/json",
                    "description": "Figure-data lineage manifest.",
                },
            ],
            "recordSet": [
                {
                    "@type": "cr:RecordSet",
                    "@id": "headline_results",
                    "name": "headline_results",
                    "description": "One record per benchmark cell.",
                    "field": croissant_headline_fields(),
                }
            ],
            "rai:dataLimitations": "P2 cases remain labelled as plausibility-only.",
            "rai:dataBiases": "The suite intentionally favours affordable physics surrogates.",
            "rai:personalSensitiveInformation": "No personal or sensitive information is included.",
            "rai:dataUseCases": "Benchmark DOE and surrogate methods under the frozen protocol.",
            "rai:dataSocialImpact": "Supports reproducible benchmark practice; over-interpretation is mitigated by evidence tiers.",
            "rai:hasSyntheticData": True,
        }),
        encoding="utf-8",
    )
    (root / "ro-crate-metadata.json").write_text(
        json.dumps({
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": [
                {
                    "@id": "ro-crate-metadata.json",
                    "@type": "CreativeWork",
                    "about": {"@id": "./"},
                },
                {
                    "@id": "./",
                    "@type": "Dataset",
                    "hasPart": [
                        {"@id": "CITATION.cff"},
                        {"@id": "codemeta.json"},
                        {"@id": "croissant-metadata.json"},
                        {"@id": "benchmark/results/full_benchmark/results.csv"},
                        {"@id": "benchmark/results/full_benchmark/RUN_MANIFEST.json"},
                        {"@id": "paper/figdata/figdata_manifest.json"},
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )


def test_server_return_package_required_paths_rejects_stale_runtime_health_run_config_fingerprint(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_required_stale_config_health",
    )
    results_root = tmp_path / "benchmark" / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = tmp_path / "paper"
    docs_dir = tmp_path / "docs"
    logs_dir = tmp_path / "logs"
    for path in [
        results_dir / "analysis",
        results_root / "ranking_reversal",
        results_root / "sample_complexity_0.20",
        results_root / "reliability_recovery",
        paper_dir / "figdata",
        paper_dir / "figures",
        docs_dir,
        logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [
        tmp_path / "CITATION.cff",
        results_dir / "results.csv",
        results_dir / "RUN_MANIFEST.json",
        results_dir / "analysis" / "test_set_sufficiency.csv",
        docs_dir / "case_validation_status.csv",
        docs_dir / "case_validation_sources.csv",
        docs_dir / "case_validation_note_sources.csv",
        docs_dir / "case_experiment_readiness.csv",
        docs_dir / "CASE_VALIDATION_SUMMARY.md",
        docs_dir / "CASE_WEB_VALIDATION_AUDIT.md",
        docs_dir / "EXPERIMENT_BENCHMARK_AUDIT.md",
        docs_dir / "PUBLICATION_LAUNCH_PREFLIGHT.md",
        docs_dir / "STATISTICAL_ANALYSIS_PLAN.md",
        paper_dir / "numbers.json",
        paper_dir / "numbers.tex",
        paper_dir / "figure_manifest.json",
        paper_dir / "figdata" / "figdata_manifest.json",
        logs_dir / "pipeline_command_history.txt",
    ]:
        path.write_text("x\n", encoding="utf-8")
    _write_minimal_machine_readable_metadata(tmp_path)
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )
    _materialize_publication_python_environment(results_dir)
    run_config_json = results_dir / "run_config.json"
    _write_minimal_publication_run_config(run_config_json)
    results_csv = results_dir / "results.csv"
    (results_dir / "runtime_health_report.txt").write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{run_config_json}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{run_config_json}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{run_config_json}\n"
        "[warn] run_config.json torch reproducibility policy recorded and valid: "
        f"{run_config_json}\n"
        "[warn] current runtime environment matches run_config.json snapshot: "
        f"{run_config_json}\n"
        "[summary] runtime failure-category telemetry summarized: "
        'unique_cells=1 failure_category_counts={"ok": 1} '
        "non_ok_failure_category_counts={}\n"
        "[summary] runtime resource telemetry summarized: "
        'rss_gb_peak={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        "[summary] runtime wall-time telemetry summarized: "
        't_total_s={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[artifact] "
        f"run_config_json={run_config_json.resolve()} "
        f"size_bytes=999 sha256={'0' * 64}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )
    (results_root / "ranking_reversal" / "sentinel.txt").write_text("x\n", encoding="utf-8")
    (results_root / "sample_complexity_0.20" / "sentinel.txt").write_text("x\n", encoding="utf-8")
    (results_root / "reliability_recovery" / "sentinel.txt").write_text("x\n", encoding="utf-8")
    (paper_dir / "figdata" / "sentinel.csv").write_text("x\n", encoding="utf-8")
    (paper_dir / "figures" / "sentinel.png").write_text("x\n", encoding="utf-8")

    errors = mod.check_required_return_paths(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        docs_dir=docs_dir,
        logs_dir=logs_dir,
    )

    assert any(
        "full-sweep runtime health report run_config.json size_bytes does not match"
        in err
        for err in errors
    )
    assert any(
        "full-sweep runtime health report run_config.json sha256 does not match" in err
        for err in errors
    )


def test_server_return_package_requires_citation_metadata(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_citation_metadata",
    )
    results_root = tmp_path / "benchmark" / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = tmp_path / "paper"
    docs_dir = tmp_path / "docs"
    logs_dir = tmp_path / "logs"
    for path in [
        results_dir / "analysis",
        results_root / "ranking_reversal",
        results_root / "sample_complexity_0.20",
        results_root / "reliability_recovery",
        paper_dir / "figdata",
        paper_dir / "figures",
        docs_dir,
        logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    errors = mod.check_required_return_paths(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        docs_dir=docs_dir,
        logs_dir=logs_dir,
    )

    assert any("citation metadata missing" in err for err in errors)

    citation = tmp_path / "CITATION.cff"
    citation.write_text("cff-version: 1.2.0\ntitle: DOE-Benchmark\n", encoding="utf-8")
    errors = mod.check_required_return_paths(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        docs_dir=docs_dir,
        logs_dir=logs_dir,
    )

    assert not any("citation metadata" in err for err in errors)


def test_server_return_package_validates_machine_readable_metadata_content(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_metadata_content",
    )
    _write_minimal_machine_readable_metadata(tmp_path)

    assert mod.check_machine_readable_metadata(tmp_path) == []

    croissant = json.loads((tmp_path / "croissant-metadata.json").read_text(encoding="utf-8"))
    croissant["recordSet"][0]["field"][0]["source"]["extract"]["column"] = "wrong_case_column"
    (tmp_path / "croissant-metadata.json").write_text(
        json.dumps(croissant),
        encoding="utf-8",
    )
    errors = mod.check_machine_readable_metadata(tmp_path)
    assert any(
        "croissant-metadata.json field source.extract.column must match field name" in err
        for err in errors
    )
    _write_minimal_machine_readable_metadata(tmp_path)

    (tmp_path / ".zenodo.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "codemeta.json").write_text(
        json.dumps({
            "@context": "https://example.org/software-schema",
            "@type": "SoftwareSourceCode",
            "name": "doe-testbed",
            "description": "x",
            "version": "0.4.0-dev",
            "codeRepository": "https://example.org/repo",
            "license": "MIT",
            "author": [],
        }),
        encoding="utf-8",
    )
    (tmp_path / "ro-crate-metadata.json").write_text(
        json.dumps({
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": [{"@id": "./", "@type": "Dataset", "hasPart": []}],
        }),
        encoding="utf-8",
    )

    errors = mod.check_machine_readable_metadata(tmp_path)

    assert any(".zenodo.json is not valid JSON" in err for err in errors)
    assert any("codemeta.json @context must reference CodeMeta" in err for err in errors)
    assert any("codemeta.json author must be a non-empty list" in err for err in errors)
    assert any("ro-crate-metadata.json missing metadata descriptor node" in err for err in errors)
    assert any("ro-crate-metadata.json root Dataset hasPart must be non-empty" in err for err in errors)

    (tmp_path / "croissant-metadata.json").write_text(
        json.dumps({
            "@context": "https://example.org/dataset-schema",
            "@type": "Dataset",
            "name": "",
            "description": "x",
            "url": "https://example.org",
            "license": "MIT",
            "version": "0.4.0",
            "conformsTo": "https://example.org/spec",
            "distribution": [],
            "recordSet": [{"@id": "rows", "field": []}],
            "rai:hasSyntheticData": "yes",
        }),
        encoding="utf-8",
    )
    errors = mod.check_machine_readable_metadata(tmp_path)
    assert any("croissant-metadata.json @context must reference Croissant" in err for err in errors)
    assert any("croissant-metadata.json @type must be 'sc:Dataset'" in err for err in errors)
    assert any("croissant-metadata.json distribution must be a non-empty list" in err for err in errors)
    assert any("croissant-metadata.json recordSet field must be non-empty" in err for err in errors)
    assert any("croissant-metadata.json missing non-empty rai:dataLimitations" in err for err in errors)
    assert any("croissant-metadata.json rai:hasSyntheticData must be boolean" in err for err in errors)


def test_server_return_package_requires_extended_croissant_headline_fields(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_extended_croissant_fields",
    )
    _write_minimal_machine_readable_metadata(tmp_path)
    croissant_path = tmp_path / "croissant-metadata.json"
    croissant = json.loads(croissant_path.read_text(encoding="utf-8"))
    fields = croissant["recordSet"][0]["field"]
    croissant["recordSet"][0]["field"] = [
        field
        for field in fields
        if field.get("name") not in {"protocol_digest", "rmse"}
    ]
    for field in croissant["recordSet"][0]["field"]:
        if field.get("name") == "nrmse":
            field["dataType"] = "sc:Text"
        if field.get("name") == "worker_pid":
            field["dataType"] = "sc:Text"
    croissant_path.write_text(json.dumps(croissant), encoding="utf-8")

    errors = mod.check_machine_readable_metadata(tmp_path)

    assert any(
        "headline_results recordSet missing field protocol_digest" in err
        for err in errors
    )
    assert any(
        "headline_results recordSet missing field rmse" in err
        for err in errors
    )
    assert any("field nrmse dataType must be sc:Float" in err for err in errors)
    assert any("field worker_pid dataType must be sc:Integer" in err for err in errors)


def test_submission_ready_validates_machine_readable_root_metadata(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_ready_metadata_content",
    )
    _write_minimal_machine_readable_metadata(tmp_path)

    assert mod.check_machine_readable_root_metadata(tmp_path) == []

    (tmp_path / "codemeta.json").write_text(
        json.dumps({
            "@context": "https://example.org/software-schema",
            "@type": "SoftwareSourceCode",
            "name": "doe-testbed",
            "description": "x",
            "version": "0.4.0-dev",
            "codeRepository": "https://example.org/repo",
            "license": "MIT",
            "author": [],
        }),
        encoding="utf-8",
    )

    errors = mod.check_machine_readable_root_metadata(tmp_path)

    assert any("codemeta.json @context must reference CodeMeta" in err for err in errors)
    assert any("codemeta.json author must be a non-empty list" in err for err in errors)


def test_submission_ready_cli_checks_machine_readable_root_metadata_by_default(
    tmp_path,
    monkeypatch,
    capsys,
):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_ready_metadata_cli",
    )
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "nature_cs_main.tex").write_text("clean manuscript text\n", encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    rc = mod._main_impl([
        str(paper),
        "--skip-source-check",
        "--skip-case-validation-check",
        "--skip-manifest-check",
        "--skip-paper-artifact-check",
    ])
    out = capsys.readouterr().out

    assert rc == 1
    assert "[metadata] .zenodo.json is missing from the repository root" in out
    assert "[summary] machine-readable metadata errors:" in out


def test_server_return_package_rechecks_p2_web_audit_evidence_tiers(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_p2_web_audit_content",
    )
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "case_validation_status.csv").write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "implicit_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_x.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    (docs_dir / "CASE_WEB_VALIDATION_AUDIT.md").write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `implicit_case` | reference checked | plausible | Keep P2 plausibility-only. |\n",
        encoding="utf-8",
    )

    errors = mod.check_case_web_validation_audit_content(docs_dir=docs_dir)

    assert any("markers do not match" in error for error in errors)
    assert any("missing `Evidence tier remains P2` marker" in error for error in errors)


def test_server_return_package_requires_geomechanics_source_snapshot(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_geomechanics_snapshot",
    )
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "case_validation_status.csv").write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "dam_seepage_6d,geomechanics,6,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_dam.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    (docs_dir / "CASE_WEB_VALIDATION_AUDIT.md").write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `dam_seepage_6d` | Evidence tier remains P2. Reference checked. | "
        "Plausible seepage stress test. | Keep P2 plausibility-only. |\n",
        encoding="utf-8",
    )

    errors = mod.check_case_web_validation_audit_content(docs_dir=docs_dir)

    assert any("geomechanics P2 source-verification" in error for error in errors)
    assert any(
        "Additional `dam_seepage_6d` 2026-06-08 public-scope boundary tightening:"
        in error
        for error in errors
    )


def test_server_return_package_requires_analytical_control_source_snapshot(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_analytical_control_snapshot",
    )
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "case_validation_status.csv").write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "multilayer_wall_4d,heat_transfer,4,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_wall.py,P2,accepted analytic conduction control,keep\n",
        encoding="utf-8",
    )
    (docs_dir / "CASE_WEB_VALIDATION_AUDIT.md").write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `multilayer_wall_4d` | Evidence tier remains P2 analytic-control. "
        "Reference checked as boundary/context evidence, not external validation. | "
        "plausible | Keep P2 analytic-control. |\n",
        encoding="utf-8",
    )

    errors = mod.check_case_web_validation_audit_content(docs_dir=docs_dir)

    assert any("analytical-control P2 source-verification" in error for error in errors)
    assert any("2026-06-08 Analytical-Control Source Verification Snapshot" in error for error in errors)


def test_server_return_package_required_paths_reject_symlinks(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_required_symlinks",
    )
    results_root = tmp_path / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = tmp_path / "paper"
    docs_dir = tmp_path / "docs"
    logs_dir = tmp_path / "logs"
    required_dirs = [
        results_dir / "analysis",
        results_root / "ranking_reversal",
        results_root / "sample_complexity_0.20",
        results_root / "reliability_recovery",
        paper_dir / "figdata",
        paper_dir / "figures",
        docs_dir,
        logs_dir,
    ]
    required_files = [
        results_dir / "results.csv",
        results_dir / "sweep_integrity_report.md",
        results_dir / "run_config.json",
        results_dir / "RUN_MANIFEST.json",
        results_dir / "analysis" / "test_set_sufficiency.csv",
        docs_dir / "case_validation_status.csv",
        docs_dir / "case_validation_sources.csv",
        docs_dir / "case_validation_note_sources.csv",
        docs_dir / "case_experiment_readiness.csv",
        docs_dir / "CASE_VALIDATION_SUMMARY.md",
        docs_dir / "CASE_WEB_VALIDATION_AUDIT.md",
        docs_dir / "EXPERIMENT_BENCHMARK_AUDIT.md",
        docs_dir / "PUBLICATION_LAUNCH_PREFLIGHT.md",
        docs_dir / "STATISTICAL_ANALYSIS_PLAN.md",
        paper_dir / "numbers.json",
        paper_dir / "numbers.tex",
        paper_dir / "figure_manifest.json",
        paper_dir / "figdata" / "figdata_manifest.json",
        logs_dir / "pipeline_command_history.txt",
    ]
    for path in required_dirs:
        path.mkdir(parents=True, exist_ok=True)
        (path / "sentinel.txt").write_text("x\n", encoding="utf-8")
    for path in required_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")

    symlink_like_file = results_dir / "results.csv"
    symlink_like_dir = paper_dir / "figures"
    path_type = type(symlink_like_file)
    original_is_symlink = path_type.is_symlink

    def fake_is_symlink(self):
        current = Path(self)
        if current in {symlink_like_file, symlink_like_dir}:
            return True
        return original_is_symlink(self)

    monkeypatch.setattr(path_type, "is_symlink", fake_is_symlink)

    errors = mod.check_required_return_paths(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        docs_dir=docs_dir,
        logs_dir=logs_dir,
    )

    assert any("full-sweep results is a symlink" in err for err in errors)
    assert any("figure directory is a symlink" in err for err in errors)


def test_server_return_package_required_base_dirs_reject_symlinks(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_base_symlinks",
    )
    results_root = tmp_path / "benchmark" / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = tmp_path / "paper"
    docs_dir = tmp_path / "docs"
    logs_dir = tmp_path / "logs"
    for path in (results_dir, paper_dir, docs_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(self):
        if self in {results_root, results_dir, paper_dir, docs_dir, logs_dir}:
            return True
        return original_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

    errors = mod.check_required_return_paths(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        docs_dir=docs_dir,
        logs_dir=logs_dir,
    )

    assert any("results root is a symlink" in err for err in errors)
    assert any("full-sweep result directory is a symlink" in err for err in errors)
    assert any("paper directory is a symlink" in err for err in errors)
    assert any("docs directory is a symlink" in err for err in errors)
    assert any("logs directory is a symlink" in err for err in errors)


def test_server_return_package_surfaces_downstream_analysis_gaps(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_downstream",
    )

    errors = mod.check_downstream_analysis_package(
        results_root=tmp_path / "results",
        results_dir=tmp_path / "results" / "full_benchmark",
        paper_dir=tmp_path / "paper",
    )

    assert any("correlation.json" in err for err in errors)
    assert any("figdata directory does not exist" in err for err in errors)
    assert any("figure_manifest.json does not exist" in err for err in errors)


def test_server_return_package_rejects_incomplete_submission_numbers(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_numbers_complete",
    )
    submission = _load_module(
        "scripts/check_submission_ready.py",
        "check_server_return_package_numbers_submission",
    )
    monkeypatch.setattr(submission, "check_numbers_consistency", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_numbers_provenance", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_numbers_tex_consistency", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_figure_manifest", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_figdata_schema", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_figdata_manifest", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_figdata_methods", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_required_analysis_artifacts", lambda *args, **kwargs: [])
    sys.modules["scripts.check_submission_ready"] = submission
    results_dir = tmp_path / "results" / "full_benchmark"
    paper_dir = tmp_path / "paper"
    results_dir.mkdir(parents=True)
    paper_dir.mkdir(parents=True)
    numbers = _complete_submission_numbers()
    numbers["dOneCILo"] = "N/A"
    (paper_dir / "numbers.json").write_text(json.dumps(numbers), encoding="utf-8")

    errors = mod.check_downstream_analysis_package(
        results_root=tmp_path / "results",
        results_dir=results_dir,
        paper_dir=paper_dir,
    )

    assert any("numbers.json dOneCILo is 'N/A'" in err for err in errors)


def test_server_return_package_rejects_numbers_provenance_drift(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_numbers_provenance",
    )
    submission = _load_module(
        "scripts/check_submission_ready.py",
        "check_server_return_package_numbers_provenance_submission",
    )
    monkeypatch.setattr(submission, "check_numbers_consistency", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_numbers_submission_completeness", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_numbers_tex_consistency", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_figure_manifest", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_figdata_schema", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_figdata_manifest", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_figdata_methods", lambda *args, **kwargs: [])
    monkeypatch.setattr(submission, "check_required_analysis_artifacts", lambda *args, **kwargs: [])
    sys.modules["scripts.check_submission_ready"] = submission
    results_root = tmp_path / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = tmp_path / "paper"
    results_dir.mkdir(parents=True)
    paper_dir.mkdir(parents=True)
    numbers = _write_numbers_with_provenance(tmp_path, submission)
    stale_input = tmp_path / "benchmark" / "results" / "full_benchmark" / "analysis" / "test_set_sufficiency.csv"
    stale_input.write_text("mutated\n", encoding="utf-8")
    target_numbers = paper_dir / "numbers.json"
    target_numbers.write_text(numbers.read_text(encoding="utf-8"), encoding="utf-8")
    original_root = submission.ROOT
    submission.ROOT = tmp_path
    try:
        errors = mod.check_downstream_analysis_package(
            results_root=results_root,
            results_dir=results_dir,
            paper_dir=paper_dir,
        )
    finally:
        submission.ROOT = original_root

    assert any("test_set_sufficiency" in err and "sha256 mismatch" in err for err in errors)


def test_server_return_package_skips_stale_downstream_checks_until_core_archive(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_skip_without_core",
    )
    results_root = tmp_path / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = tmp_path / "paper"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    paper_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (results_dir / "RUN_MANIFEST.json").write_text("{}", encoding="utf-8")

    groups = mod.check_server_return_package(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        logs_dir=logs_dir,
    )

    assert any("results.csv" in err for err in groups["required-paths"])
    assert groups["manifest"] == [
        "Skipped until results.csv, sweep_integrity_report.md, and RUN_MANIFEST.json "
        "are all present and non-empty."
    ]
    assert groups["downstream-analysis"] == [
        "Skipped until the core full-sweep archive is restored."
    ]


def test_server_return_package_skips_downstream_until_full_sweep_gate_passes(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_skip_until_full_sweep_pass",
    )
    results_root = tmp_path / "benchmark" / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = tmp_path / "paper"
    docs_dir = tmp_path / "docs"
    logs_dir = tmp_path / "logs"
    for path in [
        results_dir / "analysis",
        results_root / "ranking_reversal",
        results_root / "sample_complexity_0.20",
        results_root / "reliability_recovery",
        paper_dir / "figdata",
        paper_dir / "figures",
        docs_dir,
        logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
        (path / ".keep").write_text("x\n", encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\ntitle: DOE-Benchmark\n",
        encoding="utf-8",
    )
    _write_minimal_machine_readable_metadata(tmp_path)
    audit_columns = [
        "budget_mult",
        "case_id",
        "case_source_path",
        "case_source_sha256",
        "dim",
        "domain",
        "error",
        "method",
        "method_family",
        "method_source_path",
        "method_source_sha256",
        "n_output",
        "n_test",
        "n_train",
        "nrmse",
        "repeat",
        "seed",
        "status",
        "t_total_s",
        "test_seed",
    ]
    (results_dir / "results.csv").write_text(
        ",".join(audit_columns) + "\n",
        encoding="utf-8",
    )
    results_sha = _file_sha256(results_dir / "results.csv")
    results_size = (results_dir / "results.csv").stat().st_size
    _materialize_publication_python_environment(results_dir)
    run_config_json = results_dir / "run_config.json"
    _write_minimal_publication_run_config(run_config_json)
    run_config_sha = _file_sha256(run_config_json)
    run_config_size = run_config_json.stat().st_size
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )
    (results_dir / "runtime_health_report.txt").write_text(
        f"[warn] run_config.json contract matched partial results: {results_dir / 'run_config.json'}\n"
        f"[warn] run_config.json runtime controls recorded and valid: {results_dir / 'run_config.json'}\n"
        f"[warn] run_config.json runtime environment recorded and valid: {results_dir / 'run_config.json'}\n"
        f"[warn] run_config.json torch reproducibility policy recorded and valid: {results_dir / 'run_config.json'}\n"
        f"[warn] current runtime environment matches run_config.json snapshot: {results_dir / 'run_config.json'}\n"
        '[summary] runtime failure-category telemetry summarized: '
        'unique_cells=1 failure_category_counts={"ok": 1} '
        'non_ok_failure_category_counts={}\n'
        '[summary] runtime resource telemetry summarized: '
        'rss_gb_peak={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        '[summary] runtime wall-time telemetry summarized: '
        't_total_s={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        "[result] PASS - runtime sweep health is acceptable.\n"
        f"[artifact] results_csv={results_dir / 'results.csv'} "
        f"size_bytes={results_size} sha256={results_sha}\n"
        f"[artifact] run_config_json={run_config_json} "
        f"size_bytes={run_config_size} sha256={run_config_sha}\n",
        encoding="utf-8",
    )
    (results_dir / "RUN_MANIFEST.json").write_text("{}", encoding="utf-8")
    for path in [
        results_dir / "analysis" / "test_set_sufficiency.csv",
        docs_dir / "case_validation_status.csv",
        docs_dir / "case_validation_sources.csv",
        docs_dir / "case_validation_note_sources.csv",
        docs_dir / "case_experiment_readiness.csv",
        docs_dir / "CASE_VALIDATION_SUMMARY.md",
        docs_dir / "CASE_WEB_VALIDATION_AUDIT.md",
        docs_dir / "EXPERIMENT_BENCHMARK_AUDIT.md",
        docs_dir / "PUBLICATION_LAUNCH_PREFLIGHT.md",
        docs_dir / "STATISTICAL_ANALYSIS_PLAN.md",
        paper_dir / "numbers.json",
        paper_dir / "numbers.tex",
        paper_dir / "figure_manifest.json",
        paper_dir / "figdata" / "figdata_manifest.json",
        logs_dir / "pipeline_command_history.txt",
    ]:
        path.write_text("placeholder\n", encoding="utf-8")

    groups = mod.check_server_return_package(
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        docs_dir=docs_dir,
        logs_dir=logs_dir,
    )

    assert groups["required-paths"] == []
    assert any("Full sweep gate is BLOCKED" in err for err in groups["full-sweep"])
    assert groups["manifest"] == [
        "Skipped until the restored full-sweep gate passes; fix the "
        "full-sweep findings first so manifest diagnostics are tied to a "
        "trustworthy results.csv."
    ]
    assert groups["downstream-analysis"] == [
        "Skipped until the restored full-sweep gate passes."
    ]


def test_server_return_package_permission_error_message_is_actionable():
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_permission_error",
    )

    message = mod._filesystem_error(
        Path("paper/figures"),
        PermissionError(13, "Access denied", "paper/figures"),
    )

    assert "paper/figures" in message
    assert "Fix the file/directory ACL" in message


def test_server_return_package_rejects_manifest_case_resolution_error(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_manifest_case_resolution",
    )
    results_dir = tmp_path / "results" / "full_benchmark"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    results_csv = results_dir / "results.csv"
    results_csv.write_text("case_id\npressure_vessel_4d\n", encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )
    manifest_path = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest_path, results_csv=results_csv)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["experiment"]["case_resolution_error"] = "ImportError: no doe_testbed"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    (logs_dir / "pipeline_command_history.txt").write_text(
        "pipeline_invocation\tpython scripts/run_pipeline.py --profile full --confirm-full-run\n",
        encoding="utf-8",
    )

    errors = mod.check_manifest_package(manifest_path, logs_dir=logs_dir)

    assert any("case_resolution_error is non-empty" in err for err in errors)


def test_server_return_package_accepts_matching_command_history(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_command_history_ok",
    )
    results_dir = tmp_path / "results" / "full_benchmark"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    command_entries = [
        {
            "label": "pipeline_invocation",
            "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
        },
        {
            "label": "validate",
            "command": "python scripts/validate_run_config.py --pool nature",
        },
    ]
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "experiment": {
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_entries": command_entries,
            },
        }),
        encoding="utf-8",
    )
    (logs_dir / "pipeline_command_history.txt").write_text(
        "# DOE-Benchmark resolved pipeline stage commands\n"
        "pipeline_invocation\tpython scripts/run_pipeline.py --profile full --confirm-full-run\n"
        "validate\tpython scripts/validate_run_config.py --pool nature\n",
        encoding="utf-8",
    )

    assert mod.check_command_history_matches_manifest(
        results_dir / "RUN_MANIFEST.json",
        logs_dir=logs_dir,
    ) == []


def test_server_return_package_rejects_mismatched_command_history(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_command_history_bad",
    )
    results_dir = tmp_path / "results" / "full_benchmark"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "experiment": {
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )
    (logs_dir / "pipeline_command_history.txt").write_text(
        "pipeline_invocation\tpython scripts/run_pipeline.py --profile full --dry-run\n",
        encoding="utf-8",
    )

    errors = mod.check_command_history_matches_manifest(
        results_dir / "RUN_MANIFEST.json",
        logs_dir=logs_dir,
    )

    assert any("entries do not match" in err for err in errors)
    assert any("dry-run pipeline_invocation" in err for err in errors)


def test_server_return_package_rejects_malformed_command_history_file(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_command_history_file_bad",
    )
    results_dir = tmp_path / "results" / "full_benchmark"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "experiment": {
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )
    (logs_dir / "pipeline_command_history.txt").write_text(
        "# DOE-Benchmark resolved pipeline stage commands\n"
        "pipeline_invocation python scripts/run_pipeline.py --profile full --confirm-full-run\n"
        "\tpython scripts/validate_run_config.py --pool nature\n"
        "validate\t\n",
        encoding="utf-8",
    )

    errors = mod.check_command_history_matches_manifest(
        results_dir / "RUN_MANIFEST.json",
        logs_dir=logs_dir,
    )

    assert any("line 2 is not label<TAB>command" in err for err in errors)
    assert any("line 3 has blank label or command" in err for err in errors)
    assert any("line 4 has blank label or command" in err for err in errors)
    assert not any("entries do not match" in err for err in errors)


def test_server_return_package_rejects_malformed_requested_command_files(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_command_files_malformed",
    )
    results_dir = tmp_path / "results" / "full_benchmark"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "experiment": {
                "command_files_requested": [
                    "logs/pipeline_command_history.txt",
                    "",
                    123,
                ],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )
    (logs_dir / "pipeline_command_history.txt").write_text(
        "pipeline_invocation\tpython scripts/run_pipeline.py --profile full --confirm-full-run\n",
        encoding="utf-8",
    )

    errors = mod.check_command_history_matches_manifest(
        results_dir / "RUN_MANIFEST.json",
        logs_dir=logs_dir,
    )

    assert errors == [
        "RUN_MANIFEST experiment.command_files_requested contains "
        "non-string or blank entries"
    ]


def test_server_return_package_rejects_nonportable_requested_command_files(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_command_files_nonportable",
    )
    results_dir = tmp_path / "results" / "full_benchmark"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "experiment": {
                "command_files_requested": [
                    "logs/pipeline_command_history.txt",
                    "../logs/pipeline_command_history.txt",
                    r"C:\DOE-benchmark\logs\pipeline_command_history.txt",
                ],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )
    (logs_dir / "pipeline_command_history.txt").write_text(
        "pipeline_invocation\tpython scripts/run_pipeline.py --profile full --confirm-full-run\n",
        encoding="utf-8",
    )

    errors = mod.check_command_history_matches_manifest(
        results_dir / "RUN_MANIFEST.json",
        logs_dir=logs_dir,
    )

    assert len(errors) == 1
    assert "non-portable paths" in errors[0]
    assert "../logs/pipeline_command_history.txt" in errors[0]
    assert "C:" in errors[0]
    assert "DOE-benchmark" in errors[0]


def test_server_return_package_rejects_malformed_manifest_command_entries(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_command_history_malformed",
    )
    results_dir = tmp_path / "results" / "full_benchmark"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "experiment": {
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_entries": [
                    {"label": "pipeline_invocation", "command": ""},
                ],
            },
        }),
        encoding="utf-8",
    )
    (logs_dir / "pipeline_command_history.txt").write_text(
        "pipeline_invocation\tpython scripts/run_pipeline.py --profile full --confirm-full-run\n",
        encoding="utf-8",
    )

    errors = mod.check_command_history_matches_manifest(
        results_dir / "RUN_MANIFEST.json",
        logs_dir=logs_dir,
    )

    assert errors == ["RUN_MANIFEST experiment.command_entries contains malformed entries"]


def test_server_return_package_rejects_bad_manifest_pipeline_invocation(tmp_path):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_command_history_pipeline",
    )
    results_dir = tmp_path / "results" / "full_benchmark"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "pipeline_command_history.txt").write_text(
        "pipeline_invocation\tpython scripts/run_pipeline.py --profile full --confirm-full-run\n",
        encoding="utf-8",
    )

    manifest = {
        "experiment": {
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_entries": [
                {
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                },
                {
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --from-stage 3 --confirm-full-run",
                },
            ],
        },
    }
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = mod.check_command_history_matches_manifest(
        results_dir / "RUN_MANIFEST.json",
        logs_dir=logs_dir,
    )
    assert errors == [
        "RUN_MANIFEST experiment.command_entries must contain exactly one "
        "pipeline_invocation entry"
    ]

    manifest["experiment"]["command_entries"] = [{
        "label": "pipeline_invocation",
        "command": "python scripts/run_pipeline.py --profile full --dry-run",
    }]
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = mod.check_command_history_matches_manifest(
        results_dir / "RUN_MANIFEST.json",
        logs_dir=logs_dir,
    )
    assert errors == [
        "RUN_MANIFEST experiment.command_entries records a dry-run pipeline_invocation"
    ]

    manifest["experiment"]["command_entries"] = [{
        "label": "pipeline_invocation",
        "command": "python benchmark/scripts/run_full_benchmark.py --pool nature",
    }]
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    errors = mod.check_command_history_matches_manifest(
        results_dir / "RUN_MANIFEST.json",
        logs_dir=logs_dir,
    )
    assert errors == [
        "RUN_MANIFEST experiment.command_entries pipeline_invocation must use "
        "the formal full-pipeline entrypoint `scripts/run_pipeline.py` or "
        "`scripts/run_pipeline.sh`"
    ]


def test_server_launch_ready_preflight_never_includes_large_sweep():
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_no_sweep",
    )

    checks = mod.build_launch_checks(python="python")
    joined = "\n".join(mod.command_to_text(check.command) for check in checks)

    assert "run_case_physics_tests.py" in joined
    assert "nature-pool-cardinality" in {check.label for check in checks}
    assert "sweep-integrity-audit-contract" in {check.label for check in checks}
    assert "server-operator-contracts" in {check.label for check in checks}
    assert "check_optional_dependencies.py" in joined
    assert "--expected-methods 20" in joined
    assert "--require-formal-case-backends" in joined
    assert "--require-runtime-monitoring" in joined
    assert "benchmark/tests/test_sweep_integrity_audit.py" in joined
    assert "test_server_return_packaging_script_contract" in joined
    assert "test_create_server_return_package_import_guard_works_outside_repo_root" in joined
    assert "test_create_server_return_package_requires_passing_sweep_audit" in joined
    assert "test_create_server_return_package_requires_passing_runtime_health_report" in joined
    assert "test_server_return_package_validates_machine_readable_metadata_content" in joined
    assert "test_submission_ready_validates_machine_readable_root_metadata" in joined
    assert "test_submission_ready_cli_checks_machine_readable_root_metadata_by_default" in joined
    assert "test_create_server_return_package_requires_valid_machine_readable_metadata" in joined
    assert "test_create_server_return_package_rejects_stale_runtime_health_report" in joined
    assert "test_create_server_return_package_rejects_stale_runtime_health_run_config_fingerprint" in joined
    assert "test_create_server_return_package_requires_runtime_control_bound_health_report" in joined
    assert "test_create_server_return_package_requires_runtime_environment_bound_health_report" in joined
    assert "test_create_server_return_package_requires_torch_repro_bound_health_report" in joined
    assert "test_create_server_return_package_requires_current_environment_matched_health_report" in joined
    assert "test_create_server_return_package_requires_runtime_failure_category_summary" in joined
    assert "test_create_server_return_package_requires_runtime_resource_summary" in joined
    assert "test_create_server_return_package_requires_runtime_wall_time_summary" in joined
    assert "test_create_server_return_package_rejects_manifest_temp_storage_drift" in joined
    assert "test_create_server_return_package_rechecks_p2_web_audit_evidence_tiers" in joined
    assert "test_create_server_return_package_excludes_optional_robustness_dirs" in joined
    assert "test_generate_run_manifest_records_and_rejects_csv_header_contract_errors" in joined
    assert "test_sweep_runtime_health_rejects_run_config_missing_temp_storage_snapshot" in joined
    assert "test_sweep_runtime_health_rejects_malformed_torch_reproducibility_snapshot" in joined
    assert "test_sweep_runtime_health_rejects_malformed_run_config_temp_storage_snapshot" in joined
    assert "test_remote_progress_monitor_runs_runtime_health_gate" in joined
    assert "test_remote_progress_monitor_validates_tail_and_pid_inputs" in joined
    assert "test_remote_progress_monitor_points_missing_pid_to_formal_pipeline" in joined
    assert "test_experiment_audit_records_neurips_datasets_benchmarks_resource_controls" in joined
    assert "test_experiment_audit_maps_external_benchmark_practices_to_project_controls" in joined
    assert "test_experiment_audit_records_current_external_source_verification_matrix" in joined
    assert "test_experiment_audit_records_fair_nature_data_package_controls" in joined
    assert "test_server_launch_ready_rejects_p2_plausibility_without_nonvalidation_boundary" in joined
    assert "test_server_launch_ready_accepts_p2_plausibility_with_nonvalidation_boundary" in joined
    assert "test_server_return_package_rechecks_p2_web_audit_evidence_tiers" in joined
    assert "test_server_return_package_rejects_tarball_without_passing_sweep_audit" in joined
    assert "test_server_return_package_rejects_tarball_without_passing_runtime_health" in joined
    assert "test_server_return_package_rejects_tarball_with_stale_figdata_results_summary" in joined
    assert "test_server_return_package_rejects_tarball_with_stale_numbers_results_provenance" in joined
    assert "test_server_return_package_rejects_tarball_with_invalid_machine_readable_metadata" in joined
    assert "test_server_return_package_rejects_tarball_with_stale_runtime_health_fingerprint" in joined
    assert "test_server_return_package_rejects_tarball_with_stale_runtime_health_run_config_fingerprint" in joined
    assert "test_server_return_package_required_paths_rejects_stale_runtime_health_run_config_fingerprint" in joined
    assert "test_server_return_package_rejects_tarball_without_runtime_control_bound_health" in joined
    assert "test_server_return_package_rejects_tarball_without_runtime_environment_bound_health" in joined
    assert "test_server_return_package_rejects_tarball_without_torch_repro_bound_health" in joined
    assert "test_server_return_package_rejects_tarball_without_runtime_telemetry_summaries" in joined
    assert "test_server_return_package_rejects_optional_robustness_members" in joined
    assert "run_pipeline.py" in joined
    assert "--to-stage 1" in joined
    assert "--dry-run" in joined
    assert "run_full_benchmark.py" not in joined
    assert "--confirm-large-run" not in joined
    assert "audit_publication_readiness.py" in joined
    assert "--launch-preflight" in joined
    assert "PUBLICATION_LAUNCH_PREFLIGHT.md" in joined
    assert mod.validate_no_sweep_commands(checks) == []
    assert mod.validate_pipeline_dry_run_stage_graph(checks) == []
    assert mod.validate_case_web_validation_audit_coverage() == []
    assert mod.validate_p2_source_verification_snapshots() == []


def test_server_launch_ready_requires_aerodynamics_p2_source_snapshot(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_aero_p2_source_snapshot",
    )
    status_csv = tmp_path / "case_validation_status.csv"
    audit_md = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    status_csv.write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "wing_vlm_6d,aerodynamics,6,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_wing.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    audit_md.write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `wing_vlm_6d` | Evidence tier remains P2. Reference checked as "
        "boundary/context evidence, not validation data. | plausible | "
        "Keep P2 plausibility-only. |\n",
        encoding="utf-8",
    )

    errors = mod.validate_p2_source_verification_snapshots(
        status_csv=status_csv,
        audit_md=audit_md,
    )

    assert any("aerodynamics P2 source-verification" in error for error in errors)
    assert any("Additional `wing_vlm_6d` 2026-06-08 official-source boundary tightening:" in error for error in errors)


def test_server_launch_ready_requires_multidisciplinary_p2_source_snapshot(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_multidisciplinary_p2_source_snapshot",
    )
    status_csv = tmp_path / "case_validation_status.csv"
    audit_md = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    status_csv.write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "solar_pv_yield_6d,multidisciplinary,6,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_pv.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    audit_md.write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `solar_pv_yield_6d` | Evidence tier remains P2. Reference checked as "
        "boundary/context evidence, not validation data. | plausible | "
        "Keep P2 plausibility-only. |\n",
        encoding="utf-8",
    )

    errors = mod.validate_p2_source_verification_snapshots(
        status_csv=status_csv,
        audit_md=audit_md,
    )

    assert any("multidisciplinary P2 source-verification" in error for error in errors)
    assert any("Additional `solar_pv_yield_6d` 2026-06-08 official-source boundary tightening:" in error for error in errors)


def test_server_launch_ready_rejects_missing_p2_web_validation_rows(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_p2_web_coverage",
    )
    status_csv = tmp_path / "case_validation_status.csv"
    audit_md = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    status_csv.write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "covered_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_x.py,P2,plausibility only,keep\n"
        "missing_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_y.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    audit_md.write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `covered_case` | Evidence tier remains P2. Reference checked, not external validation. | plausible | Keep P2 plausibility-only. |\n",
        encoding="utf-8",
    )

    errors = mod.validate_case_web_validation_audit_coverage(
        status_csv=status_csv,
        audit_md=audit_md,
    )

    assert any("missing_case" in error for error in errors)


def test_server_launch_ready_rejects_p2_web_rows_without_decision(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_p2_web_decision",
    )
    status_csv = tmp_path / "case_validation_status.csv"
    audit_md = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    status_csv.write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "weak_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_x.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    audit_md.write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `weak_case` | reference checked | plausible | Still under review. |\n",
        encoding="utf-8",
    )

    errors = mod.validate_case_web_validation_audit_coverage(
        status_csv=status_csv,
        audit_md=audit_md,
    )

    assert any("without an explicit keep/fix/resweep judgment" in error for error in errors)


def test_server_launch_ready_rejects_p2_web_rows_without_evidence_tier(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_p2_web_tier",
    )
    status_csv = tmp_path / "case_validation_status.csv"
    audit_md = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    status_csv.write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "implicit_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_x.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    audit_md.write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `implicit_case` | reference checked | plausible | Keep P2 plausibility-only. |\n",
        encoding="utf-8",
    )

    errors = mod.validate_case_web_validation_audit_coverage(
        status_csv=status_csv,
        audit_md=audit_md,
    )

    assert any("markers do not match" in error for error in errors)
    assert any("missing `Evidence tier remains P2` marker" in error for error in errors)


def test_server_launch_ready_rejects_p2_web_analytic_tier_drift(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_p2_web_analytic_tier",
    )
    status_csv = tmp_path / "case_validation_status.csv"
    audit_md = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    status_csv.write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "analytic_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_x.py,P2,analytical control; not external validation,keep\n",
        encoding="utf-8",
    )
    audit_md.write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `analytic_case` | Evidence tier remains P2. Reference checked. | plausible | Keep P2. |\n",
        encoding="utf-8",
    )

    errors = mod.validate_case_web_validation_audit_coverage(
        status_csv=status_csv,
        audit_md=audit_md,
    )

    assert any("ledger tier targeted_analytical_control requires" in error for error in errors)


def test_server_launch_ready_rejects_p2_plausibility_described_as_analytic(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_p2_web_plausibility_tier",
    )
    status_csv = tmp_path / "case_validation_status.csv"
    audit_md = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    status_csv.write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "plausible_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_x.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    audit_md.write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `plausible_case` | Evidence tier remains P2 analytic-control. Reference checked. | plausible | Keep P2 plausibility-only. |\n",
        encoding="utf-8",
    )

    errors = mod.validate_case_web_validation_audit_coverage(
        status_csv=status_csv,
        audit_md=audit_md,
    )

    assert any("targeted_plausibility_only must not be described" in error for error in errors)


def test_server_launch_ready_rejects_p2_plausibility_without_nonvalidation_boundary(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_p2_web_nonvalidation_boundary",
    )
    status_csv = tmp_path / "case_validation_status.csv"
    audit_md = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    status_csv.write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "plausible_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_x.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    audit_md.write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `plausible_case` | Evidence tier remains P2. Reference checked. | plausible | Keep P2 for launch. |\n",
        encoding="utf-8",
    )

    errors = mod.validate_case_web_validation_audit_coverage(
        status_csv=status_csv,
        audit_md=audit_md,
    )

    assert any("requires explicit plausibility-only wording" in error for error in errors)
    assert any("non-validation boundary marker" in error for error in errors)


def test_server_launch_ready_accepts_p2_plausibility_with_nonvalidation_boundary(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_p2_web_nonvalidation_boundary_ok",
    )
    status_csv = tmp_path / "case_validation_status.csv"
    audit_md = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    status_csv.write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "plausible_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_x.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    audit_md.write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `plausible_case` | Evidence tier remains P2. Reference checked as boundary/context evidence, not validation data. | plausible | Keep P2 plausibility-only. |\n",
        encoding="utf-8",
    )

    errors = mod.validate_case_web_validation_audit_coverage(
        status_csv=status_csv,
        audit_md=audit_md,
    )

    assert errors == []


def test_local_pre_server_ready_wraps_requested_no_sweep_checklist():
    mod = _load_module(
        "scripts/check_local_pre_server_ready.py",
        "check_local_pre_server_ready_contract",
    )

    checks = mod.build_local_checks(python="python")
    joined = "\n".join(mod.command_to_text(check.command) for check in checks)
    labels = {check.label for check in checks}

    assert labels == {
        "doc-links",
        "case-source-fingerprints",
        "case-note-fingerprints",
        "case-validation-and-prereg",
        "pipeline-orchestrator",
        "sweep-integrity-audit-contract",
        "runtime-row-contracts",
        "server-operator-contracts",
        "d1-pairing-modes",
        "case-physics-tests",
        "paper-draft-ready",
    }
    for expected in [
        "check_doc_links.py",
        "update_case_validation_sources.py",
        "update_case_validation_note_sources.py",
        "tests/test_case_validation_status.py",
        "benchmark/tests/test_prereg_compliance.py",
        "benchmark/tests/test_pipeline_orchestrator.py",
        "benchmark/tests/test_sweep_integrity_audit.py",
        "benchmark/tests/test_runtime_contracts.py",
        "test_server_return_packaging_script_contract",
        "test_create_server_return_package_import_guard_works_outside_repo_root",
        "test_create_server_return_package_requires_passing_runtime_health_report",
        "test_server_return_package_validates_machine_readable_metadata_content",
        "test_submission_ready_validates_machine_readable_root_metadata",
        "test_submission_ready_cli_checks_machine_readable_root_metadata_by_default",
        "test_create_server_return_package_requires_valid_machine_readable_metadata",
        "test_create_server_return_package_rejects_stale_runtime_health_report",
        "test_create_server_return_package_rejects_stale_runtime_health_run_config_fingerprint",
        "test_create_server_return_package_requires_runtime_control_bound_health_report",
        "test_create_server_return_package_requires_runtime_environment_bound_health_report",
        "test_create_server_return_package_requires_torch_repro_bound_health_report",
        "test_create_server_return_package_requires_current_environment_matched_health_report",
        "test_create_server_return_package_requires_runtime_failure_category_summary",
        "test_create_server_return_package_requires_runtime_resource_summary",
        "test_create_server_return_package_requires_runtime_wall_time_summary",
        "test_create_server_return_package_rejects_manifest_temp_storage_drift",
        "test_create_server_return_package_rechecks_p2_web_audit_evidence_tiers",
        "test_create_server_return_package_excludes_optional_robustness_dirs",
        "test_generate_run_manifest_records_and_rejects_csv_header_contract_errors",
        "test_sweep_runtime_health_rejects_malformed_torch_reproducibility_snapshot",
        "test_remote_progress_monitor_runs_runtime_health_gate",
        "test_remote_progress_monitor_validates_tail_and_pid_inputs",
        "test_remote_progress_monitor_points_missing_pid_to_formal_pipeline",
        "test_experiment_audit_records_neurips_datasets_benchmarks_resource_controls",
        "test_experiment_audit_maps_external_benchmark_practices_to_project_controls",
        "test_experiment_audit_records_current_external_source_verification_matrix",
        "test_experiment_audit_records_fair_nature_data_package_controls",
        "test_server_launch_ready_rejects_p2_plausibility_without_nonvalidation_boundary",
        "test_server_launch_ready_accepts_p2_plausibility_with_nonvalidation_boundary",
        "test_server_return_package_rechecks_p2_web_audit_evidence_tiers",
        "test_server_return_package_rejects_tarball_without_passing_runtime_health",
        "test_server_return_package_rejects_tarball_with_stale_figdata_results_summary",
        "test_server_return_package_rejects_tarball_with_stale_numbers_results_provenance",
        "test_server_return_package_rejects_tarball_with_invalid_machine_readable_metadata",
        "test_server_return_package_rejects_tarball_with_stale_runtime_health_fingerprint",
        "test_server_return_package_rejects_tarball_with_stale_runtime_health_run_config_fingerprint",
        "test_server_return_package_required_paths_rejects_stale_runtime_health_run_config_fingerprint",
        "test_server_return_package_rejects_tarball_without_runtime_control_bound_health",
        "test_server_return_package_rejects_tarball_without_runtime_environment_bound_health",
        "test_server_return_package_rejects_tarball_without_torch_repro_bound_health",
        "test_server_return_package_rejects_tarball_without_runtime_telemetry_summaries",
        "test_server_return_package_rejects_optional_robustness_members",
        "benchmark/tests/test_new_methods.py::TestD1PairingModes",
        "run_case_physics_tests.py",
        "check_paper_draft_ready.py",
    ]:
        assert expected in joined
    assert "run_full_benchmark.py" not in joined
    assert "--confirm-large-run" not in joined
    assert "--confirm-full-run" not in joined
    assert mod.validate_no_sweep_commands(checks) == []


def test_local_and_server_preflights_share_server_operator_contract_list():
    local_mod = _load_module(
        "scripts/check_local_pre_server_ready.py",
        "check_local_pre_server_ready_operator_contract_list",
    )
    server_mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_operator_contract_list",
    )

    local_command = next(
        check.command
        for check in local_mod.build_local_checks(
            python="python",
            include_doc_links=False,
            include_case_physics=False,
            include_paper_draft=False,
        )
        if check.label == "server-operator-contracts"
    )
    server_command = next(
        check.command
        for check in server_mod.build_launch_checks(
            python="python",
            include_case_physics=False,
            include_pipeline_dry_run=False,
            include_publication_preflight=False,
        )
        if check.label == "server-operator-contracts"
    )

    def nodeids(command):
        return [
            item
            for item in command
            if "benchmark/tests/test_experiment_contracts.py::" in item
        ]

    assert nodeids(local_command) == nodeids(server_command)


def test_local_pre_server_ready_rejects_full_sweep_commands():
    mod = _load_module(
        "scripts/check_local_pre_server_ready.py",
        "check_local_pre_server_ready_safety",
    )
    checks = [
        mod.LocalCheck(
            "bad-sweep",
            ["python", "benchmark/scripts/run_full_benchmark.py", "--confirm-large-run"],
        ),
        mod.LocalCheck(
            "bad-confirm",
            ["python", "scripts/run_pipeline.py", "--profile", "full", "--confirm-full-run"],
        ),
    ]

    errors = mod.validate_no_sweep_commands(checks)

    assert any("run_full_benchmark.py" in err for err in errors)
    assert any("--confirm-large-run" in err or "--confirm-full-run" in err for err in errors)


def test_server_launch_ready_rejects_incomplete_full_dry_run_stage_graph(monkeypatch):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_stage_graph",
    )
    monkeypatch.setattr(
        mod,
        "REQUIRED_FULL_DRY_RUN_STAGES",
        (*mod.REQUIRED_FULL_DRY_RUN_STAGES, "missing_freeze_stage"),
    )
    checks = mod.build_launch_checks(python="python")

    errors = mod.validate_pipeline_dry_run_stage_graph(checks)

    assert any("missing_freeze_stage" in err for err in errors)


def test_server_launch_ready_indexes_required_shell_syntax_files():
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_shell_files",
    )

    required = set(mod.REQUIRED_SHELL_SYNTAX_FILES)

    assert "scripts/run_pipeline.sh" in required
    assert "scripts/check_progress_remote.sh" in required
    assert "scripts/create_server_return_package.sh" in required
    assert "scripts/launch_sweep_remote.sh" in required
    assert "scripts/launch_sota_robustness.sh" in required
    assert "scripts/launch_mfbo.sh" in required
    assert "scripts/launch_d1_pairings.sh" in required
    assert "scripts/_remote_lib.sh" in required


def test_server_launch_ready_allows_missing_results_csv_before_first_launch(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_missing_results",
    )

    errors, warnings = mod.validate_existing_results_resume_health(
        tmp_path / "missing_results.csv"
    )

    assert errors == []
    assert any("results.csv is missing" in warning for warning in warnings)


def test_server_launch_ready_warns_on_stale_derived_artifacts_without_results(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_stale_derived_without_results",
    )
    (tmp_path / "run_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "python_environment_pip_freeze.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "RUN_MANIFEST.json").write_text("{}", encoding="utf-8")
    (tmp_path / "report").mkdir()

    errors, warnings = mod.validate_existing_results_resume_health(
        tmp_path / "results.csv"
    )

    assert errors == []
    assert any("stale derived full-sweep artifact" in warning for warning in warnings)
    assert any("RUN_MANIFEST.json" in warning and "report" in warning for warning in warnings)
    assert any(
        "run_config.json" in warning and "python_environment_pip_freeze.txt" in warning
        for warning in warnings
    )


def test_server_launch_ready_can_require_clean_sweep_dir_without_results(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_requires_clean_missing_results",
    )
    (tmp_path / "run_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "python_environment_pip_freeze.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "RUN_MANIFEST.json").write_text("{}", encoding="utf-8")

    errors, warnings = mod.validate_existing_results_resume_health(
        tmp_path / "results.csv",
        require_clean_missing_results=True,
    )

    assert any("results.csv is missing" in warning for warning in warnings)
    assert any("clean full-sweep output directory required" in error for error in errors)
    assert any("RUN_MANIFEST.json" in error for error in errors)
    assert any(
        "run_config.json" in error and "python_environment_pip_freeze.txt" in error
        for error in errors
    )


def test_server_launch_ready_run_checks_honors_clean_sweep_dir_gate(monkeypatch):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_clean_sweep_dir_gate",
    )
    monkeypatch.setattr(mod, "validate_case_web_validation_audit_coverage", lambda: [])
    monkeypatch.setattr(mod, "validate_p2_source_verification_snapshots", lambda: [])
    monkeypatch.setattr(mod, "validate_shell_script_syntax", lambda: ([], []))
    monkeypatch.setattr(mod, "validate_server_resource_capacity", lambda: ([], []))

    observed = {}

    def fake_resume_health(*, require_clean_missing_results=False):
        observed["require_clean_missing_results"] = require_clean_missing_results
        if require_clean_missing_results:
            return ["clean full-sweep output directory required"], []
        return [], ["stale derived full-sweep artifact"]

    monkeypatch.setattr(mod, "validate_existing_results_resume_health", fake_resume_health)

    rc = mod.run_checks([], require_clean_sweep_dir=True)

    assert rc == 1
    assert observed == {"require_clean_missing_results": True}


def test_server_launch_ready_reports_clean_first_launch_missing_results_as_info(
    monkeypatch, capsys
):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_clean_first_launch_info",
    )
    monkeypatch.setattr(mod, "validate_case_web_validation_audit_coverage", lambda: [])
    monkeypatch.setattr(mod, "validate_p2_source_verification_snapshots", lambda: [])
    monkeypatch.setattr(mod, "validate_shell_script_syntax", lambda: ([], []))
    monkeypatch.setattr(mod, "validate_server_resource_capacity", lambda: ([], []))
    monkeypatch.setattr(
        mod,
        "validate_existing_results_resume_health",
        lambda **_kwargs: ([], ["results.csv is missing: /tmp/full_benchmark/results.csv"]),
    )

    rc = mod.run_checks([])

    out = capsys.readouterr().out
    assert rc == 0
    assert "[resume-health] INFO results.csv is missing" in out
    assert "[resume-health] WARN results.csv is missing" not in out


def test_server_launch_ready_times_out_hung_subprocess(monkeypatch, capsys):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_subprocess_timeout",
    )
    monkeypatch.setattr(mod, "validate_case_web_validation_audit_coverage", lambda: [])
    monkeypatch.setattr(mod, "validate_p2_source_verification_snapshots", lambda: [])
    monkeypatch.setattr(mod, "validate_shell_script_syntax", lambda: ([], []))
    monkeypatch.setattr(mod, "validate_server_resource_capacity", lambda: ([], []))
    monkeypatch.setattr(
        mod,
        "validate_existing_results_resume_health",
        lambda **_kwargs: ([], []),
    )

    def fake_run(command, *args, **kwargs):
        raise mod.subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    checks = [mod.LaunchCheck("timeout-case", ["python", "-c", "pass"])]

    rc = mod.run_checks(checks, check_timeout_s=0.1)

    out = capsys.readouterr().out
    assert rc == 1
    assert "[timeout-case] TIMEOUT" in out
    assert "limit=0.1s" in out
    assert "timeout-case:124" in out


def test_server_launch_ready_requires_run_config_for_existing_results_resume(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_missing_run_config",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status\n"
        "case_a,m1,5,0,ok\n",
        encoding="utf-8",
    )

    errors, warnings = mod.validate_existing_results_resume_health(csv_path)

    assert any("run_config.json is missing" in error for error in errors)
    assert any("results.csv has no t_total_s column yet" in warning for warning in warnings)


def test_server_launch_ready_rejects_corrupt_existing_results_csv(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_corrupt_results",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text('case_id,method\n"unterminated,m1\n', encoding="utf-8")

    errors, warnings = mod.validate_existing_results_resume_health(csv_path)

    assert any("different number of fields than the header" in error for error in errors)
    assert warnings == []


def test_server_launch_ready_blocks_when_existing_results_health_fails(monkeypatch):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resume_health_gate",
    )
    monkeypatch.setattr(mod, "validate_case_web_validation_audit_coverage", lambda: [])
    monkeypatch.setattr(mod, "validate_shell_script_syntax", lambda: ([], []))
    monkeypatch.setattr(
        mod,
        "validate_existing_results_resume_health",
        lambda *args, **kwargs: (["duplicate protocol rows"], []),
    )

    rc = mod.run_checks([])

    assert rc == 1


def test_server_launch_ready_shell_syntax_skips_missing_bash_on_windows(monkeypatch):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_shell_skip_windows",
    )
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(mod.os, "name", "nt")

    errors, messages = mod.validate_shell_script_syntax()

    assert errors == []
    assert any("shell syntax will be checked on Linux server" in msg for msg in messages)
    assert any("LF line endings PASS" in msg for msg in messages)


def test_server_launch_ready_shell_syntax_skips_unusable_windows_bash(monkeypatch):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_shell_skip_unusable_windows_bash",
    )

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "Windows Subsystem for Linux has no installed distributions"

    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        return Completed()

    monkeypatch.setattr(mod.shutil, "which", lambda name: r"C:\Windows\System32\bash.exe")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.os, "name", "nt")

    errors, messages = mod.validate_shell_script_syntax()

    assert errors == []
    assert calls == [[r"C:\Windows\System32\bash.exe", "--version"]]
    assert any("not usable" in msg for msg in messages)
    assert any("shell syntax will be checked on Linux server" in msg for msg in messages)


def test_server_launch_ready_shell_syntax_uses_working_windows_bash(monkeypatch):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_shell_uses_working_windows_bash",
    )

    class Completed:
        returncode = 0
        stdout = "GNU bash"
        stderr = ""

    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        return Completed()

    monkeypatch.setattr(mod.shutil, "which", lambda name: r"C:\Program Files\Git\bin\bash.exe")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.os, "name", "nt")
    monkeypatch.setattr(mod, "REQUIRED_SHELL_SYNTAX_FILES", ("scripts/run_pipeline.sh",))

    errors, messages = mod.validate_shell_script_syntax()

    assert errors == []
    assert calls == [
        [r"C:\Program Files\Git\bin\bash.exe", "--version"],
        [r"C:\Program Files\Git\bin\bash.exe", "-n", str(mod.ROOT / "scripts/run_pipeline.sh")],
    ]
    assert any("scripts/run_pipeline.sh: bash -n PASS" in msg for msg in messages)


def test_server_launch_ready_rejects_crlf_shell_wrappers_even_on_windows(
    tmp_path,
    monkeypatch,
):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_shell_crlf",
    )
    script = tmp_path / "run_pipeline.sh"
    script.write_bytes(b"#!/usr/bin/env bash\r\necho bad\r\n")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "REQUIRED_SHELL_SYNTAX_FILES", ("run_pipeline.sh",))
    monkeypatch.setattr(mod.os, "name", "nt")
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    errors, messages = mod.validate_shell_script_syntax()

    assert any("CRLF/CR line endings detected" in err for err in errors)
    assert any("shell syntax will be checked on Linux server" in msg for msg in messages)


def test_server_launch_ready_shell_syntax_requires_bash_on_linux(monkeypatch):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_shell_requires_bash",
    )
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(mod.os, "name", "posix")

    errors, messages = mod.validate_shell_script_syntax()

    assert any("LF line endings PASS" in msg for msg in messages)
    assert errors == [
        "bash executable not found; install bash before running the server launch wrappers"
    ]


def test_server_launch_ready_shell_syntax_surfaces_bash_n_failures(monkeypatch):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_shell_bash_n_failure",
    )

    class Completed:
        returncode = 2
        stdout = ""
        stderr = "syntax error near unexpected token\n"

    monkeypatch.setattr(mod.shutil, "which", lambda name: "/bin/bash")
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(mod, "REQUIRED_SHELL_SYNTAX_FILES", ("scripts/run_pipeline.sh",))

    errors, messages = mod.validate_shell_script_syntax(bash_executable="/bin/bash")

    assert any("LF line endings PASS" in msg for msg in messages)
    assert errors == [
        "scripts/run_pipeline.sh: bash -n failed with exit=2: "
        "syntax error near unexpected token"
    ]


def test_server_launch_ready_resource_check_skips_on_windows(monkeypatch):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resource_skip_windows",
    )
    monkeypatch.setattr(mod.os, "name", "nt")

    errors, messages = mod.validate_server_resource_capacity()

    assert errors == []
    assert any("RAM/disk capacity will be checked on Linux server" in msg for msg in messages)


def test_server_launch_ready_resource_check_rejects_low_ram(monkeypatch, tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resource_low_ram",
    )
    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod, "_detect_total_ram_gb", lambda: 31.5)

    errors, messages = mod.validate_server_resource_capacity(
        min_ram_gb=64.0,
        min_disk_free_gb=1.0,
        paths=(tmp_path,),
    )

    assert any("effective server RAM 31.5 GB" in err and "below the launch minimum 64.0 GB" in err for err in errors)
    assert any("free disk space" in msg for msg in messages)


def test_server_launch_ready_resource_check_rejects_low_disk(monkeypatch, tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resource_low_disk",
    )

    class Usage:
        total = 100 * 1024**3
        used = 99 * 1024**3
        free = 1 * 1024**3

    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod, "_detect_total_ram_gb", lambda: 128.0)
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda path: Usage())

    errors, messages = mod.validate_server_resource_capacity(
        min_ram_gb=64.0,
        min_disk_free_gb=25.0,
        paths=(tmp_path,),
    )

    assert any("free disk space" in err and "below the launch minimum 25.0 GB" in err for err in errors)
    assert any("effective server RAM 128.0 GB" in msg and ">= required 64.0 GB" in msg for msg in messages)


def test_server_launch_ready_resource_check_uses_cgroup_limit(monkeypatch, tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resource_cgroup_limit",
    )
    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod, "_detect_total_ram_gb", lambda: 256.0)
    monkeypatch.setattr(
        mod,
        "_detect_cgroup_memory_limit_gb",
        lambda: (31.5, "cgroup v2 memory.max"),
    )

    errors, messages = mod.validate_server_resource_capacity(
        min_ram_gb=64.0,
        min_disk_free_gb=1.0,
        paths=(tmp_path,),
    )

    assert any("effective server RAM 31.5 GB" in err for err in errors)
    assert any("cgroup v2 memory.max" in err for err in errors)
    assert any("physical RAM 256.0 GB" in err for err in errors)
    assert any("free disk space" in msg for msg in messages)


def test_server_launch_ready_resource_check_ignores_unlimited_cgroup(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resource_unlimited_cgroup",
    )
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "memory.limit_in_bytes").write_text(str(2**63 - 1), encoding="utf-8")

    limit_gb, source = mod._detect_cgroup_memory_limit_gb(tmp_path)

    assert limit_gb is None
    assert source is None


def test_server_launch_ready_resource_check_reads_current_v1_memory_cgroup(tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_current_v1_cgroup",
    )
    proc_cgroup = tmp_path / "proc_self_cgroup"
    proc_cgroup.write_text(
        "7:cpu,cpuacct:/ignored\n"
        "5:memory:/slurm/uid_1000/job_7\n",
        encoding="utf-8",
    )
    root_limit = tmp_path / "memory" / "memory.limit_in_bytes"
    root_limit.parent.mkdir(parents=True)
    root_limit.write_text(str(512 * 1024**3), encoding="utf-8")
    job_dir = tmp_path / "memory" / "slurm" / "uid_1000" / "job_7"
    job_dir.mkdir(parents=True)
    (job_dir / "memory.limit_in_bytes").write_text(str(24 * 1024**3), encoding="utf-8")

    limit_gb, source = mod._detect_cgroup_memory_limit_gb(tmp_path, proc_cgroup)

    assert limit_gb == 24.0
    assert source == "cgroup v1 memory.limit_in_bytes"


def test_server_launch_ready_resource_check_rejects_bad_env_threshold(monkeypatch, tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resource_bad_env",
    )
    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setenv("DOE_BENCHMARK_MIN_SERVER_RAM_GB", "not-a-number")
    monkeypatch.setenv("DOE_BENCHMARK_MIN_SERVER_DISK_FREE_GB", "-1")
    monkeypatch.setattr(mod, "_detect_total_ram_gb", lambda: 128.0)

    errors, _messages = mod.validate_server_resource_capacity(paths=(tmp_path,))

    assert any("DOE_BENCHMARK_MIN_SERVER_RAM_GB must be a positive finite number" in err for err in errors)
    assert any("DOE_BENCHMARK_MIN_SERVER_DISK_FREE_GB must be a positive finite number" in err for err in errors)


def test_server_launch_ready_resource_check_includes_temp_storage_disk(monkeypatch, tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resource_temp_storage_disk",
    )

    class Usage:
        total = 100 * 1024**3
        used = 98 * 1024**3
        free = 2 * 1024**3

    scratch = tmp_path / "scratch" / "job_42" / "tmp"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod, "_detect_total_ram_gb", lambda: 128.0)
    monkeypatch.setattr(mod, "_detect_cgroup_memory_limit_gb", lambda: (None, None))
    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(scratch))
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda path: Usage())

    errors, messages = mod.validate_server_resource_capacity(
        min_ram_gb=64.0,
        min_disk_free_gb=25.0,
        paths=(),
    )

    assert any(
        "free disk space" in err and str(scratch) in err and "below the launch minimum 25.0 GB" in err
        for err in errors
    )
    assert any("effective server RAM 128.0 GB" in msg for msg in messages)


def test_server_launch_ready_resource_check_rejects_unwritable_temp_storage(monkeypatch, tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resource_unwritable_temp_storage",
    )
    scratch = tmp_path / "scratch" / "job_7" / "tmp"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod, "_detect_total_ram_gb", lambda: 128.0)
    monkeypatch.setattr(mod, "_detect_cgroup_memory_limit_gb", lambda: (None, None))
    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(scratch))
    original_named_temporary_file = mod.tempfile.NamedTemporaryFile
    scratch_text = str(scratch).replace("\\", "/").rstrip("/")

    def fake_named_temporary_file(*args, **kwargs):
        directory = str(kwargs.get("dir", "")).replace("\\", "/").rstrip("/")
        if directory == scratch_text:
            raise PermissionError("simulated no write access")
        return original_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(mod.tempfile, "NamedTemporaryFile", fake_named_temporary_file)

    errors, _messages = mod.validate_server_resource_capacity(
        min_ram_gb=64.0,
        min_disk_free_gb=0.1,
        paths=(),
    )

    assert any(
        "temporary/scratch path is not writable" in err and str(scratch) in err
        for err in errors
    )


def test_server_launch_ready_resource_check_write_probe_cleans_up(monkeypatch, tmp_path):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_resource_write_probe_cleanup",
    )
    scratch = tmp_path / "scratch" / "job_8" / "tmp"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod, "_detect_total_ram_gb", lambda: 128.0)
    monkeypatch.setattr(mod, "_detect_cgroup_memory_limit_gb", lambda: (None, None))
    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(scratch))

    errors, _messages = mod.validate_server_resource_capacity(
        min_ram_gb=64.0,
        min_disk_free_gb=0.1,
        paths=(),
    )

    assert errors == []
    assert not list(scratch.glob(".doe_benchmark_write_probe_*.tmp"))


def test_server_launch_ready_runtime_safety_rejects_large_sweep_commands():
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_runtime_safety_large",
    )

    checks = [
        mod.LaunchCheck(
            "bad-sweep",
            [
                "python",
                "benchmark/scripts/run_full_benchmark.py",
                "--confirm-large-run",
            ],
        )
    ]

    errors = mod.validate_no_sweep_commands(checks)

    assert any("run_full_benchmark.py" in err for err in errors)
    assert any("--confirm-large-run" in err for err in errors)


def test_server_launch_ready_runtime_safety_rejects_unbounded_full_pipeline(capsys):
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_runtime_safety_pipeline",
    )

    checks = [
        mod.LaunchCheck(
            "bad-pipeline",
            [
                "python",
                "scripts/run_pipeline.py",
                "--profile",
                "full",
                "--confirm-full-run",
            ],
        )
    ]

    assert mod.run_checks(checks, dry_run=True) == 1
    out = capsys.readouterr().out
    assert "confirmed full pipeline command is not limited" in out
    assert "DRY-RUN - no checks executed" not in out


def test_server_launch_ready_can_skip_case_physics():
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_skip_physics",
    )

    checks = mod.build_launch_checks(
        python="python",
        include_case_physics=False,
    )
    joined = "\n".join(mod.command_to_text(check.command) for check in checks)

    assert "run_case_physics_tests.py" not in joined
    assert "tests/test_case_validation_status.py" in joined


def test_server_launch_ready_can_skip_pipeline_dry_run():
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_skip_dry_run",
    )

    checks = mod.build_launch_checks(
        python="python",
        include_pipeline_dry_run=False,
    )
    joined = "\n".join(mod.command_to_text(check.command) for check in checks)

    assert "--to-stage 1" in joined
    assert "--dry-run" not in joined


def test_server_launch_ready_can_skip_publication_preflight():
    mod = _load_module(
        "scripts/check_server_launch_ready.py",
        "check_server_launch_ready_skip_publication",
    )

    checks = mod.build_launch_checks(
        python="python",
        include_publication_preflight=False,
    )
    joined = "\n".join(mod.command_to_text(check.command) for check in checks)

    assert "audit_publication_readiness.py" not in joined
    assert "--launch-preflight" not in joined


def test_repository_gitattributes_preserves_linux_server_line_endings():
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "* text=auto eol=lf" in normalized
    assert "*.sh text eol=lf" not in normalized
    assert "*.ps1 text eol=crlf" in normalized
    assert "*.png binary" in normalized
    assert "*.pdf binary" in normalized


def test_server_launch_ready_docs_describe_publication_preflight():
    launch = (ROOT / "scripts" / "check_server_launch_ready.py").read_text(encoding="utf-8")
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "docs" / "PROJECT_STATUS.md").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")
    reproducibility = (ROOT / "docs" / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    normalized_handoff = " ".join(handoff.split())

    assert "publication launch-preflight" in launch
    assert "publication launch-preflight" in readme
    assert "publication launch-preflight" in status
    assert "PUBLICATION_LAUNCH_PREFLIGHT.md" in readme
    assert "--expected-methods 20" in handoff
    assert "--require-formal-case-backends" in handoff
    assert "--require-runtime-monitoring" in handoff
    assert "D1 pairing-mode tests" in handoff
    assert "64 GB effective RAM" in normalized_handoff
    assert "25 GB free disk space" in normalized_handoff
    assert "DOE_BENCHMARK_MIN_SERVER_RAM_GB" in handoff
    assert "DOE_BENCHMARK_MIN_SERVER_DISK_FREE_GB" in handoff
    assert "resource-capacity check is skipped" in handoff
    assert "20-method" in readme
    assert "without absolute paths or `../` parent" in handoff
    assert "Date signed (UTC+08:00)" in handoff
    assert "2026-06-03T10:00:00+08:00" in handoff
    assert "Date signed (UTC+08:00)" in reproducibility
    assert "+08:00" in reproducibility


def test_optional_dependency_preflight_allows_nature_without_botorch(monkeypatch):
    mod = _load_module("scripts/check_optional_dependencies.py", "optional_deps_no_botorch")
    monkeypatch.setattr(
        mod,
        "_installed_modules",
        lambda: {"torch": True, "gpytorch": False, "botorch": False},
    )

    rc, messages = mod.check_pool("nature")

    assert rc == 0
    assert any("diagnostic fallback" in msg for msg in messages)


def test_optional_dependency_preflight_can_require_botorch_for_formal_nature(monkeypatch):
    mod = _load_module("scripts/check_optional_dependencies.py", "optional_deps_require_botorch")
    monkeypatch.setattr(
        mod,
        "_installed_modules",
        lambda: {"torch": True, "gpytorch": False, "botorch": False},
    )

    rc, messages = mod.check_pool("nature", require_botorch=True)

    assert rc == 1
    assert any("requires BoTorch-compatible methods" in msg for msg in messages)


def test_optional_dependency_preflight_rejects_wrong_method_count(monkeypatch):
    mod = _load_module("scripts/check_optional_dependencies.py", "optional_deps_method_count")
    monkeypatch.setattr(
        mod,
        "_installed_modules",
        lambda: {"torch": True, "gpytorch": True, "botorch": True},
    )
    monkeypatch.setattr(mod, "_probe_botorch_imports", lambda: [])
    monkeypatch.setattr(
        mod,
        "_pool_method_names",
        lambda pool: (["botorch_qei", "botorch_saasbo"], None),
    )

    rc, messages = mod.check_pool("nature", require_botorch=True, expected_methods=20)

    assert rc == 1
    assert any("expected 20" in msg for msg in messages)


def test_optional_dependency_preflight_rejects_incompatible_botorch(monkeypatch):
    mod = _load_module("scripts/check_optional_dependencies.py", "optional_deps_bad_botorch")
    monkeypatch.setattr(
        mod,
        "_installed_modules",
        lambda: {"torch": True, "gpytorch": True, "botorch": True},
    )
    monkeypatch.setattr(
        mod,
        "_probe_botorch_imports",
        lambda: ["BoTorch core imports failed: cannot import name 'MAP_LOCATION'"],
    )

    rc, messages = mod.check_pool("nature")

    assert rc == 1
    assert any("MAP_LOCATION" in msg for msg in messages)
    assert any("Fix the environment before launching long sweeps" in msg for msg in messages)


def test_optional_dependency_preflight_requires_sota_methods(monkeypatch):
    mod = _load_module("scripts/check_optional_dependencies.py", "optional_deps_sota_methods")
    monkeypatch.setattr(
        mod,
        "_installed_modules",
        lambda: {"torch": True, "gpytorch": True, "botorch": True},
    )
    monkeypatch.setattr(mod, "_probe_botorch_imports", lambda: [])
    monkeypatch.setattr(
        mod,
        "_pool_method_names",
        lambda pool: (["botorch_qei", "botorch_saasbo"], None),
    )

    rc, messages = mod.check_pool("sota")

    assert rc == 1
    assert any("botorch_turbo" in msg for msg in messages)


def test_optional_dependency_preflight_can_require_formal_case_backends(monkeypatch):
    mod = _load_module(
        "scripts/check_optional_dependencies.py",
        "optional_deps_formal_case_backends",
    )
    monkeypatch.setattr(
        mod,
        "_installed_modules",
        lambda names=mod.BO_MODULES: {name: True for name in names},
    )
    monkeypatch.setattr(mod.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(mod, "_probe_botorch_imports", lambda: [])
    monkeypatch.setattr(
        mod,
        "_pool_method_names",
        lambda pool: (["botorch_qei", "botorch_saasbo"] + [f"m{i}" for i in range(18)], None),
    )

    rc, messages = mod.check_pool(
        "nature",
        require_botorch=True,
        expected_methods=20,
        require_formal_case_backends=True,
    )

    assert rc == 0
    assert any("formal case backend preflight PASS" in msg for msg in messages)


def test_optional_dependency_preflight_rejects_missing_formal_case_backend(monkeypatch):
    mod = _load_module(
        "scripts/check_optional_dependencies.py",
        "optional_deps_missing_formal_case_backend",
    )
    monkeypatch.setattr(
        mod,
        "_installed_modules",
        lambda names=mod.BO_MODULES: {name: (name != "neuralfoil") for name in names},
    )

    rc, messages = mod.check_pool("default", require_formal_case_backends=True)

    assert rc == 1
    assert any("formal case backend 'neuralfoil' is missing" in msg for msg in messages)


def test_optional_dependency_preflight_can_require_runtime_monitoring(monkeypatch):
    mod = _load_module(
        "scripts/check_optional_dependencies.py",
        "optional_deps_runtime_monitoring",
    )
    monkeypatch.setattr(
        mod,
        "_installed_modules",
        lambda names=mod.BO_MODULES: {name: True for name in names},
    )
    monkeypatch.setattr(mod.importlib, "import_module", lambda name: object())

    rc, messages = mod.check_pool("default", require_runtime_monitoring=True)

    assert rc == 0
    assert any("runtime monitoring preflight PASS" in msg for msg in messages)


def test_optional_dependency_preflight_rejects_missing_runtime_monitoring(monkeypatch):
    mod = _load_module(
        "scripts/check_optional_dependencies.py",
        "optional_deps_missing_runtime_monitoring",
    )
    monkeypatch.setattr(
        mod,
        "_installed_modules",
        lambda names=mod.BO_MODULES: {name: False for name in names},
    )

    rc, messages = mod.check_pool("default", require_runtime_monitoring=True)

    assert rc == 1
    assert any("runtime monitoring dependency 'psutil' is missing" in msg for msg in messages)
    assert any("RSS traces and memory caps" in msg for msg in messages)


def test_remote_optional_launchers_run_dependency_preflight():
    sota = (ROOT / "scripts" / "launch_sota_robustness.sh").read_text(encoding="utf-8")
    mfbo = (ROOT / "scripts" / "launch_mfbo.sh").read_text(encoding="utf-8")
    d1 = (ROOT / "scripts" / "launch_d1_pairings.sh").read_text(encoding="utf-8")

    assert "check_optional_dependencies.py\" --pool sota" in sota
    assert "check_optional_dependencies.py\" --pool mf" in mfbo
    assert "check_optional_dependencies.py\"" in d1
    assert "--pool nature --require-botorch --expected-methods 20" in d1


def test_remote_mf_launcher_caps_default_parallelism_after_autotune():
    text = (ROOT / "scripts" / "launch_mfbo.sh").read_text(encoding="utf-8")
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    assert 'USER_SET_N_JOBS="${N_JOBS+x}"' in text
    assert "auto_tune_defaults" in text
    assert 'if [[ -z "${USER_SET_N_JOBS}" ]]; then' in text
    assert "N_JOBS=5" in text
    assert "export N_JOBS" in text
    assert "unless the operator explicitly" in text
    assert text.index("auto_tune_defaults") < text.index('if [[ -z "${USER_SET_N_JOBS}" ]]; then')
    assert text.index('if [[ -z "${USER_SET_N_JOBS}" ]]; then') < text.index(
        "validate_remote_launch_tunables"
    )
    assert "default-capped to `N_JOBS=5` unless the operator explicitly overrides it" in readme


def test_remote_d1_pairing_launcher_matches_formal_stage4_protocol():
    text = (ROOT / "scripts" / "launch_d1_pairings.sh").read_text(encoding="utf-8")

    assert "--pool nature" in text
    assert "--pairing-mode all" in text
    assert "--budgets 10 20" in text
    assert "--n-test 1000" in text
    assert "--seed-strategy case_repeat" in text
    assert "--seed-start 1000" in text
    assert "--skip-existing" in text
    assert "benchmark/results/ranking_reversal" in text
    assert "benchmark/results/full_benchmark" in text
    assert "benchmark/results/sota_robustness" in text
    assert "benchmark/results/mf_supplementary" in text
    assert "D1 pairing output must not be benchmark/results/full_benchmark" in text


def test_remote_d1_pairing_launcher_defines_shared_tunables_before_validation():
    text = (ROOT / "scripts" / "launch_d1_pairings.sh").read_text(encoding="utf-8")

    for assignment in [
        'WALL_H="${WALL_H:-24}"',
        'HEARTBEAT_S="${HEARTBEAT_S:-60}"',
    ]:
        assert assignment in text
        assert text.index(assignment) < text.index("validate_remote_launch_tunables")


def test_remote_full_launcher_requires_botorch_preflight():
    text = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")

    assert "scripts/check_optional_dependencies.py" in text
    assert "--pool nature" in text
    assert "--require-botorch" in text
    assert "--expected-methods 20" in text
    assert "--require-formal-case-backends" in text
    assert "--require-runtime-monitoring" in text
    assert "--confirm-large-run" in text


def test_remote_full_launcher_runs_case_readiness_preflight():
    text = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")

    assert "scripts/audit_case_experiment_readiness.py" in text
    assert "--output docs/case_experiment_readiness.csv" in text
    assert "--fail-on-warning" in text
    assert "case experiment-readiness pre-flight failed" in text
    assert text.index("scripts/check_optional_dependencies.py") < text.index(
        "scripts/audit_case_experiment_readiness.py"
    )
    assert text.index("scripts/audit_case_experiment_readiness.py") < text.index("PY_CMD=(")


def test_remote_full_launcher_checks_existing_results_before_resume():
    text = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")

    assert "scripts/check_sweep_runtime_health.py" in text
    assert '"${OUT_DIR}/results.csv"' in text
    assert '--run-config-json "${OUT_DIR}/run_config.json"' in text
    assert "--require-run-config" in text
    assert "--validate-current-environment" in text
    assert "--allow-missing" in text
    assert "existing results.csv is not safe to resume" in text
    assert text.index("scripts/audit_case_experiment_readiness.py") < text.index(
        "scripts/check_sweep_runtime_health.py"
    )
    assert text.index("scripts/check_sweep_runtime_health.py") < text.index("PY_CMD=(")
    assert "checks any existing headline `results.csv` with the runtime health gate" in readme
    assert "`launch_sweep_remote.sh` wrapper runs the same read-only runtime health gate" in handoff
    assert "valid `run_config.json` startup snapshot" in handoff
    assert "bad-contract partial CSV blocks resume" in handoff


def test_remote_full_launcher_prints_memory_and_disk_monitor_command():
    text = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")
    status = (ROOT / "docs" / "PROJECT_STATUS.md").read_text(encoding="utf-8")

    assert "MEM_CAP_GB=${MEM_CAP_GB} DISK_WARNING_GB=${DISK_WARNING_GB:-25} bash scripts/check_progress_remote.sh -w" in text
    assert (
        "VALIDATE_CURRENT_ENV=1 MEM_CAP_GB=${MEM_CAP_GB} "
        "DISK_WARNING_GB=${DISK_WARNING_GB:-25} bash scripts/check_progress_remote.sh -w"
    ) in text
    assert "MEM_CAP_GB=56 DISK_WARNING_GB=25 bash scripts/check_progress_remote.sh -w" in status
    assert "VALIDATE_CURRENT_ENV=1 MEM_CAP_GB=56 DISK_WARNING_GB=25 bash scripts/check_progress_remote.sh -w" in status
    assert (
        "MEM_CAP_GB=${MEM_CAP_GB} DISK_WARNING_GB=${DISK_WARNING_GB:-25} "
        "STALE_AFTER_MIN=${STALE_AFTER_MIN:-45} FAIL_ON_STALE=1 "
        "bash scripts/check_progress_remote.sh -w"
    ) in text
    assert (
        "MEM_CAP_GB=56 DISK_WARNING_GB=25 STALE_AFTER_MIN=45 FAIL_ON_STALE=1 "
        "bash scripts/check_progress_remote.sh -w"
    ) in status


def test_remote_launchers_keep_auto_bo_parallelism_positive():
    remote = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")
    shared = (ROOT / "scripts" / "_remote_lib.sh").read_text(encoding="utf-8")
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    assert '[[ "${BO_N_JOBS_AUTO}" -lt 1 ]] && BO_N_JOBS_AUTO=1' in remote
    assert '[[ "${bo_n_jobs_auto}" -lt 1 ]] && bo_n_jobs_auto=1' in shared
    assert remote.index("RAM_LIMITED_BO") < remote.rindex(
        '[[ "${BO_N_JOBS_AUTO}" -lt 1 ]] && BO_N_JOBS_AUTO=1'
    )
    assert shared.index("ram_limited") < shared.rindex(
        '[[ "${bo_n_jobs_auto}" -lt 1 ]] && bo_n_jobs_auto=1'
    )
    assert "keeping BoTorch parallelism at a positive integer" in readme
    assert "rejects non-positive/invalid environment overrides" in readme
    assert "rejects unknown persistence modes" in readme


def test_remote_launchers_autotune_ram_from_current_cgroup_limit():
    remote = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")
    shared = (ROOT / "scripts" / "_remote_lib.sh").read_text(encoding="utf-8")
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")

    for text in (remote, shared):
        assert "detect_cgroup_memory_limit_gb" in text
        assert "emit_cgroup_memory_limit_candidates" in text
        assert "CGROUP_PROC_SELF" in text
        assert "CGROUP_ROOT" in text
        assert "memory.max" in text
        assert "memory.limit_in_bytes" in text
        assert "1152921504606846976" in text
        assert 'cgroup_limit_gb="$(detect_cgroup_memory_limit_gb || true)"' in text
        assert text.index('cgroup_limit_gb="$(detect_cgroup_memory_limit_gb || true)"') < text.index("if [[ -r /proc/meminfo ]]")

    assert "auto-tunes `N_JOBS`, `BO_N_JOBS`, and `MEM_CAP_GB` from the effective RAM" in readme
    assert "low-level remote launchers also auto-tune from that effective RAM" in handoff


def test_remote_launchers_validate_tunable_overrides_before_dispatch():
    shared = (ROOT / "scripts" / "_remote_lib.sh").read_text(encoding="utf-8")
    remote = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")
    sota = (ROOT / "scripts" / "launch_sota_robustness.sh").read_text(encoding="utf-8")
    mfbo = (ROOT / "scripts" / "launch_mfbo.sh").read_text(encoding="utf-8")
    d1 = (ROOT / "scripts" / "launch_d1_pairings.sh").read_text(encoding="utf-8")

    for text in (shared, remote):
        assert "require_positive_int_env" in text
        assert "require_positive_number_env" in text
        assert "must be a positive integer" in text
        assert "must be a positive number" in text
        for name in [
            "N_JOBS",
            "BO_N_JOBS",
            "MEM_CAP_GB",
            "WALL_H",
            "TIMEOUT_S",
            "REF_TIMEOUT_S",
            "HEARTBEAT_S",
            "N_REPEATS",
        ]:
            assert name in text

    for text in (remote, sota, mfbo, d1):
        assert "validate_remote_launch_tunables" in text
        assert text.index("validate_remote_launch_tunables") < text.index("PY_CMD=(")


def test_formal_pipeline_wrapper_accepts_validated_env_tunables():
    text = (ROOT / "scripts" / "run_pipeline.sh").read_text(encoding="utf-8")
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")

    assert "inject_env_int_arg N_JOBS --n-jobs" in text
    assert "inject_env_int_arg BO_N_JOBS --bo-n-jobs" in text
    assert "inject_env_number_arg MEM_CAP_GB --mem-cap-gb" in text
    assert "inject_env_number_arg TIMEOUT_S --timeout-s" in text
    assert "inject_env_number_arg BO_TIMEOUT_S --bo-timeout-s" in text
    assert "inject_env_number_arg REF_TIMEOUT_S --ref-timeout-s" in text
    assert "inject_env_number_arg HEARTBEAT_S --heartbeat-s" in text
    assert "inject_env_number_arg MAX_WALL_CLOCK_H --max-wall-clock-h" in text
    assert "has_cli_flag" in text
    assert "require_positive_int_value" in text
    assert "require_positive_number_value" in text
    assert text.index("inject_env_int_arg N_JOBS --n-jobs") < text.index(
        'exec "${PYTHON}" "${SCRIPT_DIR}/run_pipeline.py" "${args[@]}"'
    )
    assert "CLI flags still win" in readme
    assert "N_JOBS=24 BO_N_JOBS=4 MEM_CAP_GB=96 bash scripts/run_pipeline.sh" in readme
    handoff_flat = " ".join(handoff.split())
    assert "lets explicit CLI flags win" in handoff_flat
    assert "run_config.json" in handoff


def test_formal_pipeline_wrapper_runs_launch_preflight_before_full_dispatch():
    text = (ROOT / "scripts" / "run_pipeline.sh").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")

    assert "should_run_launch_preflight" in text
    assert "check_server_launch_ready.py" in text
    assert "DOE_BENCHMARK_REQUIRE_CLEAN_SWEEP_DIR" in text
    assert "DOE_BENCHMARK_SKIP_LAUNCH_PREFLIGHT" in text
    assert "has_cli_flag --dry-run && return 1" in text
    assert "has_cli_flag --list && return 1" in text
    assert "has_cli_flag --help && return 1" in text
    assert "has_cli_flag -h && return 1" in text
    assert text.index("check_server_launch_ready.py") < text.index(
        'exec "${PYTHON}" "${SCRIPT_DIR}/run_pipeline.py" "${args[@]}"'
    )
    assert "DOE_BENCHMARK_REQUIRE_CLEAN_SWEEP_DIR=1 bash scripts/run_pipeline.sh" in handoff
    assert "`--dry-run`, `--list`, and help invocations skip" in handoff


def test_formal_pipeline_wrapper_overrides_inherited_blas_thread_caps():
    text = (ROOT / "scripts" / "run_pipeline.sh").read_text(encoding="utf-8")
    orchestrator = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert f"{name}=1" in text
        assert f': "${{{name}:=1}}"' not in text
        assert f'os.environ[var] = "1"' in orchestrator


def test_remote_launchers_reject_unknown_persistence_mode_before_dispatch():
    shared = (ROOT / "scripts" / "_remote_lib.sh").read_text(encoding="utf-8")
    remote = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")
    sota = (ROOT / "scripts" / "launch_sota_robustness.sh").read_text(encoding="utf-8")
    mfbo = (ROOT / "scripts" / "launch_mfbo.sh").read_text(encoding="utf-8")
    d1 = (ROOT / "scripts" / "launch_d1_pairings.sh").read_text(encoding="utf-8")

    for text in (shared, remote):
        assert "validate_persistence_mode" in text
        assert "persistence mode must be one of: nohup, tmux, fg" in text
        assert "nohup|*)" not in text

    for text in (remote, sota, mfbo, d1):
        assert 'validate_persistence_mode "${PERSISTENCE_MODE}"' in text
        assert text.index("validate_persistence_mode") < text.index("PY_CMD=(")


def test_remote_launchers_refuse_existing_live_pid_before_any_dispatch_mode():
    shared = (ROOT / "scripts" / "_remote_lib.sh").read_text(encoding="utf-8")
    remote = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    for text in (shared, remote):
        assert "refuse_live_pid_file" in text
        assert "kill -0" in text
        assert "refusing to start a second writer" in text
        assert "stale or malformed pid file will be replaced" in text
        assert "tr -d '[:space:]'" in text

    assert shared.index('refuse_live_pid_file "${pid_file}"') < shared.rindex('case "${mode}"')
    assert remote.index('refuse_live_pid_file "${OUT_DIR}/sweep.pid"') < remote.index(
        'case "${PERSISTENCE_MODE}"'
    )
    assert remote.index('refuse_live_pid_file "${OUT_DIR}/sweep.pid"') < remote.index(
        'tmux new-session'
    )
    assert "refuses launch when the PID file still points to a live process" in readme


def test_remote_tmux_launchers_report_python_exit_code_not_tee_status():
    remote = (ROOT / "scripts" / "launch_sweep_remote.sh").read_text(encoding="utf-8")
    shared = (ROOT / "scripts" / "_remote_lib.sh").read_text(encoding="utf-8")
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    for text in (remote, shared):
        assert "set -o pipefail" in text
        assert "PIPESTATUS[0]" in text
        assert "tee -a" in text
        assert "exit=\\$?" not in text

    assert 'tmux new-session -d -s "${session}" -c "${PWD}"' in shared
    assert 'tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}"' in remote
    assert "starts tmux sessions from the current repo directory" in readme


def test_run_full_benchmark_rejects_invalid_numeric_args_before_output(tmp_path):
    mod = _load_module(
        "benchmark/scripts/run_full_benchmark.py",
        "run_full_benchmark_invalid_args",
    )
    out = tmp_path / "bad_run"

    rc = mod.main([
        "--cases", "cantilever_beam_5d",
        "--pool", "default",
        "--budgets", "0",
        "--n-repeats", "0",
        "--n-test", "0",
        "--n-jobs", "0",
        "--output-dir", str(out),
    ])

    assert rc == 2
    assert not out.exists()


def test_run_full_benchmark_rejects_duplicate_budgets_before_output(tmp_path):
    mod = _load_module(
        "benchmark/scripts/run_full_benchmark.py",
        "run_full_benchmark_duplicate_budgets",
    )
    out = tmp_path / "bad_run"

    rc = mod.main([
        "--cases", "cantilever_beam_5d",
        "--pool", "default",
        "--budgets", "5", "5",
        "--n-repeats", "1",
        "--n-test", "20",
        "--output-dir", str(out),
    ])

    assert rc == 2
    assert not out.exists()


def test_progress_helpers_derive_target_grid_from_current_registries():
    mod = _load_module("scripts/_check_sweep_progress.py", "check_sweep_progress_contract")
    from benchmark.methods import nature_pool

    expected_cases = {case.case_id for case in db.list_cases()}
    expected_methods = {method.name for method in nature_pool()}

    assert mod.PREREG_CASES == expected_cases
    assert mod.PREREG_METHODS == expected_methods
    assert mod.PREREG_TARGET == (
        len(expected_cases) * len(expected_methods) * len(mod.PREREG_BUDGETS) * mod.PREREG_REPS
    )


def test_progress_summary_counts_unique_protocol_cells_not_raw_rows():
    mod = _load_module("scripts/_check_sweep_progress.py", "check_sweep_progress_summary")
    import pandas as pd

    df = pd.DataFrame([
        {
            "case_id": "case_a",
            "method": "method_a",
            "budget_mult": 5,
            "repeat": 0,
            "status": "ok",
        },
        {
            "case_id": "case_a",
            "method": "method_a",
            "budget_mult": 5,
            "repeat": 0,
            "status": "failed",
        },
        {
            "case_id": "case_a",
            "method": "method_a",
            "budget_mult": 10,
            "repeat": 0,
            "status": "timeout",
        },
        {
            "case_id": "case_a",
            "method": "method_a",
            "budget_mult": 50,
            "repeat": 99,
            "status": "ok",
        },
        {
            "case_id": "case_a",
            "method": "extra_method",
            "budget_mult": 5,
            "repeat": 0,
            "status": "ok",
        },
    ])

    summary = mod.summarize_progress(
        df,
        cases={"case_a"},
        methods={"method_a"},
        budgets={5, 10},
        n_repeats=2,
    )

    assert summary["target_cells"] == 4
    assert summary["protocol_rows"] == 3
    assert summary["protocol_unique_cells"] == 2
    assert summary["duplicate_protocol_rows"] == 1
    assert summary["out_of_protocol_rows"] == 2
    assert summary["status_counts_unique"] == {"failed": 1, "ok": 0, "timeout": 1}


def test_progress_main_handles_malformed_results_csv_without_crashing(
    tmp_path,
    capsys,
):
    mod = _load_module(
        "scripts/_check_sweep_progress.py",
        "check_sweep_progress_malformed_csv",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text('case_id,method\n"unterminated,m1\n', encoding="utf-8")

    rc = mod.main([str(csv_path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "could not parse results.csv" in out
    assert "formal read-only health gate" in out


def test_progress_main_accepts_explicit_results_csv_path(tmp_path, capsys):
    mod = _load_module(
        "scripts/_check_sweep_progress.py",
        "check_sweep_progress_explicit_path",
    )
    csv_path = tmp_path / "explicit_results.csv"
    csv_path.write_text("case_id,method,budget_mult,repeat\n", encoding="utf-8")

    rc = mod.main([str(csv_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert f"CSV path        : {csv_path}" in out
    assert "No data rows yet" in out


def test_progress_main_handles_missing_progress_columns_without_crashing(
    tmp_path,
    capsys,
):
    mod = _load_module(
        "scripts/_check_sweep_progress.py",
        "check_sweep_progress_missing_columns",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("case_id,method\ncase_a,m1\n", encoding="utf-8")

    rc = mod.main([str(csv_path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "Malformed results.csv" in out
    assert "missing required progress columns" in out
    assert "formal read-only health gate" in out


def test_progress_breakdown_no_longer_hardcodes_20_method_target():
    text = (ROOT / "scripts" / "sweep_progress_breakdown.py").read_text(encoding="utf-8")

    assert "N_METHODS = 20" not in text
    assert "METHODS = [m.name for m in nature_pool()]" in text
    assert "Unexpected methods outside nature_pool()" in text
    assert "drop_duplicates(GRID_COLUMNS, keep=\"last\")" in text


def test_progress_breakdown_handles_malformed_results_csv_without_crashing(tmp_path, capsys):
    mod = _load_module(
        "scripts/sweep_progress_breakdown.py",
        "sweep_progress_breakdown_malformed_csv",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text('case_id,method\n"unterminated,m1\n', encoding="utf-8")

    rc = mod.main([str(csv_path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "could not parse results.csv" in out
    assert "formal read-only health gate" in out


def test_progress_breakdown_handles_header_only_startup_without_crashing(tmp_path, capsys):
    mod = _load_module(
        "scripts/sweep_progress_breakdown.py",
        "sweep_progress_breakdown_header_only",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("case_id,method,budget_mult,repeat,t_total_s\n", encoding="utf-8")

    rc = mod.main([str(csv_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "No data rows yet" in out
    assert "--allow-missing" in out


def test_progress_breakdown_handles_missing_columns_without_crashing(tmp_path, capsys):
    mod = _load_module(
        "scripts/sweep_progress_breakdown.py",
        "sweep_progress_breakdown_missing_columns",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("case_id,method\ncase_a,m1\n", encoding="utf-8")

    rc = mod.main([str(csv_path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "Malformed results.csv" in out
    assert "missing required progress columns" in out
    assert "formal read-only health gate" in out


def test_progress_helper_suggestions_preserve_formal_server_pipeline():
    text = (ROOT / "scripts" / "_check_sweep_progress.py").read_text(encoding="utf-8")

    assert "bash scripts/run_pipeline.sh" in text
    assert "--from-stage 3 --confirm-full-run" in text
    assert "local review workstation" in text
    assert "launch_sweep_local_16gb.ps1" not in text
    assert "snakemake --cores" not in text


def test_sweep_runtime_health_accepts_clean_partial_csv(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_clean",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    case_fingerprint = case_source_fingerprint(case)
    method_fingerprint = mod.PREREG_METHOD_FINGERPRINTS[method]
    timeout_trace = _runtime_trace_fields()
    timeout_trace["cell_finished_utc"] = "2026-06-01T00:10:00+00:00"
    rows = [
        {
            "case_id": case_id,
            **case_fingerprint,
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **method_fingerprint,
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
        {
            "case_id": case_id,
            **case_fingerprint,
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **method_fingerprint,
            "budget_mult": 10,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 10 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "timeout",
            "failure_category": "timeout",
            "nrmse": "",
            "t_total_s": 600.0,
            "error": "timeout after 600s",
            **timeout_trace,
        },
    ]
    _results_frame(rows).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        max_timeout_fraction=0.60,
    )

    assert rc == 0
    assert errors == []
    assert any("fraction gate deferred" in warn for warn in warnings)


def test_sweep_runtime_health_warns_when_rss_nears_memory_cap(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_mem_cap_warning",
    )
    from benchmark.core import derive_seed

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    trace = _runtime_trace_fields()
    trace["rss_gb_peak"] = 9.25
    _results_frame([
        {
            "case_id": case_id,
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **mod.PREREG_METHOD_FINGERPRINTS[method],
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **trace,
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        mem_cap_gb=10.0,
        mem_warning_fraction=0.90,
    )

    assert rc == 0
    assert errors == []
    assert any("observed rss_gb_peak is close" in warn for warn in warnings)
    assert any("reducing n_jobs/bo_n_jobs" in warn for warn in warnings)


def test_sweep_runtime_health_warns_when_results_filesystem_is_low_on_space(
    tmp_path,
    monkeypatch,
):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_disk_warning",
    )
    from benchmark.core import derive_seed

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    _results_frame([
        {
            "case_id": case_id,
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **mod.PREREG_METHOD_FINGERPRINTS[method],
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
    ]).to_csv(csv_path, index=False)

    class Usage:
        total = 100 * 1024**3
        used = 99 * 1024**3
        free = 1 * 1024**3

    monkeypatch.setattr(mod.shutil, "disk_usage", lambda _path: Usage())

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        disk_warning_gb=5.0,
    )

    assert rc == 0
    assert errors == []
    assert any("free disk space near results.csv is below" in warn for warn in warnings)
    assert any("storage exhaustion corrupts the next streamed row" in warn for warn in warnings)


def test_sweep_runtime_health_warns_when_temp_storage_is_low_on_space(
    tmp_path,
    monkeypatch,
):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_temp_storage_disk_warning",
    )
    from benchmark.core import derive_seed

    csv_path = tmp_path / "results.csv"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    _results_frame([
        {
            "case_id": case_id,
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **mod.PREREG_METHOD_FINGERPRINTS[method],
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
    ]).to_csv(csv_path, index=False)

    class Usage:
        def __init__(self, free_gb: float):
            self.total = 100 * 1024**3
            self.used = int((100.0 - free_gb) * 1024**3)
            self.free = int(free_gb * 1024**3)

    original_disk_usage = mod.shutil.disk_usage

    def fake_disk_usage(path):
        probe = Path(path)
        if probe == scratch:
            return Usage(1.0)
        return original_disk_usage(path)

    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(scratch))
    monkeypatch.setattr(mod.shutil, "disk_usage", fake_disk_usage)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        disk_warning_gb=5.0,
    )

    assert rc == 0
    assert errors == []
    assert any("free disk space near temp/scratch" in warn for warn in warnings)
    assert any(str(scratch) in warn for warn in warnings)


def _write_runtime_health_csv_and_run_config(tmp_path: Path, module_name: str):
    mod = _load_module("scripts/check_sweep_runtime_health.py", module_name)
    from benchmark.core import derive_seed

    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    case_fp = case_source_fingerprint(case)
    method_fp = mod.PREREG_METHOD_FINGERPRINTS[method]
    row = {
        "case_id": case_id,
        **case_fp,
        "domain": case.meta.domain,
        "method": method,
        "method_family": mod.PREREG_METHOD_FAMILIES[method],
        **method_fp,
        "budget_mult": 5,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "n_train": 5 * case.meta.dim_input,
        "repeat": 0,
        "n_test": 1000,
        "test_seed": 0,
        "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
        "status": "ok",
        "failure_category": "ok",
        "nrmse": 0.1,
        "t_total_s": 1.0,
        "error": "",
        **_runtime_trace_fields(),
    }
    csv_path = tmp_path / "results.csv"
    _results_frame([row]).to_csv(csv_path, index=False)
    run_config = {
        "artifact": "doe-benchmark-stage2-run-config",
        "created_utc": "2026-06-01T00:00:00+00:00",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "total_cells": 1,
        "cases": [{
            "case_id": case_id,
            "domain": case.meta.domain,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            **case_fp,
        }],
        "methods": [{
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            "parallel_backend": "loky",
            "memory_class": "light",
            **method_fp,
        }],
        "budgets": [5],
        "n_repeats": 1,
        "n_test": 1000,
        "test_seed": 0,
        "seed_start": 1000,
        "seed_strategy": "case_repeat",
        "n_jobs": 1,
        "bo_n_jobs": 1,
        "timeout_s": 600.0,
        "bo_timeout_s": 1800.0,
        "ref_timeout_s": 600.0,
        "watchdog_s": None,
        "heartbeat_s": 60.0,
        "max_wall_clock_s": None,
        "mem_cap_gb": 56.0,
        "skip_existing": True,
        "output_dir": str(tmp_path),
        "environment": {
            "schema_version": 1,
            "python_version": "3.11.9",
            "python_executable": "/usr/bin/python",
            "platform": "Linux-6.8-x86_64",
            "machine": "x86_64",
            "node": "server-a",
            "cwd": str(tmp_path),
            "packages": {
                package: {"status": "installed", "version": "1.0.0"}
                for package in (
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
            },
        },
        "temp_storage": {
            "schema_version": 1,
            "resolved_path": str(tmp_path),
            "exists": True,
            "writable": True,
            "free_gb": 100.0,
            "env": {
                "TMPDIR": None,
                "TMP": None,
                "TEMP": None,
                "TEMPDIR": None,
                "SLURM_TMPDIR": None,
                "SCRATCH": None,
                "LOCAL_SCRATCH": None,
                "JOBSCRATCH": None,
            },
        },
        "torch_reproducibility": _torch_reproducibility_snapshot(),
        "thread_env": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }
    run_config_path = tmp_path / "run_config.json"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    return mod, csv_path, run_config_path, run_config


def test_sweep_runtime_health_validates_matching_run_config_snapshot(tmp_path):
    mod, csv_path, run_config_path, _run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_ok",
    )

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 0
    assert errors == []
    assert any("run_config.json contract matched partial results" in warn for warn in warnings)


def test_sweep_runtime_health_can_validate_current_environment_snapshot(
    tmp_path, monkeypatch
):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_validate_current_environment",
    )
    monkeypatch.setattr(
        mod,
        "_runtime_environment_snapshot",
        lambda: run_config["environment"],
    )
    for name, value in run_config["thread_env"].items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
        validate_current_environment=True,
    )

    assert rc == 0
    assert errors == []
    assert any(
        "current runtime environment matches run_config.json snapshot"
        in warn
        for warn in warnings
    )


def test_sweep_runtime_health_rejects_current_environment_drift(
    tmp_path, monkeypatch
):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_environment_drift",
    )
    drifted_environment = json.loads(json.dumps(run_config["environment"]))
    drifted_environment["python_version"] = "3.12.0"
    drifted_environment["packages"]["botorch"]["version"] = "9.9.9"
    monkeypatch.setattr(
        mod,
        "_runtime_environment_snapshot",
        lambda: drifted_environment,
    )
    for name, value in run_config["thread_env"].items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
        validate_current_environment=True,
    )

    assert rc == 1
    assert any("current environment python_version drifted" in err for err in errors)
    assert any("package version drifted for botorch" in err for err in errors)
    assert any(
        "current thread env drifted for OMP_NUM_THREADS" in err for err in errors
    )
    assert warnings == []


def test_sweep_runtime_health_rejects_run_config_protocol_drift(tmp_path):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_bad",
    )
    run_config["budgets"] = [10]
    run_config["cases"][0]["case_source_sha256"] = "0" * 64
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any("budget_mult values outside run_config.json" in err for err in errors)
    assert any("case_source_sha256" in err and "run_config.json" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_malformed_run_config_scalar_protocol_fields(tmp_path):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_bad_scalar_fields",
    )
    run_config["n_test"] = "1000"
    run_config["test_seed"] = -1
    run_config["seed_start"] = 1000.0
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any(
        "run_config.json n_test must be a positive integer" in err
        for err in errors
    )
    assert any(
        "run_config.json test_seed must be a non-negative integer" in err
        for err in errors
    )
    assert any(
        "run_config.json seed_start must be a non-negative integer" in err
        for err in errors
    )
    assert warnings == []


def test_sweep_runtime_health_rejects_duplicate_run_config_case_or_method_entries(tmp_path):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_duplicate_grid_entries",
    )
    run_config["cases"].append(dict(run_config["cases"][0]))
    run_config["methods"].append(dict(run_config["methods"][0]))
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any("run_config.json has malformed cases entries" in err for err in errors)
    assert any("run_config.json has malformed methods entries" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_run_config_missing_runtime_controls(tmp_path):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_runtime_controls",
    )
    for key in (
        "timeout_s",
        "bo_timeout_s",
        "ref_timeout_s",
        "watchdog_s",
        "heartbeat_s",
        "max_wall_clock_s",
        "mem_cap_gb",
        "skip_existing",
        "thread_env",
    ):
        run_config.pop(key, None)
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any("run_config.json missing timeout_s" in err for err in errors)
    assert any("run_config.json skip_existing must be recorded as a boolean" in err for err in errors)
    assert any("run_config.json thread_env must be an object" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_run_config_string_or_boolean_runtime_controls(tmp_path):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_runtime_control_types",
    )
    run_config.update({
        "n_jobs": "8",
        "bo_n_jobs": True,
        "timeout_s": "600",
        "bo_timeout_s": "1800",
        "ref_timeout_s": False,
        "watchdog_s": "120",
        "heartbeat_s": "60",
        "max_wall_clock_s": "86400",
        "mem_cap_gb": "56",
    })
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any("run_config.json n_jobs must be a JSON positive integer" in err for err in errors)
    assert any("run_config.json bo_n_jobs must be a JSON positive integer" in err for err in errors)
    assert any("run_config.json timeout_s must be null or a finite positive JSON number" in err for err in errors)
    assert any("run_config.json bo_timeout_s must be null or a finite positive JSON number" in err for err in errors)
    assert any("run_config.json ref_timeout_s must be null or a finite positive JSON number" in err for err in errors)
    assert any("run_config.json watchdog_s must be null or a finite positive JSON number" in err for err in errors)
    assert any("run_config.json heartbeat_s must be null or a finite non-negative JSON number" in err for err in errors)
    assert any("run_config.json max_wall_clock_s must be null or a finite non-negative JSON number" in err for err in errors)
    assert any("run_config.json mem_cap_gb must be null or a finite positive JSON number" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_run_config_missing_required_environment_package(
    tmp_path,
):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_missing_package",
    )
    run_config["environment"]["packages"].pop("psutil", None)
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any(
        "run_config.json environment.packages missing object for psutil" in err
        for err in errors
    )
    assert warnings == []


def test_sweep_runtime_health_rejects_run_config_non_installed_required_package(
    tmp_path,
):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_non_installed_package",
    )
    run_config["environment"]["packages"]["neuralfoil"] = {
        "status": "missing",
        "version": "",
    }
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any(
        "run_config.json environment.packages neuralfoil.status must be 'installed'"
        in err
        for err in errors
    )
    assert warnings == []


def test_sweep_runtime_health_rejects_run_config_missing_temp_storage_snapshot(
    tmp_path,
):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_missing_temp_storage",
    )
    run_config.pop("temp_storage", None)
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any("run_config.json temp_storage must be an object" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_missing_torch_reproducibility_snapshot(
    tmp_path,
):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_missing_torch_repro",
    )
    run_config.pop("torch_reproducibility", None)
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any(
        "run_config.json torch_reproducibility must be an object" in err
        for err in errors
    )
    assert warnings == []


def test_sweep_runtime_health_rejects_malformed_torch_reproducibility_snapshot(
    tmp_path,
):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_bad_torch_repro",
    )
    run_config["torch_reproducibility"] = {
        "schema_version": 0,
        "status": "ok",
        "policy": {
            "deterministic_algorithms": False,
            "deterministic_warn_only": True,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "allow_tf32": True,
            "float32_matmul_precision": "medium",
            "cublas_workspace_config": "",
        },
        "applied": True,
        "errors": [],
        "state": {
            "deterministic_algorithms": False,
            "cudnn_benchmark": True,
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
            "float32_matmul_precision": "medium",
        },
    }
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any("torch_reproducibility.schema_version must be 1" in err for err in errors)
    assert any(
        "torch_reproducibility.policy.deterministic_algorithms must be True" in err
        for err in errors
    )
    assert any("torch cudnn benchmark is enabled" in err for err in errors)
    assert any("torch CUDA matmul TF32 is enabled" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_malformed_run_config_temp_storage_snapshot(
    tmp_path,
):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_run_config_bad_temp_storage",
    )
    run_config["temp_storage"] = {
        "schema_version": 0,
        "resolved_path": "",
        "exists": "yes",
        "writable": "yes",
        "free_gb": -1,
        "env": {},
    }
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        require_run_config=True,
    )

    assert rc == 1
    assert any("run_config.json temp_storage.schema_version must be 1" in err for err in errors)
    assert any("run_config.json temp_storage.resolved_path must be non-empty" in err for err in errors)
    assert any("run_config.json temp_storage.exists must be boolean" in err for err in errors)
    assert any("run_config.json temp_storage.free_gb must be null or a finite non-negative number" in err for err in errors)
    assert any("run_config.json temp_storage.env missing TMPDIR" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_duplicate_header_columns(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_duplicate_header",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,status\n"
        "case_a,method_a,5,0,ok,failed\n",
        encoding="utf-8",
    )

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("duplicate header column name" in error for error in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_blank_header_columns(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_blank_header",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,,repeat,status\n"
        "case_a,method_a,5,0,ok\n",
        encoding="utf-8",
    )

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("blank header column name" in error for error in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_ragged_rows_before_pandas_parsing(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_ragged_row",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status\n"
        "case_a,method_a,5,0,ok,extra_field\n",
        encoding="utf-8",
    )

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("different number of fields than the header" in error for error in errors)
    assert warnings == []


def test_sweep_runtime_health_can_fail_on_stale_nonempty_csv(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_stale_nonempty",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    _results_frame([
        {
            "case_id": case_id,
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **mod.PREREG_METHOD_FINGERPRINTS[method],
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        }
    ]).to_csv(csv_path, index=False)
    stale_time = csv_path.stat().st_mtime - 120.0
    os.utime(csv_path, (stale_time, stale_time))

    rc_warn, errors_warn, warnings_warn = mod.check_runtime_health(
        csv_path,
        stale_after_min=0.5,
    )
    rc_fail, errors_fail, warnings_fail = mod.check_runtime_health(
        csv_path,
        stale_after_min=0.5,
        fail_on_stale=True,
    )

    assert rc_warn == 0
    assert errors_warn == []
    assert any("results.csv has not been updated" in warn for warn in warnings_warn)
    assert rc_fail == 1
    assert any("results.csv has not been updated" in err for err in errors_fail)
    assert warnings_fail == []


def test_sweep_runtime_health_rejects_finished_before_started(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_bad_time_order",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    trace = _runtime_trace_fields()
    trace["cell_started_utc"] = "2026-06-01T00:00:02+00:00"
    trace["cell_finished_utc"] = "2026-06-01T00:00:01+00:00"
    _results_frame([
        {
            "case_id": case_id,
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **mod.PREREG_METHOD_FINGERPRINTS[method],
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **trace,
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("cell_finished_utc earlier than cell_started_utc" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_runtime_duration_drift(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_bad_duration",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    trace = _runtime_trace_fields()
    trace["cell_started_utc"] = "2026-06-01T00:00:00+00:00"
    trace["cell_finished_utc"] = "2026-06-01T00:00:01+00:00"
    _results_frame([
        {
            "case_id": case_id,
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **mod.PREREG_METHOD_FINGERPRINTS[method],
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 100.0,
            "error": "",
            **trace,
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("t_total_s inconsistent" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_flags_protocol_metadata_drift(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_protocol_metadata",
    )
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    _results_frame([
        {
            "case_id": case_id,
            "method": method,
            "budget_mult": 5,
            "repeat": 0,
            "n_test": 999,
            "test_seed": 123,
            "seed": 9999,
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("n_test != 1000" in err for err in errors)
    assert any("test_seed != 0" in err for err in errors)
    assert any("seed not matching case_repeat" in err for err in errors)


def test_sweep_runtime_health_default_protocol_rejects_smoke_n_test(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_default_rejects_smoke_protocol",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    _results_frame([
        {
            "case_id": case_id,
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **mod.PREREG_METHOD_FINGERPRINTS[method],
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 100,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        min_cells_for_fraction_gate=1,
        max_timeout_fraction=1.0,
        max_failed_fraction=1.0,
    )

    assert rc == 1
    assert any("n_test != 1000" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_accepts_explicit_smoke_protocol(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_explicit_smoke_protocol",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    _results_frame([
        {
            "case_id": case_id,
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **mod.PREREG_METHOD_FINGERPRINTS[method],
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 100,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        expected_n_test=100,
        expected_test_seed=0,
        seed_strategy="case_repeat",
        seed_start=1000,
    )

    assert rc == 0
    assert errors == []
    assert warnings == []


def test_sweep_runtime_health_flags_n_train_budget_drift(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_n_train",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    method_family = mod.PREREG_METHOD_FAMILIES[method]
    case_fingerprint = case_source_fingerprint(case)
    method_fingerprint = mod.PREREG_METHOD_FINGERPRINTS[method]
    _results_frame([
        {
            "case_id": case_id,
            **case_fingerprint,
            "domain": case.meta.domain,
            "method": method,
            "method_family": method_family,
            **method_fingerprint,
            "budget_mult": 5,
            "dim": 4,
            "n_output": case.meta.dim_output,
            "n_train": 21,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("n_train != budget_mult * dim" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_flags_case_metadata_drift(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_case_metadata",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    case = db.get_case(case_id)
    method = sorted(mod.PREREG_METHODS)[0]
    method_family = mod.PREREG_METHOD_FAMILIES[method]
    case_fingerprint = case_source_fingerprint(case)
    method_fingerprint = mod.PREREG_METHOD_FINGERPRINTS[method]
    _results_frame([
        {
            "case_id": case_id,
            **case_fingerprint,
            "domain": "wrong_domain",
            "method": method,
            "method_family": method_family,
            **method_fingerprint,
            "budget_mult": 5,
            "dim": case.meta.dim_input + 1,
            "n_output": case.meta.dim_output + 1,
            "n_train": 5 * (case.meta.dim_input + 1),
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("case metadata columns mismatching the current registry" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_flags_method_family_drift(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_method_family",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    case = db.get_case(case_id)
    method = sorted(mod.PREREG_METHODS)[0]
    case_fingerprint = case_source_fingerprint(case)
    method_fingerprint = mod.PREREG_METHOD_FINGERPRINTS[method]
    _results_frame([
        {
            "case_id": case_id,
            **case_fingerprint,
            "domain": case.meta.domain,
            "method": method,
            **method_fingerprint,
            "method_family": "wrong_family",
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("method_family values mismatching the current nature_pool registry" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_flags_source_fingerprint_drift(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_source_fingerprint",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    case = db.get_case(case_id)
    method = sorted(mod.PREREG_METHODS)[0]
    case_fingerprint = case_source_fingerprint(case)
    method_fingerprint = dict(mod.PREREG_METHOD_FINGERPRINTS[method])
    case_fingerprint["case_source_sha256"] = "0" * 64
    method_fingerprint["method_source_sha256"] = "0" * 64
    _results_frame([
        {
            "case_id": case_id,
            **case_fingerprint,
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **method_fingerprint,
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("case source fingerprints mismatching the current registry" in err for err in errors)
    assert any("method source fingerprints mismatching the current nature_pool registry" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_accepts_absolute_source_paths_with_matching_hashes(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_absolute_source_paths",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    case = db.get_case(case_id)
    method = sorted(mod.PREREG_METHODS)[0]
    case_fingerprint = dict(case_source_fingerprint(case))
    method_fingerprint = dict(mod.PREREG_METHOD_FINGERPRINTS[method])
    case_fingerprint["case_source_path"] = (ROOT / case_fingerprint["case_source_path"]).as_posix()
    method_fingerprint["method_source_path"] = (ROOT / method_fingerprint["method_source_path"]).as_posix()
    _results_frame([
        {
            "case_id": case_id,
            **case_fingerprint,
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **method_fingerprint,
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 0
    assert errors == []
    assert warnings == []


def test_sweep_runtime_health_emits_failure_resource_and_wall_time_summaries():
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_runtime_summaries",
    )

    frame = _results_frame([
        {
            "case_id": "case_a",
            "method": "method_a",
            "budget_mult": 5,
            "repeat": 0,
            "status": "ok",
            "failure_category": "ok",
            "error": "",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            **_runtime_trace_fields(),
        },
        {
            "case_id": "case_a",
            "method": "method_b",
            "budget_mult": 5,
            "repeat": 0,
            "status": "failed",
            "failure_category": "oom",
            "error": "slurmstepd: error: Detected 1 oom-kill event",
            "nrmse": "",
            "t_total_s": 1.0,
            **{**_runtime_trace_fields(), "rss_gb_peak": 12.5},
        },
    ])

    lines = mod._runtime_summary_lines(frame)

    joined = "\n".join(lines)
    assert "runtime failure-category telemetry summarized" in joined
    assert '"ok": 1' in joined
    assert '"oom": 1' in joined
    assert "non_ok_failure_category_counts" in joined
    assert "runtime resource telemetry summarized" in joined
    assert '"max": 12.5' in joined
    assert '"method_b": 12.5' in joined
    assert "runtime wall-time telemetry summarized" in joined
    assert 't_total_s={"max": 1.0' in joined
    assert '"method_a": 1.0' in joined


def test_sweep_runtime_health_fails_when_runtime_contract_columns_missing(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_n_train_missing",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    _results_frame([
        {
            "case_id": case_id,
            "method": method,
            "budget_mult": 5,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("missing required runtime contract columns" in err for err in errors)
    assert any("case_source_path" in err for err in errors)
    assert any("method_source_sha256" in err for err in errors)
    assert any("n_train" in err for err in errors)


def test_sweep_runtime_health_rejects_tampered_protocol_digest(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_bad_protocol_digest",
    )
    from benchmark.core import derive_seed

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    case = db.get_case(case_id)
    method = sorted(mod.PREREG_METHODS)[0]
    df = _results_frame([
        {
            "case_id": case_id,
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **mod.PREREG_METHOD_FINGERPRINTS[method],
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
            **_runtime_trace_fields(),
        },
    ])
    df.loc[0, "protocol_digest"] = "0" * 64
    df.to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("protocol_digest inconsistent" in err for err in errors)
    assert warnings == []


def test_prepare_figdata_contract_rejects_tampered_protocol_digest():
    mod = _load_module("scripts/prepare_figdata.py", "prepare_figdata_bad_digest")
    case = db.get_case("cantilever_beam_5d")
    method = nature_pool()[0]
    df = _results_frame([
        {
            "case_id": "cantilever_beam_5d",
            **case_source_fingerprint(case),
            "domain": case.meta.domain,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "method": method.name,
            "method_family": method.family,
            **method_source_fingerprint(method),
            "budget_mult": 5,
            "n_train": 5 * case.meta.dim_input,
            "n_test": 1000,
            "repeat": 0,
            "seed": 1000,
            "test_seed": 0,
            "status": "ok",
            "failure_category": "ok",
            "error": "",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            **_runtime_trace_fields(),
        },
    ])
    df.loc[0, "protocol_digest"] = "0" * 64

    with pytest.raises(AssertionError, match="protocol_digest must match"):
        mod.validate_results_contract(df)


def test_sweep_runtime_health_flags_invalid_runtime_trace_values(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_bad_runtime_trace",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    case = db.get_case(case_id)
    method = sorted(mod.PREREG_METHODS)[0]
    row = {
        "case_id": case_id,
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "method": method,
        "method_family": mod.PREREG_METHOD_FAMILIES[method],
        **mod.PREREG_METHOD_FINGERPRINTS[method],
        "budget_mult": 5,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "n_train": 5 * case.meta.dim_input,
        "repeat": 0,
        "n_test": 1000,
        "test_seed": 0,
        "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
        "status": "ok",
        "failure_category": "ok",
        "nrmse": 0.1,
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        "worker_host": "",
        "worker_pid": -1,
        "cell_started_utc": "not-a-time",
        "cell_finished_utc": "",
        "rss_gb_start": -0.1,
        "rss_gb_end": "nan",
        "rss_gb_peak": "bad",
    }
    _results_frame([row]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("empty worker_host" in err for err in errors)
    assert any("worker_pid" in err for err in errors)
    assert any("cell_started_utc" in err for err in errors)
    assert any("cell_finished_utc" in err for err in errors)
    assert any("rss_gb_start" in err for err in errors)
    assert any("rss_gb_end" in err for err in errors)
    assert any("rss_gb_peak" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_boolean_integer_fields(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_bool_integer_fields",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    case = db.get_case(case_id)
    method = sorted(mod.PREREG_METHODS)[0]
    row = {
        "case_id": case_id,
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "method": method,
        "method_family": mod.PREREG_METHOD_FAMILIES[method],
        **mod.PREREG_METHOD_FINGERPRINTS[method],
        "budget_mult": True,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "n_train": case.meta.dim_input,
        "repeat": True,
        "n_test": 1000,
        "test_seed": 0,
        "seed": derive_seed(case_id, 1, strategy="case_repeat", seed_start=1000),
        "status": "ok",
        "failure_category": "ok",
        "nrmse": 0.1,
        "t_total_s": 1.0,
        "error": "",
        **_runtime_trace_fields(),
    }
    row["worker_pid"] = True
    _results_frame([row]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("n_train != budget_mult * dim" in err for err in errors)
    assert any("seed not matching" in err for err in errors)
    assert any("worker_pid" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_flags_protocol_and_status_problems(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_bad",
    )
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    _results_frame([
        {
            "case_id": case_id,
            "method": method,
            "budget_mult": 5,
            "repeat": 0,
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
        },
        {
            "case_id": case_id,
            "method": method,
            "budget_mult": 5,
            "repeat": 0,
            "status": "failed",
            "failure_category": "exception",
            "nrmse": "",
            "t_total_s": -1.0,
            "error": "",
        },
        {
            "case_id": case_id,
            "method": "not_in_pool",
            "budget_mult": 5,
            "repeat": 0,
            "status": "weird",
            "failure_category": "exception",
            "nrmse": 0.2,
            "t_total_s": 1.0,
            "error": "",
        },
        {
            "case_id": case_id,
            "method": method,
            "budget_mult": 7,
            "repeat": 30,
            "status": "ok",
            "failure_category": "ok",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            "error": "",
        },
        {
            "case_id": case_id,
            "method": method,
            "budget_mult": 10,
            "repeat": 0,
            "status": "ok",
            "failure_category": "ok",
            "nrmse": -0.1,
            "t_total_s": 1.0,
            "error": "",
        },
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("outside the preregistered" in err for err in errors)
    assert any("method values outside the frozen nature_pool()" in err for err in errors)
    assert any("budget_mult values outside the preregistered set" in err for err in errors)
    assert any("repeat values outside the preregistered range" in err for err in errors)
    assert any("invalid status values" in err for err in errors)
    assert any("invalid t_total_s" in err for err in errors)
    assert any("invalid nrmse" in err for err in errors)
    assert any("empty error" in err for err in errors)
    assert any("duplicate protocol" in err for err in errors)


def test_sweep_runtime_health_rejects_invalid_optional_metric_values(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_bad_optional_metrics",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    base = {
        "case_id": case_id,
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "method": method,
        "method_family": mod.PREREG_METHOD_FAMILIES[method],
        **mod.PREREG_METHOD_FINGERPRINTS[method],
        "budget_mult": 5,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "status": "ok",
        "failure_category": "ok",
        "nrmse": 0.1,
        "t_total_s": 1.0,
        "error": "",
        **_runtime_trace_fields(),
    }
    rows = [
        {
            **base,
            "repeat": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "rmse": -0.2,
            "mae": 0.1,
            "maxae": 0.5,
            "q95ae": 0.4,
            "r2": 0.9,
        },
        {
            **base,
            "repeat": 1,
            "seed": derive_seed(case_id, 1, strategy="case_repeat", seed_start=1000),
            "rmse": 0.2,
            "mae": "not-a-number",
            "maxae": 0.5,
            "q95ae": 0.4,
            "r2": 0.8,
        },
        {
            **base,
            "repeat": 2,
            "seed": derive_seed(case_id, 2, strategy="case_repeat", seed_start=1000),
            "rmse": 0.2,
            "mae": 0.1,
            "maxae": float("inf"),
            "q95ae": "",
            "r2": "not-finite",
        },
    ]
    _results_frame(rows, seed_strategy="legacy").to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        min_cells_for_fraction_gate=1,
        max_timeout_fraction=1.0,
        max_failed_fraction=1.0,
    )

    assert rc == 1
    assert any("invalid rmse" in err for err in errors)
    assert any("invalid mae" in err for err in errors)
    assert any("invalid maxae" in err for err in errors)
    assert any("invalid q95ae" in err for err in errors)
    assert any("invalid r2" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_metrics_on_failed_or_timeout_rows(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_non_ok_metrics",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    base = {
        "case_id": case_id,
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "method": method,
        "method_family": mod.PREREG_METHOD_FAMILIES[method],
        **mod.PREREG_METHOD_FINGERPRINTS[method],
        "budget_mult": 5,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "t_total_s": 1.0,
        **_runtime_trace_fields(),
    }
    rows = [
        {
            **base,
            "repeat": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "failed",
            "failure_category": "exception",
            "error": "RuntimeError: synthetic failure",
            "nrmse": 0.2,
            "rmse": "",
        },
        {
            **base,
            "repeat": 1,
            "seed": derive_seed(case_id, 1, strategy="case_repeat", seed_start=1000),
            "status": "timeout",
            "failure_category": "timeout",
            "error": "timeout after 600s",
            "nrmse": "",
            "rmse": 0.3,
        },
    ]
    _results_frame(rows, seed_strategy="legacy").to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        min_cells_for_fraction_gate=1,
        max_timeout_fraction=1.0,
        max_failed_fraction=1.0,
    )

    assert rc == 1
    assert any("non-empty nrmse" in err for err in errors)
    assert any("non-empty rmse" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_rejects_blank_or_missing_status_values(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_blank_status",
    )
    from benchmark.core import derive_seed
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    case_fingerprint = case_source_fingerprint(case)
    method_fingerprint = mod.PREREG_METHOD_FINGERPRINTS[method]
    base = {
        "case_id": case_id,
        **case_fingerprint,
        "domain": case.meta.domain,
        "method": method,
        "method_family": mod.PREREG_METHOD_FAMILIES[method],
        **method_fingerprint,
        "budget_mult": 5,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
        "nrmse": 0.1,
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        **_runtime_trace_fields(),
    }
    _results_frame([
        {**base, "repeat": 0, "status": " "},
        {**base, "repeat": 1, "status": pd.NA},
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("missing/blank status values: 2" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_allow_missing_is_nonblocking(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_missing",
    )

    rc, errors, warnings = mod.check_runtime_health(
        tmp_path / "missing.csv",
        allow_missing=True,
    )

    assert rc == 0
    assert errors == []
    assert any("missing" in warn for warn in warnings)


def test_sweep_runtime_health_allow_missing_still_requires_run_config_when_requested(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_missing_requires_run_config",
    )

    rc, errors, warnings = mod.check_runtime_health(
        tmp_path / "results.csv",
        allow_missing=True,
        require_run_config=True,
    )

    assert rc == 1
    assert any("run_config.json is required" in err for err in errors)
    assert any("results.csv is missing" in warn for warn in warnings)


def test_sweep_runtime_health_allow_missing_validates_existing_run_config(tmp_path):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_missing_bad_run_config",
    )
    csv_path.unlink()
    run_config["temp_storage"].pop("resolved_path")
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        allow_missing=True,
        require_run_config=True,
    )

    assert rc == 1
    assert any("run_config.json temp_storage.resolved_path" in err for err in errors)
    assert any("results.csv is missing" in warn for warn in warnings)


def test_sweep_runtime_health_allow_missing_reports_malformed_run_config_grid(tmp_path):
    mod, csv_path, run_config_path, run_config = _write_runtime_health_csv_and_run_config(
        tmp_path,
        "check_sweep_runtime_health_missing_bad_run_config_grid",
    )
    csv_path.unlink()
    run_config.pop("budgets")
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        run_config_json=run_config_path,
        allow_missing=True,
        require_run_config=True,
    )

    assert rc == 1
    assert any("run_config.json missing non-empty budgets" in err for err in errors)
    assert any("results.csv is missing" in warn for warn in warnings)


def test_sweep_runtime_health_rejects_symlinked_results_csv(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_symlink",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("case_id,method,budget_mult,repeat,status\n", encoding="utf-8")
    real_is_symlink = Path.is_symlink

    def fake_is_symlink(path):
        if Path(path) == csv_path:
            return True
        return real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        allow_missing=True,
    )

    assert rc == 1
    assert any("is a symlink and cannot be trusted" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_cli_preserves_symlink_path_before_resolve(
    tmp_path,
    monkeypatch,
    capsys,
):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_cli_symlink",
    )
    csv_path = tmp_path / "results.csv"
    target_path = tmp_path / "external_target.csv"
    csv_path.write_text("case_id,method,budget_mult,repeat,status\n", encoding="utf-8")
    real_is_symlink = Path.is_symlink
    real_resolve = Path.resolve

    def fake_is_symlink(path):
        if Path(path) == csv_path:
            return True
        return real_is_symlink(path)

    def fake_resolve(path, *args, **kwargs):
        if Path(path) == csv_path:
            return target_path
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "resolve", fake_resolve)

    rc = mod.main([str(csv_path), "--allow-missing"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "is a symlink and cannot be trusted" in out
    assert str(csv_path) in out
    assert str(target_path) not in out


def test_sweep_runtime_health_rejects_header_only_results_csv(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_header_only",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("case_id,method,budget_mult,repeat,status\n", encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(csv_path)

    assert rc == 1
    assert any("header but no data rows" in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_allow_missing_tolerates_header_only_startup(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_header_only_startup",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("case_id,method,budget_mult,repeat,status\n", encoding="utf-8")

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        allow_missing=True,
    )

    assert rc == 0
    assert errors == []
    assert any("header but no data rows" in warn for warn in warnings)


def test_sweep_runtime_health_allow_missing_rejects_stale_header_only_results_csv(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_stale_header_only",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("case_id,method,budget_mult,repeat,status\n", encoding="utf-8")
    stale_time = csv_path.stat().st_mtime - 120.0
    os.utime(csv_path, (stale_time, stale_time))

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        stale_after_min=0.5,
        allow_missing=True,
    )

    assert rc == 1
    assert any("header but no data rows" in err for err in errors)
    assert any("threshold 0.5 min" in err for err in errors)
    assert warnings == []


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"stale_after_min": -1}, "stale_after_min"),
        ({"stale_after_min": float("inf")}, "stale_after_min"),
        ({"max_timeout_fraction": -0.1}, "max_timeout_fraction"),
        ({"max_timeout_fraction": 1.1}, "max_timeout_fraction"),
        ({"max_failed_fraction": float("nan")}, "max_failed_fraction"),
        ({"min_cells_for_fraction_gate": 0}, "min_cells_for_fraction_gate"),
        ({"min_cells_for_fraction_gate": True}, "min_cells_for_fraction_gate"),
        ({"expected_n_test": -1}, "expected_n_test"),
        ({"expected_test_seed": -1}, "expected_test_seed"),
        ({"seed_start": -1}, "seed_start"),
        ({"seed_strategy": "unknown"}, "seed_strategy"),
        ({"mem_cap_gb": 0}, "mem_cap_gb"),
        ({"mem_cap_gb": float("nan")}, "mem_cap_gb"),
        ({"mem_warning_fraction": 0}, "mem_warning_fraction"),
        ({"mem_warning_fraction": 1.1}, "mem_warning_fraction"),
        ({"disk_warning_gb": -1}, "disk_warning_gb"),
        ({"disk_warning_gb": float("nan")}, "disk_warning_gb"),
    ],
)
def test_sweep_runtime_health_rejects_invalid_runtime_options(
    tmp_path,
    kwargs,
    expected,
):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_options",
    )

    rc, errors, warnings = mod.check_runtime_health(
        tmp_path / "missing.csv",
        allow_missing=True,
        **kwargs,
    )

    assert rc == 1
    assert any(expected in err for err in errors)
    assert warnings == []


def test_sweep_runtime_health_fraction_gate_waits_for_enough_cells(tmp_path):
    mod = _load_module(
        "scripts/check_sweep_runtime_health.py",
        "check_sweep_runtime_health_fraction_gate",
    )
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = sorted(mod.PREREG_CASES)[0]
    method = sorted(mod.PREREG_METHODS)[0]
    case = db.get_case(case_id)
    case_fingerprint = case_source_fingerprint(case)
    method_fingerprint = mod.PREREG_METHOD_FINGERPRINTS[method]
    from benchmark.core import derive_seed
    timeout_trace = _runtime_trace_fields()
    timeout_trace["cell_finished_utc"] = "2026-06-01T00:10:00+00:00"

    _results_frame([
        {
            "case_id": case_id,
            **case_fingerprint,
            "domain": case.meta.domain,
            "method": method,
            "method_family": mod.PREREG_METHOD_FAMILIES[method],
            **method_fingerprint,
            "budget_mult": 5,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "n_train": 5 * case.meta.dim_input,
            "repeat": 0,
            "n_test": 1000,
            "test_seed": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "status": "timeout",
            "failure_category": "timeout",
            "nrmse": "",
            "t_total_s": 600.0,
            "error": "timeout after 600s",
            **timeout_trace,
        }
    ]).to_csv(csv_path, index=False)

    rc, errors, warnings = mod.check_runtime_health(
        csv_path,
        max_timeout_fraction=0.01,
        min_cells_for_fraction_gate=10,
    )

    assert rc == 0
    assert errors == []
    assert any("fraction gate deferred" in warn for warn in warnings)


def test_remote_progress_monitor_runs_runtime_health_gate():
    text = (ROOT / "scripts" / "check_progress_remote.sh").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")

    assert "check_sweep_runtime_health.py" in text
    assert "--allow-missing" in text
    assert "does not launch/resume" in text
    assert "PYTHON_CMD" in text
    assert "latest_pipeline_log" in text
    assert "logs/pipeline_*.log" in text
    assert "tail of latest pipeline log" in text
    assert '"${PYTHON_CMD}" scripts/_check_sweep_progress.py || true' in text
    assert "HEALTH_ARGS=(--allow-missing)" in text
    assert 'RUN_CONFIG="benchmark/results/full_benchmark/run_config.json"' in text
    assert 'HEALTH_ARGS+=(--run-config-json "${RUN_CONFIG}" --require-run-config)' in text
    assert "VALIDATE_CURRENT_ENV" in text
    assert 'HEALTH_ARGS+=(--validate-current-environment)' in text
    assert 'HEALTH_ARGS+=(--mem-cap-gb "${MEM_CAP_GB}")' in text
    assert 'HEALTH_ARGS+=(--disk-warning-gb "${DISK_WARNING_GB}")' in text
    assert 'HEALTH_ARGS+=(--stale-after-min "${STALE_AFTER_MIN}")' in text
    assert 'HEALTH_ARGS+=(--fail-on-stale)' in text
    assert '"${PYTHON_CMD}" scripts/check_sweep_runtime_health.py "${HEALTH_ARGS[@]}"' in text
    assert "HEALTH_RC=$?" in text
    assert "runtime health gate failed with exit=" in text
    assert "PROCESS_DEAD=0" in text
    assert "DEAD_INCOMPLETE_RC=0" in text
    assert "PROCESS_DEAD=1" in text
    assert "dead-process completion gate" in text
    assert "PREREG_TARGET" in text
    assert "summarize_progress" in text
    assert "sweep.pid is dead and headline sweep is incomplete" in text
    assert 'return "${DEAD_INCOMPLETE_RC}"' in text
    assert "snapshot || true" in text
    assert 'exit "${RC}"' in text
    assert "python scripts/_check_sweep_progress.py /path/to/results.csv" in handoff
    assert "python scripts/check_sweep_runtime_health.py /path/to/results.csv" in handoff
    assert "dead while the headline sweep is incomplete" in handoff
    assert "STALE_AFTER_MIN=45 FAIL_ON_STALE=1 bash scripts/check_progress_remote.sh -w" in handoff
    assert "forwards `--stale-after-min <minutes>`" in handoff
    assert "VALIDATE_CURRENT_ENV=1 bash scripts/check_progress_remote.sh -w" in handoff
    assert "forwards `--validate-current-environment`" in handoff


def test_remote_progress_monitor_validates_tail_and_pid_inputs():
    text = (ROOT / "scripts" / "check_progress_remote.sh").read_text(encoding="utf-8")

    assert "--tail requires a positive integer line count" in text
    assert "[[ $# -lt 2 || ! \"$2\" =~ ^[0-9]+$ || \"$2\" -lt 1 ]]" in text
    assert "tr -d '[:space:]' < \"${PID_FILE}\"" in text
    assert "malformed sweep.pid" in text
    assert "[[ ! \"${PID}\" =~ ^[0-9]+$ ]]" in text
    assert "pgrep -af" in text
    assert "scripts/run_pipeline.py|benchmark/scripts/run_full_benchmark.py|scripts/run_pipeline.sh" in text
    assert "active process(es) found without sweep.pid" in text
    assert "ps(first)" in text


def test_remote_progress_monitor_points_missing_pid_to_formal_pipeline():
    text = (ROOT / "scripts" / "check_progress_remote.sh").read_text(encoding="utf-8")

    assert "run run_pipeline.sh" in text
    assert "low-level Stage 2 restart" in text
    assert "no active formal pipeline process matched" in text


def test_scripts_readme_indexes_server_operator_tools():
    text = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    for name in [
        "check_server_launch_ready.py",
        "check_sweep_runtime_health.py",
        "check_server_return_package.py",
        "create_server_return_package.py",
        "create_server_return_package.sh",
    ]:
        assert f"[`{name}`]({name})" in text
    assert (
        "forwards `MEM_CAP_GB`, `DISK_WARNING_GB`, `STALE_AFTER_MIN`, and `FAIL_ON_STALE`"
        in text
    )


def test_operator_docs_include_neurips_benchmark_resource_audit_gate():
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    for text in (handoff, scripts_readme):
        assert "NeurIPS 2026 Evaluations & Datasets-style" in text
        assert "benchmark/reusable-artifact audit contract" in text
        assert "docs/EXPERIMENT_BENCHMARK_AUDIT.md" in text
    for marker in (
        "code/executable access",
        "accessibility",
        "transparency",
        "usability",
        "reproducibility",
        "benchmark-analysis",
        "stress-testing",
        "explicit limitations/failure modes",
    ):
        assert marker in handoff
    assert "check_local_pre_server_ready.py" in handoff
    assert "check_server_launch_ready.py" in handoff
    assert "check_local_pre_server_ready.py" in scripts_readme
    assert "check_server_launch_ready.py" in scripts_readme


def test_server_return_packaging_script_contract():
    script = (ROOT / "scripts" / "create_server_return_package.sh").read_text(encoding="utf-8")
    helper = (ROOT / "scripts" / "create_server_return_package.py").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")

    for required in [
        "benchmark/results/full_benchmark/results.csv",
        "benchmark/results/full_benchmark/sweep_integrity_report.md",
        "benchmark/results/full_benchmark/runtime_health_report.txt",
        "benchmark/results/full_benchmark/run_config.json",
        "benchmark/results/full_benchmark/RUN_MANIFEST.json",
        "benchmark/results/full_benchmark/analysis/test_set_sufficiency.csv",
        "docs/case_validation_status.csv",
        "docs/case_validation_sources.csv",
        "docs/case_validation_note_sources.csv",
        "docs/case_experiment_readiness.csv",
        "docs/CASE_VALIDATION_SUMMARY.md",
        "docs/CASE_WEB_VALIDATION_AUDIT.md",
        "docs/EXPERIMENT_BENCHMARK_AUDIT.md",
        "docs/PUBLICATION_LAUNCH_PREFLIGHT.md",
        "docs/STATISTICAL_ANALYSIS_PLAN.md",
        "paper/figdata/figdata_manifest.json",
        "paper/figure_manifest.json",
        "paper/numbers.json",
        "logs/pipeline_command_history.txt",
    ]:
        assert required in helper
        assert required in handoff
    assert "required file missing" in helper
    assert "required directory is empty" in helper
    assert "declared PASS verdict" in helper
    assert "sweep_integrity_report.md" in helper
    assert "runtime_health_report.txt" in helper
    assert "runtime health PASS verdict" in helper
    assert "runtime_health_report.txt" in handoff
    assert "runtime-health PASS verdict" in handoff
    assert "run_config.json contract matched partial results" in helper
    assert "run_config.json contract matched partial results" in handoff
    assert "run_config.json runtime controls recorded and valid" in helper
    assert "run_config.json runtime controls recorded and valid" in handoff
    assert "run_config.json runtime environment recorded and valid" in helper
    assert "run_config.json runtime environment recorded and valid" in handoff
    assert "check_sweep_runtime_health.py" in script
    assert "--run-config-json" in script
    assert "--validate-current-environment" in script
    assert "--require-run-config" in script
    assert "resolve_python()" in script
    assert ".venv/bin/python" in script
    assert 'PYTHON_CMD="$(resolve_python)"' in script
    assert "cleanup_tmp_health()" in script
    assert 'trap cleanup_tmp_health EXIT HUP INT TERM' in script
    assert 'TMP_HEALTH=""' in script
    assert '"${PYTHON_CMD}" scripts/check_sweep_runtime_health.py' in script
    assert 'exec "${PYTHON_CMD}" scripts/create_server_return_package.py' in script
    assert "check_case_web_validation_audit_content" in helper
    assert "server_return_package_" in helper
    assert ".tar.gz" in helper
    assert "hashlib.sha256" in helper
    assert "check_return_tarball" in helper
    assert "Removing the invalid package" in helper
    assert "overwrite packaged artifact" in helper
    assert "must not be written inside packaged" in helper
    assert "must end with .tar.gz" in helper
    assert "return package output is a symlink" in helper
    assert "return package output parent directory is a symlink" in helper
    assert "return package sha256 sidecar is a symlink" in helper
    assert "sha256 sidecar already exists" in helper
    assert "python scripts/check_server_return_package.py" in helper
    assert "create_server_return_package.py" in script
    assert "bash scripts/create_server_return_package.sh" in handoff
    assert "rejects symlinked output paths" in handoff
    assert "symlinked output parent directories" in handoff
    assert "must end in" in handoff
    assert "existing output or existing sidecar" in handoff
    assert "check_server_return_package.py --package" in handoff
    assert "package-filename binding" in handoff
    assert "cleans that temporary file on interruption" in handoff
    assert "place the tarball inside a packaged directory" in handoff
    assert "rechecks the P2" in handoff
    assert "failed, missing, or stale final" in handoff
    assert "health report is rejected" in handoff
    assert "Evidence tier remains P2" in handoff
    assert "duplicate member paths" in handoff
    assert "unsupported special-file members" in handoff
    assert "required directory paths are real directories" in handoff
    assert "package path and" in handoff
    assert "sidecar path themselves are also rejected if they are symlinks" in handoff
    assert "rejects symlinked required files or directories" in handoff
    assert "--package server_return_package_<UTC>.tar.gz" in (
        ROOT / "scripts" / "README.md"
    ).read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    assert "sidecar digest and filename binding" in scripts_readme
    assert "duplicate member paths" in scripts_readme
    assert "unsupported special members" in scripts_readme
    assert "required directory members are real directories" in scripts_readme
    assert "rechecks P2 web-validation audit" in scripts_readme
    assert "runtime-health report" in scripts_readme
    assert "runtime_health_report.txt" in scripts_readme
    assert "evidence-tier markers against the case-validation ledger" in scripts_readme
    assert "rejects symlinked package/sidecar paths" in scripts_readme
    assert "rejects symlinked base directories" in scripts_readme
    assert "rejects symlinked required files/directories" in scripts_readme
    assert "symlinked output paths" in scripts_readme
    assert "symlinked output parent directories" in scripts_readme
    assert "symlinked sidecar paths" in scripts_readme
    assert "existing output or sidecar" in scripts_readme


def test_server_return_package_hashing_streams_large_tarballs():
    create = (ROOT / "scripts" / "create_server_return_package.py").read_text(encoding="utf-8")
    check = (ROOT / "scripts" / "check_server_return_package.py").read_text(encoding="utf-8")

    for text in (create, check):
        assert "def _sha256_file" in text
        assert "f.read(1024 * 1024)" in text
    assert "tmp_output.read_bytes()" not in create
    assert "package_path.read_bytes()" not in check


def test_create_server_return_package_import_guard_works_outside_repo_root(tmp_path):
    code = (
        "import importlib.util, pathlib; "
        f"root=pathlib.Path({str(ROOT)!r}); "
        "spec=importlib.util.spec_from_file_location("
        "'csr_pkg', root/'scripts'/'create_server_return_package.py'"
        "); "
        "mod=importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "errors=mod.check_required_artifacts(pathlib.Path.cwd()); "
        "print(errors); "
        "raise SystemExit(0 if any('required file missing' in err for err in errors) else 1)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def _write_minimal_return_package_artifacts(root: Path, mod) -> None:
    for rel_path in mod.REQUIRED_FILES:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel_path.as_posix() == "benchmark/results/full_benchmark/sweep_integrity_report.md":
            path.write_text("PASS: sweep satisfies the declared integrity gate.\n", encoding="utf-8")
        elif rel_path.as_posix() == "benchmark/results/full_benchmark/runtime_health_report.txt":
            continue
        elif rel_path.as_posix() == "benchmark/results/full_benchmark/results.csv":
            path.write_text(
                "case_id,method,budget_mult,repeat,status,failure_category\n"
                "minimal_case,m1,10,0,ok,ok\n"
                "minimal_case,m1,10,1,ok,ok\n",
                encoding="utf-8",
            )
        elif rel_path.as_posix() == "benchmark/results/full_benchmark/run_config.json":
            _write_minimal_publication_run_config(path)
        elif rel_path.as_posix() == "benchmark/results/full_benchmark/RUN_MANIFEST.json":
            continue
        elif rel_path.as_posix() == "benchmark/results/full_benchmark/python_environment_pip_freeze.txt":
            path.write_text(_publication_pip_freeze_contents(), encoding="utf-8")
        elif rel_path.as_posix() == "docs/case_validation_status.csv":
            path.write_text(
                "case_id,domain,dim,validation_note_path,external_validation_scope,"
                "formula_status,code_status,boundary_test_status,targeted_physics_test,"
                "risk_level,final_disposition,next_action\n"
                "minimal_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
                "ok,cross_validation_present,tests/test_x.py,P2,plausibility only,keep\n",
                encoding="utf-8",
            )
        elif rel_path.as_posix() == "docs/CASE_WEB_VALIDATION_AUDIT.md":
            path.write_text(
                "| Case | Web/source check | Code judgment | Action |\n"
                "|---|---|---|---|\n"
                "| `minimal_case` | Evidence tier remains P2. Reference checked. | "
                "Plausible stress test. | Keep P2 plausibility-only. |\n",
                encoding="utf-8",
            )
        elif rel_path.as_posix() == "paper/figdata/figdata_manifest.json":
            results_csv = root / "benchmark" / "results" / "full_benchmark" / "results.csv"
            path.write_text(
                json.dumps({
                    "status": "complete",
                    "results_summary": {
                        "exists": True,
                        "n_rows": 2,
                        "n_cases": 1,
                        "n_methods": 1,
                        "budgets": [10],
                        "repeats": [0, 1],
                        "status_counts": {"ok": 2},
                        "failure_category_counts": {"ok": 2},
                    },
                    "inputs": [{
                        "label": "results_csv",
                        "path": "benchmark/results/full_benchmark/results.csv",
                        "exists": True,
                        "size_bytes": results_csv.stat().st_size,
                        "sha256": _file_sha256(results_csv),
                    }],
                    "figdata_files": [],
                    "errors": [],
                }),
                encoding="utf-8",
            )
        elif rel_path.as_posix() == "paper/numbers.json":
            results_csv = root / "benchmark" / "results" / "full_benchmark" / "results.csv"
            path.write_text(
                json.dumps({
                    "nCases": 1,
                    "nMethods": 1,
                    "nRepeats": 2,
                    "nBudgets": 1,
                    "budgetList": "10",
                    "totalCells": 2,
                    "totalRows": 2,
                    "okRows": 2,
                    "timeoutRows": 0,
                    "failedRows": 0,
                    "_provenance": {
                        "script": "scripts/fill_paper_numbers.py",
                        "generated_at": "2026-06-02T00:00:00+00:00",
                        "inputs": [{
                            "label": "results_csv",
                            "path": "benchmark/results/full_benchmark/results.csv",
                            "exists": True,
                            "size_bytes": results_csv.stat().st_size,
                            "sha256": _file_sha256(results_csv),
                        }],
                    },
                }),
                encoding="utf-8",
            )
        elif rel_path.as_posix() == "logs/pipeline_command_history.txt":
            path.write_text(
                "pipeline_invocation\t"
                "python scripts/run_pipeline.py --profile full --confirm-full-run\n",
                encoding="utf-8",
            )
        else:
            path.write_text("artifact\n", encoding="utf-8")
    _write_minimal_machine_readable_metadata(root)
    results_csv = root / "benchmark" / "results" / "full_benchmark" / "results.csv"
    run_config_json = (
        root / "benchmark" / "results" / "full_benchmark" / "run_config.json"
    )
    _write_minimal_publication_manifest(
        root / "benchmark" / "results" / "full_benchmark" / "RUN_MANIFEST.json",
        results_csv=results_csv,
    )
    runtime_health = root / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    runtime_health.parent.mkdir(parents=True, exist_ok=True)
    runtime_health.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{runtime_health.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{runtime_health.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{runtime_health.parent / 'run_config.json'}\n"
        "[warn] run_config.json torch reproducibility policy recorded and valid: "
        f"{runtime_health.parent / 'run_config.json'}\n"
        "[warn] current runtime environment matches run_config.json snapshot: "
        f"{runtime_health.parent / 'run_config.json'}\n"
        '[summary] runtime failure-category telemetry summarized: '
        'unique_cells=1 failure_category_counts={"ok": 1} '
        'non_ok_failure_category_counts={}\n'
        '[summary] runtime resource telemetry summarized: '
        'rss_gb_peak={"max": 1.75, "min": 1.75, "n_valid": 1, '
        '"p50": 1.75, "p95": 1.75, "top_methods": {"m1": 1.75}}\n'
        '[summary] runtime wall-time telemetry summarized: '
        't_total_s={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[artifact] "
        f"run_config_json={run_config_json.resolve()} "
        f"size_bytes={run_config_json.stat().st_size} "
        f"sha256={_file_sha256(run_config_json)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )
    for rel_path in mod.REQUIRED_DIRS:
        path = root / rel_path
        path.mkdir(parents=True, exist_ok=True)
        (path / "artifact.txt").write_text("artifact\n", encoding="utf-8")


def test_create_server_return_package_fails_before_writing_when_required_artifacts_missing(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_missing",
    )
    output = tmp_path / "return.tar.gz"

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)

    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_requires_passing_sweep_audit(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_sweep_audit_required",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "sweep_integrity_report.md"
    report.write_text(
        "# Sweep Integrity Report\n\n## Findings\n\n### Errors\n\n- duplicate cell\n",
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("does not contain the declared PASS verdict" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_requires_passing_runtime_health_report(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_health_required",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    report.write_text(
        "[error] latest cells with protocol_digest inconsistent with row-level protocol fields: 1\n"
        "[result] FAIL - runtime sweep health needs attention.\n",
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("runtime health PASS verdict" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_requires_valid_machine_readable_metadata(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_metadata_required",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    (tmp_path / "codemeta.json").write_text(
        json.dumps({
            "@context": "https://example.org/software-schema",
            "@type": "SoftwareSourceCode",
            "name": "doe-testbed",
            "description": "x",
            "version": "0.4.0-dev",
            "codeRepository": "https://example.org/repo",
            "license": "MIT",
            "author": [],
        }),
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("codemeta.json @context must reference CodeMeta" in err for err in errors)
    assert any("codemeta.json author must be a non-empty list" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()


def test_create_server_return_package_rejects_stale_runtime_health_report(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_health_stale",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    report.write_text(
        "[artifact] results_csv=/old/results.csv size_bytes=999 sha256="
        f"{'0' * 64}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("size_bytes does not match current results.csv" in err for err in errors)
    assert any("sha256 does not match current results.csv" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_rejects_stale_runtime_health_run_config_fingerprint(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_health_stale_run_config",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    run_config_json = report.parent / "run_config.json"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{run_config_json}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{run_config_json}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{run_config_json}\n"
        "[warn] current runtime environment matches run_config.json snapshot: "
        f"{run_config_json}\n"
        '[summary] runtime failure-category telemetry summarized: '
        'unique_cells=1 failure_category_counts={"ok": 1} '
        'non_ok_failure_category_counts={}\n'
        '[summary] runtime resource telemetry summarized: '
        'rss_gb_peak={"max": 1.75, "min": 1.75, "n_valid": 1, '
        '"p50": 1.75, "p95": 1.75, "top_methods": {"m1": 1.75}}\n'
        '[summary] runtime wall-time telemetry summarized: '
        't_total_s={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[artifact] "
        f"run_config_json={run_config_json.resolve()} "
        "size_bytes=999 sha256="
        f"{'0' * 64}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    errors = mod.check_required_artifacts(tmp_path)

    assert any("run_config.json size_bytes does not match" in err for err in errors)
    assert any("run_config.json sha256 does not match" in err for err in errors)


def test_create_server_return_package_rejects_noncanonical_runtime_health_results_path(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_health_noncanonical_path",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] current runtime environment matches run_config.json snapshot: "
        f"{report.parent / 'run_config.json'}\n"
        '[summary] runtime failure-category telemetry summarized: '
        'unique_cells=1 failure_category_counts={"ok": 1} '
        'non_ok_failure_category_counts={}\n'
        '[summary] runtime resource telemetry summarized: '
        'rss_gb_peak={"max": 1.75, "min": 1.75, "n_valid": 1, '
        '"p50": 1.75, "p95": 1.75, "top_methods": {"m1": 1.75}}\n'
        '[summary] runtime wall-time telemetry summarized: '
        't_total_s={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        "[artifact] "
        f"results_csv={tmp_path / 'benchmark' / 'results' / 'shadow_copy' / 'results.csv'} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("canonical headline archive location" in err for err in errors)
    assert any("does not match current results.csv" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_requires_run_config_bound_runtime_health_report(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_health_run_config_bound",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("checked against run_config.json" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_requires_runtime_control_bound_health_report(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_health_controls_bound",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("runtime controls were recorded and validated" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_requires_runtime_environment_bound_health_report(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_health_environment_bound",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("runtime environment was recorded and validated" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_requires_torch_repro_bound_health_report(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_health_torch_repro_bound",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    text = report.read_text(encoding="utf-8")
    text = "\n".join(
        line
        for line in text.splitlines()
        if "run_config.json torch reproducibility policy recorded and valid" not in line
    )
    report.write_text(text + "\n", encoding="utf-8")
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("torch reproducibility policy was recorded and validated" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_requires_current_environment_matched_health_report(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_current_environment_match_required",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("current packaging environment matched the run_config.json startup snapshot" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_requires_runtime_failure_category_summary(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_failure_summary",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[summary] runtime resource telemetry summarized: "
        'rss_gb_peak={"max": 1.75, "min": 1.75, "n_valid": 1}\n'
        "[summary] runtime wall-time telemetry summarized: "
        't_total_s={"max": 1.0, "min": 1.0, "n_valid": 1}\n'
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    errors = mod.check_required_artifacts(tmp_path)

    assert any("failure-category telemetry" in err for err in errors)


def test_create_server_return_package_requires_runtime_resource_summary(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_resource_summary",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[summary] runtime failure-category telemetry summarized: "
        'unique_cells=1 failure_category_counts={"ok": 1} '
        "non_ok_failure_category_counts={}\n"
        "[summary] runtime wall-time telemetry summarized: "
        't_total_s={"max": 1.0, "min": 1.0, "n_valid": 1}\n'
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    errors = mod.check_required_artifacts(tmp_path)

    assert any("runtime resource telemetry" in err for err in errors)


def test_create_server_return_package_requires_runtime_wall_time_summary(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_runtime_wall_time_summary",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[summary] runtime failure-category telemetry summarized: "
        'unique_cells=1 failure_category_counts={"ok": 1} '
        "non_ok_failure_category_counts={}\n"
        "[summary] runtime resource telemetry summarized: "
        'rss_gb_peak={"max": 1.75, "min": 1.75, "n_valid": 1}\n'
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    errors = mod.check_required_artifacts(tmp_path)

    assert any("runtime wall-time telemetry" in err for err in errors)


def test_create_server_return_package_rejects_python_environment_lock_drift(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_python_environment_drift",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    pip_freeze = tmp_path / "benchmark" / "results" / "full_benchmark" / "python_environment_pip_freeze.txt"
    pip_freeze.write_text(
        _publication_pip_freeze_contents(overrides={"botorch": "9.9.9"}),
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any(
        "python environment lock version does not match run_config.json for botorch"
        in err
        for err in errors
    )
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_rejects_manifest_temp_storage_drift(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_manifest_temp_storage_drift",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    manifest = tmp_path / "benchmark" / "results" / "full_benchmark" / "RUN_MANIFEST.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["machine"]["temp_storage"]["resolved_path"] = "/tmp/posthoc-manifest-scratch"
    payload["machine"]["temp_storage"]["env"]["SCRATCH"] = "/tmp/posthoc-manifest-scratch"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("machine.temp_storage.resolved_path does not match" in err for err in errors)
    assert any("temp_storage.env for SCRATCH" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_rechecks_p2_web_audit_evidence_tiers(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_p2_web_audit_content",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    docs_dir = tmp_path / "docs"
    (docs_dir / "case_validation_status.csv").write_text(
        "case_id,domain,dim,validation_note_path,external_validation_scope,"
        "formula_status,code_status,boundary_test_status,targeted_physics_test,"
        "risk_level,final_disposition,next_action\n"
        "implicit_case,structures,1,NONE,scope,targeted_tested_no_code_change,"
        "ok,cross_validation_present,tests/test_x.py,P2,plausibility only,keep\n",
        encoding="utf-8",
    )
    (docs_dir / "CASE_WEB_VALIDATION_AUDIT.md").write_text(
        "| Case | Web/source check | Code judgment | Action |\n"
        "|---|---|---|---|\n"
        "| `implicit_case` | reference checked | plausible | Keep P2 plausibility-only. |\n",
        encoding="utf-8",
    )
    output = tmp_path / "return.tar.gz"

    errors = mod.check_required_artifacts(tmp_path)

    assert any("markers do not match" in err for err in errors)
    assert any("missing `Evidence tier remains P2` marker" in err for err in errors)
    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)
    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_rejects_symlinks_before_writing(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_symlink",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    link_like = tmp_path / "benchmark" / "results" / "ranking_reversal" / "link.txt"
    link_like.write_text("looks like a link to the checker\n", encoding="utf-8")
    output = tmp_path / "return.tar.gz"
    path_type = type(link_like)
    original_is_symlink = path_type.is_symlink

    def fake_is_symlink(self):
        if Path(self) == link_like:
            return True
        return original_is_symlink(self)

    monkeypatch.setattr(path_type, "is_symlink", fake_is_symlink)

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)

    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_rejects_symlinked_output_path(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_output_symlink",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"existing link target stand-in")
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(self):
        if self == output:
            return True
        return original_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output, overwrite=True)

    assert excinfo.value.code == 2
    assert output.read_bytes() == b"existing link target stand-in"
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_rejects_symlinked_output_parent(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_output_parent_symlink",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    output_dir = tmp_path / "linked-output"
    output_dir.mkdir()
    output = output_dir / "return.tar.gz"
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(self):
        if self == output_dir:
            return True
        return original_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)

    assert excinfo.value.code == 2
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_rejects_symlinked_sidecar_path(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_sidecar_symlink",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    sidecar = output.with_name(f"{output.name}.sha256")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("existing sidecar target stand-in\n", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def fake_is_symlink(self):
        if self == sidecar:
            return True
        return original_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output, overwrite=True)

    assert excinfo.value.code == 2
    assert not output.exists()
    assert sidecar.read_text(encoding="utf-8") == "existing sidecar target stand-in\n"


def test_create_server_return_package_refuses_existing_sidecar_without_overwrite(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_existing_sidecar",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    sidecar = output.with_name(f"{output.name}.sha256")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("previous digest\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)

    assert excinfo.value.code == 2
    assert not output.exists()
    assert sidecar.read_text(encoding="utf-8") == "previous digest\n"

    package = mod.create_package(tmp_path, output, overwrite=True)
    assert package == output.resolve()
    assert sidecar.read_text(encoding="utf-8") != "previous digest\n"


def test_create_server_return_package_rejects_special_members_before_writing(
    tmp_path, monkeypatch
):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_special_member",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    special_like = tmp_path / "benchmark" / "results" / "ranking_reversal" / "device"
    special_like.write_text("simulated special tar member\n", encoding="utf-8")
    output = tmp_path / "return.tar.gz"
    path_type = type(special_like)
    original_is_file = path_type.is_file
    original_is_dir = path_type.is_dir

    def fake_is_file(self):
        if Path(self) == special_like:
            return False
        return original_is_file(self)

    def fake_is_dir(self):
        if Path(self) == special_like:
            return False
        return original_is_dir(self)

    monkeypatch.setattr(path_type, "is_file", fake_is_file)
    monkeypatch.setattr(path_type, "is_dir", fake_is_dir)

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)

    assert excinfo.value.code == 1
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_writes_tarball_and_sha256(tmp_path):
    import tarfile

    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_success",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"

    package = mod.create_package(tmp_path, output)

    assert package == output.resolve()
    sidecar = output.with_name(f"{output.name}.sha256")
    assert sidecar.is_file()
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  {output.name}\n"
    with tarfile.open(output, "r:gz") as tar:
        names = set(tar.getnames())
    assert ".zenodo.json" in names
    assert "CITATION.cff" in names
    assert "codemeta.json" in names
    assert "croissant-metadata.json" in names
    assert "ro-crate-metadata.json" in names
    assert "benchmark/results/full_benchmark/results.csv" in names
    assert "benchmark/results/full_benchmark/runtime_health_report.txt" in names
    assert "benchmark/results/full_benchmark/run_config.json" in names
    assert "docs/CASE_VALIDATION_SUMMARY.md" in names
    assert "docs/CASE_WEB_VALIDATION_AUDIT.md" in names
    assert "docs/case_validation_status.csv" in names
    assert "logs/pipeline_command_history.txt" in names
    assert "paper/figdata/figdata_manifest.json" in names
    assert "paper/figures/artifact.txt" in names
    assert all(not Path(name).is_absolute() for name in names)


def test_create_server_return_package_excludes_optional_robustness_dirs(tmp_path):
    import tarfile

    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_excludes_robustness",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    for optional_dir in [
        tmp_path / "benchmark" / "results" / "sota_robustness",
        tmp_path / "benchmark" / "results" / "mf_supplementary",
    ]:
        optional_dir.mkdir(parents=True, exist_ok=True)
        (optional_dir / "results.csv").write_text("case_id,method\nx,y\n", encoding="utf-8")

    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    mod.create_package(tmp_path, output)

    with tarfile.open(output, "r:gz") as tar:
        names = set(tar.getnames())
    assert not any(name.startswith("benchmark/results/sota_robustness") for name in names)
    assert not any(name.startswith("benchmark/results/mf_supplementary") for name in names)
    required_paths = {path.as_posix() for path in mod.REQUIRED_FILES + mod.REQUIRED_DIRS}
    assert "benchmark/results/sota_robustness" not in required_paths
    assert "benchmark/results/mf_supplementary" not in required_paths


def test_server_return_package_can_verify_tarball_and_sha256(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_for_check",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_tarball_ok",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"

    package = create.create_package(tmp_path, output)

    assert check.check_return_tarball(package) == []


def test_server_return_package_rejects_tarball_with_stale_figdata_results_summary(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_stale_figdata_summary_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_stale_figdata_summary",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    manifest = tmp_path / "paper" / "figdata" / "figdata_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["results_summary"]["status_counts"] = {"timeout": 2}
    manifest.write_text(json.dumps(data), encoding="utf-8")

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any(
        "figdata_manifest.json results_summary status_counts mismatch against packaged results.csv"
        in err
        for err in errors
    )


def test_server_return_package_rejects_tarball_with_stale_numbers_results_provenance(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_stale_numbers_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_stale_numbers",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    numbers = tmp_path / "paper" / "numbers.json"
    data = json.loads(numbers.read_text(encoding="utf-8"))
    data["_provenance"]["inputs"][0]["sha256"] = "0" * 64
    numbers.write_text(json.dumps(data), encoding="utf-8")

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any(
        "paper/numbers.json results_csv provenance sha256 does not match packaged results.csv"
        in err
        for err in errors
    )


def test_server_return_package_rejects_tarball_with_invalid_machine_readable_metadata(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_bad_metadata_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_bad_metadata_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    (tmp_path / "ro-crate-metadata.json").write_text(
        json.dumps({
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": [
                {
                    "@id": "ro-crate-metadata.json",
                    "@type": "CreativeWork",
                    "about": {"@id": "./"},
                },
                {"@id": "./", "@type": "Dataset", "hasPart": []},
            ],
        }),
        encoding="utf-8",
    )

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any(
        "server return package ro-crate-metadata.json root Dataset hasPart must be non-empty"
        in err
        for err in errors
    )


def test_server_return_package_rejects_tarball_without_passing_sweep_audit(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_failed_audit_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_failed_audit_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "sweep_integrity_report.md"
    report.write_text("PASS: sweep satisfies the declared integrity gate.\n", encoding="utf-8")
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report.write_text("# Sweep Integrity Report\n\n### Errors\n\n- stale hash\n", encoding="utf-8")

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("sweep_integrity_report.md does not contain the declared PASS verdict" in err for err in errors)


def test_server_return_package_rejects_tarball_without_passing_runtime_health(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_failed_health_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_failed_health_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    report.write_text(
        "[error] latest cells with invalid status values: ['partial']\n"
        "[result] FAIL - runtime sweep health needs attention.\n",
        encoding="utf-8",
    )

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("runtime_health_report.txt does not contain the runtime health PASS verdict" in err for err in errors)


def test_server_return_package_rejects_tarball_with_stale_runtime_health_fingerprint(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_stale_health_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_stale_health_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    report.write_text(
        "[artifact] results_csv=/old/results.csv size_bytes=999 sha256="
        f"{'0' * 64}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("runtime_health_report.txt results.csv size_bytes does not match" in err for err in errors)
    assert any("runtime_health_report.txt results.csv sha256 does not match" in err for err in errors)


def test_server_return_package_rejects_tarball_with_stale_runtime_health_run_config_fingerprint(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_stale_config_health_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_stale_config_health_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    run_config_json = report.parent / "run_config.json"
    text = report.read_text(encoding="utf-8")
    text = re.sub(
        r"^\[artifact\]\s+run_config_json=.*$",
        lambda _match: (
            f"[artifact] run_config_json={run_config_json.resolve()} "
            f"size_bytes=999 sha256={'0' * 64}"
        ),
        text,
        flags=re.MULTILINE,
    )
    report.write_text(text, encoding="utf-8")

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("runtime_health_report.txt run_config.json size_bytes does not match" in err for err in errors)
    assert any("runtime_health_report.txt run_config.json sha256 does not match" in err for err in errors)


def test_server_return_package_rejects_tarball_with_noncanonical_runtime_health_results_path(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_noncanonical_health_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_noncanonical_health_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] current runtime environment matches run_config.json snapshot: "
        f"{report.parent / 'run_config.json'}\n"
        '[summary] runtime failure-category telemetry summarized: '
        'unique_cells=1 failure_category_counts={"ok": 1} '
        'non_ok_failure_category_counts={}\n'
        '[summary] runtime resource telemetry summarized: '
        'rss_gb_peak={"max": 1.75, "min": 1.75, "n_valid": 1, '
        '"p50": 1.75, "p95": 1.75, "top_methods": {"m1": 1.75}}\n'
        '[summary] runtime wall-time telemetry summarized: '
        't_total_s={"max": 1.0, "min": 1.0, "n_valid": 1, '
        '"p50": 1.0, "p95": 1.0, "top_methods": {"m1": 1.0}}\n'
        "[artifact] "
        f"results_csv=/tmp/shadow/results.csv "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("results.csv path does not point at the canonical headline archive location" in err for err in errors)


def test_server_return_package_rejects_tarball_without_run_config_bound_runtime_health(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_unbound_health_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_unbound_health_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("checked against run_config.json" in err for err in errors)


def test_server_return_package_rejects_tarball_without_runtime_control_bound_health(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_unbound_runtime_controls_health_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_unbound_runtime_controls_health_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("runtime controls were recorded and validated" in err for err in errors)


def test_server_return_package_rejects_tarball_without_runtime_environment_bound_health(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_unbound_runtime_environment_health_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_unbound_runtime_environment_health_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("runtime environment was recorded and validated" in err for err in errors)


def test_server_return_package_rejects_tarball_without_torch_repro_bound_health(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_unbound_torch_repro_health_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_unbound_torch_repro_health_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    text = report.read_text(encoding="utf-8")
    text = "\n".join(
        line
        for line in text.splitlines()
        if "run_config.json torch reproducibility policy recorded and valid" not in line
    )
    report.write_text(text + "\n", encoding="utf-8")

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("torch reproducibility policy was recorded and validated" in err for err in errors)


def test_server_return_package_rejects_tarball_without_current_environment_matched_health(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_unbound_current_environment_health_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_unbound_current_environment_health_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("current packaging environment matched run_config.json" in err for err in errors)


def test_server_return_package_rejects_tarball_without_runtime_telemetry_summaries(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_unbound_runtime_summary_tar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_unbound_runtime_summary_tar",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    report = tmp_path / "benchmark" / "results" / "full_benchmark" / "runtime_health_report.txt"
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    report.write_text(
        "[warn] run_config.json contract matched partial results: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime controls recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[warn] run_config.json runtime environment recorded and valid: "
        f"{report.parent / 'run_config.json'}\n"
        "[artifact] "
        f"results_csv={results_csv.resolve()} "
        f"size_bytes={results_csv.stat().st_size} "
        f"sha256={_file_sha256(results_csv)}\n"
        "[result] PASS - runtime sweep health is acceptable.\n",
        encoding="utf-8",
    )

    import tarfile

    with tarfile.open(package, "w:gz") as tar:
        for rel_path in create.REQUIRED_FILES + create.REQUIRED_DIRS:
            tar.add(tmp_path / rel_path, arcname=rel_path.as_posix(), recursive=True)
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("failure-category telemetry" in err for err in errors)
    assert any("runtime resource telemetry" in err for err in errors)
    assert any("runtime wall-time telemetry" in err for err in errors)


def test_server_return_package_rejects_symlinked_tarball_or_sidecar(tmp_path, monkeypatch):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_symlink_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_tarball_symlinks",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    sidecar = package.with_name(f"{package.name}.sha256")
    original_is_symlink = Path.is_symlink

    def fake_package_symlink(self):
        if self == package:
            return True
        return original_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_package_symlink)
    errors = check.check_return_tarball(package)
    assert any("server return package is a symlink" in err for err in errors)

    def fake_sidecar_symlink(self):
        if self == sidecar:
            return True
        return original_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_sidecar_symlink)
    errors = check.check_return_tarball(package)
    assert any("sha256 sidecar is a symlink" in err for err in errors)


def test_server_return_package_cli_preserves_package_path_before_resolve(
    tmp_path,
    monkeypatch,
    capsys,
):
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_cli_symlink",
    )
    package = tmp_path / "return.tar.gz"
    target = tmp_path / "external_target.tar.gz"
    package.write_bytes(b"not a real tarball")
    package.with_name(f"{package.name}.sha256").write_text(f"{'0' * 64}  {package.name}\n", encoding="utf-8")
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve

    def fake_is_symlink(self):
        if self == package:
            return True
        return original_is_symlink(self)

    def fake_resolve(self, *args, **kwargs):
        if self == package:
            return target
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "resolve", fake_resolve)

    rc = check.main(["--package", str(package)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "server return package is a symlink" in out
    assert str(package) in out
    assert str(target) not in out


def test_server_return_package_rejects_sidecar_filename_mismatch(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_sidecar_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_sidecar_filename",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    sidecar = package.with_name(f"{package.name}.sha256")
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  other_return.tar.gz\n", encoding="utf-8")

    errors = check.check_return_tarball(package)

    assert any("sidecar filename does not match package" in err for err in errors)
    assert any("other_return.tar.gz" in err for err in errors)


def test_server_return_package_rejects_digest_only_sidecar(tmp_path):
    create = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_digest_only_source",
    )
    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_digest_only",
    )
    _write_minimal_return_package_artifacts(tmp_path, create)
    output = tmp_path / "benchmark" / "results" / "return.tar.gz"
    package = create.create_package(tmp_path, output)
    sidecar = package.with_name(f"{package.name}.sha256")
    sidecar.write_text(f"{hashlib.sha256(package.read_bytes()).hexdigest()}\n", encoding="utf-8")

    errors = check.check_return_tarball(package)

    assert any("sha256 sidecar is malformed" in err for err in errors)


def test_server_return_package_rejects_corrupt_or_unsafe_tarball(tmp_path):
    import io
    import tarfile

    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_tarball_bad",
    )
    package = tmp_path / "bad.tar.gz"
    safe_payload = tmp_path / "payload.txt"
    safe_payload.write_text("payload\n", encoding="utf-8")
    with tarfile.open(package, "w:gz") as tar:
        tar.add(safe_payload, arcname="../escape.txt")
        windows_payload = b"windows absolute member\n"
        windows_member = tarfile.TarInfo("C:/escape.txt")
        windows_member.size = len(windows_payload)
        tar.addfile(windows_member, io.BytesIO(windows_payload))
        drive_payload = b"windows drive member\n"
        drive_member = tarfile.TarInfo("D:escape.txt")
        drive_member.size = len(drive_payload)
        tar.addfile(drive_member, io.BytesIO(drive_payload))
        special = tarfile.TarInfo("logs/device")
        special.type = tarfile.CHRTYPE
        special.devmajor = 1
        special.devminor = 3
        tar.addfile(special)
    (tmp_path / "bad.tar.gz.sha256").write_text(
        f"{'0' * 64}  bad.tar.gz\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("sha256 mismatch" in err for err in errors)
    assert any("unsafe member path" in err for err in errors)
    assert any("C:/escape.txt" in err for err in errors)
    assert any("D:escape.txt" in err for err in errors)
    assert any("unsupported special member" in err for err in errors)
    assert any("missing required file" in err for err in errors)


def test_server_return_package_rejects_duplicate_tarball_members(tmp_path):
    import io
    import tarfile

    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_duplicate_members",
    )
    package = tmp_path / "duplicate.tar.gz"
    package_path = "benchmark/results/full_benchmark/results.csv"
    with tarfile.open(package, "w:gz") as tar:
        for payload in (b"case_id\nfirst\n", b"case_id\nsecond\n"):
            member = tarfile.TarInfo(package_path)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("duplicate member path" in err for err in errors)
    assert any(package_path in err for err in errors)


def test_server_return_package_rejects_optional_robustness_members(tmp_path):
    import io
    import tarfile

    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_no_robustness_dirs",
    )
    package = tmp_path / "return.tar.gz"
    with tarfile.open(package, "w:gz") as tar:
        payload = b"case_id,method\nx,y\n"
        member = tarfile.TarInfo("benchmark/results/sota_robustness/results.csv")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("must not include optional robustness directory" in err for err in errors)
    assert any("benchmark/results/sota_robustness" in err for err in errors)


def test_server_return_package_tarball_rejects_command_history_manifest_drift(tmp_path):
    import io
    import tarfile

    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_tarball_command_history_drift",
    )
    package = tmp_path / "return.tar.gz"
    formal_command = "python scripts/run_pipeline.py --profile full --confirm-full-run"
    manifest = {
        "experiment": {
            "command_files_requested": ["logs/pipeline_command_history.txt"],
            "command_entries": [
                {"label": "pipeline_invocation", "command": formal_command},
            ],
        }
    }
    bad_history = (
        "pipeline_invocation\t"
        "python scripts/run_pipeline.py --profile full --confirm-full-run --dry-run\n"
    )
    with tarfile.open(package, "w:gz") as tar:
        manifest_payload = json.dumps(manifest).encode("utf-8")
        manifest_member = tarfile.TarInfo(
            "benchmark/results/full_benchmark/RUN_MANIFEST.json"
        )
        manifest_member.size = len(manifest_payload)
        tar.addfile(manifest_member, io.BytesIO(manifest_payload))

        history_payload = bad_history.encode("utf-8")
        history_member = tarfile.TarInfo("logs/pipeline_command_history.txt")
        history_member.size = len(history_payload)
        tar.addfile(history_member, io.BytesIO(history_payload))
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any(
        "server return package pipeline_command_history.txt entries do not match"
        in err
        for err in errors
    )
    assert any(
        "server return package pipeline_command_history.txt records a dry-run"
        in err
        for err in errors
    )


def test_server_return_package_rejects_required_dir_member_as_file(tmp_path):
    import io
    import tarfile

    check = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_required_dir_as_file",
    )
    package = tmp_path / "return.tar.gz"
    dir_path = "paper/figdata"
    with tarfile.open(package, "w:gz") as tar:
        payload = b"this path must be a directory\n"
        member = tarfile.TarInfo(dir_path)
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    package.with_name(f"{package.name}.sha256").write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )

    errors = check.check_return_tarball(package)

    assert any("required path is not a directory" in err for err in errors)
    assert any(dir_path in err for err in errors)


def test_server_return_package_optional_tarball_group_is_reported(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/check_server_return_package.py",
        "check_server_return_package_optional_tarball",
    )
    monkeypatch.setattr(mod, "_core_full_sweep_missing", lambda **kwargs: True)
    monkeypatch.setattr(mod, "check_full_sweep_gate", lambda results_dir: ["blocked"])
    package = tmp_path / "missing.tar.gz"

    groups = mod.check_server_return_package(
        results_root=tmp_path / "results",
        results_dir=tmp_path / "results" / "full_benchmark",
        paper_dir=tmp_path / "paper",
        logs_dir=tmp_path / "logs",
        package_path=package,
    )

    assert "package" in groups
    assert any("missing or empty" in err for err in groups["package"])


def test_create_server_return_package_rejects_self_including_output_paths(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_self_include",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    output = tmp_path / "paper" / "figures" / "return.tar.gz"

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)

    assert excinfo.value.code == 2
    assert not output.exists()


def test_create_server_return_package_rejects_non_tar_gz_output_suffix(tmp_path):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_bad_suffix",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    output = tmp_path / "benchmark" / "results" / "return_package.zip"

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)

    assert excinfo.value.code == 2
    assert not output.exists()
    assert not output.with_name(f"{output.name}.sha256").exists()


def test_create_server_return_package_rejects_packaged_artifact_output_even_overwrite(
    tmp_path,
):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_overwrite_artifact",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    results_csv = tmp_path / "benchmark" / "results" / "full_benchmark" / "results.csv"
    original = results_csv.read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, results_csv, overwrite=True)

    assert excinfo.value.code == 2
    assert results_csv.read_text(encoding="utf-8") == original


def test_create_server_return_package_refuses_existing_output_without_overwrite(
    tmp_path, monkeypatch
):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_existing",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    output = tmp_path / "return.tar.gz"
    output.write_bytes(b"existing")
    monkeypatch.delenv("OVERWRITE_RETURN_PACKAGE", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        mod.create_package(tmp_path, output)

    assert excinfo.value.code == 2
    assert output.read_bytes() == b"existing"

    package = mod.create_package(tmp_path, output, overwrite=True)
    assert package == output.resolve()
    assert output.read_bytes() != b"existing"


def test_create_server_return_package_preserves_existing_output_on_write_failure(
    tmp_path, monkeypatch
):
    mod = _load_module(
        "scripts/create_server_return_package.py",
        "create_server_return_package_atomic_failure",
    )
    _write_minimal_return_package_artifacts(tmp_path, mod)
    output = tmp_path / "return.tar.gz"
    sidecar = output.with_name(f"{output.name}.sha256")
    output.write_bytes(b"previous-package")
    sidecar.write_text("previous-digest  return.tar.gz\n", encoding="utf-8")
    original_open = mod.tarfile.open

    def fail_open(*args, **kwargs):
        if len(args) >= 2 and args[1] == "w:gz":
            raise RuntimeError("simulated tar writer crash")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(mod.tarfile, "open", fail_open)

    with pytest.raises(RuntimeError, match="simulated tar writer crash"):
        mod.create_package(tmp_path, output, overwrite=True)

    assert output.read_bytes() == b"previous-package"
    assert sidecar.read_text(encoding="utf-8") == "previous-digest  return.tar.gz\n"
    assert not list(output.parent.glob(".return.tar.gz.tmp.*"))


def test_scripts_readme_keeps_local_launcher_out_of_headline_path():
    text = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    assert "Nature CS headline" in text
    assert "designated Linux" in text
    assert "server, not by a local review workstation launcher" in text
    assert "Legacy constrained Windows Stage-2 launcher for diagnostics only" in text
    assert "not the Nature CS headline run path" in text
    assert "Windows 16 GB headline sweep" not in text


def test_server_entrypoint_docs_prefer_pipeline_for_formal_archive():
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    benchmark_readme = (ROOT / "benchmark" / "README.md").read_text(encoding="utf-8")
    root_readme_zh = (ROOT / "README_zh.md").read_text(encoding="utf-8")
    paper_readme = (ROOT / "paper" / "README.md").read_text(encoding="utf-8")
    reproducibility = (ROOT / "docs" / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    faq = (ROOT / "docs" / "FAQ.md").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")
    project_status = (ROOT / "docs" / "PROJECT_STATUS.md").read_text(encoding="utf-8")
    setup_remote = (ROOT / "scripts" / "setup_remote.sh").read_text(encoding="utf-8")
    run_pipeline_sh = (ROOT / "scripts" / "run_pipeline.sh").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    assert "python scripts/check_server_launch_ready.py" in root_readme
    assert "python scripts/check_server_launch_ready.py --require-clean-sweep-dir" in root_readme
    assert "python scripts/check_local_pre_server_ready.py" in root_readme
    assert "bash scripts/run_pipeline.sh" in root_readme
    assert "bash scripts/create_server_return_package.sh" in root_readme
    assert "formal headline entry" in root_readme
    assert "Low-level Stage 2 restart" in root_readme

    assert "formal Nature CS archives should enter through" in benchmark_readme
    assert "python scripts/check_local_pre_server_ready.py" in benchmark_readme
    assert "python scripts/check_server_launch_ready.py" in benchmark_readme
    assert "python scripts/check_server_launch_ready.py --require-clean-sweep-dir" in benchmark_readme
    assert "bash scripts/run_pipeline.sh" in benchmark_readme
    assert "bash scripts/create_server_return_package.sh" in benchmark_readme

    assert "python scripts/check_local_pre_server_ready.py" in root_readme_zh
    assert "python scripts/check_server_launch_ready.py" in root_readme_zh
    assert "python scripts/check_server_launch_ready.py --require-clean-sweep-dir" in root_readme_zh
    assert "bash scripts/run_pipeline.sh" in root_readme_zh
    assert "bash scripts/create_server_return_package.sh" in root_readme_zh
    assert "8 核约 4 小时" not in root_readme_zh

    assert "python scripts/check_server_launch_ready.py --require-clean-sweep-dir" in paper_readme
    assert "bash scripts/run_pipeline.sh" in paper_readme
    assert "bash scripts/create_server_return_package.sh" in paper_readme
    assert "python scripts/run_pipeline.py --profile full --confirm-full-run" not in paper_readme

    assert "本机交接前" in project_status
    assert "python scripts/check_local_pre_server_ready.py" in project_status

    assert "| Linux ≥ 32 c / ≥ 64 GB | `bash scripts/run_pipeline.sh`" in reproducibility
    assert "default 56 GB; override explicitly with `MEM_CAP_GB=...`" in reproducibility
    assert "N_JOBS=24 BO_N_JOBS=4 MEM_CAP_GB=96 bash scripts/run_pipeline.sh" in reproducibility
    assert "auto-detect, ≤ 56 GB" not in reproducibility
    assert "no formal full sweep; use smoke/preflight only" in reproducibility
    assert "Windows workstations are for smoke, unit, physics, and" in reproducibility
    assert "not the preferred way to produce the complete paper" in reproducibility
    reproducibility_flat = " ".join(reproducibility.split())
    assert "archive because it does not run the Stage 2b-10 gates" in reproducibility_flat

    assert "| Windows workstation | 16 GB | 8+ | n/a | Smoke, unit, physics, dry-run/preflight checks only |" in faq
    assert "not a local workstation job" in faq
    assert "bash scripts/run_pipeline.sh" in faq
    assert "Full sweep on Windows 16 GB" not in faq

    assert "the formal archive entry points are:" in handoff
    assert "runs this strict variant by default" in handoff
    assert "DOE_BENCHMARK_ALLOW_STALE_DERIVED=1" in handoff

    assert "DOE_BENCHMARK_ALLOW_STALE_DERIVED" in run_pipeline_sh
    assert "--require-clean-sweep-dir" in run_pipeline_sh
    assert "valid partial results.csv is still treated as a resume candidate" in run_pipeline_sh

    assert "repeats strict clean-sweep preflight" in scripts_readme


def test_statistical_analysis_plan_marks_low_level_stage2_as_diagnostic():
    sap = (ROOT / "docs" / "STATISTICAL_ANALYSIS_PLAN.md").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")
    setup_remote = (ROOT / "scripts" / "setup_remote.sh").read_text(encoding="utf-8")

    assert "Formal server launch path" in sap
    assert "bash scripts/run_pipeline.sh" in sap
    assert "low-level command below is only for a" in sap
    assert "reviewed Stage 2 resume/diagnostic" in sap
    assert "does not create the complete paper" in sap
    assert "reserved for a reviewed low-level sweep resume or" in handoff
    assert "does not build the complete paper archive by itself" in handoff

    assert "mkdir -p \"${SMOKE_OUT}\"" in setup_remote
    assert "Launch the formal full archive pipeline with:" in setup_remote
    assert "bash scripts/run_pipeline.sh" in setup_remote


def test_resume_contract_docs_reject_blank_protocol_keys():
    handoff = (ROOT / "docs" / "SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    assert "blank or missing `case_id`, `method`, `budget_mult`, or `repeat`" in handoff
    assert "blank protocol keys" in handoff
    assert "symlinked CSV paths" in handoff
    assert "blank protocol keys" in scripts_readme
    assert "symlinked CSV path" in scripts_readme


def test_experiment_audit_records_external_benchmark_launch_and_streaming_controls():
    audit = (ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md").read_text(encoding="utf-8")
    scripts_readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    normalized_scripts_readme = " ".join(scripts_readme.split())
    full_runner = (ROOT / "benchmark" / "scripts" / "run_full_benchmark.py").read_text(
        encoding="utf-8"
    )

    assert "Bayesmark" in audit
    assert "job-file" in audit
    assert "dry runs for cluster execution" in audit
    assert "function-call budget" in audit
    assert "Cluster-scale launch discipline" in audit
    assert "Streamed archive precedence" in audit
    assert "P2 web-validation audit `Evidence tier remains P2`" in audit
    assert "on-disk streamed `results.csv` is the canonical Stage-2 artifact" in audit
    assert "The streamed on-disk `results.csv` is the canonical Stage-2 artifact" in normalized_scripts_readme
    assert "Streamed CSV is the canonical artifact" in full_runner
    assert "overwriting from `result.df` would lose history" in full_runner


def test_experiment_audit_records_neurips_datasets_benchmarks_resource_controls():
    audit = (ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md").read_text(encoding="utf-8")

    assert "NeurIPS 2026 Evaluations & Datasets guidance" in audit
    assert "https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets" in audit
    assert "https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines" in audit
    assert "https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting" in audit
    for review_dimension in (
        "code release is required at submission",
        "benchmark suite",
        "evaluation environment",
        "software tool",
        "accessibility",
        "transparency",
        "usability",
        "reproducibility",
        "benchmark design",
        "benchmark analysis",
        "auditing",
        "stress-testing",
        "limitations",
        "failure modes",
        "machine-readable metadata",
    ):
        assert review_dimension in audit
    for control in (
        "public case API",
        "case registry",
        "evidence-tier ledger",
        "preregistered protocol",
        "smoke/full pipeline split",
        "row-level run records",
        "runtime-health telemetry",
        "return-package gate",
        "generated figdata/numbers/SI",
        "P2 limitations and server failures visible",
    ):
        assert control in audit
    for proof_artifact in (
        "`docs/CASE_VALIDATION_SUMMARY.md`",
        "`docs/CASE_WEB_VALIDATION_AUDIT.md`",
        "`docs/SERVER_RUN_HANDOFF.md`",
        "`docs/EXPERIMENT_BENCHMARK_AUDIT.md`",
        "`scripts/check_server_return_package.py`",
    ):
        assert proof_artifact in audit
    assert "Benchmark papers should make the resource inspectable and usable by others" in audit
    assert "without overstating validation or hiding server failures" in audit


def test_experiment_audit_maps_external_benchmark_practices_to_project_controls():
    audit = (ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md").read_text(encoding="utf-8")

    assert "External Benchmark Practice Mapping" in audit
    for anchor in (
        "OpenML benchmark suites and run records",
        "COCO/BBOB",
        "HPOBench and YAHPO Gym",
        "Bayesmark",
        "MLCommons/MLPerf-style audit discipline",
        "ACM artifact review and Nature/FAIR expectations",
        "NeurIPS 2026 Evaluations & Datasets-style benchmark/reusable-artifact review",
    ):
        assert anchor in audit
    for control in (
        "`run_config.json`",
        "`RUN_MANIFEST.json`",
        "`full_benchmark/results.csv`",
        "`protocol_digest`",
        "reusable run record",
        "flows/parameter settings",
        "evaluation metrics",
        "runtime metadata",
        "server return-package checks",
        "SHA-256 sidecars",
    ):
        assert control in audit
    assert "does not claim that" in audit
    assert "fixed function-evaluation budgets" in audit
    assert "independent post-processing" in audit
    assert "uncontrolled drift a gate failure" in audit


def test_experiment_audit_records_current_external_source_verification_matrix():
    audit = (ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md").read_text(encoding="utf-8")

    assert "Current check date: 2026-06-09" in audit
    assert "External Source Verification Matrix" in audit
    assert "Source rechecked on 2026-06-09" in audit
    assert "Nature Computational Science Fit and Reporting Controls" in audit
    for source_anchor in (
        "Nature Portfolio reporting standards",
        "Nature Computational Science submission guidance",
        "development and use of computational techniques and mathematical models",
        "GO FAIR FAIR Principles",
        "ACM artifact review badging",
        "NeurIPS 2026 Evaluations & Datasets benchmark/reusable-artifact guidance",
        "OpenML benchmark suites and run records",
        "MLCommons/MLPerf benchmark practice",
        "COCO/BBOB, HPOBench, YAHPO Gym, and Bayesmark",
    ):
        assert source_anchor in audit
    for project_control in (
        "`check_submission_ready.py` lineage checks",
        "`check_paper_draft_ready.py`",
        "`figdata_manifest.json`",
        "Linux launch preflight",
        "row-level `results.csv`",
        "runtime trace",
        "run-config binding",
        "Stage-2b audit",
        "Code Availability section",
        "permanent identifiers",
        "reviewer rerun path",
    ):
        assert project_control in audit
    assert "cross-domain engineering/physics DOE resource" in audit
    assert "It must not be framed as 34 externally validated simulators" in audit
    assert "standardized train-test splits" in audit
    assert "what can change and how performance is measured" in audit
    assert "Supplementary Information should carry material needed" in audit
    assert "machine-checkable" in audit
    assert "What is a reusable run record?" in audit
    assert "case/method source hashes" in audit
    assert "without uploading to OpenML" in audit
    assert "Independently re-evaluable runs" in audit
    assert "without access to the live server or hidden notebooks" in audit
    assert "task, method, seed, metric, status, failure, runtime, environment, and source-provenance fields" in audit
    assert "python scripts/check_server_launch_ready.py --require-clean-sweep-dir" in audit
    assert "bash scripts/run_pipeline.sh" in audit
    assert "strict clean-sweep preflight" in audit
    assert "DOE_BENCHMARK_ALLOW_STALE_DERIVED=1" in audit
    assert "dead while the headline grid is missing" in audit
    assert "watch mode prints that warning and continues" in audit
    for cluster_failure_marker in (
        "TERM_MEMLIMIT",
        "TERM_RUNLIMIT",
        "h_vmem",
        "h_rt",
    ):
        assert cluster_failure_marker in audit


def test_experiment_audit_records_reviewer_prelaunch_acceptance_checklist():
    audit = (ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md").read_text(encoding="utf-8")

    assert "Reviewer-Facing Pre-Launch Acceptance Checklist" in audit
    assert "Server-launch proof obligation" in audit
    for principle in (
        "Nature Portfolio availability policy",
        "ACM artifact-evaluation criteria",
        "NeurIPS 2026 Evaluations & Datasets guidance",
        "FAIR principles",
        "OpenML benchmark-suite practice",
        "COCO/BBOB optimizer-benchmark practice",
        "MLCommons/MLPerf audit discipline",
        "Bayesmark/HPOBench/YAHPO-style scalable benchmark practice",
        "Physics-benchmark honesty",
    ):
        assert principle in audit
    for proof in (
        "prose-only readiness is insufficient",
        "wall time/RSS are diagnostics rather than hidden budgets",
        "run-config/report mismatch",
        "resume only valid partial archives",
        "public sources used as boundary/context evidence",
    ):
        assert proof in audit
    for gate in (
        "`scripts/check_local_pre_server_ready.py`",
        "`scripts/check_server_launch_ready.py`",
        "`scripts/check_sweep_runtime_health.py`",
        "`scripts/check_server_return_package.py`",
        "`scripts/check_submission_ready.py`",
        "`docs/CASE_WEB_VALIDATION_AUDIT.md`",
    ):
        assert gate in audit


def test_experiment_audit_reviewer_smoke_health_command_matches_current_cli():
    audit = (ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md").read_text(encoding="utf-8")
    smoke_row = next(
        line
        for line in audit.splitlines()
        if line.startswith("| Smoke archive health check |")
    )

    assert (
        "python scripts/check_sweep_runtime_health.py "
        "benchmark/results/_smoke/results.csv --allow-missing "
        "--expected-n-test 200 --seed-strategy case_repeat --seed-start 1000"
    ) in smoke_row
    for unsupported_flag in (
        "--results-csv",
        "--expected-cases",
        "--expected-methods",
        "--expected-budgets",
        "--expected-repeats",
    ):
        assert unsupported_flag not in smoke_row
    assert "sweep_integrity_report.md" in smoke_row
    assert "case/method/budget/repeat grid" in smoke_row


def test_experiment_audit_records_server_failure_mode_drills():
    audit = (ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit.split())

    assert "Server Failure-Mode Drill Matrix" in audit
    assert "2026-06-09 matrix" in audit
    assert "fail loudly" in audit
    assert "resume conservatively" in audit
    assert "reviewer to diagnose the run without access to the live server" in normalized
    for failure_mode in (
        "Method cell timeout or watchdog expiry",
        "CPU/GPU/cgroup OOM or killed worker",
        "Scheduler preemption, cancellation, node failure, dependency-chain failure, or launch failure",
        "Storage, quota, stale filesystem, or temp/scratch exhaustion",
        "Crash-truncated, hand-edited, stale, or header-only `results.csv`",
        "Partial run appears alive but stops making progress",
        "Post-return artifact drift or mixed analysis archives",
        "Paper claims generated from stale or partial data",
    ):
        assert failure_mode in audit
    for evidence in (
        "`status=timeout`",
        "`failure_category=timeout`",
        "`rss_gb_peak`",
        "PREEMPTED",
        "DUE TO PREEMPTION",
        "NODE_FAIL",
        "ReqNodeNotAvail",
        "JobLaunchFailure",
        "QOSGrpCpuLimit",
        "LAUNCH_FAILED",
        "BOOT_FAIL",
        "returncode=124",
        "ExitCode=0:9",
        "ENOSPC",
        "EDQUOT",
        "ESTALE",
        "database or disk is full",
        "Transport endpoint is not connected",
        "create/write/flush/fsync/delete probe",
        "raw row field counts must match the header",
        "`--fail-on-stale --stale-after-min 45`",
        "SOTA/MF robustness directories must not contaminate",
        "`RESULT_PENDING` remains",
    ):
        assert evidence in audit
    for gate in (
        "`BenchmarkRunner --skip-existing`",
        "`check_sweep_runtime_health.py --mem-cap-gb ...`",
        "`check_sweep_runtime_health.py --disk-warning-gb ...`",
        "`create_server_return_package.py`",
        "`check_server_return_package.py`",
        "`check_submission_ready.py paper`",
    ):
        assert gate in audit


def test_experiment_audit_records_fair_nature_data_package_controls():
    audit = (ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md").read_text(encoding="utf-8")

    assert "Nature Research data-availability guidance" in audit
    assert "FAIR Guiding Principles" in audit
    assert "artifact-review-and-badging-current" in audit
    assert "software, scripts, input data, raw experimental data, or analysis scripts" in audit
    assert "personal web page" in audit
    assert "findable, accessible, interoperable, and reusable" in audit
    assert "FAIR/Nature data package" in audit
    assert "Zenodo/DOI deposition remains a final post-run requirement" in audit
    for artifact in (
        "`results.csv`",
        "`RUN_MANIFEST.json`",
        "figdata CSVs",
        "paper numbers",
        "SI tables",
        "figure manifests",
        "command history",
        "source fingerprints",
        "sha256 sidecars",
    ):
        assert artifact in audit
    assert "available on request" in audit
    assert "immutable DOI" in audit


def test_sweep_integrity_audit_accepts_complete_grid(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_ok")
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool

    csv_path = tmp_path / "results.csv"
    case_ids = ["cantilever_beam_5d", "pressure_vessel_4d"]
    cases = {case_id: db.get_case(case_id) for case_id in case_ids}
    method_objects = {m.name: m for m in default_method_pool()[:2]}
    rows = []
    for case in case_ids:
        meta = cases[case].meta
        fingerprint = case_source_fingerprint(cases[case])
        for method_name, method in method_objects.items():
            for budget in [5, 10]:
                for repeat in [0, 1]:
                    rows.append({
                        "case_id": case,
                        **fingerprint,
                        "domain": meta.domain,
                        "dim": meta.dim_input,
                        "n_output": meta.dim_output,
                        "method": method_name,
                        "method_family": method.family,
                        **method_source_fingerprint(method),
                        "budget_mult": budget,
                        "n_train": budget * meta.dim_input,
                        "n_test": 1000,
                        "test_seed": 0,
                        "repeat": repeat,
                        "seed": 1000 + repeat,
                        "status": "ok",
                        "failure_category": "ok",
                        "t_total_s": 1.0,
                        "error": "",
                        "nrmse": 0.1,
                        **_runtime_trace_fields(),
                    })
    import pandas as pd
    _results_frame(rows, seed_strategy="legacy").to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=case_ids,
        expected_methods=list(method_objects),
        expected_method_families={name: method.family for name, method in method_objects.items()},
        expected_budgets=[5, 10],
        expected_repeats=2,
        expected_n_test=1000,
        expected_test_seed=0,
        seed_strategy="legacy",
        seed_start=1000,
        min_repeats_per_cell=2,
        min_ok_fraction=0.95,
    )

    assert rc == 0
    assert "PASS" in report


def test_sweep_integrity_audit_rejects_missing_protocol_key_values(tmp_path):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_missing_protocol_keys",
    )
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case = db.get_case("cantilever_beam_5d")
    method = default_method_pool()[0]
    _results_frame([{
        "case_id": "",
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "method": method.name,
        "method_family": method.family,
        **method_source_fingerprint(method),
        "budget_mult": 5,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "repeat": 0,
        "seed": 1000,
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        "nrmse": 0.1,
        **_runtime_trace_fields(),
    }]).to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["cantilever_beam_5d"],
        expected_methods=[method.name],
        expected_method_families={method.name: method.family},
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
    )

    assert rc == 1
    assert "protocol key values must be non-empty" in report
    assert "case_id=1" in report


def test_sweep_integrity_report_write_is_atomic_on_replace_failure(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_atomic_report",
    )
    output = tmp_path / "sweep_integrity_report.md"
    output.write_text("old report\n", encoding="utf-8")

    def fail_move(src, dst):
        raise RuntimeError(f"cannot replace {dst}")

    monkeypatch.setattr(mod.shutil, "move", fail_move)

    with pytest.raises(RuntimeError, match="cannot replace"):
        mod.write_report_atomic("new report\n", output)

    assert output.read_text(encoding="utf-8") == "old report\n"
    assert not list(tmp_path.glob(".sweep_integrity_report.md.tmp.*"))


def test_sweep_integrity_audit_rejects_stale_case_source_hashes(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_case_source")
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool

    csv_path = tmp_path / "results.csv"
    case = db.get_case("cantilever_beam_5d")
    fingerprint = case_source_fingerprint(case)
    method = default_method_pool()[0]
    import pandas as pd

    _results_frame([{
        "case_id": "cantilever_beam_5d",
        "case_source_path": fingerprint["case_source_path"],
        "case_source_sha256": "0" * 64,
        "domain": case.meta.domain,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "method": method.name,
        "method_family": method.family,
        **method_source_fingerprint(method),
        "budget_mult": 5,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "repeat": 0,
        "seed": 1000,
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        "nrmse": 0.1,
    }]).to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["cantilever_beam_5d"],
        expected_methods=[method.name],
        expected_method_families={method.name: method.family},
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        seed_strategy="legacy",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "case_source_sha256 mismatch for cantilever_beam_5d" in report
    assert "Case Source Fingerprint Mismatches" in report


def test_sweep_integrity_audit_rejects_case_source_path_mismatches(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_case_source_path")
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool

    csv_path = tmp_path / "results.csv"
    case = db.get_case("cantilever_beam_5d")
    fingerprint = case_source_fingerprint(case)
    method = default_method_pool()[0]
    import pandas as pd

    _results_frame([{
        "case_id": "cantilever_beam_5d",
        "case_source_path": "doe_testbed/cases/structures/not_the_case.py",
        "case_source_sha256": fingerprint["case_source_sha256"],
        "domain": case.meta.domain,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "method": method.name,
        "method_family": method.family,
        **method_source_fingerprint(method),
        "budget_mult": 5,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "repeat": 0,
        "seed": 1000,
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        "nrmse": 0.1,
    }]).to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["cantilever_beam_5d"],
        expected_methods=[method.name],
        expected_method_families={method.name: method.family},
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        seed_strategy="legacy",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "case_source_path mismatch for cantilever_beam_5d" in report
    assert "not_the_case.py" in report


def test_sweep_integrity_audit_rejects_method_source_hash_mismatches(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_method_source")
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool

    csv_path = tmp_path / "results.csv"
    case = db.get_case("cantilever_beam_5d")
    method = default_method_pool()[0]
    method_fingerprint = method_source_fingerprint(method)
    import pandas as pd

    _results_frame([{
        "case_id": "cantilever_beam_5d",
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "method": method.name,
        "method_family": method.family,
        "method_source_path": method_fingerprint["method_source_path"],
        "method_source_sha256": "0" * 64,
        "budget_mult": 5,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "repeat": 0,
        "seed": 1000,
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        "nrmse": 0.1,
    }]).to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["cantilever_beam_5d"],
        expected_methods=[method.name],
        expected_method_families={method.name: method.family},
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        seed_strategy="legacy",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert f"method_source_sha256 mismatch for {method.name}" in report
    assert "Method Source Fingerprint Mismatches" in report


def test_sweep_integrity_audit_accepts_absolute_source_paths_with_matching_hashes(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_absolute_source_paths")
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool

    csv_path = tmp_path / "results.csv"
    case = db.get_case("cantilever_beam_5d")
    case_fingerprint = dict(case_source_fingerprint(case))
    method = default_method_pool()[0]
    method_fingerprint = dict(method_source_fingerprint(method))
    case_fingerprint["case_source_path"] = (ROOT / case_fingerprint["case_source_path"]).as_posix()
    method_fingerprint["method_source_path"] = (ROOT / method_fingerprint["method_source_path"]).as_posix()
    import pandas as pd

    _results_frame([{
        "case_id": "cantilever_beam_5d",
        **case_fingerprint,
        "domain": case.meta.domain,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "method": method.name,
        "method_family": method.family,
        **method_fingerprint,
        "budget_mult": 5,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "repeat": 0,
        "seed": 1000,
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        "nrmse": 0.1,
        **_runtime_trace_fields(),
    }], seed_strategy="legacy").to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["cantilever_beam_5d"],
        expected_methods=[method.name],
        expected_method_families={method.name: method.family},
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        seed_strategy="legacy",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 0
    assert "source_path mismatch" not in report


def test_sweep_integrity_audit_rejects_invalid_runtime_trace_values(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_runtime_trace")
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case = db.get_case("cantilever_beam_5d")
    method = default_method_pool()[0]
    _results_frame([{
        "case_id": "cantilever_beam_5d",
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "method": method.name,
        "method_family": method.family,
        **method_source_fingerprint(method),
        "budget_mult": 5,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "repeat": 0,
        "seed": 1000,
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        "nrmse": 0.1,
        "worker_host": "",
        "worker_pid": 0,
        "cell_started_utc": "bad",
        "cell_finished_utc": "bad",
        "rss_gb_start": -1,
        "rss_gb_end": "bad",
        "rss_gb_peak": "bad",
    }]).to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["cantilever_beam_5d"],
        expected_methods=[method.name],
        expected_method_families={method.name: method.family},
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        seed_strategy="legacy",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "worker_host" in report
    assert "worker_pid" in report
    assert "cell_started_utc" in report
    assert "cell_finished_utc" in report
    assert "rss_gb_start" in report
    assert "rss_gb_end" in report
    assert "rss_gb_peak" in report


def test_sweep_integrity_audit_rejects_tampered_protocol_digest(tmp_path):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_bad_protocol_digest",
    )
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool

    csv_path = tmp_path / "results.csv"
    case_id = "cantilever_beam_5d"
    case = db.get_case(case_id)
    method = default_method_pool()[0]
    df = _results_frame([{
        "case_id": case_id,
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "method": method.name,
        "method_family": method.family,
        **method_source_fingerprint(method),
        "budget_mult": 5,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "repeat": 0,
        "seed": mod._derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        "nrmse": 0.1,
        **_runtime_trace_fields(),
    }])
    df.loc[0, "protocol_digest"] = "0" * 64
    df.to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=[case_id],
        expected_methods=[method.name],
        expected_method_families={method.name: method.family},
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
    )

    assert rc == 1
    assert "protocol_digest must match" in report
    assert "Protocol Digest Mismatches" in report


def test_sweep_integrity_audit_rejects_rss_peak_below_start_or_end(tmp_path):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_bad_rss_peak_order",
    )
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case = db.get_case("cantilever_beam_5d")
    method = default_method_pool()[0]
    row = {
        "case_id": "cantilever_beam_5d",
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "method": method.name,
        "method_family": method.family,
        **method_source_fingerprint(method),
        "budget_mult": 5,
        "n_train": 5 * case.meta.dim_input,
        "n_test": 1000,
        "test_seed": 0,
        "repeat": 0,
        "seed": 1000,
        "status": "ok",
        "t_total_s": 1.0,
        "error": "",
        "nrmse": 0.1,
        **_runtime_trace_fields(),
        "rss_gb_peak": 1.0,
    }
    _results_frame([row]).to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["cantilever_beam_5d"],
        expected_methods=[method.name],
        expected_method_families={method.name: method.family},
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        seed_strategy="legacy",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "rss_gb_peak >= rss_gb_start" in report


def test_sweep_integrity_audit_rejects_missing_case_source_fingerprint_columns(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_missing_case_source")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,domain,dim,n_output,method,method_family,budget_mult,n_train,n_test,test_seed,repeat,seed,status,t_total_s,error,nrmse\n"
        "cantilever_beam_5d,structures,5,1,m1,test_family,5,25,1000,0,0,1000,ok,1.0,,0.1\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["cantilever_beam_5d"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "missing required columns" in report
    assert "case_source_path" in report
    assert "case_source_sha256" in report


def test_sweep_integrity_audit_rejects_missing_cells_and_low_ok(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n"
        "case_a,m1,5,1,timeout,2.0,timeout reason,\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1", "m2"],
        expected_budgets=[5],
        expected_repeats=2,
        min_repeats_per_cell=2,
        min_ok_fraction=0.95,
        max_timeout_fraction=0.05,
    )

    assert rc == 1
    assert "OK fraction" in report
    assert "missing methods" in report
    assert "timeout fraction" in report


def test_sweep_integrity_audit_reports_missing_core_columns_without_traceback(tmp_path):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_missing_core_cols",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "method,budget_mult,t_total_s,nrmse\n"
        "m1,5,1.0,0.1\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        seed_strategy="case_repeat",
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
    )

    assert rc == 1
    assert "missing required columns" in report
    assert "case_id" in report
    assert "status" in report
    assert "seed" in report
    assert "Traceback" not in report


def test_sweep_integrity_audit_reports_header_only_results_csv(tmp_path):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_header_only",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(",".join(sorted(mod.REQUIRED_COLUMNS)) + "\n", encoding="utf-8")

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
    )

    assert rc == 1
    assert "results.csv has no data rows" in report
    assert "Traceback" not in report


def test_sweep_integrity_audit_rejects_unexpected_extra_methods(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_extra_method")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n"
        "case_a,m_extra,5,0,ok,1.2,,0.2\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "unexpected extra methods" in report


def test_sweep_integrity_audit_treats_row_count_drift_as_error(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_row_count")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=2,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "### Errors" in report
    assert "row count differs from declared full grid" in report


def test_sweep_integrity_audit_rejects_duplicate_repeat_cells(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_duplicate")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n"
        "case_a,m1,5,0,ok,1.1,,0.2\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "duplicate (case, method, budget, repeat) cells" in report
    assert "Duplicate Cells" in report


def test_sweep_integrity_audit_rejects_out_of_range_repeat_ids(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_repeat")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n"
        "case_a,m1,5,2,ok,1.0,,0.2\n"
        "case_a,m1,5,-1,ok,1.0,,0.3\n"
        "case_a,m1,5,0.5,ok,1.0,,0.4\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=2,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "repeat values must be integer IDs in [0, 1]" in report
    assert "Invalid Repeat IDs" in report


def test_sweep_integrity_audit_rejects_invalid_budget_values(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_budget")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n"
        "case_a,m1,5.5,1,ok,1.0,,0.2\n"
        "case_a,m1,20,0,ok,1.0,,0.3\n"
        "case_a,m1,not-a-budget,1,ok,1.0,,0.4\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5, 10],
        expected_repeats=2,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "budget_mult values must be integer members of [5, 10]" in report
    assert "Invalid Budgets" in report


def test_sweep_integrity_audit_rejects_unknown_status_values(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_status")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n"
        "case_a,m1,5,1,success,1.0,,0.2\n"
        "case_a,m1,5,2,,1.0,,0.3\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=3,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
    )

    assert rc == 1
    assert "status values must be one of ['failed', 'ok', 'timeout']" in report
    assert "Invalid Status Values" in report


def test_sweep_integrity_audit_rejects_n_train_budget_dim_mismatch(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_n_train")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,dim,n_train,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,4,20,0,ok,1.0,,0.1\n"
        "case_a,m1,5,4,21,1,ok,1.0,,0.2\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=2,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "n_train must equal budget_mult * dim" in report
    assert "Invalid Training Sizes" in report


def test_sweep_integrity_audit_rejects_case_metadata_mismatches(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_case_meta")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,domain,dim,n_output,method,budget_mult,n_train,repeat,status,t_total_s,error,nrmse\n"
        "cantilever_beam_5d,aerodynamics,999,99,m1,5,4995,0,ok,1.0,,0.1\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["cantilever_beam_5d"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "case metadata columns do not match the current registry" in report
    assert "Case Metadata Mismatches" in report
    assert "cantilever_beam_5d" in report
    assert "domain" in report
    assert "dim" in report
    assert "n_output" in report


def test_sweep_integrity_audit_rejects_method_family_mismatches(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_method_family")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,method_family,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,wrong_family,5,0,ok,1.0,,0.1\n"
        "case_a,m2,expected_family,5,0,ok,1.0,,0.2\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1", "m2"],
        expected_method_families={"m1": "expected_family", "m2": "expected_family"},
        expected_budgets=[5],
        expected_repeats=1,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "method_family values do not match the declared method pool" in report
    assert "Method Family Mismatches" in report
    assert "m1" in report
    assert "wrong_family" in report


def test_sweep_integrity_audit_rejects_missing_or_mismatched_n_test(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_n_test")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n",
        encoding="utf-8",
    )

    rc_missing, report_missing = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc_missing == 1
    assert "results.csv has no n_test column" in report_missing

    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse,n_test\n"
        "case_a,m1,5,0,ok,1.0,,0.1,1000\n"
        "case_a,m1,5,1,ok,1.0,,0.2,999\n"
        "case_a,m1,5,2,ok,1.0,,0.3,not-an-int\n",
        encoding="utf-8",
    )

    rc_bad, report_bad = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=3,
        expected_n_test=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc_bad == 1
    assert "n_test must equal declared expected_n_test=1000" in report_bad
    assert "found 2 invalid rows" in report_bad
    assert "Invalid Test Set Sizes" in report_bad


def test_sweep_integrity_audit_rejects_missing_or_mismatched_test_seed(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_test_seed")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n",
        encoding="utf-8",
    )

    rc_missing, report_missing = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        expected_test_seed=0,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc_missing == 1
    assert "results.csv has no test_seed column" in report_missing

    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse,test_seed\n"
        "case_a,m1,5,0,ok,1.0,,0.1,0\n"
        "case_a,m1,5,1,ok,1.0,,0.2,1\n"
        "case_a,m1,5,2,ok,1.0,,0.3,not-an-int\n",
        encoding="utf-8",
    )

    rc_bad, report_bad = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=3,
        expected_test_seed=0,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc_bad == 1
    assert "test_seed must equal declared expected_test_seed=0" in report_bad
    assert "found 2 invalid rows" in report_bad
    assert "Invalid Test Seeds" in report_bad


def test_sweep_integrity_audit_validates_case_repeat_seeds(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_seed")
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.methods import default_method_pool
    import pandas as pd

    csv_path = tmp_path / "results.csv"
    case_id = "cantilever_beam_5d"
    case = db.get_case(case_id)
    fingerprint = case_source_fingerprint(case)
    method = default_method_pool()[0]
    method_fingerprint = method_source_fingerprint(method)
    seed0 = mod._derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000)
    seed1 = mod._derive_seed(case_id, 1, strategy="case_repeat", seed_start=1000)
    _results_frame([
        {
            "case_id": case_id,
            **fingerprint,
            "domain": case.meta.domain,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "method": method.name,
            "method_family": method.family,
            **method_fingerprint,
            "budget_mult": 5,
            "n_train": 5 * case.meta.dim_input,
            "n_test": 1000,
            "test_seed": 0,
            "repeat": 0,
            "seed": seed0,
            "status": "ok",
            "failure_category": "ok",
            "t_total_s": 1.0,
            "error": "",
            "nrmse": 0.1,
            **_runtime_trace_fields(),
        },
        {
            "case_id": case_id,
            **fingerprint,
            "domain": case.meta.domain,
            "dim": case.meta.dim_input,
            "n_output": case.meta.dim_output,
            "method": method.name,
            "method_family": method.family,
            **method_fingerprint,
            "budget_mult": 5,
            "n_train": 5 * case.meta.dim_input,
            "n_test": 1000,
            "test_seed": 0,
            "repeat": 1,
            "seed": seed1,
            "status": "ok",
            "failure_category": "ok",
            "t_total_s": 1.2,
            "error": "",
            "nrmse": 0.2,
            **_runtime_trace_fields(),
        },
    ]).to_csv(csv_path, index=False)

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=[case_id],
        expected_methods=[method.name],
        expected_budgets=[5],
        expected_repeats=2,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method.name: method.family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 0
    assert "PASS" in report


def test_sweep_integrity_audit_rejects_bad_seed_values(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_seed")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,dim,n_train,repeat,seed,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,2,10,0,123,ok,1.0,,0.1\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "seed values do not match case_repeat seed derivation" in report
    assert "Invalid Seeds" in report


def test_sweep_integrity_audit_requires_seed_column_when_strategy_declared(tmp_path):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_missing_seed_column",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,dim,n_train,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,2,10,0,ok,1.0,,0.1\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "results.csv has no seed column" in report
    assert "paired-repeat reproducibility cannot be audited" in report


def test_sweep_integrity_audit_rejects_invalid_ok_nrmse_values(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_nrmse")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n"
        "case_a,m1,5,1,ok,1.0,,\n"
        "case_a,m1,5,2,ok,1.0,,inf\n"
        "case_a,m1,5,3,ok,1.0,,-0.2\n"
        "case_a,m1,5,4,timeout,1.0,timeout reason,\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=5,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
        max_timeout_fraction=1.0,
    )

    assert rc == 1
    assert "ok rows must have finite non-negative nrmse" in report
    assert "found 3 invalid rows" in report
    assert "Invalid NRMSE Values" in report


def test_sweep_integrity_audit_rejects_invalid_optional_metric_values(tmp_path):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_bad_optional_metrics",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse,rmse,mae,maxae,q95ae,r2\n"
        "case_a,m1,5,0,ok,1.0,,0.1,0.2,0.1,0.5,0.4,0.9\n"
        "case_a,m1,5,1,ok,1.0,,0.2,-0.2,0.1,0.5,0.4,0.8\n"
        "case_a,m1,5,2,ok,1.0,,0.3,0.2,not-a-number,0.5,0.4,0.7\n"
        "case_a,m1,5,3,ok,1.0,,0.4,0.2,0.1,inf,0.4,0.6\n"
        "case_a,m1,5,4,ok,1.0,,0.5,0.2,0.1,0.5,,0.5\n"
        "case_a,m1,5,5,ok,1.0,,0.6,0.2,0.1,0.5,0.4,not-finite\n"
        "case_a,m1,5,6,failed,1.0,failed reason,,nan,nan,nan,nan,nan\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=7,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
        max_timeout_fraction=1.0,
    )

    assert rc == 1
    assert "metric column 'rmse' must have finite non-negative values" in report
    assert "metric column 'mae' must have finite non-negative values" in report
    assert "metric column 'maxae' must have finite non-negative values" in report
    assert "metric column 'q95ae' must have finite non-negative values" in report
    assert "metric column 'r2' must have finite numeric values" in report
    assert "Invalid Optional Metric Values" in report


def test_sweep_integrity_audit_rejects_invalid_timing_values(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_bad_time")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n"
        "case_a,m1,5,1,timeout,,timeout reason,\n"
        "case_a,m1,5,2,failed,inf,failed reason,\n"
        "case_a,m1,5,3,ok,-0.5,,0.2\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=4,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
        max_timeout_fraction=1.0,
    )

    assert rc == 1
    assert "finite non-negative t_total_s" in report
    assert "found 3 invalid rows" in report
    assert "Invalid Timing Values" in report


def test_sweep_integrity_audit_rejects_runtime_time_order(tmp_path):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_bad_runtime_time_order",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse,"
        "worker_host,worker_pid,cell_started_utc,cell_finished_utc,"
        "rss_gb_start,rss_gb_end,rss_gb_peak\n"
        "case_a,m1,5,0,ok,1.0,,0.1,"
        "server-a,12345,2026-06-01T00:00:02+00:00,"
        "2026-06-01T00:00:01+00:00,1.0,1.1,1.2\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
        max_timeout_fraction=1.0,
    )

    assert rc == 1
    assert "cell_finished_utc must not be earlier than cell_started_utc" in report
    assert "Invalid Runtime Time Order" in report


def test_sweep_integrity_audit_rejects_runtime_duration_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_sweep_integrity.py",
        "audit_sweep_integrity_bad_runtime_duration",
    )
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse,"
        "worker_host,worker_pid,cell_started_utc,cell_finished_utc,"
        "rss_gb_start,rss_gb_end,rss_gb_peak\n"
        "case_a,m1,5,0,ok,100.0,,0.1,"
        "server-a,12345,2026-06-01T00:00:00+00:00,"
        "2026-06-01T00:00:01+00:00,1.0,1.1,1.2\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=1,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
        max_timeout_fraction=1.0,
    )

    assert rc == 1
    assert "t_total_s must agree with cell_finished_utc - cell_started_utc" in report
    assert "Invalid Runtime Duration Consistency" in report


def test_sweep_integrity_audit_rejects_missing_non_ok_error_reasons(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_missing_errors")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,,0.1\n"
        "case_a,m1,5,1,timeout,2.0,,\n"
        "case_a,m1,5,2,failed,0.2,   ,\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=3,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
        max_timeout_fraction=1.0,
    )

    assert rc == 1
    assert "timeout/failed rows must include a non-empty error reason" in report
    assert "found 2 invalid rows" in report
    assert "Missing Failure Reasons" in report


def test_sweep_integrity_audit_rejects_ok_rows_with_error_text(tmp_path):
    mod = _load_module("scripts/audit_sweep_integrity.py", "audit_sweep_integrity_ok_error_text")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status,t_total_s,error,nrmse\n"
        "case_a,m1,5,0,ok,1.0,RuntimeWarning leaked into success,0.1\n"
        "case_a,m1,5,1,ok,1.1,,0.2\n",
        encoding="utf-8",
    )

    rc, report = mod.audit_results(
        csv_path,
        expected_cases=["case_a"],
        expected_methods=["m1"],
        expected_budgets=[5],
        expected_repeats=2,
        min_repeats_per_cell=1,
        min_ok_fraction=0.95,
    )

    assert rc == 1
    assert "ok rows must leave error blank" in report
    assert "found 1 invalid rows" in report
    assert "OK Rows With Error Text" in report


def test_submission_ready_rejects_missing_manifest_artifact(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_missing_artifact")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": 123,
                        "sha256": "0" * 64,
                    }
                ]
            }
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_artifacts(manifest)

    assert errors
    assert "manifest artifact missing" in errors[0]


def test_submission_ready_scans_unresolved_tokens_but_skips_templates(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_token_scan")
    paper = tmp_path / "paper"
    paper.mkdir()
    manuscript = paper / "submission.md"
    manuscript.write_text(
        "Data DOI: TBD\n"
        "Result: [X.XXX] with CI [L, H]\n"
        "Reviewer note XXX\n",
        encoding="utf-8",
    )
    template = paper / "cover_letter.template.md"
    template.write_text("This TEMPLATE contains [X.XXX] and TBD by design.\n", encoding="utf-8")

    candidates = list(mod.iter_candidate_files(paper))
    hits = mod.scan_file(manuscript)

    assert manuscript in candidates
    assert template not in candidates
    kinds = [h[1] for h in hits]
    assert "TBD" in kinds
    assert "BRACKETED_RESULT" in kinds
    assert "XXX" in kinds


def test_submission_ready_ignores_latex_num_placeholder_fallback(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_token_stub")
    tex = tmp_path / "paper.tex"
    tex.write_text(
        "\\newcommand{\\NumNCases}{PLACEHOLDER}\n"
        "Actual manuscript text has no unresolved tokens.\n",
        encoding="utf-8",
    )

    assert mod.scan_file(tex) == []


def test_submission_ready_rejects_result_pending_draft_markers(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_token_pending")
    tex = tmp_path / "paper.tex"
    tex.write_text(
        "\\newcommand{\\ResultPending}{\\texttt{RESULT\\_PENDING}}\n"
        "The final D1 claim is \\ResultPending{} until the sweep finishes.\n",
        encoding="utf-8",
    )

    hits = mod.scan_file(tex)

    assert any(kind == "RESULT_PENDING" for _, kind, _ in hits)


def test_submission_ready_rejects_missing_required_submission_sources(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_sources_missing")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "cover_letter.template.md").write_text(
        "Template is not enough.\n", encoding="utf-8",
    )

    errors = mod.check_required_submission_sources(paper)

    assert any("nature_cs_main.tex" in err for err in errors)
    assert any("preregistration_decisions.md" in err for err in errors)
    assert any("significance_statement.md" in err for err in errors)
    assert any("cover_letter.md" in err for err in errors)


def test_submission_ready_accepts_required_submission_sources(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_sources_ok")
    paper = tmp_path / "paper"
    paper.mkdir()
    for rel in mod.REQUIRED_SUBMISSION_SOURCES:
        (paper / rel).write_text("submission text\n", encoding="utf-8")

    assert mod.check_required_submission_sources(paper) == []


def test_submission_ready_accepts_signed_preregistration_decisions(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_decisions_ok")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "preregistration_decisions.md").write_text(
        "# Pre-registration decisions\n\n"
        "## Provenance\n\n"
        "- **Run manifest**: `benchmark/results/full_benchmark/RUN_MANIFEST.json`\n"
        "- **Sweep started (UTC)**: 2026-06-01T00:00:00Z\n"
        "- **Sweep finished (UTC)**: 2026-06-02T00:00:00Z\n"
        "- **Cluster / host**: server-a, 64 cores, 256 GB RAM\n"
        "- **Author signing this decision document**: Signed Author, ORCID 0000-0000-0000-0000\n"
        "- **Date signed (UTC+08:00)**: 2026-06-03T10:00:00+08:00\n\n"
        "## D1 - H1\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D2 - H2\n\n"
        "- **Headline decision**: Falsified\n\n"
        "## D3 - H3\n\n"
        "- **Headline decision**: Inconclusive\n\n"
        "## Holm-Bonferroni ledger across {{D1, D2, D3}}\n\n"
        "| Discovery | Raw p | Rank | Holm threshold | Pass? |\n"
        "|---|---:|---:|---:|---|\n"
        "| D1 | 0.001 | 1 | 0.0167 | Y |\n"
        "| D2 | 0.020 | 2 | 0.0250 | N |\n"
        "| D3 | 0.030 | 3 | 0.0500 | Yes |\n\n"
        "## Falsification log\n\n"
        "### D2 falsification\n"
        "- §6 clause triggered: D2 CP upper bound exceeded the preregistered threshold.\n"
        "- New manuscript headline: The 10d rule was falsified under the archived sweep.\n\n"
        "### D3 falsification\n"
        "- SAP Section 6 clause triggered: D3 bias direction was inconclusive.\n"
        "- New manuscript headline: Failure-probability bias was inconclusive.\n\n"
        "## Sign-off\n\n"
        "By committing this filled-in document I attest that:\n\n"
        "1. The numbers above are read directly from the artefacts under "
        "`benchmark/results/{{full_benchmark, ranking_reversal, "
        "sample_complexity_0.20, reliability_recovery}}/` whose SHA-256s are "
        "pinned in `RUN_MANIFEST.json`.\n"
        "2. No manuscript text was amended before this document was committed.\n"
        "3. Any departure from the SAP is recorded in "
        "`preregistration_amendments.md`.\n\n"
        "-- Signed Author, 2026-06-03\n",
        encoding="utf-8",
    )

    assert mod.check_preregistration_decisions(paper, manifest_path=None) == []


def test_submission_ready_rejects_signature_datetime_without_utc8_offset(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_decisions_signature_tz")
    paper = tmp_path / "paper"
    paper.mkdir()
    base_text = (
        "# Pre-registration decisions\n\n"
        "## Provenance\n\n"
        "- **Run manifest**: `benchmark/results/full_benchmark/RUN_MANIFEST.json`\n"
        "- **Sweep started (UTC)**: 2026-06-01T00:00:00Z\n"
        "- **Sweep finished (UTC)**: 2026-06-02T00:00:00Z\n"
        "- **Cluster / host**: server-a, 64 cores, 256 GB RAM\n"
        "- **Author signing this decision document**: Signed Author, ORCID 0000-0000-0000-0000\n"
        "- **Date signed (UTC+08:00)**: {signature_datetime}\n\n"
        "## D1 - H1\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D2 - H2\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D3 - H3\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## Holm-Bonferroni ledger across {{D1, D2, D3}}\n\n"
        "| Discovery | Raw p | Rank | Holm threshold | Pass? |\n"
        "|---|---:|---:|---:|---|\n"
        "| D1 | 0.001 | 1 | 0.0167 | Y |\n"
        "| D2 | 0.002 | 2 | 0.0250 | Y |\n"
        "| D3 | 0.003 | 3 | 0.0500 | Y |\n\n"
        "## Sign-off\n\n"
        "By committing this filled-in document I attest that:\n\n"
        "1. The numbers above are read directly from the artefacts under "
        "`benchmark/results/{{full_benchmark, ranking_reversal, "
        "sample_complexity_0.20, reliability_recovery}}/` whose SHA-256s are "
        "pinned in `RUN_MANIFEST.json`.\n"
        "2. No manuscript text was amended before this document was committed.\n"
    )

    for signature_datetime in (
        "2026-06-03 10:00",
        "2026-06-03T02:00:00Z",
        "2026-06-03T10:00:00+09:00",
    ):
        (paper / "preregistration_decisions.md").write_text(
            base_text.format(signature_datetime=signature_datetime),
            encoding="utf-8",
        )

        errors = mod.check_preregistration_decisions(paper, manifest_path=None)

        assert any(
            "Provenance missing required UTC+08:00 signature datetime" in err
            for err in errors
        )


def test_submission_ready_rejects_incomplete_preregistration_decisions(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_decisions_bad")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "preregistration_decisions.md").write_text(
        "# Pre-registration decisions\n\n"
        "## Provenance\n\n"
        "- **Run manifest**: RESULT_PENDING\n"
        "- **Sweep started (UTC)**: yesterday\n"
        "- **Sweep finished (UTC)**: RESULT_PENDING\n"
        "- **Cluster / host**: [hostname]\n"
        "- **Author signing this decision document**: RESULT_PENDING\n"
        "- **Date signed (UTC+08:00)**: [YYYY-MM-DD HH:MM]\n\n"
        "## D1 - H1\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D2 - H2\n\n"
        "- **Headline decision**: Maybe\n\n"
        "## D3 - H3\n\n"
        "- **Rationale**: Filled without decision.\n\n"
        "## Holm-Bonferroni ledger across {D1, D2, D3}\n\n"
        "| Discovery | Raw p | Rank | Holm threshold | Pass? |\n"
        "|---|---:|---:|---:|---|\n"
        "| D1 | 0.001 | 1 | 0.0167 | RESULT_PENDING |\n"
        "| D2 | 0.020 | 2 | 0.0250 | maybe |\n\n"
        "## Sign-off\n\n"
        "Signed after manuscript edits.\n",
        encoding="utf-8",
    )

    errors = mod.check_preregistration_decisions(paper, manifest_path=None)

    assert any("D2 Headline decision must be one of" in err for err in errors)
    assert any("D3 missing Headline decision" in err for err in errors)
    assert any("Holm-Bonferroni ledger missing Discovery rows: ['D3']" in err for err in errors)
    assert any("D1 Pass? must be Y/N-like" in err for err in errors)
    assert any("D2 Pass? must be Y/N-like" in err for err in errors)
    assert any("Provenance missing required concrete run-manifest path" in err for err in errors)
    assert any("Provenance missing required ISO-like sweep start time" in err for err in errors)
    assert any("Provenance missing required ISO-like sweep finish time" in err for err in errors)
    assert any("Provenance missing required cluster/host record" in err for err in errors)
    assert any("Provenance missing required signing author identity" in err for err in errors)
    assert any("Provenance missing required UTC+08:00 signature datetime" in err for err in errors)
    assert any("Sign-off missing required archive-artifact attestation" in err for err in errors)
    assert any("Sign-off missing required pre-manuscript-edit attestation" in err for err in errors)


def test_submission_ready_rejects_missing_falsification_log_entries(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_decisions_falsification")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "preregistration_decisions.md").write_text(
        "# Pre-registration decisions\n\n"
        "## Provenance\n\n"
        "- **Run manifest**: `benchmark/results/full_benchmark/RUN_MANIFEST.json`\n"
        "- **Sweep started (UTC)**: 2026-06-01T00:00:00Z\n"
        "- **Sweep finished (UTC)**: 2026-06-02T00:00:00Z\n"
        "- **Cluster / host**: server-a, 64 cores, 256 GB RAM\n"
        "- **Author signing this decision document**: Signed Author, ORCID 0000-0000-0000-0000\n"
        "- **Date signed (UTC+08:00)**: 2026-06-03T10:00:00+08:00\n\n"
        "## D1 - H1\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D2 - H2\n\n"
        "- **Headline decision**: Falsified\n\n"
        "## D3 - H3\n\n"
        "- **Headline decision**: Inconclusive\n\n"
        "## Holm-Bonferroni ledger across {D1, D2, D3}\n\n"
        "| Discovery | Raw p | Rank | Holm threshold | Pass? |\n"
        "|---|---:|---:|---:|---|\n"
        "| D1 | 0.001 | 1 | 0.0167 | Y |\n"
        "| D2 | 0.020 | 2 | 0.0250 | N |\n"
        "| D3 | 0.030 | 3 | 0.0500 | No |\n\n"
        "## Falsification log\n\n"
        "### D2 falsification\n"
        "- New manuscript headline: D2 failed.\n\n"
        "## Sign-off\n\n"
        "By committing this filled-in document I attest that:\n\n"
        "1. The numbers above are read directly from the artefacts under "
        "`benchmark/results/{full_benchmark, ranking_reversal, "
        "sample_complexity_0.20, reliability_recovery}/` whose SHA-256s are "
        "pinned in `RUN_MANIFEST.json`.\n"
        "2. No manuscript text was amended before this document was committed.\n",
        encoding="utf-8",
    )

    errors = mod.check_preregistration_decisions(paper, manifest_path=None)

    assert any("D2 entry missing SAP Section 6 clause" in err for err in errors)
    assert any("Falsification log missing D3 falsification entry" in err for err in errors)


def test_submission_ready_validates_decision_results_manifest_hash(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_decisions_hash")
    paper = tmp_path / "paper"
    paper.mkdir()
    manifest = tmp_path / "RUN_MANIFEST.json"
    expected_sha = "a" * 64
    manifest.write_text(
        json.dumps(_decision_manifest_payload(expected_sha)),
        encoding="utf-8",
    )
    (paper / "preregistration_decisions.md").write_text(
        "# Pre-registration decisions\n\n"
        "## Provenance\n\n"
        f"- **Run manifest**: `{manifest}`\n"
        f"- **Manifest sha256 for results.csv**: {expected_sha}\n\n"
        "- **Sweep started (UTC)**: 2026-06-01T00:00:00Z\n"
        "- **Sweep finished (UTC)**: 2026-06-02T00:00:00Z\n"
        "- **Cluster / host**: server-a, 64 cores, 256 GB RAM\n"
        "- **Author signing this decision document**: Signed Author, ORCID 0000-0000-0000-0000\n"
        "- **Date signed (UTC+08:00)**: 2026-06-03T10:00:00+08:00\n\n"
        "## D1 - H1\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D2 - H2\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D3 - H3\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## Holm-Bonferroni ledger across {D1, D2, D3}\n\n"
        "| Discovery | Raw p | Rank | Holm threshold | Pass? |\n"
        "|---|---:|---:|---:|---|\n"
        "| D1 | 0.001 | 1 | 0.0167 | Y |\n"
        "| D2 | 0.002 | 2 | 0.0250 | Y |\n"
        "| D3 | 0.003 | 3 | 0.0500 | Y |\n\n"
        "## Sign-off\n\n"
        "By committing this filled-in document I attest that:\n\n"
        "1. The numbers above are read directly from the artefacts under "
        "`benchmark/results/{full_benchmark, ranking_reversal, "
        "sample_complexity_0.20, reliability_recovery}/` whose SHA-256s are "
        "pinned in `RUN_MANIFEST.json`.\n"
        "2. No manuscript text was amended before this document was committed.\n",
        encoding="utf-8",
    )

    assert mod.check_preregistration_decisions(paper, manifest) == []

    text = (paper / "preregistration_decisions.md").read_text(encoding="utf-8")
    (paper / "preregistration_decisions.md").write_text(
        text.replace(expected_sha, "b" * 64),
        encoding="utf-8",
    )

    errors = mod.check_preregistration_decisions(paper, manifest)

    assert any("Manifest sha256 for results.csv does not match" in err for err in errors)


def test_submission_ready_rejects_decision_manifest_path_drift(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_decisions_manifest_path")
    paper = tmp_path / "paper"
    paper.mkdir()
    manifest = tmp_path / "RUN_MANIFEST.json"
    expected_sha = "a" * 64
    manifest.write_text(
        json.dumps(_decision_manifest_payload(expected_sha)),
        encoding="utf-8",
    )
    (paper / "preregistration_decisions.md").write_text(
        "# Pre-registration decisions\n\n"
        "## Provenance\n\n"
        "- **Run manifest**: `benchmark/results/old_benchmark/RUN_MANIFEST.json`\n"
        f"- **Manifest sha256 for results.csv**: {expected_sha}\n\n"
        "- **Sweep started (UTC)**: 2026-06-01T00:00:00Z\n"
        "- **Sweep finished (UTC)**: 2026-06-02T00:00:00Z\n"
        "- **Cluster / host**: server-a, 64 cores, 256 GB RAM\n"
        "- **Author signing this decision document**: Signed Author, ORCID 0000-0000-0000-0000\n"
        "- **Date signed (UTC+08:00)**: 2026-06-03T10:00:00+08:00\n\n"
        "## D1 - H1\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D2 - H2\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D3 - H3\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## Holm-Bonferroni ledger across {D1, D2, D3}\n\n"
        "| Discovery | Raw p | Rank | Holm threshold | Pass? |\n"
        "|---|---:|---:|---:|---|\n"
        "| D1 | 0.001 | 1 | 0.0167 | Y |\n"
        "| D2 | 0.002 | 2 | 0.0250 | Y |\n"
        "| D3 | 0.003 | 3 | 0.0500 | Y |\n\n"
        "## Sign-off\n\n"
        "By committing this filled-in document I attest that:\n\n"
        "1. The numbers above are read directly from the artefacts under "
        "`benchmark/results/{full_benchmark, ranking_reversal, "
        "sample_complexity_0.20, reliability_recovery}/` whose SHA-256s are "
        "pinned in `RUN_MANIFEST.json`.\n"
        "2. No manuscript text was amended before this document was committed.\n",
        encoding="utf-8",
    )

    errors = mod.check_preregistration_decisions(paper, manifest)

    assert any("Provenance Run manifest does not match" in err for err in errors)


def test_submission_ready_rejects_decision_manifest_provenance_drift(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_decisions_provenance_drift")
    paper = tmp_path / "paper"
    paper.mkdir()
    manifest = tmp_path / "RUN_MANIFEST.json"
    expected_sha = "a" * 64
    manifest.write_text(
        json.dumps(_decision_manifest_payload(expected_sha)),
        encoding="utf-8",
    )
    (paper / "preregistration_decisions.md").write_text(
        "# Pre-registration decisions\n\n"
        "## Provenance\n\n"
        f"- **Run manifest**: `{manifest}`\n"
        f"- **Manifest sha256 for results.csv**: {expected_sha}\n\n"
        "- **Sweep started (UTC)**: 2026-06-01T01:00:00Z\n"
        "- **Sweep finished (UTC)**: 2026-06-02T01:00:00Z\n"
        "- **Cluster / host**: other-host, 64 cores, 256 GB RAM\n"
        "- **Author signing this decision document**: Signed Author, ORCID 0000-0000-0000-0000\n"
        "- **Date signed (UTC+08:00)**: 2026-06-03T10:00:00+08:00\n\n"
        "## D1 - H1\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D2 - H2\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D3 - H3\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## Holm-Bonferroni ledger across {D1, D2, D3}\n\n"
        "| Discovery | Raw p | Rank | Holm threshold | Pass? |\n"
        "|---|---:|---:|---:|---|\n"
        "| D1 | 0.001 | 1 | 0.0167 | Y |\n"
        "| D2 | 0.002 | 2 | 0.0250 | Y |\n"
        "| D3 | 0.003 | 3 | 0.0500 | Y |\n\n"
        "## Sign-off\n\n"
        "By committing this filled-in document I attest that:\n\n"
        "1. The numbers above are read directly from the artefacts under "
        "`benchmark/results/{full_benchmark, ranking_reversal, "
        "sample_complexity_0.20, reliability_recovery}/` whose SHA-256s are "
        "pinned in `RUN_MANIFEST.json`.\n"
        "2. No manuscript text was amended before this document was committed.\n",
        encoding="utf-8",
    )

    errors = mod.check_preregistration_decisions(paper, manifest)

    assert any("Sweep started (UTC) does not match" in err for err in errors)
    assert any("Sweep finished (UTC) does not match" in err for err in errors)
    assert any("Cluster / host does not mention" in err for err in errors)


def test_submission_ready_rejects_missing_decision_results_manifest_hash(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_decisions_hash_missing")
    paper = tmp_path / "paper"
    paper.mkdir()
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps(_decision_manifest_payload("a" * 64)),
        encoding="utf-8",
    )
    (paper / "preregistration_decisions.md").write_text(
        "# Pre-registration decisions\n\n"
        "## Provenance\n\n"
        f"- **Run manifest**: `{manifest}`\n"
        "- **Sweep started (UTC)**: 2026-06-01T00:00:00Z\n"
        "- **Sweep finished (UTC)**: 2026-06-02T00:00:00Z\n"
        "- **Cluster / host**: server-a, 64 cores, 256 GB RAM\n"
        "- **Author signing this decision document**: Signed Author, ORCID 0000-0000-0000-0000\n"
        "- **Date signed (UTC+08:00)**: 2026-06-03T10:00:00+08:00\n\n"
        "## D1 - H1\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D2 - H2\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## D3 - H3\n\n"
        "- **Headline decision**: Confirmed\n\n"
        "## Holm-Bonferroni ledger across {D1, D2, D3}\n\n"
        "| Discovery | Raw p | Rank | Holm threshold | Pass? |\n"
        "|---|---:|---:|---:|---|\n"
        "| D1 | 0.001 | 1 | 0.0167 | Y |\n"
        "| D2 | 0.002 | 2 | 0.0250 | Y |\n"
        "| D3 | 0.003 | 3 | 0.0500 | Y |\n\n"
        "## Sign-off\n\n"
        "By committing this filled-in document I attest that:\n\n"
        "1. The numbers above are read directly from the artefacts under "
        "`benchmark/results/{full_benchmark, ranking_reversal, "
        "sample_complexity_0.20, reliability_recovery}/` whose SHA-256s are "
        "pinned in `RUN_MANIFEST.json`.\n"
        "2. No manuscript text was amended before this document was committed.\n",
        encoding="utf-8",
    )

    errors = mod.check_preregistration_decisions(paper, manifest)

    assert any("missing concrete Manifest sha256 for results.csv" in err for err in errors)


def test_submission_ready_accepts_editorial_material_format(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_editorial_ok")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "significance_statement.md").write_text(
        "# Significance Statement\n\n"
        + " ".join(["clear"] * 120)
        + "\n",
        encoding="utf-8",
    )
    (paper / "cover_letter.md").write_text(
        "# Cover Letter\n\n"
        "Dear Editor,\n\n"
        "We submit DOE-Benchmark, a physics-grounded resource for evaluating "
        "20 design-of-experiments methods across 34 physics-grounded engineering "
        "simulations. The RUN_MANIFEST.json records the raw data, figures "
        "and analysis scripts, with a SHA-256 checksum. A versioned delivery "
        "archive will be assembled for reproducible review.\n\n"
        "Yours sincerely,\n"
        "The authors\n",
        encoding="utf-8",
    )

    assert mod.check_editorial_submission_materials(paper) == []


def test_submission_ready_rejects_editorial_material_format_errors(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_editorial_bad")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "significance_statement.md").write_text(
        "# Significance Statement\n\n"
        + " ".join(["word"] * 220)
        + "\n",
        encoding="utf-8",
    )
    (paper / "cover_letter.md").write_text(
        "# Cover Letter\n\n"
        "Hello,\n\n"
        + " ".join(["word"] * 451)
        + "\n\n"
        "The final case count is \\NumNCases{}.\n\n"
        "Regards,\n",
        encoding="utf-8",
    )

    errors = mod.check_editorial_submission_materials(paper)

    assert any("significance_statement.md body word count outside" in err for err in errors)
    assert any("cover_letter.md body word count exceeds" in err for err in errors)
    assert any("cover_letter.md missing required element: editor salutation" in err for err in errors)
    assert any("cover_letter.md missing required element: closing signature line" in err for err in errors)
    assert any("cover_letter.md contains LaTeX number macros" in err for err in errors)


def test_submission_ready_accepts_case_validation_caveat(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_caveat_ok")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "nature_cs_main.tex").write_text(
        "The evidence tiers separate P2 plausibility-only cases; "
        "these support benchmark stress testing but not strong external "
        "case-level validation claims.\n",
        encoding="utf-8",
    )

    assert mod.check_manuscript_case_validation_caveat(paper) == []


def test_submission_ready_rejects_missing_case_validation_caveat(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_caveat_bad")
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "nature_cs_main.tex").write_text(
        "The benchmark includes case validation and physics checks.\n",
        encoding="utf-8",
    )

    errors = mod.check_manuscript_case_validation_caveat(paper)

    assert any("case-validation evidence-tier caveat" in err for err in errors)


def test_paper_draft_ready_rejects_result_filled_package():
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_current")
    errors = mod.check_draft_package(ROOT / "paper")

    # The production manuscript is result-filled; it must not be treated as
    # a pre-results draft merely to satisfy this legacy-stage gate.
    assert any("no RESULT_PENDING marker" in err for err in errors)


def test_paper_draft_ready_permission_error_is_actionable(monkeypatch):
    mod = _load_module(
        "scripts/check_paper_draft_ready.py",
        "paper_draft_ready_permission_error",
    )

    def deny(_paper_dir):
        raise PermissionError(13, "Access denied", "paper/figures/pdf/fig1.pdf")

    monkeypatch.setattr(mod, "_check_draft_package_impl", deny)

    errors = mod.check_draft_package(ROOT / "paper")

    assert len(errors) == 1
    assert "paper/figures/pdf/fig1.pdf" in errors[0]
    assert "Fix the file/directory ACL" in errors[0]


def test_paper_draft_ready_rejects_stale_si_case_evidence_tiers(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_si_evidence")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    si_cases = paper / "si_cases.md"
    si_cases.write_text(
        si_cases.read_text(encoding="utf-8").replace(
            "| `pressure_vessel_4d` | multidisciplinary | 4 | 3 | analytical | "
            "Thin-walled cylindrical pressure-vessel membrane-stress surr | L0 | "
            "✓ | `targeted_analytical_control` |",
            "| `pressure_vessel_4d` | multidisciplinary | 4 | 3 | analytical | "
            "Thin-walled cylindrical pressure-vessel membrane-stress surr | L0 | "
            "✓ | `targeted_plausibility_only` |",
        ),
        encoding="utf-8",
    )

    errors = mod.check_draft_package(paper)

    assert any("draft SI package drift" in err for err in errors)
    assert any("pressure_vessel_4d" in err for err in errors)


def test_paper_draft_ready_rejects_numbers_tex_json_drift(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_numbers_drift")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    numbers_tex = paper / "numbers.tex"
    numbers_tex.write_text(
        numbers_tex.read_text(encoding="utf-8").replace(
            r"\newcommand{\NumNCases}{34}",
            r"\newcommand{\NumNCases}{33}",
        ),
        encoding="utf-8",
    )

    errors = mod.check_draft_package(paper)

    assert any(
        "draft numbers package drift" in err
        and r"\NumNCases" in err
        and "mismatch" in err
        for err in errors
    )


def test_paper_draft_ready_rejects_figure_manifest_drift(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_figure_drift")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    manifest_path = paper / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figures"][0]["pdf_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = mod.check_draft_package(paper)

    assert any(
        "draft figure package drift" in err
        and "fig1" in err
        and "pdf sha256 mismatch" in err
        for err in errors
    )


def test_paper_draft_ready_allows_draft_pending_figdata_manifest(tmp_path):
    draft = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_figdata_pending")
    submission = _load_module("scripts/check_submission_ready.py", "submission_figdata_pending_final")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)

    # Convert the copied complete figdata manifest into an explicit draft
    # fixture and refresh the parent manifest fingerprint.  The production
    # figure package is complete and must remain so.
    figdata_path = paper / "figdata" / "figdata_manifest.json"
    figdata = json.loads(figdata_path.read_text(encoding="utf-8"))
    figdata["status"] = "draft_pending_full_results"
    figdata_path.write_text(json.dumps(figdata), encoding="utf-8")
    manifest_path = paper / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figdata_manifest"]["sha256"] = hashlib.sha256(
        figdata_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    draft_errors = draft.check_draft_package(paper)
    final_figure_errors = submission.check_figure_manifest(
        paper / "figure_manifest.json",
        paper,
    )

    assert not any("draft figure package drift" in err for err in draft_errors)
    assert any("linked figdata_manifest status is 'draft_pending_full_results'" in err for err in final_figure_errors)


def test_paper_draft_ready_rejects_oversized_significance_statement(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_significance_words")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    statement = paper / "significance_statement.md"
    statement.write_text(
        "# Significance Statement\n\n"
        "> **Status**: DRAFT / RESULT_PENDING.\n\n"
        + " ".join(["word"] * 220)
        + " RESULT_PENDING\n",
        encoding="utf-8",
    )

    errors = mod.check_draft_package(paper)

    assert any(
        "significance_statement.md body word count outside Nature-family" in err
        for err in errors
    )


def test_paper_draft_ready_rejects_oversized_cover_letter(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_cover_words")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    cover_letter = paper / "cover_letter.md"
    cover_letter.write_text(
        cover_letter.read_text(encoding="utf-8")
        + "\n\n"
        + " ".join(["word"] * 451)
        + " RESULT_PENDING\n",
        encoding="utf-8",
    )

    errors = mod.check_draft_package(paper)

    assert any("cover_letter.md body word count exceeds Nature-family" in err for err in errors)


def test_paper_draft_ready_rejects_cover_letter_missing_structure(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_cover_structure")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    cover_letter = paper / "cover_letter.md"
    cover_letter.write_text(
        cover_letter.read_text(encoding="utf-8")
        .replace("Dear Editor,", "Hello,")
        .replace("Yours sincerely,", "Regards,"),
        encoding="utf-8",
    )

    errors = mod.check_draft_package(paper)

    assert any("cover_letter.md missing required element: editor salutation" in err for err in errors)
    assert any(
        "cover_letter.md missing required element: closing signature line" in err
        for err in errors
    )


def test_paper_draft_ready_rejects_cover_letter_number_macros(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_cover_macros")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    cover_letter = paper / "cover_letter.md"
    cover_letter.write_text(
        cover_letter.read_text(encoding="utf-8")
        + "\n\nThe final case count is \\NumNCases{} after RESULT_PENDING.\n",
        encoding="utf-8",
    )

    errors = mod.check_draft_package(paper)

    assert any(
        "cover_letter.md contains LaTeX number macros" in err
        and r"\NumNCases" in err
        for err in errors
    )


def test_paper_draft_ready_rejects_decision_ledger_missing_d1_pairings(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_decision_d1")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    decisions = paper / "preregistration_decisions.md"
    decisions.write_text(
        re.sub(
            r"^\|\s*(?:dimension|topology|random)\s*\|.*\n?",
            "",
            decisions.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        ),
        encoding="utf-8",
    )

    errors = mod.check_draft_package(paper)

    assert any(
        "preregistration_decisions.md missing required decision-ledger element: D1 pairing rows"
        in err
        for err in errors
    )


def test_paper_draft_ready_rejects_decision_ledger_missing_signoff_chain(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_decision_signoff")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    decisions = paper / "preregistration_decisions.md"
    text = decisions.read_text(encoding="utf-8")
    text = re.sub(
        r"\n## Holm-Bonferroni Ledger.*",
        "\n## Reviewer Notes\n\nRESULT_PENDING\n",
        text,
        flags=re.S,
    )
    decisions.write_text(text, encoding="utf-8")

    errors = mod.check_draft_package(paper)

    assert any(
        "preregistration_decisions.md missing required decision-ledger element: "
        "Holm-Bonferroni ledger"
        in err
        for err in errors
    )
    assert any(
        "preregistration_decisions.md missing required decision-ledger element: sign-off section"
        in err
        for err in errors
    )
    assert any(
        "preregistration_decisions.md missing required decision-ledger element: "
        "pre-manuscript-rewrite attestation"
        in err
        for err in errors
    )


def test_paper_draft_ready_requires_statistical_explanations(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_stats")
    paper = tmp_path / "paper"
    shutil.copytree(ROOT / "paper", paper)
    main = paper / "nature_cs_main.tex"
    text = main.read_text(encoding="utf-8")
    text = text.replace("Spearman $\\rho$ is interpreted as the D1", "Spearman $\\rho$ is the D1")
    text = text.replace("limited power", "small-sample caveat")
    text = text.replace("Test-set\nMonte-Carlo noise is reported", "Test-set\nnoise is summarized")
    text = text.replace("Test-set Monte-Carlo noise is reported", "Test-set noise is summarized")
    text = re.sub(r"cluster-vs-i\.i\.d\.\s+CI-width\s+ratio",
                  "bootstrap sensitivity", text)
    text = text.replace("the multiple-testing policy", "the inference policy")
    text = text.replace("the D2 exploratory-topology denominator policy", "the D2 policy")
    text = text.replace("the test-set Monte-Carlo noise floor", "the test-set noise floor")
    main.write_text(text, encoding="utf-8")

    errors = mod.check_draft_package(paper)

    assert any("D1 Spearman rho effect-size" in err for err in errors)
    assert any("D2 fixed-topology limited-power" in err for err in errors)
    assert any("test-set MC-noise" in err for err in errors)
    assert any("cluster-vs-iid sensitivity" in err for err in errors)
    assert any("Supplementary Information multiple-testing" in err for err in errors)
    assert any("Supplementary Information D2 policy" in err for err in errors)
    assert any("Supplementary Information MC-noise" in err for err in errors)


def test_paper_draft_ready_requires_pending_markers(tmp_path):
    mod = _load_module("scripts/check_paper_draft_ready.py", "paper_draft_ready_pending")
    submission = _load_module("scripts/check_submission_ready.py", "paper_draft_ready_sources")
    paper = tmp_path / "paper"
    paper.mkdir()
    for rel in submission.REQUIRED_SUBMISSION_SOURCES:
        (paper / rel).write_text("Draft source without marker.\n", encoding="utf-8")
    for rel in submission.REQUIRED_SI_FILES:
        (paper / rel).write_text("generated SI\n", encoding="utf-8")
    (paper / "README.md").write_text("readme\n", encoding="utf-8")
    (paper / "numbers.tex").write_text("", encoding="utf-8")
    (paper / "numbers.json").write_text(
        json.dumps({
            "nCases": 34,
            "nMethods": 20,
            "nRepeats": 30,
            "nBudgets": 4,
            "budgetList": "5, 10, 20, 50",
            "totalCells": 81600,
            "primaryFailurePolicy": "penalized_failed_in_denominator",
            "dOneRankPolicy": "complete_case",
        }),
        encoding="utf-8",
    )

    errors = mod.check_draft_package(paper)

    assert any("no RESULT_PENDING marker" in err for err in errors)


def test_case_experiment_readiness_accepts_registered_cases_subset(tmp_path):
    mod = _load_module("scripts/audit_case_experiment_readiness.py", "case_experiment_ready_subset")
    rows = [
        mod.audit_case("pressure_vessel_4d", n_samples=4, seed=0),
        mod.audit_case("cantilever_beam_5d", n_samples=4, seed=0),
    ]
    output = tmp_path / "case_readiness.csv"

    mod.write_rows(rows, output)

    assert output.exists()
    assert {row["status"] for row in rows} <= {"ok", "timing_warning"}
    assert all(row["n_nan"] == 0 for row in rows)
    assert all(row["n_inf"] == 0 for row in rows)
    assert all(row["determinism_n_mismatch"] == 0 for row in rows)
    assert all(row["n_expected_range_violations"] == 0 for row in rows)
    assert all(row["response_topology"] for row in rows)
    assert all(row["n_topology_tags"] >= 1 for row in rows)
    assert all(row["pde_type"] for row in rows)
    assert all(row["n_physics_references"] >= 1 for row in rows)


def test_case_physics_runner_rejects_missing_ledger_target(tmp_path):
    mod = _load_module("scripts/run_case_physics_tests.py", "case_physics_runner_coverage")
    root = tmp_path / "repo"
    tests_dir = root / "tests"
    docs_dir = root / "docs"
    tests_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    discovered = tests_dir / "test_existing_physics.py"
    discovered.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    envelope = tests_dir / "test_expected_range_envelopes.py"
    envelope.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    ledger = docs_dir / "case_validation_status.csv"
    ledger.write_text(
        "case_id,targeted_physics_test\n"
        "case_a,tests/test_existing_physics.py\n"
        "case_b,tests/test_missing_physics.py\n",
        encoding="utf-8",
    )

    errors = mod.validate_ledger_test_coverage(
        [discovered, envelope],
        root,
        ledger,
    )

    assert errors == [
        "case physics runner does not discover targeted ledger test: "
        "tests/test_missing_physics.py"
    ]


def test_case_physics_runner_rejects_missing_expected_range_test(tmp_path):
    mod = _load_module("scripts/run_case_physics_tests.py", "case_physics_runner_envelope")
    root = tmp_path / "repo"
    tests_dir = root / "tests"
    docs_dir = root / "docs"
    tests_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    discovered = tests_dir / "test_existing_physics.py"
    discovered.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    ledger = docs_dir / "case_validation_status.csv"
    ledger.write_text(
        "case_id,targeted_physics_test\n"
        "case_a,tests/test_existing_physics.py\n",
        encoding="utf-8",
    )

    errors = mod.validate_ledger_test_coverage([discovered], root, ledger)

    assert errors == [
        "case physics runner missing envelope test: tests/test_expected_range_envelopes.py"
    ]


def test_case_experiment_readiness_flags_nonfinite_outputs(monkeypatch):
    mod = _load_module("scripts/audit_case_experiment_readiness.py", "case_experiment_ready_bad")

    class _Output:
        failure_threshold = None
        failure_direction = None
        name = "y"

    class _Meta:
        case_id = "bad_case"
        domain = "test"
        dim_input = 2
        dim_output = 1
        input_variables = [object(), object()]
        output_variables = [_Output()]
        has_failure_boundary = False
        supports_multifidelity = False
        fidelity_levels = ["default"]
        typical_eval_time_ms = {"default": 1.0}

    class _Case:
        meta = _Meta()

        def evaluate(self, X):
            return np.full((len(X), 1), np.nan)

    monkeypatch.setattr(mod.db, "get_case", lambda case_id: _Case())

    row = mod.audit_case("bad_case", n_samples=4, seed=0)

    assert row["status"] == "nonfinite"
    assert row["n_nan"] == 4
    assert "non-finite outputs" in row["message"]


def test_case_experiment_readiness_flags_nondeterministic_outputs(monkeypatch):
    mod = _load_module(
        "scripts/audit_case_experiment_readiness.py",
        "case_experiment_ready_nondeterministic",
    )

    class _Output:
        failure_threshold = None
        failure_direction = None
        name = "y"

    class _Meta:
        case_id = "random_case"
        domain = "test"
        dim_input = 2
        dim_output = 1
        input_variables = [object(), object()]
        output_variables = [_Output()]
        has_failure_boundary = False
        supports_multifidelity = False
        fidelity_levels = ["default"]
        typical_eval_time_ms = {"default": 1.0}

    class _Case:
        meta = _Meta()

        def __init__(self):
            self.calls = 0

        def evaluate(self, X):
            self.calls += 1
            return np.full((len(X), 1), float(self.calls))

    monkeypatch.setattr(mod.db, "get_case", lambda case_id: _Case())

    row = mod.audit_case("random_case", n_samples=4, seed=0)

    assert row["status"] == "nondeterministic"
    assert row["determinism_n_mismatch"] == 4
    assert "repeat evaluate is not deterministic" in row["message"]


def test_case_experiment_readiness_flags_input_mutation(monkeypatch):
    mod = _load_module(
        "scripts/audit_case_experiment_readiness.py",
        "case_experiment_ready_input_mutation",
    )

    class _Input:
        name = "x"
        description = "input"
        unit = "-"
        lower = 0.0
        upper = 1.0
        log_scale = False

    class _Output:
        name = "y"
        description = "output"
        unit = "-"
        expected_range = (0.0, 1.0)
        failure_threshold = None
        failure_direction = None

    class _Meta:
        case_id = "mutating_case"
        name = "mutating case"
        domain = "structures"
        pde_type = "algebraic"
        governing_equation = "y = x"
        physics_references = ["test reference"]
        response_topology = ["smooth_monotone"]
        dim_input = 1
        dim_output = 1
        input_variables = [_Input()]
        output_variables = [_Output()]
        has_failure_boundary = False
        supports_multifidelity = False
        fidelity_levels = ["default"]
        typical_eval_time_ms = {"default": 1.0}

    class _Case:
        meta = _Meta()

        def normalize(self, X):
            return X

        def evaluate(self, X):
            X[:, 0] = 0.25
            return np.full((len(X), 1), 0.25)

    monkeypatch.setattr(mod.db, "get_case", lambda case_id: _Case())

    row = mod.audit_case("mutating_case", n_samples=4, seed=0)

    assert row["status"] == "input_mutation"
    assert row["input_mutation_n_changed"] > 0
    assert "evaluate mutated its input array" in row["message"]


def test_case_experiment_readiness_flags_expected_range_violation(monkeypatch):
    mod = _load_module(
        "scripts/audit_case_experiment_readiness.py",
        "case_experiment_ready_expected_range",
    )

    class _Input:
        lower = 0.0
        upper = 1.0

    class _Output:
        failure_threshold = None
        failure_direction = None
        name = "y"
        expected_range = (0.0, 1.0)

    class _Meta:
        case_id = "range_case"
        domain = "test"
        dim_input = 1
        dim_output = 1
        input_variables = [_Input()]
        output_variables = [_Output()]
        has_failure_boundary = False
        supports_multifidelity = False
        fidelity_levels = ["default"]
        typical_eval_time_ms = {"default": 1.0}

    class _Case:
        meta = _Meta()

        def normalize(self, X):
            return X

        def evaluate(self, X):
            return np.full((len(X), 1), 2.0)

    monkeypatch.setattr(mod.db, "get_case", lambda case_id: _Case())

    row = mod.audit_case("range_case", n_samples=4, seed=0)

    assert row["status"] == "metadata_warning"
    assert row["n_expected_range_checked"] == 1
    assert row["n_expected_range_violations"] == 1
    assert "outside expected_range" in row["message"]


def test_case_experiment_readiness_flags_analysis_metadata_drift(monkeypatch):
    mod = _load_module(
        "scripts/audit_case_experiment_readiness.py",
        "case_experiment_ready_metadata_drift",
    )

    class _Input:
        name = "x"
        description = "input"
        unit = "-"
        lower = 0.0
        upper = 1.0
        log_scale = False

    class _Output:
        name = "y"
        description = "output"
        unit = "-"
        expected_range = (0.0, 1.0)
        failure_threshold = 2.0
        failure_direction = "above"

    class _Meta:
        case_id = "metadata_drift_case"
        name = "metadata drift"
        domain = "test_domain"
        pde_type = "made_up"
        governing_equation = ""
        physics_references = []
        response_topology = ["smooth_monotone", "not_a_topology", "smooth_monotone"]
        dim_input = 2
        dim_output = 1
        input_variables = [_Input(), _Input()]
        output_variables = [_Output()]
        has_failure_boundary = True
        supports_multifidelity = False
        fidelity_levels = ["default"]
        typical_eval_time_ms = {"default": 1.0}

    class _Case:
        meta = _Meta()

        def normalize(self, X):
            return X

        def evaluate(self, X):
            return np.full((len(X), 1), 0.5)

    monkeypatch.setattr(mod.db, "get_case", lambda case_id: _Case())

    row = mod.audit_case("metadata_drift_case", n_samples=4, seed=0)

    assert row["status"] == "metadata_warning"
    assert "domain 'test_domain' is not in the closed vocabulary" in row["message"]
    assert "unknown response_topology tags ['not_a_topology']" in row["message"]
    assert "duplicate input variable names" in row["message"]
    assert "failure_threshold 2 outside expected_range" in row["message"]


def test_case_experiment_readiness_can_promote_timing_warning_to_failure():
    mod = _load_module(
        "scripts/audit_case_experiment_readiness.py",
        "case_experiment_ready_warning_policy",
    )
    rows = [
        {"case_id": "ok_case", "status": "ok"},
        {"case_id": "slow_case", "status": "timing_warning"},
    ]

    fatal, warnings = mod.summarize_statuses(rows, fail_on_warning=False)
    strict_fatal, strict_warnings = mod.summarize_statuses(rows, fail_on_warning=True)

    assert fatal == []
    assert [row["case_id"] for row in warnings] == ["slow_case"]
    assert [row["case_id"] for row in strict_fatal] == ["slow_case"]
    assert [row["case_id"] for row in strict_warnings] == ["slow_case"]


def test_case_experiment_readiness_write_failure_preserves_existing_csv(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_case_experiment_readiness.py",
        "case_experiment_ready_atomic_write",
    )
    output = tmp_path / "case_readiness.csv"
    output.write_text("old csv\n", encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("csv write failed")

    monkeypatch.setattr(mod.csv.DictWriter, "writerows", boom)

    with pytest.raises(RuntimeError, match="csv write failed"):
        mod.write_rows([
            {column: "" for column in mod.REQUIRED_COLUMNS} | {"case_id": "c1", "status": "ok"}
        ], output)

    assert output.read_text(encoding="utf-8") == "old csv\n"
    assert not list(tmp_path.glob(".case_readiness.csv.tmp.*"))


def test_publication_readiness_report_marks_missing_sweep_as_blocked(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_missing_sweep",
    )
    missing_results = tmp_path / "full_benchmark"

    gate = mod.summarize_sweep_artifacts(missing_results)
    report = mod.render_report([
        gate,
        mod.GateResult("Example pass", "PASS", "ok", "none"),
    ])

    assert gate.status == "BLOCKED"
    assert "results.csv" in gate.evidence
    assert "Overall status: **BLOCKED**" in report
    assert "| Full sweep results | BLOCKED |" in report


def test_publication_readiness_blocks_unreadable_results_csv(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_unreadable_results_csv",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    (results_dir / "results.csv").write_bytes(b"\xff\xfe\x00\x00not utf-8\n")

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "could not read" in gate.evidence
    assert "results.csv" in gate.evidence
    assert "Restore a valid UTF-8 CSV" in gate.next_action


def test_publication_readiness_permission_error_is_actionable():
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_permission_error",
    )

    gate = mod._filesystem_error_gate(
        "Smoke pipeline",
        PermissionError(13, "Access denied", "paper/figures"),
    )

    assert gate.status == "BLOCKED"
    assert "paper/figures" in gate.evidence
    assert "Fix the file/directory ACL" in gate.next_action


def test_publication_readiness_requires_passing_full_sweep_audit_report(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_audit_required",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    _write_results_rows(results_dir / "results.csv", rows)
    _write_minimal_publication_manifest(
        results_dir / "RUN_MANIFEST.json",
        results_dir / "results.csv",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "# Sweep Integrity Report\n\n## Findings\n\n### Errors\n\n- stale source hash\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "did not contain a PASS verdict" in gate.evidence
    assert "rerun `sweep_audit`" in gate.next_action


def test_publication_readiness_reruns_sweep_integrity_before_accepting_report(
    tmp_path,
    monkeypatch,
):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_fresh_audit",
    )
    monkeypatch.setattr(
        mod,
        "_rerun_sweep_integrity",
        lambda results_csv: (1, "duplicate protocol cells"),
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    _write_results_rows(results_dir / "results.csv", rows)
    _write_minimal_publication_manifest(
        results_dir / "RUN_MANIFEST.json",
        results_dir / "results.csv",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "Fresh local sweep-integrity audit failed" in gate.evidence


def test_publication_readiness_accepts_passing_full_sweep_audit_report(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_audit_pass",
    )
    monkeypatch.setattr(mod, "_rerun_sweep_integrity", lambda results_csv: (0, "PASS"))
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    _write_results_rows(results_dir / "results.csv", rows)
    _write_minimal_publication_manifest(
        results_dir / "RUN_MANIFEST.json",
        results_dir / "results.csv",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "PASS"
    assert "sweep_integrity_report.md" in gate.evidence
    assert "fresh local sweep-integrity audit PASS" in gate.evidence


def test_publication_readiness_defers_server_return_until_full_sweep_passes():
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_server_return_deferred",
    )

    gate = mod.summarize_server_return_package(
        mod.GateResult("Full sweep results", "BLOCKED", "missing", "restore")
    )

    assert gate.status == "DEFERRED"
    assert "Full sweep results" in gate.evidence


def test_publication_readiness_cli_accepts_logs_dir():
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_cli_logs_dir",
    )

    args = mod.parse_args(["--logs-dir", "returned_logs"])

    assert args.logs_dir == Path("returned_logs")


def test_publication_readiness_main_passes_custom_logs_dir(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_main_logs_dir",
    )
    seen: dict[str, Path] = {}
    monkeypatch.setattr(
        mod,
        "summarize_sweep_artifacts",
        lambda results_dir: mod.GateResult("Full sweep results", "BLOCKED", "missing", "restore"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_validation",
        lambda: mod.GateResult("Case physics validation ledger", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_readiness",
        lambda: mod.GateResult("Case experiment readiness", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_smoke_pipeline",
        lambda results_root: mod.GateResult("Smoke pipeline", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_draft_gate",
        lambda paper_dir: mod.GateResult("Pre-results paper draft", "PASS", "ok", "none"),
    )

    def fake_server_return(sweep_gate, *, logs_dir, **kwargs):
        seen["logs_dir"] = logs_dir
        return mod.GateResult("Server return package", "DEFERRED", "ok", "none")

    monkeypatch.setattr(mod, "summarize_server_return_package", fake_server_return)

    rc = mod.main([
        "--logs-dir", str(tmp_path / "returned_logs"),
        "--output", str(tmp_path / "audit.md"),
        "--no-submission-gate",
    ])

    assert rc == 1
    assert seen["logs_dir"] == tmp_path / "returned_logs"


def test_publication_readiness_server_return_preserves_paths_before_checker(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_server_return_preserve_paths",
    )
    server = _load_module(
        "scripts/check_server_return_package.py",
        "publication_readiness_server_return_preserve_checker",
    )
    seen = {}

    def fake_check_server_return_package(**kwargs):
        seen.update(kwargs)
        return {}

    server.check_server_return_package = fake_check_server_return_package
    sys.modules["scripts.check_server_return_package"] = server
    raw_base = tmp_path / "returned_link" / ".." / "returned"
    results_root = raw_base / "benchmark" / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = raw_base / "paper"
    logs_dir = raw_base / "logs"

    gate = mod.summarize_server_return_package(
        mod.GateResult("Full sweep results", "PASS", "ready", "No action."),
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        logs_dir=logs_dir,
    )

    assert gate.status == "PASS"
    assert seen["results_root"] == results_root
    assert seen["results_dir"] == results_dir
    assert seen["paper_dir"] == paper_dir
    assert seen["logs_dir"] == logs_dir
    assert seen["results_root"] != results_root.resolve()


def test_publication_readiness_results_preserves_server_return_paths(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_results_preserve_paths",
    )
    raw_base = tmp_path / "returned_link" / ".." / "returned"
    results_dir = raw_base / "benchmark" / "results" / "full_benchmark"
    paper_dir = raw_base / "paper"
    logs_dir = raw_base / "logs"
    seen = {}

    monkeypatch.setattr(
        mod,
        "summarize_sweep_artifacts",
        lambda path: mod.GateResult("Full sweep results", "PASS", "ready", "No action."),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_validation",
        lambda: mod.GateResult("Case physics validation ledger", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_readiness",
        lambda: mod.GateResult("Case experiment readiness", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_smoke_pipeline",
        lambda results_root: mod.GateResult("Smoke pipeline", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_draft_gate",
        lambda paper_path: mod.GateResult("Pre-results paper draft", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_submission_gate",
        lambda paper_path, sweep_path: mod.GateResult("Final submission gate", "PASS", "ok", "none"),
    )

    def fake_server_return(sweep_gate, **kwargs):
        seen.update(kwargs)
        return mod.GateResult("Server return package", "PASS", "ok", "none")

    monkeypatch.setattr(mod, "summarize_server_return_package", fake_server_return)
    args = mod.parse_args([
        "--results-dir", str(results_dir),
        "--paper-dir", str(paper_dir),
        "--logs-dir", str(logs_dir),
    ])

    results = mod._publication_readiness_results(args)

    assert {item.name for item in results} >= {"Server return package", "Final submission gate"}
    assert seen["results_root"] == results_dir.parent
    assert seen["results_dir"] == results_dir
    assert seen["paper_dir"] == paper_dir
    assert seen["logs_dir"] == logs_dir
    assert seen["results_dir"] != results_dir.resolve()


def test_publication_readiness_uses_results_filled_stage_when_headline_exists(
    tmp_path, monkeypatch
):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_results_filled_stage",
    )
    results_dir = tmp_path / "benchmark" / "results" / "full_benchmark"
    results_dir.mkdir(parents=True)
    (results_dir / "results.csv").write_text("case_id\nexample\n", encoding="utf-8")
    paper_dir = tmp_path / "paper"
    logs_dir = tmp_path / "logs"

    monkeypatch.setattr(
        mod,
        "summarize_sweep_artifacts",
        lambda path: mod.GateResult("Full sweep results", "BLOCKED", "provenance", "repair"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_validation",
        lambda: mod.GateResult("Case physics validation ledger", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_evidence_tiers",
        lambda: mod.GateResult("Case evidence tiers", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_readiness",
        lambda: mod.GateResult("Case experiment readiness", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_smoke_pipeline",
        lambda results_root: mod.GateResult("Smoke pipeline", "WARN", "ok", "none"),
    )

    def unexpected_draft_gate(_paper_dir):
        raise AssertionError("pre-results draft gate must not run for a filled headline")

    monkeypatch.setattr(mod, "summarize_draft_gate", unexpected_draft_gate)
    monkeypatch.setattr(
        mod,
        "summarize_server_return_package",
        lambda sweep_gate, **kwargs: mod.GateResult(
            "Server return package", "DEFERRED", "upstream", "repair"
        ),
    )

    args = mod.parse_args(
        [
            "--results-dir",
            str(results_dir),
            "--paper-dir",
            str(paper_dir),
            "--logs-dir",
            str(logs_dir),
            "--no-submission-gate",
        ]
    )
    results = mod._publication_readiness_results(args)
    filled = next(item for item in results if item.name == "Results-filled manuscript")
    assert filled.status == "WARN"
    assert "1 data rows" in filled.evidence


def test_publication_readiness_surfaces_server_return_command_history_errors(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_server_return_command_history",
    )
    server = _load_module(
        "scripts/check_server_return_package.py",
        "publication_readiness_server_return_checker",
    )
    server.check_server_return_package = lambda **kwargs: {
        "manifest": [
            "pipeline_command_history.txt entries do not match RUN_MANIFEST experiment.command_entries",
            "pipeline_command_history.txt records a dry-run pipeline_invocation",
        ]
    }
    sys.modules["scripts.check_server_return_package"] = server
    results_root = tmp_path / "results"
    results_dir = results_root / "full_benchmark"
    paper_dir = tmp_path / "paper"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    paper_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    gate = mod.summarize_server_return_package(
        mod.GateResult("Full sweep results", "PASS", "ready", "No action."),
        results_root=results_root,
        results_dir=results_dir,
        paper_dir=paper_dir,
        logs_dir=logs_dir,
    )

    assert gate.status == "BLOCKED"
    assert "pipeline_command_history.txt entries do not match" in gate.evidence
    assert "dry-run pipeline_invocation" in gate.evidence


def test_publication_readiness_blocks_passing_sweep_without_manifest(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_missing_manifest",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    with (results_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("case_id\n")
        for i in range(rows):
            f.write(f"case_{i}\n")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "RUN_MANIFEST.json" in gate.evidence
    assert "manifest generation" in gate.next_action


def test_publication_readiness_blocks_empty_manifest(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_empty_manifest",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    with (results_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("case_id\n")
        for i in range(rows):
            f.write(f"case_{i}\n")
    (results_dir / "RUN_MANIFEST.json").write_text("", encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "empty" in gate.evidence
    assert "--require-complete-archive" in gate.next_action


def test_publication_readiness_blocks_malformed_manifest_json(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_bad_manifest_json",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    with (results_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("case_id\n")
        for i in range(rows):
            f.write(f"case_{i}\n")
    (results_dir / "RUN_MANIFEST.json").write_text("{not json\n", encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "could not parse" in gate.evidence
    assert "--require-complete-archive" in gate.next_action


def test_publication_readiness_blocks_manifest_missing_archive_structure(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_structure",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    with (results_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("case_id\n")
        for i in range(rows):
            f.write(f"case_{i}\n")
    (results_dir / "RUN_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "experiment section" in gate.evidence
    assert "results_summary section" in gate.evidence
    assert "artifacts.files" in gate.evidence


def test_publication_readiness_blocks_manifest_without_command_history(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_missing_commands",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(["", "   "]),
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "experiment.commands is empty" in gate.evidence


def test_publication_readiness_blocks_manifest_with_dry_run_command_history(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_dry_run_commands",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["commands"] = [
        "python scripts/run_pipeline.py --profile full --confirm-full-run --dry-run",
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "--dry-run" in gate.evidence


def test_publication_readiness_blocks_manifest_with_missing_command_file(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_missing_command_file",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["command_files_missing"] = ["logs/pipeline_command_history.txt"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "command_files_missing" in gate.evidence


def test_publication_readiness_blocks_manifest_with_malformed_command_file_fields(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_command_file_fields",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["command_files_requested"] = "logs/pipeline_command_history.txt"
    data["experiment"].pop("command_files_missing")
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "command_files_requested" in gate.evidence
    assert "command_files_missing" in gate.evidence


def test_publication_readiness_blocks_manifest_with_command_file_parse_errors(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_command_file_parse_errors",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, 34 * 20 * 4 * 30)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["command_file_parse_errors"] = [
        "logs/pipeline_command_history.txt: line 2 is not label<TAB>command"
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "command_file_parse_errors" in gate.evidence


def test_publication_readiness_blocks_manifest_with_case_resolution_error(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_case_resolution_error",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, 34 * 20 * 4 * 30)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["case_resolution_error"] = "ImportError: no doe_testbed"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "case_resolution_error" in gate.evidence


def test_publication_readiness_blocks_manifest_with_empty_command_file_request(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_empty_command_file_request",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["command_files_requested"] = []
    data["experiment"]["command_files_missing"] = []
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "command_files_requested is empty" in gate.evidence


def test_publication_readiness_blocks_manifest_with_bad_command_file_entries(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_bad_command_file_entries",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["command_files_requested"] = [
        "logs/pipeline_command_history.dry_run.txt",
        "../logs/pipeline_command_history.txt",
        r"C:\DOE-benchmark\logs\pipeline_command_history.txt",
        "",
        123,
    ]
    data["experiment"]["command_files_missing"] = []
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "blank or non-string" in gate.evidence
    assert "non-portable paths" in gate.evidence
    assert "dry-run command history" in gate.evidence


def test_publication_readiness_blocks_manifest_without_pipeline_invocation_entry(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_pipeline_invocation",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["command_entries"] = [{
        "label": "sweep",
        "command": "python benchmark/scripts/run_full_benchmark.py --pool nature",
    }]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "pipeline_invocation" in gate.evidence


def test_publication_readiness_blocks_low_level_pipeline_invocation(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_low_level_pipeline_invocation",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["commands"] = [
        "python benchmark/scripts/run_full_benchmark.py --pool nature",
    ]
    data["experiment"]["command_entries"] = [{
        "label": "pipeline_invocation",
        "command": "python benchmark/scripts/run_full_benchmark.py --pool nature",
    }]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "formal full-pipeline entrypoint" in gate.evidence


def test_publication_readiness_blocks_manifest_with_malformed_command_entries(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_malformed_command_entries",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["command_entries"] = [
        {"label": "pipeline_invocation", "command": ""},
        {"label": "", "command": "python scripts/run_pipeline.py --profile full --confirm-full-run"},
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "malformed entries" in gate.evidence
    assert "do not match experiment.commands" in gate.evidence


def test_publication_readiness_blocks_manifest_with_duplicate_pipeline_invocation_entries(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_duplicate_pipeline_invocation_entries",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    manifest = results_dir / "RUN_MANIFEST.json"
    _write_minimal_publication_manifest(manifest, results_csv)
    command = "python scripts/run_pipeline.py --profile full --confirm-full-run"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["experiment"]["commands"] = [command, command]
    data["experiment"]["command_entries"] = [
        {"label": "pipeline_invocation", "command": command},
        {"label": "pipeline_invocation", "command": command},
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "multiple pipeline_invocation" in gate.evidence


def test_publication_readiness_blocks_manifest_without_manifest_command(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_missing_manifest_command",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(["python scripts/run_pipeline.py"]),
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "manifest_command is missing" in gate.evidence


def test_publication_readiness_blocks_source_dirty_manifest(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_source_dirty_manifest",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_manifest_with_publication_python_environment(
        results_dir / "RUN_MANIFEST.json",
        {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": {
                "commit": "abcdef123456",
                "dirty": True,
                "source_dirty": True,
                "source_dirty_entries": [" M scripts/run_pipeline.py"],
            },
            "experiment": _publication_experiment(["python scripts/run_pipeline.py"]),
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        },
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "dirty source/config state" in gate.evidence


def test_publication_readiness_blocks_manifest_without_git_commit(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_missing_git_commit",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    git_info = _publication_clean_git()
    git_info["commit"] = ""
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": git_info,
            "experiment": _publication_experiment(["python scripts/run_pipeline.py"]),
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "git.commit is missing" in gate.evidence


def test_publication_readiness_blocks_legacy_dirty_manifest(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_legacy_dirty_manifest",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": True,
                "status_porcelain": [" M scripts/run_pipeline.py"],
            },
            "experiment": _publication_experiment(["python scripts/run_pipeline.py"]),
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "dirty git worktree" in gate.evidence


def test_publication_readiness_blocks_manifest_without_results_artifact(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_missing_results_artifact",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    _write_results_rows(results_dir / "results.csv", rows)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(["python scripts/run_pipeline.py"]),
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {"files": [{"path": "report/REPORT.md"}]},
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "does not archive results.csv" in gate.evidence


def test_publication_readiness_blocks_manifest_row_count_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_row_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    _write_results_rows(results_dir / "results.csv", rows)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(["python scripts/run_pipeline.py"]),
            "results_summary": {"exists": True, "n_rows": rows - 1},
            "artifacts": {"files": [{"path": "results.csv"}]},
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "n_rows does not match" in gate.evidence


def test_publication_readiness_blocks_results_artifact_size_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_size_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(["python scripts/run_pipeline.py"]),
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size + 1,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "size_bytes does not match" in gate.evidence


def test_publication_readiness_blocks_results_artifact_sha_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_manifest_sha_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(["python scripts/run_pipeline.py"]),
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": "0" * 64,
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "sha256 does not match" in gate.evidence


def test_publication_readiness_blocks_malformed_artifact_file_entries(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_malformed_artifact_entries",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["files"].extend(["not-an-object", {}, {"path": "results.csv"}])
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "contains a non-object entry" in gate.evidence
    assert "entry without path" in gate.evidence
    assert "duplicate paths" in gate.evidence


def test_publication_readiness_blocks_missing_sweep_report_artifact(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_missing_sweep_report_artifact",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["files"] = [
        item for item in manifest["artifacts"]["files"]
        if item.get("path") != "sweep_integrity_report.md"
    ]
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "does not archive sweep_integrity_report.md" in gate.evidence


def test_publication_readiness_blocks_sweep_report_artifact_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_report_artifact_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["artifacts"]["files"]:
        if item.get("path") == "sweep_integrity_report.md":
            item["size_bytes"] = (results_dir / "sweep_integrity_report.md").stat().st_size + 1
            item["sha256"] = "0" * 64
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "sweep_integrity_report.md artifact size_bytes does not match" in gate.evidence
    assert "sweep_integrity_report.md artifact sha256 does not match" in gate.evidence


def test_publication_readiness_revalidates_current_results_csv_contract(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_results_contract_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    lines = results_csv.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[1]
    results_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "row-level contract failed" in gate.evidence
    assert "duplicate" in gate.evidence and "cells" in gate.evidence


def test_publication_readiness_blocks_preregistered_protocol_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_protocol_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    experiment = _publication_experiment(["python scripts/run_pipeline.py"])
    experiment["pool"] = "default"
    experiment["budgets"] = [5, 10, 20]
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": experiment,
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "experiment.pool deviates" in gate.evidence
    assert "experiment.budgets deviates" in gate.evidence


def test_publication_readiness_blocks_declared_case_count_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_case_count_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    experiment = _publication_experiment(["python scripts/run_pipeline.py"])
    experiment["n_cases_declared"] = 33
    experiment["case_ids_declared"] = experiment["case_ids_declared"][:-1]
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": experiment,
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "experiment.n_cases_declared deviates" in gate.evidence
    assert "experiment.case_ids_declared count deviates" in gate.evidence


def test_publication_readiness_blocks_declared_case_identity_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_case_identity_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    experiment = _publication_experiment(["python scripts/run_pipeline.py"])
    case_ids = list(experiment["case_ids_declared"])
    case_ids[-1] = "not_a_registered_case"
    experiment["case_ids_declared"] = case_ids
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": experiment,
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "case_ids_declared does not match" in gate.evidence
    assert "not_a_registered_case" in gate.evidence


def test_publication_readiness_blocks_declared_method_identity_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_method_identity_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    experiment = _publication_experiment(["python scripts/run_pipeline.py"])
    methods = list(experiment["methods_declared"])
    methods[-1] = "not_a_registered_method"
    experiment["methods_declared"] = methods
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": experiment,
            "results_summary": {"exists": True, "n_rows": rows},
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "methods_declared does not match" in gate.evidence
    assert "not_a_registered_method" in gate.evidence


def test_publication_readiness_blocks_observed_grid_summary_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_summary_grid_drift",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    (results_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(["python scripts/run_pipeline.py"]),
            "results_summary": {
                "exists": True,
                "n_rows": rows,
                "status_counts": {"ok": rows - 1, "rerun": 1},
                "n_cases_observed": 34,
                "case_ids_observed": PREREG_CASE_IDS,
                "n_methods_observed": 20,
                "methods_observed": PREREG_METHOD_NAMES[:-1] + ["not_a_registered_method"],
                "budgets_observed": [5, 10, 20],
                "repeats_observed": list(range(29)),
                "n_invalid_budget_mult_values": 1,
                "n_invalid_repeat_values": 1,
                "n_missing_protocol_key_values": 1,
            },
            "artifacts": {
                "files": [{
                    "path": "results.csv",
                    "size_bytes": results_csv.stat().st_size,
                    "sha256": _file_sha256(results_csv),
                }]
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "method set drifts" in gate.evidence
    assert "budget set drifts" in gate.evidence
    assert "repeat set drifts" in gate.evidence
    assert "unknown statuses" in gate.evidence
    assert "invalid budget_mult" in gate.evidence
    assert "invalid repeat" in gate.evidence
    assert "missing protocol key values" in gate.evidence


def test_publication_readiness_blocks_malformed_invalid_grid_counts(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_malformed_invalid_grid_counts",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["results_summary"]["n_invalid_budget_mult_values"] = "not-an-int"
    manifest["results_summary"]["n_invalid_repeat_values"] = -1
    manifest["results_summary"]["n_missing_protocol_key_values"] = "not-an-int"
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "n_invalid_budget_mult_values must be a non-negative integer" in gate.evidence
    assert "n_invalid_repeat_values must be a non-negative integer" in gate.evidence
    assert "n_missing_protocol_key_values must be a non-negative integer" in gate.evidence


def test_publication_readiness_blocks_malformed_status_counts(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_malformed_status_counts",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["results_summary"]["status_counts"] = {
        "ok": rows,
        "timeout": -1,
        "failed": "not-an-int",
    }
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "status_counts has non-negative integer count required for 'timeout'" in gate.evidence
    assert "status_counts has non-negative integer count required for 'failed'" in gate.evidence


def test_publication_readiness_blocks_missing_status_counts(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_missing_status_counts",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["results_summary"]["status_counts"] = ["ok", rows]
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "results_summary.status_counts is missing or malformed" in gate.evidence


def test_publication_readiness_blocks_missing_observed_sets(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_missing_observed_sets",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    summary = manifest["results_summary"]
    summary.pop("case_ids_observed")
    summary["methods_observed"] = "not-a-list"
    summary["budgets_observed"] = None
    summary["repeats_observed"] = "0..29"
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "case observed set is missing or malformed" in gate.evidence
    assert "method observed set is missing or malformed" in gate.evidence
    assert "budget observed set is missing or malformed" in gate.evidence
    assert "repeat observed set is missing or malformed" in gate.evidence


def test_publication_readiness_blocks_duplicate_observed_sets_and_count_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_duplicate_observed_sets",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    summary = manifest["results_summary"]
    summary["case_ids_observed"] = PREREG_CASE_IDS[:-1] + [PREREG_CASE_IDS[0]]
    summary["n_cases_observed"] = 35
    summary["methods_observed"] = PREREG_METHOD_NAMES[:-1] + [PREREG_METHOD_NAMES[0]]
    summary["n_methods_observed"] = 21
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "case observed set contains duplicates" in gate.evidence
    assert "method observed set contains duplicates" in gate.evidence
    assert "case observed count does not match" in gate.evidence
    assert "method observed count does not match" in gate.evidence
    assert "case count drifts" in gate.evidence
    assert "method count drifts" in gate.evidence


def test_publication_readiness_blocks_missing_source_fingerprints(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_missing_source_fingerprints",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest.pop("case_sources")
    manifest.pop("method_sources")
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "case source provenance failed" in gate.evidence
    assert "method source provenance failed" in gate.evidence


def test_publication_readiness_blocks_stale_source_fingerprints(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_full_sweep_stale_source_fingerprints",
    )
    results_dir = tmp_path / "full_benchmark"
    results_dir.mkdir()
    rows = 34 * 20 * 4 * 30
    results_csv = results_dir / "results.csv"
    _write_results_rows(results_csv, rows)
    _write_minimal_publication_manifest(results_dir / "RUN_MANIFEST.json", results_csv)
    manifest = json.loads((results_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["case_sources"]["cases"][0]["sha256"] = "0" * 64
    manifest["method_sources"]["methods"][0]["method_source_sha256"] = "0" * 64
    (results_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results_dir / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_sweep_artifacts(results_dir)

    assert gate.status == "BLOCKED"
    assert "case source hash mismatch" in gate.evidence
    assert "method source hash mismatch" in gate.evidence


def test_publication_readiness_report_allows_warning_without_blocking():
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_warn_only",
    )

    report = mod.render_report([
        mod.GateResult("Example warning", "WARN", "usable", "review"),
        mod.GateResult("Example pass", "PASS", "ok", "none"),
    ])

    assert "Overall status: **WARN**" in report
    assert "Overall status: **BLOCKED**" not in report


def test_publication_readiness_launch_preflight_excludes_result_dependent_gates(
    tmp_path,
    monkeypatch,
):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_launch_preflight",
    )
    calls: list[str] = []

    def pass_gate(name: str):
        def _inner(*args, **kwargs):
            calls.append(name)
            return mod.GateResult(name, "PASS", "ok", "none")

        return _inner

    monkeypatch.setattr(mod, "summarize_case_validation", pass_gate("Case physics validation ledger"))
    monkeypatch.setattr(mod, "summarize_case_evidence_tiers", pass_gate("Case evidence tiers"))
    monkeypatch.setattr(mod, "summarize_case_readiness", pass_gate("Case experiment readiness"))
    monkeypatch.setattr(mod, "summarize_benchmark_protocol_audit", pass_gate("Benchmark protocol audit"))
    monkeypatch.setattr(mod, "summarize_smoke_pipeline", pass_gate("Smoke pipeline"))
    monkeypatch.setattr(mod, "summarize_draft_gate", pass_gate("Pre-results paper draft"))
    monkeypatch.setattr(
        mod,
        "summarize_sweep_artifacts",
        lambda *args, **kwargs: pytest.fail("launch preflight must not inspect full sweep"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_server_return_package",
        lambda *args, **kwargs: pytest.fail("launch preflight must not inspect server return"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_submission_gate",
        lambda *args, **kwargs: pytest.fail("launch preflight must not inspect submission gate"),
    )

    rc = mod.main(["--launch-preflight", "--output", str(tmp_path / "preflight.md")])

    assert rc == 0
    assert calls == [
        "Case physics validation ledger",
        "Case evidence tiers",
        "Case experiment readiness",
        "Benchmark protocol audit",
        "Smoke pipeline",
        "Pre-results paper draft",
    ]


def test_publication_readiness_launch_preflight_uses_distinct_default_report(
    tmp_path,
    monkeypatch,
):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_launch_preflight_output",
    )
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "DEFAULT_OUTPUT", tmp_path / "PUBLICATION_READINESS_AUDIT.md")
    monkeypatch.setattr(
        mod,
        "DEFAULT_LAUNCH_OUTPUT",
        tmp_path / "PUBLICATION_LAUNCH_PREFLIGHT.md",
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_validation",
        lambda: mod.GateResult("Case physics validation ledger", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_evidence_tiers",
        lambda: mod.GateResult("Case evidence tiers", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_readiness",
        lambda: mod.GateResult("Case experiment readiness", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_benchmark_protocol_audit",
        lambda: mod.GateResult("Benchmark protocol audit", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_smoke_pipeline",
        lambda results_root: mod.GateResult("Smoke pipeline", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_draft_gate",
        lambda paper_dir: mod.GateResult("Pre-results paper draft", "PASS", "ok", "none"),
    )

    rc = mod.main(["--launch-preflight"])

    assert rc == 0
    assert not (tmp_path / "PUBLICATION_READINESS_AUDIT.md").exists()
    launch_report = tmp_path / "PUBLICATION_LAUNCH_PREFLIGHT.md"
    assert launch_report.exists()
    assert "# Publication Launch Preflight" in launch_report.read_text(encoding="utf-8")


def test_publication_readiness_launch_preflight_fails_on_warning(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_launch_preflight_warn",
    )

    monkeypatch.setattr(
        mod,
        "summarize_case_validation",
        lambda: mod.GateResult("Case physics validation ledger", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_evidence_tiers",
        lambda: mod.GateResult("Case evidence tiers", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_case_readiness",
        lambda: mod.GateResult("Case experiment readiness", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_benchmark_protocol_audit",
        lambda: mod.GateResult("Benchmark protocol audit", "PASS", "ok", "none"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_smoke_pipeline",
        lambda results_root: mod.GateResult("Smoke pipeline", "WARN", "stale", "rerun"),
    )
    monkeypatch.setattr(
        mod,
        "summarize_draft_gate",
        lambda paper_dir: mod.GateResult("Pre-results paper draft", "PASS", "ok", "none"),
    )

    rc = mod.main(["--launch-preflight", "--output", str(tmp_path / "preflight.md")])

    assert rc == 1


def test_publication_readiness_benchmark_protocol_audit_gate_blocks_missing_controls(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_benchmark_protocol_audit",
    )
    audit = tmp_path / "EXPERIMENT_BENCHMARK_AUDIT.md"
    audit.write_text(
        "# Experiment Benchmark Audit\n\nCurrent check date: 2026-06-07\n",
        encoding="utf-8",
    )

    gate = mod.summarize_benchmark_protocol_audit(audit)

    assert gate.name == "Benchmark protocol audit"
    assert gate.status == "BLOCKED"
    assert "External Source Verification Matrix" in gate.evidence
    assert "runtime" in gate.next_action


def test_publication_readiness_benchmark_protocol_audit_gate_accepts_current_doc():
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_benchmark_protocol_audit_current",
    )

    gate = mod.summarize_benchmark_protocol_audit()

    assert gate.name == "Benchmark protocol audit"
    assert gate.status == "PASS"
    assert "External-source matrix" in gate.evidence
    assert "FAIR data package" in gate.evidence


def test_publication_readiness_benchmark_protocol_audit_requires_strict_launch_markers(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_benchmark_protocol_audit_strict_launch",
    )
    current = ROOT / "docs" / "EXPERIMENT_BENCHMARK_AUDIT.md"
    audit = tmp_path / "EXPERIMENT_BENCHMARK_AUDIT.md"
    text = current.read_text(encoding="utf-8")
    text = text.replace(
        "python scripts/check_server_launch_ready.py --require-clean-sweep-dir",
        "python scripts/check_server_launch_ready.py",
    )
    text = text.replace("DOE_BENCHMARK_ALLOW_STALE_DERIVED=1", "")
    audit.write_text(text, encoding="utf-8")

    gate = mod.summarize_benchmark_protocol_audit(audit)

    assert gate.status == "BLOCKED"
    assert "--require-clean-sweep-dir" in gate.evidence


def _write_case_evidence_tier_artifacts(
    tmp_path: Path,
    *,
    rows: list[dict[str, str]] | None = None,
    audit_line: str | None = None,
) -> tuple[Path, Path, Path]:
    if rows is None:
        rows = [
            {
                "case_id": "validated_case",
                "validation_note_path": "案例正确性校验/example",
                "external_validation_scope": "docx plus references",
                "targeted_physics_test": "tests/test_validated_case_physics.py",
                "risk_level": "P1",
                "final_disposition": "accepted after resweep",
            },
            {
                "case_id": "p2_case",
                "validation_note_path": "NONE",
                "external_validation_scope": "no prior validation-note folder found",
                "targeted_physics_test": "tests/test_p2_case_physics.py",
                "risk_level": "P2",
                "final_disposition": "plausibility only",
            },
        ]
    status_csv = tmp_path / "case_validation_status.csv"
    fieldnames = [
        "case_id",
        "validation_note_path",
        "external_validation_scope",
        "targeted_physics_test",
        "risk_level",
        "final_disposition",
    ]
    with status_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = tmp_path / "CASE_VALIDATION_SUMMARY.md"
    summary.write_text(
        "\n".join([
            "# Case Validation Summary",
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
            "| `targeted_plausibility_only` | 1 | do not claim strong external case validation. |",
        ]),
        encoding="utf-8",
    )

    audit = tmp_path / "CASE_WEB_VALIDATION_AUDIT.md"
    if audit_line is None:
        audit_line = (
            "| `p2_case` | Evidence tier remains P2. Reference checked. | "
            "plausible trend audit | Keep P2 plausibility-only. |"
        )
    audit.write_text(
        "\n".join([
            "# Case Web Validation Audit",
            "",
            "| Case | Evidence tier | Web/literature check | Decision |",
            "|---|---|---|---|",
            audit_line,
        ]),
        encoding="utf-8",
    )
    return status_csv, summary, audit


def test_publication_readiness_case_evidence_tiers_gate_accepts_current_docs():
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_evidence_current",
    )

    gate = mod.summarize_case_evidence_tiers()

    assert gate.name == "Case evidence tiers"
    assert gate.status == "PASS"
    assert "P0: 0, P1: 17, P2: 17" in gate.evidence
    assert "targeted_plausibility_only" in gate.evidence


def test_case_web_validation_audit_records_geomechanics_source_snapshot():
    audit_text = (ROOT / "docs" / "CASE_WEB_VALIDATION_AUDIT.md").read_text(
        encoding="utf-8"
    )

    assert "## 2026-06-08 Geomechanics Source Verification Snapshot" in audit_text
    assert "USACE EM 1110-2-1901" in audit_text
    assert "FERC Engineering Guidelines Chapter 3" in audit_text
    assert "USBR Design Standards No. 13" in audit_text
    assert "USGS Circular 1224" in audit_text
    assert "OPM Flow overview/manual pages" in audit_text
    assert "boundary/context evidence, not external case validation" in audit_text
    assert "`dam_seepage_6d`" in audit_text
    assert "`groundwater_contamination_6d`, `oil_reservoir_9d`" in audit_text
    assert "`oil_reservoir_9d`" in audit_text
    assert "Evidence tier remains P2 plausibility-only" in audit_text
    assert "OPM/MRST/SPE10 are scope/context anchors only" in audit_text
    assert "case-specific validation-note folder" in audit_text


def test_case_web_validation_audit_records_analytical_control_source_snapshot():
    audit_text = (ROOT / "docs" / "CASE_WEB_VALIDATION_AUDIT.md").read_text(
        encoding="utf-8"
    )

    assert "## 2026-06-08 Analytical-Control Source Verification Snapshot" in audit_text
    assert "MIT thermal-resistance-circuit notes" in audit_text
    assert "NPTEL composite-wall conduction material" in audit_text
    assert "NASA Passive Thermal Control Engineering Guidebook" in audit_text
    assert "ASME BPVC Section VIII" in audit_text
    assert "Itasca FLAC3D one-dimensional consolidation example" in audit_text
    assert "`multilayer_wall_4d`, `multilayer_wall_8d`, `multilayer_wall_16d`" in audit_text
    assert "`pressure_vessel_4d`" in audit_text
    assert "`soil_consolidation_5d`" in audit_text
    assert "Evidence tier remains P2 analytic-control" in audit_text
    assert "not external building-physics validation" in audit_text


def test_publication_readiness_case_evidence_tiers_gate_blocks_missing_p2_marker(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_evidence_p2_marker",
    )
    status_csv, summary, audit = _write_case_evidence_tier_artifacts(
        tmp_path,
        audit_line=(
            "| `p2_case` | Reference checked against public formulas. | "
            "plausible trend audit | Keep P2 plausibility-only. |"
        ),
    )

    gate = mod.summarize_case_evidence_tiers(
        status_csv,
        summary,
        audit,
        expected_n_cases=2,
    )

    assert gate.status == "BLOCKED"
    assert "missing `Evidence tier remains P2` marker" in gate.evidence


def test_publication_readiness_case_evidence_tiers_gate_blocks_p0_rows(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_evidence_p0",
    )
    rows = [
        {
            "case_id": "bad_case",
            "validation_note_path": "NONE",
            "external_validation_scope": "no prior validation-note folder found",
            "targeted_physics_test": "tests/test_bad_case_physics.py",
            "risk_level": "P0",
            "final_disposition": "unresolved",
        },
        {
            "case_id": "p2_case",
            "validation_note_path": "NONE",
            "external_validation_scope": "no prior validation-note folder found",
            "targeted_physics_test": "tests/test_p2_case_physics.py",
            "risk_level": "P2",
            "final_disposition": "plausibility only",
        },
    ]
    status_csv, summary, audit = _write_case_evidence_tier_artifacts(tmp_path, rows=rows)

    gate = mod.summarize_case_evidence_tiers(
        status_csv,
        summary,
        audit,
        expected_n_cases=2,
    )

    assert gate.status == "BLOCKED"
    assert "P0/unaccepted case rows remain" in gate.evidence
    assert "bad_case" in gate.evidence


def test_publication_readiness_case_evidence_tiers_gate_blocks_missing_targeted_tests(
    tmp_path,
):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_evidence_targeted",
    )
    rows = [
        {
            "case_id": "validated_case",
            "validation_note_path": "案例正确性校验/example",
            "external_validation_scope": "docx plus references",
            "targeted_physics_test": "NONE",
            "risk_level": "P1",
            "final_disposition": "accepted after resweep",
        },
        {
            "case_id": "p2_case",
            "validation_note_path": "NONE",
            "external_validation_scope": "no prior validation-note folder found",
            "targeted_physics_test": "tests/test_p2_case_physics.py",
            "risk_level": "P2",
            "final_disposition": "plausibility only",
        },
    ]
    status_csv, summary, audit = _write_case_evidence_tier_artifacts(tmp_path, rows=rows)

    gate = mod.summarize_case_evidence_tiers(
        status_csv,
        summary,
        audit,
        expected_n_cases=2,
    )

    assert gate.status == "BLOCKED"
    assert "missing targeted physics tests" in gate.evidence
    assert "validated_case" in gate.evidence


def test_publication_readiness_report_write_is_atomic_on_replace_failure(
    tmp_path, monkeypatch
):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_atomic_report",
    )
    output = tmp_path / "PUBLICATION_READINESS_AUDIT.md"
    output.write_text("old readiness report\n", encoding="utf-8")

    def fail_move(src, dst):
        raise RuntimeError(f"cannot replace {dst}")

    monkeypatch.setattr(mod.shutil, "move", fail_move)

    with pytest.raises(RuntimeError, match="cannot replace"):
        mod.write_report_atomic("new readiness report\n", output)

    assert output.read_text(encoding="utf-8") == "old readiness report\n"
    assert not list(tmp_path.glob(".PUBLICATION_READINESS_AUDIT.md.tmp.*"))


def test_publication_readiness_defers_submission_gate_until_sweep_passes(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_defer_submission",
    )
    sweep_gate = mod.summarize_sweep_artifacts(tmp_path / "full_benchmark")

    gate = mod.defer_submission_gate(sweep_gate)
    report = mod.render_report([sweep_gate, gate])

    assert gate.status == "DEFERRED"
    assert "Strict submission-ready checks depend on a passed full-sweep gate" in gate.evidence
    assert "| Final submission gate | DEFERRED |" in report
    assert "Overall status: **BLOCKED**" in report


def test_publication_readiness_case_readiness_blocks_timing_warning(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_warning",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    _write_case_readiness_rows(
        audit,
        [
            *(_case_readiness_row(case_id) for case_id in PREREG_CASE_IDS[:-1]),
            _case_readiness_row(PREREG_CASE_IDS[-1], status="timing_warning"),
        ],
    )

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "timing_warning" in gate.evidence
    assert "timing metadata drift" in gate.next_action


def test_publication_readiness_case_readiness_rejects_fatal_status(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_fatal",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    _write_case_readiness_rows(
        audit,
        [
            *(_case_readiness_row(case_id) for case_id in PREREG_CASE_IDS[:-1]),
            _case_readiness_row(PREREG_CASE_IDS[-1], status="nonfinite", n_nan=1),
        ],
    )

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "nonfinite" in gate.evidence


def test_publication_readiness_case_readiness_rejects_old_schema(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_old_schema",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    with audit.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "status"])
        writer.writeheader()
        for case_id in PREREG_CASE_IDS:
            writer.writerow({"case_id": case_id, "status": "ok"})

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "Missing required columns" in gate.evidence
    assert "determinism_n_mismatch" in gate.evidence


def test_publication_readiness_case_readiness_rejects_unreadable_csv(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_unreadable_csv",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    audit.write_bytes(b"\xff\xfe\x00\x00not utf-8\n")

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "could not be read as UTF-8 CSV" in gate.evidence
    assert "audit_case_experiment_readiness.py" in gate.next_action


def test_publication_readiness_case_readiness_requires_header_row(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_missing_header",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    audit.write_text("", encoding="utf-8")

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "missing a CSV header row" in gate.evidence


def test_publication_readiness_case_readiness_rejects_invalid_header_contract(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_bad_header_contract",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    audit.write_text(
        "case_id,,status,case_id\n"
        "airfoil_stall_2d,heat_transfer,ok,airfoil_stall_2d\n",
        encoding="utf-8",
    )

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "invalid CSV header" in gate.evidence
    assert "blank header" in gate.evidence
    assert "duplicate header" in gate.evidence


def test_publication_readiness_case_readiness_rejects_ragged_rows(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_ragged_rows",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    audit.write_text(
        "case_id,domain,pde_type,response_topology,n_topology_tags,governing_equation,"
        "n_physics_references,status,n_nan,n_inf,input_mutation_max_abs_delta,"
        "input_mutation_n_changed,determinism_max_abs_delta,determinism_n_mismatch,"
        "n_expected_range_checked,n_expected_range_violations,expected_range_max_abs_violation\n"
        "airfoil_stall_2d,aerodynamics,algebraic,smooth_monotone,1,test equation,1,ok,0,0,0,0,0,0,1,0,0,EXTRA\n",
        encoding="utf-8",
    )

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "ragged CSV row" in gate.evidence
    assert "line(s) [2]" in gate.evidence


def test_publication_readiness_case_readiness_requires_metadata_columns(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_metadata_columns",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    old_columns = [
        col for col in CASE_READINESS_TEST_COLUMNS
        if col not in {
            "domain",
            "pde_type",
            "response_topology",
            "n_topology_tags",
            "governing_equation",
            "n_physics_references",
        }
    ]
    with audit.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_columns)
        writer.writeheader()
        for case_id in PREREG_CASE_IDS:
            row = _case_readiness_row(case_id)
            writer.writerow({col: row[col] for col in old_columns})

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "Missing required columns" in gate.evidence
    assert "response_topology" in gate.evidence
    assert "n_physics_references" in gate.evidence


def test_publication_readiness_case_readiness_rejects_empty_metadata_values(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_metadata_values",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    _write_case_readiness_rows(
        audit,
        [
            *(_case_readiness_row(case_id) for case_id in PREREG_CASE_IDS[:-1]),
            _case_readiness_row(
                PREREG_CASE_IDS[-1],
                response_topology="",
                n_topology_tags=1,
                governing_equation="",
                n_physics_references=0,
            ),
        ],
    )

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "Case metadata readiness errors" in gate.evidence
    assert "response_topology is empty" in gate.evidence
    assert "n_physics_references=0" in gate.evidence
    assert "reference metadata drift" in gate.next_action


def test_publication_readiness_case_readiness_rejects_registry_metadata_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_registry_metadata_drift",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    rows = [_case_readiness_row(case_id) for case_id in PREREG_CASE_IDS]
    drift_case = rows[-1]["case_id"]
    meta = db.get_case_meta(str(drift_case))
    rows[-1].update({
        "domain": meta.domain,
        "pde_type": meta.pde_type,
        "response_topology": "smooth_monotone",
        "n_topology_tags": 1,
        "governing_equation": "stale equation",
        "n_physics_references": len(meta.physics_references),
    })
    _write_case_readiness_rows(audit, rows)

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "Case metadata drift vs registry" in gate.evidence
    assert str(drift_case) in gate.evidence
    assert "response_topology" in gate.evidence or "governing_equation" in gate.evidence


def test_submission_ready_accepts_case_experiment_readiness_ledger(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_case_readiness_ok",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    _write_case_readiness_rows(
        audit,
        [_case_readiness_row(case_id) for case_id in PREREG_CASE_IDS],
    )

    assert mod.check_case_experiment_readiness(audit) == []


def test_submission_ready_rejects_case_experiment_readiness_drift(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_case_readiness_bad",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    _write_case_readiness_rows(
        audit,
        [
            *(_case_readiness_row(case_id) for case_id in PREREG_CASE_IDS[:-1]),
            _case_readiness_row(
                PREREG_CASE_IDS[-1],
                response_topology="",
                n_topology_tags=0,
            ),
        ],
    )

    errors = mod.check_case_experiment_readiness(audit)

    assert any("case experiment readiness is not submission-ready" in err for err in errors)
    assert any("Case metadata readiness errors" in err for err in errors)


def test_publication_readiness_case_readiness_rejects_determinism_mismatch(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_determinism",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    _write_case_readiness_rows(
        audit,
        [
            *(_case_readiness_row(case_id) for case_id in PREREG_CASE_IDS[:-1]),
            _case_readiness_row(
                PREREG_CASE_IDS[-1],
                determinism_max_abs_delta=1.0,
                determinism_n_mismatch=4,
            ),
        ],
    )

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert f"{PREREG_CASE_IDS[-1]}: determinism_n_mismatch=4" in gate.evidence


def test_publication_readiness_case_readiness_rejects_input_mutation(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_input_mutation",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    _write_case_readiness_rows(
        audit,
        [
            _case_readiness_row(case_id)
            for case_id in PREREG_CASE_IDS[:-1]
        ] + [
            _case_readiness_row(
                PREREG_CASE_IDS[-1],
                input_mutation_max_abs_delta=0.25,
                input_mutation_n_changed=3,
            )
        ],
    )

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert f"{PREREG_CASE_IDS[-1]}: input_mutation_n_changed=3" in gate.evidence


def test_publication_readiness_case_readiness_rejects_expected_range_violation(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_expected_range",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    _write_case_readiness_rows(
        audit,
        [
            *(_case_readiness_row(case_id) for case_id in PREREG_CASE_IDS[:-1]),
            _case_readiness_row(
                PREREG_CASE_IDS[-1],
                n_expected_range_checked=1,
                n_expected_range_violations=1,
                expected_range_max_abs_violation=0.5,
            ),
        ],
    )

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert f"{PREREG_CASE_IDS[-1]}: n_expected_range_violations=1" in gate.evidence
    assert "expected_range metadata drift" in gate.next_action


def test_publication_readiness_case_readiness_rejects_case_id_drift(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_id_drift",
    )
    audit = tmp_path / "case_experiment_readiness.csv"
    _write_case_readiness_rows(
        audit,
        [
            *(_case_readiness_row(case_id) for case_id in PREREG_CASE_IDS[:-1]),
            _case_readiness_row("unknown_case"),
        ],
    )

    gate = mod.summarize_case_readiness(audit)

    assert gate.status == "BLOCKED"
    assert "Case coverage drift" in gate.evidence
    assert "unknown_case" in gate.evidence
    assert PREREG_CASE_IDS[-1] in gate.evidence


@requires_validation_notes
def test_publication_readiness_case_validation_blocks_stale_source_hash(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_validation_stale_source",
    )
    status = tmp_path / "case_validation_status.csv"
    sources = tmp_path / "case_validation_sources.csv"
    note_sources = tmp_path / "case_validation_note_sources.csv"
    summary = tmp_path / "CASE_VALIDATION_SUMMARY.md"
    _write_case_validation_status_copy(status)
    _write_case_validation_sources_copy(sources)
    _write_case_validation_note_sources_copy(note_sources)
    _write_case_validation_summary_copy(summary)

    with sources.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    rows[0]["case_source_sha256"] = "0" * 64
    with sources.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    gate = mod.summarize_case_validation(status, sources, note_sources, summary)

    assert gate.status == "BLOCKED"
    assert "Strict case-validation errors" in gate.evidence
    assert "case validation source hash mismatch" in gate.evidence


@requires_validation_notes
def test_publication_readiness_case_validation_blocks_stale_note_source_hash(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_validation_stale_note_source",
    )
    status = tmp_path / "case_validation_status.csv"
    sources = tmp_path / "case_validation_sources.csv"
    note_sources = tmp_path / "case_validation_note_sources.csv"
    summary = tmp_path / "CASE_VALIDATION_SUMMARY.md"
    _write_case_validation_status_copy(status)
    _write_case_validation_sources_copy(sources)
    _write_case_validation_note_sources_copy(note_sources)
    _write_case_validation_summary_copy(summary)

    with note_sources.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    rows[0]["sha256"] = "0" * 64
    with note_sources.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    gate = mod.summarize_case_validation(status, sources, note_sources, summary)

    assert gate.status == "BLOCKED"
    assert "Strict case-validation errors" in gate.evidence
    assert "case validation note-source hash mismatch" in gate.evidence


@requires_validation_notes
def test_publication_readiness_case_validation_blocks_stale_summary(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_case_validation_stale_summary",
    )
    status = tmp_path / "case_validation_status.csv"
    sources = tmp_path / "case_validation_sources.csv"
    note_sources = tmp_path / "case_validation_note_sources.csv"
    summary = tmp_path / "CASE_VALIDATION_SUMMARY.md"
    _write_case_validation_status_copy(status)
    _write_case_validation_sources_copy(sources)
    _write_case_validation_note_sources_copy(note_sources)
    _write_case_validation_summary_copy(summary)
    summary.write_text(summary.read_text(encoding="utf-8") + "\nmanual edit\n", encoding="utf-8")

    gate = mod.summarize_case_validation(status, sources, note_sources, summary)

    assert gate.status == "BLOCKED"
    assert "case validation summary is stale" in gate.evidence


def test_publication_readiness_smoke_pipeline_accepts_latest_pass(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_smoke_pass",
    )
    monkeypatch.setattr(mod, "_rerun_smoke_integrity", lambda results_csv: (0, "PASS"))
    older = tmp_path / "_smoke_old"
    latest = tmp_path / "_smoke_new"
    for directory in (older, latest):
        directory.mkdir()
        (directory / "results.csv").write_text("case_id\nc0\n", encoding="utf-8")
    (older / "sweep_integrity_report.md").write_text("old report\n", encoding="utf-8")
    (latest / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )
    (latest / "report" / "figs").mkdir(parents=True)
    (latest / "report" / "REPORT.md").write_text("report\n", encoding="utf-8")
    (latest / "report" / "figs" / "fig.png").write_bytes(b"png")

    gate = mod.summarize_smoke_pipeline(tmp_path)

    assert gate.status == "PASS"
    assert "_smoke_new" in gate.evidence
    assert "REPORT.md" in gate.evidence


def test_publication_readiness_smoke_pipeline_rejects_stale_pass_report(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_smoke_stale_pass",
    )
    smoke = tmp_path / "_smoke"
    smoke.mkdir()
    (smoke / "results.csv").write_text("case_id\nc0\n", encoding="utf-8")
    (smoke / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )
    (smoke / "report" / "figs").mkdir(parents=True)
    (smoke / "report" / "REPORT.md").write_text("report\n", encoding="utf-8")
    (smoke / "report" / "figs" / "fig.png").write_bytes(b"png")

    gate = mod.summarize_smoke_pipeline(tmp_path)

    assert gate.status == "WARN"
    assert "fresh smoke audit failed" in gate.evidence


def test_publication_readiness_smoke_fresh_audit_uses_profile_protocol(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_smoke_profile_protocol",
    )
    captured = {}

    def fake_run(cmd, *, timeout_s=60):
        captured["cmd"] = cmd
        captured["timeout_s"] = timeout_s
        return 0, "PASS"

    monkeypatch.setattr(mod, "_run_short_gate", fake_run)

    rc, _ = mod._rerun_smoke_integrity(tmp_path / "results.csv")

    assert rc == 0
    cmd = captured["cmd"]
    assert "--expected-cases" in cmd
    assert cmd[cmd.index("--expected-cases") + 1] == "cantilever_beam_5d"
    assert "--expected-methods" in cmd
    methods = cmd[cmd.index("--expected-methods") + 1].split(",")
    assert len(methods) == 10
    assert "random+gp" in methods
    assert "--expected-method-families" in cmd
    families = cmd[cmd.index("--expected-method-families") + 1].split(",")
    assert len(families) == 10
    assert "random+gp:static_design" in families
    assert cmd[cmd.index("--expected-repeats") + 1] == "2"
    assert cmd[cmd.index("--expected-n-test") + 1] == "100"
    assert cmd[cmd.index("--seed-strategy") + 1] == "case_repeat"


def test_publication_readiness_smoke_pipeline_prefers_latest_passing_audit(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_smoke_latest_pass",
    )
    monkeypatch.setattr(
        mod,
        "_rerun_smoke_integrity",
        lambda results_csv: (0, "PASS") if "_smoke_pass" in str(results_csv) else (1, "bad"),
    )
    passing = tmp_path / "_smoke_pass"
    newer_incomplete = tmp_path / "_smoke_newer_incomplete"
    for directory in (passing, newer_incomplete):
        directory.mkdir()
        (directory / "results.csv").write_text("case_id\nc0\n", encoding="utf-8")
    (passing / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )
    (passing / "report" / "figs").mkdir(parents=True)
    (passing / "report" / "REPORT.md").write_text("report\n", encoding="utf-8")
    (passing / "report" / "figs" / "fig.png").write_bytes(b"png")
    # Make the incomplete directory newest.
    (newer_incomplete / "touch.txt").write_text("newer\n", encoding="utf-8")

    gate = mod.summarize_smoke_pipeline(tmp_path)

    assert gate.status == "PASS"
    assert "_smoke_pass" in gate.evidence
    assert "_smoke_newer_incomplete" not in gate.evidence


def test_publication_readiness_smoke_pipeline_warns_without_report(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_smoke_no_report",
    )
    monkeypatch.setattr(mod, "_rerun_smoke_integrity", lambda results_csv: (0, "PASS"))
    smoke = tmp_path / "_smoke"
    smoke.mkdir()
    (smoke / "results.csv").write_text("case_id\nc0\n", encoding="utf-8")
    (smoke / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    gate = mod.summarize_smoke_pipeline(tmp_path)

    assert gate.status == "WARN"
    assert "REPORT.md" in gate.evidence


def test_publication_readiness_smoke_pipeline_warns_without_audit(tmp_path):
    mod = _load_module(
        "scripts/audit_publication_readiness.py",
        "publication_readiness_smoke_warn",
    )
    smoke = tmp_path / "_smoke"
    smoke.mkdir()
    (smoke / "results.csv").write_text("case_id\nc0\n", encoding="utf-8")

    gate = mod.summarize_smoke_pipeline(tmp_path)

    assert gate.status == "WARN"
    assert "sweep_integrity_report.md" in gate.evidence


def test_submission_ready_rejects_missing_required_si_files(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_si_missing")
    paper = tmp_path / "paper"
    paper.mkdir()

    errors = mod.check_required_si_files(paper)

    assert any("si_hyperparameters.tex" in err for err in errors)
    assert any("generate_si.py" in err for err in errors)


def test_submission_ready_accepts_required_si_files(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_si_ok")
    paper = tmp_path / "paper"
    paper.mkdir()
    for rel in mod.REQUIRED_SI_FILES:
        (paper / rel).write_text("generated SI\n", encoding="utf-8")
    expected_tiers = mod._expected_case_evidence_tiers()
    md_lines = [
        "| Case | Domain | d | m | PDE | Governing eq. | Cost | Failure? | Evidence tier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for case_id, tier in sorted(expected_tiers.items()):
        md_lines.append(f"| `{case_id}` | test | 1 | 1 | algebraic | test | L0 | -- | `{tier}` |")
    (paper / "si_cases.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    (paper / "si_cases.tex").write_text(
        r"\textbf{Evidence tier} \\"
        "\n"
        r"\texttt{external\_note\_targeted\_regression} \\"
        "\n"
        r"\texttt{external\_note\_analytic\_control} \\"
        "\n"
        r"\texttt{targeted\_analytical\_control} \\"
        "\n"
        r"\texttt{targeted\_plausibility\_only} \\"
        "\n",
        encoding="utf-8",
    )

    assert mod.check_required_si_files(paper) == []


def test_submission_ready_rejects_si_cases_without_evidence_tiers(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_si_evidence_bad")
    paper = tmp_path / "paper"
    paper.mkdir()
    for rel in mod.REQUIRED_SI_FILES:
        (paper / rel).write_text("generated SI\n", encoding="utf-8")
    (paper / "si_cases.md").write_text("| Case | Domain |\n|---|---|\n", encoding="utf-8")
    (paper / "si_cases.tex").write_text(r"\textbf{Case} & \textbf{Domain} \\", encoding="utf-8")

    errors = mod.check_required_si_files(paper)

    assert any("si_cases.md missing Evidence tier column" in err for err in errors)
    assert any("si_cases.tex missing Evidence tier column" in err for err in errors)
    assert any("missing required tier" in err for err in errors)


def test_submission_ready_rejects_si_case_evidence_tier_drift(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_si_evidence_drift")
    paper = tmp_path / "paper"
    paper.mkdir()
    for rel in mod.REQUIRED_SI_FILES:
        (paper / rel).write_text("generated SI\n", encoding="utf-8")
    expected_tiers = mod._expected_case_evidence_tiers()
    md_lines = [
        "| Case | Domain | d | m | PDE | Governing eq. | Cost | Failure? | Evidence tier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for case_id, tier in sorted(expected_tiers.items()):
        observed = "targeted_plausibility_only" if case_id == "pressure_vessel_4d" else tier
        md_lines.append(f"| `{case_id}` | test | 1 | 1 | algebraic | test | L0 | -- | `{observed}` |")
    (paper / "si_cases.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (paper / "si_cases.tex").write_text(
        r"\textbf{Evidence tier} \\"
        "\n"
        r"\texttt{external\_note\_targeted\_regression} \\"
        "\n"
        r"\texttt{external\_note\_analytic\_control} \\"
        "\n"
        r"\texttt{targeted\_analytical\_control} \\"
        "\n"
        r"\texttt{targeted\_plausibility\_only} \\"
        "\n",
        encoding="utf-8",
    )

    errors = mod.check_required_si_files(paper)

    assert any("si_cases.md evidence-tier drift" in err for err in errors)
    assert any("pressure_vessel_4d" in err for err in errors)


def test_submission_ready_rejects_manifest_hash_mismatch(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_hash_mismatch")
    artifact = tmp_path / "results.csv"
    artifact.write_text("not the archived result\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ]
            }
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_artifacts(manifest)

    assert errors
    assert any("sha256 mismatch" in err for err in errors)


def test_submission_ready_rejects_manifest_artifact_malformed_size(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_hash_bad_size")
    artifact = tmp_path / "results.csv"
    artifact.write_text("case_id,method\ncase_a,m1\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": "bad",
                        "sha256": mod._sha256(artifact),
                    }
                ]
            }
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_artifacts(manifest)

    assert any("size_bytes is not an integer" in err for err in errors)


def test_submission_ready_rejects_manifest_symlink_artifact_entry(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_hash_symlink_entry",
    )
    artifact = tmp_path / "results.csv"
    artifact.write_text("case_id,method\ncase_a,m1\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": None,
                        "sha256": None,
                        "is_symlink": True,
                        "symlink_target": "../external/results.csv",
                    }
                ]
            }
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_artifacts(manifest)

    assert any("records a symlink" in err for err in errors)


def test_submission_ready_rejects_manifest_duplicate_artifact_paths(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_hash_duplicate_paths")
    artifact = tmp_path / "results.csv"
    artifact.write_text("case_id,method\ncase_a,m1\n", encoding="utf-8")
    entry = {
        "path": "results.csv",
        "size_bytes": artifact.stat().st_size,
        "sha256": mod._sha256(artifact),
    }
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({"artifacts": {"files": [entry, dict(entry)]}}),
        encoding="utf-8",
    )

    errors = mod.check_manifest_artifacts(manifest)

    assert any("duplicate paths" in err and "results.csv" in err for err in errors)


def test_submission_ready_rejects_manifest_artifact_paths_outside_archive(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_hash_path_escape")
    archive = tmp_path / "archive"
    archive.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("case_id,method\ncase_a,m1\n", encoding="utf-8")
    manifest = archive / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "artifacts": {
                "files": [
                    {
                        "path": "../outside.csv",
                        "size_bytes": outside.stat().st_size,
                        "sha256": mod._sha256(outside),
                    },
                    {
                        "path": str(outside),
                        "size_bytes": outside.stat().st_size,
                        "sha256": mod._sha256(outside),
                    },
                ]
            }
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_artifacts(manifest)

    assert sum("without parent traversal" in err for err in errors) == 2


def test_submission_ready_accepts_manifest_artifact_with_matching_hash(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_hash_ok")
    artifact = tmp_path / "results.csv"
    artifact.write_text("case_id,method\ncase_a,m1\n", encoding="utf-8")
    expected_hash = mod._sha256(artifact)
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": expected_hash,
                    }
                ]
            }
        }),
        encoding="utf-8",
    )

    assert mod.check_manifest_artifacts(manifest) == []


def test_submission_ready_rejects_dirty_manifest_and_empty_commands(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_provenance_bad")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": True,
                "status_porcelain": ["M scripts/run_pipeline.py"],
            },
            "experiment": {"commands": []},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("dirty git worktree" in err for err in errors)
    assert any("experiment.commands is empty" in err for err in errors)


def test_submission_ready_rejects_dry_run_manifest_command_history(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_provenance_dry_run")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "python_environment": _publication_python_environment(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": [
                    "python scripts/run_pipeline.py --profile full --confirm-full-run --dry-run",
                ]
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("--dry-run" in err for err in errors)


def test_submission_ready_rejects_missing_manifest_command_file(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_missing_command_file")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
                "command_files_missing": ["logs/pipeline_command_history.txt"],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("command_files_missing" in err for err in errors)


def test_submission_ready_rejects_malformed_manifest_command_file_fields(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_command_file_fields")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
                "command_files_requested": "logs/pipeline_command_history.txt",
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("command_files_requested" in err for err in errors)
    assert any("command_files_missing" in err for err in errors)


def test_submission_ready_rejects_manifest_command_file_parse_errors(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_command_file_parse_errors")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
                "command_file_parse_errors": [
                    "logs/pipeline_command_history.txt: line 2 is not label<TAB>command"
                ],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("command_file_parse_errors is non-empty" in err for err in errors)


def test_submission_ready_rejects_manifest_case_resolution_error(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_case_resolution_error")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
                "case_resolution_error": "ImportError: no doe_testbed",
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
                "command_file_parse_errors": [],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("case_resolution_error is non-empty" in err for err in errors)


def test_submission_ready_rejects_manifest_without_python_environment_lock(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_missing_python_environment",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "run_window": _publication_run_window(),
            "git": _publication_clean_git(),
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
                "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
                "command_file_parse_errors": [],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("python_environment section is missing" in err for err in errors)


def test_submission_ready_rejects_manifest_without_server_environment_provenance(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_missing_server_environment",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "python_environment": _publication_python_environment(),
            "run_window": _publication_run_window(),
            "git": _publication_clean_git(),
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
                "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
                "command_file_parse_errors": [],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("scheduler section is missing" in err for err in errors)
    assert any("accelerator section is missing" in err for err in errors)
    assert any("container section is missing" in err for err in errors)


def test_submission_ready_rejects_malformed_machine_resource_provenance(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_bad_machine_resource_provenance",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    machine = _publication_machine()
    machine.update({
        "effective_ram_gb": "not-a-number",
        "effective_ram_source": "",
        "cgroup_memory": {
            "schema_version": 2,
            "detected": "yes",
            "limit_gb": -1,
            "unlimited": "no",
        },
        "resource_limits": {"schema_version": 2, "status": "", "limits": []},
        "temp_storage": {
            "schema_version": 2,
            "resolved_path": "",
            "exists": "yes",
            "writable": "yes",
            "free_gb": -1,
            "env": [],
        },
    })
    manifest.write_text(
        json.dumps({
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": machine,
            "python_environment": _publication_python_environment(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": _publication_clean_git(),
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
                "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
                "command_file_parse_errors": [],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("RUN_MANIFEST machine.effective_ram_gb" in err for err in errors)
    assert any("RUN_MANIFEST machine.effective_ram_source" in err for err in errors)
    assert any("RUN_MANIFEST machine.cgroup_memory.schema_version" in err for err in errors)
    assert any("RUN_MANIFEST machine.cgroup_memory.detected" in err for err in errors)
    assert any("RUN_MANIFEST machine.cgroup_memory.limit_gb" in err for err in errors)
    assert any("RUN_MANIFEST machine.cgroup_memory.unlimited" in err for err in errors)
    assert any("RUN_MANIFEST machine.resource_limits.schema_version" in err for err in errors)
    assert any("RUN_MANIFEST machine.resource_limits.status" in err for err in errors)
    assert any("RUN_MANIFEST machine.resource_limits.limits" in err for err in errors)
    assert any("RUN_MANIFEST machine.temp_storage.schema_version" in err for err in errors)
    assert any("RUN_MANIFEST machine.temp_storage.resolved_path" in err for err in errors)
    assert any("RUN_MANIFEST machine.temp_storage.exists" in err for err in errors)
    assert any("RUN_MANIFEST machine.temp_storage.writable" in err for err in errors)
    assert any("RUN_MANIFEST machine.temp_storage.free_gb" in err for err in errors)
    assert any("RUN_MANIFEST machine.temp_storage.env" in err for err in errors)


def test_generate_run_manifest_strict_rejects_bad_python_environment_lock():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_bad_python_environment",
    )
    manifest = {
        "git": _publication_clean_git(),
        "python_environment": {
            "status": "ok",
            "command": "python -m pip freeze --all",
            "package_count": 10,
            "pip_freeze_sha256": "not-a-sha",
        },
        "run_window": _publication_run_window(),
        "experiment": _publication_experiment(),
        "results_summary": {"exists": False},
        "artifacts": {"files": []},
        "case_sources": {"status": "ok", "cases": [{"status": "ok", "sha256": "a" * 64}]},
        "method_sources": {
            "status": "ok",
            "methods": [{"status": "ok", "method_source_sha256": "a" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("python_environment.pip_freeze_sha256" in err for err in errors)


def test_generate_run_manifest_strict_rejects_missing_python_environment_lock_path():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_missing_python_environment_lock_path",
    )
    python_environment, pip_freeze_artifact = _materialize_publication_python_environment(ROOT)
    python_environment.pop("pip_freeze_path", None)
    manifest = {
        "git": _publication_clean_git(),
        "machine": _publication_machine(),
        "python_environment": python_environment,
        "scheduler": _publication_scheduler(),
        "accelerator": _publication_accelerator(),
        "container": _publication_container(),
        "run_window": _publication_run_window(),
        "experiment": _publication_experiment(),
        "results_summary": {"exists": False},
        "artifacts": {"files": [pip_freeze_artifact]},
        "case_sources": {"status": "ok", "cases": [{"status": "ok", "sha256": "a" * 64}]},
        "method_sources": {
            "status": "ok",
            "methods": [{"status": "ok", "method_source_sha256": "a" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("python_environment.pip_freeze_path" in err for err in errors)


def test_generate_run_manifest_strict_rejects_missing_server_environment_provenance():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_missing_server_environment",
    )
    manifest = {
        "git": _publication_clean_git(),
        "python_environment": _publication_python_environment(),
        "run_window": _publication_run_window(),
        "experiment": _publication_experiment(),
        "results_summary": {"exists": False},
        "artifacts": {"files": []},
        "case_sources": {"status": "ok", "cases": [{"status": "ok", "sha256": "a" * 64}]},
        "method_sources": {
            "status": "ok",
            "methods": [{"status": "ok", "method_source_sha256": "a" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("scheduler section is missing" in err for err in errors)
    assert any("accelerator section is missing" in err for err in errors)
    assert any("container section is missing" in err for err in errors)


def test_generate_run_manifest_strict_rejects_malformed_server_environment_provenance():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_bad_server_environment",
    )
    manifest = {
        "git": _publication_clean_git(),
        "python_environment": _publication_python_environment(),
        "scheduler": {"schema_version": 2, "name": "", "detected": "yes", "env": []},
        "accelerator": {
            "schema_version": 2,
            "nvidia_smi": "missing",
            "torch_cuda": {"status": "ok", "cuda_available": "yes", "device_count": -1},
        },
        "container": {"schema_version": 2, "detected": "no", "env": [], "marker_files": []},
        "run_window": _publication_run_window(),
        "experiment": _publication_experiment(),
        "results_summary": {"exists": False},
        "artifacts": {"files": []},
        "case_sources": {"status": "ok", "cases": [{"status": "ok", "sha256": "a" * 64}]},
        "method_sources": {
            "status": "ok",
            "methods": [{"status": "ok", "method_source_sha256": "a" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("scheduler.schema_version" in err for err in errors)
    assert any("scheduler.detected" in err for err in errors)
    assert any("accelerator.nvidia_smi" in err for err in errors)
    assert any("accelerator.torch_cuda.device_count" in err for err in errors)
    assert any("container.marker_files" in err for err in errors)


def test_generate_run_manifest_strict_rejects_malformed_machine_resource_provenance():
    mod = _load_module(
        "scripts/generate_run_manifest.py",
        "generate_run_manifest_bad_machine_resource_provenance",
    )
    machine = _publication_machine()
    machine.update({
        "effective_ram_gb": "not-a-number",
        "effective_ram_source": "",
        "cgroup_memory": {
            "schema_version": 2,
            "detected": "yes",
            "limit_gb": -1,
            "unlimited": "no",
        },
        "resource_limits": {"schema_version": 2, "status": "", "limits": []},
        "temp_storage": {
            "schema_version": 2,
            "resolved_path": "",
            "exists": "yes",
            "writable": "yes",
            "free_gb": -1,
            "env": [],
        },
    })
    manifest = {
        "git": _publication_clean_git(),
        "machine": machine,
        "python_environment": _publication_python_environment(),
        "scheduler": _publication_scheduler(),
        "accelerator": _publication_accelerator(),
        "container": _publication_container(),
        "run_window": _publication_run_window(),
        "experiment": _publication_experiment(),
        "results_summary": {"exists": False},
        "artifacts": {"files": []},
        "case_sources": {"status": "ok", "cases": [{"status": "ok", "sha256": "a" * 64}]},
        "method_sources": {
            "status": "ok",
            "methods": [{"status": "ok", "method_source_sha256": "a" * 64}],
        },
    }

    errors = mod._strict_archive_errors(manifest)

    assert any("machine.effective_ram_gb" in err for err in errors)
    assert any("machine.effective_ram_source" in err for err in errors)
    assert any("machine.cgroup_memory.schema_version" in err for err in errors)
    assert any("machine.cgroup_memory.detected" in err for err in errors)
    assert any("machine.cgroup_memory.limit_gb" in err for err in errors)
    assert any("machine.cgroup_memory.unlimited" in err for err in errors)
    assert any("machine.resource_limits.schema_version" in err for err in errors)
    assert any("machine.resource_limits.status" in err for err in errors)
    assert any("machine.resource_limits.limits" in err for err in errors)
    assert any("machine.temp_storage.schema_version" in err for err in errors)
    assert any("machine.temp_storage.resolved_path" in err for err in errors)
    assert any("machine.temp_storage.exists" in err for err in errors)
    assert any("machine.temp_storage.writable" in err for err in errors)
    assert any("machine.temp_storage.free_gb" in err for err in errors)
    assert any("machine.temp_storage.env" in err for err in errors)


def test_submission_ready_rejects_empty_manifest_command_file_request(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_empty_command_file_request")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
                "command_files_requested": [],
                "command_files_missing": [],
            "command_file_parse_errors": [],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("command_files_requested is empty" in err for err in errors)


def test_submission_ready_rejects_bad_manifest_command_file_entries(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_bad_command_file_entries")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
                "command_files_requested": [
                    "logs/pipeline_command_history.dry_run.txt",
                    "../logs/pipeline_command_history.txt",
                    r"C:\DOE-benchmark\logs\pipeline_command_history.txt",
                    "",
                    123,
                ],
                "command_files_missing": [],
            "command_file_parse_errors": [],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("blank or non-string" in err for err in errors)
    assert any("non-portable paths" in err for err in errors)
    assert any("dry-run command history" in err for err in errors)


def test_submission_ready_rejects_manifest_without_pipeline_invocation_entry(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_pipeline_invocation")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
            "command_file_parse_errors": [],
                "command_entries": [{
                    "label": "sweep",
                    "command": "python benchmark/scripts/run_full_benchmark.py --pool nature",
                }],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("pipeline_invocation" in err for err in errors)


def test_submission_ready_rejects_low_level_manifest_pipeline_invocation(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_low_level_pipeline")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": ["python benchmark/scripts/run_full_benchmark.py --pool nature"],
                "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
                "command_file_parse_errors": [],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python benchmark/scripts/run_full_benchmark.py --pool nature",
                }],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("formal full-pipeline entrypoint" in err for err in errors)


def test_submission_ready_rejects_malformed_manifest_command_entries(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_malformed_command_entries")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
            "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
            "command_file_parse_errors": [],
                "command_entries": [
                    {"label": "pipeline_invocation", "command": ""},
                    {"label": "", "command": "python scripts/run_pipeline.py --profile full --confirm-full-run"},
                ],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("malformed entries" in err for err in errors)
    assert any("do not match experiment.commands" in err for err in errors)


def test_submission_ready_rejects_duplicate_manifest_pipeline_invocation_entries(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_duplicate_pipeline_invocation_entries",
    )
    command = "python scripts/run_pipeline.py --profile full --confirm-full-run"
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "manifest_command": "python scripts/generate_run_manifest.py",
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
                "source_dirty": False,
            },
            "experiment": {
                "commands": [command, command],
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
            "command_file_parse_errors": [],
                "command_entries": [
                    {"label": "pipeline_invocation", "command": command},
                    {"label": "pipeline_invocation", "command": command},
                ],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("multiple pipeline_invocation" in err for err in errors)


def test_submission_ready_accepts_manifest_with_only_derived_dirty_entries(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_provenance_derived_dirty")
    manifest = tmp_path / "RUN_MANIFEST.json"
    _write_manifest_with_publication_python_environment(
        manifest,
        {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": {
                "commit": "abcdef123456",
                "dirty": True,
                "status_porcelain": [" M paper/numbers.json", " M docs/case_validation_note_sources.csv"],
                "source_dirty": False,
                "source_dirty_entries": [],
                "derived_dirty_entries": [" M paper/numbers.json", " M docs/case_validation_note_sources.csv"],
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
                "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
                "command_file_parse_errors": [],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        },
    )

    assert mod.check_manifest_provenance(manifest) == []


def test_submission_ready_rejects_manifest_without_archived_python_environment_lock_entry(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_missing_python_environment_lock_artifact",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    _write_manifest_with_publication_python_environment(
        manifest,
        {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(),
        },
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["artifacts"]["files"] = [
        item
        for item in data["artifacts"]["files"]
        if item.get("path") != "python_environment_pip_freeze.txt"
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    errors = mod.check_manifest_provenance(manifest)

    assert any("does not archive the python environment lock" in err for err in errors)


def test_submission_ready_rejects_manifest_without_archived_python_environment_lock_file(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_missing_python_environment_lock_file",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    _write_manifest_with_publication_python_environment(
        manifest,
        {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(),
            "results_summary": {
                "exists": True,
                "n_rows": 0,
                "n_cases_observed": 0,
                "case_ids_observed": [],
                "n_methods_observed": 0,
                "methods_observed": [],
                "budgets_observed": [],
                "repeats_observed": [],
                "status_counts": {},
                "failure_category_counts": {},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_missing_protocol_key_values": 0,
                "n_invalid_status_values": 0,
                "n_missing_failure_category_values": 0,
                "n_invalid_failure_category_values": 0,
                "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_invalid_runtime_duration_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {
                "files": [
                    {"path": "results.csv"},
                    {"path": "sweep_integrity_report.md"},
                ]
            },
        },
    )
    (tmp_path / "python_environment_pip_freeze.txt").unlink()
    (tmp_path / "results.csv").write_text("case_id,method,budget_mult,repeat,status\n", encoding="utf-8")
    (tmp_path / "sweep_integrity_report.md").write_text(
        "PASS: sweep satisfies the declared integrity gate.\n",
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("archived python environment lock is missing beside manifest" in err for err in errors)


def test_submission_ready_rejects_manifest_with_mismatched_python_environment_lock_sha(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_mismatched_python_environment_lock_sha",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    _write_manifest_with_publication_python_environment(
        manifest,
        {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(),
        },
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["artifacts"]["files"] = [
        {
            **item,
            "sha256": "0" * 64,
        }
        if item.get("path") == "python_environment_pip_freeze.txt"
        else item
        for item in data["artifacts"]["files"]
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    errors = mod.check_manifest_provenance(manifest)

    assert any("does not match the archived python environment lock artifact" in err for err in errors)


def test_submission_ready_rejects_python_environment_lock_package_drift_from_run_config(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_python_environment_run_config_drift",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    _write_manifest_with_publication_python_environment(
        manifest,
        {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(),
            "results_summary": {"exists": False},
        },
        contents=_publication_pip_freeze_contents(overrides={"botorch": "9.9.9"}),
    )
    run_config = {
        "environment": {
            "packages": {
                package: {"status": "installed", "version": "1.0.0"}
                for package in (
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
            }
        }
    }
    (tmp_path / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")

    errors = mod.check_manifest_result_contract(manifest)

    assert any(
        "python environment lock version does not match run_config.json for botorch"
        in err
        for err in errors
    )


def test_submission_ready_accepts_manifest_temp_storage_matching_run_config(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_temp_storage_ok",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    _write_manifest_with_publication_python_environment(
        manifest,
        {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(),
            "results_summary": {"exists": False},
        },
    )
    run_config = tmp_path / "run_config.json"
    _write_minimal_publication_run_config(run_config)

    assert mod.check_manifest_temp_storage_matches_run_config(manifest, run_config) == []


def test_submission_ready_rejects_manifest_temp_storage_drift_from_run_config(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_temp_storage_drift",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    _write_manifest_with_publication_python_environment(
        manifest,
        {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": _publication_clean_git(),
            "experiment": _publication_experiment(),
            "results_summary": {"exists": False},
        },
    )
    run_config = tmp_path / "run_config.json"
    _write_minimal_publication_run_config(run_config)
    payload = json.loads(run_config.read_text(encoding="utf-8"))
    payload["temp_storage"]["resolved_path"] = "/tmp/not-the-stage2-scratch"
    payload["temp_storage"]["env"]["SLURM_TMPDIR"] = "/tmp/not-the-stage2-scratch"
    run_config.write_text(json.dumps(payload), encoding="utf-8")

    errors = mod.check_manifest_result_contract(manifest)

    assert any("machine.temp_storage.resolved_path does not match" in err for err in errors)
    assert any("temp_storage.env for SLURM_TMPDIR" in err for err in errors)


def test_submission_ready_accepts_clean_manifest_provenance(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_provenance_ok")
    manifest = tmp_path / "RUN_MANIFEST.json"
    _write_manifest_with_publication_python_environment(
        manifest,
        {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "scheduler": _publication_scheduler(),
            "accelerator": _publication_accelerator(),
            "container": _publication_container(),
            "run_window": _publication_run_window(),
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
                "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
                "command_file_parse_errors": [],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        },
    )

    assert mod.check_manifest_provenance(manifest) == []


def test_submission_ready_rejects_manifest_without_run_window(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_run_window")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "generated_at": "2026-06-02T00:00:00+00:00",
            "manifest_command": "python scripts/generate_run_manifest.py",
            "machine": _publication_machine(),
            "git": {
                "commit": "abcdef123456",
                "dirty": False,
                "status_porcelain": [],
            },
            "experiment": {
                "commands": ["python scripts/run_pipeline.py --profile full --confirm-full-run"],
                "case_resolution_error": None,
                "command_files_requested": ["logs/pipeline_command_history.txt"],
                "command_files_missing": [],
                "command_file_parse_errors": [],
                "command_entries": [{
                    "label": "pipeline_invocation",
                    "command": "python scripts/run_pipeline.py --profile full --confirm-full-run",
                }],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_provenance(manifest)

    assert any("run_window section is missing" in err for err in errors)


def test_submission_ready_rejects_manifest_result_contract_drift(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_result_contract_bad")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 2,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 3,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 2,
                "methods_observed": ["m1", "m2"],
                "budgets_observed": [5],
                "repeats_observed": [0, 1],
                "status_counts": {"ok": 3},
            },
            "artifacts": {"files": []},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("does not archive results.csv" in err for err in errors)
    assert any("archived results.csv is missing" in err for err in errors)
    assert any("result grid mismatch" in err for err in errors)
    assert any("method set drift" in err and "m2" in err for err in errors)


def test_submission_ready_rejects_malformed_manifest_status_counts(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_status_counts_bad")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": -1, "timeout": "bad", "rerun": 2},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {"files": [{"path": "results.csv"}]},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("non-negative integer count required for 'ok'" in err for err in errors)
    assert any("non-negative integer count required for 'timeout'" in err for err in errors)
    assert any("unknown statuses" in err and "rerun" in err for err in errors)


def test_submission_ready_rejects_missing_manifest_status_counts(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_status_counts_missing")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": ["ok", 1],
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {"files": [{"path": "results.csv"}]},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("status_counts is missing or malformed" in err for err in errors)


def test_submission_ready_rejects_malformed_manifest_grid_summary(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_grid_malformed")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5, "bad"],
                "n_repeats": "bad",
            },
            "results_summary": {
                "exists": True,
                "n_rows": "bad",
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5, "bad"],
                "repeats_observed": [0, "bad"],
                "status_counts": {"ok": 1},
                "failure_category_counts": {"ok": 1},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {"files": [{"path": "results.csv"}]},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("experiment.budgets must be a list of integers" in err for err in errors)
    assert any("experiment.n_repeats must be a non-negative integer" in err for err in errors)
    assert any("results_summary.budgets_observed must be a list of integers" in err for err in errors)
    assert any("results_summary.repeats_observed must be a list of integers" in err for err in errors)
    assert any("results_summary.n_rows must be a non-negative integer" in err for err in errors)


def test_submission_ready_rejects_missing_manifest_observed_sets(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_missing_observed_sets",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "methods_observed": "not-a-list",
                "budgets_observed": None,
                "repeats_observed": "0..29",
                "status_counts": {"ok": 1},
                "failure_category_counts": {"ok": 1},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_missing_protocol_key_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {"files": [{"path": "results.csv"}]},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("case_ids_observed is missing or malformed" in err for err in errors)
    assert any("n_cases_observed is missing or malformed" in err for err in errors)
    assert any("methods_observed is missing or malformed" in err for err in errors)
    assert any("n_methods_observed is missing or malformed" in err for err in errors)
    assert any("budgets_observed is missing or malformed" in err for err in errors)
    assert any("repeats_observed is missing or malformed" in err for err in errors)


def test_submission_ready_rejects_duplicate_manifest_observed_sets_and_count_drift(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_observed_set_count_drift",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 2,
                "case_ids_declared": ["case_a", "case_b"],
                "n_methods_declared": 2,
                "methods_declared": ["m1", "m2"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 4,
                "n_cases_observed": 3,
                "case_ids_observed": ["case_a", "case_a"],
                "n_methods_observed": 3,
                "methods_observed": ["m1", "m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": 4},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_missing_protocol_key_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {"files": [{"path": "results.csv"}]},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("case set drift" in err and "case_b" in err for err in errors)
    assert any("method set drift" in err and "m2" in err for err in errors)
    assert any("case_ids_observed contains duplicates" in err for err in errors)
    assert any("methods_observed contains duplicates" in err for err in errors)
    assert any("n_cases_observed does not match" in err for err in errors)
    assert any("n_methods_observed does not match" in err for err in errors)
    assert any("case-count drift" in err for err in errors)
    assert any("method-count drift" in err for err in errors)


def test_submission_ready_rejects_manifest_row_contract_counts(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_row_contract_bad")
    results = tmp_path / "results.csv"
    results.write_text("case_id,method,budget_mult,repeat,status\ncase_a,m1,5,0,ok\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": 1},
                "failure_category_counts": {"ok": 1},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_missing_protocol_key_values": 6,
                "n_invalid_status_values": 1,
                "n_invalid_ok_nrmse_values": 2,
                "n_invalid_t_total_s_values": 3,
                "n_invalid_runtime_duration_values": 1,
                "n_missing_failure_error_values": 4,
                "n_ok_rows_with_error_values": 5,
            },
            "artifacts": {"files": [{"path": "results.csv"}]},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("missing protocol key values" in err for err in errors)
    assert any("invalid status values" in err for err in errors)
    assert any("invalid ok-row nrmse values" in err for err in errors)
    assert any("invalid t_total_s values" in err for err in errors)
    assert any("t_total_s inconsistent with runtime timestamps" in err for err in errors)
    assert any("timeout/failed rows without error reasons" in err for err in errors)
    assert any("ok rows with non-empty errors" in err for err in errors)


def test_submission_ready_rejects_malformed_manifest_row_contract_counts(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_row_contract_malformed",
    )
    results = tmp_path / "results.csv"
    results.write_text("case_id,method,budget_mult,repeat,status\ncase_a,m1,5,0,ok\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": 1},
                "failure_category_counts": {"ok": 1},
                "n_invalid_budget_mult_values": "bad",
                "n_invalid_repeat_values": -1,
                "n_missing_protocol_key_values": "bad",
                "n_invalid_status_values": "bad",
            "n_missing_failure_category_values": "bad",
            "n_invalid_failure_category_values": -1,
            "n_failure_category_status_mismatch_values": "bad",
                "n_invalid_ok_nrmse_values": -1,
                "n_invalid_t_total_s_values": "bad",
                "n_missing_failure_error_values": -1,
                "n_ok_rows_with_error_values": "bad",
            },
            "artifacts": {"files": [{"path": "results.csv"}]},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("n_invalid_budget_mult_values must be a non-negative integer" in err for err in errors)
    assert any("n_invalid_repeat_values must be a non-negative integer" in err for err in errors)
    assert any("n_missing_protocol_key_values must be a non-negative integer" in err for err in errors)
    assert any("n_invalid_status_values must be a non-negative integer" in err for err in errors)
    assert any("n_invalid_ok_nrmse_values must be a non-negative integer" in err for err in errors)
    assert any("n_invalid_t_total_s_values must be a non-negative integer" in err for err in errors)
    assert any("n_missing_failure_error_values must be a non-negative integer" in err for err in errors)
    assert any("n_ok_rows_with_error_values must be a non-negative integer" in err for err in errors)


def test_submission_ready_accepts_manifest_result_contract(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_manifest_result_contract_ok")
    results = tmp_path / "results.csv"
    results.write_text(
        _runtime_results_csv(["case_a,m1,5,0,ok,,0.1,1.0"]),
        encoding="utf-8",
    )
    sweep_report = tmp_path / "sweep_integrity_report.md"
    sweep_report.write_text("# Sweep Integrity Audit\n\nOverall status: **PASS**\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": 1},
                "failure_category_counts": {"ok": 1},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": results.stat().st_size,
                        "sha256": mod._sha256(results),
                    },
                    {
                        "path": "sweep_integrity_report.md",
                        "size_bytes": sweep_report.stat().st_size,
                        "sha256": mod._sha256(sweep_report),
                    },
                ]
            },
        }),
        encoding="utf-8",
    )

    assert mod.check_manifest_result_contract(manifest) == []


def test_submission_ready_rejects_manifest_runtime_resource_summary_drift(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_runtime_resource_summary_drift",
    )
    results = tmp_path / "results.csv"
    results.write_text(
        _runtime_results_csv(["case_a,m1,5,0,ok,,0.1,1.0"]),
        encoding="utf-8",
    )
    sweep_report = tmp_path / "sweep_integrity_report.md"
    sweep_report.write_text("# Sweep Integrity Audit\n\nOverall status: **PASS**\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": 1},
                "failure_category_counts": {"ok": 1},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_invalid_rss_gb_values": 0,
                "n_invalid_rss_gb_peak_order_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
                "runtime_resource_summary": {
                    "rss_trace_columns_present": True,
                    "missing_rss_trace_columns": [],
                    "n_rows": 1,
                    "n_rows_with_valid_rss_gb_peak": 1,
                    "rss_gb_peak_min": 1.0,
                    "rss_gb_peak_p50": 1.0,
                    "rss_gb_peak_p95": 1.0,
                    "rss_gb_peak_max": 1.0,
                    "rss_gb_peak_max_by_method": {"m1": 1.0},
                },
            },
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": results.stat().st_size,
                        "sha256": mod._sha256(results),
                    },
                    {
                        "path": "sweep_integrity_report.md",
                        "size_bytes": sweep_report.stat().st_size,
                        "sha256": mod._sha256(sweep_report),
                    },
                ]
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("rss_gb_peak_max does not match current results.csv" in err for err in errors)


def test_submission_ready_rejects_manifest_runtime_timestamp_summary_drift(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_runtime_timestamp_summary_drift",
    )
    results = tmp_path / "results.csv"
    results.write_text(
        _runtime_results_csv(["case_a,m1,5,0,ok,,0.1,1.0"]),
        encoding="utf-8",
    )
    sweep_report = tmp_path / "sweep_integrity_report.md"
    sweep_report.write_text("# Sweep Integrity Audit\n\nOverall status: **PASS**\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": 1},
                "failure_category_counts": {"ok": 1},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_invalid_runtime_timestamp_values": 1,
                "n_invalid_runtime_time_order_values": 0,
                "n_invalid_runtime_duration_values": 0,
                "n_invalid_rss_gb_values": 0,
                "n_invalid_rss_gb_peak_order_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
                "runtime_timestamp_summary": {
                    "timestamp_trace_columns_present": True,
                    "missing_timestamp_trace_columns": [],
                    "n_rows": 1,
                    "n_invalid_runtime_timestamp_values": 1,
                    "n_invalid_runtime_time_order_values": 0,
                    "n_invalid_runtime_duration_values": 0,
                },
                "runtime_resource_summary": {
                    "rss_trace_columns_present": True,
                    "missing_rss_trace_columns": [],
                    "n_rows": 1,
                    "n_rows_with_valid_rss_gb_peak": 1,
                    "rss_gb_peak_min": 1.75,
                    "rss_gb_peak_p50": 1.75,
                    "rss_gb_peak_p95": 1.75,
                    "rss_gb_peak_max": 1.75,
                    "rss_gb_peak_max_by_method": {"m1": 1.75},
                },
            },
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": results.stat().st_size,
                        "sha256": mod._sha256(results),
                    },
                    {
                        "path": "sweep_integrity_report.md",
                        "size_bytes": sweep_report.stat().st_size,
                        "sha256": mod._sha256(sweep_report),
                    },
                ]
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("invalid runtime timestamp values" in err for err in errors)
    assert any("n_invalid_runtime_timestamp_values does not match current results.csv" in err for err in errors)


def test_submission_ready_rejects_current_results_csv_contract_drift(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_current_results_contract_bad",
    )
    results = tmp_path / "results.csv"
    results.write_text(
        _runtime_results_csv([
            "case_a,m1,5,0,ok,,0.1,1.0",
            "case_a,m1,5,0,ok,,0.2,1.0",
        ]),
        encoding="utf-8",
    )
    sweep_report = tmp_path / "sweep_integrity_report.md"
    sweep_report.write_text("# Sweep Integrity Audit\n\nOverall status: **PASS**\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 2,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": 2},
                "failure_category_counts": {"ok": 2},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": results.stat().st_size,
                        "sha256": mod._sha256(results),
                    },
                    {
                        "path": "sweep_integrity_report.md",
                        "size_bytes": sweep_report.stat().st_size,
                        "sha256": mod._sha256(sweep_report),
                    },
                ]
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("current results.csv row-level contract failed" in err for err in errors)
    assert any("duplicate" in err and "cells" in err for err in errors)


def test_submission_ready_rejects_manifest_summary_drift_from_current_results(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_current_results_summary_drift",
    )
    results = tmp_path / "results.csv"
    results.write_text(
        _runtime_results_csv(["case_a,m1,5,0,ok,,0.1,1.0"]),
        encoding="utf-8",
    )
    sweep_report = tmp_path / "sweep_integrity_report.md"
    sweep_report.write_text("# Sweep Integrity Audit\n\nOverall status: **PASS**\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m2"],
                "budgets_observed": [10],
                "repeats_observed": [0],
                "status_counts": {"failed": 1},
                "failure_category_counts": {"exception": 1},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": results.stat().st_size,
                        "sha256": mod._sha256(results),
                    },
                    {
                        "path": "sweep_integrity_report.md",
                        "size_bytes": sweep_report.stat().st_size,
                        "sha256": mod._sha256(sweep_report),
                    },
                ]
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("methods_observed" in err and "current results.csv method set" in err for err in errors)
    assert any("budgets_observed" in err and "current results.csv budget set" in err for err in errors)
    assert any("status_counts" in err and "current results.csv status counts" in err for err in errors)


def test_submission_ready_default_manifest_enforces_headline_results_source(tmp_path, monkeypatch):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_default_headline_source",
    )
    results = tmp_path / "results.csv"
    results.write_text(
        _runtime_results_csv(["case_a,m1,5,0,ok,,0.1,1.0"]),
        encoding="utf-8",
    )
    sweep_report = tmp_path / "sweep_integrity_report.md"
    sweep_report.write_text("# Sweep Integrity Audit\n\nOverall status: **PASS**\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    monkeypatch.setattr(mod, "MANIFEST_DEFAULT", manifest)
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": 1},
                "failure_category_counts": {"ok": 1},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "size_bytes": results.stat().st_size,
                        "sha256": mod._sha256(results),
                    },
                    {
                        "path": "sweep_integrity_report.md",
                        "size_bytes": sweep_report.stat().st_size,
                        "sha256": mod._sha256(sweep_report),
                    },
                ]
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("canonical results source" in err for err in errors)


def test_submission_ready_rejects_manifest_without_sweep_integrity_artifact(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_manifest_missing_sweep_artifact",
    )
    results = tmp_path / "results.csv"
    results.write_text("case_id,method,budget_mult,repeat,status\ncase_a,m1,5,0,ok\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
                "n_methods_declared": 1,
                "methods_declared": ["m1"],
                "budgets": [5],
                "n_repeats": 1,
            },
            "results_summary": {
                "exists": True,
                "n_rows": 1,
                "n_cases_observed": 1,
                "case_ids_observed": ["case_a"],
                "n_methods_observed": 1,
                "methods_observed": ["m1"],
                "budgets_observed": [5],
                "repeats_observed": [0],
                "status_counts": {"ok": 1},
                "failure_category_counts": {"ok": 1},
                "n_invalid_budget_mult_values": 0,
                "n_invalid_repeat_values": 0,
                "n_invalid_status_values": 0,
            "n_missing_failure_category_values": 0,
            "n_invalid_failure_category_values": 0,
            "n_failure_category_status_mismatch_values": 0,
                "n_invalid_ok_nrmse_values": 0,
                "n_invalid_t_total_s_values": 0,
                "n_missing_failure_error_values": 0,
                "n_ok_rows_with_error_values": 0,
            },
            "artifacts": {"files": [{"path": "results.csv"}]},
        }),
        encoding="utf-8",
    )

    errors = mod.check_manifest_result_contract(manifest)

    assert any("sweep_integrity_report.md" in err for err in errors)


def test_submission_ready_rejects_missing_case_source_fingerprints(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_sources_missing")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({"experiment": {"case_ids_declared": ["pressure_vessel_4d"]}}),
        encoding="utf-8",
    )

    errors = mod.check_case_source_fingerprints(manifest)

    assert errors == ["RUN_MANIFEST case_sources fingerprints are missing"]


def test_submission_ready_rejects_case_source_hash_mismatch(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_sources_bad")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {"case_ids_declared": ["pressure_vessel_4d"]},
            "case_sources": {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "pressure_vessel_4d",
                        "status": "ok",
                        "source_path": "doe_testbed/cases/multidisciplinary/pressure_vessel_4d.py",
                        "sha256": "0" * 64,
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_case_source_fingerprints(manifest)

    assert any("case source hash mismatch for pressure_vessel_4d" in err for err in errors)


def test_submission_ready_rejects_case_source_path_mismatch(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_sources_bad_path")
    source, sha, error = mod._current_case_source_hash("pressure_vessel_4d")
    assert error is None
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {"case_ids_declared": ["pressure_vessel_4d"]},
            "case_sources": {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "pressure_vessel_4d",
                        "status": "ok",
                        "source_path": "doe_testbed/cases/multidisciplinary/not_the_case.py",
                        "sha256": sha,
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_case_source_fingerprints(manifest)

    assert source is not None
    assert any("case source path mismatch for pressure_vessel_4d" in err for err in errors)


def test_submission_ready_accepts_matching_case_source_fingerprints(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_sources_ok")
    source, sha, error = mod._current_case_source_hash("pressure_vessel_4d")
    assert error is None
    assert source is not None
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {"case_ids_declared": ["pressure_vessel_4d"]},
            "case_sources": {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "pressure_vessel_4d",
                        "status": "ok",
                        "source_path": source.resolve().relative_to(ROOT).as_posix(),
                        "sha256": sha,
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    assert mod.check_case_source_fingerprints(manifest) == []


def test_submission_ready_rejects_missing_method_source_fingerprints(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_method_sources_missing")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "pool": "default",
                "methods_declared": ["random+gp"],
            }
        }),
        encoding="utf-8",
    )

    errors = mod.check_method_source_fingerprints(manifest)

    assert errors == ["RUN_MANIFEST method_sources fingerprints are missing"]


def test_submission_ready_rejects_method_source_hash_mismatch(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_method_sources_bad")
    from benchmark.core import method_source_components, method_source_fingerprint
    from benchmark.methods import default_method_pool

    method = default_method_pool()[0]
    fingerprint = method_source_fingerprint(method)
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "pool": "default",
                "methods_declared": [method.name],
            },
            "method_sources": {
                "schema_version": 1,
                "methods": [
                    {
                        "method": method.name,
                        "family": method.family,
                        "status": "ok",
                        "components": method_source_components(method),
                        "method_source_path": fingerprint["method_source_path"],
                        "method_source_sha256": "0" * 64,
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_method_source_fingerprints(manifest)

    assert any(f"method source hash mismatch for {method.name}" in err for err in errors)


def test_submission_ready_rejects_missing_method_source_components(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_method_sources_components")
    from benchmark.core import method_source_fingerprint
    from benchmark.methods import default_method_pool

    method = default_method_pool()[0]
    fingerprint = method_source_fingerprint(method)
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "pool": "default",
                "methods_declared": [method.name],
            },
            "method_sources": {
                "schema_version": 1,
                "methods": [
                    {
                        "method": method.name,
                        "family": method.family,
                        "status": "ok",
                        **fingerprint,
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    errors = mod.check_method_source_fingerprints(manifest)

    assert any(f"method source components are missing for {method.name}" in err for err in errors)


def test_submission_ready_accepts_matching_method_source_fingerprints(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_method_sources_ok")
    from benchmark.core import method_source_components, method_source_fingerprint
    from benchmark.methods import default_method_pool

    method = default_method_pool()[0]
    fingerprint = method_source_fingerprint(method)
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "pool": "default",
                "methods_declared": [method.name],
            },
            "method_sources": {
                "schema_version": 1,
                "methods": [
                    {
                        "method": method.name,
                        "family": method.family,
                        "status": "ok",
                        "components": method_source_components(method),
                        **fingerprint,
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    assert mod.check_method_source_fingerprints(manifest) == []


def _write_case_validation_status_copy(path: Path, *, resolve_p0: bool = False) -> None:
    source = ROOT / "docs" / "case_validation_status.csv"
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if resolve_p0:
        for row in rows:
            if row["risk_level"] == "P0":
                row["risk_level"] = "P1"
                row["next_action"] = "resolved for submission readiness fixture"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_case_validation_sources_copy(path: Path) -> None:
    source = ROOT / "docs" / "case_validation_sources.csv"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_case_validation_note_sources_copy(path: Path) -> None:
    source = ROOT / "docs" / "case_validation_note_sources.csv"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_case_validation_summary_copy(path: Path) -> None:
    source = ROOT / "docs" / "CASE_VALIDATION_SUMMARY.md"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


@requires_validation_notes
def test_submission_ready_case_validation_has_no_current_p0_rows():
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_current")

    errors = mod.check_case_validation_status(ROOT / "docs" / "case_validation_status.csv")

    assert not any("rocket_isp_4d: risk_level 'P0'" in err for err in errors)
    assert not any("supersonic_nozzle_5d: risk_level 'P0'" in err for err in errors)
    assert not any("thin_plate_buckling_5d: risk_level 'P0'" in err for err in errors)
    assert errors == []
    assert mod.check_case_validation_sources(
        ROOT / "docs" / "case_validation_status.csv",
        ROOT / "docs" / "case_validation_sources.csv",
    ) == []
    assert mod.check_case_validation_note_sources(
        ROOT / "docs" / "case_validation_note_sources.csv",
    ) == []
    assert mod.check_case_validation_summary(
        ROOT / "docs" / "case_validation_status.csv",
        ROOT / "docs" / "CASE_VALIDATION_SUMMARY.md",
        ROOT / "docs" / "case_validation_note_sources.csv",
    ) == []


@requires_validation_notes
def test_submission_ready_accepts_resolved_case_validation_ledger(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_ok")
    status = tmp_path / "case_validation_status.csv"
    _write_case_validation_status_copy(status, resolve_p0=True)
    sources = tmp_path / "case_validation_sources.csv"
    _write_case_validation_sources_copy(sources)
    note_sources = tmp_path / "case_validation_note_sources.csv"
    _write_case_validation_note_sources_copy(note_sources)
    summary = tmp_path / "CASE_VALIDATION_SUMMARY.md"
    _write_case_validation_summary_copy(summary)

    assert mod.check_case_validation_status(status) == []
    assert mod.check_case_validation_sources(status, sources) == []
    assert mod.check_case_validation_note_sources(note_sources) == []
    assert mod.check_case_validation_summary(status, summary, note_sources) == []


def test_submission_ready_rejects_missing_case_validation_rows(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_missing")
    status = tmp_path / "case_validation_status.csv"
    _write_case_validation_status_copy(status, resolve_p0=True)
    rows: list[dict[str, str]]
    with status.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row["case_id"] != "cantilever_beam_5d"]
        fieldnames = list(reader.fieldnames or [])
    with status.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = mod.check_case_validation_status(status)

    assert any("missing registered cases" in err and "cantilever_beam_5d" in err for err in errors)
    assert any("row count mismatch" in err for err in errors)


def test_submission_ready_rejects_fixed_case_without_targeted_test(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_no_test")
    status = tmp_path / "case_validation_status.csv"
    _write_case_validation_status_copy(status, resolve_p0=True)
    with status.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        if row["case_id"] == "cantilever_beam_5d":
            row["targeted_physics_test"] = "NONE"
    with status.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = mod.check_case_validation_status(status)

    assert any("cantilever_beam_5d: fixed/targeted formula row lacks targeted physics test" in err for err in errors)


def test_submission_ready_rejects_case_validation_evidence_tier_drift(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_evidence")
    status = tmp_path / "case_validation_status.csv"
    _write_case_validation_status_copy(status, resolve_p0=True)
    with status.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        if row["case_id"] == "aircraft_conceptual_8d":
            row["risk_level"] = "P1"
            row["final_disposition"] = "accepted after resweep"
        if row["case_id"] == "cantilever_beam_5d":
            row["final_disposition"] = "needs review"
    with status.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = mod.check_case_validation_status(status)

    assert any(
        "aircraft_conceptual_8d: P1 case-validation risk requires" in err
        for err in errors
    )
    assert any(
        "cantilever_beam_5d: prior-note P1 row must have an accepted" in err
        for err in errors
    )


def test_submission_ready_accepts_p2_analytic_without_validation_note(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_p2_note")
    status = tmp_path / "case_validation_status.csv"
    _write_case_validation_status_copy(status, resolve_p0=True)
    with status.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        if row["case_id"] == "aircraft_conceptual_8d":
            row["final_disposition"] = "accepted analytic control"
    with status.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = mod.check_case_validation_status(status)

    assert not any("aircraft_conceptual_8d:" in err for err in errors)


def test_submission_ready_rejects_no_note_p2_external_validation_claim(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_case_validation_p2_external_claim",
    )
    status = tmp_path / "case_validation_status.csv"
    _write_case_validation_status_copy(status, resolve_p0=True)
    with status.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        if row["case_id"] == "aircraft_conceptual_8d":
            row["final_disposition"] = "externally validated"
    with status.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = mod.check_case_validation_status(status)

    assert any(
        "aircraft_conceptual_8d: P2 case-validation risk must be marked "
        "plausibility-only or analytic/control" in err
        for err in errors
    )


def test_submission_ready_rejects_thin_p2_plausibility_test_file(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_p2_thin_test")
    status = tmp_path / "case_validation_status.csv"
    thin_test = tmp_path / "test_thin_p2_physics.py"
    thin_test.write_text(
        "def test_declared_output_ranges_cover_domain_corners():\n"
        "    pass\n\n"
        "def test_neuralfoil_backend_is_used_when_available():\n"
        "    pass\n",
        encoding="utf-8",
    )
    _write_case_validation_status_copy(status, resolve_p0=True)
    with status.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        if row["case_id"] == "aircraft_conceptual_8d":
            row["targeted_physics_test"] = thin_test.as_posix()
    with status.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = mod.check_case_validation_status(status)

    assert any(
        "aircraft_conceptual_8d: P2 plausibility-only targeted physics test has only "
        "2 test function(s)" in err
        for err in errors
    )
    assert any(
        "aircraft_conceptual_8d: P2 plausibility-only targeted physics test needs "
        "at least 2 non-generic" in err
        for err in errors
    )


def test_submission_ready_rejects_stale_case_validation_source_hash(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_sources_bad")
    status = tmp_path / "case_validation_status.csv"
    sources = tmp_path / "case_validation_sources.csv"
    _write_case_validation_status_copy(status, resolve_p0=True)
    _write_case_validation_sources_copy(sources)

    with sources.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        if row["case_id"] == "cantilever_beam_5d":
            row["case_source_sha256"] = "0" * 64
    with sources.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = mod.check_case_validation_sources(status, sources)

    assert any("cantilever_beam_5d: case validation source hash mismatch" in err for err in errors)


@requires_validation_notes
def test_submission_ready_rejects_stale_case_validation_note_source_hash(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_case_validation_note_sources_bad",
    )
    note_sources = tmp_path / "case_validation_note_sources.csv"
    _write_case_validation_note_sources_copy(note_sources)

    with note_sources.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    rows[0]["sha256"] = "0" * 64
    with note_sources.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = mod.check_case_validation_note_sources(note_sources)

    assert any("case validation note-source hash mismatch" in err for err in errors)


@requires_validation_notes
def test_submission_ready_rejects_stale_case_validation_summary(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_summary_bad")
    status = tmp_path / "case_validation_status.csv"
    _write_case_validation_status_copy(status, resolve_p0=True)
    note_sources = tmp_path / "case_validation_note_sources.csv"
    _write_case_validation_note_sources_copy(note_sources)
    summary = tmp_path / "CASE_VALIDATION_SUMMARY.md"
    _write_case_validation_summary_copy(summary)
    summary.write_text(summary.read_text(encoding="utf-8") + "\nmanual edit\n", encoding="utf-8")

    errors = mod.check_case_validation_summary(status, summary, note_sources)

    assert errors == [
        "case validation summary is stale; regenerate with "
        "python scripts/generate_case_validation_summary.py"
    ]


def test_submission_ready_case_validation_summary_runs_strict_evidence_chain(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_case_validation_summary_strict")
    generator = _load_module(
        "scripts/generate_case_validation_summary.py",
        "check_submission_case_validation_summary_generator",
    )
    note_dir = ROOT / "案例正确性校验" / "tmp_strict_summary_no_refs"
    note_dir.mkdir(parents=True, exist_ok=True)
    note_docx = note_dir / "tmp_strict_summary_no_refs.docx"
    note_docx.write_bytes(b"docx placeholder")
    status = tmp_path / "case_validation_status.csv"
    summary = tmp_path / "CASE_VALIDATION_SUMMARY.md"
    note_sources = tmp_path / "case_validation_note_sources.csv"
    note_sources.write_text("path,size_bytes,sha256\n", encoding="utf-8")
    try:
        with (ROOT / "docs" / "case_validation_status.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as src:
            reader = csv.DictReader(src)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        rows[0]["validation_note_path"] = note_dir.relative_to(ROOT).as_posix()
        with status.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary.write_text(
            generator.render_markdown(rows, note_source_rows=0),
            encoding="utf-8",
        )

        errors = mod.check_case_validation_summary(status, summary, note_sources)
    finally:
        note_docx.unlink(missing_ok=True)
        note_dir.rmdir()

    assert any("strict evidence-chain errors" in err for err in errors)
    assert any("has no reference attachments" in err for err in errors)


@requires_validation_notes
def test_submission_ready_case_summary_check_works_when_script_runs_directly():
    code = (
        "import importlib.util, pathlib; "
        "root=pathlib.Path.cwd(); "
        "spec=importlib.util.spec_from_file_location('csr', root/'scripts'/'check_submission_ready.py'); "
        "mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
        "errors=mod.check_case_validation_summary("
        "root/'docs'/'case_validation_status.csv', "
        "root/'docs'/'CASE_VALIDATION_SUMMARY.md', "
        "root/'docs'/'case_validation_note_sources.csv'); "
        "print(errors); "
        "raise SystemExit(0 if errors == [] else 1)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_submission_ready_rejects_numbers_manifest_method_drift(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_drift")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "git": {"commit": "abcdef123456"},
            "experiment": {
                "n_cases_declared": 2,
                "n_methods_declared": 2,
                "n_repeats": 3,
                "budgets": [5, 10],
                "seed_start": 1000,
            },
            "results_summary": {
                "n_rows": 24,
                "n_cases_observed": 2,
                "n_methods_observed": 3,
                "status_counts": {"ok": 20, "timeout": 4},
                "failure_category_counts": {"ok": 20, "timeout": 4},
            },
        }),
        encoding="utf-8",
    )
    numbers = tmp_path / "numbers.json"
    numbers.write_text(
        json.dumps({
            "nCases": 2,
            "nMethods": 2,
            "nRepeats": 3,
            "nBudgets": 2,
            "budgetList": "5, 10",
            "seedStart": 1000,
            "totalCells": 24,
            "totalRows": 24,
            "okRows": 20,
            "timeoutRows": 4,
            "failedRows": 0,
            "nCasesCovered": 2,
            "nMethodsCovered": 2,
            "gitCommit": "abcdef1",
        }),
        encoding="utf-8",
    )

    errors = mod.check_numbers_consistency(numbers, manifest)
    manifest_errors = mod.check_manifest_result_contract(manifest)

    assert any("method-count drift" in err for err in manifest_errors)
    assert any("nMethodsCovered" in err for err in errors)


def test_submission_ready_rejects_manifest_invalid_grid_values(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_invalid_grid_values")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "git": {"commit": "abcdef123456"},
            "experiment": {
                "n_cases_declared": 1,
                "n_methods_declared": 1,
                "n_repeats": 2,
                "budgets": [5],
                "seed_start": 1000,
            },
            "results_summary": {
                "n_rows": 2,
                "n_cases_observed": 1,
                "n_methods_observed": 1,
                "status_counts": {"ok": 2},
                "failure_category_counts": {"ok": 2},
                "n_invalid_budget_mult_values": 1,
                "n_invalid_repeat_values": 1,
            },
        }),
        encoding="utf-8",
    )
    numbers = tmp_path / "numbers.json"
    numbers.write_text(
        json.dumps({
            "nCases": 1,
            "nMethods": 1,
            "nRepeats": 2,
            "nBudgets": 1,
            "budgetList": "5",
            "seedStart": 1000,
            "totalCells": 2,
            "totalRows": 2,
            "okRows": 2,
            "timeoutRows": 0,
            "failedRows": 0,
            "nCasesCovered": 1,
            "nMethodsCovered": 1,
            "gitCommit": "abcdef1",
        }),
        encoding="utf-8",
    )

    errors = mod.check_numbers_consistency(numbers, manifest)
    manifest_errors = mod.check_manifest_result_contract(manifest)

    assert errors == []
    assert any("invalid budget_mult" in err for err in manifest_errors)
    assert any("invalid repeat" in err for err in manifest_errors)


def test_submission_ready_accepts_numbers_consistent_with_manifest(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_ok")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "git": {"commit": "abcdef123456"},
            "experiment": {
                "n_cases_declared": 2,
                "n_methods_declared": 2,
                "n_repeats": 3,
                "budgets": [5, 10],
                "seed_start": 1000,
            },
            "results_summary": {
                "n_rows": 24,
                "n_cases_observed": 2,
                "n_methods_observed": 2,
                "status_counts": {"ok": 20, "timeout": 4},
                "failure_category_counts": {"ok": 20, "timeout": 4},
            },
        }),
        encoding="utf-8",
    )
    numbers = tmp_path / "numbers.json"
    numbers.write_text(
        json.dumps({
            "nCases": 2,
            "nMethods": 2,
            "nRepeats": 3,
            "nBudgets": 2,
            "budgetList": "5, 10",
            "seedStart": 1000,
            "totalCells": 24,
            "totalRows": 24,
            "okRows": 20,
            "timeoutRows": 4,
            "failedRows": 0,
            "nCasesCovered": 2,
            "nMethodsCovered": 2,
            "gitCommit": "abcdef1",
        }),
        encoding="utf-8",
    )

    assert mod.check_numbers_consistency(numbers, manifest) == []


def _complete_submission_numbers() -> dict[str, object]:
    return {
        "totalRows": 81600,
        "okRows": 80000,
        "timeoutRows": 1600,
        "failedRows": 0,
        "okFrac": 98.0,
        "nCasesCovered": 34,
        "nMethodsCovered": 20,
        "dOneRho": 0.62,
        "dOneCILo": 0.5,
        "dOneCIHi": 0.7,
        "dOneDecision": "Divergence",
        "dOneVerb": "did not transfer",
        "dOneNMethods": 20,
        "dTwoNMainTopos": 6,
        "dTwoNExploTopos": 1,
        "dTwoFracPassMain": "67%",
        "dTwoDecision": "Rule Holds",
        "dTwoMedianNStar": 10.0,
        "dThreeNSigMethods": 3,
        "dThreeNTested": 20,
        "dThreeNLowN": 0,
        "dThreeDecision": "Bias Exists",
        "dThreeDirection": "over",
        "mcNoiseNTest": 1000,
        "mcNoiseMedianRelWidthPct": 3.1,
        "mcNoiseMaxRelWidthPct": 5.2,
        "mcNoiseMaxAbsWidth": 0.003,
        "mcNoiseNCases": 34,
    }


def test_submission_ready_rejects_submission_critical_numbers_na(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_complete_bad")
    numbers = _complete_submission_numbers()
    numbers["dOneCILo"] = "N/A"
    numbers["mcNoiseNTest"] = "N/A"
    path = tmp_path / "numbers.json"
    path.write_text(json.dumps(numbers), encoding="utf-8")

    errors = mod.check_numbers_submission_completeness(path)

    assert any("dOneCILo" in err and "N/A" in err for err in errors)
    assert any("mcNoiseNTest" in err and "N/A" in err for err in errors)


def test_submission_ready_accepts_submission_complete_numbers(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_complete_ok")
    path = tmp_path / "numbers.json"
    path.write_text(json.dumps(_complete_submission_numbers()), encoding="utf-8")

    assert mod.check_numbers_submission_completeness(path) == []


def _write_numbers_with_provenance(tmp_path: Path, mod) -> Path:
    numbers = _complete_submission_numbers()
    inputs = []
    for label, name in [
        ("results_csv", "benchmark/results/full_benchmark/results.csv"),
        ("d1_correlation", "benchmark/results/ranking_reversal/correlation.json"),
        ("d2_rule_summary", "benchmark/results/sample_complexity_0.20/rule_summary.csv"),
        ("d3_bias_numbers", "paper/figdata/fig4_d3_bias.csv"),
        ("test_set_sufficiency", "benchmark/results/full_benchmark/analysis/test_set_sufficiency.csv"),
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{label}\n", encoding="utf-8")
        inputs.append({
            "label": label,
            "path": name,
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": mod._sha256(path),
        })
    numbers["_provenance"] = {
        "script": "scripts/fill_paper_numbers.py",
        "generated_at": "2026-06-01T00:00:00+00:00",
        "inputs": inputs,
    }
    path = tmp_path / "paper" / "numbers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(numbers), encoding="utf-8")
    return path


def test_submission_ready_rejects_numbers_missing_provenance(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_provenance_missing")
    paper = tmp_path / "paper"
    paper.mkdir()
    path = paper / "numbers.json"
    path.write_text(json.dumps(_complete_submission_numbers()), encoding="utf-8")
    monkeypatch_root = tmp_path

    original_root = mod.ROOT
    mod.ROOT = monkeypatch_root
    try:
        errors = mod.check_numbers_provenance(path, paper)
    finally:
        mod.ROOT = original_root

    assert any("missing _provenance" in err for err in errors)


def test_submission_ready_rejects_numbers_provenance_hash_drift(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_provenance_hash")
    paper = tmp_path / "paper"
    numbers = _write_numbers_with_provenance(tmp_path, mod)
    (tmp_path / "benchmark" / "results" / "ranking_reversal" / "correlation.json").write_text(
        "mutated\n",
        encoding="utf-8",
    )

    original_root = mod.ROOT
    mod.ROOT = tmp_path
    try:
        errors = mod.check_numbers_provenance(numbers, paper)
    finally:
        mod.ROOT = original_root

    assert any("d1_correlation" in err and "sha256 mismatch" in err for err in errors)


def test_submission_ready_rejects_numbers_provenance_missing_sha256(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_provenance_no_sha")
    paper = tmp_path / "paper"
    numbers = _write_numbers_with_provenance(tmp_path, mod)
    data = json.loads(numbers.read_text(encoding="utf-8"))
    del data["_provenance"]["inputs"][0]["sha256"]
    numbers.write_text(json.dumps(data), encoding="utf-8")

    original_root = mod.ROOT
    mod.ROOT = tmp_path
    try:
        errors = mod.check_numbers_provenance(numbers, paper)
    finally:
        mod.ROOT = original_root

    assert any("results_csv" in err and "missing or invalid sha256" in err for err in errors)


def test_submission_ready_rejects_numbers_provenance_duplicate_labels(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_provenance_dup")
    paper = tmp_path / "paper"
    numbers = _write_numbers_with_provenance(tmp_path, mod)
    data = json.loads(numbers.read_text(encoding="utf-8"))
    data["_provenance"]["inputs"].append(dict(data["_provenance"]["inputs"][0]))
    numbers.write_text(json.dumps(data), encoding="utf-8")

    original_root = mod.ROOT
    mod.ROOT = tmp_path
    try:
        errors = mod.check_numbers_provenance(numbers, paper)
    finally:
        mod.ROOT = original_root

    assert any("duplicate input labels" in err and "results_csv" in err for err in errors)


def test_submission_ready_rejects_numbers_provenance_missing_exists_true(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_provenance_exists")
    paper = tmp_path / "paper"
    numbers = _write_numbers_with_provenance(tmp_path, mod)
    data = json.loads(numbers.read_text(encoding="utf-8"))
    del data["_provenance"]["inputs"][1]["exists"]
    numbers.write_text(json.dumps(data), encoding="utf-8")

    original_root = mod.ROOT
    mod.ROOT = tmp_path
    try:
        errors = mod.check_numbers_provenance(numbers, paper)
    finally:
        mod.ROOT = original_root

    assert any("d1_correlation" in err and "exists must be true" in err for err in errors)


def test_submission_ready_accepts_clean_numbers_provenance(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_provenance_ok")
    paper = tmp_path / "paper"
    numbers = _write_numbers_with_provenance(tmp_path, mod)

    original_root = mod.ROOT
    mod.ROOT = tmp_path
    try:
        assert mod.check_numbers_provenance(numbers, paper) == []
    finally:
        mod.ROOT = original_root


def _write_minimal_lineage_artifacts(
    tmp_path: Path,
    *,
    manifest_sha: str = "a" * 64,
    numbers_sha: str | None = None,
    figdata_sha: str | None = None,
) -> tuple[Path, Path, Path]:
    if numbers_sha is None:
        numbers_sha = manifest_sha
    if figdata_sha is None:
        figdata_sha = manifest_sha

    manifest = tmp_path / "benchmark" / "results" / "full_benchmark" / "RUN_MANIFEST.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({
            "artifacts": {
                "files": [
                    {
                        "path": "results.csv",
                        "sha256": manifest_sha,
                    }
                ]
            }
        }),
        encoding="utf-8",
    )

    paper = tmp_path / "paper"
    figdata = paper / "figdata"
    figdata.mkdir(parents=True, exist_ok=True)
    numbers = paper / "numbers.json"
    numbers.write_text(
        json.dumps({
            "_provenance": {
                "inputs": [
                    {
                        "label": "results_csv",
                        "path": "benchmark/results/full_benchmark/results.csv",
                        "exists": True,
                        "sha256": numbers_sha,
                    }
                ]
            }
        }),
        encoding="utf-8",
    )
    (figdata / "figdata_manifest.json").write_text(
        json.dumps({
            "inputs": [
                {
                    "label": "results_csv",
                    "path": "benchmark/results/full_benchmark/results.csv",
                    "exists": True,
                    "sha256": figdata_sha,
                }
            ]
        }),
        encoding="utf-8",
    )
    return manifest, numbers, figdata


def test_submission_ready_accepts_paper_artifacts_bound_to_manifest_results(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_lineage_ok",
    )
    manifest, numbers, figdata = _write_minimal_lineage_artifacts(tmp_path)

    assert mod.check_paper_artifact_lineage(manifest, numbers, figdata) == []


def test_submission_ready_rejects_numbers_and_figdata_results_sha_drift(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_lineage_drift",
    )
    manifest, numbers, figdata = _write_minimal_lineage_artifacts(
        tmp_path,
        manifest_sha="a" * 64,
        numbers_sha="b" * 64,
        figdata_sha="c" * 64,
    )

    errors = mod.check_paper_artifact_lineage(manifest, numbers, figdata)

    assert any("numbers.json results_csv provenance sha256" in err for err in errors)
    assert any("figdata_manifest.json results_csv provenance sha256" in err for err in errors)


def test_submission_ready_rejects_prereg_manifest_config_drift(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_prereg_drift")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "pool": "default",
                "budgets": [5, 10],
                "n_repeats": 2,
                "n_test": 100,
                "seed_strategy": "global",
                "seed_start": 42,
                "n_cases_declared": 1,
                "case_ids_declared": ["case_a"],
            }
        }),
        encoding="utf-8",
    )

    errors = mod.check_prereg_manifest_config(manifest)

    assert any("experiment.pool" in err for err in errors)
    assert any("experiment.budgets" in err for err in errors)
    assert any("case_ids_declared count" in err for err in errors)


def test_submission_ready_rejects_prereg_manifest_method_pool_drift(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_prereg_method_drift")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "pool": "nature",
                "budgets": [5, 10, 20, 50],
                "n_repeats": 30,
                "n_test": 1000,
                "seed_strategy": "case_repeat",
                "seed_start": 1000,
                "n_cases_declared": 34,
                "case_ids_declared": PREREG_CASE_IDS,
                "n_methods_declared": 21,
                "methods_declared": PREREG_METHOD_NAMES + ["botorch_sequential"],
            }
        }),
        encoding="utf-8",
    )

    errors = mod.check_prereg_manifest_config(manifest)

    assert any("experiment.n_methods_declared" in err for err in errors)
    assert any("methods_declared count" in err for err in errors)
    assert any("methods_declared deviates from current nature_pool" in err for err in errors)
    assert any("botorch_sequential" in err for err in errors)


def test_submission_ready_accepts_prereg_manifest_config(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_prereg_ok")
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({
            "experiment": {
                "pool": "nature",
                "budgets": [5, 10, 20, 50],
                "n_repeats": 30,
                "n_test": 1000,
                "seed_strategy": "case_repeat",
                "seed_start": 1000,
                "n_cases_declared": 34,
                "case_ids_declared": [f"case_{i}" for i in range(34)],
                "n_methods_declared": 20,
                "methods_declared": PREREG_METHOD_NAMES,
            }
        }),
        encoding="utf-8",
    )

    assert mod.check_prereg_manifest_config(manifest) == []


def test_submission_ready_rejects_numbers_tex_json_mismatch(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_tex_bad")
    numbers = tmp_path / "numbers.json"
    numbers.write_text(
        json.dumps({"nCases": 34, "dOneRho": 0.903}),
        encoding="utf-8",
    )
    numbers_tex = tmp_path / "numbers.tex"
    numbers_tex.write_text(
        "\\newcommand{\\nCases}{33}\n"
        "\\newcommand{\\NumNCases}{34}\n"
        "\\newcommand{\\dOneRho}{0.903}\n"
        "\\newcommand{\\NumDOneRho}{0.904}\n",
        encoding="utf-8",
    )

    errors = mod.check_numbers_tex_consistency(numbers_tex, numbers)

    assert any("\\nCases" in err and "mismatch" in err for err in errors)
    assert any("\\NumDOneRho" in err and "mismatch" in err for err in errors)


def test_submission_ready_rejects_duplicate_numbers_tex_macros(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_numbers_tex_duplicate",
    )
    numbers = tmp_path / "numbers.json"
    numbers.write_text(
        json.dumps({
            "nCases": 34,
            "dOneRho": 0.903,
        }),
        encoding="utf-8",
    )
    numbers_tex = tmp_path / "numbers.tex"
    numbers_tex.write_text(
        "\\newcommand{\\nCases}{34}\n"
        "\\newcommand{\\NumNCases}{33}\n"
        "\\newcommand{\\NumNCases}{34}\n"
        "\\newcommand{\\dOneRho}{0.903}\n"
        "\\newcommand{\\NumDOneRho}{0.903}\n",
        encoding="utf-8",
    )

    errors = mod.check_numbers_tex_consistency(numbers_tex, numbers)

    assert any("\\NumNCases" in err and "more than once" in err for err in errors)


def test_submission_ready_accepts_numbers_tex_json_consistency(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_numbers_tex_ok")
    numbers = tmp_path / "numbers.json"
    numbers.write_text(
        json.dumps({
            "nCases": 34,
            "dOneRho": 0.903,
            "dOneCILo": "N/A",
            "dOneRankPolicy": "complete_case",
            "holmFWER": 0.05,
            "holmFamilySize": 3,
            "holmThresholds": "0.0167, 0.025, 0.05",
            "mcNoiseMaxRelWidthPct": 4.2,
        }),
        encoding="utf-8",
    )
    numbers_tex = tmp_path / "numbers.tex"
    numbers_tex.write_text(
        "\\newcommand{\\nCases}{34}\n"
        "\\newcommand{\\NumNCases}{34}\n"
        "\\newcommand{\\dOneRho}{0.903}\n"
        "\\newcommand{\\NumDOneRho}{0.903}\n"
        "\\newcommand{\\dOneCILo}{N/A}\n"
        "\\newcommand{\\NumDOneCILo}{N/A}\n"
        "\\newcommand{\\dOneRankPolicy}{complete_case}\n"
        "\\newcommand{\\NumDOneRankPolicy}{complete_case}\n"
        "\\newcommand{\\HolmFWER}{0.05}\n"
        "\\newcommand{\\NumHolmFWER}{0.05}\n"
        "\\newcommand{\\HolmFamilySize}{3}\n"
        "\\newcommand{\\NumHolmFamilySize}{3}\n"
        "\\newcommand{\\HolmThresholds}{0.0167, 0.025, 0.05}\n"
        "\\newcommand{\\NumHolmThresholds}{0.0167, 0.025, 0.05}\n"
        "\\newcommand{\\MCNoiseMaxRelWidthPct}{4.2}\n"
        "\\newcommand{\\NumMCNoiseMaxRelWidthPct}{4.2}\n",
        encoding="utf-8",
    )

    assert mod.check_numbers_tex_consistency(numbers_tex, numbers) == []


def _write_minimal_figdata(figdata_dir: Path) -> None:
    figdata_dir.mkdir(parents=True)
    fixtures = {
        "fig1_case_catalog.csv": (
            "case_id,domain,dim,n_output,topology,has_failure,supports_mf,_source,_generated_at\n"
            "case_a,structures,2,1,smooth,False,False,test,2026-06-01T00:00:00+00:00\n"
        ),
        "fig1_method_catalog.csv": (
            "method,family,surrogate,pool,_source,_generated_at\n"
            "m1,static,gp,nature,test,2026-06-01T00:00:00+00:00\n"
        ),
        "fig2_d1_spearman.csv": (
            "pairing_mode,budget_mult,spearman_rho,spearman_p,kendall_tau,analysis_type,is_primary,_source,_generated_at\n"
            "dimension,10,0.8,0.01,0.7,complete_case,True,test,2026-06-01T00:00:00+00:00\n"
        ),
        "fig2_d1_ranks.csv": (
            "pairing_mode,budget_mult,method,rank_analytical,rank_physical,analysis_type,is_primary,_source,_generated_at\n"
            "dimension,10,m1,1,1,complete_case,True,test,2026-06-01T00:00:00+00:00\n"
        ),
        "fig3_d2_learning.csv": (
            "topology,method,budget_mult,median,ci_lo,ci_hi,n_cases,_source,_generated_at\n"
            "smooth,m1,10,0.1,0.08,0.12,3,test,2026-06-01T00:00:00+00:00\n"
        ),
        "fig3_d2_rule.csv": (
            "topology,frac_pass_at_10d,median_n_star,n_cases,exploratory,analysis_type,is_primary,_source,_generated_at\n"
            "smooth,0.9,10,3,False,penalized_failed_in_denominator,True,test,2026-06-01T00:00:00+00:00\n"
        ),
        "fig4_d3_bias.csv": (
            "method,signed_bias,ci_lo,ci_hi,p_holm,bias_direction,n_runs,low_n,_source,_generated_at\n"
            "m1,-0.01,-0.02,0.0,0.2,under,10,False,test,2026-06-01T00:00:00+00:00\n"
        ),
        "fig5_mechanism.csv": (
            "topology,method,budget_mult,n_cases,mean_rank,_source,_generated_at\n"
            "smooth,m1,50,3,1.0,test,2026-06-01T00:00:00+00:00\n"
        ),
        "fig6_pareto.csv": (
            "method,nrmse_median,budget_mult,analysis_type,on_pareto,is_primary,time_median_s,_source,_generated_at\n"
            "m1,0.1,10,penalized_failed_in_denominator,True,True,12.0,test,2026-06-01T00:00:00+00:00\n"
        ),
    }
    for name, text in fixtures.items():
        (figdata_dir / name).write_text(text, encoding="utf-8")


def _write_minimal_figdata_manifest(figdata_dir: Path, mod) -> None:
    files = []
    for path in sorted(figdata_dir.glob("*.csv")):
        files.append({
            "label": "figdata_csv",
            "path": f"figdata/{path.name}",
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": mod._sha256(path),
        })
    inputs = []
    for label, name in [
        ("results_csv", "results.csv"),
        ("d1_correlation", "correlation.json"),
        ("d2_learning_curves", "learning_curves.csv"),
        ("d2_rule_summary", "rule_summary.csv"),
        ("d3_reliability_per_method", "reliability_per_method.csv"),
    ]:
        path = figdata_dir.parent / name
        if label == "results_csv":
            path.write_text(
                "case_id,method,budget_mult,repeat,status,failure_category\n"
                "c1,m1,10,0,ok,ok\n"
                "c1,m1,10,1,ok,ok\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"{name}\n", encoding="utf-8")
        inputs.append({
            "label": label,
            "path": name,
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": mod._sha256(path),
        })
    for mode in ["dimension", "topology", "random"]:
        path = figdata_dir.parent / f"pairing_{mode}" / "rank_comparison.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"rank {mode}\n", encoding="utf-8")
        inputs.append({
            "label": "d1_rank_comparison",
            "path": f"pairing_{mode}/rank_comparison.csv",
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": mod._sha256(path),
        })
    (figdata_dir / "figdata_manifest.json").write_text(
        json.dumps({
            "generated_at": "2026-06-01T00:00:00+00:00",
            "script": "scripts/prepare_figdata.py",
            "status": "complete",
            "analysis_policy": {
                "schema_version": 1,
                "d1_primary_analysis": "complete_case",
                "d2_primary_analysis": "penalized_failed_in_denominator",
                "fig6_primary_analysis": "penalized_failed_in_denominator",
                "d3_primary_cluster_by": "case_id",
                "d3_sensitivity_cluster_by": "iid",
            },
            "results_summary": {
                "exists": True,
                "n_rows": 2,
                "n_cases": 1,
                "n_methods": 1,
                "budgets": [10],
                "repeats": [0, 1],
                "status_counts": {"ok": 2},
                "failure_category_counts": {"ok": 2},
            },
            "inputs": inputs,
            "figdata_files": files,
            "errors": [],
        }),
        encoding="utf-8",
    )


def test_submission_ready_accepts_figdata_schema(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_figdata_ok")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)

    assert mod.check_figdata_schema(figdata) == []


def test_submission_ready_accepts_clean_figdata_manifest(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_figdata_manifest_ok")
    paper = tmp_path / "paper"
    figdata = paper / "figdata"
    _write_minimal_figdata(figdata)
    _write_minimal_figdata_manifest(figdata, mod)

    assert mod.check_figdata_manifest(figdata, paper) == []


def test_submission_ready_rejects_figdata_manifest_hash_mismatch(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_figdata_manifest_bad")
    paper = tmp_path / "paper"
    figdata = paper / "figdata"
    _write_minimal_figdata(figdata)
    _write_minimal_figdata_manifest(figdata, mod)
    (figdata / "fig6_pareto.csv").write_text("mutated\n", encoding="utf-8")

    errors = mod.check_figdata_manifest(figdata, paper)

    assert any("fig6_pareto.csv" in err and "sha256 mismatch" in err for err in errors)


def test_submission_ready_rejects_figdata_manifest_missing_d1_rank_inputs(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_figdata_manifest_missing_d1_rank",
    )
    paper = tmp_path / "paper"
    figdata = paper / "figdata"
    _write_minimal_figdata(figdata)
    _write_minimal_figdata_manifest(figdata, mod)
    manifest_path = figdata / "figdata_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["inputs"] = [
        item
        for item in data["inputs"]
        if not (
            item.get("label") == "d1_rank_comparison"
            and "pairing_topology" in item.get("path", "")
        )
    ]
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    errors = mod.check_figdata_manifest(figdata, paper)

    assert any("d1_rank_comparison inputs" in err and "topology" in err for err in errors)


def test_submission_ready_rejects_figdata_manifest_missing_results_summary(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_figdata_manifest_missing_results_summary",
    )
    paper = tmp_path / "paper"
    figdata = paper / "figdata"
    _write_minimal_figdata(figdata)
    _write_minimal_figdata_manifest(figdata, mod)
    manifest_path = figdata / "figdata_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.pop("results_summary")
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    errors = mod.check_figdata_manifest(figdata, paper)

    assert any("missing results_summary" in err for err in errors)


def test_submission_ready_rejects_figdata_manifest_results_summary_count_mismatch(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_figdata_manifest_results_summary_count_mismatch",
    )
    paper = tmp_path / "paper"
    figdata = paper / "figdata"
    _write_minimal_figdata(figdata)
    _write_minimal_figdata_manifest(figdata, mod)
    manifest_path = figdata / "figdata_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["results_summary"]["status_counts"] = {"ok": 1}
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    errors = mod.check_figdata_manifest(figdata, paper)

    assert any("results_summary.status_counts total 1 does not match n_rows 2" in err for err in errors)


def test_submission_ready_rejects_figdata_manifest_results_summary_current_csv_drift(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_figdata_manifest_results_summary_current_csv_drift",
    )
    paper = tmp_path / "paper"
    figdata = paper / "figdata"
    _write_minimal_figdata(figdata)
    _write_minimal_figdata_manifest(figdata, mod)
    manifest_path = figdata / "figdata_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["results_summary"]["status_counts"] = {"timeout": 2}
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    errors = mod.check_figdata_manifest(figdata, paper)

    assert any("results_summary status_counts mismatch against current results_csv" in err for err in errors)


def test_submission_ready_rejects_partial_figdata_manifest(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_figdata_manifest_partial")
    paper = tmp_path / "paper"
    figdata = paper / "figdata"
    _write_minimal_figdata(figdata)
    _write_minimal_figdata_manifest(figdata, mod)
    data = json.loads((figdata / "figdata_manifest.json").read_text(encoding="utf-8"))
    data["status"] = "partial_skip_selection"
    data["errors"] = ["partial figdata package produced with --skip"]
    (figdata / "figdata_manifest.json").write_text(json.dumps(data), encoding="utf-8")

    errors = mod.check_figdata_manifest(figdata, paper)

    assert any("generation errors" in err and "--skip" in err for err in errors)
    assert any("status is 'partial_skip_selection'" in err for err in errors)


def test_submission_ready_rejects_figdata_manifest_analysis_policy_drift(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_figdata_manifest_policy_drift",
    )
    paper = tmp_path / "paper"
    figdata = paper / "figdata"
    _write_minimal_figdata(figdata)
    _write_minimal_figdata_manifest(figdata, mod)
    manifest_path = figdata / "figdata_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["analysis_policy"]["d1_primary_analysis"] = "penalized_failed_in_denominator"
    data["analysis_policy"]["d2_primary_analysis"] = "complete_case"
    data["analysis_policy"]["fig6_primary_analysis"] = "complete_case"
    data["analysis_policy"]["d3_primary_cluster_by"] = "iid"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    errors = mod.check_figdata_manifest(figdata, paper)

    assert any("d1_primary_analysis mismatch" in err for err in errors)
    assert any("d2_primary_analysis mismatch" in err for err in errors)
    assert any("fig6_primary_analysis mismatch" in err for err in errors)
    assert any("d3_primary_cluster_by mismatch" in err for err in errors)


def _write_minimal_analysis_artifacts(
    d1_dir: Path,
    d2_dir: Path,
    d3_dir: Path,
    protocol_dir: Path,
) -> None:
    d1_dir.mkdir(parents=True)
    d2_dir.mkdir(parents=True)
    d3_dir.mkdir(parents=True)
    protocol_dir.mkdir(parents=True)
    corr_rows = []
    for mode in ("dimension", "topology", "random"):
        mode_dir = d1_dir / f"pairing_{mode}"
        mode_dir.mkdir()
        (mode_dir / "rank_comparison.csv").write_text(
            "pairing_mode,budget_mult,method,rank_analytical,rank_physical,"
            "rank_shift,analysis_type,is_primary\n"
            f"{mode},10,m1,1,2,1,complete_case,True\n"
            f"{mode},20,m1,1,1,0,complete_case,True\n",
            encoding="utf-8",
        )
        for budget in (10, 20):
            corr_rows.append({
                "pairing_mode": mode,
                "budget_mult": budget,
                "spearman_rho": 0.5,
                "spearman_rho_ci_lo": 0.1,
                "spearman_rho_ci_hi": 0.9,
                "kendall_tau": 0.4,
                "kendall_tau_ci_lo": 0.0,
                "kendall_tau_ci_hi": 0.8,
                "analysis_type": "complete_case",
                "is_primary": True,
            })
    (d1_dir / "correlation.json").write_text(
        json.dumps(corr_rows),
        encoding="utf-8",
    )
    (d2_dir / "learning_curves.csv").write_text(
        "topology,method,budget_mult,median,ci_lo,ci_hi,n_cases,n_runs\n"
        "smooth,m1,10,0.04,0.03,0.05,3,30\n",
        encoding="utf-8",
    )
    (d2_dir / "rule_summary.csv").write_text(
        "topology,analysis_type,is_primary,methods_total,methods_pass_at_10d,"
        "frac_pass_at_10d,median_n_star,max_n_star,n_methods_never_pass,"
        "n_cases,case_ids,exploratory,n_failed_cells_per_topology\n"
        "smooth,penalized_failed_in_denominator,True,1,1,1.0,10,10,0,"
        "3,case_a;case_b;case_c,False,0\n",
        encoding="utf-8",
    )
    per_method_text = (
        "method,method_family,n_runs,low_n,pf_abs_err_mean,pf_abs_err_ci_lo,"
        "pf_abs_err_ci_hi,signed_bias_mean,signed_bias_ci_lo,signed_bias_ci_hi,"
        "signed_bias_pvalue_neg,signed_bias_pvalue_pos,"
        "signed_bias_pvalue_two_sided,p_holm\n"
        "m1,gp,30,False,0.1,0.05,0.2,-0.03,-0.08,0.01,0.99,0.02,0.04,0.04\n"
    )
    (d3_dir / "reliability_per_method.csv").write_text(
        per_method_text,
        encoding="utf-8",
    )
    (d3_dir / "reliability_per_method_iid.csv").write_text(
        per_method_text,
        encoding="utf-8",
    )
    (d3_dir / "reliability_ci_sensitivity.csv").write_text(
        "method,primary_cluster_by,sensitivity_analysis,"
        "signed_bias_ci_width_cluster,signed_bias_ci_width_iid,"
        "signed_bias_ci_width_ratio_cluster_over_iid,"
        "pf_abs_err_ci_width_cluster,pf_abs_err_ci_width_iid,"
        "pf_abs_err_ci_width_ratio_cluster_over_iid\n"
        "m1,case_id,iid_bootstrap,0.2,0.1,2.0,0.3,0.2,1.5\n",
        encoding="utf-8",
    )
    (protocol_dir / "test_set_sufficiency.csv").write_text(
        "case_id,n_test,nrmse_ci_width,nrmse_relative_ci_width\n"
        "case_a,1000,0.01,0.02\n",
        encoding="utf-8",
    )
    (protocol_dir / "cross_metric_ranks.csv").write_text(
        "method,nrmse,rmse,mae,maxae,q95ae,r2,span\n"
        "m1,1,1,1,1,1,1,0\n",
        encoding="utf-8",
    )
    (protocol_dir / "cross_metric_rank_correlation.csv").write_text(
        "metric,nrmse,rmse,mae,maxae,q95ae,r2\n"
        "nrmse,1,1,1,1,1,1\n",
        encoding="utf-8",
    )
    (protocol_dir / "method_robustness.csv").write_text(
        "method,mean_nrmse,mean_per_rep_std,median_per_rep_std,max_per_rep_std,mean_cv\n"
        "m1,0.1,0.01,0.01,0.02,0.1\n",
        encoding="utf-8",
    )


def test_submission_ready_accepts_required_analysis_artifacts(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_analysis_ok")
    d1_dir = tmp_path / "d1"
    d2_dir = tmp_path / "d2"
    d3_dir = tmp_path / "d3"
    protocol_dir = tmp_path / "analysis"
    _write_minimal_analysis_artifacts(d1_dir, d2_dir, d3_dir, protocol_dir)

    assert mod.check_required_analysis_artifacts(d1_dir, d2_dir, d3_dir, protocol_dir) == []


def test_submission_ready_rejects_missing_or_invalid_analysis_artifacts(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_analysis_bad")
    d1_dir = tmp_path / "d1"
    d2_dir = tmp_path / "d2"
    d3_dir = tmp_path / "d3"
    protocol_dir = tmp_path / "analysis"
    _write_minimal_analysis_artifacts(d1_dir, d2_dir, d3_dir, protocol_dir)
    (d1_dir / "correlation.json").write_text(
        json.dumps([{
            "pairing_mode": "dimension",
            "budget_mult": 10,
            "spearman_rho": 0.5,
            "analysis_type": "legacy",
            "is_primary": True,
        }]),
        encoding="utf-8",
    )
    (d2_dir / "rule_summary.csv").write_text(
        "topology,analysis_type,is_primary,methods_total,methods_pass_at_10d,"
        "frac_pass_at_10d,median_n_star,max_n_star,n_methods_never_pass,"
        "n_cases,case_ids,exploratory,n_failed_cells_per_topology\n"
        "smooth,legacy,True,1,1,1.0,10,10,0,3,case_a;case_b;case_c,False,0\n",
        encoding="utf-8",
    )
    (d3_dir / "reliability_ci_sensitivity.csv").write_text(
        "method,primary_cluster_by,sensitivity_analysis,"
        "signed_bias_ci_width_cluster,signed_bias_ci_width_iid,"
        "signed_bias_ci_width_ratio_cluster_over_iid,"
        "pf_abs_err_ci_width_cluster,pf_abs_err_ci_width_iid,"
        "pf_abs_err_ci_width_ratio_cluster_over_iid\n"
        "m1,repeat,iid_bootstrap,0.2,notfinite,2.0,0.3,0.2,1.5\n",
        encoding="utf-8",
    )
    (protocol_dir / "test_set_sufficiency.csv").write_text(
        "case_id,n_test,nrmse_ci_width,nrmse_relative_ci_width\n"
        "case_a,500,0.01,0.02\n",
        encoding="utf-8",
    )
    (protocol_dir / "cross_metric_ranks.csv").write_text(
        "method,nrmse,rmse,mae,maxae,q95ae,r2,span\n"
        "m1,1,notfinite,1,1,1,1,0\n",
        encoding="utf-8",
    )
    (protocol_dir / "cross_metric_rank_correlation.csv").write_text(
        "metric,nrmse,rmse,mae,maxae,q95ae,r2\n"
        "nrmse,1,1,1,1,1,1\n",
        encoding="utf-8",
    )
    (protocol_dir / "method_robustness.csv").write_text(
        "method,mean_nrmse,mean_per_rep_std,median_per_rep_std,max_per_rep_std,mean_cv\n"
        "m1,0.1,0.01,0.01,0.02,notfinite\n",
        encoding="utf-8",
    )

    errors = mod.check_required_analysis_artifacts(d1_dir, d2_dir, d3_dir, protocol_dir)

    assert any("correlation.json" in err and "missing required keys" in err for err in errors)
    assert any("correlation.json" in err and "missing pairing_mode records" in err for err in errors)
    assert any("rule_summary.csv" in err and "unexpected analysis_type" in err for err in errors)
    assert any("non-finite numeric values" in err for err in errors)
    assert any("unexpected primary_cluster_by" in err for err in errors)
    assert any("no n_test=1000 rows" in err for err in errors)
    assert any("cross_metric_ranks.csv" in err and "rmse" in err for err in errors)
    assert any("method_robustness.csv" in err and "mean_cv" in err for err in errors)


def test_submission_ready_rejects_incomplete_decision_artifact_semantics(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_analysis_semantics")
    d1_dir = tmp_path / "d1"
    d2_dir = tmp_path / "d2"
    d3_dir = tmp_path / "d3"
    protocol_dir = tmp_path / "analysis"
    _write_minimal_analysis_artifacts(d1_dir, d2_dir, d3_dir, protocol_dir)

    corr = json.loads((d1_dir / "correlation.json").read_text(encoding="utf-8"))
    corr = [
        row for row in corr
        if not (row["pairing_mode"] == "topology" and row["budget_mult"] == 20)
    ]
    (d1_dir / "correlation.json").write_text(json.dumps(corr), encoding="utf-8")
    (d1_dir / "pairing_random" / "rank_comparison.csv").write_text(
        "pairing_mode,budget_mult,method,rank_analytical,rank_physical,"
        "rank_shift,analysis_type,is_primary\n"
        "dimension,10,m1,1,2,1,complete_case,True\n",
        encoding="utf-8",
    )
    (d3_dir / "reliability_per_method.csv").write_text(
        "method,method_family,n_runs,low_n,pf_abs_err_mean,pf_abs_err_ci_lo,"
        "pf_abs_err_ci_hi,signed_bias_mean,signed_bias_ci_lo,signed_bias_ci_hi,"
        "signed_bias_pvalue_two_sided,p_holm\n"
        "m1,gp,30,True,0.1,0.05,0.2,-0.03,-0.08,0.01,0.02,0.04\n",
        encoding="utf-8",
    )

    errors = mod.check_required_analysis_artifacts(d1_dir, d2_dir, d3_dir, protocol_dir)

    assert any("correlation.json" in err and "topology:20" in err for err in errors)
    assert any("rank_comparison.csv" in err and "missing budget_mult rows [20]" in err for err in errors)
    assert any("rank_comparison.csv" in err and "expected pairing_mode='random'" in err for err in errors)
    assert any("reliability_per_method.csv" in err and "low_n must equal n_runs < 10" in err for err in errors)


def test_submission_ready_rejects_d1_artifacts_without_primary_pairs(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_d1_primary_pairs")
    d1_dir = tmp_path / "d1"
    d2_dir = tmp_path / "d2"
    d3_dir = tmp_path / "d3"
    protocol_dir = tmp_path / "analysis"
    _write_minimal_analysis_artifacts(d1_dir, d2_dir, d3_dir, protocol_dir)

    corr = json.loads((d1_dir / "correlation.json").read_text(encoding="utf-8"))
    for row in corr:
        if row["pairing_mode"] == "dimension" and row["budget_mult"] == 10:
            row["is_primary"] = False
    (d1_dir / "correlation.json").write_text(json.dumps(corr), encoding="utf-8")
    (d1_dir / "pairing_topology" / "rank_comparison.csv").write_text(
        "pairing_mode,budget_mult,method,rank_analytical,rank_physical,"
        "rank_shift,analysis_type,is_primary\n"
        "topology,10,m1,1,2,1,complete_case,True\n"
        "topology,20,m1,1,1,0,complete_case,False\n",
        encoding="utf-8",
    )

    errors = mod.check_required_analysis_artifacts(d1_dir, d2_dir, d3_dir, protocol_dir)

    assert any(
        "correlation.json" in err
        and "missing primary pairing_mode/budget_mult records" in err
        and "dimension:10" in err
        for err in errors
    )
    assert any(
        "rank_comparison.csv" in err
        and "missing primary budget_mult rows [20]" in err
        and "pairing_mode='topology'" in err
        for err in errors
    )


def test_submission_ready_rejects_figdata_schema_errors(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_figdata_bad")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)
    (figdata / "fig2_d1_ranks.csv").write_text(
        "pairing_mode,budget_mult,method,rank_analytical,_source,_generated_at\n"
        "dimension,10,m1,not-a-number,test,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )
    (figdata / "fig5_mechanism.csv").write_text(
        "topology,method,budget_mult,n_cases,_source,_generated_at\n"
        "smooth,m1,50,3,test,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )
    (figdata / "fig6_pareto.csv").write_text(
        "method,nrmse_median,budget_mult,analysis_type,on_pareto,is_primary,time_median_s,_source,_generated_at\n",
        encoding="utf-8",
    )

    errors = mod.check_figdata_schema(figdata)

    assert any("rank_physical" in err for err in errors)
    assert any("rank_analytical" in err and "non-finite" in err for err in errors)
    assert any("missing at least one" in err and "mean_rank" in err for err in errors)
    assert any("fig6_pareto.csv: CSV has no data rows" in err for err in errors)


def test_submission_ready_rejects_inconsistent_d3_low_n_flag(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_figdata_low_n")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)
    (figdata / "fig4_d3_bias.csv").write_text(
        "method,signed_bias,ci_lo,ci_hi,p_holm,bias_direction,n_runs,low_n,_source,_generated_at\n"
        "m1,-0.01,-0.02,0.0,0.2,under,9,False,test,2026-06-01T00:00:00+00:00\n"
        "m2,0.01,0.0,0.02,0.2,over,10,True,test,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )

    errors = mod.check_figdata_schema(figdata)

    assert any("fig4_d3_bias.csv" in err and "low_n must equal n_runs < 10" in err for err in errors)


def test_submission_ready_rejects_non_penalized_primary_d2(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_d2_primary")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)
    (figdata / "fig3_d2_rule.csv").write_text(
        "topology,frac_pass_at_10d,median_n_star,n_cases,exploratory,analysis_type,is_primary,_source,_generated_at\n"
        "smooth,0.9,10,3,False,complete_case,True,test,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )

    errors = mod.check_figdata_schema(figdata)

    assert any("fig3_d2_rule.csv" in err and "only 'penalized_failed_in_denominator'" in err for err in errors)
    assert any("fig3_d2_rule.csv" in err and "no primary rows" in err for err in errors)


def test_submission_ready_rejects_non_complete_case_primary_d1(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_d1_primary")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)
    (figdata / "fig2_d1_spearman.csv").write_text(
        "pairing_mode,budget_mult,spearman_rho,spearman_p,kendall_tau,analysis_type,is_primary,_source,_generated_at\n"
        "dimension,10,0.8,0.01,0.7,penalized_failed_in_denominator,True,test,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )
    (figdata / "fig2_d1_ranks.csv").write_text(
        "pairing_mode,budget_mult,method,rank_analytical,rank_physical,analysis_type,is_primary,_source,_generated_at\n"
        "dimension,10,m1,1,1,penalized_failed_in_denominator,True,test,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )

    errors = mod.check_figdata_schema(figdata)

    assert any("fig2_d1_spearman.csv" in err and "only 'complete_case'" in err for err in errors)
    assert any("fig2_d1_spearman.csv" in err and "no primary rows" in err for err in errors)
    assert any("fig2_d1_ranks.csv" in err and "only 'complete_case'" in err for err in errors)


def test_submission_ready_rejects_non_penalized_primary_fig6(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_fig6_primary")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)
    (figdata / "fig6_pareto.csv").write_text(
        "method,nrmse_median,budget_mult,analysis_type,on_pareto,is_primary,time_median_s,_source,_generated_at\n"
        "m1,0.1,10,complete_case,True,True,12.0,test,2026-06-01T00:00:00+00:00\n"
        "m1,0.2,10,penalized_failed_in_denominator,True,False,12.0,test,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )

    errors = mod.check_figdata_schema(figdata)

    assert any("only 'penalized_failed_in_denominator' rows may have is_primary=True" in err for err in errors)
    assert any("no primary rows for 'penalized_failed_in_denominator'" in err for err in errors)


def test_submission_ready_rejects_invalid_fig6_timing_and_boolean_values(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_fig6_values")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)
    (figdata / "fig6_pareto.csv").write_text(
        "method,nrmse_median,budget_mult,analysis_type,on_pareto,is_primary,time_median_s,_source,_generated_at\n"
        "m1,0.1,10,penalized_failed_in_denominator,maybe,True,notfinite,test,2026-06-01T00:00:00+00:00\n"
        "m2,0.2,10,penalized_failed_in_denominator,False,sure,12.0,test,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )

    errors = mod.check_figdata_schema(figdata)

    assert any("fig6_pareto.csv" in err and "time_median_s" in err and "non-finite" in err for err in errors)
    assert any("fig6_pareto.csv" in err and "on_pareto" in err and "invalid boolean" in err for err in errors)
    assert any("fig6_pareto.csv" in err and "is_primary" in err and "invalid boolean" in err for err in errors)


def test_submission_ready_rejects_figdata_missing_provenance_columns(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_figdata_provenance")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)
    (figdata / "fig1_method_catalog.csv").write_text(
        "method,family,surrogate,pool\n"
        "m1,static,gp,nature\n",
        encoding="utf-8",
    )

    errors = mod.check_figdata_schema(figdata)

    assert any("fig1_method_catalog.csv" in err and "_source" in err for err in errors)
    assert any("fig1_method_catalog.csv" in err and "_generated_at" in err for err in errors)


def test_submission_ready_rejects_invalid_figdata_provenance_values(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_figdata_provenance_values")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)
    (figdata / "fig1_method_catalog.csv").write_text(
        "method,family,surrogate,pool,_source,_generated_at\n"
        "m1,static,gp,nature,,2026-06-01T00:00:00+00:00\n"
        "m2,static,rf,nature,test,not-a-time\n",
        encoding="utf-8",
    )

    errors = mod.check_figdata_schema(figdata)

    assert any("fig1_method_catalog.csv" in err and "empty _source" in err for err in errors)
    assert any("fig1_method_catalog.csv" in err and "invalid _generated_at" in err for err in errors)


def test_submission_ready_rejects_figdata_methods_outside_manifest_pool(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_figdata_methods")
    figdata = tmp_path / "figdata"
    _write_minimal_figdata(figdata)
    (figdata / "fig6_pareto.csv").write_text(
        "method,nrmse_median,budget_mult,analysis_type,on_pareto,is_primary,time_median_s,_source,_generated_at\n"
        "m1,0.1,10,penalized_failed_in_denominator,True,True,12.0,test,2026-06-01T00:00:00+00:00\n"
        "botorch_sequential,0.2,10,penalized_failed_in_denominator,False,True,20.0,test,2026-06-01T00:00:00+00:00\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps({"experiment": {"methods_declared": ["m1"]}}),
        encoding="utf-8",
    )

    errors = mod.check_figdata_methods(figdata, manifest)

    assert any("fig6_pareto.csv" in err and "botorch_sequential" in err for err in errors)


def test_submission_ready_rejects_missing_declared_figure_format(tmp_path):
    """A declared svg/tiff entry that is absent must fail the gate.

    Regression: ``_portable_path`` initially resolved only pdf/png, so svg and
    tiff entries silently pointed at ``paper/figN.<ext>`` and the missing-file
    check never fired for those formats.
    """
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_fig_formats")
    paper = tmp_path / "paper"
    (paper / "figures" / "pdf").mkdir(parents=True)
    (paper / "figures" / "png").mkdir(parents=True)
    (paper / "figures" / "svg").mkdir(parents=True)
    (paper / "figures" / "tiff").mkdir(parents=True)
    (paper / "figdata").mkdir()
    for i in range(1, 7):
        (paper / "figures" / "pdf" / f"fig{i}.pdf").write_text(f"pdf {i}\n", encoding="utf-8")
        (paper / "figures" / "png" / f"fig{i}.png").write_text(f"png {i}\n", encoding="utf-8")
        (paper / "figures" / "svg" / f"fig{i}.svg").write_text(f"svg {i}\n", encoding="utf-8")
        # fig1.tiff is deliberately not written.
        if i != 1:
            (paper / "figures" / "tiff" / f"fig{i}.tiff").write_text(f"tiff {i}\n", encoding="utf-8")
        (paper / "figdata" / f"fig{i}.csv").write_text(f"x\n{i}\n", encoding="utf-8")

    figures = []
    for i in range(1, 7):
        entry = {
            "figure": f"fig{i}",
            "pdf": f"figures/pdf/fig{i}.pdf",
            "svg": f"figures/svg/fig{i}.svg",
            "png": f"figures/png/fig{i}.png",
            "tiff": f"figures/tiff/fig{i}.tiff",
            "pdf_sha256": mod._sha256(paper / "figures" / "pdf" / f"fig{i}.pdf"),
            "svg_sha256": mod._sha256(paper / "figures" / "svg" / f"fig{i}.svg"),
            "png_sha256": mod._sha256(paper / "figures" / "png" / f"fig{i}.png"),
            "source_figdata": [{
                "path": f"figdata/fig{i}.csv",
                "sha256": mod._sha256(paper / "figdata" / f"fig{i}.csv"),
            }],
        }
        if i != 1:
            entry["tiff_sha256"] = mod._sha256(paper / "figures" / "tiff" / f"fig{i}.tiff")
        figures.append(entry)

    manifest = paper / "figure_manifest.json"
    manifest.write_text(json.dumps({"figures": figures, "errors": []}), encoding="utf-8")

    errors = mod.check_figure_manifest(manifest, paper)

    assert any("fig1: missing tiff artifact" in err for err in errors), errors
    # The failure must name the real figures/tiff location.  Resolving to
    # paper/fig1.tiff would also "fail", so assert the path to keep this test
    # discriminating rather than incidentally green.
    assert any(
        "fig1: missing tiff artifact" in err and "figures" in err
        for err in errors
    ), errors


def test_submission_ready_rejects_declared_figure_format_hash_drift(tmp_path):
    """A declared svg entry whose bytes change must fail the gate."""
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_fig_svg_drift")
    paper = tmp_path / "paper"
    for sub in ("pdf", "svg", "png"):
        (paper / "figures" / sub).mkdir(parents=True)
    (paper / "figdata").mkdir()
    for i in range(1, 7):
        (paper / "figures" / "pdf" / f"fig{i}.pdf").write_text(f"pdf {i}\n", encoding="utf-8")
        (paper / "figures" / "png" / f"fig{i}.png").write_text(f"png {i}\n", encoding="utf-8")
        (paper / "figures" / "svg" / f"fig{i}.svg").write_text(f"svg {i}\n", encoding="utf-8")
        (paper / "figdata" / f"fig{i}.csv").write_text(f"x\n{i}\n", encoding="utf-8")

    figures = []
    for i in range(1, 7):
        figures.append({
            "figure": f"fig{i}",
            "pdf": f"figures/pdf/fig{i}.pdf",
            "svg": f"figures/svg/fig{i}.svg",
            "png": f"figures/png/fig{i}.png",
            "pdf_sha256": mod._sha256(paper / "figures" / "pdf" / f"fig{i}.pdf"),
            "svg_sha256": ("0" * 64 if i == 2
                           else mod._sha256(paper / "figures" / "svg" / f"fig{i}.svg")),
            "png_sha256": mod._sha256(paper / "figures" / "png" / f"fig{i}.png"),
            "source_figdata": [{
                "path": f"figdata/fig{i}.csv",
                "sha256": mod._sha256(paper / "figdata" / f"fig{i}.csv"),
            }],
        })

    manifest = paper / "figure_manifest.json"
    manifest.write_text(json.dumps({"figures": figures, "errors": []}), encoding="utf-8")

    errors = mod.check_figure_manifest(manifest, paper)

    assert any("fig2: svg sha256 mismatch" in err for err in errors), errors


def test_submission_ready_rejects_figure_manifest_hash_mismatch(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_fig_manifest_bad")
    paper = tmp_path / "paper"
    (paper / "figures" / "pdf").mkdir(parents=True)
    (paper / "figures" / "png").mkdir(parents=True)
    (paper / "figdata").mkdir()
    for i in range(1, 7):
        (paper / "figures" / "pdf" / f"fig{i}.pdf").write_text(f"pdf {i}\n", encoding="utf-8")
        (paper / "figures" / "png" / f"fig{i}.png").write_text(f"png {i}\n", encoding="utf-8")
        (paper / "figdata" / f"fig{i}.csv").write_text(f"x\n{i}\n", encoding="utf-8")
    manifest = paper / "figure_manifest.json"
    figures = []
    for i in range(1, 7):
        figures.append({
            "figure": f"fig{i}",
            "pdf": f"/old/root/paper/figures/pdf/fig{i}.pdf",
            "png": f"/old/root/paper/figures/png/fig{i}.png",
            "pdf_sha256": "0" * 64 if i == 1 else mod._sha256(paper / "figures" / "pdf" / f"fig{i}.pdf"),
            "png_sha256": mod._sha256(paper / "figures" / "png" / f"fig{i}.png"),
            "source_figdata": [
                {
                    "path": f"/old/root/paper/figdata/fig{i}.csv",
                    "sha256": mod._sha256(paper / "figdata" / f"fig{i}.csv"),
                }
            ],
        })
    manifest.write_text(json.dumps({"figures": figures, "errors": []}), encoding="utf-8")

    errors = mod.check_figure_manifest(manifest, paper)

    assert any("fig1: pdf sha256 mismatch" in err for err in errors)


def test_submission_ready_rejects_partial_figure_manifest_status(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_fig_manifest_partial")
    paper = tmp_path / "paper"
    (paper / "figures" / "pdf").mkdir(parents=True)
    (paper / "figures" / "png").mkdir(parents=True)
    (paper / "figdata").mkdir()
    figdata_manifest = paper / "figdata" / "figdata_manifest.json"
    figdata_manifest.write_text(
        '{"status": "complete", "figdata_files": [], "inputs": [], "errors": []}\n',
        encoding="utf-8",
    )
    pdf = paper / "figures" / "pdf" / "fig1.pdf"
    png = paper / "figures" / "png" / "fig1.png"
    source = paper / "figdata" / "fig1.csv"
    pdf.write_text("pdf\n", encoding="utf-8")
    png.write_text("png\n", encoding="utf-8")
    source.write_text("x\n1\n", encoding="utf-8")
    manifest = paper / "figure_manifest.json"
    manifest.write_text(
        json.dumps({
            "status": "partial_fig_selection",
            "figdata_manifest": {
                "path": "figdata/figdata_manifest.json",
                "sha256": mod._sha256(figdata_manifest),
            },
            "figures": [{
                "figure": "fig1",
                "pdf": "figures/pdf/fig1.pdf",
                "png": "figures/png/fig1.png",
                "pdf_sha256": mod._sha256(pdf),
                "png_sha256": mod._sha256(png),
                "source_figdata": [{
                    "path": "figdata/fig1.csv",
                    "sha256": mod._sha256(source),
                }],
            }],
            "errors": ["partial figure package produced with --fig"],
        }),
        encoding="utf-8",
    )

    errors = mod.check_figure_manifest(manifest, paper)

    assert any("status is 'partial_fig_selection'" in err for err in errors)
    assert any("generation errors" in err and "--fig" in err for err in errors)


def test_submission_ready_rejects_figure_manifest_linked_partial_figdata(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_fig_manifest_linked_partial")
    paper = tmp_path / "paper"
    (paper / "figures" / "pdf").mkdir(parents=True)
    (paper / "figures" / "png").mkdir(parents=True)
    (paper / "figdata").mkdir()
    figdata_manifest = paper / "figdata" / "figdata_manifest.json"
    figdata_manifest.write_text(
        json.dumps({
            "status": "partial_skip_selection",
            "figdata_files": [],
            "inputs": [],
            "errors": ["partial figdata package produced with --skip"],
        }),
        encoding="utf-8",
    )
    figures = []
    for i in range(1, 7):
        pdf = paper / "figures" / "pdf" / f"fig{i}.pdf"
        png = paper / "figures" / "png" / f"fig{i}.png"
        csv = paper / "figdata" / f"fig{i}.csv"
        pdf.write_text(f"pdf {i}\n", encoding="utf-8")
        png.write_text(f"png {i}\n", encoding="utf-8")
        csv.write_text(f"x\n{i}\n", encoding="utf-8")
        figures.append({
            "figure": f"fig{i}",
            "pdf": f"figures/pdf/fig{i}.pdf",
            "png": f"figures/png/fig{i}.png",
            "pdf_sha256": mod._sha256(pdf),
            "png_sha256": mod._sha256(png),
            "source_figdata": [{
                "path": f"figdata/fig{i}.csv",
                "sha256": mod._sha256(csv),
            }],
        })
    manifest = paper / "figure_manifest.json"
    manifest.write_text(
        json.dumps({
            "status": "complete",
            "figdata_manifest": {
                "path": "figdata/figdata_manifest.json",
                "sha256": mod._sha256(figdata_manifest),
            },
            "figures": figures,
            "errors": [],
        }),
        encoding="utf-8",
    )

    errors = mod.check_figure_manifest(manifest, paper)

    assert any("linked figdata_manifest contains generation errors" in err for err in errors)
    assert any("linked figdata_manifest status is 'partial_skip_selection'" in err for err in errors)


def test_plots_nature_marks_partial_fig_selection_manifest():
    text = (ROOT / "scripts" / "plots_nature.py").read_text(encoding="utf-8")
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    assert '"status": status' in text
    assert 'status="partial_fig_selection"' in text
    assert "partial figure package produced with --fig" in text
    assert "rerun all figures before submission" in text
    assert "partial `--fig` selections are marked" in readme


def test_submission_ready_accepts_clean_figure_manifest_with_portable_paths(tmp_path):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_fig_manifest_ok")
    paper = tmp_path / "paper"
    (paper / "figures" / "pdf").mkdir(parents=True)
    (paper / "figures" / "png").mkdir(parents=True)
    (paper / "figdata").mkdir()
    policy = dict(mod.EXPECTED_FIGDATA_ANALYSIS_POLICY)
    figdata_manifest = paper / "figdata" / "figdata_manifest.json"
    figdata_manifest.write_text(
        json.dumps({
            "status": "complete",
            "analysis_policy": policy,
            "figdata_files": [],
            "inputs": [],
            "errors": [],
        }),
        encoding="utf-8",
    )
    figures = []
    for i in range(1, 7):
        pdf = paper / "figures" / "pdf" / f"fig{i}.pdf"
        png = paper / "figures" / "png" / f"fig{i}.png"
        csv = paper / "figdata" / f"fig{i}.csv"
        pdf.write_text(f"pdf {i}\n", encoding="utf-8")
        png.write_text(f"png {i}\n", encoding="utf-8")
        csv.write_text(f"x\n{i}\n", encoding="utf-8")
        figures.append({
            "figure": f"fig{i}",
            "pdf": f"/old/root/paper/figures/pdf/fig{i}.pdf",
            "png": f"/old/root/paper/figures/png/fig{i}.png",
            "pdf_sha256": mod._sha256(pdf),
            "png_sha256": mod._sha256(png),
            "source_figdata": [
                {
                    "path": f"/old/root/paper/figdata/fig{i}.csv",
                    "sha256": mod._sha256(csv),
                }
            ],
        })
    manifest = paper / "figure_manifest.json"
    manifest.write_text(
        json.dumps({
            "figdata_manifest": {
                "path": "figdata/figdata_manifest.json",
                "sha256": mod._sha256(figdata_manifest),
            },
            "analysis_policy": policy,
            "figures": figures,
            "errors": [],
        }),
        encoding="utf-8",
    )

    assert mod.check_figure_manifest(manifest, paper) == []


def test_submission_ready_rejects_figure_manifest_missing_analysis_policy(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_fig_manifest_missing_policy",
    )
    paper = tmp_path / "paper"
    (paper / "figdata").mkdir(parents=True)
    figdata_manifest = paper / "figdata" / "figdata_manifest.json"
    figdata_manifest.write_text(
        json.dumps({
            "status": "complete",
            "analysis_policy": dict(mod.EXPECTED_FIGDATA_ANALYSIS_POLICY),
            "figdata_files": [],
            "inputs": [],
            "errors": [],
        }),
        encoding="utf-8",
    )
    manifest = paper / "figure_manifest.json"
    manifest.write_text(
        json.dumps({
            "figdata_manifest": {
                "path": "figdata/figdata_manifest.json",
                "sha256": mod._sha256(figdata_manifest),
            },
            "figures": [],
            "errors": [],
        }),
        encoding="utf-8",
    )

    errors = mod.check_figure_manifest(manifest, paper)

    assert any("missing analysis_policy copied from figdata_manifest.json" in err for err in errors)


def test_submission_ready_rejects_figure_manifest_analysis_policy_drift(tmp_path):
    mod = _load_module(
        "scripts/check_submission_ready.py",
        "check_submission_fig_manifest_policy_drift",
    )
    paper = tmp_path / "paper"
    (paper / "figdata").mkdir(parents=True)
    policy = dict(mod.EXPECTED_FIGDATA_ANALYSIS_POLICY)
    figdata_manifest = paper / "figdata" / "figdata_manifest.json"
    figdata_manifest.write_text(
        json.dumps({
            "status": "complete",
            "analysis_policy": policy,
            "figdata_files": [],
            "inputs": [],
            "errors": [],
        }),
        encoding="utf-8",
    )
    drifted = dict(policy)
    drifted["d3_primary_cluster_by"] = "iid"
    manifest = paper / "figure_manifest.json"
    manifest.write_text(
        json.dumps({
            "figdata_manifest": {
                "path": "figdata/figdata_manifest.json",
                "sha256": mod._sha256(figdata_manifest),
            },
            "analysis_policy": drifted,
            "figures": [],
            "errors": [],
        }),
        encoding="utf-8",
    )

    errors = mod.check_figure_manifest(manifest, paper)

    assert any("analysis_policy does not match linked figdata_manifest.json" in err for err in errors)
    assert any("d3_primary_cluster_by mismatch" in err for err in errors)


def test_submission_ready_rejects_unsafe_figure_palette_contract(tmp_path, monkeypatch):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_palette_bad")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "plots_nature.py").write_text(
        "\n".join([
            "WONG_PALETTE = ('#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7')",
            "PALETTE_B = ('#0072B2', '#56B4E9', '#D55E00', '#CC79A7')",
            "CVD_SEQUENTIAL_CMAP = 'RdYlBu_r'",
            "LEGACY = '#4575b4'",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    errors = mod.check_figure_palette_contract()

    assert any("CVD_SEQUENTIAL_CMAP" in err for err in errors)
    assert any("legacy hard-coded colours" in err for err in errors)


def test_submission_ready_accepts_safe_figure_palette_contract(tmp_path, monkeypatch):
    mod = _load_module("scripts/check_submission_ready.py", "check_submission_palette_ok")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "plots_nature.py").write_text(
        "\n".join([
            "WONG_PALETTE = ('#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7')",
            "PALETTE_B = ('#0072B2', '#56B4E9', '#D55E00', '#CC79A7')",
            "CVD_SEQUENTIAL_CMAP = 'cividis_r'",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    assert mod.check_figure_palette_contract() == []
