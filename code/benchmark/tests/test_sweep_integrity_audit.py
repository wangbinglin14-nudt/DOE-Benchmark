from __future__ import annotations

import json

import pandas as pd
from pathlib import Path

import doe_testbed as db
from benchmark.core import case_source_fingerprint, derive_seed, method_source_fingerprint
from benchmark.methods import nature_pool
from benchmark.runtime_contracts import RESULT_SCHEMA_VERSION, protocol_digest_for_row
from scripts.audit_sweep_integrity import audit_results, main


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


def _write_one_cell_archive(tmp_path: Path) -> tuple[Path, Path, str, str, str]:
    case_id = "pressure_vessel_4d"
    method = next(m for m in nature_pool() if m.name == "random+linear")
    meta = db.get_case_meta(case_id)
    case_hash = case_source_fingerprint(db.get_case(case_id))
    method_hash = method_source_fingerprint(method)
    seed = derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000)
    row = {
        "case_id": case_id,
        "case_source_path": case_hash["case_source_path"],
        "case_source_sha256": case_hash["case_source_sha256"],
        "domain": meta.domain,
        "dim": meta.dim_input,
        "n_output": meta.dim_output,
        "method": method.name,
        "method_family": method.family,
        "method_source_path": method_hash["method_source_path"],
        "method_source_sha256": method_hash["method_source_sha256"],
        "budget_mult": 5,
        "n_train": 5 * meta.dim_input,
        "n_test": 1000,
        "repeat": 0,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 1.0,
        "test_seed": 0,
        "error": "",
        "n_output": meta.dim_output,
        "nrmse": 0.1,
        "seed": seed,
        "seed_strategy": "case_repeat",
        "seed_start": 1000,
        "protocol_digest": "",
        "worker_host": "server-a",
        "worker_pid": 1234,
        "cell_started_utc": "2026-06-06T00:00:00Z",
        "cell_finished_utc": "2026-06-06T00:00:01Z",
        "rss_gb_start": 1.0,
        "rss_gb_end": 1.1,
        "rss_gb_peak": 1.2,
    }
    row["protocol_digest"] = protocol_digest_for_row(row)
    results_csv = tmp_path / "results.csv"
    pd.DataFrame([row]).to_csv(results_csv, index=False)
    run_config = {
        "artifact": "doe-benchmark-stage2-run-config",
        "created_utc": "2026-06-06T00:00:00Z",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "n_test": 1000,
        "test_seed": 0,
        "seed_start": 1000,
        "seed_strategy": "case_repeat",
        "budgets": [5],
        "n_repeats": 1,
        "cases": [{
            "case_id": case_id,
            "case_source_path": case_hash["case_source_path"],
            "case_source_sha256": case_hash["case_source_sha256"],
            "domain": meta.domain,
            "dim": meta.dim_input,
            "n_output": meta.dim_output,
        }],
        "methods": [{
            "method": method.name,
            "method_family": method.family,
            "method_source_path": method_hash["method_source_path"],
            "method_source_sha256": method_hash["method_source_sha256"],
        }],
        "total_cells": 1,
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
            "processor": "x86_64",
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
    run_config_json = tmp_path / "run_config.json"
    run_config_json.write_text(json.dumps(run_config), encoding="utf-8")
    return results_csv, run_config_json, case_id, method.name, method.family


def test_audit_results_rejects_symlinked_results_csv(tmp_path, monkeypatch):
    results_csv = tmp_path / "results.csv"
    results_csv.write_text("case_id,method,budget_mult,repeat,status\n", encoding="utf-8")
    real_is_symlink = Path.is_symlink

    def fake_is_symlink(path):
        if Path(path) == results_csv:
            return True
        return real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

    rc, report = audit_results(results_csv)

    assert rc == 1
    assert "results.csv is a symlink and cannot be trusted" in report


def test_audit_results_rejects_duplicate_header_columns(tmp_path):
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "case_id,method,budget_mult,repeat,status,status\n"
        "case_a,method_a,5,0,ok,failed\n",
        encoding="utf-8",
    )

    rc, report = audit_results(results_csv)

    assert rc == 1
    assert "results.csv has duplicate header column name(s): ['status']" in report


def test_audit_results_rejects_blank_header_columns(tmp_path):
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "case_id,method,,repeat,status\n"
        "case_a,method_a,5,0,ok\n",
        encoding="utf-8",
    )

    rc, report = audit_results(results_csv)

    assert rc == 1
    assert "results.csv has blank header column name(s)" in report


def test_audit_results_rejects_ragged_rows_before_pandas_parsing(tmp_path):
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "case_id,method,budget_mult,repeat,status\n"
        "case_a,method_a,5,0,ok,extra_field\n",
        encoding="utf-8",
    )

    rc, report = audit_results(results_csv)

    assert rc == 1
    assert "different number of fields than the header" in report


def test_audit_results_rejects_invalid_threshold_configuration(tmp_path):
    results_csv = tmp_path / "missing.csv"

    rc, report = audit_results(
        results_csv,
        expected_budgets=[5, 5, 0],
        expected_repeats=0,
        min_repeats_per_cell=2,
        min_ok_fraction=-0.1,
        max_timeout_fraction=1.5,
    )

    assert rc == 1
    assert "expected_repeats must be >= 1" in report
    assert "min_repeats_per_cell must not exceed expected_repeats" in report
    assert "min_ok_fraction must be a finite fraction in [0, 1]" in report
    assert "max_timeout_fraction must be a finite fraction in [0, 1]" in report
    assert "expected_budgets must not contain duplicates" in report


def test_audit_cli_rejects_invalid_threshold_configuration(tmp_path, capsys):
    results_csv = tmp_path / "missing.csv"
    out = tmp_path / "report.md"

    rc = main(
        [
            str(results_csv),
            "--expected-budgets",
            "5",
            "0",
            "--expected-repeats",
            "0",
            "--min-repeats-per-cell",
            "2",
            "--min-ok-fraction",
            "-0.25",
            "--max-timeout-fraction",
            "1.25",
            "--output",
            str(out),
        ]
    )

    captured = capsys.readouterr().out
    report = out.read_text(encoding="utf-8")
    assert rc == 1
    assert "expected_repeats must be >= 1" in captured
    assert "expected_budgets must contain positive integers" in report
    assert "min_ok_fraction must be a finite fraction in [0, 1]" in report


def test_audit_results_accepts_matching_run_config_snapshot(tmp_path):
    results_csv, run_config_json, case_id, method_name, method_family = _write_one_cell_archive(tmp_path)

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: method_family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
        max_timeout_fraction=0.0,
        run_config_json=run_config_json,
        require_run_config=True,
    )

    assert rc == 0
    assert "run_config.json contract matched partial results" in report
    assert "PASS: sweep satisfies the declared integrity gate." in report


def test_audit_results_requires_run_config_when_formal_gate_demands_it(tmp_path):
    results_csv, run_config_json, case_id, method_name, method_family = _write_one_cell_archive(tmp_path)
    run_config_json.unlink()

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: method_family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
        max_timeout_fraction=0.0,
        require_run_config=True,
    )

    assert rc == 1
    assert "run_config.json is required for the final sweep integrity gate" in report


def test_audit_results_rejects_run_config_without_environment_snapshot(tmp_path):
    results_csv, run_config_json, case_id, method_name, method_family = _write_one_cell_archive(tmp_path)
    payload = json.loads(run_config_json.read_text(encoding="utf-8"))
    payload.pop("environment")
    run_config_json.write_text(json.dumps(payload), encoding="utf-8")

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: method_family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
        max_timeout_fraction=0.0,
        run_config_json=run_config_json,
        require_run_config=True,
    )

    assert rc == 1
    assert "run_config.json environment must be an object" in report


def test_audit_results_rejects_run_config_without_temp_storage_snapshot(tmp_path):
    results_csv, run_config_json, case_id, method_name, method_family = _write_one_cell_archive(tmp_path)
    payload = json.loads(run_config_json.read_text(encoding="utf-8"))
    payload.pop("temp_storage")
    run_config_json.write_text(json.dumps(payload), encoding="utf-8")

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: method_family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
        max_timeout_fraction=0.0,
        run_config_json=run_config_json,
        require_run_config=True,
    )

    assert rc == 1
    assert "run_config.json temp_storage must be an object" in report


def test_audit_results_rejects_run_config_snapshot_protocol_drift(tmp_path):
    results_csv, run_config_json, case_id, method_name, method_family = _write_one_cell_archive(tmp_path)
    payload = json.loads(run_config_json.read_text(encoding="utf-8"))
    payload["budgets"] = [10]
    payload["cases"][0]["case_source_sha256"] = "0" * 64
    run_config_json.write_text(json.dumps(payload), encoding="utf-8")

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: method_family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
        max_timeout_fraction=0.0,
        run_config_json=run_config_json,
        require_run_config=True,
    )

    assert rc == 1
    assert "budget_mult values outside run_config.json" in report
    assert "case_source_sha256" in report
    assert "run_config.json" in report


def test_audit_results_rejects_non_integer_run_config_grid_counts(tmp_path):
    results_csv, run_config_json, case_id, method_name, method_family = _write_one_cell_archive(tmp_path)
    payload = json.loads(run_config_json.read_text(encoding="utf-8"))
    payload["budgets"] = [5, 5, 10.0]
    payload["n_repeats"] = "1"
    payload["total_cells"] = 3.0
    run_config_json.write_text(json.dumps(payload), encoding="utf-8")

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: method_family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
        max_timeout_fraction=0.0,
        run_config_json=run_config_json,
        require_run_config=True,
    )

    assert rc == 1
    assert "run_config.json budgets must be positive integers" in report
    assert "run_config.json n_repeats must be a positive integer" in report
    assert "run_config.json total_cells must be a positive integer" in report


def test_audit_results_rejects_duplicate_run_config_budgets(tmp_path):
    results_csv, run_config_json, case_id, method_name, method_family = _write_one_cell_archive(tmp_path)
    payload = json.loads(run_config_json.read_text(encoding="utf-8"))
    payload["budgets"] = [5, 5]
    payload["total_cells"] = 2
    run_config_json.write_text(json.dumps(payload), encoding="utf-8")

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: method_family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
        max_timeout_fraction=0.0,
        run_config_json=run_config_json,
        require_run_config=True,
    )

    assert rc == 1
    assert "run_config.json budgets must not contain duplicates" in report


def test_audit_results_rejects_malformed_run_config_scalar_protocol_fields(tmp_path):
    results_csv, run_config_json, case_id, method_name, method_family = _write_one_cell_archive(tmp_path)
    payload = json.loads(run_config_json.read_text(encoding="utf-8"))
    payload["n_test"] = "1000"
    payload["test_seed"] = -1
    payload["seed_start"] = 1000.0
    run_config_json.write_text(json.dumps(payload), encoding="utf-8")

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: method_family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
        max_timeout_fraction=0.0,
        run_config_json=run_config_json,
        require_run_config=True,
    )

    assert rc == 1
    assert "run_config.json n_test must be a positive integer" in report
    assert "run_config.json test_seed must be a non-negative integer" in report
    assert "run_config.json seed_start must be a non-negative integer" in report


def test_audit_results_rejects_duplicate_run_config_case_or_method_entries(tmp_path):
    results_csv, run_config_json, case_id, method_name, method_family = _write_one_cell_archive(tmp_path)
    payload = json.loads(run_config_json.read_text(encoding="utf-8"))
    payload["cases"].append(dict(payload["cases"][0]))
    payload["methods"].append(dict(payload["methods"][0]))
    run_config_json.write_text(json.dumps(payload), encoding="utf-8")

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: method_family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=1.0,
        max_timeout_fraction=0.0,
        run_config_json=run_config_json,
        require_run_config=True,
    )

    assert rc == 1
    assert "run_config.json has malformed cases entries" in report
    assert "run_config.json has malformed methods entries" in report


def test_audit_results_rejects_corrupt_server_return_rows(tmp_path):
    case_id = "pressure_vessel_4d"
    method = next(m for m in nature_pool() if m.name == "random+linear")
    method_name = method.name
    case_hash = case_source_fingerprint(db.get_case(case_id))
    method_hash = method_source_fingerprint(method)
    columns = [
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
        "status",
        "failure_category",
        "t_total_s",
        "error",
        "nrmse",
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
    ]
    rows = [
        {
            "case_id": case_id,
            "case_source_path": case_hash["case_source_path"],
            "case_source_sha256": "0" * 64,
            "domain": "structures",
            "dim": 4,
            "n_output": 2,
            "method": method_name,
            "method_family": "classical",
            "method_source_path": method_hash["method_source_path"],
            "method_source_sha256": "1" * 64,
            "budget_mult": 5,
            "n_train": 20,
            "n_test": 1000,
            "repeat": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "test_seed": 0,
            "status": "ok",
            "failure_category": "ok",
            "t_total_s": 1.0,
            "error": "should be blank for ok rows",
            "nrmse": -0.1,
            "rmse": -1.0,
            "mae": 0.1,
            "maxae": 0.2,
            "q95ae": 0.15,
            "r2": 0.9,
            "worker_host": "server-a",
            "worker_pid": 1234,
            "cell_started_utc": "2026-06-06T00:00:00Z",
            "cell_finished_utc": "2026-06-06T00:00:30Z",
            "rss_gb_start": 2.0,
            "rss_gb_end": 2.5,
            "rss_gb_peak": 1.5,
        },
        {
            "case_id": case_id,
            "case_source_path": case_hash["case_source_path"],
            "case_source_sha256": case_hash["case_source_sha256"],
            "domain": "multidisciplinary",
            "dim": 4,
            "n_output": 2,
            "method": method_name,
            "method_family": "classical",
            "method_source_path": method_hash["method_source_path"],
            "method_source_sha256": method_hash["method_source_sha256"],
            "budget_mult": 5,
            "n_train": 20,
            "n_test": 1000,
            "repeat": 0,
            "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
            "test_seed": 0,
            "status": "failed",
            "failure_category": "ok",
            "t_total_s": -3.0,
            "error": "",
            "nrmse": "0.25",
            "rmse": "0.5",
            "mae": "",
            "maxae": "",
            "q95ae": "",
            "r2": "",
            "worker_host": "",
            "worker_pid": 0,
            "cell_started_utc": "not-a-time",
            "cell_finished_utc": "2026-06-06T00:00:01Z",
            "rss_gb_start": -1.0,
            "rss_gb_end": 2.0,
            "rss_gb_peak": 2.5,
        },
        {
            "case_id": case_id,
            "case_source_path": case_hash["case_source_path"],
            "case_source_sha256": case_hash["case_source_sha256"],
            "domain": "multidisciplinary",
            "dim": 4,
            "n_output": 2,
            "method": method_name,
            "method_family": "classical",
            "method_source_path": method_hash["method_source_path"],
            "method_source_sha256": method_hash["method_source_sha256"],
            "budget_mult": 5,
            "n_train": 20,
            "n_test": 999,
            "repeat": 2,
            "seed": 123,
            "test_seed": 1,
            "status": "crashed",
            "failure_category": "worker_crash",
            "t_total_s": 1.0,
            "error": "worker crashed",
            "nrmse": "",
            "rmse": "",
            "mae": "",
            "maxae": "",
            "q95ae": "",
            "r2": "",
            "worker_host": "server-a",
            "worker_pid": 1235,
            "cell_started_utc": "2026-06-06T00:01:00Z",
            "cell_finished_utc": "2026-06-06T00:00:00Z",
            "rss_gb_start": 1.0,
            "rss_gb_end": 1.2,
            "rss_gb_peak": 1.3,
        },
    ]
    results_csv = tmp_path / "results.csv"
    pd.DataFrame(rows, columns=columns).to_csv(results_csv, index=False)

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method_name],
        expected_budgets=[5],
        expected_repeats=3,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method_name: "classical"},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=3,
        min_ok_fraction=0.0,
        max_timeout_fraction=1.0,
    )

    assert rc == 1
    assert "case_source_sha256 mismatch" in report
    assert "method_source_sha256 mismatch" in report
    assert "status values must be one of" in report
    assert "failure_category must match the canonical status/error taxonomy" in report
    assert "duplicate (case, method, budget, repeat) cells found" in report
    assert "cells have fewer than 3 repeats or are missing" in report
    assert "ok rows must have finite non-negative nrmse" in report
    assert "timeout/failed rows must leave nrmse blank" in report
    assert "timeout/failed rows must leave metric column 'rmse' blank" in report
    assert "ok rows with metric column 'rmse'" in report
    assert "all rows must have finite non-negative t_total_s" in report
    assert "timeout/failed rows must include a non-empty error reason" in report
    assert "ok rows must leave error blank" in report
    assert "all rows must include a non-empty worker_host" in report
    assert "all rows must include a positive-integer worker_pid" in report
    assert "all rows must include parseable UTC cell_started_utc" in report
    assert "cell_finished_utc must not be earlier than cell_started_utc" in report
    assert "t_total_s must agree with cell_finished_utc - cell_started_utc" in report
    assert "all rows must include finite non-negative rss_gb_start" in report
    assert "all rows must have rss_gb_peak >= rss_gb_start and rss_gb_end" in report
    assert "n_test must equal declared expected_n_test=1000" in report
    assert "test_seed must equal declared expected_test_seed=0" in report
    assert "seed values do not match case_repeat seed derivation" in report


def test_audit_results_rejects_parseable_truncated_tail_row(tmp_path):
    """A dead-machine tail row with NaN runtime fields must not pass audit."""
    case_id = "pressure_vessel_4d"
    method = next(m for m in nature_pool() if m.name == "random+linear")
    case_hash = case_source_fingerprint(db.get_case(case_id))
    method_hash = method_source_fingerprint(method)
    results_csv = tmp_path / "results.csv"
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
    ]
    partial_row = [
        case_id,
        case_hash["case_source_path"],
        case_hash["case_source_sha256"],
        "structures",
        "4",
        "2",
        method.name,
        method.family,
        method_hash["method_source_path"],
        method_hash["method_source_sha256"],
    ]
    results_csv.write_text(
        ",".join(header) + "\n" + ",".join(partial_row) + "\n",
        encoding="utf-8",
    )

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method.name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method.name: method.family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=1,
        min_ok_fraction=0.0,
        max_timeout_fraction=1.0,
    )

    assert rc == 1
    assert "different number of fields than the header" in report
    assert "expected 28 field(s), observed 10" in report


def test_audit_results_rejects_boolean_integer_protocol_fields(tmp_path):
    case_id = "pressure_vessel_4d"
    method = next(m for m in nature_pool() if m.name == "random+linear")
    case_hash = case_source_fingerprint(db.get_case(case_id))
    method_hash = method_source_fingerprint(method)
    results_csv = tmp_path / "results.csv"
    pd.DataFrame([{
        "case_id": case_id,
        "case_source_path": case_hash["case_source_path"],
        "case_source_sha256": case_hash["case_source_sha256"],
        "domain": db.get_case_meta(case_id).domain,
        "dim": db.get_case_meta(case_id).dim_input,
        "n_output": db.get_case_meta(case_id).dim_output,
        "method": method.name,
        "method_family": method.family,
        "method_source_path": method_hash["method_source_path"],
        "method_source_sha256": method_hash["method_source_sha256"],
        "budget_mult": True,
        "n_train": db.get_case_meta(case_id).dim_input,
        "n_test": 1000,
        "repeat": True,
        "seed": derive_seed(case_id, 0, strategy="case_repeat", seed_start=1000),
        "test_seed": 0,
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 1.0,
        "error": "",
        "nrmse": 0.1,
        "worker_host": "server-a",
        "worker_pid": True,
        "cell_started_utc": "2026-06-06T00:00:00Z",
        "cell_finished_utc": "2026-06-06T00:00:01Z",
        "rss_gb_start": 1.0,
        "rss_gb_end": 1.1,
        "rss_gb_peak": 1.2,
    }]).to_csv(results_csv, index=False)

    rc, report = audit_results(
        results_csv,
        expected_cases=[case_id],
        expected_methods=[method.name],
        expected_budgets=[5],
        expected_repeats=1,
        expected_n_test=1000,
        expected_test_seed=0,
        expected_method_families={method.name: method.family},
        seed_strategy="case_repeat",
        seed_start=1000,
        min_repeats_per_cell=0,
        min_ok_fraction=0.0,
        max_timeout_fraction=1.0,
    )

    assert rc == 1
    assert "budget_mult values must be integer members" in report
    assert "repeat values must be integer IDs" in report
    assert "positive-integer worker_pid" in report


def test_sweep_audit_docs_record_invalid_threshold_policy():
    handoff = Path("docs/SERVER_RUN_HANDOFF.md").read_text(encoding="utf-8")
    audit = Path("docs/EXPERIMENT_BENCHMARK_AUDIT.md").read_text(encoding="utf-8")

    assert "duplicate/zero expected budgets" in handoff
    assert "repeat" in handoff
    assert "floor larger than the declared" in handoff
    assert "rejects invalid audit thresholds" in audit
