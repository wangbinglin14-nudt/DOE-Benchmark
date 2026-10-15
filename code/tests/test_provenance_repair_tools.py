from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_repair_rejects_wrong_case_grid(tmp_path: Path):
    mod = _load("scripts/merge_case_repair.py", "merge_case_repair_test")
    canonical = tmp_path / "canonical.csv"
    repair = tmp_path / "repair.csv"
    output = tmp_path / "merged.csv"
    frame = {"case_id": ["not_a_repair_case"], "method": ["m"], "budget_mult": [5], "repeat": [0]}
    pd.DataFrame(frame).to_csv(canonical, index=False)
    pd.DataFrame(frame).to_csv(repair, index=False)
    with pytest.raises(SystemExit, match="cases mismatch"):
        mod.merge(canonical, repair, output, in_place=False)
    assert not output.exists()


def test_run_config_repair_replaces_only_repaired_cases(tmp_path: Path):
    mod = _load("scripts/merge_run_config_repair.py", "merge_run_config_repair_test")
    mod.REPAIR_CASES = {"repair_case"}
    mod.EXPECTED_TOTAL_CELLS = 2
    mod.EXPECTED_REPAIR_CELLS = 1
    mod.EXPECTED_CANONICAL_CASES = 2

    common = {
        "artifact": "doe-benchmark-stage2-run-config",
        "bo_n_jobs": 1,
        "bo_timeout_s": 2.0,
        "budgets": [5],
        "environment": {"node": "linux"},
        "heartbeat_s": 1.0,
        "max_wall_clock_s": None,
        "mem_cap_gb": 1.0,
        "methods": [{"method": "m", "method_source_sha256": "msha"}],
        "n_jobs": 1,
        "n_repeats": 1,
        "n_test": 10,
        "ref_timeout_s": 2.0,
        "result_schema_version": 1,
        "seed_start": 1000,
        "seed_strategy": "case_repeat",
        "temp_storage": {"resolved_path": "/tmp"},
        "test_seed": 0,
        "thread_env": {},
        "timeout_s": 2.0,
        "torch_reproducibility": {"status": "ok"},
        "watchdog_s": 2.0,
    }
    canonical = dict(common, total_cells=2, created_utc="old", output_dir="old", skip_existing=False)
    canonical["cases"] = [
        {"case_id": "keep_case", "case_source_sha256": "keep-old"},
        {"case_id": "repair_case", "case_source_sha256": "repair-old"},
    ]
    repair = dict(common, total_cells=1, created_utc="repair", output_dir="repair", skip_existing=False)
    repair["cases"] = [{"case_id": "repair_case", "case_source_sha256": "repair-new"}]
    canonical_path = tmp_path / "canonical.json"
    repair_path = tmp_path / "repair.json"
    output_path = tmp_path / "merged.json"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    repair_path.write_text(json.dumps(repair), encoding="utf-8")

    mod.merge(canonical_path, repair_path, output_path)
    merged = json.loads(output_path.read_text(encoding="utf-8"))
    by_case = {row["case_id"]: row for row in merged["cases"]}
    assert by_case["keep_case"]["case_source_sha256"] == "keep-old"
    assert by_case["repair_case"]["case_source_sha256"] == "repair-new"
    assert merged["total_cells"] == 2
    assert merged["provenance_repair"]["repair_cases"] == ["repair_case"]
