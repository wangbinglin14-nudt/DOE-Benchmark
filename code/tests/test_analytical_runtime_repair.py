from __future__ import annotations

import json

import pandas as pd

from scripts.repair_analytical_runtime_trace import repair


def test_repair_only_raises_endpoint_inconsistent_peak(tmp_path):
    source = tmp_path / "analytical.csv"
    output = tmp_path / "repaired.csv"
    audit = tmp_path / "audit.json"
    pd.DataFrame([
        {"case_id": "a", "method": "m", "budget_mult": 10, "repeat": 0,
         "rss_gb_start": 1.2, "rss_gb_end": 1.1, "rss_gb_peak": 1.0,
         "nrmse": 0.25, "status": "ok"},
        {"case_id": "b", "method": "m", "budget_mult": 10, "repeat": 0,
         "rss_gb_start": 1.0, "rss_gb_end": 1.1, "rss_gb_peak": 1.2,
         "nrmse": 0.5, "status": "ok"},
    ]).to_csv(source, index=False)

    result = repair(source, output, audit)
    repaired = pd.read_csv(output)
    assert result["n_changed"] == 1
    assert repaired.loc[0, "rss_gb_peak"] == 1.2
    assert repaired.loc[1, "rss_gb_peak"] == 1.2
    assert repaired["nrmse"].tolist() == [0.25, 0.5]
    assert json.loads(audit.read_text(encoding="utf-8"))["scientific_columns_unchanged"] is True


def test_repair_is_noop_for_valid_trace(tmp_path):
    source = tmp_path / "analytical.csv"
    output = tmp_path / "repaired.csv"
    audit = tmp_path / "audit.json"
    pd.DataFrame([
        {"case_id": "a", "method": "m", "budget_mult": 10, "repeat": 0,
         "rss_gb_start": 1.0, "rss_gb_end": 1.1, "rss_gb_peak": 1.2,
         "nrmse": 0.25, "status": "ok"},
    ]).to_csv(source, index=False)

    result = repair(source, output, audit)
    assert result["n_changed"] == 0
    pd.testing.assert_frame_equal(pd.read_csv(source), pd.read_csv(output))
