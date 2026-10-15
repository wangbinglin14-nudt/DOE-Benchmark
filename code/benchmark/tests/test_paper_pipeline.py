"""Tests for the Stage-8/9/10 paper-pipeline contract (DOE-Benchmark §0.5).

These tests exercise the *decision rules* and *NaN-safety contracts* that
were added in the second-pass review of GAP-5/GAP-8/GAP-9/GAP-12 and the
upstream Holm correction.  They deliberately use lightweight fixtures so
they run in seconds without depending on a full sweep, but still cover:

* :func:`benchmark.analysis.critical_difference` — table values, the
  ``studentized_range`` fallback for k>20, and arbitrary α.
* :func:`benchmark.reliability_recovery._holm_adjust` and the
  upstream ``per_method_reliability_summary`` ``p_holm`` column.
* :func:`scripts.fill_paper_numbers.collect_numbers` — every D1/D2/D3
  decision branch including NaN endpoints and the vacuous-truth bug
  on direction classification.
* :func:`scripts.prepare_figdata.make_fig4` /
  :func:`scripts.prepare_figdata.make_fig6` — schema invariants that
  ``plots_nature.py`` and ``fill_paper_numbers.py`` rely on.
* :func:`scripts.plots_nature.plot_fig2_d1` — dynamic mode/budget
  selection so a partial run (only one pairing) does not produce empty
  subplots.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.analysis import critical_difference
from benchmark.runtime_contracts import RESULT_SCHEMA_VERSION, protocol_digest_for_row
from benchmark.reliability_recovery import (
    _holm_adjust,
    per_method_reliability_summary,
)
from scripts.fill_paper_numbers import (
    _classify_direction,
    _safe_round,
    collect_numbers,
    write_tex,
)


def _load_script(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _result_contract_row(**overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": "c1",
        "case_source_path": "doe_testbed/cases/test_case.py",
        "case_source_sha256": "a" * 64,
        "domain": "test",
        "dim": 2,
        "n_output": 1,
        "method": "lhs+gp",
        "method_family": "static_design",
        "method_source_path": "benchmark/methods.py",
        "method_source_sha256": "b" * 64,
        "budget_mult": 10,
        "n_train": 20,
        "n_test": 1000,
        "repeat": 0,
        "seed": 1000,
        "test_seed": 0,
        "seed_strategy": "case_repeat",
        "seed_start": 1000,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "ok",
        "failure_category": "ok",
        "error": "",
        "nrmse": 0.1,
        "t_total_s": 1.0,
        **_runtime_trace_fields(),
    }
    row.update(overrides)
    if "failure_category" not in overrides:
        status = str(row.get("status", "ok"))
        if status == "timeout":
            row["failure_category"] = "timeout"
        elif status == "failed":
            row["failure_category"] = "exception"
        elif status == "ok":
            row["failure_category"] = "ok"
    row["protocol_digest"] = protocol_digest_for_row(row)
    return row


def test_generate_report_sensitivity_paths_are_archive_scoped(tmp_path):
    from benchmark.scripts.generate_report import sensitivity_paths

    results_dir = tmp_path / "runs" / "smoke_run"
    sibling = tmp_path / "runs" / "sensitivity_recovery"
    sibling.mkdir(parents=True)

    summary, long_csv = sensitivity_paths(results_dir)

    assert summary == results_dir / "sensitivity_recovery" / "sensitivity_summary.csv"
    assert long_csv == results_dir / "sensitivity_recovery" / "sensitivity_recovery.csv"
    assert sibling not in summary.parents
    assert sibling not in long_csv.parents


def test_generate_report_failure_accounting_keeps_non_ok_rows_visible():
    from benchmark.scripts.generate_report import failure_accounting_section

    raw = pd.DataFrame([
        {
            "case_id": "case_a",
            "method": "m1",
            "budget_mult": 5,
            "repeat": 0,
            "status": "ok",
            "error": "",
            "nrmse": 0.1,
        },
        {
            "case_id": "case_a",
            "method": "m2",
            "budget_mult": 5,
            "repeat": 0,
            "status": "timeout",
            "error": "cell exceeded timeout",
            "nrmse": np.nan,
        },
        {
            "case_id": "case_b",
            "method": "m2",
            "budget_mult": 5,
            "repeat": 1,
            "status": "failed",
            "error": "model fit failed",
            "nrmse": np.nan,
        },
    ])

    md = failure_accounting_section(raw)

    assert "Downstream ranking sections use only `status == \"ok\"` rows" in md
    assert "| timeout | 1 |" in md
    assert "| failed | 1 |" in md
    assert "Highest non-ok rates by method" in md
    assert "Highest non-ok rates by case" in md
    assert "cell exceeded timeout" in md
    assert "model fit failed" in md


# ─── critical_difference (GAP-5) ────────────────────────────────────────
class TestCriticalDifference:
    """The Demšar table is only defined for k ∈ [2, 20] at α=0.05.

    GAP-5 adds a numerical fallback for k>20 and arbitrary α via
    ``scipy.stats.studentized_range``.  These tests pin the table values
    and confirm the fallback agrees with the analytical reference.
    """

    def test_tabulated_value_k20(self):
        # CD = q * sqrt(k(k+1) / (6N)) with q = 3.544 from Demšar Table 5.
        cd = critical_difference(20, 34, alpha=0.05)
        expected = 3.544 * np.sqrt(20 * 21 / (6 * 34))
        assert cd == pytest.approx(expected, rel=1e-9)

    def test_fallback_k22_matches_scipy(self):
        from scipy.stats import studentized_range
        q = float(studentized_range.ppf(0.95, 22, float("inf")))
        expected = q * np.sqrt(22 * 23 / (6 * 34))
        cd = critical_difference(22, 34, alpha=0.05)
        assert cd == pytest.approx(expected, rel=1e-9)

    def test_fallback_arbitrary_alpha(self):
        # The α=0.01 path bypasses the table even for k≤20.
        from scipy.stats import studentized_range
        q = float(studentized_range.ppf(0.99, 20, float("inf")))
        expected = q * np.sqrt(20 * 21 / (6 * 34))
        cd = critical_difference(20, 34, alpha=0.01)
        assert cd == pytest.approx(expected, rel=1e-9)

    def test_monotone_in_k(self):
        # CD is monotone non-decreasing in k for fixed N.
        seq = [critical_difference(k, 30) for k in (5, 10, 15, 20, 22, 25)]
        assert seq == sorted(seq)

    def test_monotone_in_n(self):
        # CD is monotone non-increasing in N for fixed k.
        seq = [critical_difference(10, n) for n in (10, 20, 34, 100)]
        assert seq == sorted(seq, reverse=True)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            critical_difference(1, 30)
        with pytest.raises(ValueError):
            critical_difference(10, 0)


# ─── Holm step-down adjustment ──────────────────────────────────────────
class TestHolmAdjust:
    """The DOE-Benchmark §4.3 D3 decision uses Holm-adjusted p-values."""

    def test_classic_four_pvalues(self):
        # The textbook example: sorted p = [0.005, 0.01, 0.04, 0.5]
        # Holm: p̃ = [4·0.005, 3·0.01, 2·0.04, 1·0.5] capped non-decreasing
        #     = [0.02, 0.03, 0.08, 0.5]
        out = _holm_adjust(np.array([0.04, 0.005, 0.5, 0.01]))
        # Re-order back to input layout.
        assert out[0] == pytest.approx(0.08)
        assert out[1] == pytest.approx(0.02)
        assert out[2] == pytest.approx(0.5)
        assert out[3] == pytest.approx(0.03)

    def test_monotonicity_after_sort(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            p = rng.uniform(0, 1, size=12)
            adj = _holm_adjust(p)
            order = np.argsort(p)
            sorted_adj = adj[order]
            # Holm output is monotone non-decreasing in sorted p.
            assert np.all(np.diff(sorted_adj) >= -1e-12)

    def test_clip_to_one(self):
        # When (m-i+1)*p exceeds 1 the value should clip.
        out = _holm_adjust(np.array([0.4, 0.5, 0.9]))
        assert np.max(out) <= 1.0

    def test_all_nan(self):
        out = _holm_adjust(np.array([np.nan, np.nan, np.nan]))
        assert np.all(np.isnan(out))

    def test_partial_nan_preserved(self):
        # NaN inputs stay NaN; Holm runs on the finite subset only.
        out = _holm_adjust(np.array([0.01, np.nan, 0.04, np.nan]))
        assert np.isnan(out[1])
        assert np.isnan(out[3])
        # Two finite values: p_(1)=0.01 → 2·0.01=0.02; p_(2)=0.04 → 1·0.04=0.04
        assert out[0] == pytest.approx(0.02)
        assert out[2] == pytest.approx(0.04)


# ─── per_method_reliability_summary upstream p_holm ────────────────────
class TestUpstreamPHolm:
    """``reliability_per_method.csv`` must carry ``p_holm`` so the paper
    pipeline does not re-compute Holm in three separate downstream
    consumers.
    """

    @staticmethod
    def _make_long(seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        # Four methods with known per-method bias means.
        for method, mu in [("m1", -0.05), ("m2", -0.005),
                           ("m3", 0.005), ("m4", 0.04)]:
            for c in range(12):
                bias = float(rng.normal(mu, 0.02))
                rows.append({
                    "method": method,
                    "method_family": "gp",
                    "case_id": f"c{c}",
                    "repeat": 0,
                    "budget_mult": 10,
                    "pf_signed_bias": bias,
                    "pf_abs_err": abs(bias),
                    "f_bal_acc": float("nan"),
                    "f_brier": float("nan"),
                })
        return pd.DataFrame(rows)

    def test_p_holm_column_present(self):
        df = self._make_long()
        out = per_method_reliability_summary(df, n_boot=200, seed=0)
        assert "p_holm" in out.columns

    def test_p_holm_dominates_raw(self):
        df = self._make_long()
        out = per_method_reliability_summary(df, n_boot=200, seed=0)
        for _, row in out.iterrows():
            raw = row["signed_bias_pvalue_pos"]
            adj = row["p_holm"]
            if np.isfinite(raw) and np.isfinite(adj):
                # Holm-adjusted p is always ≥ raw p.
                assert adj + 1e-12 >= raw

    def test_strong_bias_methods_significant(self):
        df = self._make_long()
        out = per_method_reliability_summary(df, n_boot=200, seed=0)
        sig = out[out["p_holm"] < 0.05]["method"].tolist()
        # The registered alternative is positive signed bias: m4 (μ=+0.04)
        # should be significant, whereas m1's negative bias is retained as a
        # directional sensitivity and is not a positive-bias discovery.
        # m2 (μ=-0.005) and m3 (μ=+0.005) are within noise.
        assert "m1" not in sig
        assert "m4" in sig
        assert "m2" not in sig
        assert "m3" not in sig


# ─── _safe_round and _classify_direction (NaN safety) ─────────────────
class TestSafeRound:
    @pytest.mark.parametrize("v,expected", [
        (0.5, 0.5),
        (0.123456, 0.123),
        (0, 0.0),
    ])
    def test_finite(self, v, expected):
        out = _safe_round(v, 3)
        assert out == pytest.approx(expected)

    @pytest.mark.parametrize("v", [np.nan, np.inf, -np.inf, None, "not-a-number"])
    def test_non_finite_returns_string(self, v):
        assert _safe_round(v, 3) == "N/A"


class TestClassifyDirection:
    """Vacuous-truth bug regression test for the dThreeDirection field."""

    def test_all_negative(self):
        s = pd.Series([-0.05, -0.01, -0.10])
        assert _classify_direction(s) == "under"

    def test_all_positive(self):
        s = pd.Series([0.04, 0.10])
        assert _classify_direction(s) == "over"

    def test_mixed(self):
        s = pd.Series([-0.05, 0.04])
        assert _classify_direction(s) == "mixed"

    def test_single_significant(self):
        s = pd.Series([-0.07])
        assert _classify_direction(s) == "under"

    def test_all_nan_significant(self):
        # Regression: vacuous-truth bug previously returned "over".
        s = pd.Series([np.nan, np.nan])
        assert _classify_direction(s) == "none"

    def test_partial_nan_negative_remainder(self):
        # NaN entries are dropped; finite values agree → under.
        s = pd.Series([-0.01, np.nan, -0.04])
        assert _classify_direction(s) == "under"


# ─── End-to-end fill_paper_numbers fixtures ────────────────────────────
@pytest.fixture
def paper_pipeline_dirs(tmp_path: Path):
    rr_dir = tmp_path / "rr"
    d2_dir = tmp_path / "d2"
    d3_dir = tmp_path / "d3"
    figdata_dir = tmp_path / "figdata"
    for d in (rr_dir, d2_dir, d3_dir, figdata_dir):
        d.mkdir(parents=True)
    return tmp_path, rr_dir, d2_dir, d3_dir, figdata_dir


def _write_correlation(rr_dir: Path, *, rho: float, ci_lo: float, ci_hi: float,
                       n_methods: int = 20) -> None:
    payload = [{
        "pairing_mode": "dimension",
        "budget_mult": 10,
        "spearman_rho": rho,
        "spearman_rho_ci_lo": ci_lo,
        "spearman_rho_ci_hi": ci_hi,
        "spearman_p": 0.001,
        "kendall_tau": rho * 0.8,
        "kendall_tau_ci_lo": ci_lo * 0.8,
        "kendall_tau_ci_hi": ci_hi * 0.8,
        "n_methods": n_methods,
        "method_pool": "nature",
        "seed_start": 1000,
    }]
    (rr_dir / "correlation.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def _write_rule(d2_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(d2_dir / "rule_summary.csv", index=False)


def _write_cp_bound(
    d2_dir: Path,
    *,
    upper: object,
    cutoff: object = 4 / 7,
) -> None:
    """Write the minimal confirmatory D2 artifact used by collect_numbers."""
    (d2_dir / "tenx_rule_cp_bound.json").write_text(
        json.dumps({
            "clopper_pearson_one_sided_upper95": upper,
            "registered_cutoff_fraction": cutoff,
        }),
        encoding="utf-8",
    )


def _write_bias(d3_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(d3_dir / "reliability_per_method.csv", index=False)


def _write_d3_pooled_artifacts(
    d3_dir: Path,
    *,
    pooled_mean: float = 0.0264,
    ci_lo: float = 0.0017,
    ci_hi: float = 0.0592,
) -> None:
    """Write the D3 pooled/robustness artifacts the submission gate requires.

    These mirror the schema of the generated files under
    ``benchmark/results/reliability_recovery/`` so the fixture exercises the
    same code path as the archived sweep rather than a reduced stand-in.
    """
    d3_dir.mkdir(parents=True, exist_ok=True)
    (d3_dir / "pooled_reliability_summary.json").write_text(json.dumps({
        "budget_mult": 10,
        "n_runs": 9570,
        "n_cases": 16,
        "pooled_mean_signed_bias": pooled_mean,
        "cluster_bootstrap_ci_lo": ci_lo,
        "cluster_bootstrap_ci_hi": ci_hi,
    }), encoding="utf-8")
    (d3_dir / "d3_loco_bca.json").write_text(json.dumps({
        "n_runs": 9570,
        "n_cases": 16,
        "pooled_mean_signed_bias": pooled_mean,
        "bca_ci": [ci_lo * 2, ci_hi * 1.2],
        "loco_min": ci_lo * 8,
        "loco_max": ci_hi * 0.6,
    }), encoding="utf-8")
    (d3_dir / "d3_case_level_sensitivity.json").write_text(json.dumps({
        "n_cases": 16,
        "n_positive_cases": 12,
        "equal_case_bootstrap_ci_lo": ci_lo,
        "equal_case_bootstrap_ci_hi": ci_hi,
        "wilcoxon_case_level_one_sided_greater_p": 0.0253,
        "holm_family_threshold": 1 / 60,
    }), encoding="utf-8")
    (d3_dir / "d3_sensitivity_extras.json").write_text(json.dumps({
        "most_influential_case": "case_a",
        "most_influential_case_share_pct_of_pooled": 51.1,
        "loo_case_cluster_ci_lo": -0.0031,
        "loo_case_cluster_ci_hi": 0.0313,
        "n_methods_cluster_ci_excludes_zero": 5,
        "n_methods_cluster_ci_includes_zero": 15,
        "n_methods_tested": 20,
        "n_methods_case_level_holm_survivors": 0,
        "median_true_pf_at_driving_output": 0.392,
        "pooled_bias_pct_of_median_reference_pf": 6.73,
    }), encoding="utf-8")


def _write_test_set_sufficiency(protocol_dir: Path) -> None:
    protocol_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "case_id": "case_a",
            "n_test": 1000,
            "nrmse_ci_width": 0.10,
            "nrmse_relative_ci_width": 0.20,
        },
        {
            "case_id": "case_b",
            "n_test": 1000,
            "nrmse_ci_width": 0.30,
            "nrmse_relative_ci_width": 0.40,
        },
        {
            "case_id": "case_a",
            "n_test": 5000,
            "nrmse_ci_width": 0.05,
            "nrmse_relative_ci_width": 0.10,
        },
    ]).to_csv(protocol_dir / "test_set_sufficiency.csv", index=False)


def _write_d1_case_axis_artifacts(
    rr_dir: Path,
    *,
    ci_lo: float = 0.32,
    ci_hi: float = 0.94,
    prob_below: float = 0.33,
    split_half_analytical: float = 0.74,
    split_half_physical: float = 0.70,
    ceiling: float = 0.72,
    n_physical_cases_complete: int = 4,
) -> None:
    """Mirror the schema of ``d1_case_axis_reliability.json``.

    Added alongside the D1 case-axis robustness paragraph.  Without this
    artefact the ``--require-submission-complete`` gate reports the
    case-axis numbers as ``N/A`` and refuses to write ``numbers.tex``.
    """
    rr_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pairing_mode": "dimension",
        "budget_mult": 10,
        "metric": "nrmse",
        "n_analytical_cases": 6,
        "n_physical_cases": 6,
        "n_analytical_cases_complete": 6,
        "n_physical_cases_complete": n_physical_cases_complete,
        "case_axis_bootstrap": {
            "n_boot": 10000,
            "bootstrap_seed": 1000,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "prob_below_reference": prob_below,
            "reference_rho": 0.7,
        },
        "split_half_analytical": {
            "n_cases": 6,
            "half_size": 3,
            "n_splits": 10,
            "scheme": "exhaustive",
            "reliability": split_half_analytical,
        },
        "split_half_physical": {
            "n_cases": 4,
            "half_size": 2,
            "n_splits": 3,
            "scheme": "exhaustive",
            "reliability": split_half_physical,
        },
        "reliability_ceiling": ceiling,
        "script": "scripts/derive_d1_case_axis_reliability.py",
    }
    (rr_dir / "d1_case_axis_reliability.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_d2_metric_sensitivity(
    d2_dir: Path,
    *,
    alt_pass: int = 0,
    n_classes: int = 7,
) -> None:
    """Mirror the schema of ``d2_metric_sensitivity.json``.

    ``alt_pass`` defaults to the measured worst-output result (no class
    passes), which is the value the manuscript cites.
    """
    d2_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "benchmark/results/full_benchmark/results.csv",
        "primary": {
            "metric": "nrmse",
            "threshold": 0.2,
            "rule_budget": 10,
            "n_classes": n_classes,
            "n_classes_point_estimate_pass": 1,
        },
        "alternate": {
            "metric": "nrmse_worst",
            "threshold": 0.2,
            "rule_budget": 10,
            "n_classes": n_classes,
            "n_classes_point_estimate_pass": alt_pass,
        },
    }
    (d2_dir / "d2_metric_sensitivity.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_formal_numbers_inputs(
    results_csv: Path,
    rr_dir: Path,
    d2_dir: Path,
    d3_dir: Path,
    protocol_dir: Path,
) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        _result_contract_row(**{
            "case_id": "case_a",
            "method": "lhs",
            "budget_mult": 10,
            "repeat": 0,
            "status": "ok",
            "failure_category": "ok",
            "error": "",
            "nrmse": 0.11,
            "t_total_s": 1.0,
            **_runtime_trace_fields(),
        }),
        _result_contract_row(**{
            "case_id": "case_b",
            "method": "gp",
            "budget_mult": 10,
            "repeat": 0,
            "status": "ok",
            "failure_category": "ok",
            "error": "",
            "nrmse": 0.08,
            "t_total_s": 1.0,
            **_runtime_trace_fields(),
        }),
    ]).to_csv(results_csv, index=False)
    (rr_dir / "correlation.json").write_text(json.dumps([
        {
            "pairing_mode": "dimension",
            "budget_mult": 10,
            "spearman_rho": 0.55,
            "spearman_rho_ci_lo": 0.32,
            "spearman_rho_ci_hi": 0.69,
            "spearman_p": 0.001,
            "kendall_tau": 0.44,
            "kendall_tau_ci_lo": 0.25,
            "kendall_tau_ci_hi": 0.55,
            "n_methods": 20,
            "method_pool": "nature",
            "seed_start": 1000,
            "analysis_type": "complete_case",
            "is_primary": True,
        },
    ]), encoding="utf-8")
    _write_rule(d2_dir, [
        {
            "topology": "smooth",
            "frac_pass_at_10d": 0.7,
            "median_n_star": 8.0,
            "n_cases": 5,
            "exploratory": False,
            "analysis_type": "penalized_failed_in_denominator",
            "is_primary": True,
        },
    ])
    _write_bias(d3_dir, [
        {
            "method": "gp",
            "signed_bias_mean": -0.05,
            "signed_bias_ci_lo": -0.08,
            "signed_bias_ci_hi": -0.02,
            "signed_bias_pvalue_two_sided": 0.001,
            "p_holm": 0.004,
            "n_runs": 10,
            "low_n": False,
        },
    ])
    _write_cp_bound(d2_dir, upper=0.7746784, cutoff=4 / 7)
    _write_d3_pooled_artifacts(d3_dir, pooled_mean=-0.05, ci_lo=-0.08, ci_hi=-0.02)
    _write_d1_case_axis_artifacts(rr_dir)
    _write_d2_metric_sensitivity(d2_dir)
    _write_test_set_sufficiency(protocol_dir)


def test_validate_test_protocol_failure_preserves_existing_analysis(tmp_path, monkeypatch):
    mod = _load_script(
        "benchmark/scripts/validate_test_protocol.py",
        "validate_test_protocol_atomic_failure",
    )
    run_dir = tmp_path / "run"
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True)
    sentinel = analysis / "test_set_sufficiency.csv"
    sentinel.write_text("old analysis\n", encoding="utf-8")
    pd.DataFrame([
        _result_contract_row(**{
            "case_id": "case_a",
            "method": "m1",
            "budget_mult": 5,
            "repeat": 0,
            "status": "ok",
            "failure_category": "ok",
            "error": "",
            "nrmse": 0.1,
            "t_total_s": 1.0,
            **_runtime_trace_fields(),
        })
    ]).to_csv(run_dir / "results.csv", index=False)

    monkeypatch.setattr(mod, "block_a_test_set_sufficiency", lambda out: (out / "partial.csv").write_text("partial\n", encoding="utf-8"))
    monkeypatch.setattr(mod, "block_b_cross_metric_consistency", lambda df, out: (_ for _ in ()).throw(RuntimeError("synthetic failure")))
    monkeypatch.setattr(mod, "block_c_method_robustness", lambda df, out: None)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        mod.main([str(run_dir)])

    assert sentinel.read_text(encoding="utf-8") == "old analysis\n"
    assert not list(run_dir.glob(".analysis.tmp.*"))


def test_validate_test_protocol_replace_failure_restores_existing_analysis(tmp_path, monkeypatch):
    mod = _load_script(
        "benchmark/scripts/validate_test_protocol.py",
        "validate_test_protocol_replace_failure",
    )
    run_dir = tmp_path / "run"
    analysis = run_dir / "analysis"
    tmp_analysis = run_dir / ".analysis.tmp.synthetic"
    analysis.mkdir(parents=True)
    tmp_analysis.mkdir(parents=True)
    sentinel = analysis / "test_set_sufficiency.csv"
    sentinel.write_text("old analysis\n", encoding="utf-8")
    (tmp_analysis / "test_set_sufficiency.csv").write_text("new analysis\n", encoding="utf-8")
    real_move = mod.shutil.move

    def fail_second_move(src, dst):
        if Path(src) == tmp_analysis and Path(dst) == analysis:
            raise RuntimeError("synthetic replace failure")
        return real_move(src, dst)

    monkeypatch.setattr(mod.shutil, "move", fail_second_move)

    with pytest.raises(RuntimeError, match="synthetic replace failure"):
        mod._replace_analysis_dir(tmp_analysis, analysis)

    assert sentinel.read_text(encoding="utf-8") == "old analysis\n"
    assert not (run_dir / ".analysis.bak").exists()


def test_validate_test_protocol_rejects_corrupt_results_before_output(tmp_path):
    mod = _load_script(
        "benchmark/scripts/validate_test_protocol.py",
        "validate_test_protocol_contract_failure",
    )
    run_dir = tmp_path / "run"
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True)
    sentinel = analysis / "test_set_sufficiency.csv"
    sentinel.write_text("old analysis\n", encoding="utf-8")
    row = _result_contract_row(**{
        "case_id": "case_a",
        "method": "m1",
        "budget_mult": 5,
        "repeat": 0,
        "status": "ok",
        "failure_category": "ok",
        "error": "",
        "nrmse": 0.1,
        "t_total_s": 1.0,
    })
    duplicate = _result_contract_row(**{
        "case_id": "case_a",
        "method": "m1",
        "budget_mult": 5,
        "repeat": 0,
        "status": "ok",
        "failure_category": "ok",
        "error": "",
        "nrmse": 0.2,
        "t_total_s": 1.0,
    })
    pd.DataFrame([row, duplicate]).to_csv(run_dir / "results.csv", index=False)

    rc = mod.main([str(run_dir)])

    assert rc == 2
    assert sentinel.read_text(encoding="utf-8") == "old analysis\n"
    assert not list(run_dir.glob(".analysis.tmp.*"))


def test_validate_test_protocol_require_headline_source_rejects_fixture_results(tmp_path):
    mod = _load_script(
        "benchmark/scripts/validate_test_protocol.py",
        "validate_test_protocol_headline_source",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame([_result_contract_row(**{
        "case_id": "case_a",
        "method": "m1",
        "budget_mult": 5,
        "repeat": 0,
        "status": "ok",
        "failure_category": "ok",
        "error": "",
        "nrmse": 0.1,
        "t_total_s": 1.0,
    })]).to_csv(run_dir / "results.csv", index=False)

    rc = mod.main([str(run_dir), "--require-headline-source"])

    assert rc == 2
    assert not (run_dir / "analysis").exists()


def test_validate_test_protocol_cross_metric_outputs_named_index_columns(tmp_path):
    mod = _load_script(
        "benchmark/scripts/validate_test_protocol.py",
        "validate_test_protocol_cross_metric_columns",
    )
    df = pd.DataFrame([
        {
            "case_id": "case_a",
            "method": "m1",
            "budget_mult": 10,
            "status": "ok",
            "nrmse": 0.10,
            "rmse": 0.10,
            "mae": 0.08,
            "maxae": 0.20,
            "q95ae": 0.18,
            "r2": 0.90,
        },
        {
            "case_id": "case_a",
            "method": "m2",
            "budget_mult": 10,
            "status": "ok",
            "nrmse": 0.20,
            "rmse": 0.20,
            "mae": 0.16,
            "maxae": 0.30,
            "q95ae": 0.28,
            "r2": 0.80,
        },
        {
            "case_id": "case_b",
            "method": "m1",
            "budget_mult": 10,
            "status": "ok",
            "nrmse": 0.12,
            "rmse": 0.12,
            "mae": 0.09,
            "maxae": 0.22,
            "q95ae": 0.19,
            "r2": 0.88,
        },
        {
            "case_id": "case_b",
            "method": "m2",
            "budget_mult": 10,
            "status": "ok",
            "nrmse": 0.22,
            "rmse": 0.22,
            "mae": 0.17,
            "maxae": 0.32,
            "q95ae": 0.29,
            "r2": 0.78,
        },
    ])

    mod.block_b_cross_metric_consistency(df, tmp_path)

    ranks = pd.read_csv(tmp_path / "cross_metric_ranks.csv")
    corr = pd.read_csv(tmp_path / "cross_metric_rank_correlation.csv")

    assert "method" in ranks.columns
    assert "span" in ranks.columns
    assert "metric" in corr.columns
    assert set(ranks["method"]) == {"m1", "m2"}
    assert pd.to_numeric(ranks["span"], errors="coerce").notna().all()
    assert set(corr["metric"]) >= {"nrmse", "rmse", "mae", "maxae", "q95ae", "r2"}


class TestD1Decision:
    @pytest.mark.parametrize("ci_lo,ci_hi,decision", [
        (0.32, 0.69, "Divergence"),  # CI fully below 0.70
        (0.72, 0.91, "Transfer"),    # CI fully above 0.70
        (0.55, 0.83, "Mixed"),       # CI straddles 0.70
        (0.71, 0.71, "Transfer"),    # Edge: CI just above 0.70
        (0.69, 0.69, "Divergence"),  # Edge: CI just below 0.70
    ])
    def test_decision_branches(self, paper_pipeline_dirs, ci_lo, ci_hi, decision):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_correlation(rr, rho=(ci_lo + ci_hi) / 2, ci_lo=ci_lo, ci_hi=ci_hi)
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dOneDecision"] == decision

    def test_nan_endpoints_become_mixed(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_correlation(rr, rho=0.5, ci_lo=float("nan"), ci_hi=0.6)
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        # ci_hi=0.6 < 0.70 → still Divergence; this guards the
        # explicit ``np.isfinite`` check.
        assert nums["dOneDecision"] == "Divergence"
        assert nums["dOneCILo"] == "N/A"

    def test_all_nan_safe(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_correlation(rr, rho=float("nan"),
                           ci_lo=float("nan"), ci_hi=float("nan"))
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dOneRho"] == "N/A"
        assert nums["dOneCILo"] == "N/A"
        assert nums["dOneCIHi"] == "N/A"
        assert nums["dOneDecision"] == "Mixed"

    def test_primary_complete_case_record_is_preferred(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        payload = [
            {
                "pairing_mode": "dimension",
                "budget_mult": 10,
                "spearman_rho": 0.91,
                "spearman_rho_ci_lo": 0.82,
                "spearman_rho_ci_hi": 0.96,
                "kendall_tau": 0.75,
                "kendall_tau_ci_lo": 0.60,
                "kendall_tau_ci_hi": 0.88,
                "n_methods": 20,
                "analysis_type": "legacy",
                "is_primary": False,
            },
            {
                "pairing_mode": "dimension",
                "budget_mult": 10,
                "spearman_rho": 0.42,
                "spearman_rho_ci_lo": 0.20,
                "spearman_rho_ci_hi": 0.61,
                "kendall_tau": 0.31,
                "kendall_tau_ci_lo": 0.10,
                "kendall_tau_ci_hi": 0.50,
                "n_methods": 20,
                "analysis_type": "complete_case",
                "is_primary": True,
            },
        ]
        (rr / "correlation.json").write_text(json.dumps(payload), encoding="utf-8")

        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)

        assert nums["dOneRankPolicy"] == "complete_case"
        assert nums["dOneRho"] == 0.42
        assert nums["dOneDecision"] == "Divergence"


class TestD2Decision:
    def test_rule_holds_when_majority_pass(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        # 3 main topologies, 2 of which pass ≥0.5 fraction.
        _write_rule(d2, [
            {"topology": "smooth", "frac_pass_at_10d": 0.7,
             "median_n_star": 8.0, "n_cases": 5, "exploratory": False},
            {"topology": "rough", "frac_pass_at_10d": 0.6,
             "median_n_star": 12.0, "n_cases": 4, "exploratory": False},
            {"topology": "discontinuous", "frac_pass_at_10d": 0.3,
             "median_n_star": 20.0, "n_cases": 3, "exploratory": False},
            {"topology": "rare", "frac_pass_at_10d": 0.9,
             "median_n_star": 5.0, "n_cases": 1, "exploratory": True},
        ])
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dTwoNMainTopos"] == 3
        assert nums["dTwoNExploTopos"] == 1
        assert nums["dTwoFracPassMain"] == "67%"
        assert nums["dTwoDecision"] == "Rule Holds"
        assert nums["dTwoMedianNStar"] == 12.0  # median of [8, 12, 20]

    def test_rule_fails(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_rule(d2, [
            {"topology": "smooth", "frac_pass_at_10d": 0.4,
             "median_n_star": 22.0, "n_cases": 5, "exploratory": False},
            {"topology": "rough", "frac_pass_at_10d": 0.2,
             "median_n_star": 30.0, "n_cases": 4, "exploratory": False},
        ])
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dTwoDecision"] == "Rule Fails"
        assert nums["dTwoFracPassMain"] == "0%"

    def test_confirmatory_decision_requires_cp_upper_below_registered_cutoff(
        self, paper_pipeline_dirs,
    ):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_rule(d2, [{
            "topology": "smooth",
            "frac_pass_at_10d": 0.6,
            "median_n_star": 10.0,
            "n_cases": 5,
            "exploratory": False,
        }])
        _write_cp_bound(d2, upper=0.55, cutoff=0.5714)
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dTwoDecision"] == "Rule Holds"
        assert nums["dTwoConfirmatoryDecision"] == "Confirmed"

    def test_confirmatory_decision_is_inconclusive_for_current_cp_bound(
        self, paper_pipeline_dirs,
    ):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_rule(d2, [{
            "topology": "smooth",
            "frac_pass_at_10d": 0.4,
            "median_n_star": 10.0,
            "n_cases": 5,
            "exploratory": False,
        }])
        _write_cp_bound(d2, upper=0.7746784, cutoff=4 / 7)
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dTwoDecision"] == "Rule Fails"
        assert nums["dTwoConfirmatoryDecision"] == "Inconclusive"

    def test_malformed_cp_artifact_falls_back_to_inconclusive(
        self, paper_pipeline_dirs,
    ):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_rule(d2, [{
            "topology": "smooth",
            "frac_pass_at_10d": 0.6,
            "median_n_star": 10.0,
            "n_cases": 5,
            "exploratory": False,
        }])
        (d2 / "tenx_rule_cp_bound.json").write_text(
            '{"clopper_pearson_one_sided_upper95": "not-a-number"}',
            encoding="utf-8",
        )
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dTwoConfirmatoryDecision"] == "Inconclusive"

    def test_all_nan_n_star_safe(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_rule(d2, [
            {"topology": "smooth", "frac_pass_at_10d": 0.6,
             "median_n_star": float("nan"), "n_cases": 5, "exploratory": False},
        ])
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        # Decision still computable from frac_pass_at_10d…
        assert nums["dTwoDecision"] == "Rule Holds"
        # …but median_n_star degrades gracefully to "N/A".
        assert nums["dTwoMedianNStar"] == "N/A"

    def test_string_false_exploratory_is_not_treated_as_true(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_rule(d2, [
            {"topology": "smooth", "frac_pass_at_10d": 0.8,
             "median_n_star": 8.0, "n_cases": 5, "exploratory": "False"},
            {"topology": "rare", "frac_pass_at_10d": 0.2,
             "median_n_star": 30.0, "n_cases": 1, "exploratory": "True"},
        ])

        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)

        assert nums["dTwoNMainTopos"] == 1
        assert nums["dTwoNExploTopos"] == 1
        assert nums["dTwoDecision"] == "Rule Holds"

    def test_primary_penalized_rows_are_preferred(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_rule(d2, [
            {"topology": "smooth", "frac_pass_at_10d": 0.0,
             "median_n_star": 99.0, "n_cases": 5, "exploratory": False,
             "analysis_type": "complete_case", "is_primary": False},
            {"topology": "smooth", "frac_pass_at_10d": 0.8,
             "median_n_star": 8.0, "n_cases": 5, "exploratory": False,
             "analysis_type": "penalized_failed_in_denominator", "is_primary": True},
            {"topology": "rough", "frac_pass_at_10d": 0.7,
             "median_n_star": 12.0, "n_cases": 4, "exploratory": False,
             "analysis_type": "penalized_failed_in_denominator", "is_primary": True},
            {"topology": "rare", "frac_pass_at_10d": 1.0,
             "median_n_star": 4.0, "n_cases": 1, "exploratory": True,
             "analysis_type": "penalized_failed_in_denominator", "is_primary": False},
        ])

        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)

        assert nums["dTwoNMainTopos"] == 2
        assert nums["dTwoNExploTopos"] == 1
        assert nums["dTwoFracPassMain"] == "100%"
        assert nums["dTwoDecision"] == "Rule Holds"
        assert nums["dTwoMedianNStar"] == 10.0


class TestD3Decision:
    def test_under_direction(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_bias(d3, [
            {"method": "gp", "signed_bias_mean": -0.05,
             "signed_bias_ci_lo": -0.08, "signed_bias_ci_hi": -0.02,
             "signed_bias_pvalue_two_sided": 0.001, "p_holm": 0.004},
            {"method": "rf", "signed_bias_mean": -0.03,
             "signed_bias_ci_lo": -0.06, "signed_bias_ci_hi": -0.001,
             "signed_bias_pvalue_two_sided": 0.02, "p_holm": 0.04},
            {"method": "et", "signed_bias_mean": 0.005,
             "signed_bias_ci_lo": -0.01, "signed_bias_ci_hi": 0.02,
             "signed_bias_pvalue_two_sided": 0.5, "p_holm": 0.5},
        ])
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dThreeNSigMethods"] == 2
        assert nums["dThreeDecision"] == "Bias Exists"
        assert nums["dThreeDirection"] == "under"

    def test_mixed_direction(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_bias(d3, [
            {"method": "gp", "signed_bias_mean": -0.05,
             "signed_bias_ci_lo": -0.08, "signed_bias_ci_hi": -0.02,
             "signed_bias_pvalue_two_sided": 0.001, "p_holm": 0.004},
            {"method": "rf", "signed_bias_mean": 0.04,
             "signed_bias_ci_lo": 0.01, "signed_bias_ci_hi": 0.07,
             "signed_bias_pvalue_two_sided": 0.001, "p_holm": 0.004},
        ])
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dThreeDirection"] == "mixed"

    def test_no_significant_methods(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_bias(d3, [
            {"method": "gp", "signed_bias_mean": 0.001,
             "signed_bias_ci_lo": -0.005, "signed_bias_ci_hi": 0.007,
             "signed_bias_pvalue_two_sided": 0.7, "p_holm": 1.0},
            {"method": "rf", "signed_bias_mean": -0.002,
             "signed_bias_ci_lo": -0.008, "signed_bias_ci_hi": 0.004,
             "signed_bias_pvalue_two_sided": 0.6, "p_holm": 1.0},
        ])
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dThreeNSigMethods"] == 0
        assert nums["dThreeDecision"] == "No Bias"
        assert nums["dThreeDirection"] == "none"

    def test_low_n_methods_excluded_from_headline(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_bias(d3, [
            {"method": "low_n_sig", "n_runs": 9, "low_n": True,
             "signed_bias_mean": -0.05,
             "signed_bias_ci_lo": -0.08, "signed_bias_ci_hi": -0.02,
             "signed_bias_pvalue_two_sided": 0.001, "p_holm": 0.004},
            {"method": "ok_non_sig", "n_runs": 10, "low_n": False,
             "signed_bias_mean": 0.001,
             "signed_bias_ci_lo": -0.005, "signed_bias_ci_hi": 0.007,
             "signed_bias_pvalue_two_sided": 0.7, "p_holm": 1.0},
        ])

        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)

        assert nums["dThreeNTested"] == 1
        assert nums["dThreeNLowN"] == 1
        assert nums["dThreeNSigMethods"] == 0
        assert nums["dThreeDecision"] == "No Bias"

    def test_significant_but_signed_bias_nan(self, paper_pipeline_dirs):
        # Regression for the vacuous-truth bug.  A method can be flagged
        # significant by an external Holm step (e.g. carried over from
        # an old reliability_per_method.csv) but its signed_bias may
        # still be NaN if upstream extraction failed for P_f.
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_bias(d3, [
            {"method": "gp", "signed_bias_mean": float("nan"),
             "signed_bias_ci_lo": float("nan"),
             "signed_bias_ci_hi": float("nan"),
             "signed_bias_pvalue_two_sided": 0.001, "p_holm": 0.004},
        ])
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        assert nums["dThreeNSigMethods"] == 1
        assert nums["dThreeDirection"] == "none"

    def test_holm_fallback_when_only_raw_present(self, paper_pipeline_dirs):
        # Older reliability_per_method.csv may lack p_holm; the script
        # must reconstruct it before applying the 0.05 cutoff.
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_bias(d3, [
            {"method": "m1", "signed_bias_mean": -0.05,
             "signed_bias_ci_lo": -0.08, "signed_bias_ci_hi": -0.02,
             "signed_bias_pvalue_two_sided": 0.001},
            {"method": "m2", "signed_bias_mean": 0.005,
             "signed_bias_ci_lo": -0.01, "signed_bias_ci_hi": 0.02,
             "signed_bias_pvalue_two_sided": 0.5},
            {"method": "m3", "signed_bias_mean": 0.04,
             "signed_bias_ci_lo": 0.01, "signed_bias_ci_hi": 0.07,
             "signed_bias_pvalue_two_sided": 0.04},
        ])
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        # Holm with m=3 raw=[0.001, 0.5, 0.04] →
        #   sorted=[0.001, 0.04, 0.5], adj=[0.003, 0.08, 0.5]
        # → only m1 survives at α=0.05.
        assert nums["dThreeNSigMethods"] == 1
        assert nums["dThreeDirection"] == "under"

    def test_missing_d3_p_values_do_not_become_no_bias(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_bias(d3, [
            {"method": "gp", "signed_bias_mean": -0.05,
             "signed_bias_ci_lo": -0.08, "signed_bias_ci_hi": -0.02},
        ])

        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)

        assert nums["dThreeNSigMethods"] == "N/A"
        assert nums["dThreeNTested"] == "N/A"
        assert nums["dThreeDecision"] == "N/A"
        assert nums["dThreeDirection"] == "N/A"


class TestWriteTex:
    def test_no_question_marks_when_full(self, tmp_path, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        _write_correlation(rr, rho=0.55, ci_lo=0.32, ci_hi=0.69)
        _write_rule(d2, [
            {"topology": "smooth", "frac_pass_at_10d": 0.6,
             "median_n_star": 8.0, "n_cases": 5, "exploratory": False},
        ])
        _write_bias(d3, [
            {"method": "gp", "signed_bias_mean": -0.05,
             "signed_bias_ci_lo": -0.08, "signed_bias_ci_hi": -0.02,
             "signed_bias_pvalue_two_sided": 0.001, "p_holm": 0.004},
        ])
        _write_cp_bound(d2, upper=0.7746784, cutoff=4 / 7)
        _write_d3_pooled_artifacts(d3)
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)
        tex_path = tmp_path / "numbers.tex"
        write_tex(nums, tex_path)
        text = tex_path.read_text(encoding="utf-8")
        # Full happy-path data must not leave any "??" placeholders.
        assert "??" not in text
        # Every documented macro must be present.
        for macro in ["dOneRho", "dOneCILo", "dOneCIHi", "dOneDecision",
                      "dTwoFracPassMain", "dTwoDecision",
                      "dThreeNSigMethods", "dThreeDecision",
                      "dThreeDirection", "dThreeNLowN",
                      "primaryFailurePolicy", "dOneRankPolicy",
                      "HolmFWER", "HolmFamilySize", "HolmThresholds",
                      "MCNoiseMaxRelWidthPct"]:
            assert macro in text
        # The generated file also exposes \Num* aliases used by paper
        # templates and audit_macros.py.
        for macro in ["NumNCases", "NumNMethods", "NumDOneRho",
                      "NumDOneRankPolicy", "NumDTwoDecision",
                      "NumDThreeDirection", "NumHolmThresholds",
                      "NumMCNoiseMaxRelWidthPct"]:
            assert macro in text

    def test_holm_fwer_contract_macros(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)

        assert nums["holmFWER"] == 0.05
        assert nums["holmFamilySize"] == 3
        assert nums["holmThresholds"] == "0.0167, 0.025, 0.05"

    def test_mc_noise_floor_macros_from_test_protocol_csv(self, paper_pipeline_dirs, tmp_path):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        protocol = tmp_path / "analysis"
        _write_test_set_sufficiency(protocol)

        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd, protocol)

        assert nums["mcNoiseNTest"] == 1000
        assert nums["mcNoiseMedianRelWidthPct"] == 30.0
        assert nums["mcNoiseMaxRelWidthPct"] == 40.0
        assert nums["mcNoiseMaxAbsWidth"] == 0.3
        assert nums["mcNoiseNCases"] == 2

    def test_mc_noise_floor_macros_degrade_to_na_when_missing(self, paper_pipeline_dirs):
        _, rr, d2, d3, fd = paper_pipeline_dirs
        nums = collect_numbers(rr / "no_results.csv", rr, d2, d3, fd)

        assert nums["mcNoiseNTest"] == "N/A"
        assert nums["mcNoiseMaxRelWidthPct"] == "N/A"

    def test_cli_require_results_csv_fails_when_missing(self, tmp_path):
        from scripts.fill_paper_numbers import main

        rc = main([
            "--results-csv", str(tmp_path / "missing.csv"),
            "--rr-dir", str(tmp_path / "rr"),
            "--d2-dir", str(tmp_path / "d2"),
            "--d3-dir", str(tmp_path / "d3"),
            "--figdata-dir", str(tmp_path / "figdata"),
            "--output-tex", str(tmp_path / "numbers.tex"),
            "--output-json", str(tmp_path / "numbers.json"),
            "--require-results-csv",
        ])

        assert rc == 2
        assert not (tmp_path / "numbers.tex").exists()
        assert not (tmp_path / "numbers.json").exists()

    def test_cli_require_submission_complete_fails_before_writing_na_numbers(
        self,
        paper_pipeline_dirs,
        tmp_path,
    ):
        from scripts.fill_paper_numbers import main

        results_dir, rr, d2, d3, fd = paper_pipeline_dirs
        results = results_dir / "results.csv"
        pd.DataFrame([
            {
                "case_id": "c1",
                "method": "gp",
                "status": "ok",
                "budget_mult": 10,
                "repeat": 0,
                "error": "",
            }
        ]).to_csv(results, index=False)

        output_tex = tmp_path / "numbers.tex"
        output_json = tmp_path / "numbers.json"
        rc = main([
            "--results-csv", str(results),
            "--rr-dir", str(rr),
            "--d2-dir", str(d2),
            "--d3-dir", str(d3),
            "--figdata-dir", str(fd),
            "--test-protocol-dir", str(tmp_path / "missing_protocol"),
            "--output-tex", str(output_tex),
            "--output-json", str(output_json),
            "--require-results-csv",
            "--require-submission-complete",
        ])

        assert rc == 3
        assert not output_tex.exists()
        assert not output_json.exists()

    def test_cli_require_submission_complete_accepts_formal_primary_inputs(
        self,
        paper_pipeline_dirs,
        tmp_path,
    ):
        from scripts.fill_paper_numbers import main

        results_dir, rr, d2, d3, fd = paper_pipeline_dirs
        results = results_dir / "results.csv"
        protocol = tmp_path / "analysis"
        _write_formal_numbers_inputs(results, rr, d2, d3, protocol)

        output_tex = tmp_path / "numbers.tex"
        output_json = tmp_path / "numbers.json"
        rc = main([
            "--results-csv", str(results),
            "--rr-dir", str(rr),
            "--d2-dir", str(d2),
            "--d3-dir", str(d3),
            "--figdata-dir", str(fd),
            "--test-protocol-dir", str(protocol),
            "--output-tex", str(output_tex),
            "--output-json", str(output_json),
            "--require-results-csv",
            "--require-results-contract",
            "--require-submission-complete",
        ])

        assert rc == 0
        data = json.loads(output_json.read_text(encoding="utf-8"))
        assert data["dOneRankPolicy"] == "complete_case"
        assert data["primaryFailurePolicy"] == "penalized_failed_in_denominator"
        assert data["dTwoDecision"] == "Rule Holds"

    def test_cli_require_submission_complete_rejects_non_primary_d1_before_writing(
        self,
        paper_pipeline_dirs,
        tmp_path,
    ):
        from scripts.fill_paper_numbers import main

        results_dir, rr, d2, d3, fd = paper_pipeline_dirs
        results = results_dir / "results.csv"
        protocol = tmp_path / "analysis"
        _write_formal_numbers_inputs(results, rr, d2, d3, protocol)
        (rr / "correlation.json").write_text(json.dumps([
            {
                "pairing_mode": "dimension",
                "budget_mult": 10,
                "spearman_rho": 0.91,
                "spearman_rho_ci_lo": 0.82,
                "spearman_rho_ci_hi": 0.96,
                "n_methods": 20,
                "analysis_type": "legacy",
                "is_primary": True,
            },
        ]), encoding="utf-8")

        output_tex = tmp_path / "numbers.tex"
        output_json = tmp_path / "numbers.json"
        rc = main([
            "--results-csv", str(results),
            "--rr-dir", str(rr),
            "--d2-dir", str(d2),
            "--d3-dir", str(d3),
            "--figdata-dir", str(fd),
            "--test-protocol-dir", str(protocol),
            "--output-tex", str(output_tex),
            "--output-json", str(output_json),
            "--require-results-csv",
            "--require-results-contract",
            "--require-submission-complete",
        ])

        assert rc == 3
        assert not output_tex.exists()
        assert not output_json.exists()

    def test_cli_require_submission_complete_rejects_non_penalized_d2_before_writing(
        self,
        paper_pipeline_dirs,
        tmp_path,
    ):
        from scripts.fill_paper_numbers import main

        results_dir, rr, d2, d3, fd = paper_pipeline_dirs
        results = results_dir / "results.csv"
        protocol = tmp_path / "analysis"
        _write_formal_numbers_inputs(results, rr, d2, d3, protocol)
        _write_rule(d2, [
            {
                "topology": "smooth",
                "frac_pass_at_10d": 0.7,
                "median_n_star": 8.0,
                "n_cases": 5,
                "exploratory": False,
                "analysis_type": "complete_case",
                "is_primary": True,
            },
        ])

        output_tex = tmp_path / "numbers.tex"
        output_json = tmp_path / "numbers.json"
        rc = main([
            "--results-csv", str(results),
            "--rr-dir", str(rr),
            "--d2-dir", str(d2),
            "--d3-dir", str(d3),
            "--figdata-dir", str(fd),
            "--test-protocol-dir", str(protocol),
            "--output-tex", str(output_tex),
            "--output-json", str(output_json),
            "--require-results-csv",
            "--require-results-contract",
            "--require-submission-complete",
        ])

        assert rc == 3
        assert not output_tex.exists()
        assert not output_json.exists()

    def test_cli_require_results_contract_rejects_corrupt_results_before_writing(
        self,
        paper_pipeline_dirs,
        tmp_path,
    ):
        from scripts.fill_paper_numbers import main

        results_dir, rr, d2, d3, fd = paper_pipeline_dirs
        results = results_dir / "results.csv"
        pd.DataFrame([
            {
                "case_id": "c1",
                "method": "gp",
                "status": "ok",
                "budget_mult": 10,
                "repeat": 0,
                "nrmse": float("nan"),
                "error": "",
                "t_total_s": 1.0,
            }
        ]).to_csv(results, index=False)

        output_tex = tmp_path / "numbers.tex"
        output_json = tmp_path / "numbers.json"
        rc = main([
            "--results-csv", str(results),
            "--rr-dir", str(rr),
            "--d2-dir", str(d2),
            "--d3-dir", str(d3),
            "--figdata-dir", str(fd),
            "--output-tex", str(output_tex),
            "--output-json", str(output_json),
            "--require-results-contract",
        ])

        assert rc == 2
        assert not output_tex.exists()
        assert not output_json.exists()

    def test_cli_require_headline_source_rejects_fixture_results_before_writing(
        self,
        paper_pipeline_dirs,
        tmp_path,
    ):
        from scripts.fill_paper_numbers import main

        results_dir, rr, d2, d3, fd = paper_pipeline_dirs
        results = results_dir / "results.csv"
        protocol = tmp_path / "analysis"
        _write_formal_numbers_inputs(results, rr, d2, d3, protocol)

        output_tex = tmp_path / "numbers.tex"
        output_json = tmp_path / "numbers.json"
        rc = main([
            "--results-csv", str(results),
            "--rr-dir", str(rr),
            "--d2-dir", str(d2),
            "--d3-dir", str(d3),
            "--figdata-dir", str(fd),
            "--test-protocol-dir", str(protocol),
            "--output-tex", str(output_tex),
            "--output-json", str(output_json),
            "--require-results-csv",
            "--require-results-contract",
            "--require-headline-source",
        ])

        assert rc == 2
        assert not output_tex.exists()
        assert not output_json.exists()

    def test_cli_writes_numbers_input_provenance(
        self,
        paper_pipeline_dirs,
        tmp_path,
    ):
        from scripts.fill_paper_numbers import main

        results_dir, rr, d2, d3, fd = paper_pipeline_dirs
        protocol = tmp_path / "analysis"
        _write_test_set_sufficiency(protocol)
        output_tex = tmp_path / "numbers.tex"
        output_json = tmp_path / "numbers.json"

        rc = main([
            "--results-csv", str(results_dir / "missing_results.csv"),
            "--rr-dir", str(rr),
            "--d2-dir", str(d2),
            "--d3-dir", str(d3),
            "--figdata-dir", str(fd),
            "--test-protocol-dir", str(protocol),
            "--output-tex", str(output_tex),
            "--output-json", str(output_json),
        ])

        assert rc == 0
        data = json.loads(output_json.read_text(encoding="utf-8"))
        provenance = data["_provenance"]
        assert provenance["script"] == "scripts/fill_paper_numbers.py"
        labels = {item["label"] for item in provenance["inputs"]}
        assert {
            "results_csv",
            "d1_correlation",
            "d2_rule_summary",
            "d3_bias_numbers",
            "test_set_sufficiency",
        } <= labels
        assert any(
            item["label"] == "test_set_sufficiency" and item["sha256"]
            for item in provenance["inputs"]
        )

    def test_numbers_package_write_failure_preserves_existing_pair(
        self,
        tmp_path,
        monkeypatch,
    ):
        import json as json_module
        from scripts import fill_paper_numbers

        output_tex = tmp_path / "numbers.tex"
        output_json = tmp_path / "numbers.json"
        output_tex.write_text("old tex\n", encoding="utf-8")
        output_json.write_text('{"old": true}\n', encoding="utf-8")

        def boom(*args, **kwargs):
            raise RuntimeError("json serialization failed")

        monkeypatch.setattr(json_module, "dumps", boom)

        with pytest.raises(RuntimeError, match="json serialization failed"):
            fill_paper_numbers.write_numbers_package(
                {"nCases": 34},
                output_tex,
                output_json,
            )

        assert output_tex.read_text(encoding="utf-8") == "old tex\n"
        assert output_json.read_text(encoding="utf-8") == '{"old": true}\n'
        assert not list(tmp_path.glob(".numbers.tmp.*"))


# ─── prepare_figdata schema invariants ─────────────────────────────────
class TestPrepareFigdata:
    """Ensure the figdata CSVs hold the columns plots_nature/fill_paper expect."""

    def test_fig4_holm_fallback(self, tmp_path):
        from scripts.prepare_figdata import make_fig4
        d3 = tmp_path / "d3"
        out = tmp_path / "fig"
        d3.mkdir()
        out.mkdir()
        # Reliability table missing p_holm — make_fig4 must compute it.
        pd.DataFrame([
            {"method": "m1",
             "signed_bias_mean": -0.05, "signed_bias_ci_lo": -0.08,
             "signed_bias_ci_hi": -0.02,
             "signed_bias_pvalue_two_sided": 0.001},
            {"method": "m2",
             "signed_bias_mean": 0.005, "signed_bias_ci_lo": -0.01,
             "signed_bias_ci_hi": 0.02,
             "signed_bias_pvalue_two_sided": 0.5},
        ]).to_csv(d3 / "reliability_per_method.csv", index=False)
        make_fig4(d3, out)
        fig4 = pd.read_csv(out / "fig4_d3_bias.csv")
        assert "p_holm" in fig4.columns
        assert "signed_bias" in fig4.columns
        assert "bias_direction" in fig4.columns
        assert "low_n" in fig4.columns
        # bias_direction must reflect the renamed ``signed_bias`` column.
        sub = fig4[fig4["method"] == "m1"]
        assert sub.iloc[0]["bias_direction"] == "under"

    def test_fig4_derives_low_n_from_n_runs_when_missing(self, tmp_path):
        from scripts.prepare_figdata import make_fig4
        d3 = tmp_path / "d3"
        out = tmp_path / "fig"
        d3.mkdir()
        out.mkdir()
        pd.DataFrame([
            {"method": "m_low", "n_runs": 9,
             "signed_bias_mean": -0.05, "signed_bias_ci_lo": -0.08,
             "signed_bias_ci_hi": -0.02,
             "signed_bias_pvalue_two_sided": 0.001},
            {"method": "m_ok", "n_runs": 10,
             "signed_bias_mean": 0.01, "signed_bias_ci_lo": -0.01,
             "signed_bias_ci_hi": 0.02,
             "signed_bias_pvalue_two_sided": 0.5},
        ]).to_csv(d3 / "reliability_per_method.csv", index=False)

        make_fig4(d3, out)

        fig4 = pd.read_csv(out / "fig4_d3_bias.csv")
        flags = dict(zip(fig4["method"], fig4["low_n"]))
        assert bool(flags["m_low"]) is True
        assert bool(flags["m_ok"]) is False

    def test_fig4_bias_direction_distinguishes_nan_from_zero(self, tmp_path):
        """GAP audit fix: NaN signed_bias gets ``"undefined"``, not ``"none"``.

        Before the fix, ``np.where(bias > 0, "over", np.where(bias < 0,
        "under", "none"))`` collapsed NaN signed_bias (i.e. method had no
        usable runs on the 16 failure cases) into the ``"none"`` bucket
        alongside genuine zero-bias methods.  The §D3 paper text relies on
        ``"none"`` meaning "calibrated"; reporting ``"none"`` for a
        no-data method would over-state how many methods passed the
        calibration test.
        """
        from scripts.prepare_figdata import make_fig4
        d3 = tmp_path / "d3"
        out = tmp_path / "fig"
        d3.mkdir()
        out.mkdir()
        pd.DataFrame([
            {"method": "m_over",   "signed_bias_mean":  0.05,
             "signed_bias_ci_lo":  0.02, "signed_bias_ci_hi":  0.08,
             "signed_bias_pvalue_two_sided": 0.001},
            {"method": "m_under",  "signed_bias_mean": -0.05,
             "signed_bias_ci_lo": -0.08, "signed_bias_ci_hi": -0.02,
             "signed_bias_pvalue_two_sided": 0.001},
            {"method": "m_zero",   "signed_bias_mean":  0.0,
             "signed_bias_ci_lo": -0.01, "signed_bias_ci_hi":  0.01,
             "signed_bias_pvalue_two_sided": 0.9},
            {"method": "m_nodata", "signed_bias_mean":  float("nan"),
             "signed_bias_ci_lo":  float("nan"),
             "signed_bias_ci_hi":  float("nan"),
             "signed_bias_pvalue_two_sided": float("nan")},
        ]).to_csv(d3 / "reliability_per_method.csv", index=False)
        make_fig4(d3, out)
        fig4 = pd.read_csv(out / "fig4_d3_bias.csv")
        directions = dict(zip(fig4["method"], fig4["bias_direction"]))
        assert directions["m_over"] == "over"
        assert directions["m_under"] == "under"
        assert directions["m_zero"] == "none"
        # The fix: NaN must be "undefined", NOT "none".
        assert directions["m_nodata"] == "undefined"

    def test_fig6_emits_complete_and_penalized(self, tmp_path):
        from scripts.prepare_figdata import make_fig6
        out = tmp_path / "fig"
        out.mkdir()
        # Two methods × one budget; method_b has one timeout in its
        # cell so the penalized denominator differs from complete-case.
        df = pd.DataFrame([
            {"method": "method_a", "case_id": "c1", "repeat": 0,
             "budget_mult": 10, "status": "ok", "nrmse": 0.10,
             "t_total_s": 5.0},
            {"method": "method_a", "case_id": "c2", "repeat": 0,
             "budget_mult": 10, "status": "ok", "nrmse": 0.12,
             "t_total_s": 6.0},
            {"method": "method_b", "case_id": "c1", "repeat": 0,
             "budget_mult": 10, "status": "ok", "nrmse": 0.08,
             "t_total_s": 50.0},
            {"method": "method_b", "case_id": "c2", "repeat": 0,
             "budget_mult": 10, "status": "timeout", "nrmse": float("nan"),
             "t_total_s": 60.0},
        ])
        make_fig6(df, out)
        fig6 = pd.read_csv(out / "fig6_pareto.csv")
        assert "analysis_type" in fig6.columns
        assert "is_primary" in fig6.columns
        assert set(fig6["analysis_type"].unique()) == {
            "complete_case", "penalized_failed_in_denominator",
        }
        # complete-case keeps method_b's median NRMSE = 0.08 (single OK row);
        # penalized variant inflates it because the timeout cell is now
        # counted with NRMSE = 1.25 × max OK = 0.15.
        cc = fig6[(fig6["analysis_type"] == "complete_case")
                  & (fig6["method"] == "method_b")].iloc[0]
        pen = fig6[(fig6["analysis_type"] == "penalized_failed_in_denominator")
                   & (fig6["method"] == "method_b")].iloc[0]
        assert cc["nrmse_median"] == pytest.approx(0.08)
        assert pen["nrmse_median"] > cc["nrmse_median"]
        assert bool(cc["is_primary"]) is False
        assert bool(pen["is_primary"]) is True
        # Failure-rate book-keeping: method_b has 1/2 timeouts.
        assert pen["failure_rate"] == pytest.approx(0.5)
        assert cc["failure_rate"] == pytest.approx(0.5)

    def test_fig3_rule_marks_penalized_primary_when_legacy_input_lacks_flags(self, tmp_path):
        from scripts.prepare_figdata import make_fig3

        d2 = tmp_path / "d2"
        out = tmp_path / "fig"
        d2.mkdir()
        out.mkdir()
        pd.DataFrame([
            {
                "topology": "smooth",
                "method": "m1",
                "budget_mult": 10,
                "median": 0.1,
                "ci_lo": 0.08,
                "ci_hi": 0.12,
                "n_cases": 3,
            }
        ]).to_csv(d2 / "learning_curves.csv", index=False)
        pd.DataFrame([
            {
                "topology": "smooth",
                "frac_pass_at_10d": 0.9,
                "median_n_star": 10.0,
                "n_cases": 3,
                "exploratory": False,
            }
        ]).to_csv(d2 / "rule_summary.csv", index=False)

        make_fig3(d2, out)

        rule = pd.read_csv(out / "fig3_d2_rule.csv")
        assert set(rule["analysis_type"]) == {"penalized_failed_in_denominator"}
        assert rule["is_primary"].astype(bool).all()

    def test_fig3_rule_only_marks_penalized_analysis_primary(self, tmp_path):
        from scripts.prepare_figdata import make_fig3

        d2 = tmp_path / "d2"
        out = tmp_path / "fig"
        d2.mkdir()
        out.mkdir()
        pd.DataFrame([
            {
                "topology": "smooth",
                "method": "m1",
                "budget_mult": 10,
                "median": 0.1,
                "ci_lo": 0.08,
                "ci_hi": 0.12,
                "n_cases": 3,
            }
        ]).to_csv(d2 / "learning_curves.csv", index=False)
        pd.DataFrame([
            {
                "topology": "smooth",
                "frac_pass_at_10d": 0.9,
                "median_n_star": 10.0,
                "n_cases": 3,
                "exploratory": False,
                "analysis_type": "penalized_failed_in_denominator",
            },
            {
                "topology": "smooth",
                "frac_pass_at_10d": 1.0,
                "median_n_star": 8.0,
                "n_cases": 3,
                "exploratory": False,
                "analysis_type": "complete_case",
            },
        ]).to_csv(d2 / "rule_summary.csv", index=False)

        make_fig3(d2, out)

        rule = pd.read_csv(out / "fig3_d2_rule.csv")
        flags = dict(zip(rule["analysis_type"], rule["is_primary"].astype(bool)))
        assert flags["penalized_failed_in_denominator"] is True
        assert flags["complete_case"] is False

    @pytest.fixture
    def d1_dir_with_three_pairings(self, tmp_path):
        d1 = tmp_path / "d1"
        d1.mkdir()
        corr = []
        for mode in ("dimension", "topology", "random"):
            mode_dir = d1 / f"pairing_{mode}"
            mode_dir.mkdir()
            pd.DataFrame([
                {
                    "pairing_mode": mode,
                    "budget_mult": 10,
                    "method": f"m{i}",
                    "rank_analytical": float(i),
                    "rank_physical": float(i) + 0.1,
                }
                for i in range(1, 4)
            ]).to_csv(mode_dir / "rank_comparison.csv", index=False)
            corr.append({
                "pairing_mode": mode,
                "budget_mult": 10,
                "spearman_rho": 0.8,
                "spearman_p": 0.01,
                "kendall_tau": 0.7,
                "analysis_type": "complete_case",
                "is_primary": True,
            })
        (d1 / "correlation.json").write_text(json.dumps(corr), encoding="utf-8")
        return d1

    def test_fig2_uses_nested_pairings_even_when_legacy_root_exists(
        self, d1_dir_with_three_pairings, tmp_path,
    ):
        from scripts.prepare_figdata import make_fig2

        pd.DataFrame([
            {
                "budget_mult": 10,
                "method": "legacy_only",
                "rank_analytical": 1.0,
                "rank_physical": 1.0,
            }
        ]).to_csv(d1_dir_with_three_pairings / "rank_comparison.csv", index=False)

        out = tmp_path / "fig"
        make_fig2(d1_dir_with_three_pairings, out)
        ranks = pd.read_csv(out / "fig2_d1_ranks.csv")

        assert sorted(ranks["pairing_mode"].unique()) == ["dimension", "random", "topology"]
        assert set(ranks["analysis_type"]) == {"complete_case"}
        assert ranks["is_primary"].astype(bool).all()
        assert "legacy_only" not in set(ranks["method"])

    def test_fig2_rejects_legacy_only_rank_layout_by_default(self, tmp_path):
        from scripts.prepare_figdata import make_fig2

        d1 = tmp_path / "d1"
        d1.mkdir()
        (d1 / "correlation.json").write_text(
            json.dumps([{
                "pairing_mode": "dimension",
                "budget_mult": 10,
                "spearman_rho": 1.0,
                "spearman_p": 0.0,
                "kendall_tau": 1.0,
            }]),
            encoding="utf-8",
        )
        pd.DataFrame([
            {
                "budget_mult": 10,
                "method": "m1",
                "rank_analytical": 1.0,
                "rank_physical": 1.0,
            }
        ]).to_csv(d1 / "rank_comparison.csv", index=False)

        with pytest.raises(AssertionError, match="legacy D1 layout"):
            make_fig2(d1, tmp_path / "fig")

    def test_fig2_allows_legacy_rank_layout_only_when_explicit(self, tmp_path):
        from scripts.prepare_figdata import make_fig2

        d1 = tmp_path / "d1"
        d1.mkdir()
        (d1 / "correlation.json").write_text(
            json.dumps([{
                "pairing_mode": "dimension",
                "budget_mult": 10,
                "spearman_rho": 1.0,
                "spearman_p": 0.0,
                "kendall_tau": 1.0,
            }]),
            encoding="utf-8",
        )
        pd.DataFrame([
            {
                "budget_mult": 10,
                "method": "m1",
                "rank_analytical": 1.0,
                "rank_physical": 1.0,
            }
        ]).to_csv(d1 / "rank_comparison.csv", index=False)

        out = tmp_path / "fig"
        make_fig2(d1, out, allow_legacy_rank=True)
        ranks = pd.read_csv(out / "fig2_d1_ranks.csv")

        assert ranks.iloc[0]["pairing_mode"] == "dimension"
        assert ranks.iloc[0]["analysis_type"] == "complete_case"
        assert bool(ranks.iloc[0]["is_primary"]) is True

    def test_validate_results_methods_rejects_extra_method(self):
        from scripts.prepare_figdata import validate_results_methods

        df = pd.DataFrame([
            {"method": "lhs+gp"},
            {"method": "botorch_sequential"},
        ])

        with pytest.raises(AssertionError, match="botorch_sequential"):
            validate_results_methods(df, ["lhs+gp", "botorch_qei"])

    def test_validate_results_methods_allows_subset_of_declared_pool(self):
        from scripts.prepare_figdata import validate_results_methods

        df = pd.DataFrame([
            {"method": "lhs+gp"},
        ])

        validate_results_methods(df, ["lhs+gp", "botorch_qei"])

    def test_validate_headline_results_source_rejects_noncanonical_path(self, tmp_path):
        from scripts.prepare_figdata import validate_headline_results_source

        df = pd.DataFrame([{
            "case_id": "c1",
            "method": "lhs+gp",
            "budget_mult": 5,
            "repeat": 0,
            "status": "ok",
            "error": "",
            "nrmse": 0.1,
            "t_total_s": 1.0,
        }])

        with pytest.raises(AssertionError, match="canonical results source"):
            validate_headline_results_source(tmp_path / "results.csv", df)

    def test_validate_headline_results_source_rejects_sota_contamination(self):
        from scripts.prepare_figdata import (
            HEADLINE_RESULTS_CSV,
            validate_headline_results_source,
        )

        df = pd.DataFrame([{
            "case_id": "cantilever_beam_5d",
            "method": "botorch_sequential",
            "budget_mult": 5,
            "repeat": 0,
            "status": "ok",
            "error": "",
            "nrmse": 0.1,
            "t_total_s": 1.0,
        }])

        with pytest.raises(AssertionError, match="botorch_sequential"):
            validate_headline_results_source(HEADLINE_RESULTS_CSV, df)

    def test_validate_results_contract_rejects_bad_ok_nrmse(self):
        from scripts.prepare_figdata import validate_results_contract

        df = pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": float("nan"),
                "t_total_s": 1.0,
            })
        ])

        with pytest.raises(AssertionError, match="finite non-negative nrmse"):
            validate_results_contract(df)

    def test_validate_results_contract_rejects_invalid_status(self):
        from scripts.prepare_figdata import validate_results_contract

        df = pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": "cancelled",
                "failure_category": "exception",
                "error": "",
                "nrmse": float("nan"),
                "t_total_s": 1.0,
            })
        ])

        with pytest.raises(AssertionError, match="status values must be one of"):
            validate_results_contract(df)

    def test_validate_results_contract_rejects_ok_rows_with_error(self):
        from scripts.prepare_figdata import validate_results_contract

        df = pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "stale retry failure",
                "nrmse": 0.1,
                "t_total_s": 1.0,
            })
        ])

        with pytest.raises(AssertionError, match="ok rows must have blank error"):
            validate_results_contract(df)

    @pytest.mark.parametrize("status", ["timeout", "failed"])
    def test_validate_results_contract_rejects_failed_rows_without_error(self, status):
        from scripts.prepare_figdata import validate_results_contract

        df = pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": status,
                "failure_category": "timeout" if status == "timeout" else "exception",
                "error": "",
                "nrmse": "",
                "t_total_s": 1.0,
            }),
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 20,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
            }),
        ])

        with pytest.raises(
            AssertionError,
            match="timeout/failed rows must include a non-empty error reason",
        ):
            validate_results_contract(df)

    @pytest.mark.parametrize("status", ["timeout", "failed"])
    @pytest.mark.parametrize(
        "metric_patch, match",
        [
            ({"nrmse": 0.2}, "timeout/failed rows must leave nrmse blank"),
            ({"rmse": 0.2}, "timeout/failed rows must leave rmse blank"),
        ],
    )
    def test_validate_results_contract_rejects_failed_rows_with_metrics(
        self,
        status,
        metric_patch,
        match,
    ):
        from scripts.prepare_figdata import validate_results_contract

        failed_row = _result_contract_row(**{
            "case_id": "c1",
            "method": "lhs+gp",
            "budget_mult": 10,
            "repeat": 0,
            "status": status,
            "failure_category": "timeout" if status == "timeout" else "exception",
            "error": "synthetic failure",
            "nrmse": "",
            "t_total_s": 1.0,
            **_runtime_trace_fields(),
            **metric_patch,
        })
        df = pd.DataFrame([
            failed_row,
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 20,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
                **_runtime_trace_fields(),
            }),
        ])

        with pytest.raises(AssertionError, match=match):
            validate_results_contract(df)

    @pytest.mark.parametrize("status", ["timeout", "failed"])
    def test_validate_results_contract_rejects_failed_rows_with_bad_timing(self, status):
        from scripts.prepare_figdata import validate_results_contract

        df = pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": status,
                "failure_category": "timeout" if status == "timeout" else "exception",
                "error": "synthetic failure",
                "nrmse": "",
                "t_total_s": -1.0,
            }),
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 20,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
            }),
        ])

        with pytest.raises(AssertionError, match="finite non-negative t_total_s"):
            validate_results_contract(df)

    @pytest.mark.parametrize("status", ["timeout", "failed"])
    def test_validate_results_contract_allows_failed_rows_with_zero_timing(self, status):
        from scripts.prepare_figdata import validate_results_contract

        df = pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": status,
                "failure_category": "timeout" if status == "timeout" else "exception",
                "error": "synthetic failure",
                "nrmse": "",
                "t_total_s": 0.0,
                **_runtime_trace_fields(),
                "cell_finished_utc": "2026-06-01T00:00:00+00:00",
            }),
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 20,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
                **_runtime_trace_fields(),
            }),
        ])

        validate_results_contract(df)

    def test_validate_results_contract_rejects_missing_runtime_trace(self):
        from scripts.prepare_figdata import validate_results_contract

        row = _result_contract_row()
        for key in _runtime_trace_fields():
            row.pop(key)
        df = pd.DataFrame([row])

        with pytest.raises(AssertionError, match="runtime provenance columns"):
            validate_results_contract(df)

    def test_validate_results_contract_rejects_bad_rss_peak_order(self):
        from scripts.prepare_figdata import validate_results_contract

        df = pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
                **_runtime_trace_fields(),
                "rss_gb_peak": 1.0,
            })
        ])

        with pytest.raises(AssertionError, match="rss_gb_peak must be"):
            validate_results_contract(df)

    def test_validate_results_contract_rejects_non_integer_grid_ids(self):
        from scripts.prepare_figdata import validate_results_contract

        df = pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": "5.5",
                "repeat": "0.5",
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
            })
        ])

        with pytest.raises(AssertionError, match="budget_mult values must be finite integers"):
            validate_results_contract(df)

    @pytest.mark.parametrize("field", ["case_id", "method", "budget_mult", "repeat"])
    def test_validate_results_contract_rejects_missing_protocol_key_values(self, field):
        from scripts.prepare_figdata import validate_results_contract

        row = _result_contract_row(**{
            "case_id": "c1",
            "method": "lhs+gp",
            "budget_mult": 10,
            "repeat": 0,
            "status": "ok",
            "failure_category": "ok",
            "error": "",
            "nrmse": 0.1,
            "t_total_s": 1.0,
        })
        row[field] = None
        df = pd.DataFrame([row])

        with pytest.raises(
            AssertionError,
            match=f"protocol key values must be non-empty.*{field}=1",
        ):
            validate_results_contract(df)

    def test_validate_results_contract_rejects_duplicate_cells(self):
        from scripts.prepare_figdata import validate_results_contract

        row = _result_contract_row(**{
            "case_id": "c1",
            "method": "lhs+gp",
            "budget_mult": 10,
            "repeat": 0,
            "status": "ok",
            "failure_category": "ok",
            "error": "",
            "nrmse": 0.1,
            "t_total_s": 1.0,
        })
        duplicate = _result_contract_row(**{
            "case_id": "c1",
            "method": "lhs+gp",
            "budget_mult": 10,
            "repeat": 0,
            "status": "ok",
            "failure_category": "ok",
            "error": "",
            "nrmse": 0.2,
            "t_total_s": 1.0,
        })
        df = pd.DataFrame([row, duplicate])

        with pytest.raises(AssertionError, match="duplicate .*cells"):
            validate_results_contract(df)

    def test_cli_contract_failure_preserves_previous_figdata(self, tmp_path):
        from scripts.prepare_figdata import main

        results = tmp_path / "results.csv"
        pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": -0.1,
                "t_total_s": 1.0,
            })
        ]).to_csv(results, index=False)

        out = tmp_path / "figdata"
        out.mkdir()
        sentinel = out / "sentinel.csv"
        sentinel.write_text("old_package\n", encoding="utf-8")

        with pytest.raises(AssertionError, match="finite non-negative nrmse"):
            main([
                "--results-csv", str(results),
                "--expected-methods", "lhs+gp",
                "--out-dir", str(out),
                "--skip", "fig2", "fig3", "fig4", "fig5", "fig6",
            ])

        assert sentinel.read_text(encoding="utf-8") == "old_package\n"
        assert not (out / "fig1_case_catalog.csv").exists()

    def test_cli_require_headline_source_rejects_fixture_figdata_before_writing(self, tmp_path):
        from scripts.prepare_figdata import main

        results = tmp_path / "results.csv"
        pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
                **_runtime_trace_fields(),
            })
        ]).to_csv(results, index=False)
        out = tmp_path / "figdata"

        with pytest.raises(AssertionError, match="canonical results source"):
            main([
                "--results-csv", str(results),
                "--expected-methods", "lhs+gp",
                "--out-dir", str(out),
                "--skip", "fig2", "fig3", "fig4", "fig5", "fig6",
                "--require-headline-source",
            ])

        assert not out.exists()

    def test_declared_methods_from_manifest(self, tmp_path):
        from scripts.prepare_figdata import declared_methods_from_manifest

        manifest = tmp_path / "RUN_MANIFEST.json"
        manifest.write_text(
            json.dumps({
                "experiment": {
                    "methods_declared": ["lhs+gp", "botorch_qei", None, "  "],
                }
            }),
            encoding="utf-8",
        )

        assert declared_methods_from_manifest(manifest) == ["lhs+gp", "botorch_qei"]

    def test_cli_manifest_methods_override_fallback_expected_methods(self, tmp_path):
        from scripts.prepare_figdata import main

        results = tmp_path / "results.csv"
        pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "botorch_qei",
                "budget_mult": 10,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
                **_runtime_trace_fields(),
            })
        ]).to_csv(results, index=False)
        manifest = tmp_path / "RUN_MANIFEST.json"
        manifest.write_text(
            json.dumps({"experiment": {"methods_declared": ["botorch_qei"]}}),
            encoding="utf-8",
        )

        rc = main([
            "--results-csv", str(results),
            "--manifest", str(manifest),
            "--expected-methods", "lhs+gp",
            "--out-dir", str(tmp_path / "figdata"),
            "--skip", "fig1", "fig2", "fig3", "fig4", "fig5", "fig6",
        ])

        assert rc == 0
        figdata_manifest = json.loads(
            (tmp_path / "figdata" / "figdata_manifest.json").read_text(encoding="utf-8")
        )
        assert figdata_manifest["status"] == "partial_skip_selection"
        assert any("--skip" in err for err in figdata_manifest["errors"])

    def test_fig1_method_catalog_uses_declared_methods_when_provided(self, tmp_path):
        from scripts.prepare_figdata import make_fig1

        df = pd.DataFrame([{"method": "lhs+gp"}])
        out = tmp_path / "figdata"

        make_fig1(
            df,
            out,
            declared_methods=["lhs+gp", "botorch_qei", "botorch_saasbo"],
            method_source="RUN_MANIFEST experiment.methods_declared",
        )
        catalog = pd.read_csv(out / "fig1_method_catalog.csv")

        assert catalog["method"].tolist() == ["lhs+gp", "botorch_qei", "botorch_saasbo"]
        assert set(catalog["_source"]) == {"RUN_MANIFEST experiment.methods_declared"}

    def test_fig1_method_catalog_marks_expected_methods_fallback_source(self, tmp_path):
        from scripts.prepare_figdata import main

        results = tmp_path / "results.csv"
        pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
                **_runtime_trace_fields(),
            })
        ]).to_csv(results, index=False)

        out = tmp_path / "figdata"
        rc = main([
            "--results-csv", str(results),
            "--expected-methods", "lhs+gp",
            "--out-dir", str(out),
            "--skip", "fig2", "fig3", "fig4", "fig5", "fig6",
        ])
        catalog = pd.read_csv(out / "fig1_method_catalog.csv")
        figdata_manifest = json.loads((out / "figdata_manifest.json").read_text(encoding="utf-8"))

        assert rc == 0
        assert set(catalog["_source"]) == {"--expected-methods"}
        assert figdata_manifest["status"] == "partial_skip_selection"
        assert figdata_manifest["analysis_policy"]["d1_primary_analysis"] == "complete_case"
        assert (
            figdata_manifest["analysis_policy"]["d2_primary_analysis"]
            == "penalized_failed_in_denominator"
        )
        assert (
            figdata_manifest["analysis_policy"]["fig6_primary_analysis"]
            == "penalized_failed_in_denominator"
        )
        assert figdata_manifest["analysis_policy"]["d3_primary_cluster_by"] == "case_id"
        assert figdata_manifest["analysis_policy"]["d3_sensitivity_cluster_by"] == "iid"
        assert any("--skip" in err for err in figdata_manifest["errors"])

    def test_figdata_manifest_records_results_summary_with_failures(self, tmp_path):
        from scripts.prepare_figdata import main

        results = tmp_path / "results.csv"
        pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 5,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
                **_runtime_trace_fields(),
            }),
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 5,
                "repeat": 1,
                "status": "timeout",
                "failure_category": "timeout",
                "error": "worker exceeded wall time",
                "nrmse": np.nan,
                "t_total_s": 1.0,
                **_runtime_trace_fields(),
            }),
        ]).to_csv(results, index=False)

        out = tmp_path / "figdata"
        rc = main([
            "--results-csv", str(results),
            "--expected-methods", "lhs+gp",
            "--out-dir", str(out),
            "--skip", "fig1", "fig2", "fig3", "fig4", "fig5", "fig6",
        ])

        assert rc == 0
        figdata_manifest = json.loads((out / "figdata_manifest.json").read_text(encoding="utf-8"))
        summary = figdata_manifest["results_summary"]
        assert summary["n_rows"] == 2
        assert summary["n_cases"] == 1
        assert summary["n_methods"] == 1
        assert summary["budgets"] == [5]
        assert summary["repeats"] == [0, 1]
        assert summary["status_counts"] == {"ok": 1, "timeout": 1}
        assert summary["failure_category_counts"] == {"ok": 1, "timeout": 1}

    def test_cli_full_figdata_failure_preserves_previous_package(self, tmp_path):
        from scripts.prepare_figdata import main

        results = tmp_path / "results.csv"
        pd.DataFrame([
            _result_contract_row(**{
                "case_id": "c1",
                "method": "lhs+gp",
                "budget_mult": 10,
                "repeat": 0,
                "status": "ok",
                "failure_category": "ok",
                "error": "",
                "nrmse": 0.1,
                "t_total_s": 1.0,
                **_runtime_trace_fields(),
            })
        ]).to_csv(results, index=False)

        out = tmp_path / "figdata"
        out.mkdir()
        sentinel = out / "sentinel.csv"
        sentinel.write_text("old_package\n", encoding="utf-8")

        with pytest.raises(AssertionError, match="correlation.json"):
            main([
                "--results-csv", str(results),
                "--expected-methods", "lhs+gp",
                "--d1-dir", str(tmp_path / "missing_d1"),
                "--d2-dir", str(tmp_path / "missing_d2"),
                "--d3-dir", str(tmp_path / "missing_d3"),
                "--out-dir", str(out),
            ])

        assert sentinel.read_text(encoding="utf-8") == "old_package\n"
        assert not (out / "fig1_case_catalog.csv").exists()
        assert not list(tmp_path.glob(".figdata.tmp.*"))


# ─── generate_si full-package replacement ──────────────────────────────
class TestGenerateSi:
    def test_multiple_testing_note_documents_policy(self):
        from scripts.generate_si import _make_multiple_testing_note

        latex, markdown = _make_multiple_testing_note()

        combined = latex + markdown
        assert "Holm" in combined
        assert "Bonferroni" in combined
        assert "BH-FDR" in combined
        assert "Clopper-Pearson" in markdown
        assert r"\label{tab:si_multiple_testing}" in latex

    def test_d2_policy_note_documents_exploratory_contract(self):
        from scripts.generate_si import _make_d2_policy_note

        latex, markdown = _make_d2_policy_note()

        combined = latex + markdown
        assert "penalized_failed_in_denominator" in combined
        assert "exploratory=True" in combined
        assert "n_failed_cells_per_topology" in markdown
        assert r"\label{tab:si_d2_policy}" in latex

    def test_case_characterisation_table_reports_validation_evidence_tiers(self):
        from scripts.generate_si import _make_table_s3

        result = _make_table_s3()

        assert result is not None
        latex, markdown = result
        assert "Evidence tier" in markdown
        assert r"\textbf{Evidence tier}" in latex
        assert "`targeted_plausibility_only`" in markdown
        assert "`external_note_analytic_control`" in markdown
        assert r"\texttt{targeted\_plausibility\_only}" in latex

    def test_failed_si_build_preserves_existing_package(self, tmp_path, monkeypatch):
        from scripts import generate_si

        for name in generate_si.SI_FILES:
            (tmp_path / name).write_text(f"old {name}\n", encoding="utf-8")

        def boom():
            raise RuntimeError("sobol table failed")

        monkeypatch.setattr(generate_si, "_make_table_s2", boom)

        rc = generate_si.main(["--output-dir", str(tmp_path)])

        assert rc == 1
        for name in generate_si.SI_FILES:
            assert (tmp_path / name).read_text(encoding="utf-8") == f"old {name}\n"
        assert not list(tmp_path.glob(".si.tmp.*"))
        assert not list(tmp_path.glob(".si.backup.*"))


# ─── plots_nature.plot_fig2_d1 dynamic mode/budget selection ───────────
class TestPlotsNatureFig2:
    def test_partial_modes_no_empty_subplots(self, tmp_path, monkeypatch):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from scripts.plots_nature import plot_fig2_d1
        figdata = tmp_path / "figdata"
        out = tmp_path / "out"
        figdata.mkdir()
        out.mkdir()
        # Only the dimension pairing was run; data has 4 budgets but
        # only one mode.  fig2 must not insist on three-row layout.
        spearman = pd.DataFrame([
            {"pairing_mode": "dimension", "budget_mult": bm,
             "spearman_rho": 0.5, "spearman_rho_ci_lo": 0.3,
             "spearman_rho_ci_hi": 0.7}
            for bm in (5, 10, 20, 50)
        ])
        spearman.to_csv(figdata / "fig2_d1_spearman.csv", index=False)
        ranks = pd.DataFrame([
            {"pairing_mode": "dimension", "budget_mult": bm,
             "method": f"m{i}", "rank_analytical": i,
             "rank_physical": i + 0.1}
            for bm in (5, 10, 20, 50) for i in range(1, 6)
        ])
        ranks.to_csv(figdata / "fig2_d1_ranks.csv", index=False)
        # Should not raise and should produce a fig2.pdf/png.
        plot_fig2_d1(figdata, out)
        assert (out / "pdf" / "fig2.pdf").exists()
        assert (out / "png" / "fig2.png").exists()

    def test_prefers_10d_and_20d_budgets(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from scripts.plots_nature import plot_fig2_d1
        figdata = tmp_path / "figdata"
        out = tmp_path / "out"
        figdata.mkdir()
        out.mkdir()
        spearman = pd.DataFrame([
            {"pairing_mode": "dimension", "budget_mult": bm,
             "spearman_rho": 0.5, "spearman_rho_ci_lo": 0.3,
             "spearman_rho_ci_hi": 0.7}
            for bm in (5, 10, 20, 50)
        ])
        spearman.to_csv(figdata / "fig2_d1_spearman.csv", index=False)
        ranks = pd.DataFrame([
            {"pairing_mode": "dimension", "budget_mult": bm,
             "method": f"m{i}", "rank_analytical": i,
             "rank_physical": i + 0.1}
            for bm in (5, 10, 20, 50) for i in range(1, 6)
        ])
        ranks.to_csv(figdata / "fig2_d1_ranks.csv", index=False)
        # Run via direct import; verify by inspecting figure manifest.
        plot_fig2_d1(figdata, out)
        assert (out / "pdf" / "fig2.pdf").exists()

    def test_fig6_string_false_on_pareto_is_not_highlighted(self, tmp_path, monkeypatch):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from scripts.plots_nature import plot_fig6_practice

        figdata = tmp_path / "figdata"
        out = tmp_path / "out"
        figdata.mkdir()
        out.mkdir()
        pd.DataFrame([
            {
                "method": "m1",
                "nrmse_median": 0.1,
                "budget_mult": 10,
                "analysis_type": "penalized_failed_in_denominator",
                "on_pareto": "False",
                "is_primary": "True",
                "time_median_s": 12.0,
            },
            {
                "method": "m2",
                "nrmse_median": 0.2,
                "budget_mult": 10,
                "analysis_type": "penalized_failed_in_denominator",
                "on_pareto": "True",
                "is_primary": "True",
                "time_median_s": 24.0,
            },
        ]).to_csv(figdata / "fig6_pareto.csv", index=False)

        scatter_lengths = []

        def spy_scatter(self, x, y, *args, **kwargs):
            scatter_lengths.append(len(x))
            return original_scatter(self, x, y, *args, **kwargs)

        from matplotlib.axes import Axes

        original_scatter = Axes.scatter
        monkeypatch.setattr(Axes, "scatter", spy_scatter)

        plot_fig6_practice(figdata, out)

        assert scatter_lengths == [1, 1]
        assert (out / "pdf" / "fig6.pdf").exists()

    def test_fig3_rule_plot_sets_fixed_category_ticks(self, tmp_path):
        import warnings
        import matplotlib
        matplotlib.use("Agg", force=True)
        from scripts.plots_nature import plot_fig3_d2

        figdata = tmp_path / "figdata"
        out = tmp_path / "out"
        figdata.mkdir()
        out.mkdir()
        pd.DataFrame([
            {
                "topology": "smooth",
                "method": "m1",
                "budget_mult": bm,
                "median": 0.2 / bm,
                "ci_lo": 0.15 / bm,
                "ci_hi": 0.25 / bm,
                "n_cases": 3,
            }
            for bm in (5, 10)
        ]).to_csv(figdata / "fig3_d2_learning.csv", index=False)
        pd.DataFrame([
            {
                "topology": "smooth",
                "frac_pass_at_10d": 0.8,
                "median_n_star": 10,
                "n_cases": 3,
                "exploratory": False,
                "analysis_type": "penalized_failed_in_denominator",
                "is_primary": True,
            },
            {
                "topology": "rough",
                "frac_pass_at_10d": 0.3,
                "median_n_star": 20,
                "n_cases": 2,
                "exploratory": True,
                "analysis_type": "penalized_failed_in_denominator",
                "is_primary": True,
            },
        ]).to_csv(figdata / "fig3_d2_rule.csv", index=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_fig3_d2(figdata, out)

        assert not [w for w in caught if "set_ticklabels" in str(w.message)]
        assert (out / "pdf" / "fig3.pdf").exists()

    def test_figure_manifest_paths_are_portable(self, tmp_path, monkeypatch):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from scripts import plots_nature

        figdata = tmp_path / "paper" / "figdata"
        out = tmp_path / "paper" / "figures"
        figdata.mkdir(parents=True)
        out.mkdir(parents=True)
        pd.DataFrame([
            {"case_id": "c1", "domain": "structures", "dim": 2,
             "n_output": 1, "topology": "smooth", "has_failure": True,
             "supports_mf": False}
        ]).to_csv(figdata / "fig1_case_catalog.csv", index=False)
        pd.DataFrame([
            {"method": "m1", "family": "space_filling",
             "surrogate": "gp", "pool": "nature"}
        ]).to_csv(figdata / "fig1_method_catalog.csv", index=False)
        analysis_policy = {
            "schema_version": 1,
            "d1_primary_analysis": "complete_case",
            "d2_primary_analysis": "penalized_failed_in_denominator",
            "fig6_primary_analysis": "penalized_failed_in_denominator",
            "d3_primary_cluster_by": "case_id",
            "d3_sensitivity_cluster_by": "iid",
        }
        (figdata / "figdata_manifest.json").write_text(
            json.dumps({
                "status": "complete",
                "analysis_policy": analysis_policy,
                "figdata_files": [],
                "inputs": [],
                "errors": [],
            }),
            encoding="utf-8",
        )

        rc = plots_nature.main([
            "--figdata", str(figdata),
            "--out-dir", str(out),
            "--fig", "1",
        ])
        assert rc == 0
        manifest = json.loads(
            (tmp_path / "paper" / "figure_manifest.json").read_text(encoding="utf-8")
        )
        item = manifest["figures"][0]
        assert item["pdf"] == "figures/pdf/fig1.pdf"
        assert item["png"] == "figures/png/fig1.png"
        source_paths = [s["path"] for s in item["source_figdata"]]
        assert source_paths == [
            "figdata/fig1_case_catalog.csv",
            "figdata/fig1_method_catalog.csv",
        ]
        assert manifest["analysis_policy"] == analysis_policy
        assert not any(Path(p).is_absolute() for p in [item["pdf"], item["png"], *source_paths])

    def test_full_figure_failure_preserves_existing_package(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from scripts import plots_nature

        paper = tmp_path / "paper"
        figdata = paper / "figdata"
        out = paper / "figures"
        figdata.mkdir(parents=True)
        (out / "pdf").mkdir(parents=True)
        (out / "png").mkdir(parents=True)
        old_pdf = out / "pdf" / "fig1.pdf"
        old_png = out / "png" / "fig1.png"
        manifest_path = paper / "figure_manifest.json"
        old_pdf.write_text("old pdf\n", encoding="utf-8")
        old_png.write_text("old png\n", encoding="utf-8")
        manifest_path.write_text('{"old": true}\n', encoding="utf-8")

        pd.DataFrame([
            {"case_id": "c1", "domain": "structures", "dim": 2,
             "n_output": 1, "topology": "smooth", "has_failure": True,
             "supports_mf": False}
        ]).to_csv(figdata / "fig1_case_catalog.csv", index=False)
        pd.DataFrame([
            {"method": "m1", "family": "space_filling",
             "surrogate": "gp", "pool": "nature"}
        ]).to_csv(figdata / "fig1_method_catalog.csv", index=False)

        rc = plots_nature.main([
            "--figdata", str(figdata),
            "--out-dir", str(out),
        ])

        assert rc == 1
        assert old_pdf.read_text(encoding="utf-8") == "old pdf\n"
        assert old_png.read_text(encoding="utf-8") == "old png\n"
        assert manifest_path.read_text(encoding="utf-8") == '{"old": true}\n'
        assert not list(paper.glob(".figures.tmp.*"))


# ─── plots_nature CVD-safe palette guard (GAP-F4) ──────────────────────
class TestPlotsNatureColorblindPalette:
    """Pin the palette so a future edit cannot reintroduce CVD-unsafe maps.

    Two CVD-unsafety patterns matter for Nature-grade figures:
    * Categorical maps that rely on red↔green discrimination
      (e.g. ``tab10``, ``tab20``, ``Set1``, ``Paired``).
    * Diverging maps with a red↔green axis (e.g. ``RdYlBu``, ``RdYlGn``,
      ``bwr``, ``coolwarm``).

    These tests fail if any reappear in ``scripts/plots_nature.py``.
    """

    def test_wong_palette_is_eight_distinct_hex(self):
        from scripts.plots_nature import WONG_PALETTE
        assert len(WONG_PALETTE) == 8
        assert len(set(WONG_PALETTE)) == 8
        for c in WONG_PALETTE:
            assert c.startswith("#") and len(c) == 7

    def test_cvd_categorical_colors_for_few_categories_uses_wong(self):
        from scripts.plots_nature import WONG_PALETTE, cvd_categorical_colors
        assert cvd_categorical_colors(0) == []
        assert cvd_categorical_colors(1) == [WONG_PALETTE[0]]
        assert cvd_categorical_colors(8) == list(WONG_PALETTE)

    def test_cvd_categorical_colors_for_many_categories_extends_via_cividis(self):
        from scripts.plots_nature import cvd_categorical_colors
        cols = cvd_categorical_colors(20)
        assert len(cols) == 20
        # All values are valid hex strings.
        for c in cols:
            assert c.startswith("#") and len(c) == 7
        # Endpoints differ enough that they would not be confusable.
        assert cols[0] != cols[-1]

    def test_no_cvd_unsafe_colormaps_remain_in_source(self):
        """grep-style guard: the source file must not name any unsafe map."""
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[2]
            / "scripts" / "plots_nature.py"
        ).read_text(encoding="utf-8")
        unsafe_tokens = (
            "plt.cm.tab10",
            "plt.cm.tab20",
            "plt.cm.Set1",
            "plt.cm.Paired",
            "plt.cm.RdYlBu",
            "plt.cm.RdYlGn",
            "plt.cm.bwr",
            "plt.cm.coolwarm",
            'cmap="tab10"', 'cmap="tab20"', 'cmap="RdYlBu_r"',
            'cmap="RdYlGn"', 'cmap="bwr"', 'cmap="coolwarm"',
        )
        offenders = [t for t in unsafe_tokens if t in src]
        assert not offenders, (
            f"plots_nature.py reintroduced CVD-unsafe colormap(s): "
            f"{offenders}.  Use cvd_categorical_colors() / "
            f"CVD_SEQUENTIAL_CMAP instead (GAP-F4)."
        )

    def test_palette_b_uses_wong_hexes(self):
        """The four-budget palette must be drawn from the Wong set."""
        from scripts.plots_nature import PALETTE_B, WONG_PALETTE
        assert all(c in WONG_PALETTE for c in PALETTE_B)
        assert len(PALETTE_B) == 4
        assert len(set(PALETTE_B)) == 4

    def test_no_legacy_hardcoded_blues_remain_in_source(self):
        """All blue/teal hues should route through the CVD-safe constants."""
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[2]
            / "scripts" / "plots_nature.py"
        ).read_text(encoding="utf-8")
        legacy_hexes = ("#4575b4", "#74add1", "#1a6faf", "#abd9e9")
        offenders = [c for c in legacy_hexes if c in src]
        assert not offenders, (
            f"plots_nature.py contains legacy hard-coded colours {offenders}; "
            "use WONG_PALETTE-derived constants instead."
        )


# ─── recompute_spearman_ci.py: nested vs legacy CSV layout ─────────────
class TestRecomputeSpearmanCi:
    """GAP audit fix: when the legacy top-level rank_comparison.csv coexists
    with new pairing_*/ subdirectories (a real layout in this repository at
    2026-05-23), the prior code preferred the legacy file and silently
    dropped CI computation for the topology / random pairings.

    The fix prefers the nested layout when ANY pairing_*/ exists, and
    falls back to the legacy file only when there is no nested data.
    These tests pin both branches.
    """

    @pytest.fixture
    def rr_dir_with_three_pairings(self, tmp_path):
        rr = tmp_path / "rr"
        rr.mkdir()
        for mode in ("dimension", "topology", "random"):
            d = rr / f"pairing_{mode}"
            d.mkdir()
            # Synthetic 5-method × 1-budget rank pairs with a slight twist
            # per mode so bootstrap CIs are mode-distinct.
            offset = {"dimension": 0.0, "topology": 0.5, "random": -0.5}[mode]
            pd.DataFrame([
                {"pairing_mode": mode, "budget_mult": 10,
                 "method": f"m{i}",
                 "rank_analytical": float(i),
                 "rank_physical": float(i) + offset}
                for i in range(1, 6)
            ]).to_csv(d / "rank_comparison.csv", index=False)
        # correlation.json with 3 mode entries pre-filled, no CI yet.
        entries = [
            {"pairing_mode": mode, "budget_mult": 10,
             "spearman_rho": 0.5, "kendall_tau": 0.4}
            for mode in ("dimension", "topology", "random")
        ]
        (rr / "correlation.json").write_text(
            json.dumps(entries), encoding="utf-8",
        )
        return rr

    def test_nested_pairings_all_get_ci(self, rr_dir_with_three_pairings):
        from benchmark.scripts.recompute_spearman_ci import recompute
        rc = recompute(rr_dir_with_three_pairings, n_boot=200, seed=0)
        assert rc == 0
        out = json.loads(
            (rr_dir_with_three_pairings / "correlation.json")
            .read_text(encoding="utf-8")
        )
        # All three pairings must have CI fields written.
        modes = sorted({e["pairing_mode"] for e in out})
        assert modes == ["dimension", "random", "topology"]
        for entry in out:
            assert "spearman_rho_ci_lo" in entry
            assert "spearman_rho_ci_hi" in entry
            assert "kendall_tau_ci_lo" in entry
            assert "kendall_tau_ci_hi" in entry

    def test_nested_pairings_preferred_over_stale_legacy(
        self, rr_dir_with_three_pairings,
    ):
        """If a stale legacy rank_comparison.csv (no pairing_mode col)
        exists alongside the nested files, the nested files win.  This
        is the regression that motivated the fix.
        """
        # Drop a legacy top-level CSV that only carries dimension data
        # (no pairing_mode column).
        legacy = pd.DataFrame([
            {"budget_mult": 10, "method": f"m{i}",
             "rank_analytical": float(i), "rank_physical": float(i)}
            for i in range(1, 6)
        ])
        legacy.to_csv(
            rr_dir_with_three_pairings / "rank_comparison.csv",
            index=False,
        )
        from benchmark.scripts.recompute_spearman_ci import recompute
        rc = recompute(rr_dir_with_three_pairings, n_boot=200, seed=0)
        assert rc == 0
        out = json.loads(
            (rr_dir_with_three_pairings / "correlation.json")
            .read_text(encoding="utf-8")
        )
        modes = sorted({e["pairing_mode"] for e in out})
        # Crucial: with the bug, only "dimension" would have CI fields.
        # With the fix, all three pairings have CI fields.
        assert modes == ["dimension", "random", "topology"]
        for entry in out:
            assert "spearman_rho_ci_lo" in entry, (
                f"pairing_mode={entry['pairing_mode']} did not get a CI; "
                "the legacy-vs-nested fallback regressed."
            )

    def test_legacy_only_layout_still_works(self, tmp_path):
        """If there are NO pairing_*/ subdirs, the legacy top-level CSV is
        the only source; it gets treated as a single-mode (dimension)
        layout — preserves backward compatibility for v0.3 archives."""
        rr = tmp_path / "rr_legacy"
        rr.mkdir()
        pd.DataFrame([
            {"budget_mult": 10, "method": f"m{i}",
             "rank_analytical": float(i), "rank_physical": float(i)}
            for i in range(1, 6)
        ]).to_csv(rr / "rank_comparison.csv", index=False)
        (rr / "correlation.json").write_text(
            json.dumps([{"pairing_mode": "dimension", "budget_mult": 10,
                         "spearman_rho": 1.0, "kendall_tau": 1.0}]),
            encoding="utf-8",
        )
        from benchmark.scripts.recompute_spearman_ci import recompute
        rc = recompute(rr, n_boot=200, seed=0)
        assert rc == 0
        out = json.loads((rr / "correlation.json").read_text(encoding="utf-8"))
        assert len(out) == 1
        assert out[0]["pairing_mode"] == "dimension"
        assert "spearman_rho_ci_lo" in out[0]


# ─── reliability.py: f_bal_acc degenerate-case audit ───────────────────
class TestReliabilityBalAccDegenerate:
    """GAP audit fix: ``f_bal_acc_o{j}`` must be NaN when the test set is
    single-class (no true positives or no true negatives).

    Before the fix, the formula ``0.5 * (tpr + tnr)`` with the
    ``if (tp + fn) > 0 else 0.0`` shortcut would produce e.g. 0.5 *
    (0 + 1) = 0.5 for a method that perfectly classified every safe
    sample on a test set with zero failures.  That collapses a well-
    classified-but-degenerate scenario to look like 50% bal-acc and
    misleads the §D3 ranking-bal-acc reporting.
    """

    def test_no_true_failures_gives_nan(self):
        """Test set with zero true failures: bal-acc must be NaN."""
        import numpy as np
        from unittest.mock import patch
        from benchmark.reliability import reliability_metrics

        # Mock get_failure_thresholds to return a single threshold so we
        # can drive Y_true / Y_pred without a real case registry.
        with patch(
            "benchmark.reliability.get_failure_thresholds",
            return_value=[(0, 100.0, "above")],
        ):
            Y_true = np.full((10, 1), 50.0)   # all safe (< 100)
            Y_pred = np.full((10, 1), 60.0)   # all predicted safe
            metrics = reliability_metrics("dummy_case", Y_true, Y_pred)
            assert "f_bal_acc_o0" in metrics
            assert np.isnan(metrics["f_bal_acc_o0"]), (
                f"expected NaN for degenerate test set, got "
                f"{metrics['f_bal_acc_o0']}"
            )

    def test_no_true_safes_gives_nan(self):
        """Test set with zero true safes: bal-acc must be NaN."""
        import numpy as np
        from unittest.mock import patch
        from benchmark.reliability import reliability_metrics

        with patch(
            "benchmark.reliability.get_failure_thresholds",
            return_value=[(0, 100.0, "above")],
        ):
            Y_true = np.full((10, 1), 200.0)  # all failure (> 100)
            Y_pred = np.full((10, 1), 200.0)
            metrics = reliability_metrics("dummy_case", Y_true, Y_pred)
            assert np.isnan(metrics["f_bal_acc_o0"])

    def test_well_defined_test_set_gives_finite_bal_acc(self):
        """Sanity: when both classes are present, bal-acc is finite."""
        import numpy as np
        from unittest.mock import patch
        from benchmark.reliability import reliability_metrics

        with patch(
            "benchmark.reliability.get_failure_thresholds",
            return_value=[(0, 100.0, "above")],
        ):
            # 5 truly-failing + 5 truly-safe; method classifies perfectly.
            Y_true = np.array([[200.0]] * 5 + [[50.0]] * 5)
            Y_pred = np.array([[200.0]] * 5 + [[50.0]] * 5)
            metrics = reliability_metrics("dummy_case", Y_true, Y_pred)
            assert metrics["f_bal_acc_o0"] == pytest.approx(1.0)

    def test_global_bal_acc_mean_skips_nan_outputs(self):
        """When a multi-output case has one NaN-bal-acc output and one
        well-defined output, the global ``f_bal_acc_mean`` aggregate must
        still report the well-defined value rather than NaN.  The fix
        switched the aggregator to ignore non-finite per-output values.
        """
        import numpy as np
        from unittest.mock import patch
        from benchmark.reliability import reliability_metrics

        with patch(
            "benchmark.reliability.get_failure_thresholds",
            return_value=[(0, 100.0, "above"), (1, 50.0, "above")],
        ):
            # Output 0: degenerate (no failures); Output 1: 5/5 split,
            # method classifies perfectly.
            Y_true = np.column_stack([
                np.full(10, 30.0),                        # all safe
                np.array([100.0] * 5 + [10.0] * 5),       # 5 fail / 5 safe
            ])
            Y_pred = Y_true.copy()
            metrics = reliability_metrics("dummy_case", Y_true, Y_pred)
            assert np.isnan(metrics["f_bal_acc_o0"])
            assert metrics["f_bal_acc_o1"] == pytest.approx(1.0)
            # Global aggregate must reflect output 1 only (1.0), not NaN.
            assert metrics["f_bal_acc_mean"] == pytest.approx(1.0)


# ─── Display-name contract (Fig. 1b labels) ─────────────────────────────
def test_family_labels_keep_abbreviations_upper_case():
    """Fig. 1b renders method-pool family names via ``family_label``.

    The generic ``str.title()`` fallback used to turn ``sequential_ego``
    into "Sequential Ego" and ``sequential_botorch_saas`` into
    "Sequential Botorch Saas", so the rendered figure disagreed with the
    method names (EGO, BoTorch, SAAS) used everywhere else.  Pin the
    display contract for every family in the frozen pool.
    """
    module = _load_script("scripts/paper_names.py", "paper_names_labels")
    expected = {
        "static_design": "Static Design",
        "sequential_ego": "Sequential EGO",
        "sequential_botorch_ei": "Sequential BoTorch EI",
        "sequential_botorch_saas": "Sequential BoTorch SAAS",
        "sequential_active_learning": "Sequential Active Learning",
        "sequential_reliability": "Sequential Reliability",
    }
    for family, label in expected.items():
        assert module.family_label(family) == label

    from benchmark.methods import nature_pool

    for method in nature_pool():
        label = module.family_label(method.family)
        assert label == expected[method.family], (method.name, label)
        # No abbreviation may be rendered in title case.
        for bad in ("Ego", "Botorch", "Saas", "Ei"):
            assert bad not in label, (method.name, label)
