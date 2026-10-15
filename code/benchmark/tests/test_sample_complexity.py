"""Tests for benchmark.sample_complexity (Discovery 2 module)."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from benchmark.sample_complexity import (
    PRIMARY_NRMSE_THRESHOLD,
    compare_to_10d_rule,
    learning_curves_by_topology,
    n_star_per_topology,
    sample_complexity_report,
)


def test_primary_nrmse_threshold_matches_preregistered_target():
    """The public API default must reproduce the registered D2 target."""
    assert PRIMARY_NRMSE_THRESHOLD == 0.20

    import inspect

    assert inspect.signature(n_star_per_topology).parameters[
        "nrmse_threshold"
    ].default == PRIMARY_NRMSE_THRESHOLD
    assert inspect.signature(compare_to_10d_rule).parameters[
        "nrmse_threshold"
    ].default == PRIMARY_NRMSE_THRESHOLD
    assert inspect.signature(sample_complexity_report).parameters[
        "nrmse_threshold"
    ].default == PRIMARY_NRMSE_THRESHOLD


def _make_synthetic_results(seed: int = 0) -> pd.DataFrame:
    """Synthetic benchmark with two methods and two topology tags.

    Method A converges faster than Method B; method B never reaches
    the 5 % threshold.  Each case has a single topology tag so the
    explode is trivial.
    """
    rng = np.random.default_rng(seed)
    case_topo = {
        "case_smooth_1": ("smooth_monotone",),
        "case_smooth_2": ("smooth_monotone",),
        "case_jump_1": ("threshold_jump",),
        "case_jump_2": ("threshold_jump",),
    }
    rows: List[dict] = []
    for case_id, topo in case_topo.items():
        d = 5
        for budget in [5, 10, 20]:
            for method, decay in [("fast", 0.05), ("slow", 0.20)]:
                # NRMSE ~ decay / sqrt(budget) with small noise
                base = decay / np.sqrt(budget)
                if "jump" in topo[0] and method == "slow":
                    # threshold_jump is harder for the slow method
                    base *= 1.5
                for r in range(4):
                    rows.append({
                        "case_id": case_id,
                        "method": method,
                        "method_family": "static_design",
                        "budget_mult": budget,
                        "repeat": r,
                        "nrmse": float(base * (1 + 0.05 * rng.standard_normal())),
                        "status": "ok",
                        # Stub topology for explode helper
                        "topology": topo,
                        "domain_attr": "structures",
                        "dim_bucket": "mid",
                        "scenarios": (),
                        "has_failure_boundary": False,
                        "cost_level": "L0",
                        "n_inputs": d,
                        "dim": d,
                    })
    return pd.DataFrame(rows)


# ─── learning_curves_by_topology ─────────────────────────────────────────
def test_learning_curves_basic_shape():
    df = _make_synthetic_results()
    learn = learning_curves_by_topology(df, n_boot=100)
    expected_cols = {"topology", "method", "budget_mult", "n_cases",
                     "n_runs", "median", "ci_lo", "ci_hi"}
    assert expected_cols <= set(learn.columns)
    # Two methods × two topologies × three budgets = 12 rows
    assert len(learn) == 12


def test_learning_curves_decreasing_with_budget():
    df = _make_synthetic_results()
    learn = learning_curves_by_topology(df, n_boot=100)
    for (topo, method), sub in learn.groupby(["topology", "method"]):
        sub_sorted = sub.sort_values("budget_mult")
        # The median should be (weakly) non-increasing with budget
        diffs = sub_sorted["median"].diff().dropna()
        # Allow tiny noise but require overall trend
        assert (diffs <= 0.05).all(), (
            f"non-monotone learning curve for {topo}/{method}: {sub_sorted}"
        )


def test_learning_curves_unknown_metric_raises():
    df = _make_synthetic_results()
    with pytest.raises(KeyError):
        learning_curves_by_topology(df, metric="banana")


# ─── n_star_per_topology ─────────────────────────────────────────────────
def test_n_star_picks_smallest_passing_budget():
    df = _make_synthetic_results()
    learn = learning_curves_by_topology(df, n_boot=100)
    n_star = n_star_per_topology(learn, nrmse_threshold=0.03)

    # 'fast' method: 0.05 / sqrt(5) ≈ 0.022 → already below at budget=5
    assert n_star.loc["fast", "smooth_monotone"] == 5.0
    # 'slow' method: 0.20 / sqrt(20) ≈ 0.045 — never reaches 0.03
    assert np.isnan(n_star.loc["slow", "smooth_monotone"])


def test_n_star_returns_df_indexed_by_method():
    df = _make_synthetic_results()
    learn = learning_curves_by_topology(df, n_boot=100)
    n_star = n_star_per_topology(learn, nrmse_threshold=0.05)
    assert "fast" in n_star.index
    assert "slow" in n_star.index
    assert {"smooth_monotone", "threshold_jump"} <= set(n_star.columns)


def test_n_star_monotone_smoothing_helps():
    """If a learning curve has noise spikes, monotone=True should still
    correctly pick the first crossing budget."""
    df = pd.DataFrame([
        # Single case, single topology, one method, three budgets
        {"case_id": "c1", "method": "m", "method_family": "x",
         "budget_mult": 5, "repeat": 0, "nrmse": 0.10, "status": "ok",
         "topology": ("smooth_monotone",), "domain_attr": "x",
         "dim_bucket": "low", "scenarios": (), "has_failure_boundary": False,
         "cost_level": "L0"},
        {"case_id": "c1", "method": "m", "method_family": "x",
         "budget_mult": 10, "repeat": 0, "nrmse": 0.04, "status": "ok",
         "topology": ("smooth_monotone",), "domain_attr": "x",
         "dim_bucket": "low", "scenarios": (), "has_failure_boundary": False,
         "cost_level": "L0"},
        {"case_id": "c1", "method": "m", "method_family": "x",
         "budget_mult": 20, "repeat": 0, "nrmse": 0.06, "status": "ok",
         "topology": ("smooth_monotone",), "domain_attr": "x",
         "dim_bucket": "low", "scenarios": (), "has_failure_boundary": False,
         "cost_level": "L0"},
    ])
    learn = learning_curves_by_topology(df, n_boot=100)
    # With monotone smoothing the budget=10 crossing is preserved
    n_star_smoothed = n_star_per_topology(learn, nrmse_threshold=0.05,
                                          monotone=True)
    assert n_star_smoothed.loc["m", "smooth_monotone"] == 10.0
    # Without smoothing, same answer (budget=10 still passes first)
    n_star_raw = n_star_per_topology(learn, nrmse_threshold=0.05,
                                     monotone=False)
    assert n_star_raw.loc["m", "smooth_monotone"] == 10.0


# ─── compare_to_10d_rule ─────────────────────────────────────────────────
def test_compare_to_10d_rule_summary_shape():
    df = _make_synthetic_results()
    learn = learning_curves_by_topology(df, n_boot=100)
    rule = compare_to_10d_rule(
        learn, nrmse_threshold=0.05, rule_budget=10, source_df=df,
    )
    expected = {"topology", "methods_total", "methods_pass_at_10d",
                "frac_pass_at_10d", "median_n_star", "max_n_star",
                "n_methods_never_pass", "n_cases", "case_ids",
                "n_failed_cells_per_topology", "exploratory",
                "analysis_type", "is_primary"}
    assert expected <= set(rule.columns)
    assert {"smooth_monotone", "threshold_jump"} <= set(rule["topology"])
    assert set(rule["analysis_type"]) == {"penalized_failed_in_denominator"}
    assert rule["is_primary"].astype(bool).all()


def test_compare_to_10d_rule_threshold_passes_for_fast():
    df = _make_synthetic_results()
    learn = learning_curves_by_topology(df, n_boot=100)
    rule = compare_to_10d_rule(
        learn, nrmse_threshold=0.05, rule_budget=10, source_df=df,
    )
    # On smooth_monotone with threshold 5%, fast should pass at <=10d
    smooth = rule[rule["topology"] == "smooth_monotone"].iloc[0]
    assert smooth["methods_pass_at_10d"] >= 1


def test_compare_to_10d_rule_uses_full_source_method_pool_denominator():
    df = _make_synthetic_results()
    extra = df.iloc[[0]].copy()
    extra["method"] = "timeout_only"
    extra["status"] = "timeout"
    extra["nrmse"] = np.nan
    # The timeout-only method exists in source_df but is absent from
    # learning_df because learning_curves_by_topology filters status != ok.
    df = pd.concat([df, extra], ignore_index=True)

    learn = learning_curves_by_topology(df, n_boot=100)
    assert "timeout_only" not in set(learn["method"])
    rule = compare_to_10d_rule(
        learn, nrmse_threshold=0.05, rule_budget=10, source_df=df,
    )
    smooth = rule[rule["topology"] == "smooth_monotone"].iloc[0]
    assert smooth["methods_total"] == 3
    assert smooth["n_methods_never_pass"] == 1
    assert smooth["methods_pass_at_10d"] == 1
    assert smooth["frac_pass_at_10d"] == pytest.approx(1 / 3)
    assert smooth["n_failed_cells_per_topology"] == 1
    assert "case_smooth_1" in smooth["case_ids"]
    assert smooth["n_cases"] == 2
    assert bool(smooth["exploratory"]) is True


def test_sample_complexity_report_rule_summary_uses_source_df():
    df = _make_synthetic_results()
    failed = df.iloc[[0]].copy()
    failed["method"] = "failed_only"
    failed["status"] = "failed"
    failed["nrmse"] = np.nan
    df = pd.concat([df, failed], ignore_index=True)

    out = sample_complexity_report(df, n_boot=100, nrmse_threshold=0.05)
    smooth = out["rule_summary"][
        out["rule_summary"]["topology"] == "smooth_monotone"
    ].iloc[0]
    # If sample_complexity_report forgot to pass source_df=df, this would be
    # 2 because failed_only is absent from learning curves after status filter.
    assert smooth["methods_total"] == 3
    assert smooth["n_failed_cells_per_topology"] == 1


# ─── Wrapper ─────────────────────────────────────────────────────────────
def test_sample_complexity_report_returns_three_keys():
    df = _make_synthetic_results()
    out = sample_complexity_report(df, n_boot=200)
    for k in ("learning", "n_star", "rule_summary"):
        assert k in out
        assert isinstance(out[k], pd.DataFrame)


def test_sample_complexity_cli_rejects_invalid_args_before_output(tmp_path):
    from benchmark.scripts.run_sample_complexity import main

    out = tmp_path / "sample_complexity"

    rc = main([
        str(tmp_path / "missing_results.csv"),
        "--threshold", "0",
        "--rule-budget", "0",
        "--output-dir", str(out),
    ])

    assert rc == 2
    assert not out.exists()


def test_sample_complexity_cli_rejects_corrupt_results_before_output(tmp_path):
    from benchmark.scripts.run_sample_complexity import main

    results = tmp_path / "results.csv"
    row = {
        "case_id": "c1",
        "method": "lhs+gp",
        "budget_mult": 10,
        "repeat": 0,
        "status": "ok",
        "error": "",
        "nrmse": 0.1,
        "t_total_s": 1.0,
        "topology": ("smooth_monotone",),
        "domain_attr": "synthetic",
        "dim_bucket": "low",
        "scenarios": (),
        "has_failure_boundary": False,
        "cost_level": "L0",
    }
    pd.DataFrame([row, dict(row, nrmse=0.2)]).to_csv(results, index=False)
    out = tmp_path / "sample_complexity"

    rc = main([
        str(results),
        "--output-dir", str(out),
    ])

    assert rc == 2
    assert not out.exists()


def test_sample_complexity_cli_require_headline_source_rejects_fixture_results(tmp_path):
    from benchmark.scripts.run_sample_complexity import main

    results = tmp_path / "results.csv"
    row = {
        "case_id": "c1",
        "method": "lhs+gp",
        "budget_mult": 10,
        "repeat": 0,
        "status": "ok",
        "error": "",
        "nrmse": 0.1,
        "t_total_s": 1.0,
        "topology": ("smooth_monotone",),
        "domain_attr": "synthetic",
        "dim_bucket": "low",
        "scenarios": (),
        "has_failure_boundary": False,
        "cost_level": "L0",
    }
    pd.DataFrame([row]).to_csv(results, index=False)
    out = tmp_path / "sample_complexity"

    rc = main([
        str(results),
        "--output-dir", str(out),
        "--require-headline-source",
    ])

    assert rc == 2
    assert not out.exists()
