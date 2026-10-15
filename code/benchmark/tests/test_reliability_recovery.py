"""Tests for benchmark.reliability_recovery (Discovery 3 module)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.reliability_recovery import (
    extract_long_signed_bias,
    per_method_reliability_summary,
    reliability_ci_sensitivity,
    reliability_ranking,
    reliability_recovery_report,
    signed_bias_by_family,
)


# ─── Helpers to build synthetic results.csv ──────────────────────────────
def _make_synthetic_results(
    n_cases: int = 4,
    n_repeats: int = 8,
    methods=("lhs_opt+gp", "lhs_opt+rf", "lhs+gp"),
    families=("static_design", "static_design", "static_design"),
    bias_per_method=(-0.05, 0.0, -0.10),  # signed P_f bias
    seed: int = 0,
) -> pd.DataFrame:
    """Build a results.csv-shaped DataFrame with controllable bias.

    Each method draws a Gaussian-noise signed bias ``b ~ N(mu_method, sigma)``
    and we set ``pf_pred_o0 = pf_true_o0 + b``.  ``pf_abs_err_o0`` is set
    consistently.  ``f_bal_acc_o0`` is randomised in [0.5, 1.0].
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cases):
        case_id = f"case_{c:02d}"
        for method, family, mu in zip(methods, families, bias_per_method, strict=False):
            for r in range(n_repeats):
                pf_true = float(rng.uniform(0.05, 0.4))
                bias = float(rng.normal(mu, 0.02))
                pf_pred = float(np.clip(pf_true + bias, 0.0, 1.0))
                bal_acc = float(rng.uniform(0.55, 0.95))
                rows.append({
                    "case_id": case_id,
                    "method": method,
                    "method_family": family,
                    "budget_mult": 10,
                    "repeat": r,
                    "pf_true_o0": pf_true,
                    "pf_pred_o0": pf_pred,
                    "pf_abs_err_o0": abs(pf_pred - pf_true),
                    "f_bal_acc_o0": bal_acc,
                    "f_brier_o0": float(rng.uniform(0.01, 0.20)),
                    "status": "ok",
                })
    return pd.DataFrame(rows)


# ─── Long-form extraction ────────────────────────────────────────────────
def test_extract_long_signed_bias_basic_shape():
    df = _make_synthetic_results(n_cases=3, n_repeats=4)
    long = extract_long_signed_bias(df)
    assert len(long) == len(df)
    for col in ("case_id", "method", "method_family", "budget_mult",
                "repeat", "pf_abs_err", "pf_signed_bias",
                "f_bal_acc", "f_brier"):
        assert col in long.columns


def test_extract_long_drops_no_threshold_rows():
    """Rows where every reliability metric is NaN must be dropped."""
    df = _make_synthetic_results(n_cases=2, n_repeats=2)
    extra = pd.DataFrame([{
        "case_id": "no_threshold",
        "method": "lhs+gp",
        "method_family": "static_design",
        "budget_mult": 10,
        "repeat": 0,
        # no pf_/f_ columns
        "status": "ok",
    }])
    df_full = pd.concat([df, extra], ignore_index=True)
    long = extract_long_signed_bias(df_full)
    assert "no_threshold" not in long["case_id"].values


def test_extract_long_excludes_non_ok_rows_even_if_metrics_are_present():
    """Inferential D3 rows must follow the registered status filter."""
    df = _make_synthetic_results(n_cases=1, n_repeats=2)
    failed = df.iloc[[0]].copy()
    failed["status"] = "failed"
    failed["pf_pred_o0"] = failed["pf_true_o0"] + 0.9
    failed["pf_abs_err_o0"] = 0.9
    long = extract_long_signed_bias(pd.concat([df, failed], ignore_index=True))
    assert len(long) == len(df)
    assert (long["pf_abs_err"] < 0.9).all()


def test_extract_long_signed_bias_aggregate_max_picks_largest_abs():
    """When two outputs disagree on sign, max-aggregate must keep the
    bias with the larger absolute value (so the worst-case dominates)."""
    df = pd.DataFrame([{
        "case_id": "dual_threshold",
        "method": "lhs+gp",
        "method_family": "static_design",
        "budget_mult": 10,
        "repeat": 0,
        "pf_true_o0": 0.20, "pf_pred_o0": 0.10,  # bias = -0.10
        "pf_abs_err_o0": 0.10,
        "pf_true_o1": 0.30, "pf_pred_o1": 0.35,  # bias = +0.05
        "pf_abs_err_o1": 0.05,
    }])
    long_max = extract_long_signed_bias(df, output_aggregate="max")
    assert long_max["pf_signed_bias"].iloc[0] == pytest.approx(-0.10)
    assert long_max["pf_abs_err"].iloc[0] == pytest.approx(0.10)

    long_mean = extract_long_signed_bias(df, output_aggregate="mean")
    assert long_mean["pf_signed_bias"].iloc[0] == pytest.approx(-0.025)


def test_extract_long_canonicalises_probability_difference_precision():
    """Decimal-grid probabilities must not create parser-dependent tails."""
    df = pd.DataFrame([{
        "case_id": "precision_case",
        "method": "lhs+gp",
        "method_family": "static_design",
        "budget_mult": 10,
        "repeat": 0,
        "pf_true_o0": 0.10,
        "pf_pred_o0": 0.07,
        "pf_abs_err_o0": 0.03,
    }])
    long = extract_long_signed_bias(df)
    assert long["pf_signed_bias"].iloc[0] == 0.07 - 0.10


def test_extract_long_signed_bias_rejects_unknown_aggregate():
    df = _make_synthetic_results(n_cases=1, n_repeats=1)
    with pytest.raises(ValueError, match="output_aggregate"):
        extract_long_signed_bias(df, output_aggregate="median")


# ─── Per-method summary ──────────────────────────────────────────────────
def test_per_method_summary_detects_known_negative_bias():
    """A method with mu=-0.10 must have signed_bias_mean ≈ -0.10
    and a small Wilcoxon one-sided p-value (< 0.05)."""
    df = _make_synthetic_results(
        n_cases=4, n_repeats=10,
        methods=("biased_neg", "unbiased"),
        families=("static_design", "static_design"),
        bias_per_method=(-0.10, 0.0),
        seed=0,
    )
    long = extract_long_signed_bias(df)
    summary = per_method_reliability_summary(long, n_boot=500, seed=0)

    biased = summary[summary["method"] == "biased_neg"].iloc[0]
    unbiased = summary[summary["method"] == "unbiased"].iloc[0]

    assert biased["signed_bias_mean"] < -0.05, (
        f"expected negative mean bias for biased_neg, got {biased['signed_bias_mean']:.3f}"
    )
    assert biased["signed_bias_pvalue_neg"] < 0.05, (
        f"Wilcoxon should reject H0 for biased_neg; p={biased['signed_bias_pvalue_neg']}"
    )
    # unbiased method must NOT have a strongly significant negative bias
    assert abs(unbiased["signed_bias_mean"]) < 0.03


def test_per_method_summary_columns_complete():
    df = _make_synthetic_results(n_cases=3, n_repeats=8)
    long = extract_long_signed_bias(df)
    summary = per_method_reliability_summary(long, n_boot=200)
    expected = {
        "method", "method_family", "n_runs",
        "pf_abs_err_mean", "pf_abs_err_ci_lo", "pf_abs_err_ci_hi",
        "signed_bias_mean", "signed_bias_ci_lo", "signed_bias_ci_hi",
        "signed_bias_pvalue_neg",
        "signed_bias_pvalue_pos",
        "signed_bias_pvalue_two_sided",
        "f_bal_acc_mean", "f_brier_mean",
        "low_n",
    }
    assert expected <= set(summary.columns)


def test_per_method_summary_flags_low_n_methods():
    df = _make_synthetic_results(n_cases=1, n_repeats=9, methods=("m_low",))
    long = extract_long_signed_bias(df)
    summary = per_method_reliability_summary(long, n_boot=100)

    row = summary.iloc[0]
    assert row["n_runs"] == 9
    assert bool(row["low_n"]) is True

    df_ok = _make_synthetic_results(n_cases=1, n_repeats=10, methods=("m_ok",))
    long_ok = extract_long_signed_bias(df_ok)
    summary_ok = per_method_reliability_summary(long_ok, n_boot=100)
    assert summary_ok.iloc[0]["n_runs"] == 10
    assert bool(summary_ok.iloc[0]["low_n"]) is False


def test_per_method_summary_reports_both_one_sided_pvalues():
    """Regression test for the v0.3 manuscript bug: only `pvalue_neg`
    was reported, and ``p ≈ 1`` was then read as "non-conservatism is
    rejected".  A method with a positive signed bias must instead show
    ``pvalue_pos`` small and ``pvalue_neg`` near 1.
    """
    df = _make_synthetic_results(
        n_cases=5, n_repeats=12,
        methods=("biased_pos",),
        families=("static_design",),
        bias_per_method=(+0.10,),
        seed=7,
    )
    long = extract_long_signed_bias(df)
    summary = per_method_reliability_summary(long, n_boot=500, seed=0)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["signed_bias_mean"] > 0.05
    # H1: bias > 0 must be strongly supported
    assert row["signed_bias_pvalue_pos"] < 0.01, (
        f"pvalue_pos should reject H0; got {row['signed_bias_pvalue_pos']:.3g}"
    )
    # H1: bias < 0 must NOT be supported (p near 1)
    assert row["signed_bias_pvalue_neg"] > 0.5, (
        f"pvalue_neg should be large for a positive-bias method; "
        f"got {row['signed_bias_pvalue_neg']:.3g}"
    )
    # Two-sided must also reject
    assert row["signed_bias_pvalue_two_sided"] < 0.05


def test_per_method_summary_sorted_ascending_by_abs_err():
    df = _make_synthetic_results(n_cases=4, n_repeats=8)
    long = extract_long_signed_bias(df)
    summary = per_method_reliability_summary(long)
    assert (summary["pf_abs_err_mean"].diff().dropna() >= -1e-9).all()


def test_per_method_summary_filters_by_budget():
    df = _make_synthetic_results(n_cases=2, n_repeats=4)
    df_b20 = df.copy()
    df_b20["budget_mult"] = 20
    df_full = pd.concat([df, df_b20], ignore_index=True)
    long = extract_long_signed_bias(df_full)
    s10 = per_method_reliability_summary(long, budget_mult=10)
    s_all = per_method_reliability_summary(long)
    # All-budgets summary has roughly 2× the runs of single-budget
    assert s10["n_runs"].sum() < s_all["n_runs"].sum()


def test_reliability_recovery_cli_rejects_invalid_args_before_output(tmp_path):
    from benchmark.scripts.run_reliability_recovery import main

    out = tmp_path / "reliability_recovery"

    rc = main([
        str(tmp_path / "missing_results.csv"),
        "--budget-mult", "0",
        "--output-dir", str(out),
    ])

    assert rc == 2
    assert not out.exists()


def test_reliability_recovery_cli_rejects_corrupt_results_before_output(tmp_path):
    from benchmark.scripts.run_reliability_recovery import main

    results = tmp_path / "results.csv"
    row = {
        "case_id": "c1",
        "method": "lhs+gp",
        "method_family": "static_design",
        "budget_mult": 10,
        "repeat": 0,
        "pf_true_o0": 0.2,
        "pf_pred_o0": 0.1,
        "pf_abs_err_o0": 0.1,
        "f_bal_acc_o0": 0.8,
        "f_brier_o0": 0.1,
        "status": "ok",
        "error": "",
        "nrmse": 0.1,
        "t_total_s": 1.0,
    }
    pd.DataFrame([row, dict(row, pf_pred_o0=0.15)]).to_csv(results, index=False)
    out = tmp_path / "reliability_recovery"

    rc = main([
        str(results),
        "--output-dir", str(out),
    ])

    assert rc == 2
    assert not out.exists()


def test_reliability_recovery_cli_require_headline_source_rejects_fixture_results(tmp_path):
    from benchmark.scripts.run_reliability_recovery import main

    results = tmp_path / "results.csv"
    row = {
        "case_id": "c1",
        "method": "lhs+gp",
        "method_family": "static_design",
        "budget_mult": 10,
        "repeat": 0,
        "pf_true_o0": 0.2,
        "pf_pred_o0": 0.1,
        "pf_abs_err_o0": 0.1,
        "f_bal_acc_o0": 0.8,
        "f_brier_o0": 0.1,
        "status": "ok",
        "error": "",
        "nrmse": 0.1,
        "t_total_s": 1.0,
    }
    pd.DataFrame([row]).to_csv(results, index=False)
    out = tmp_path / "reliability_recovery"

    rc = main([
        str(results),
        "--output-dir", str(out),
        "--require-headline-source",
    ])

    assert rc == 2
    assert not out.exists()


# ─── Reliability ranking ─────────────────────────────────────────────────
def test_ranking_lower_is_better_for_abs_err():
    df = _make_synthetic_results(
        n_cases=6, n_repeats=4,
        methods=("low_err", "high_err"),
        families=("static_design", "static_design"),
        bias_per_method=(0.0, 0.20),  # high_err has much larger |bias|
        seed=42,
    )
    long = extract_long_signed_bias(df)
    rk = reliability_ranking(long, metric="pf_abs_err")
    rk_sorted = rk.sort_values("mean_rank")
    assert rk_sorted.iloc[0]["method"] == "low_err"


def test_ranking_higher_is_better_for_bal_acc():
    """For bal_acc, the ranking helper should treat high values as good."""
    rng = np.random.default_rng(0)
    rows = []
    for c in range(5):
        for r in range(4):
            for method, mu in [("good", 0.85), ("poor", 0.55)]:
                rows.append({
                    "case_id": f"case_{c}",
                    "method": method,
                    "method_family": "static_design",
                    "budget_mult": 10,
                    "repeat": r,
                    "pf_true_o0": 0.2, "pf_pred_o0": 0.2,
                    "pf_abs_err_o0": 0.0,
                    "f_bal_acc_o0": float(np.clip(rng.normal(mu, 0.02), 0, 1)),
                })
    df = pd.DataFrame(rows)
    long = extract_long_signed_bias(df)
    rk = reliability_ranking(long, metric="f_bal_acc")
    assert rk.iloc[0]["method"] == "good", (
        f"high-bal-acc method should rank first; got {rk.iloc[0]['method']}"
    )


# ─── Family bias ─────────────────────────────────────────────────────────
def test_family_bias_groups_by_family():
    df = _make_synthetic_results(
        n_cases=4, n_repeats=6,
        methods=("gp_a", "gp_b", "rf_a"),
        families=("gp", "gp", "rf"),
        bias_per_method=(-0.10, -0.10, +0.02),
        seed=1,
    )
    long = extract_long_signed_bias(df)
    fam = signed_bias_by_family(long)
    assert {"gp", "rf"} <= set(fam["method_family"])
    gp_row = fam[fam["method_family"] == "gp"].iloc[0]
    rf_row = fam[fam["method_family"] == "rf"].iloc[0]
    assert gp_row["signed_bias_mean"] < rf_row["signed_bias_mean"]
    assert gp_row["pvalue_one_sided_neg"] < 0.1


# ─── Top-level wrapper ───────────────────────────────────────────────────
def test_recovery_report_returns_all_keys():
    df = _make_synthetic_results(n_cases=3, n_repeats=6)
    out = reliability_recovery_report(df)
    for k in ("long", "per_method", "ranking_abs_err",
              "ranking_bal_acc", "family_bias"):
        assert k in out
        assert isinstance(out[k], pd.DataFrame)
    assert not out["per_method"].empty


def test_reliability_ci_sensitivity_reports_cluster_iid_width_ratio():
    rng = np.random.default_rng(13)
    per_case_bias = rng.normal(0.10, 0.10, size=6)
    rows = []
    for c, mu in enumerate(per_case_bias):
        pf_true = float(rng.uniform(0.05, 0.4))
        for r in range(15):
            noise = float(rng.normal(0.0, 0.002))
            pf_pred = float(np.clip(pf_true + mu + noise, 0.0, 1.0))
            rows.append({
                "case_id": f"case_{c:02d}",
                "method": "only_method",
                "method_family": "static_design",
                "budget_mult": 10,
                "repeat": r,
                "pf_true_o0": pf_true,
                "pf_pred_o0": pf_pred,
                "pf_abs_err_o0": abs(pf_pred - pf_true),
                "status": "ok",
            })
    long = extract_long_signed_bias(pd.DataFrame(rows))

    sens = reliability_ci_sensitivity(
        long,
        budget_mult=10,
        primary_cluster_by="case_id",
        n_boot=1000,
        seed=0,
    )

    assert list(sens["method"]) == ["only_method"]
    assert sens.iloc[0]["primary_cluster_by"] == "case_id"
    assert sens.iloc[0]["sensitivity_analysis"] == "iid_bootstrap"
    assert sens.iloc[0]["signed_bias_ci_width_ratio_cluster_over_iid"] > 1.5


# ─── Cluster bootstrap (regression-test for the 30-rep pipeline) ─────────
def test_cluster_bootstrap_widens_ci_when_within_case_correlation_is_strong():
    r"""Cluster resampling should produce **wider** CIs than i.i.d.
    resampling when observations within a cluster are positively
    correlated.  This is the central statistical reason D3 uses
    cluster_by='case_id' for the Nature-grade sweep.

    Setup: build a 5-case × 20-repeat × 1-method dataset in which the
    signed bias is drawn as a single per-case value plus a tiny
    i.i.d.\ noise.  So within-case variance is near zero but
    between-case variance is large.  Naïve i.i.d.\ bootstrap sees
    100 points with small s.e.; cluster bootstrap sees effectively 5
    independent clusters with much larger s.e.
    """
    rng = np.random.default_rng(42)
    per_case_bias = rng.normal(0.10, 0.08, size=5)   # 5 cluster means
    rows = []
    for c, mu in enumerate(per_case_bias):
        pf_true = float(rng.uniform(0.05, 0.4))
        for r in range(20):
            noise = float(rng.normal(0.0, 0.002))   # tiny within-case var
            pf_pred = float(np.clip(pf_true + mu + noise, 0.0, 1.0))
            rows.append({
                "case_id": f"case_{c:02d}",
                "method": "only_method",
                "method_family": "static_design",
                "budget_mult": 10,
                "repeat": r,
                "pf_true_o0": pf_true,
                "pf_pred_o0": pf_pred,
                "pf_abs_err_o0": abs(pf_pred - pf_true),
                "status": "ok",
            })
    df = pd.DataFrame(rows)
    long = extract_long_signed_bias(df)
    s_iid = per_method_reliability_summary(long, n_boot=1000, seed=0)
    s_clu = per_method_reliability_summary(
        long, n_boot=1000, seed=0, cluster_by="case_id",
    )
    w_iid = s_iid["signed_bias_ci_hi"].iloc[0] - s_iid["signed_bias_ci_lo"].iloc[0]
    w_clu = s_clu["signed_bias_ci_hi"].iloc[0] - s_clu["signed_bias_ci_lo"].iloc[0]
    assert w_clu > 2.0 * w_iid, (
        f"cluster CI should be much wider than i.i.d. CI when ICC is high; "
        f"got iid={w_iid:.4f}, clu={w_clu:.4f}"
    )


def test_cluster_bootstrap_matches_iid_when_clusters_singletons():
    """Degenerate case: if every observation is its own cluster, the
    cluster bootstrap reduces to the i.i.d. bootstrap.  CIs should agree
    to within bootstrap noise.
    """
    df = _make_synthetic_results(n_cases=5, n_repeats=6, seed=3)
    long = extract_long_signed_bias(df)
    # Make each row its own "cluster" by inventing a unique column
    long = long.reset_index(drop=True)
    long["row_id"] = long.index.astype(str)
    s_iid = per_method_reliability_summary(long, n_boot=2000, seed=1)
    s_clu = per_method_reliability_summary(
        long, n_boot=2000, seed=1, cluster_by="row_id",
    )
    for method in s_iid["method"]:
        iid_row = s_iid[s_iid["method"] == method].iloc[0]
        clu_row = s_clu[s_clu["method"] == method].iloc[0]
        w_iid = iid_row["signed_bias_ci_hi"] - iid_row["signed_bias_ci_lo"]
        w_clu = clu_row["signed_bias_ci_hi"] - clu_row["signed_bias_ci_lo"]
        assert 0.5 * w_iid <= w_clu <= 2.0 * w_iid, (
            f"singleton-cluster bootstrap should match i.i.d.; "
            f"method={method} iid={w_iid:.4f} clu={w_clu:.4f}"
        )


def test_cluster_bootstrap_cli_flag_is_wired():
    """Regression test: the ``cluster_by`` argument in
    :func:`reliability_recovery_report` reaches
    :func:`per_method_reliability_summary` and actually changes the
    downstream CIs when the within-cluster correlation is non-trivial.
    """
    # Build a high-ICC dataset: per-case mean bias is highly variable,
    # within-case noise is tiny.  Cluster CI must be substantially
    # wider than i.i.d. CI on this data.
    rng = np.random.default_rng(11)
    per_case_bias = rng.normal(0.10, 0.12, size=6)
    rows = []
    for c, mu in enumerate(per_case_bias):
        pf_true = float(rng.uniform(0.05, 0.4))
        for r in range(15):
            noise = float(rng.normal(0.0, 0.003))
            pf_pred = float(np.clip(pf_true + mu + noise, 0.0, 1.0))
            rows.append({
                "case_id": f"case_{c:02d}",
                "method": "only_method",
                "method_family": "static_design",
                "budget_mult": 10,
                "repeat": r,
                "pf_true_o0": pf_true,
                "pf_pred_o0": pf_pred,
                "pf_abs_err_o0": abs(pf_pred - pf_true),
                "status": "ok",
            })
    df = pd.DataFrame(rows)
    out_iid = reliability_recovery_report(df, cluster_by=None)
    out_clu = reliability_recovery_report(df, cluster_by="case_id")
    w_iid = (
        out_iid["per_method"]["signed_bias_ci_hi"].iloc[0]
        - out_iid["per_method"]["signed_bias_ci_lo"].iloc[0]
    )
    w_clu = (
        out_clu["per_method"]["signed_bias_ci_hi"].iloc[0]
        - out_clu["per_method"]["signed_bias_ci_lo"].iloc[0]
    )
    assert w_clu > 1.5 * w_iid, (
        f"CLI flag 'cluster_by=case_id' should substantially widen CI on "
        f"high-ICC data; iid={w_iid:.4f} clu={w_clu:.4f}"
    )


def test_cluster_bootstrap_raises_on_length_mismatch():
    """Direct unit test on the low-level helper."""
    from benchmark.reliability_recovery import _bootstrap_ci

    values = np.array([0.1, 0.2, 0.3, 0.4])
    bad_clusters = np.array([0, 0, 1])   # len 3 vs len 4
    with pytest.raises(ValueError, match="same length"):
        _bootstrap_ci(values, clusters=bad_clusters)
