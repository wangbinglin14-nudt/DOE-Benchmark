"""Unit tests for the rigour module."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from benchmark.rigour import (
    bootstrap_metric_ci,
    cliffs_delta,
    cliffs_delta_ci_matrix,
    dominance_score,
    failure_rate_per_method,
    fit_quality_flags,
    pairwise_signed_rank_test,
    pairwise_win_rate_matrix,
    pareto_front,
    per_output_nrmse,
    per_output_rank_shifts,
    per_output_summary,
)

warnings.filterwarnings("ignore")


def _toy_df():
    rng = np.random.default_rng(0)
    rows = []
    for case in ["A", "B", "C", "D"]:
        for method in ["m1", "m2", "m3"]:
            for rep in range(5):
                # Different methods have different mean error
                base = {"m1": 0.2, "m2": 0.3, "m3": 0.6}[method]
                rows.append({
                    "case_id": case,
                    "method": method,
                    "budget_mult": 20,
                    "repeat": rep,
                    "nrmse": base + 0.05 * rng.standard_normal(),
                    "rmse": rng.random(),
                    "r2": 0.8,
                    "picp_95": 0.95,
                    "t_total_s": {"m1": 1.0, "m2": 0.5, "m3": 0.1}[method],
                    "status": "ok",
                })
    return pd.DataFrame(rows)


# ─── Cliff's delta ────────────────────────────────────────────────────────
def test_cliffs_delta_known_dominance():
    a = np.array([1.0, 2.0, 3.0])  # smaller = "better"
    b = np.array([10.0, 11.0, 12.0])
    # All a < b → P(a<b)=1, P(a>b)=0 → delta = 1 (a fully dominates b)
    assert cliffs_delta(a, b) == 1.0


def test_cliffs_delta_identical():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    delta = cliffs_delta(a, b)
    # All ties → delta = 0
    assert abs(delta) < 1e-12


def test_cliffs_delta_with_nan():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([10.0, 11.0])
    # Only finite a values used: 1, 3 vs 10, 11 → all a<b → delta=1
    assert cliffs_delta(a, b) == 1.0


# ─── Cliff's delta with bootstrap CI ─────────────────────────────────────
def test_cliffs_delta_ci_matrix_returns_expected_keys():
    rng = np.random.default_rng(0)
    a_vals = rng.normal(0.10, 0.02, 24)  # A is "better"
    b_vals = rng.normal(0.30, 0.03, 24)  # B is worse
    df = pd.DataFrame({
        "case_id": list(np.repeat([f"c{i}" for i in range(8)], 3)) * 2,
        "method": ["A"] * 24 + ["B"] * 24,
        "budget_mult": 10,
        "repeat": list(np.tile(np.arange(3), 8)) * 2,
        "nrmse": np.concatenate([a_vals, b_vals]),
        "status": "ok",
    })
    out = cliffs_delta_ci_matrix(df, n_boot=200, seed=0)
    for key in ("delta", "ci_lo", "ci_hi", "non_negligible", "magnitude"):
        assert key in out
    # A vs B: A has lower NRMSE → cliffs_delta(A, B) ≈ +1 (A dominates B)
    delta_AB = out["delta"].loc["A", "B"]
    assert delta_AB > 0.5, f"expected A to dominate B, got {delta_AB}"
    # CI should exclude 0
    assert out["ci_lo"].loc["A", "B"] > 0
    # Significance flag set
    assert bool(out["non_negligible"].loc["A", "B"]) is True
    # Magnitude is at least "small" given the strong effect
    assert out["magnitude"].loc["A", "B"] in {"small", "medium", "large"}


def test_cliffs_delta_ci_matrix_diagonal_is_zero():
    df = pd.DataFrame({
        "case_id": ["c1", "c2", "c1", "c2"],
        "method": ["A", "A", "B", "B"],
        "budget_mult": 10,
        "nrmse": [0.1, 0.2, 0.15, 0.25],
        "status": "ok",
    })
    out = cliffs_delta_ci_matrix(df, n_boot=50, seed=0)
    assert out["delta"].loc["A", "A"] == 0.0
    assert out["delta"].loc["B", "B"] == 0.0
    # Diagonal magnitude is the sentinel "self"
    assert out["magnitude"].loc["A", "A"] == "self"


# ─── Pairwise win-rate ────────────────────────────────────────────────────
def test_pairwise_win_rate_matrix_shape_and_diagonal():
    df = _toy_df()
    win = pairwise_win_rate_matrix(df, "nrmse", budget_value=20)
    assert win.shape == (3, 3)
    # Diagonal must be 0.5
    for m in win.index:
        assert win.loc[m, m] == 0.5


def test_pairwise_win_rate_orderings():
    df = _toy_df()
    win = pairwise_win_rate_matrix(df, "nrmse", budget_value=20)
    # m1 (best) should beat m2 (mid) and m3 (worst) on most cases
    assert win.loc["m1", "m2"] > 0.5
    assert win.loc["m1", "m3"] > 0.5
    assert win.loc["m2", "m3"] > 0.5
    # Symmetry: win[A,B] + win[B,A] = 1
    for a in ["m1", "m2", "m3"]:
        for b in ["m1", "m2", "m3"]:
            if a != b:
                assert abs(win.loc[a, b] + win.loc[b, a] - 1.0) < 1e-9


def test_dominance_score_orders_methods():
    df = _toy_df()
    win = pairwise_win_rate_matrix(df, "nrmse", budget_value=20)
    dom = dominance_score(win)
    # Best method should rank first
    assert dom.index[0] == "m1"
    assert dom.index[-1] == "m3"


# ─── Pareto ────────────────────────────────────────────────────────────────
def test_pareto_front_includes_extremes():
    df = _toy_df()
    pf = pareto_front(df, "nrmse", "t_total_s", budget_value=20)
    # Best accuracy and cheapest must both be on the front
    methods = pf[pf["on_pareto"]]["method"].tolist()
    # m3 is cheapest; m1 is most accurate
    assert "m1" in methods or "m2" in methods  # most accurate
    assert "m3" in methods  # cheapest


def test_pareto_front_dominance_logic():
    df = _toy_df()
    pf = pareto_front(df, "nrmse", "t_total_s", budget_value=20)
    front = pf[pf["on_pareto"]].sort_values("mean_cost")
    # Along the Pareto front, increasing cost ⇒ strictly decreasing metric
    metrics = front["mean_metric"].tolist()
    assert metrics == sorted(metrics, reverse=True)


# ─── Fit-quality flags ────────────────────────────────────────────────────
def test_fit_quality_flags_clean_runs():
    df = _toy_df()
    flags = fit_quality_flags(df)
    assert (flags["flags"] == "ok").all()


def test_fit_quality_flags_blowup_detection():
    df = _toy_df()
    # Inject a blowup
    df.loc[df["method"] == "m3", "nrmse"] = 7.0
    flags = fit_quality_flags(df)
    flagged = flags[flags["flags"].str.contains("blowup")]
    assert len(flagged) > 0
    # Only m3 should be flagged
    assert (flagged["method"] == "m3").all()


# ─── Failure rate ─────────────────────────────────────────────────────────
def test_failure_rate_per_method_clean():
    df = _toy_df()
    fr = failure_rate_per_method(df)
    assert (fr["total_failure_rate"] == 0).all()


def test_failure_rate_per_method_with_blowup():
    df = _toy_df()
    df.loc[df["method"] == "m3", "nrmse"] = 8.0
    fr = failure_rate_per_method(df)
    m3_rate = fr[fr["method"] == "m3"]["total_failure_rate"].values[0]
    assert m3_rate == 1.0
    m1_rate = fr[fr["method"] == "m1"]["total_failure_rate"].values[0]
    assert m1_rate == 0.0


# ─── Test-set MC error ────────────────────────────────────────────────────
def test_testset_mc_error_decreases_with_n():
    rng = np.random.default_rng(0)
    # Generate synthetic data
    Y_true = rng.random((100, 1))
    Y_pred = Y_true + 0.1 * rng.standard_normal((100, 1))
    out_small = bootstrap_metric_ci(Y_true, Y_pred, n_boot=200, seed=0)

    Y_true_big = rng.random((10000, 1))
    Y_pred_big = Y_true_big + 0.1 * rng.standard_normal((10000, 1))
    out_big = bootstrap_metric_ci(Y_true_big, Y_pred_big, n_boot=200, seed=0)

    # CI width should shrink with larger n_test
    assert out_big["nrmse_ci_width"] < out_small["nrmse_ci_width"]


# ─── Per-output NRMSE ─────────────────────────────────────────────────────
def test_per_output_nrmse_shape():
    rng = np.random.default_rng(0)
    Y_true = rng.random((100, 3))
    Y_pred = Y_true + 0.1 * rng.standard_normal((100, 3))
    nrmse = per_output_nrmse(Y_true, Y_pred)
    assert nrmse.shape == (3,)
    # Should all be small (we added small noise)
    assert (nrmse < 0.5).all()


def test_per_output_nrmse_perfect():
    Y = np.linspace(0, 1, 100).reshape(-1, 2)
    nrmse = per_output_nrmse(Y, Y.copy())
    assert (nrmse < 1e-10).all()


# ─── Per-output summary / rank shifts ────────────────────────────────────
def _toy_per_output_df():
    """A toy DF where method m1 is great on out0 but terrible on out1."""
    rng = np.random.default_rng(0)
    rows = []
    for case in ["A", "B", "C", "D"]:
        for method in ["m1", "m2"]:
            for rep in range(3):
                if method == "m1":
                    nrmse_out0 = 0.05 + 0.01 * rng.standard_normal()
                    nrmse_out1 = 0.80 + 0.01 * rng.standard_normal()
                else:
                    nrmse_out0 = 0.30 + 0.01 * rng.standard_normal()
                    nrmse_out1 = 0.30 + 0.01 * rng.standard_normal()
                rows.append({
                    "case_id": case, "method": method,
                    "budget_mult": 20, "repeat": rep, "status": "ok",
                    "nrmse": (nrmse_out0 + nrmse_out1) / 2,
                    "nrmse_out0": nrmse_out0,
                    "nrmse_out1": nrmse_out1,
                })
    return pd.DataFrame(rows)


def test_per_output_summary_detects_imbalance():
    df = _toy_per_output_df()
    s = per_output_summary(df, budget_value=20)
    assert "mean_worst" in s.columns
    # m1 has imbalance (0.05 / 0.80) → larger spread than m2 (~0)
    m1 = s[s["method"] == "m1"].iloc[0]
    m2 = s[s["method"] == "m2"].iloc[0]
    assert m1["mean_spread"] > m2["mean_spread"]
    assert m1["mean_worst"] > m2["mean_worst"]  # m1 worse on worst output


def test_per_output_rank_shifts_detect_swap():
    df = _toy_per_output_df()
    rs = per_output_rank_shifts(df, budget_value=20)
    # At least one method must show a positive rank_shift (worse on its
    # worst output than on the average).
    assert rs["rank_shift"].max() > 0
    # Both methods' rank_max should be >= rank_avg (by definition)
    assert ((rs["rank_max"] - rs["rank_avg"]) >= -1e-9).all()


def test_metrics_includes_worst_output_aggregations():
    """compute_all_metrics should expose nrmse_worst / nrmse_best / nrmse_spread."""
    from benchmark.metrics import compute_all_metrics
    rng = np.random.default_rng(0)
    Y_true = rng.random((100, 3))
    Y_pred = Y_true.copy()
    # Inject error into output 1 only
    Y_pred[:, 1] += 0.5 * rng.standard_normal(100)
    m = compute_all_metrics(Y_true, Y_pred)
    assert "nrmse_worst" in m
    assert "nrmse_best" in m
    assert "nrmse_spread" in m
    # Worst output (output 1, with injected error) > best output
    assert m["nrmse_worst"] > m["nrmse_best"]
    assert m["nrmse_spread"] > 0
    # nrmse_worst should equal max of per-output nrmse
    out_vals = [m["nrmse_out0"], m["nrmse_out1"], m["nrmse_out2"]]
    assert abs(m["nrmse_worst"] - max(out_vals)) < 1e-9


def test_nrmse_relation_uses_sample_standard_deviation_convention():
    """NRMSE and R2 agree under the documented ddof=1 convention."""
    from benchmark.metrics import nrmse, r2

    y_true = np.arange(1.0, 6.0).reshape(-1, 1)
    y_pred = y_true + np.array([[0.0], [1.0], [0.0], [-1.0], [0.0]])
    observed_r2 = float(r2(y_true, y_pred))
    observed_nrmse = float(nrmse(y_true, y_pred))
    expected = np.sqrt((len(y_true) - 1) / len(y_true) * (1.0 - observed_r2))
    assert np.isclose(observed_nrmse, expected)


# ─── Pairwise signed-rank test (Holm-Bonferroni) ─────────────────────────
def test_pairwise_signed_rank_returns_expected_shape():
    df = _toy_df()
    out = pairwise_signed_rank_test(df, metric="nrmse", budget_value=20)
    assert "p_raw" in out
    assert "p_adjusted" in out
    assert "significant" in out
    assert "summary" in out
    methods = ["m1", "m2", "m3"]
    assert list(out["p_raw"].columns) == methods
    assert list(out["p_adjusted"].index) == methods
    # diagonal is zero
    for m in methods:
        assert out["p_raw"].loc[m, m] in (0, 1)
        assert out["p_adjusted"].loc[m, m] == 0
    # Symmetry
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            assert out["p_raw"].loc[m1, m2] == out["p_raw"].loc[m2, m1]
            assert out["p_adjusted"].loc[m1, m2] == out["p_adjusted"].loc[m2, m1]


def test_pairwise_signed_rank_holm_correction_increases_p():
    """Holm-corrected p-values must be >= raw p-values."""
    df = _toy_df()
    out = pairwise_signed_rank_test(df, metric="nrmse", budget_value=20)
    summary = out["summary"]
    assert (summary["p_holm"] >= summary["p_raw"] - 1e-9).all()
    # Holm p-values must be monotonically non-decreasing
    # (they are the running max of corrected raw p)
    sorted_summary = summary.sort_values("p_raw")
    diffs = np.diff(sorted_summary["p_holm"].to_numpy())
    assert (diffs >= -1e-9).all()


def test_pairwise_signed_rank_detects_strong_difference():
    """In _toy_df, m1, m2, m3 are well-separated; the (m1, m3) pair
    should have the *largest* mean_diff in absolute value, even if
    the smallest-p tie on a 4-case sample is shared."""
    df = _toy_df()
    out = pairwise_signed_rank_test(df, metric="nrmse", budget_value=20,
                                     alpha=0.05)
    summary = out["summary"]
    # Find the (m1, m3) row
    m13 = summary[((summary["method_A"] == "m1") & (summary["method_B"] == "m3"))
                  | ((summary["method_A"] == "m3") & (summary["method_B"] == "m1"))]
    m12 = summary[((summary["method_A"] == "m1") & (summary["method_B"] == "m2"))
                  | ((summary["method_A"] == "m2") & (summary["method_B"] == "m1"))]
    assert not m13.empty
    assert not m12.empty
    # |mean_diff| for (m1, m3) > (m1, m2)
    assert abs(float(m13["mean_diff"].iloc[0])) > abs(float(m12["mean_diff"].iloc[0]))
