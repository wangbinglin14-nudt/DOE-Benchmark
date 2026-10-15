"""Smoke tests for the extension modules: surrogates, stratified, importance, reliability."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import doe_testbed as db
from benchmark.importance import (
    compute_importance_recovery,
    true_sobol_ST,
)
from benchmark.methods import (
    default_method_pool,
    extended_method_pool,
    full_method_pool,
)
from benchmark.reliability import get_failure_thresholds, reliability_metrics
from benchmark.stratified import (
    annotate_results,
    bootstrap_rank_ci,
    bootstrap_rank_shift_ci,
    stratified_ranks,
    winner_table,
)
from benchmark.surrogates import (
    GPRBFSurrogate,
    GradientBoostingSurrogate,
    KNNSurrogate,
    PolyChaosSurrogate,
    SVRSurrogate,
)

warnings.filterwarnings("ignore")


# ─── New surrogates ──────────────────────────────────────────────────────
def test_new_surrogates_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.random((30, 4))
    Y = rng.random((30, 2))
    Xt = rng.random((10, 4))
    for cls in [GPRBFSurrogate, SVRSurrogate, KNNSurrogate,
                GradientBoostingSurrogate, PolyChaosSurrogate]:
        s = cls()
        s.fit(X, Y)
        mu, sd = s.predict(Xt)
        assert mu.shape == (10, 2), f"{cls.__name__} mean shape wrong"
        if sd is not None:
            assert sd.shape == (10, 2), f"{cls.__name__} std shape wrong"


def test_method_pools_consistent():
    a = default_method_pool()
    b = extended_method_pool()
    c = full_method_pool()
    assert len(a) == 10
    assert len(b) == 15
    assert len(c) == 18
    # All names unique
    for pool in [a, b, c]:
        names = [m.name for m in pool]
        assert len(set(names)) == len(names)


# ─── Stratified analysis ─────────────────────────────────────────────────
def _toy_results_df():
    """Construct a tiny benchmark-style DataFrame for unit tests."""
    rows = []
    for case_id in ["cantilever_beam_5d", "fin_heat_sink_4d", "wing_vlm_6d"]:
        meta = db.get_case_meta(case_id)
        for method in ["lhs+gp", "random+gp", "lhs_opt+rbf"]:
            for budget in [10, 20]:
                for rep in range(3):
                    rows.append({
                        "case_id": case_id,
                        "domain": meta.domain,
                        "dim": meta.dim_input,
                        "method": method,
                        "budget_mult": budget,
                        "repeat": rep,
                        "nrmse": np.random.default_rng(hash((case_id, method, budget, rep)) & 0xffff).random(),
                        "rmse": np.random.default_rng().random(),
                        "status": "ok",
                    })
    return pd.DataFrame(rows)


def test_annotate_adds_attribute_columns():
    df = _toy_results_df()
    out = annotate_results(df)
    for col in ["domain_attr", "dim_bucket", "topology", "scenarios",
                "has_failure_boundary", "cost_level"]:
        assert col in out.columns


def test_stratified_ranks_by_domain():
    df = _toy_results_df()
    s = stratified_ranks(df, stratify_by="domain", budget_value=20)
    assert "stratum" in s.columns
    assert "n_cases" in s.columns
    # All methods should appear as columns
    for m in ["lhs+gp", "random+gp", "lhs_opt+rbf"]:
        assert m in s.columns


def test_stratified_ranks_by_vs_scenario():
    df = _toy_results_df()
    s = stratified_ranks(df, stratify_by="vs_scenario", budget_value=20)
    # At least one VS row should exist (these cases have VS tags)
    assert len(s) >= 1


def test_winner_table_one_per_case():
    df = _toy_results_df()
    w = winner_table(df, budget_value=20)
    assert len(w) == 3  # 3 unique cases
    assert set(w.columns) == {"case_id", "winner", "best_nrmse"}


def test_bootstrap_rank_ci_shape():
    df = _toy_results_df()
    out = bootstrap_rank_ci(df, budget_value=20, n_boot=50)
    assert "method" in out.columns
    assert "ci_low" in out.columns
    assert "ci_high" in out.columns
    assert (out["ci_high"] >= out["ci_low"]).all()


def test_bootstrap_rank_shift_ci_shape():
    df = _toy_results_df()
    out = bootstrap_rank_shift_ci(df, df, budget_value=20, n_boot=50)
    assert "shift" in out.columns
    assert "shift_ci_low" in out.columns
    assert "shift_ci_high" in out.columns


# ─── Reliability metrics ─────────────────────────────────────────────────
def test_failure_thresholds_lookup():
    # cantilever has 2 failure thresholds
    th = get_failure_thresholds("cantilever_beam_5d")
    assert len(th) >= 1
    # wing_vlm_6d has none (no failure boundary)
    th_none = get_failure_thresholds("wing_vlm_6d")
    assert isinstance(th_none, list)


def test_reliability_metrics_perfect_prediction():
    case = db.get_case("cantilever_beam_5d")
    X = case.sample(200, method="sobol", seed=0)
    Y = case.evaluate(X)
    # Y_pred = Y → perfect → pf_abs_err = 0
    out = reliability_metrics("cantilever_beam_5d", Y, Y, None)
    for k, v in out.items():
        if k.startswith("pf_abs_err_o"):
            assert v < 1e-10, f"{k}={v} should be 0"


def test_reliability_metrics_no_threshold_returns_empty():
    case = db.get_case("wing_vlm_6d")
    X = case.sample(100, method="sobol", seed=0)
    Y = case.evaluate(X)
    out = reliability_metrics("wing_vlm_6d", Y, Y, None)
    assert out == {}, f"expected empty dict, got {out}"


# ─── Variable-importance recovery ────────────────────────────────────────
def test_true_sobol_ST_lookup():
    st = true_sobol_ST("cantilever_beam_5d", output_idx=0)
    if st is not None:
        assert st.shape == (5,)
        assert (st >= 0).all()
        assert (st <= 1.5).all()  # ST can mildly exceed 1 from noise


def test_estimate_sobol_via_surrogate_perfect_recovery():
    """For a perfectly-fitting surrogate, Sobol indices should approximately match."""
    from benchmark.surrogates import GPSurrogate
    case = db.get_case("cantilever_beam_5d")
    X_train = case.sample(200, method="sobol", seed=0)
    Y_train = case.evaluate(X_train)

    s = GPSurrogate()
    s.fit(X_train, Y_train)

    out = compute_importance_recovery(s, "cantilever_beam_5d", n_base=256, output_idx=0)
    # With 200 training points on a smooth case, recovery should be reasonably good
    assert out["mae_st"] < 0.5  # absolute Sobol estimate error
    # Some decent agreement
    assert out["top_k_overlap"] >= 0.4
