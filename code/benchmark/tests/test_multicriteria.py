"""Tests for benchmark.multicriteria."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from benchmark.multicriteria import (
    accuracy_vs_sensitivity_pareto,
    compose_method_scorecard,
    stability_vs_quality_pareto,
)

warnings.filterwarnings("ignore")


@pytest.fixture
def synthetic_inputs():
    nrmse_ranks = pd.Series(
        {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0},
        name="mean_rank",
    )
    sr_summary = pd.DataFrame({
        "method":    ["A",  "B",  "C",  "D",  "E"],
        "tau_mean":  [0.4,  0.9,  0.5,  0.6,  0.2],   # B wins on τ
        "top1_rate": [0.8,  0.95, 0.5,  0.7,  0.3],
        "flat_rate": [0.0,  0.5,  0.0,  0.05, 0.0],   # B is brittle
    })
    return nrmse_ranks, sr_summary


# ─── accuracy_vs_sensitivity_pareto ──────────────────────────────────────
def test_pareto_marks_optimal_methods(synthetic_inputs):
    nrmse_ranks, sr_summary = synthetic_inputs
    out = accuracy_vs_sensitivity_pareto(nrmse_ranks, sr_summary)
    pareto = set(out[out["on_pareto"]]["method"])
    # A: best NRMSE rank (1.0) → automatically Pareto-optimal
    assert "A" in pareto
    # B: best τ (0.9) → automatically Pareto-optimal
    assert "B" in pareto
    # E is dominated by A on both axes → NOT on Pareto
    assert "E" not in pareto


def test_pareto_dominator_assigned(synthetic_inputs):
    nrmse_ranks, sr_summary = synthetic_inputs
    out = accuracy_vs_sensitivity_pareto(nrmse_ranks, sr_summary)
    # Every non-Pareto method has a non-empty dominator
    non_pareto = out[~out["on_pareto"]]
    assert (non_pareto["dominated_by"] != "").all()


def test_pareto_returns_empty_on_disjoint():
    nrmse_ranks = pd.Series({"X": 1.0, "Y": 2.0})
    sr = pd.DataFrame({"method": ["A"], "tau_mean": [0.5]})
    out = accuracy_vs_sensitivity_pareto(nrmse_ranks, sr)
    assert out.empty


# ─── stability_vs_quality_pareto ─────────────────────────────────────────
def test_stability_pareto_excludes_brittle(synthetic_inputs):
    _, sr_summary = synthetic_inputs
    out = stability_vs_quality_pareto(sr_summary)
    pareto = set(out[out["on_pareto"]]["method"])
    # B has the highest τ but flat-rate=0.5 (brittle)
    # D has τ=0.6, flat-rate=0.05 — much more stable
    # B is still Pareto (it's *the* highest τ)
    assert "B" in pareto
    # E: τ=0.2, flat=0.0 — dominated by A (τ=0.4, flat=0.0)
    assert "E" not in pareto


# ─── compose_method_scorecard ────────────────────────────────────────────
def test_scorecard_structure(synthetic_inputs):
    nrmse_ranks, sr_summary = synthetic_inputs
    cost = pd.Series({"A": 0.1, "B": 0.5, "C": 0.2, "D": 1.0, "E": 0.05})
    sc = compose_method_scorecard(nrmse_ranks, sr_summary, cost_summary=cost)
    # Shape: 5 methods, expected columns
    assert len(sc) == 5
    assert "nrmse_mean_rank" in sc.columns
    assert "tau_mean" in sc.columns
    assert "compute_s" in sc.columns
    # Percentile columns added
    assert "tau_mean_pct" in sc.columns
    assert "compute_s_pct" in sc.columns
    # Higher τ → higher pct
    sc_sorted = sc.sort_values("tau_mean")
    assert sc_sorted["tau_mean_pct"].is_monotonic_increasing
    # Lower compute → higher pct
    sc_sorted_c = sc.sort_values("compute_s")
    assert sc_sorted_c["compute_s_pct"].is_monotonic_decreasing


def test_scorecard_handles_missing_inputs():
    nrmse_ranks = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    # Only NRMSE provided
    sc = compose_method_scorecard(nrmse_ranks)
    assert len(sc) == 3
    assert "nrmse_mean_rank" in sc.columns
    # No SR or cost columns expected
    assert "tau_mean" not in sc.columns
    assert "compute_s" not in sc.columns
