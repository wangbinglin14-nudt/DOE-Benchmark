"""Tests for the dim_scaling module."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from benchmark.dim_scaling import (
    alpha_vs_dim,
    default_dim_bands,
    nrmse_vs_dim,
    rank_by_dimension_band,
)

warnings.filterwarnings("ignore")


@pytest.fixture
def synthetic_results() -> pd.DataFrame:
    """6 cases x 3 methods x 3 budgets x 2 reps; method A wins on
    low-d, method B wins on high-d, method C is consistently worst."""
    rng = np.random.default_rng(0)
    rows = []
    cases = [
        ("c1", 2), ("c2", 3),                       # low d
        ("c3", 5), ("c4", 6),                       # mid d
        ("c5", 9), ("c6", 12),                      # high d
    ]
    for case_id, d in cases:
        for budget in [5, 10, 20]:
            for method, base in [("A", 0.1),
                                  ("B", 0.2),
                                  ("C", 0.5)]:
                for rep in range(2):
                    if method == "A":
                        nrmse = base + 0.02 * d + rng.normal(0, 0.005)
                        # Method A degrades fast with d
                    elif method == "B":
                        nrmse = base + 0.005 * d + rng.normal(0, 0.005)
                        # Method B is dimension-tolerant
                    else:
                        nrmse = base + 0.04 * d + rng.normal(0, 0.005)
                        # Method C is bad and degrades
                    nrmse *= (5 / budget) ** 0.5
                    rows.append({
                        "case_id": case_id, "method": method,
                        "dim": d, "budget_mult": budget,
                        "n_train": budget * d,
                        "repeat": rep, "nrmse": max(nrmse, 1e-4),
                        "status": "ok",
                    })
    return pd.DataFrame(rows)


def test_default_dim_bands(synthetic_results):
    bands = default_dim_bands(synthetic_results)
    assert len(bands) >= 2
    # Bands should cover the full range
    lo = min(synthetic_results["dim"])
    hi = max(synthetic_results["dim"])
    band_lo = min(b[0] for b in bands.values())
    band_hi = max(b[1] for b in bands.values())
    assert band_lo == lo
    assert band_hi == hi


def test_rank_by_dimension_band_detects_flip(synthetic_results):
    """Method A wins low-d, B wins high-d (by construction)."""
    bands = {"low": (2, 4), "high": (8, 12)}
    rk = rank_by_dimension_band(synthetic_results, bands=bands,
                                 budget_value=20)
    assert "low" in rk.columns
    assert "high" in rk.columns
    # In low-d, A should be better (lower rank)
    assert rk.loc["A", "low"] < rk.loc["B", "low"]
    # In high-d, B should be better
    assert rk.loc["B", "high"] < rk.loc["A", "high"]
    # The 'range' column should reflect the flip
    assert rk.loc["A", "range"] > 0.5


def test_nrmse_vs_dim_slopes(synthetic_results):
    """Method A degrades faster with d than method B (by construction)."""
    out = nrmse_vs_dim(synthetic_results, budget_value=20)
    assert not out.empty
    slope_A = float(out.loc[out["method"] == "A", "log_slope"].iloc[0])
    slope_B = float(out.loc[out["method"] == "B", "log_slope"].iloc[0])
    # A should have larger positive slope (worse with dim) than B
    assert slope_A > slope_B
    # All slopes should be positive (NRMSE grows with d)
    assert (out["log_slope"] > 0).all()


def test_alpha_vs_dim_returns_dataframe(synthetic_results):
    out = alpha_vs_dim(synthetic_results)
    # With 3 budgets per case, we should get α fits
    assert isinstance(out, pd.DataFrame)
    assert "method" in out.columns
    assert "spearman_rho_alpha_vs_dim" in out.columns
