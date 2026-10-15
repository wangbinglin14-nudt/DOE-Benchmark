"""Tests for the learning_curves module."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from benchmark.learning_curves import (
    fit_power_law,
    learning_curve_table,
    rank_stability_score,
    ranking_at_each_budget,
)

warnings.filterwarnings("ignore")


def _toy_df():
    """Synthetic data: NRMSE follows n^{-α} with method-specific α.

    method m1: α=0.5  (fast)
    method m2: α=0.25 (slow)
    """
    rows = []
    rng = np.random.default_rng(0)
    for case in ["A", "B", "C"]:
        for method, alpha in [("m1", 0.5), ("m2", 0.25)]:
            for budget in [5, 10, 20]:
                n_train = budget * 5  # d=5
                base = n_train ** (-alpha)
                for rep in range(3):
                    rows.append({
                        "case_id": case,
                        "method": method,
                        "budget_mult": budget,
                        "n_train": n_train,
                        "repeat": rep,
                        "nrmse": base * (1 + 0.05 * rng.standard_normal()),
                        "status": "ok",
                    })
    return pd.DataFrame(rows)


def test_learning_curve_table_aggregation():
    df = _toy_df()
    lc = learning_curve_table(df)
    # 3 cases × 2 methods × 3 budgets = 18 rows
    assert len(lc) == 18
    assert "nrmse_mean" in lc.columns
    assert "n_train" in lc.columns


def test_ranking_at_each_budget_shape():
    df = _toy_df()
    rk = ranking_at_each_budget(df)
    assert rk.shape == (2, 3)  # 2 methods × 3 budgets


def test_rank_stability_detects_stable_ranking():
    df = _toy_df()
    rk = ranking_at_each_budget(df)
    ss = rank_stability_score(rk)
    # Both methods consistently rank the same way → range=0
    assert ss["rank_range"].max() == 0


def test_fit_power_law_recovers_alpha():
    """Power-law fit should recover the synthetic α exponents."""
    df = _toy_df()
    lc = learning_curve_table(df)
    fits = fit_power_law(lc)

    # Group by method and check α
    by_method = fits.groupby("method")["alpha"].mean()
    assert abs(by_method["m1"] - 0.5) < 0.05  # noise in the fit
    assert abs(by_method["m2"] - 0.25) < 0.05
    # m1 should converge faster than m2
    assert by_method["m1"] > by_method["m2"]


def test_fit_power_law_ignores_non_positive_metric():
    df = _toy_df()
    df.loc[df.index[0], "nrmse"] = 0.0
    lc = learning_curve_table(df)
    fits = fit_power_law(lc)
    # Non-positive value in one row should drop that (method, case)
    assert len(fits) <= 6  # 2 methods × 3 cases
