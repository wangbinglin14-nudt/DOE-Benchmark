"""
benchmark.learning_curves
==========================
Learning-curve analysis: NRMSE as a function of training budget.

This module answers asymptotic questions that single-budget tables
cannot:

  - Does method M improve at the rate n^{-α} expected from theory?
  - Where does each method *plateau*?
  - Does the cross-method ranking change as n grows?  (No-free-lunch
    detector.)
  - Does any method exhibit *double-descent* or non-monotone curves?

Public API
----------
- ``learning_curve_table`` — long-form DataFrame indexed by
  (method, case, budget) with mean NRMSE and bootstrap CI per
  budget.
- ``fit_power_law`` — fit NRMSE ≈ A · n^{-α} + ε per (method, case)
  and return the exponent α.
- ``ranking_at_each_budget`` — mean rank per method at each budget;
  reveals whether the ranking is stable or budget-dependent.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from benchmark.analysis import pivot_metric, rank_methods


def learning_curve_table(
    df: pd.DataFrame,
    metric: str = "nrmse",
) -> pd.DataFrame:
    """Aggregate (method, case, budget) → mean ± std of metric.

    Returns columns: method, case_id, budget_mult, n_train,
    metric_mean, metric_median, metric_std, metric_min, metric_max.
    """
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    grp = (df.groupby(["method", "case_id", "budget_mult"])[metric]
             .agg(metric_mean="mean",
                  metric_median="median",
                  metric_std="std",
                  metric_min="min",
                  metric_max="max",
                  n_reps="count")
             .reset_index())

    # Attach mean n_train within group
    n_train_col = "n_train" if "n_train" in df.columns else "n_train_effective"
    if n_train_col in df.columns:
        n_grp = (df.groupby(["method", "case_id", "budget_mult"])[n_train_col]
                   .mean().reset_index().rename(columns={n_train_col: "n_train"}))
        grp = grp.merge(n_grp, on=["method", "case_id", "budget_mult"], how="left")

    return grp.rename(columns={
        "metric_mean": f"{metric}_mean",
        "metric_median": f"{metric}_median",
        "metric_std": f"{metric}_std",
        "metric_min": f"{metric}_min",
        "metric_max": f"{metric}_max",
    })


def ranking_at_each_budget(
    df: pd.DataFrame,
    metric: str = "nrmse",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Mean rank per method × budget.

    A method whose mean rank shifts monotonically with budget is
    revealing scaling behavior.  A method whose rank reverses at
    larger budgets contradicts the small-budget take-away.

    Returns a DataFrame indexed by method, columns are budget_mult.
    """
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    out: Dict[int, pd.Series] = {}
    for b in sorted(df["budget_mult"].unique()):
        piv = pivot_metric(df, metric, budget_value=int(b))
        ranks = rank_methods(piv, lower_is_better=lower_is_better)
        out[int(b)] = ranks
    rank_df = pd.DataFrame(out)
    rank_df.columns.name = "budget_mult"
    return rank_df


def rank_stability_score(
    rank_df: pd.DataFrame,
) -> pd.DataFrame:
    """For each method, compute rank min / max / range across budgets.

    Methods with small range are *budget-stable*; large range means
    the ranking depends on budget.
    """
    out = pd.DataFrame({
        "rank_min": rank_df.min(axis=1),
        "rank_max": rank_df.max(axis=1),
        "rank_range": rank_df.max(axis=1) - rank_df.min(axis=1),
        "rank_first": rank_df.iloc[:, 0],
        "rank_last": rank_df.iloc[:, -1],
    })
    out["rank_trend"] = out["rank_last"] - out["rank_first"]
    return out.sort_values("rank_range", ascending=False)


def fit_power_law(
    lc_df: pd.DataFrame,
    metric: str = "nrmse",
    min_points: int = 2,
) -> pd.DataFrame:
    """Fit NRMSE ≈ A · n^{-α} per (method, case).

    Uses log-log linear regression on the budget points, requires at
    least ``min_points`` distinct budgets per (method, case).

    Returns columns: method, case_id, alpha, A, R2, n_points.
    A larger α indicates faster convergence with budget.
    """
    rows = []
    if "n_train" not in lc_df.columns:
        # Fallback to budget_mult * d implicit; just use budget_mult
        lc_df = lc_df.copy()
        lc_df["n_train"] = lc_df["budget_mult"]

    metric_col = f"{metric}_mean"
    for (method, case), sub in lc_df.groupby(["method", "case_id"]):
        if len(sub) < min_points:
            continue
        x = np.log(sub["n_train"].astype(float).to_numpy())
        y_raw = sub[metric_col].astype(float).to_numpy()
        if np.any(y_raw <= 0) or np.any(~np.isfinite(y_raw)):
            continue
        y = np.log(y_raw)
        # OLS slope
        slope, intercept = np.polyfit(x, y, 1)
        # R²
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rows.append({
            "method": method,
            "case_id": case,
            "alpha": -float(slope),  # NRMSE ≈ n^{-α}, so slope = -α
            "A": float(np.exp(intercept)),
            "R2": float(r2),
            "n_points": int(len(sub)),
        })
    return pd.DataFrame(rows)


__all__ = [
    "learning_curve_table",
    "ranking_at_each_budget",
    "rank_stability_score",
    "fit_power_law",
]
