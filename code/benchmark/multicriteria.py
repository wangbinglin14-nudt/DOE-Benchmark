"""
benchmark.multicriteria
=======================
Multi-criteria recommendation analyses.

The headline benchmark ranks methods by NRMSE.  But practitioners
often care about *several* properties simultaneously:

  - low NRMSE                (point accuracy)
  - high Sobol-ST recovery τ (correct importance ranking)
  - low compute cost
  - low surrogate-collapse rate

This module combines those signals into Pareto fronts and
human-readable decision summaries.

Key APIs
--------
- ``accuracy_vs_sensitivity_pareto(nrmse_ranks, sr_summary)``
    2-criteria Pareto: methods that are Pareto-optimal on
    (NRMSE rank, mean Kendall τ).

- ``stability_vs_quality_pareto(sr_summary)``
    2-criteria Pareto on (mean τ, flat-rate) — methods that are
    *both* accurate at importance recovery and *robust* (don't
    silently collapse).

- ``compose_method_scorecard(...)``
    Per-method scorecard combining all criteria into one table
    suitable for a paper "summary" panel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ─── Internal helpers ────────────────────────────────────────────────────
def _is_pareto_min(points: np.ndarray) -> np.ndarray:
    """Pareto-optimal indicator for a *minimisation* problem.

    `points`: (N, D) array; lower-is-better in every dimension.
    Returns a boolean array of length N.
    """
    n = len(points)
    out = np.ones(n, dtype=bool)
    for i in range(n):
        if not out[i]:
            continue
        # Any other point that dominates i?
        dominators = np.all(points <= points[i], axis=1) & \
                     np.any(points <  points[i], axis=1)
        if dominators.any():
            out[i] = False
    return out


# ─── Accuracy vs Sensitivity Pareto ──────────────────────────────────────
def accuracy_vs_sensitivity_pareto(
    nrmse_ranks: pd.Series,
    sr_summary: pd.DataFrame,
    sr_metric: str = "tau_mean",
) -> pd.DataFrame:
    """Identify methods Pareto-optimal on (NRMSE rank, sensitivity τ).

    Parameters
    ----------
    nrmse_ranks
        Per-method mean rank on NRMSE (e.g. from ``rank_methods``).
        Lower is better.
    sr_summary
        Per-method summary from ``sensitivity_recovery``; must contain
        a ``method`` column and the chosen ``sr_metric`` column
        (default ``tau_mean``; higher is better).

    Returns
    -------
    DataFrame with columns
    ``method, nrmse_rank, tau_mean, on_pareto, dominated_by``.
    """
    if "method" not in sr_summary.columns:
        raise ValueError("sr_summary must contain a 'method' column")
    if sr_metric not in sr_summary.columns:
        raise ValueError(f"sr_summary must contain a '{sr_metric}' column")
    sr = sr_summary.set_index("method")[sr_metric]
    common = nrmse_ranks.index.intersection(sr.index)
    if len(common) < 2:
        return pd.DataFrame()

    df = pd.DataFrame({
        "method":     list(common),
        "nrmse_rank": nrmse_ranks.loc[common].to_numpy(),
        sr_metric:    sr.loc[common].to_numpy(),
    })
    # Convert to a minimisation problem: lower NRMSE rank, lower (-τ)
    pts = np.column_stack([df["nrmse_rank"].to_numpy(),
                            -df[sr_metric].to_numpy()])
    on_pareto = _is_pareto_min(pts)
    df["on_pareto"] = on_pareto

    # Identify a "dominated_by" example for non-Pareto methods
    dominators = []
    for _, row in df.iterrows():
        if row["on_pareto"]:
            dominators.append("")
            continue
        dom = df[(df["nrmse_rank"] <= row["nrmse_rank"]) &
                 (df[sr_metric]   >= row[sr_metric])]
        dom = dom[dom["method"] != row["method"]]
        if not dom.empty:
            best_dom = dom.sort_values("nrmse_rank").iloc[0]
            dominators.append(str(best_dom["method"]))
        else:
            dominators.append("")
    df["dominated_by"] = dominators

    return df.sort_values(["on_pareto", "nrmse_rank"],
                           ascending=[False, True]).reset_index(drop=True)


# ─── Stability vs Quality Pareto ─────────────────────────────────────────
def stability_vs_quality_pareto(
    sr_summary: pd.DataFrame,
    quality_metric: str = "tau_mean",
    stability_metric: str = "flat_rate",
) -> pd.DataFrame:
    """Pareto on (sensitivity quality, surrogate-stability rate).

    A method that has high mean τ *but* high flat-rate is brittle.
    """
    needed = {"method", quality_metric, stability_metric}
    if not needed.issubset(sr_summary.columns):
        return pd.DataFrame()
    df = sr_summary[["method", quality_metric, stability_metric]].copy()
    pts = np.column_stack([
        -df[quality_metric].to_numpy(),    # higher τ is better → minimise -τ
         df[stability_metric].to_numpy(),  # lower flat-rate is better
    ])
    df["on_pareto"] = _is_pareto_min(pts)
    return df.sort_values(["on_pareto", quality_metric],
                           ascending=[False, False]).reset_index(drop=True)


# ─── Method scorecard ────────────────────────────────────────────────────
def compose_method_scorecard(
    nrmse_ranks: pd.Series,
    sr_summary: Optional[pd.DataFrame] = None,
    cost_summary: Optional[pd.Series] = None,
    dim_scaling_slopes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Single-row-per-method scorecard combining the major criteria.

    Each non-trivial column is also given a percentile rank within
    the cohort so that a paper can colour-code "best/middle/worst"
    consistently.

    Parameters
    ----------
    nrmse_ranks
        Mean NRMSE rank per method (lower = better).
    sr_summary
        Output of sensitivity_recovery summary (per-method).
    cost_summary
        Optional per-method total compute (mean ``t_total_s``).
    dim_scaling_slopes
        Optional output of ``benchmark.dim_scaling.nrmse_vs_dim``
        with ``method`` and ``log_slope`` columns.
    """
    methods = list(nrmse_ranks.index)
    out = pd.DataFrame({"method": methods,
                         "nrmse_mean_rank": nrmse_ranks.to_numpy()})

    if sr_summary is not None and "method" in sr_summary.columns:
        sr = sr_summary.set_index("method")
        for col in ["tau_mean", "top1_rate", "flat_rate"]:
            if col in sr.columns:
                out[col] = out["method"].map(sr[col]).to_numpy()

    if cost_summary is not None:
        out["compute_s"] = out["method"].map(cost_summary).to_numpy()

    if dim_scaling_slopes is not None and "method" in dim_scaling_slopes.columns:
        ds = dim_scaling_slopes.set_index("method")
        if "log_slope" in ds.columns:
            out["dim_slope"] = out["method"].map(ds["log_slope"]).to_numpy()

    # Percentile-rank each numeric column for colour-coding.
    higher_better = {"tau_mean", "top1_rate"}
    lower_better = {"nrmse_mean_rank", "flat_rate", "compute_s", "dim_slope"}
    for col in list(out.columns):
        if col == "method" or out[col].dtype.kind not in "fc":
            continue
        if col in higher_better:
            out[f"{col}_pct"] = out[col].rank(pct=True)
        elif col in lower_better:
            out[f"{col}_pct"] = (1 - out[col].rank(pct=True))

    return out.sort_values("nrmse_mean_rank").reset_index(drop=True)


__all__ = [
    "accuracy_vs_sensitivity_pareto",
    "compose_method_scorecard",
    "stability_vs_quality_pareto",
]
