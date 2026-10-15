"""
benchmark.reliability_recovery
==============================
Discovery 3 of the benchmark: how do surrogate models' reliability
probability estimates behave near engineering failure boundaries?

The registered analysis evaluates signed bias without presupposing its
direction.  For the 16 ``doe_testbed`` cases that declare
``failure_threshold`` metadata, this module computes reproducible quantities
from the per-row reliability metrics that :class:`benchmark.core.BenchmarkRunner`
streams to ``results.csv``.  Negative values indicate under-prediction and
positive values indicate over-prediction.

Key quantities
--------------
- **Signed bias**  ``b_ij = pf_pred_ij - pf_true_ij`` for run *i* of
  method *j*.  We report the per-method mean signed bias with a 95 %
  bootstrap CI and one-sided tests for both negative and positive bias.
- **Mean absolute P_f error** ``E[|b_j|]`` and balanced accuracy
  ``E[bal_acc_j]`` rank methods by reliability quality.
- **Bias × bias-class breakdown**: separates space-filling, surrogate
  ablation, sequential and BO method families.

Outputs
-------
:func:`per_method_reliability_summary` returns a tidy DataFrame with
columns ``method``, ``n_runs``, ``pf_abs_err_mean``,
``pf_abs_err_ci_lo``, ``pf_abs_err_ci_hi``, ``signed_bias_mean``,
``signed_bias_ci_lo``, ``signed_bias_ci_hi``, ``signed_bias_pvalue``,
``f_bal_acc_mean``, ``f_brier_mean``.

:func:`reliability_ranking` returns one row per method per metric
with mean rank across cases (Friedman-compatible).

:func:`signed_bias_by_family` aggregates the signed bias by method
family for a descriptive comparison (for example, GP vs RF vs Linear vs BO).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

D3_MIN_RUNS_PER_METHOD = 10
# The reference and predicted failure probabilities are fractions of the
# fixed 1,000-point test set.  Canonicalising their difference avoids
# parser-dependent floating-point tails changing Wilcoxon tie ranks.
PROBABILITY_DECIMAL_PLACES = 12


# ─── Long-form (run-level) signed bias ───────────────────────────────────
def extract_long_signed_bias(
    df: pd.DataFrame,
    *,
    output_aggregate: str = "max",
) -> pd.DataFrame:
    """Convert the wide reliability columns into a long-form table.

    Reliability metrics in ``results.csv`` are stored as wide columns
    keyed by output index (e.g. ``pf_abs_err_o0``, ``pf_abs_err_o1``).
    For methods comparison we typically want one row per
    (case, method, repeat, budget) cell with a single aggregated bias.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw ``results.csv`` from :class:`BenchmarkRunner`.
    output_aggregate : str
        Either ``'max'`` (most conservative — worst output dominates)
        or ``'mean'``.  Defaults to ``'max'``.

    Returns
    -------
    pandas.DataFrame
        Columns ``case_id``, ``method``, ``method_family``, ``budget_mult``,
        ``repeat``, ``pf_abs_err``, ``pf_signed_bias``, ``f_bal_acc``,
        ``f_brier`` (NaN if the case lacks a threshold for that output).
        Rows for cases without any failure threshold are dropped.
    """
    if output_aggregate not in {"max", "mean"}:
        raise ValueError("output_aggregate must be 'max' or 'mean'")

    pf_abs = [c for c in df.columns if c.startswith("pf_abs_err_o")]
    pf_pred = [c for c in df.columns if c.startswith("pf_pred_o")]
    pf_true = [c for c in df.columns if c.startswith("pf_true_o")]
    bal_acc = [c for c in df.columns if c.startswith("f_bal_acc_o")]
    brier = [c for c in df.columns if c.startswith("f_brier_o")]

    if not pf_abs:
        return pd.DataFrame()

    agg_fn = np.nanmax if output_aggregate == "max" else np.nanmean

    df = df.copy()
    # The registered inferential population is limited to completed cells.
    # Keep failed/timeout rows in the raw benchmark archive and failure-rate
    # summaries, but never let residual metrics from a non-ok row enter D3.
    if "status" in df.columns:
        df = df[df["status"].eq("ok")].copy()

    def _safe_aggregate(arr_cols: list, fn) -> np.ndarray:
        """Aggregate the listed columns with NaN-safe ``fn``; rows that are
        entirely NaN return NaN cleanly without numpy warnings."""
        if not arr_cols:
            return np.full(len(df), np.nan)
        arr = df[arr_cols].to_numpy()
        out = np.full(arr.shape[0], np.nan)
        all_nan = np.all(np.isnan(arr), axis=1)
        if (~all_nan).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[~all_nan] = fn(arr[~all_nan], axis=1)
        return out

    df["pf_abs_err"] = _safe_aggregate(pf_abs, agg_fn)
    if pf_pred and pf_true:
        # Aggregate signed bias by the output that drives the max abs err
        # to retain its sign.
        pred = df[pf_pred].to_numpy()
        true = df[pf_true].to_numpy()
        bias = np.round(pred - true, decimals=PROBABILITY_DECIMAL_PLACES)
        all_nan_rows = np.all(np.isnan(bias), axis=1)
        if output_aggregate == "max":
            signed = np.full(bias.shape[0], np.nan)
            if (~all_nan_rows).any():
                # nanargmax is NaN-aware; rows with only NaN are handled above
                idx = np.nanargmax(np.abs(bias[~all_nan_rows]), axis=1)
                rows = np.where(~all_nan_rows)[0]
                signed[rows] = bias[rows, idx]
            df["pf_signed_bias"] = signed
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                df["pf_signed_bias"] = np.nanmean(bias, axis=1)
    else:
        df["pf_signed_bias"] = np.nan
    df["f_bal_acc"] = _safe_aggregate(bal_acc, agg_fn)
    df["f_brier"] = _safe_aggregate(brier, agg_fn)

    keep = [
        "case_id", "method", "method_family", "budget_mult", "repeat",
        "pf_abs_err", "pf_signed_bias", "f_bal_acc", "f_brier",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep]
    # Drop rows where every reliability metric is NaN (case has no threshold)
    metric_cols = [c for c in ["pf_abs_err", "pf_signed_bias",
                               "f_bal_acc", "f_brier"] if c in out.columns]
    if metric_cols:
        out = out.dropna(subset=metric_cols, how="all").reset_index(drop=True)
    return out


def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    clusters: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    r"""Percentile bootstrap CI on a sample mean.

    If ``clusters`` is provided (an integer array of the same length as
    ``values``), the bootstrap uses a *cluster bootstrap*: clusters are
    resampled with replacement and all observations within each chosen
    cluster are taken together.  This is the correct estimator when
    observations within a cluster are positively correlated (e.g. all
    method-repeat cells inside the same case share the case's
    intrinsic difficulty).  Naïve i.i.d.\ bootstrap underestimates the
    s.e.\ in that situation.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if clusters is not None:
        clusters = np.asarray(clusters)
        if clusters.size != values.size:
            raise ValueError("clusters must have the same length as values")
        clusters = clusters[finite]
    values = values[finite]
    if values.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    n = values.size

    if clusters is not None and clusters.size > 0:
        unique_c, inv = np.unique(clusters, return_inverse=True)
        groups = [np.where(inv == k)[0] for k in range(unique_c.size)]
        n_g = len(groups)
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            chosen = rng.integers(0, n_g, size=n_g)
            idx = np.concatenate([groups[c] for c in chosen])
            boot_means[i] = values[idx].mean()
    else:
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            boot_means[i] = values[rng.integers(0, n, size=n)].mean()

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_lo": float(np.quantile(boot_means, alpha)),
        "ci_hi": float(np.quantile(boot_means, 1.0 - alpha)),
    }


# ─── Per-method reliability summary ──────────────────────────────────────
def per_method_reliability_summary(
    df_long: pd.DataFrame,
    *,
    budget_mult: Optional[int] = None,
    n_boot: int = 2000,
    seed: int = 0,
    cluster_by: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate the long-form bias table into a per-method summary.

    Parameters
    ----------
    df_long : DataFrame
        Output of :func:`extract_long_signed_bias`.
    budget_mult : int or None
        If given, restrict the analysis to one budget; otherwise pool
        across all budgets.
    n_boot : int
        Number of bootstrap resamples for CI estimation.
    cluster_by : str, optional
        Column name to use as the cluster identifier for a cluster
        bootstrap (e.g. ``"case_id"``).  Recommended when the long table
        contains multiple repeats per case, since per-case observations
        share within-case variance.  If ``None`` (default), the i.i.d.
        bootstrap is used; this matches the historical behaviour but is
        anti-conservative when within-cluster correlation is positive.

    Returns
    -------
    DataFrame
        One row per method with summary columns.  Sorted ascending by
        ``pf_abs_err_mean``.
    """
    df = df_long.copy()
    if budget_mult is not None:
        df = df[df["budget_mult"] == budget_mult]
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    for method, sub in df.groupby("method"):
        abs_err = sub["pf_abs_err"].to_numpy()
        bias = sub["pf_signed_bias"].to_numpy()
        bal_acc = sub.get("f_bal_acc", pd.Series(dtype=float)).to_numpy()
        brier = sub.get("f_brier", pd.Series(dtype=float)).to_numpy()
        clusters = (
            sub[cluster_by].to_numpy() if cluster_by and cluster_by in sub.columns
            else None
        )

        ci_abs = _bootstrap_ci(abs_err, n_boot=n_boot, seed=seed, clusters=clusters)
        ci_bias = _bootstrap_ci(bias, n_boot=n_boot, seed=seed, clusters=clusters)

        # Wilcoxon one-sided tests on the signed bias.
        #
        # Reliability engineering cares about both directions:
        #   * H1: bias < 0 (``pvalue_neg``) — surrogate is
        #     \emph{non-conservative}: P_f^surr < P_f^true, designs
        #     may be accepted that should be rejected.
        #   * H1: bias > 0 (``pvalue_pos``) — surrogate is
        #     \emph{conservative}: P_f^surr > P_f^true, designs
        #     may be rejected that should be accepted.
        # We report both.  Historical manuscripts only reported the
        # "non-conservative" test and then mis-interpreted p≈1 as
        # "strong evidence against non-conservatism" when in fact it
        # is "consistent with the opposite (conservative) alternative".
        bias_clean = bias[np.isfinite(bias)]
        if bias_clean.size >= 6 and not np.all(bias_clean == 0):
            try:
                _, p_neg = wilcoxon(
                    bias_clean, alternative="less", zero_method="wilcox",
                )
                _, p_pos = wilcoxon(
                    bias_clean, alternative="greater", zero_method="wilcox",
                )
                _, p_any = wilcoxon(
                    bias_clean, alternative="two-sided", zero_method="wilcox",
                )
            except ValueError:
                p_neg = p_pos = p_any = float("nan")
        else:
            p_neg = p_pos = p_any = float("nan")

        rows.append({
            "method": method,
            "method_family": (
                sub["method_family"].iloc[0]
                if "method_family" in sub.columns else "unknown"
            ),
            "n_runs": int(len(sub)),
            "low_n": bool(len(sub) < D3_MIN_RUNS_PER_METHOD),
            "pf_abs_err_mean": ci_abs["mean"],
            "pf_abs_err_ci_lo": ci_abs["ci_lo"],
            "pf_abs_err_ci_hi": ci_abs["ci_hi"],
            "signed_bias_mean": ci_bias["mean"],
            "signed_bias_ci_lo": ci_bias["ci_lo"],
            "signed_bias_ci_hi": ci_bias["ci_hi"],
            "signed_bias_pvalue_neg": float(p_neg),
            "signed_bias_pvalue_pos": float(p_pos),
            "signed_bias_pvalue_two_sided": float(p_any),
            "f_bal_acc_mean": (
                float(np.nanmean(bal_acc)) if bal_acc.size else float("nan")
            ),
            "f_brier_mean": (
                float(np.nanmean(brier)) if brier.size else float("nan")
            ),
        })
    out = (
        pd.DataFrame(rows)
        .sort_values("pf_abs_err_mean", ascending=True)
        .reset_index(drop=True)
    )
    # Holm-Bonferroni correction on the registered positive one-sided p-values
    # (DOE-Benchmark §4.3 D3 decision rule: p_adj < 0.05 ∧ CI ∌ 0).
    # Keep the two-sided p-value in its own column for transparent sensitivity
    # reporting, but do not silently substitute it for the registered
    # alternative ``median(signed_bias) > 0``.
    # We compute this once at the upstream so all downstream consumers
    # (prepare_figdata.py, fill_paper_numbers.py, plots_nature.py,
    # Stage 11 templates) see the same single source of truth.
    out["p_holm"] = _holm_adjust(out["signed_bias_pvalue_pos"].to_numpy())
    return out


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Step-down Holm-Bonferroni adjustment.

    Returns an array of the same shape as ``pvals``.  NaN inputs are
    preserved.  Adjusted values are clipped to ``[0, 1]``.  Matches the
    DOE-Benchmark §4.3 D3 decision rule exactly so downstream scripts
    (prepare_figdata.py, fill_paper_numbers.py) reproduce identical
    significant-method sets without re-implementing the algorithm.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    vals = p[finite]
    order = np.argsort(vals)
    m = len(vals)
    adjusted = np.empty_like(vals)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * vals[idx])
        adjusted[idx] = min(running, 1.0)
    out[finite] = adjusted
    return out


# ─── Per-method ranking across cases ─────────────────────────────────────
def reliability_ranking(
    df_long: pd.DataFrame,
    *,
    metric: str = "pf_abs_err",
    budget_mult: Optional[int] = None,
) -> pd.DataFrame:
    """Compute the mean rank of each method across cases for one metric.

    The rank is computed *per (case_id, repeat)* tuple so paired tests
    such as Friedman remain valid.  Lower rank = better.

    Returns a DataFrame with columns ``method``, ``mean_rank``,
    ``rank_std``, ``n_cases``.
    """
    df = df_long.copy()
    if budget_mult is not None:
        df = df[df["budget_mult"] == budget_mult]
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    direction = -1 if metric == "f_bal_acc" else 1  # higher is better for bal_acc
    df["_score"] = direction * df[metric]
    # Average across repeats within each (case, method)
    avg = df.groupby(["case_id", "method"], as_index=False)["_score"].mean()
    avg["rank"] = avg.groupby("case_id")["_score"].rank(method="average")
    out = (
        avg.groupby("method")
        .agg(mean_rank=("rank", "mean"),
             rank_std=("rank", "std"),
             n_cases=("case_id", "nunique"))
        .reset_index()
        .sort_values("mean_rank", ascending=True)
        .reset_index(drop=True)
    )
    return out


# ─── Family-level signed-bias breakdown (GP vs RF vs ...) ────────────────
def signed_bias_by_family(
    df_long: pd.DataFrame,
    *,
    budget_mult: Optional[int] = None,
    n_boot: int = 2000,
    cluster_by: Optional[str] = None,
) -> pd.DataFrame:
    r"""Aggregate signed P_f bias by method family.

    This is a descriptive family-level breakdown.  It does not by itself
    identify a causal surrogate-family effect, because acquisition strategy
    and surrogate family are not fully crossed in the benchmark design.

    Returns columns ``method_family``, ``n_runs``, ``signed_bias_mean``,
    ``signed_bias_ci_lo``, ``signed_bias_ci_hi``,
    ``pvalue_one_sided_neg``.

    Set ``cluster_by="case_id"`` to switch from i.i.d.\ to cluster
    bootstrap; see :func:`per_method_reliability_summary` for the
    rationale.
    """
    df = df_long.copy()
    if budget_mult is not None:
        df = df[df["budget_mult"] == budget_mult]
    if df.empty:
        return pd.DataFrame()
    family_col = (
        "method_family" if "method_family" in df.columns else "method"
    )
    rows: List[Dict] = []
    for fam, sub in df.groupby(family_col):
        bias = sub["pf_signed_bias"].to_numpy()
        clusters = (
            sub[cluster_by].to_numpy() if cluster_by and cluster_by in sub.columns
            else None
        )
        ci = _bootstrap_ci(bias, n_boot=n_boot, clusters=clusters)
        bias_clean = bias[np.isfinite(bias)]
        if bias_clean.size >= 6 and not np.all(bias_clean == 0):
            try:
                _, p_neg = wilcoxon(bias_clean, alternative="less")
                _, p_pos = wilcoxon(bias_clean, alternative="greater")
            except ValueError:
                p_neg = p_pos = float("nan")
        else:
            p_neg = p_pos = float("nan")
        rows.append({
            "method_family": fam,
            "n_runs": int(len(sub)),
            "signed_bias_mean": ci["mean"],
            "signed_bias_ci_lo": ci["ci_lo"],
            "signed_bias_ci_hi": ci["ci_hi"],
            "pvalue_one_sided_neg": float(p_neg),
            "pvalue_one_sided_pos": float(p_pos),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("signed_bias_mean")
        .reset_index(drop=True)
    )


# ─── Convenience top-level wrapper ───────────────────────────────────────
def reliability_recovery_report(
    results_csv: pd.DataFrame,
    *,
    budget_mult: Optional[int] = None,
    output_aggregate: str = "max",
    cluster_by: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the full Discovery-3 analysis on a results.csv DataFrame.

    Set ``cluster_by="case_id"`` for a cluster bootstrap on the per-case
    cluster (recommended whenever the input has multiple repeats per case).

    Returns a dict with keys ``long``, ``per_method``, ``ranking_abs_err``,
    ``ranking_bal_acc``, ``family_bias``.
    """
    long = extract_long_signed_bias(results_csv, output_aggregate=output_aggregate)
    per_method = per_method_reliability_summary(
        long, budget_mult=budget_mult, cluster_by=cluster_by,
    )
    ranking_abs = reliability_ranking(long, metric="pf_abs_err",
                                      budget_mult=budget_mult)
    ranking_bal = reliability_ranking(long, metric="f_bal_acc",
                                      budget_mult=budget_mult)
    family = signed_bias_by_family(
        long, budget_mult=budget_mult, cluster_by=cluster_by,
    )
    return {
        "long": long,
        "per_method": per_method,
        "ranking_abs_err": ranking_abs,
        "ranking_bal_acc": ranking_bal,
        "family_bias": family,
    }


def reliability_ci_sensitivity(
    long: pd.DataFrame,
    *,
    budget_mult: Optional[int] = None,
    primary_cluster_by: str = "case_id",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare D3 primary cluster bootstrap CIs against i.i.d. CIs.

    The SAP requires cluster-aware D3 CIs as the primary estimator and
    an i.i.d. bootstrap as sensitivity.  This helper produces the SI-ready
    width-ratio table without changing the primary per-method summary.
    """
    primary = per_method_reliability_summary(
        long,
        budget_mult=budget_mult,
        n_boot=n_boot,
        seed=seed,
        cluster_by=primary_cluster_by,
    )
    iid = per_method_reliability_summary(
        long,
        budget_mult=budget_mult,
        n_boot=n_boot,
        seed=seed,
        cluster_by=None,
    )
    if primary.empty or iid.empty:
        return pd.DataFrame()

    primary_cols = [
        "method",
        "signed_bias_ci_lo",
        "signed_bias_ci_hi",
        "pf_abs_err_ci_lo",
        "pf_abs_err_ci_hi",
    ]
    iid_cols = primary_cols
    merged = primary[primary_cols].merge(
        iid[iid_cols],
        on="method",
        how="inner",
        suffixes=("_cluster", "_iid"),
    )
    merged["signed_bias_ci_width_cluster"] = (
        merged["signed_bias_ci_hi_cluster"] - merged["signed_bias_ci_lo_cluster"]
    )
    merged["signed_bias_ci_width_iid"] = (
        merged["signed_bias_ci_hi_iid"] - merged["signed_bias_ci_lo_iid"]
    )
    merged["signed_bias_ci_width_ratio_cluster_over_iid"] = (
        merged["signed_bias_ci_width_cluster"] / merged["signed_bias_ci_width_iid"]
    ).replace([np.inf, -np.inf], np.nan)
    merged["pf_abs_err_ci_width_cluster"] = (
        merged["pf_abs_err_ci_hi_cluster"] - merged["pf_abs_err_ci_lo_cluster"]
    )
    merged["pf_abs_err_ci_width_iid"] = (
        merged["pf_abs_err_ci_hi_iid"] - merged["pf_abs_err_ci_lo_iid"]
    )
    merged["pf_abs_err_ci_width_ratio_cluster_over_iid"] = (
        merged["pf_abs_err_ci_width_cluster"] / merged["pf_abs_err_ci_width_iid"]
    ).replace([np.inf, -np.inf], np.nan)
    merged["primary_cluster_by"] = primary_cluster_by
    merged["sensitivity_analysis"] = "iid_bootstrap"
    keep = [
        "method",
        "primary_cluster_by",
        "sensitivity_analysis",
        "signed_bias_ci_width_cluster",
        "signed_bias_ci_width_iid",
        "signed_bias_ci_width_ratio_cluster_over_iid",
        "pf_abs_err_ci_width_cluster",
        "pf_abs_err_ci_width_iid",
        "pf_abs_err_ci_width_ratio_cluster_over_iid",
    ]
    return merged[keep].sort_values("method").reset_index(drop=True)


__all__ = [
    "PROBABILITY_DECIMAL_PLACES",
    "extract_long_signed_bias",
    "per_method_reliability_summary",
    "reliability_ranking",
    "signed_bias_by_family",
    "reliability_recovery_report",
    "reliability_ci_sensitivity",
]
