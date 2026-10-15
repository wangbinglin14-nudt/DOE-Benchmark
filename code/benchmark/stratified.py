"""
benchmark.stratified
====================
Multi-dimensional stratified analysis of benchmark results.

The flat "ranks per method" view of :mod:`benchmark.analysis` answers
*how does method X compare to method Y across the whole library?*  This
module answers the more nuanced question

  *On which classes of problems does method X dominate?*

by stratifying the rank computation over case attributes:

- **VS scenario** (VS-A through VS-H) — what kind of DOE method
  research the case is most relevant to.
- **Response-surface topology** (smooth_monotone, threshold_jump,
  heterogeneous, ...) — geometric features of the response.
- **Domain** (structures, aerodynamics, heat_transfer, ...).
- **Dimension bucket** (low: d≤4, mid: 5≤d≤8, high: d≥9).

For each stratum a separate Friedman test, mean-rank table, and
critical-difference value is computed.  The output is a *per-stratum
ranks panel* that turns the benchmark from a single number into a
diagnostic of method-problem matching.

Bootstrap confidence intervals on rank shifts are also provided so
that paper figures can quote ranks with statistical uncertainty.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

import doe_testbed as db
from benchmark.analysis import critical_difference, friedman_test, pivot_metric, rank_methods


# ─── Case-attribute lookup ────────────────────────────────────────────────
def get_case_attributes(case_id: str) -> Dict[str, object]:
    """Return a flat dict of stratification attributes for one case."""
    try:
        meta = db.get_case_meta(case_id)
    except Exception:
        return {
            "domain": "unknown",
            "dim_bucket": "unknown",
            "applicable_scenarios": (),
            "response_topology": (),
            "has_failure_boundary": False,
            "cost_level": "unknown",
        }
    d = meta.dim_input
    if d <= 4:
        bucket = "low"
    elif d <= 8:
        bucket = "mid"
    else:
        bucket = "high"
    return {
        "domain": getattr(meta, "domain", "unknown"),
        "dim_bucket": bucket,
        "applicable_scenarios": tuple(meta.applicable_scenarios or ()),
        "response_topology": tuple(meta.response_topology or ()),
        "has_failure_boundary": bool(getattr(meta, "has_failure_boundary", False)),
        "cost_level": getattr(meta, "cost_level", "unknown"),
    }


def annotate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Add case-attribute columns (in-place safe; returns a new DataFrame)."""
    df = df.copy()
    if "case_id" not in df.columns:
        return df
    cache: Dict[str, Dict] = {}

    def lookup(case_id: str, key: str):
        if case_id not in cache:
            cache[case_id] = get_case_attributes(case_id)
        return cache[case_id].get(key)

    df["domain_attr"] = df["case_id"].map(lambda c: lookup(c, "domain"))
    df["dim_bucket"] = df["case_id"].map(lambda c: lookup(c, "dim_bucket"))
    df["topology"] = df["case_id"].map(lambda c: lookup(c, "response_topology"))
    df["scenarios"] = df["case_id"].map(lambda c: lookup(c, "applicable_scenarios"))
    df["has_failure_boundary"] = df["case_id"].map(
        lambda c: lookup(c, "has_failure_boundary")
    )
    df["cost_level"] = df["case_id"].map(lambda c: lookup(c, "cost_level"))
    return df


# ─── Per-stratum ranks ────────────────────────────────────────────────────
def stratified_ranks(
    df: pd.DataFrame,
    metric: str = "nrmse",
    lower_is_better: bool = True,
    budget_value: Optional[int] = None,
    stratify_by: str = "vs_scenario",
) -> pd.DataFrame:
    """Compute mean ranks for each method *within each stratum*.

    Parameters
    ----------
    stratify_by :
        ``'vs_scenario'`` — one row per VS tag (a case may belong to ≥1)
        ``'topology'``    — one row per topology tag (a case may belong to ≥1)
        ``'domain'``      — one row per domain
        ``'dim_bucket'``  — low / mid / high dimension

    Returns
    -------
    DataFrame indexed by stratum, columns = method names, values =
    mean rank within that stratum.  An additional ``n_cases`` column
    reports the stratum size, and ``critical_difference`` reports the
    Nemenyi CD for that stratum.
    """
    df = annotate_results(df)
    if budget_value is not None and "budget_mult" in df.columns:
        df = df[df["budget_mult"] == budget_value]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    methods = sorted(df["method"].unique())
    rows = []

    if stratify_by in ("vs_scenario", "topology"):
        # Multi-tag attributes: a case can be in multiple strata
        col = "scenarios" if stratify_by == "vs_scenario" else "topology"
        all_tags = sorted({tag for tags in df[col] for tag in tags})
        for tag in all_tags:
            sub = df[df[col].apply(lambda tags, tag=tag: tag in tags)]
            cases_in = sub["case_id"].unique()
            if len(cases_in) < 2:
                continue
            pivot = pivot_metric(sub, metric)
            ranks = rank_methods(pivot, lower_is_better=lower_is_better)
            fstat, fp = friedman_test(pivot)
            try:
                cd = critical_difference(len(ranks), len(pivot.dropna()))
            except (KeyError, ValueError):
                cd = float("nan")
            row = {"stratum": tag, "n_cases": len(cases_in),
                   "friedman_chi2": fstat, "friedman_p": fp,
                   "critical_difference": cd}
            for m in methods:
                row[m] = float(ranks.get(m, float("nan")))
            rows.append(row)
    else:
        col = stratify_by  # 'domain' or 'dim_bucket'
        col_actual = "domain_attr" if col == "domain" else col
        for tag, sub in df.groupby(col_actual, dropna=False):
            cases_in = sub["case_id"].unique()
            if len(cases_in) < 2:
                continue
            pivot = pivot_metric(sub, metric)
            ranks = rank_methods(pivot, lower_is_better=lower_is_better)
            fstat, fp = friedman_test(pivot)
            try:
                cd = critical_difference(len(ranks), len(pivot.dropna()))
            except (KeyError, ValueError):
                cd = float("nan")
            row = {"stratum": str(tag), "n_cases": len(cases_in),
                   "friedman_chi2": fstat, "friedman_p": fp,
                   "critical_difference": cd}
            for m in methods:
                row[m] = float(ranks.get(m, float("nan")))
            rows.append(row)

    cols = ["stratum", "n_cases", "friedman_chi2", "friedman_p", "critical_difference", *methods]
    return pd.DataFrame(rows, columns=cols)


# ─── Bootstrap CI on rank shifts ─────────────────────────────────────────
def bootstrap_rank_ci(
    df: pd.DataFrame,
    metric: str = "nrmse",
    lower_is_better: bool = True,
    budget_value: Optional[int] = None,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap confidence intervals on each method's mean rank.

    Resamples cases with replacement (clustered bootstrap) to obtain
    a distribution of mean ranks per method.

    Returns
    -------
    DataFrame with columns:
      - method, mean_rank, ci_low, ci_high, ci_width
    """
    if budget_value is not None and "budget_mult" in df.columns:
        df = df[df["budget_mult"] == budget_value]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    pivot = pivot_metric(df, metric)
    methods = pivot.columns.tolist()
    cases = pivot.index.tolist()
    n_cases = len(cases)
    if n_cases < 2:
        return pd.DataFrame(
            {"method": methods, "mean_rank": [float("nan")] * len(methods),
             "ci_low": [float("nan")] * len(methods),
             "ci_high": [float("nan")] * len(methods),
             "ci_width": [float("nan")] * len(methods)}
        )

    rng = np.random.default_rng(seed)
    boot_ranks = np.zeros((n_boot, len(methods)))
    for b in range(n_boot):
        idx = rng.integers(0, n_cases, size=n_cases)
        sample = pivot.iloc[idx]
        ranks = rank_methods(sample, lower_is_better=lower_is_better)
        boot_ranks[b, :] = [ranks.get(m, float("nan")) for m in methods]

    # Point estimate from the original
    point = rank_methods(pivot, lower_is_better=lower_is_better)
    alpha = 1.0 - ci
    lo = np.nanpercentile(boot_ranks, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot_ranks, 100 * (1 - alpha / 2), axis=0)

    return pd.DataFrame({
        "method": methods,
        "mean_rank": [point.get(m, float("nan")) for m in methods],
        "ci_low": lo,
        "ci_high": hi,
        "ci_width": hi - lo,
    }).sort_values("mean_rank")


def bootstrap_rank_shift_ci(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str = "nrmse",
    lower_is_better: bool = True,
    budget_value: Optional[int] = None,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap CI on rank-shifts between two case populations (e.g. analytical vs physical).

    Returns
    -------
    DataFrame with: method, rank_a, rank_b, shift, shift_ci_low, shift_ci_high.
    """
    if budget_value is not None:
        df_a = df_a[df_a["budget_mult"] == budget_value]
        df_b = df_b[df_b["budget_mult"] == budget_value]
    df_a = df_a[df_a["status"] == "ok"] if "status" in df_a.columns else df_a
    df_b = df_b[df_b["status"] == "ok"] if "status" in df_b.columns else df_b

    pivot_a = pivot_metric(df_a, metric)
    pivot_b = pivot_metric(df_b, metric)

    methods = sorted(set(pivot_a.columns) & set(pivot_b.columns))
    pivot_a = pivot_a[methods]
    pivot_b = pivot_b[methods]

    rng = np.random.default_rng(seed)
    boot_shifts = np.zeros((n_boot, len(methods)))
    n_a, n_b = len(pivot_a), len(pivot_b)
    for k in range(n_boot):
        ia = rng.integers(0, n_a, size=n_a)
        ib = rng.integers(0, n_b, size=n_b)
        ra = rank_methods(pivot_a.iloc[ia], lower_is_better=lower_is_better)
        rb = rank_methods(pivot_b.iloc[ib], lower_is_better=lower_is_better)
        boot_shifts[k, :] = [
            (rb.get(m, float("nan")) - ra.get(m, float("nan"))) for m in methods
        ]

    rank_a_point = rank_methods(pivot_a, lower_is_better=lower_is_better)
    rank_b_point = rank_methods(pivot_b, lower_is_better=lower_is_better)

    alpha = 1.0 - ci
    lo = np.nanpercentile(boot_shifts, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot_shifts, 100 * (1 - alpha / 2), axis=0)

    out = pd.DataFrame({
        "method": methods,
        "rank_a": [rank_a_point.get(m, float("nan")) for m in methods],
        "rank_b": [rank_b_point.get(m, float("nan")) for m in methods],
        "shift": [
            rank_b_point.get(m, float("nan")) - rank_a_point.get(m, float("nan"))
            for m in methods
        ],
        "shift_ci_low": lo,
        "shift_ci_high": hi,
    })
    out["shift_significant"] = (out["shift_ci_low"] > 0) | (out["shift_ci_high"] < 0)
    return out.sort_values("shift")


# ─── Per-case best counts (winner heatmap) ───────────────────────────────
def winner_table(
    df: pd.DataFrame,
    metric: str = "nrmse",
    lower_is_better: bool = True,
    budget_value: Optional[int] = None,
) -> pd.DataFrame:
    """For each case, who is the best method? (one-row-per-case)."""
    if budget_value is not None and "budget_mult" in df.columns:
        df = df[df["budget_mult"] == budget_value]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    pivot = pivot_metric(df, metric)
    if lower_is_better:
        winners = pivot.idxmin(axis=1)
        scores = pivot.min(axis=1)
    else:
        winners = pivot.idxmax(axis=1)
        scores = pivot.max(axis=1)
    return pd.DataFrame({
        "case_id": pivot.index,
        "winner": winners.values,
        f"best_{metric}": scores.values,
    })


__all__ = [
    "annotate_results",
    "bootstrap_rank_ci",
    "bootstrap_rank_shift_ci",
    "get_case_attributes",
    "stratified_ranks",
    "winner_table",
]
