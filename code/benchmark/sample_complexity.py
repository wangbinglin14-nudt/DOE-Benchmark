"""
benchmark.sample_complexity
===========================
Discovery 2 of the Nature-grade benchmark: how many training points
``n* / d`` does each method need to reach a target NRMSE, and how
strongly does that requirement depend on the response-surface topology
of the case?

Background
----------
Loeppky et al. (*Choosing the sample size of a computer experiment: a
practical guide*, Technometrics 2009) recommend ``n = 10·d`` as a
rule of thumb for global-accuracy DOE.  Their evidence base was
analytical test functions.  This module re-checks the rule on
physics-grounded engineering cases stratified by response-surface
topology.

Approach
--------
For every (method, topology) pair we fit a learning curve to the
median NRMSE versus budget multiplier ``m`` (the budget expressed as
``n / d``).  We then read off ``m*`` — the smallest multiplier that
brings the median NRMSE under a user-supplied threshold.

A monotone non-increasing learning curve is enforced via a cumulative
minimum on the budget axis (small noisy upticks at large ``m`` are
ignored — the method has already "passed" once below the threshold).

Outputs
-------
:func:`learning_curves_by_topology`  → long-form per-(topology, method,
budget) median + 95 % bootstrap CI.

:func:`n_star_per_topology`          → wide-form ``m*`` table:
rows are methods, columns are topology tags, values are
``n* / d`` (or ``∞`` when no budget reached the threshold).

:func:`compare_to_10d_rule`          → one-row-per-(topology) summary
giving the fraction of methods that meet the threshold at ``m=10``,
the median ``m*``, and the worst-case ``m*``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from benchmark.stratified import annotate_results

PRIMARY_SAMPLE_COMPLEXITY_ANALYSIS = "penalized_failed_in_denominator"
# The preregistered D2 target.  The 0.05 engineering target remains available
# as an explicit sensitivity analysis, but must not be the library default.
PRIMARY_NRMSE_THRESHOLD = 0.20


# ─── Helpers ─────────────────────────────────────────────────────────────
def _bootstrap_median_ci(
    x: np.ndarray,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = np.median(x[rng.integers(0, x.size, size=x.size)])
    alpha = (1.0 - ci) / 2.0
    return {
        "median": float(np.median(x)),
        "ci_lo": float(np.quantile(boot, alpha)),
        "ci_hi": float(np.quantile(boot, 1.0 - alpha)),
    }


def _explode_topology(df: pd.DataFrame) -> pd.DataFrame:
    """Unpack the tuple-valued ``topology`` column into one row per tag.

    Cases that carry multiple topology tags appear once per tag.
    Cases without any tag are kept under ``topology='untagged'`` so
    the totals match the input row-count.
    """
    if "topology" not in df.columns:
        df = annotate_results(df)
    df = df.copy()

    def _normalise_tags(t):
        if isinstance(t, str):
            return (t,) if t else ()
        if isinstance(t, (tuple, list, set)):
            return tuple(t)
        return ()
    df["topology"] = df["topology"].apply(_normalise_tags)
    df["topology"] = df["topology"].apply(lambda t: t if t else ("untagged",))
    return df.explode("topology").reset_index(drop=True)


# ─── Public API ──────────────────────────────────────────────────────────
def learning_curves_by_topology(
    df: pd.DataFrame,
    *,
    metric: str = "nrmse",
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Median ``metric`` versus budget for every (topology, method).

    Rows with ``status != 'ok'`` are excluded from the performance median
    because they have no valid metric.  The primary D2 denominator policy is
    enforced downstream by :func:`compare_to_10d_rule`: such cells count as
    non-passing while the full method pool remains the denominator.  No
    synthetic metric or pseudo-penalty is written into the learning curve.

    Parameters
    ----------
    df : DataFrame
        ``results.csv`` from :class:`BenchmarkRunner`.
    metric : str
        The accuracy metric to study (default ``'nrmse'``).
    n_boot : int
        Bootstrap replicates for the 95 % CI.

    Returns
    -------
    DataFrame
        Columns ``topology``, ``method``, ``budget_mult``, ``n_cases``,
        ``n_runs``, ``median``, ``ci_lo``, ``ci_hi``.
    """
    if metric not in df.columns:
        raise KeyError(f"{metric!r} not in dataframe columns")
    long = _explode_topology(df)
    if "status" in long.columns:
        long = long[long["status"] == "ok"]

    rows: List[Dict] = []
    for (topo, method, b), sub in long.groupby(["topology", "method", "budget_mult"]):
        ci = _bootstrap_median_ci(sub[metric].to_numpy(),
                                  n_boot=n_boot, seed=seed)
        rows.append({
            "topology": topo,
            "method": method,
            "budget_mult": int(b),
            "median": ci["median"],
            "ci_lo": ci["ci_lo"],
            "ci_hi": ci["ci_hi"],
            "n_cases": int(sub["case_id"].nunique()),
            "n_runs": int(len(sub)),
        })
    if not rows:
        return pd.DataFrame(
            columns=[
                "topology",
                "method",
                "budget_mult",
                "median",
                "ci_lo",
                "ci_hi",
                "n_cases",
                "n_runs",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["topology", "method", "budget_mult"])
        .reset_index(drop=True)
    )


def n_star_per_topology(
    learning_df: pd.DataFrame,
    *,
    nrmse_threshold: float = PRIMARY_NRMSE_THRESHOLD,
    monotone: bool = False,
) -> pd.DataFrame:
    """Compute ``m*`` (= ``n*/d``) per (method, topology).

    Parameters
    ----------
    learning_df : DataFrame
        Output of :func:`learning_curves_by_topology`.
    nrmse_threshold : float
        Target NRMSE (default 0.20 = 20 %, the preregistered D2 target).
    monotone : bool
        If True, use a cumulative minimum on the budget axis before reading
        off ``m*`` as an optional sensitivity analysis.  The registered
        primary definition uses the observed budget medians directly, so the
        default is False.

    Returns
    -------
    DataFrame
        Pivoted: rows = methods, columns = topology tags, values =
        ``m*`` (NaN if no observed budget reaches the threshold).
    """
    df = learning_df.copy()
    if monotone:
        df = df.sort_values(["topology", "method", "budget_mult"])
        df["median"] = (
            df.groupby(["topology", "method"])["median"]
            .transform(lambda s: s.cummin())
        )

    rows: List[Dict] = []
    for (topo, method), sub in df.groupby(["topology", "method"]):
        ok = sub[sub["median"] <= nrmse_threshold]
        if ok.empty:
            n_star = float("nan")
        else:
            n_star = float(ok["budget_mult"].min())
        rows.append({
            "topology": topo,
            "method": method,
            "n_star_over_d": n_star,
        })
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    return long.pivot(index="method", columns="topology", values="n_star_over_d")


def compare_to_10d_rule(
    learning_df: pd.DataFrame,
    *,
    nrmse_threshold: float = PRIMARY_NRMSE_THRESHOLD,
    rule_budget: int = 10,
    source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-topology summary of the 10·d rule of thumb (Loeppky 2009).

    For each topology tag we report:
    - ``methods_total``       — full method-pool size from ``source_df``
    - ``methods_pass_at_10d`` — number meeting the threshold at m=10
    - ``frac_pass_at_10d``    — pass count divided by the full method pool
    - ``median_n_star``       — across passing-ever methods only
    - ``max_n_star``          — worst-case passing-ever method
    - ``n_methods_never_pass`` — full-pool methods with NaN ``n*``
    - ``n_cases`` / ``case_ids`` / ``exploratory``
    - ``n_failed_cells_per_topology`` — failed or timeout source rows

    ``penalized_failed_in_denominator`` means that failed/timeout cells are
    treated as non-passing for the pass fraction and retained in the method
    denominator.  They are not assigned an artificial NRMSE value; medians
    are computed only from valid ``ok`` rows and the failure count is audited
    explicitly.
    """
    # The primary D2 estimand is the first observed budget whose median meets
    # the threshold; monotone smoothing is available only as an explicit
    # sensitivity through n_star_per_topology(..., monotone=True).
    n_star = n_star_per_topology(
        learning_df, nrmse_threshold=nrmse_threshold, monotone=False
    )
    source_long = _explode_topology(source_df) if source_df is not None else None
    if source_long is not None:
        method_pool = sorted(source_long["method"].dropna().astype(str).unique())
        topologies = sorted(set(n_star.columns) | set(source_long["topology"].dropna()))
    else:
        method_pool = sorted(learning_df["method"].dropna().astype(str).unique())
        topologies = sorted(n_star.columns)

    columns = [
        "topology",
        "analysis_type",
        "is_primary",
        "methods_total",
        "methods_pass_at_10d",
        "frac_pass_at_10d",
        "median_n_star",
        "max_n_star",
        "n_methods_never_pass",
        "n_cases",
        "case_ids",
        "exploratory",
        "n_failed_cells_per_topology",
    ]
    if not method_pool:
        return pd.DataFrame(columns=columns)

    rows: List[Dict] = []
    for topo in topologies:
        if source_long is not None:
            src_topo = source_long[source_long["topology"] == topo]
            case_ids = sorted(src_topo["case_id"].dropna().astype(str).unique())
            n_cases = len(case_ids)
            if "status" in src_topo.columns:
                n_failed = int(src_topo["status"].ne("ok").sum())
            else:
                n_failed = 0
        else:
            learn_topo = learning_df[learning_df["topology"] == topo]
            case_ids = []
            n_cases = (
                int(learn_topo["n_cases"].max())
                if "n_cases" in learn_topo.columns and len(learn_topo)
                else 0
            )
            n_failed = 0

        n_methods = len(method_pool)
        if topo in n_star.columns:
            col_all = n_star[topo].reindex(method_pool)
        else:
            col_all = pd.Series(index=method_pool, dtype=float)
        finite_n_star = col_all.dropna()
        passes = col_all <= rule_budget
        n_pass = int(passes.fillna(False).sum())
        rows.append({
            "topology": topo,
            "analysis_type": PRIMARY_SAMPLE_COMPLEXITY_ANALYSIS,
            "is_primary": True,
            "methods_total": int(n_methods),
            "methods_pass_at_10d": n_pass,
            "frac_pass_at_10d": float(n_pass / n_methods),
            "median_n_star": (
                float(finite_n_star.median()) if len(finite_n_star) else float("nan")
            ),
            "max_n_star": (
                float(finite_n_star.max()) if len(finite_n_star) else float("nan")
            ),
            "n_methods_never_pass": int(n_methods - len(finite_n_star)),
            "n_cases": int(n_cases),
            "case_ids": ";".join(case_ids),
            "exploratory": bool(n_cases < 3),
            "n_failed_cells_per_topology": int(n_failed),
        })
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("median_n_star", ascending=True, na_position="last")
        .reset_index(drop=True)
    )


def sample_complexity_report(
    df: pd.DataFrame,
    *,
    metric: str = "nrmse",
    nrmse_threshold: float = PRIMARY_NRMSE_THRESHOLD,
    rule_budget: int = 10,
    n_boot: int = 1000,
) -> Dict[str, pd.DataFrame]:
    """Convenience top-level wrapper.

    Returns a dict with keys ``learning``, ``n_star``, ``rule_summary``.
    """
    learning = learning_curves_by_topology(df, metric=metric, n_boot=n_boot)
    n_star = n_star_per_topology(
        learning, nrmse_threshold=nrmse_threshold, monotone=False
    )
    rule = compare_to_10d_rule(
        learning, nrmse_threshold=nrmse_threshold, rule_budget=rule_budget,
        source_df=df,
    )
    return {"learning": learning, "n_star": n_star, "rule_summary": rule}


__all__ = [
    "PRIMARY_SAMPLE_COMPLEXITY_ANALYSIS",
    "PRIMARY_NRMSE_THRESHOLD",
    "learning_curves_by_topology",
    "n_star_per_topology",
    "compare_to_10d_rule",
    "sample_complexity_report",
]
