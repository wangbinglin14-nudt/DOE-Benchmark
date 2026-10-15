"""
benchmark.analysis
==================
Statistical analysis of benchmark results.

Provides:
- :func:`friedman_test`         — Friedman omnibus test for k methods on N cases
- :func:`nemenyi_post_hoc`      — Nemenyi pairwise post-hoc test
- :func:`critical_difference`   — Critical-difference value at significance α
- :func:`rank_methods`          — Compute mean ranks per method (lower = better)
- :func:`pivot_metric`          — Pivot results into (case × method) matrix

Reference: Demšar (2006), "Statistical Comparisons of Classifiers over
Multiple Data Sets", *Journal of Machine Learning Research*, 7:1–30.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata


def pivot_metric(
    df: pd.DataFrame,
    metric: str,
    aggfunc: str = "median",
    case_col: str = "case_id",
    method_col: str = "method",
    budget_col: Optional[str] = "budget_mult",
    budget_value: Optional[int] = None,
) -> pd.DataFrame:
    """Pivot benchmark results into a (case × method) matrix for one metric.

    If ``budget_col`` and ``budget_value`` are provided, only that budget
    is included.
    """
    if budget_col is not None and budget_value is not None:
        df = df[df[budget_col] == budget_value]
    df = df[df["status"] == "ok"] if "status" in df.columns else df
    pivot = df.pivot_table(
        index=case_col,
        columns=method_col,
        values=metric,
        aggfunc=aggfunc,
    )
    return pivot


def rank_methods(
    pivot: pd.DataFrame,
    lower_is_better: bool = True,
) -> pd.Series:
    """Per-case ranks averaged over cases.

    Parameters
    ----------
    pivot : DataFrame
        Rows = cases, columns = methods, values = metric.
    lower_is_better : bool
        For RMSE/NRMSE/MAE this is True; for R² use False.

    Returns
    -------
    mean_ranks : Series indexed by method name (lower = better).

    Notes
    -----
    Cases with any NaN are dropped before ranking, to avoid bias from
    methods that failed on a subset of cases.
    """
    clean = pivot.dropna(how="any")
    if clean.empty:
        # Fall back to per-method nanmean ranking if no complete cases
        return pd.Series(np.nan, index=pivot.columns).sort_values()
    arr = clean.to_numpy()
    if not lower_is_better:
        arr = -arr
    # Rank within each row (case); ties get average rank.
    ranks_per_case = np.apply_along_axis(rankdata, 1, arr)
    mean_ranks = ranks_per_case.mean(axis=0)
    return pd.Series(mean_ranks, index=clean.columns).sort_values()


def friedman_test(pivot: pd.DataFrame) -> Tuple[float, float]:
    """Friedman test on a (cases × methods) matrix.

    Returns (statistic, p-value).  Drops cases with any NaN.
    """
    clean = pivot.dropna()
    if len(clean) < 2:
        return float("nan"), float("nan")
    cols = [clean.iloc[:, j].to_numpy() for j in range(clean.shape[1])]
    stat, p = friedmanchisquare(*cols)
    return float(stat), float(p)


def critical_difference(
    n_methods: int,
    n_cases: int,
    alpha: float = 0.05,
) -> float:
    """Nemenyi critical-difference value for the CD diagram.

    CD = q_α * sqrt(k * (k + 1) / (6 * N))

    where ``q_α`` is the Studentised range statistic.

    For ``α = 0.05`` and ``k ≤ 20`` we use the tabulated values from
    Demšar (2006), Table 5 — the canonical reference for the
    Friedman–Nemenyi CD diagram.  For ``k > 20`` (e.g., the
    post-registration ``sota_pool`` with 22 methods, see DOE-Benchmark
    §11 GAP-5) or non-default ``α``, we fall back to a numerical
    evaluation via ``scipy.stats.studentized_range`` with ``df = ∞``,
    which matches the asymptotic q-table used by Demšar.
    """
    if n_methods < 2:
        raise ValueError(f"n_methods must be >= 2, got {n_methods}")
    if n_cases < 1:
        raise ValueError(f"n_cases must be >= 1, got {n_cases}")
    # Table 5 in Demšar (2006), q_α at α=0.05 for k methods (infinite N).
    Q_TABLE_05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
        11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
        15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489,
        19: 3.517, 20: 3.544,
    }
    if alpha == 0.05 and n_methods in Q_TABLE_05:
        q = Q_TABLE_05[n_methods]
    else:
        # Numerical fallback: studentised range with df → ∞ matches the
        # asymptotic q-table used in Demšar (2006).  Required for the
        # post-registration sota_pool (k=22) and any non-default α.
        try:
            from scipy.stats import studentized_range
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "critical_difference requires scipy>=1.7 for the "
                "studentized_range fallback when k>20 or alpha != 0.05; "
                f"scipy import failed: {exc}"
            ) from exc
        q = float(studentized_range.ppf(1.0 - alpha, n_methods, float("inf")))
    return float(q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_cases)))


def nemenyi_post_hoc(
    pivot: pd.DataFrame,
    alpha: float = 0.05,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Pairwise Nemenyi post-hoc test.

    Returns a (k × k) DataFrame ``D`` of rank differences.  Pairs with
    ``|D_ij| > CD`` are statistically different at level ``alpha``.
    """
    ranks = rank_methods(pivot, lower_is_better=lower_is_better)
    methods = ranks.index.tolist()
    k = len(methods)
    n_cases = pivot.dropna().shape[0]
    cd = critical_difference(k, n_cases, alpha=alpha)

    diff_matrix = np.zeros((k, k))
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            diff_matrix[i, j] = ranks[mi] - ranks[mj]
    df = pd.DataFrame(diff_matrix, index=methods, columns=methods)
    df.attrs["critical_difference"] = cd
    df.attrs["alpha"] = alpha
    df.attrs["n_cases"] = n_cases
    return df


def cd_diagram_data(
    pivot: pd.DataFrame,
    alpha: float = 0.05,
    lower_is_better: bool = True,
) -> Dict:
    """Compute CD-diagram inputs (used by visualisation code).

    Returns
    -------
    dict with keys: ``ranks`` (Series), ``cd`` (float),
    ``cliques`` (list of lists of method names that are not significantly
    different from each other).
    """
    ranks = rank_methods(pivot, lower_is_better=lower_is_better)
    n_methods = len(ranks)
    n_cases = pivot.dropna().shape[0]
    cd = critical_difference(n_methods, n_cases, alpha=alpha)

    # Build cliques: methods within CD of each other are connected.
    methods = ranks.index.tolist()
    cliques: List[List[str]] = []
    for i in range(n_methods):
        clique = [methods[i]]
        for j in range(i + 1, n_methods):
            if abs(ranks[methods[i]] - ranks[methods[j]]) <= cd:
                clique.append(methods[j])
            else:
                break
        if len(clique) > 1:
            cliques.append(clique)

    return {
        "ranks": ranks,
        "cd": cd,
        "n_cases": n_cases,
        "n_methods": n_methods,
        "cliques": cliques,
    }


# ─── High-level summary helper ───────────────────────────────────────────
def summarise(
    df: pd.DataFrame,
    metric: str = "nrmse",
    lower_is_better: bool = True,
    budget_value: Optional[int] = None,
) -> Dict:
    """One-call summary: ranks + Friedman + CD diagram data + per-case best.

    Parameters
    ----------
    df : DataFrame from BenchmarkResult.df
    metric : metric column name to analyse.
    lower_is_better : True for error metrics; False for R².
    budget_value : if given, restrict to one budget.
    """
    pivot = pivot_metric(df, metric, budget_value=budget_value)
    ranks = rank_methods(pivot, lower_is_better=lower_is_better)
    fstat, fp = friedman_test(pivot)
    cd_data = cd_diagram_data(pivot, lower_is_better=lower_is_better)

    # Best method per case
    if lower_is_better:
        best_per_case = pivot.idxmin(axis=1)
    else:
        best_per_case = pivot.idxmax(axis=1)
    win_count = best_per_case.value_counts()

    return {
        "metric": metric,
        "pivot": pivot,
        "ranks": ranks,
        "friedman_stat": fstat,
        "friedman_p": fp,
        "critical_difference": cd_data["cd"],
        "n_cases": cd_data["n_cases"],
        "n_methods": cd_data["n_methods"],
        "win_count": win_count,
        "best_per_case": best_per_case,
    }
