"""
benchmark.rigour
================
First-principles rigour checks beyond mean-rank analysis.

The flat ranks/Friedman pipeline answers *which method is best on
average*, but it summarises away three things a careful reader will
ask about:

1. **Per-pair dominance** — *On what fraction of cases does method A
   beat method B?*  This is robust to ties and outliers and does not
   depend on metric scale.
2. **Compute-vs-accuracy trade-off** — A "best method" that costs
   100× more is rarely the right recommendation.  We need the Pareto
   front in (accuracy, compute) space.
3. **Effect size** — Even a Friedman-significant difference may be
   practically negligible.  Cliff's delta is a non-parametric
   complement that quantifies *how much* better A is than B.

Plus three test-fidelity diagnostics:

4. **Test-set MC error** — Estimates the MC noise of NRMSE for a
   given ``n_test``, so we know whether the comparison is
   test-noise-limited.
5. **Fit-quality flags** — Heuristics that flag degenerate fits
   (constant predictions, numerical blow-up, training-set memorisation).
6. **Per-output disaggregation** — For multi-output cases, the
   mean-NRMSE can hide a method that is great on output-0 and awful
   on output-1.  We report both per-output and worst-output metrics.

These checks are computed *post-hoc* from the saved CSV so existing
benchmark runs benefit without re-running.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from benchmark.analysis import pivot_metric


# ─── Pairwise win-rate ────────────────────────────────────────────────────
def pairwise_win_rate_matrix(
    df: pd.DataFrame,
    metric: str = "nrmse",
    lower_is_better: bool = True,
    budget_value: Optional[int] = None,
) -> pd.DataFrame:
    """For every (method_A, method_B) pair, the fraction of cases where A beats B.

    Tied cases are counted as 0.5 to A and 0.5 to B.

    Returns a square DataFrame indexed *and* columned by method name.
    Diagonal is 0.5 by convention.
    """
    pivot = pivot_metric(df, metric, budget_value=budget_value)
    methods = pivot.columns.tolist()
    n_methods = len(methods)
    arr = np.full((n_methods, n_methods), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                arr[i, j] = 0.5
                continue
            valid = pivot[[a, b]].dropna()
            if len(valid) == 0:
                continue
            if lower_is_better:
                wins = (valid[a] < valid[b]).sum()
                ties = (valid[a] == valid[b]).sum()
            else:
                wins = (valid[a] > valid[b]).sum()
                ties = (valid[a] == valid[b]).sum()
            arr[i, j] = (wins + 0.5 * ties) / len(valid)
    return pd.DataFrame(arr, index=methods, columns=methods)


def dominance_score(win_matrix: pd.DataFrame) -> pd.Series:
    """Per-method dominance = mean win-rate against every other method.

    Range [0, 1]; 0.5 = average; > 0.5 = beats more methods than it loses to.
    """
    # Subtract 0.5 self-entry to avoid bias (treats diagonal correctly)
    methods = win_matrix.index.tolist()
    n = len(methods)
    if n <= 1:
        return pd.Series([0.5] * n, index=methods)
    score = (win_matrix.sum(axis=1) - 0.5) / (n - 1)
    return score.sort_values(ascending=False)


# ─── Effect size (Cliff's delta) ─────────────────────────────────────────
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta non-parametric effect size.

    delta = P(A < B) - P(A > B), in [-1, 1].
    Convention here: positive when A is *better* (lower error) than B.

    |delta| < 0.147   → negligible
    |delta| < 0.330   → small
    |delta| < 0.474   → medium
    |delta| ≥ 0.474   → large
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    n_lt = np.sum(diff < 0)
    n_gt = np.sum(diff > 0)
    n_total = len(a) * len(b)
    return float((n_lt - n_gt) / n_total)


def cliffs_delta_matrix(
    df: pd.DataFrame,
    metric: str = "nrmse",
    budget_value: Optional[int] = None,
) -> pd.DataFrame:
    """Pairwise Cliff's delta across cases (treating each case as one observation).

    Note: for one observation per case this collapses to the
    pairwise win-rate (it is then just the sign of the difference);
    the meaningful application is over the full ``case × repeat``
    population without collapsing repeats.
    """
    if budget_value is not None and "budget_mult" in df.columns:
        df = df[df["budget_mult"] == budget_value]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    methods = sorted(df["method"].unique())
    arr = np.full((len(methods), len(methods)), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                arr[i, j] = 0.0
                continue
            va = df.loc[df["method"] == a, metric].to_numpy()
            vb = df.loc[df["method"] == b, metric].to_numpy()
            arr[i, j] = cliffs_delta(va, vb)
    return pd.DataFrame(arr, index=methods, columns=methods)


def cliffs_delta_ci_matrix(
    df: pd.DataFrame,
    metric: str = "nrmse",
    budget_value: Optional[int] = None,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, pd.DataFrame]:
    """Pairwise Cliff's delta + 95 % bootstrap CI matrix.

    For each method pair we draw ``n_boot`` paired bootstrap resamples
    of the case set (keeping the case-pair structure intact so the
    bootstrap respects the dependency between methods on the same
    case).  This gives a CI on the effect size that accompanies the
    Holm-Bonferroni significance test.

    Returns a dict with keys:
      - ``"delta"``                — point estimate (M × M)
      - ``"ci_lo"``                — bootstrap lower limit (M × M)
      - ``"ci_hi"``                — bootstrap upper limit (M × M)
      - ``"non_negligible"``       — boolean: |delta| ≥ 0.147 *and* CI excludes 0
      - ``"magnitude"``            — categorical {'negligible',
        'small', 'medium', 'large'} based on the *lower bound* of |δ|
        (more conservative than the point estimate)
    """
    if budget_value is not None and "budget_mult" in df.columns:
        df = df[df["budget_mult"] == budget_value]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    methods = sorted(df["method"].unique())
    M = len(methods)
    rng = np.random.default_rng(seed)

    # Build per-method arrays once for speed
    per_method = {m: df.loc[df["method"] == m, metric].to_numpy() for m in methods}

    delta = np.zeros((M, M))
    lo = np.zeros((M, M))
    hi = np.zeros((M, M))
    alpha = (1.0 - ci) / 2.0

    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                continue
            va = per_method[a]
            vb = per_method[b]
            if len(va) == 0 or len(vb) == 0:
                delta[i, j] = lo[i, j] = hi[i, j] = float("nan")
                continue
            delta[i, j] = cliffs_delta(va, vb)
            # Independent bootstrap on each marginal; the methods'
            # values come from different runs so the marginal samples
            # are independent of one another.
            boot_d = np.empty(n_boot)
            for k in range(n_boot):
                ra = rng.integers(0, len(va), size=len(va))
                rb = rng.integers(0, len(vb), size=len(vb))
                boot_d[k] = cliffs_delta(va[ra], vb[rb])
            lo[i, j] = float(np.quantile(boot_d, alpha))
            hi[i, j] = float(np.quantile(boot_d, 1.0 - alpha))

    delta_df = pd.DataFrame(delta, index=methods, columns=methods)
    lo_df = pd.DataFrame(lo, index=methods, columns=methods)
    hi_df = pd.DataFrame(hi, index=methods, columns=methods)

    # CI excludes 0 ⇒ statistically distinguishable
    ci_excludes_zero = (lo_df > 0) | (hi_df < 0)
    nn_arr = ((delta_df.abs() >= 0.147) & ci_excludes_zero).to_numpy().copy()
    np.fill_diagonal(nn_arr, False)
    non_negligible = pd.DataFrame(nn_arr, index=methods, columns=methods)

    # Magnitude based on the more *conservative* CI bound
    abs_lo = np.minimum(np.abs(lo_df), np.abs(hi_df))
    mag = pd.DataFrame("negligible", index=methods, columns=methods, dtype=object)
    mag = mag.mask(abs_lo >= 0.147, "small")
    mag = mag.mask(abs_lo >= 0.330, "medium")
    mag = mag.mask(abs_lo >= 0.474, "large")
    mag_arr = mag.to_numpy(dtype=object).copy()
    np.fill_diagonal(mag_arr, "self")
    mag = pd.DataFrame(mag_arr, index=methods, columns=methods)

    return {
        "delta": delta_df,
        "ci_lo": lo_df,
        "ci_hi": hi_df,
        "non_negligible": non_negligible,
        "magnitude": mag,
    }


# ─── Compute-vs-accuracy Pareto ──────────────────────────────────────────
def pareto_front(
    df: pd.DataFrame,
    metric: str = "nrmse",
    cost_col: str = "t_total_s",
    budget_value: Optional[int] = None,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Aggregate per-method (mean metric, mean cost) and return Pareto-optimal subset.

    Returns a DataFrame with columns:
      method, mean_metric, mean_cost, on_pareto (bool).
    """
    if budget_value is not None and "budget_mult" in df.columns:
        df = df[df["budget_mult"] == budget_value]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    agg = df.groupby("method").agg(
        mean_metric=(metric, "mean"),
        median_metric=(metric, "median"),
        mean_cost=(cost_col, "mean"),
        median_cost=(cost_col, "median"),
        n=(metric, "count"),
    ).reset_index()

    # Pareto front: minimize (metric if lower_is_better else -metric) AND cost
    if not lower_is_better:
        agg["__metric_sign"] = -agg["mean_metric"]
    else:
        agg["__metric_sign"] = agg["mean_metric"]
    agg = agg.sort_values("mean_cost").reset_index(drop=True)
    pareto = []
    best = float("inf")
    for _, row in agg.iterrows():
        v = row["__metric_sign"]
        if v < best:
            best = v
            pareto.append(True)
        else:
            pareto.append(False)
    agg["on_pareto"] = pareto
    return agg.drop(columns=["__metric_sign"]).sort_values("mean_metric")


# ─── Fit-quality flags ───────────────────────────────────────────────────
def fit_quality_flags(
    df: pd.DataFrame,
    metric: str = "nrmse",
) -> pd.DataFrame:
    """Heuristic flags for degenerate or anomalous fits.

    Returns a DataFrame with one row per (method, case, budget) and
    a column ``flags`` containing a comma-separated list of issues.

    Heuristics:
      - ``nrmse_blowup``     : NRMSE > 5 (model worse than constant)
      - ``nrmse_perfect``    : NRMSE < 1e-4 (likely interpolated test set)
      - ``r2_negative_bad``  : R² < -1 (worse than mean predictor)
      - ``high_var_across_reps`` : std-NRMSE across reps > mean-NRMSE
      - ``picp_outside``     : PICP-95 outside [0.7, 0.99] (badly calibrated)
    """
    if "status" in df.columns:
        ok = df[df["status"] == "ok"].copy()
    else:
        ok = df.copy()

    # Aggregate per (method, case, budget)
    grp = ok.groupby(["method", "case_id", "budget_mult"])
    summary = grp.agg(
        nrmse_mean=(metric, "mean"),
        nrmse_std=(metric, "std"),
        nrmse_max=(metric, "max"),
        r2_mean=("r2", "mean") if "r2" in ok.columns else (metric, "size"),
        picp_mean=("picp_95", "mean") if "picp_95" in ok.columns else (metric, "size"),
        n_reps=(metric, "count"),
    ).reset_index()

    flags_list = []
    for _, row in summary.iterrows():
        flags = []
        if not np.isnan(row["nrmse_max"]) and row["nrmse_max"] > 5:
            flags.append("nrmse_blowup")
        if not np.isnan(row["nrmse_mean"]) and row["nrmse_mean"] < 1e-4:
            flags.append("nrmse_perfect")
        if "r2" in ok.columns and not np.isnan(row["r2_mean"]) and row["r2_mean"] < -1:
            flags.append("r2_negative_bad")
        if (not np.isnan(row.get("nrmse_std", np.nan))
                and row["nrmse_std"] > row["nrmse_mean"]):
            flags.append("high_var_across_reps")
        if "picp_95" in ok.columns:
            picp = row["picp_mean"]
            if not np.isnan(picp) and (picp < 0.7 or picp > 0.99):
                flags.append("picp_outside")
        flags_list.append(",".join(flags) if flags else "ok")
    summary["flags"] = flags_list
    return summary


# ─── Test-set MC error ───────────────────────────────────────────────────
def bootstrap_metric_ci(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap CI on NRMSE / RMSE estimates from one fit's test predictions.

    Use this to verify ``n_test`` is large enough that test-set MC
    noise is small relative to between-method differences.

    Returns dict with: nrmse_mean, nrmse_ci_low, nrmse_ci_high,
    nrmse_relative_ci_width.
    """
    rng = np.random.default_rng(seed)
    n_test = len(Y_true)
    if Y_true.ndim == 2 and Y_true.shape[1] == 1:
        Y_true = Y_true.ravel()
        Y_pred = Y_pred.ravel()
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_test, size=n_test)
        yt = Y_true[idx] if Y_true.ndim == 1 else Y_true[idx, :]
        yp = Y_pred[idx] if Y_pred.ndim == 1 else Y_pred[idx, :]
        sse = np.sum((yt - yp) ** 2)
        var = np.sum((yt - yt.mean(axis=0)) ** 2)
        if var > 0:
            boot[b] = float(np.sqrt(sse / var))
        else:
            boot[b] = float("nan")
    nrmse = boot[~np.isnan(boot)]
    mean = float(np.mean(nrmse))
    lo = float(np.percentile(nrmse, 2.5))
    hi = float(np.percentile(nrmse, 97.5))
    return {
        "nrmse_mean": mean,
        "nrmse_ci_low": lo,
        "nrmse_ci_high": hi,
        "nrmse_ci_width": hi - lo,
        "nrmse_relative_ci_width": (hi - lo) / mean if mean > 0 else float("nan"),
    }


# ─── Per-output metrics (multi-output disaggregation) ────────────────────
def per_output_nrmse(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> np.ndarray:
    """Compute NRMSE for each output column separately."""
    if Y_true.ndim == 1:
        Y_true = Y_true[:, None]
        Y_pred = Y_pred[:, None]
    n_out = Y_true.shape[1]
    nrmse = np.zeros(n_out)
    for j in range(n_out):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        sse = np.sum((yt - yp) ** 2)
        var = np.sum((yt - yt.mean()) ** 2)
        nrmse[j] = float(np.sqrt(sse / var)) if var > 0 else float("nan")
    return nrmse


def per_output_summary(
    df: pd.DataFrame,
    budget_value: Optional[int] = None,
    metric_prefix: str = "nrmse_out",
) -> pd.DataFrame:
    """Per-method × per-output mean error from a benchmark CSV.

    Looks for columns named ``{metric_prefix}{j}`` for j = 0, 1, 2 ...
    and aggregates over (case, repeat) to give one row per method.

    Returns a DataFrame indexed by method, columns are
    ``out0_mean``, ``out1_mean``, ``out2_mean``, ``mean_avg``,
    ``mean_worst``, ``mean_spread``.  ``mean_worst`` is the
    practitioner-relevant guarantee: even on the worst output the
    method delivers this NRMSE.
    """
    if budget_value is not None and "budget_mult" in df.columns:
        df = df[df["budget_mult"] == budget_value]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    out_cols = [c for c in df.columns if c.startswith(metric_prefix)]
    if not out_cols:
        return pd.DataFrame()

    grp = df.groupby("method")[out_cols].mean()
    grp.columns = [f"{c}_mean" for c in grp.columns]
    # Aggregate stats per method (worst across outputs)
    grp_arr = grp.to_numpy()
    grp["mean_avg"] = np.nanmean(grp_arr, axis=1)
    grp["mean_worst"] = np.nanmax(grp_arr, axis=1)
    grp["mean_best"] = np.nanmin(grp_arr, axis=1)
    grp["mean_spread"] = grp["mean_worst"] - grp["mean_best"]
    return grp.sort_values("mean_worst").reset_index()


def per_output_rank_shifts(
    df: pd.DataFrame,
    budget_value: Optional[int] = None,
    metric_prefix: str = "nrmse_out",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """For each output dimension, compute the per-method mean rank.

    Returns a DataFrame with columns:
      method, rank_avg (the standard mean rank on the average metric),
      rank_outJ (mean rank on output J), rank_max (worst rank across
      outputs), rank_shift (rank_max - rank_avg).

    A large positive ``rank_shift`` means the method is much worse on
    its worst output than the headline rank suggests.
    """
    if budget_value is not None and "budget_mult" in df.columns:
        df = df[df["budget_mult"] == budget_value]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    out_cols = sorted([c for c in df.columns if c.startswith(metric_prefix)])
    if not out_cols:
        return pd.DataFrame()

    methods = sorted(df["method"].unique())
    rows: List[Dict] = []

    # Mean-aggregated metric (standard headline)
    avg_pivot = df.pivot_table(
        index="case_id", columns="method", values="nrmse", aggfunc="mean",
    )
    avg_ranks = avg_pivot.rank(axis=1, ascending=lower_is_better).mean(axis=0)

    for m in methods:
        row = {"method": m, "rank_avg": float(avg_ranks.get(m, float("nan")))}
        out_ranks = []
        for col in out_cols:
            piv = df.pivot_table(
                index="case_id", columns="method", values=col, aggfunc="mean",
            )
            r = piv.rank(axis=1, ascending=lower_is_better).mean(axis=0)
            row[f"rank_{col.replace(metric_prefix, 'out')}"] = float(r.get(m, float("nan")))
            out_ranks.append(float(r.get(m, float("nan"))))
        finite = [v for v in out_ranks if not np.isnan(v)]
        if finite:
            row["rank_max"] = max(finite)
            row["rank_min"] = min(finite)
            row["rank_shift"] = row["rank_max"] - row["rank_avg"]
        else:
            row["rank_max"] = row["rank_min"] = row["rank_shift"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rank_shift", ascending=False)


# ─── Pairwise signed-rank test with Holm-Bonferroni correction ───────────
def pairwise_signed_rank_test(
    df: pd.DataFrame,
    metric: str = "nrmse",
    budget_value: Optional[int] = None,
    alpha: float = 0.05,
    lower_is_better: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.

    For every pair of methods (M*(M-1)/2 pairs), a paired Wilcoxon
    signed-rank test on per-case mean ``metric`` values is computed.
    P-values are then corrected with the Holm-Bonferroni procedure
    to control family-wise error rate at ``alpha``.

    Compared to the Nemenyi CD test (used for the CD diagram), this
    is more powerful for moderate sample sizes (< 30 cases) because
    it uses paired-difference information instead of average ranks.

    Returns
    -------
    A dict with keys:
      - ``"p_raw"``       : unadjusted p-values (M × M matrix)
      - ``"p_adjusted"``  : Holm-corrected p-values (M × M matrix)
      - ``"significant"`` : boolean M × M matrix (True ⇒ different at α)
      - ``"summary"``     : long-form DataFrame of pair-wise results
    """
    from scipy.stats import wilcoxon

    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    if budget_value is not None:
        df = df[df["budget_mult"] == budget_value]
    if df.empty:
        return {}

    # Average over repeats per (case, method)
    pivot = df.pivot_table(index="case_id", columns="method",
                            values=metric, aggfunc="mean")
    pivot = pivot.dropna(how="any")
    methods = list(pivot.columns)
    M = len(methods)
    if M < 2 or len(pivot) < 2:
        return {}

    # Compute raw p-values for every pair
    pair_rows: list = []
    p_raw = pd.DataFrame(np.eye(M), index=methods, columns=methods)
    for i in range(M):
        for j in range(i + 1, M):
            a = pivot[methods[i]].to_numpy()
            b = pivot[methods[j]].to_numpy()
            diff = a - b
            if np.allclose(diff, 0):
                p = 1.0
                stat = 0.0
            else:
                try:
                    stat, p = wilcoxon(a, b, zero_method="wilcox",
                                        alternative="two-sided")
                except ValueError:
                    p = 1.0
                    stat = 0.0
            p_raw.iat[i, j] = p
            p_raw.iat[j, i] = p
            mean_diff = float(np.mean(diff))
            pair_rows.append({
                "method_A":   methods[i],
                "method_B":   methods[j],
                "n_cases":    int(len(diff)),
                "mean_diff":  mean_diff,
                "median_diff": float(np.median(diff)),
                "stat":       float(stat),
                "p_raw":      float(p),
            })

    # Holm-Bonferroni correction across all pairs
    pairs_sorted = sorted(pair_rows, key=lambda r: r["p_raw"])
    n_pairs = len(pairs_sorted)
    p_adj_lookup: Dict[Tuple[str, str], float] = {}
    running_max = 0.0
    for k, row in enumerate(pairs_sorted):
        adj = min(1.0, row["p_raw"] * (n_pairs - k))
        running_max = max(running_max, adj)
        p_adj_lookup[(row["method_A"], row["method_B"])] = running_max
        row["p_holm"] = running_max
        row["significant_holm"] = running_max < alpha

    p_adj_arr = np.zeros((M, M), dtype=float)
    sig_arr = np.zeros((M, M), dtype=bool)
    for (a, b), p in p_adj_lookup.items():
        ia, ib = methods.index(a), methods.index(b)
        p_adj_arr[ia, ib] = p
        p_adj_arr[ib, ia] = p
        is_sig = bool(p < alpha)
        sig_arr[ia, ib] = is_sig
        sig_arr[ib, ia] = is_sig
    p_adj = pd.DataFrame(p_adj_arr, index=methods, columns=methods)
    sig = pd.DataFrame(sig_arr, index=methods, columns=methods)

    summary = pd.DataFrame(pair_rows).sort_values("p_holm").reset_index(drop=True)
    return {
        "p_raw":       p_raw,
        "p_adjusted":  p_adj,
        "significant": sig,
        "summary":     summary,
    }


# ─── Tail-risk metric: failure rate ──────────────────────────────────────
def failure_rate_per_method(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """For each method, fraction of (case, repeat) runs that crashed or were degenerate.

    A "failure" here is broader than `status='failed'` — it also flags
    runs whose NRMSE > 5 (worse than naive constant predictor).
    """
    if "status" not in df.columns:
        df = df.copy()
        df["status"] = "ok"
    df = df.copy()
    df["is_crash"] = df["status"] != "ok"
    df["is_blowup"] = (df["status"] == "ok") & (df.get("nrmse", 0).fillna(0) > 5)
    summary = df.groupby("method").agg(
        n_runs=("status", "size"),
        n_crash=("is_crash", "sum"),
        n_blowup=("is_blowup", "sum"),
    ).reset_index()
    summary["crash_rate"] = summary["n_crash"] / summary["n_runs"]
    summary["blowup_rate"] = summary["n_blowup"] / summary["n_runs"]
    summary["total_failure_rate"] = (summary["n_crash"] + summary["n_blowup"]) / summary["n_runs"]
    return summary.sort_values("total_failure_rate", ascending=True)


__all__ = [
    "cliffs_delta",
    "cliffs_delta_matrix",
    "cliffs_delta_ci_matrix",
    "dominance_score",
    "failure_rate_per_method",
    "fit_quality_flags",
    "pairwise_win_rate_matrix",
    "pareto_front",
    "per_output_nrmse",
    "per_output_summary",
    "per_output_rank_shifts",
    "bootstrap_metric_ci",
]
