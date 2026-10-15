"""
benchmark.dim_scaling
=====================
Does method ranking depend on problem dimension d?  And does the
power-law convergence rate α (NRMSE ∝ n^{-α}) degrade with d?

These are first-principles questions that go beyond aggregate
ranking — they reveal whether a recommendation generalises.

Key APIs
--------
- ``rank_by_dimension_band(df, bands, budget_value=None)``
    For each dimension band (e.g. low: d ≤ 4; mid: 5–8; high: ≥ 9),
    compute mean rank per method.  Highlights methods whose ranking
    flips with dimension.

- ``alpha_vs_dim(df)``
    Fit power-law convergence rate α per (case, method), then
    correlate with case dimension.  Is α a function of d?

- ``nrmse_vs_dim(df, budget_value)``
    Per-method scatter of NRMSE vs case dimension at a given budget.
    The slope tells you whether NRMSE *grows* with d at a fixed
    budget (curse-of-dimensionality severity).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ─── Dimension bands ─────────────────────────────────────────────────────
def default_dim_bands(df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """Default banding by case dimension, picking thresholds that
    distribute cases as evenly as possible across bands.

    Returns a dict ``{band_name: (lo, hi)}`` (inclusive on both ends).
    """
    if "dim" not in df.columns:
        raise ValueError("df has no 'dim' column; "
                         "annotate with stratified.annotate_results")
    case_dims = (df.drop_duplicates("case_id")[["case_id", "dim"]]
                   .sort_values("dim")["dim"].to_numpy(dtype=int))
    if len(case_dims) <= 1:
        d = int(case_dims[0])
        return {f"d={d}": (d, d)}
    lo, hi = int(case_dims.min()), int(case_dims.max())
    if hi - lo <= 2:
        # Use one band per unique dimension when range is small
        return {f"d={int(d)}": (int(d), int(d)) for d in np.unique(case_dims)}

    # Choose thresholds that balance case counts across 3 bands.
    # Greedy: walk through sorted dims, advance threshold whenever
    # the current band has reached ⌈N/3⌉ cases.
    n = len(case_dims)
    target = int(np.ceil(n / 3))
    cuts = []  # value of dim where each cut occurs
    count = 0
    for i, d in enumerate(case_dims):
        count += 1
        if count >= target and len(cuts) < 2 and i + 1 < n:
            # Place cut between this dim and the next *different* dim
            d_int = int(d)
            d_next = int(case_dims[i + 1])
            if d_next > d_int:
                cuts.append(d_int)
                count = 0
    if len(cuts) == 2:
        c1, c2 = cuts
        return {
            f"low (d≤{c1})":          (lo, c1),
            f"mid ({c1 + 1}–{c2})":   (c1 + 1, c2),
            f"high (d≥{c2 + 1})":     (c2 + 1, hi),
        }
    if len(cuts) == 1:
        c1 = cuts[0]
        return {
            f"low (d≤{c1})":     (lo, c1),
            f"high (d≥{c1 + 1})": (c1 + 1, hi),
        }
    # Fallback: split on median
    med = int(np.median(case_dims))
    return {
        f"low (d≤{med})":      (lo, med),
        f"high (d≥{med + 1})": (med + 1, hi),
    }


# ─── Rank-by-dimension table ─────────────────────────────────────────────
def rank_by_dimension_band(
    df: pd.DataFrame,
    bands: Optional[Dict[str, Tuple[int, int]]] = None,
    budget_value: Optional[int] = None,
    metric: str = "nrmse",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Mean rank per method, broken down by dimension band.

    Returns wide dataframe: index=method, columns=band_name, value=mean rank.
    """
    df = df[df["status"] == "ok"].copy()
    if budget_value is not None:
        df = df[df["budget_mult"] == budget_value]
    if df.empty:
        return pd.DataFrame()

    bands = bands or default_dim_bands(df)

    out: Dict[str, pd.Series] = {}
    for name, (lo, hi) in bands.items():
        sub = df[(df["dim"] >= lo) & (df["dim"] <= hi)]
        if sub.empty:
            continue
        # Pivot to (case, method) → metric, then rank within each case
        pivot = sub.pivot_table(index="case_id", columns="method",
                                values=metric, aggfunc="mean")
        ranks = pivot.rank(axis=1, method="average",
                           ascending=lower_is_better)
        out[name] = ranks.mean(axis=0)

    if not out:
        return pd.DataFrame()
    rk = pd.DataFrame(out)
    rk["range"] = rk.max(axis=1) - rk.min(axis=1)
    rk = rk.sort_values(rk.columns[0])
    return rk


# ─── α (power-law) vs dimension ─────────────────────────────────────────
def alpha_vs_dim(
    df: pd.DataFrame,
    metric: str = "nrmse",
) -> pd.DataFrame:
    """For each (method, case), fit α in NRMSE ∝ n^{-α} and tabulate
    against case dimension.  Returns a DataFrame keyed by method with
    α-statistics binned by dimension.
    """
    from benchmark.learning_curves import fit_power_law, learning_curve_table

    df_ok = df[df["status"] == "ok"].copy()
    if df_ok["budget_mult"].nunique() < 2:
        return pd.DataFrame()
    lc = learning_curve_table(df_ok, metric=metric)
    fits = fit_power_law(lc, metric=metric)
    if fits.empty:
        return pd.DataFrame()
    if "dim" not in fits.columns:
        # Merge dim from df_ok
        case_dim = df_ok.drop_duplicates("case_id")[["case_id", "dim"]]
        fits = fits.merge(case_dim, on="case_id", how="left")
    # Spearman ρ of α vs dim, per method
    rows = []
    for m, sub in fits.groupby("method"):
        if sub["dim"].nunique() < 3:
            continue
        rho, p = spearmanr(sub["dim"], sub["alpha"])
        rows.append({
            "method":      m,
            "n_cases":     len(sub),
            "alpha_mean":  float(sub["alpha"].mean()),
            "alpha_low_d": float(sub.loc[sub["dim"] <= sub["dim"].median(), "alpha"].mean()),
            "alpha_high_d": float(sub.loc[sub["dim"] > sub["dim"].median(), "alpha"].mean()),
            "spearman_rho_alpha_vs_dim": float(rho) if np.isfinite(rho) else float("nan"),
            "p_value": float(p) if np.isfinite(p) else float("nan"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("alpha_mean", ascending=False)


# ─── NRMSE vs dim regression at a given budget ───────────────────────────
def nrmse_vs_dim(
    df: pd.DataFrame,
    budget_value: int,
    metric: str = "nrmse",
) -> pd.DataFrame:
    """For each method, regress log(metric) on dim to get the
    "curse-of-dimensionality severity" slope.

    A high slope means the method's accuracy degrades quickly with
    dimension at the given budget.  Slope ≈ 0 means the method's
    accuracy is approximately dimension-independent (highly desirable).
    """
    df_b = df[(df["status"] == "ok") &
              (df["budget_mult"] == budget_value)].copy()
    if df_b.empty:
        return pd.DataFrame()

    # Per (case, method) mean of metric across repeats
    agg = (df_b.groupby(["case_id", "method", "dim"])[metric]
                .mean()
                .reset_index())

    rows = []
    for m, sub in agg.groupby("method"):
        if sub["dim"].nunique() < 3:
            continue
        x = sub["dim"].to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)
        # Skip pathological cases
        valid = (y > 0) & np.isfinite(y)
        if valid.sum() < 3:
            continue
        x, y = x[valid], y[valid]
        log_y = np.log(y)
        slope, intercept = np.polyfit(x, log_y, 1)
        # R²
        y_hat = slope * x + intercept
        ss_res = float(np.sum((log_y - y_hat) ** 2))
        ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rho, p = spearmanr(x, y)
        rows.append({
            "method":       m,
            "n_cases":      int(valid.sum()),
            "log_slope":    float(slope),
            "intercept":    float(intercept),
            "R2":           float(r2),
            "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
            "p_value":      float(p) if np.isfinite(p) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("log_slope")


__all__ = [
    "alpha_vs_dim",
    "default_dim_bands",
    "nrmse_vs_dim",
    "rank_by_dimension_band",
]
