"""
benchmark.metrics
=================
Standardised evaluation metrics for the DOE benchmark.

Each metric accepts ``Y_true``, ``Y_pred`` (shape ``(n, m)``), and
optionally ``Y_std``, and returns either:
  - a scalar (averaged over outputs), or
  - a per-output array.

The :func:`compute_all_metrics` helper computes the full metric suite
in one pass and returns a flat dict suitable for inclusion in a
benchmark results table.

Core metrics
------------
- ``rmse``   : root-mean-square error
- ``nrmse``  : RMSE normalised by the sample std(Y_true).  With the
               ``ddof=1`` convention used here,
               ``nrmse = sqrt((n-1)/n * (1 - R²))`` for non-constant outputs;
               it is not ``1 - sqrt(R²)``.
- ``mae``    : mean absolute error
- ``r2``     : coefficient of determination
- ``maxae``  : max absolute error (worst-case)
- ``q95ae``  : 95-th percentile absolute error
- ``nll``    : Gaussian negative log-likelihood (requires Y_std)
- ``picp_95``: prediction-interval coverage probability at 95 %
- ``mpiw_95``: mean prediction interval width
- ``kendall_tau_st`` : Kendall τ between true and predicted variable importances
                       (computed elsewhere via Sobol / regression coefficients)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

# ─── Per-output primitives ────────────────────────────────────────────────


def _rmse_per_output(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))


def _mae_per_output(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(Y_true - Y_pred), axis=0)


def _maxae_per_output(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    return np.max(np.abs(Y_true - Y_pred), axis=0)


def _q95ae_per_output(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    return np.quantile(np.abs(Y_true - Y_pred), 0.95, axis=0)


def _r2_per_output(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """R² = 1 - SS_res / SS_tot.  Negative when prediction is worse than mean."""
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    y_mean = np.mean(Y_true, axis=0, keepdims=True)
    ss_tot = np.sum((Y_true - y_mean) ** 2, axis=0)
    # Avoid divide-by-zero when an output is constant
    out = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)
    return out


def _nrmse_per_output(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """RMSE normalised by std(Y_true).  Output: scale-free, comparable across cases."""
    rmse = _rmse_per_output(Y_true, Y_pred)
    sd = np.std(Y_true, axis=0, ddof=1)
    return np.where(sd > 0, rmse / np.where(sd > 0, sd, 1.0), np.nan)


def _nll_per_output(
    Y_true: np.ndarray, Y_pred: np.ndarray, Y_std: np.ndarray
) -> np.ndarray:
    """Gaussian negative log-likelihood per output (lower is better).

    nll_j = mean_i [ 0.5 * log(2π σ_ij²) + 0.5 * (y_ij - ŷ_ij)² / σ_ij² ]
    """
    var = np.maximum(Y_std ** 2, 1e-12)
    diff2 = (Y_true - Y_pred) ** 2
    log_term = 0.5 * np.log(2.0 * np.pi * var)
    quad_term = 0.5 * diff2 / var
    return np.mean(log_term + quad_term, axis=0)


def _picp_per_output(
    Y_true: np.ndarray, Y_pred: np.ndarray, Y_std: np.ndarray, alpha: float = 0.95
) -> np.ndarray:
    """Prediction-interval coverage probability at level ``alpha``."""
    z = float(np.abs(np.quantile(np.random.default_rng(0).standard_normal(int(1e6)),
                                  (1 + alpha) / 2)))
    # Use stable analytic z-score instead
    from scipy.stats import norm
    z = float(norm.ppf((1 + alpha) / 2))
    lower = Y_pred - z * Y_std
    upper = Y_pred + z * Y_std
    inside = (Y_true >= lower) & (Y_true <= upper)
    return np.mean(inside, axis=0)


def _mpiw_per_output(
    Y_pred: np.ndarray, Y_std: np.ndarray, alpha: float = 0.95
) -> np.ndarray:
    """Mean prediction-interval width at level ``alpha``."""
    from scipy.stats import norm
    z = float(norm.ppf((1 + alpha) / 2))
    return np.mean(2.0 * z * Y_std, axis=0)


# ─── Aggregation: average over outputs ────────────────────────────────────
def _scalar_avg(arr: np.ndarray) -> float:
    """Mean over outputs, ignoring NaN."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


# ─── Public scalar metrics ────────────────────────────────────────────────
def rmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return _scalar_avg(_rmse_per_output(Y_true, Y_pred))


def nrmse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return _scalar_avg(_nrmse_per_output(Y_true, Y_pred))


def mae(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return _scalar_avg(_mae_per_output(Y_true, Y_pred))


def maxae(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return _scalar_avg(_maxae_per_output(Y_true, Y_pred))


def q95ae(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return _scalar_avg(_q95ae_per_output(Y_true, Y_pred))


def r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return _scalar_avg(_r2_per_output(Y_true, Y_pred))


# Registry for programmatic access
METRIC_REGISTRY: Dict[str, Callable] = {
    "rmse": rmse,
    "nrmse": nrmse,
    "mae": mae,
    "maxae": maxae,
    "q95ae": q95ae,
    "r2": r2,
}


# ─── Top-level helper ─────────────────────────────────────────────────────
def compute_all_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_std: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the full metric suite and return a flat dict.

    Both *averaged-over-outputs* scalars and *per-output* values are
    included; the latter are keyed as ``f"{metric}_out{j}"``.
    """
    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            f"shape mismatch: Y_true {Y_true.shape}, Y_pred {Y_pred.shape}"
        )

    out: Dict[str, float] = {}

    # Per-output
    rmse_v = _rmse_per_output(Y_true, Y_pred)
    nrmse_v = _nrmse_per_output(Y_true, Y_pred)
    mae_v = _mae_per_output(Y_true, Y_pred)
    maxae_v = _maxae_per_output(Y_true, Y_pred)
    q95ae_v = _q95ae_per_output(Y_true, Y_pred)
    r2_v = _r2_per_output(Y_true, Y_pred)

    # Scalar summaries (mean over outputs)
    out["rmse"] = _scalar_avg(rmse_v)
    out["nrmse"] = _scalar_avg(nrmse_v)
    out["mae"] = _scalar_avg(mae_v)
    out["maxae"] = _scalar_avg(maxae_v)
    out["q95ae"] = _scalar_avg(q95ae_v)
    out["r2"] = _scalar_avg(r2_v)

    # Worst-output / best-output aggregations (NaN-aware) — these answer
    # "what is the bound on the *worst* output the method handles?".
    # A method with low *mean* NRMSE but high *worst* NRMSE is dangerous
    # for a practitioner who can't ignore any output dimension.
    if np.any(np.isfinite(nrmse_v)):
        out["nrmse_worst"] = float(np.nanmax(nrmse_v))
        out["nrmse_best"] = float(np.nanmin(nrmse_v))
        out["nrmse_spread"] = out["nrmse_worst"] - out["nrmse_best"]
    else:
        out["nrmse_worst"] = float("nan")
        out["nrmse_best"] = float("nan")
        out["nrmse_spread"] = float("nan")
    if np.any(np.isfinite(rmse_v)):
        out["rmse_worst"] = float(np.nanmax(rmse_v))
    else:
        out["rmse_worst"] = float("nan")
    if np.any(np.isfinite(r2_v)):
        out["r2_worst"] = float(np.nanmin(r2_v))  # worst R² = lowest
    else:
        out["r2_worst"] = float("nan")

    # Per-output (keep only first 3 to limit columns; most cases have m=3)
    for j in range(min(Y_true.shape[1], 3)):
        out[f"rmse_out{j}"] = float(rmse_v[j])
        out[f"nrmse_out{j}"] = float(nrmse_v[j])
        out[f"r2_out{j}"] = float(r2_v[j])

    # Uncertainty-quality metrics (only if std is provided)
    if Y_std is not None and Y_std.shape == Y_pred.shape:
        nll_v = _nll_per_output(Y_true, Y_pred, Y_std)
        picp_v = _picp_per_output(Y_true, Y_pred, Y_std, alpha=0.95)
        mpiw_v = _mpiw_per_output(Y_pred, Y_std, alpha=0.95)
        out["nll"] = _scalar_avg(nll_v)
        out["picp_95"] = _scalar_avg(picp_v)
        out["mpiw_95"] = _scalar_avg(mpiw_v)
    else:
        out["nll"] = float("nan")
        out["picp_95"] = float("nan")
        out["mpiw_95"] = float("nan")

    return out


# ─── Failure-probability metrics (case-aware) ────────────────────────────
def failure_probability_error(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    threshold: float,
    direction: str,
) -> Dict[str, float]:
    """Compare empirical failure probabilities of predictions vs truth.

    Parameters
    ----------
    threshold : float
    direction : 'above' or 'below'
    """
    if direction == "above":
        true_fail = Y_true > threshold
        pred_fail = Y_pred > threshold
    elif direction == "below":
        true_fail = Y_true < threshold
        pred_fail = Y_pred < threshold
    else:
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
    pf_true = float(np.mean(true_fail))
    pf_pred = float(np.mean(pred_fail))
    return {
        "pf_true": pf_true,
        "pf_pred": pf_pred,
        "pf_abs_error": abs(pf_true - pf_pred),
        "pf_rel_error": (
            abs(pf_true - pf_pred) / max(pf_true, 1e-6)
            if pf_true > 0
            else float("nan")
        ),
    }
