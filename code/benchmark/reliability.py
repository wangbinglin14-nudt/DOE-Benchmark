r"""
benchmark.reliability
=====================
Failure-probability sub-metrics for the 16 cases that have a stored
``failure_threshold`` and ``failure_direction`` in their CaseMeta.

For each surrogate prediction we compute
  $\hat{P_f}^{\text{surr}}$ = fraction of the test-set predicted to fail
  $\hat{P_f}^{\text{true}}$ = fraction of the test-set true outputs that fail
and report
  - absolute and relative errors
  - confusion-matrix style classification metrics treating failure
    as the "positive" class (precision, recall, F1, balanced accuracy)
  - Brier score on the failure indicator (when ``Y_std`` is available
    and Gaussian)

This is the *reliability quality* axis: an EI-based BO method might
predict means well but be calibrated wrongly near the failure boundary.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

import doe_testbed as db


def get_failure_thresholds(case_id: str) -> List[Tuple[int, float, str]]:
    """Return list of (output_idx, threshold, direction) for one case.

    A case may have multiple outputs with thresholds (e.g.
    ``cantilever_beam_5d`` has σ_max>250 and δ_max>20).  This returns
    one tuple per such output, or an empty list if none.
    """
    try:
        meta = db.get_case_meta(case_id)
    except Exception:
        return []
    out: List[Tuple[int, float, str]] = []
    for j, ov in enumerate(meta.output_variables):
        thr = getattr(ov, "failure_threshold", None)
        dirn = getattr(ov, "failure_direction", None)
        if thr is None or dirn is None:
            continue
        out.append((j, float(thr), str(dirn)))
    return out


def _failure_indicator(Y: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "above":
        return Y > threshold
    if direction == "below":
        return Y < threshold
    raise ValueError(f"unknown direction {direction!r}")


def reliability_metrics(
    case_id: str,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_std: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute failure-probability metrics for all output thresholds of a case.

    Returns a flat dict where keys include the output index, e.g.
    ``"pf_true_o0"``, ``"pf_pred_o0"``, ``"pf_abs_err_o0"``, ...

    Returns an empty dict when the case has no failure thresholds.
    """
    out: Dict[str, float] = {}
    thresholds = get_failure_thresholds(case_id)
    if not thresholds:
        return out

    for j, thr, dirn in thresholds:
        if j >= Y_true.shape[1]:
            continue
        true_fail = _failure_indicator(Y_true[:, j], thr, dirn)
        pred_fail = _failure_indicator(Y_pred[:, j], thr, dirn)

        pf_true = float(np.mean(true_fail))
        pf_pred = float(np.mean(pred_fail))
        out[f"pf_true_o{j}"] = pf_true
        out[f"pf_pred_o{j}"] = pf_pred
        out[f"pf_abs_err_o{j}"] = abs(pf_pred - pf_true)
        if pf_true > 1e-6:
            out[f"pf_rel_err_o{j}"] = abs(pf_pred - pf_true) / pf_true
        else:
            out[f"pf_rel_err_o{j}"] = float("nan")

        # Classification metrics treating failure as positive class
        tp = int(np.sum(true_fail & pred_fail))
        fp = int(np.sum(~true_fail & pred_fail))
        fn = int(np.sum(true_fail & ~pred_fail))
        tn = int(np.sum(~true_fail & ~pred_fail))
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else 0.0
        )
        # Balanced accuracy = (TPR + TNR) / 2.
        #
        # Bal-acc is *undefined* when one of the two truth classes is
        # empty in the test set: with zero true failures the per-positive
        # recall has no denominator and bal-acc collapses to ``0.5 * tnr``
        # which systematically underrates the method even though it
        # correctly classified every safe sample (TNR = 1).  Likewise for
        # the all-failures case.  We emit NaN in either degenerate path
        # so downstream Friedman / rank tests drop the cell instead of
        # treating it as "method got everything wrong" — see GAP audit
        # 2026-05-23 entry "f_bal_acc underrated on degenerate test sets".
        has_pos = (tp + fn) > 0
        has_neg = (tn + fp) > 0
        if has_pos and has_neg:
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            bal_acc = 0.5 * (tpr + tnr)
        else:
            bal_acc = float("nan")
        out[f"f_precision_o{j}"] = float(precision) if not np.isnan(precision) else float("nan")
        out[f"f_recall_o{j}"] = float(recall) if not np.isnan(recall) else float("nan")
        out[f"f_f1_o{j}"] = float(f1)
        out[f"f_bal_acc_o{j}"] = float(bal_acc)

        # Brier score on failure indicator (probabilistic, requires std)
        if Y_std is not None and Y_std.shape == Y_pred.shape:
            from scipy.stats import norm
            std = np.maximum(Y_std[:, j], 1e-9)
            if dirn == "above":
                p_fail = 1.0 - norm.cdf((thr - Y_pred[:, j]) / std)
            else:
                p_fail = norm.cdf((thr - Y_pred[:, j]) / std)
            brier = float(np.mean((p_fail - true_fail.astype(float)) ** 2))
            out[f"f_brier_o{j}"] = brier

    # Global aggregates across all thresholded outputs.  We use NaN-aware
    # reductions because individual ``f_bal_acc_o{j}`` entries can be NaN
    # when output ``j`` of the test set is single-class (see the f_bal_acc
    # branch above).  ``np.mean`` would otherwise propagate NaN and turn
    # the entire global aggregate into NaN even when other outputs are
    # well-defined.
    abs_errs = [v for k, v in out.items() if k.startswith("pf_abs_err_")]
    if abs_errs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out["pf_abs_err_mean"] = float(np.nanmean(abs_errs))
            out["pf_abs_err_max"] = float(np.nanmax(abs_errs))
    bal_accs = [v for k, v in out.items() if k.startswith("f_bal_acc_")]
    if bal_accs:
        finite = [v for v in bal_accs if np.isfinite(v)]
        out["f_bal_acc_mean"] = float(np.mean(finite)) if finite else float("nan")

    return out


__all__ = ["get_failure_thresholds", "reliability_metrics"]
