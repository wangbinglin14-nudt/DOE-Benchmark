"""
benchmark.sensitivity_recovery
================================
Did the surrogate correctly identify *which variables matter*?

This module tests a fundamentally different aspect of method quality
than NRMSE: a surrogate may have low prediction error on average yet
attribute the variance to the wrong inputs.  Engineering decisions
informed by a surrogate (e.g. tolerance allocation, what-if studies)
depend critically on variable importance — so this is a first-class
quality criterion.

Approach
--------
For each (case, method, budget):

  1. Fit the surrogate on a design produced by the method.
  2. Generate Saltelli samples in [0, 1]^d (SALib).
  3. Evaluate the surrogate on those samples.
  4. Compute total-order Sobol indices ``S_T`` from the surrogate
     predictions.
  5. Compare with the ground-truth ``sobol_ST`` from
     ``doe_testbed`` metadata using:

     - Kendall τ between *rankings*
     - Spearman ρ between *values*
     - Top-k Jaccard overlap (top-1, top-3)
     - L1 / L2 distance between value vectors

Outputs
-------
A long-form DataFrame with columns ``case_id``, ``method``,
``output_idx``, ``kendall_tau``, ``spearman_rho``, ``top1_correct``,
``top3_jaccard``, ``L1``, ``L2``, ``n_inputs``.

Limitations
-----------
- Sobol on the surrogate measures recovery of *the surrogate's*
  variance attribution, not of the true function's.  Consistent with
  ground truth only if the surrogate is accurate.  We report this
  alongside NRMSE so the reader can see both.
- Saltelli sampling needs ``N(2D+2)`` evaluations.  With N = 256 and
  D ≤ 10, this is ~5,000 cheap surrogate calls.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import doe_testbed as db


# ─── Sobol on a fitted surrogate ─────────────────────────────────────────
def surrogate_sobol_indices(
    surrogate: Any,
    n_inputs: int,
    n_base: int = 256,
    output_idx: int = 0,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Compute total-order Sobol indices ``S_T`` on a fitted surrogate.

    Parameters
    ----------
    surrogate : object with ``predict(X) -> (mean, std)``
    n_inputs  : input dimension d
    n_base    : N in SALib's saltelli sampler; total samples = N·(2d+2)
    output_idx: which output column of the surrogate to analyse
    seed      : RNG seed for reproducibility

    Returns
    -------
    np.ndarray of shape (n_inputs,) with total-order Sobol indices,
    or None if the computation fails.
    """
    try:
        from SALib.analyze import sobol as sobol_an
        from SALib.sample import saltelli
    except ImportError:
        warnings.warn("SALib not installed; skipping sensitivity recovery")
        return None

    problem = {
        "num_vars": n_inputs,
        "names": [f"x{i}" for i in range(n_inputs)],
        "bounds": [[0.0, 1.0]] * n_inputs,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Saltelli's sampler is deterministic (Sobol' sequence) — no seed arg
        X = saltelli.sample(problem, n_base, calc_second_order=False)

    # Predict in batches to limit memory
    Y_pred, _ = surrogate.predict(X)
    if Y_pred.ndim == 1:
        y = Y_pred
    else:
        y = Y_pred[:, output_idx]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sobol_an.analyze(problem, y, calc_second_order=False, seed=seed)
        st = np.asarray(res["ST"], dtype=float)
        # Clamp negative spurious values to 0 (Monte Carlo noise floor)
        st = np.clip(st, 0.0, None)
        return st
    except Exception as exc:
        warnings.warn(f"Sobol analyze failed: {exc}")
        return None


def surrogate_response_range(
    surrogate: Any,
    n_inputs: int,
    output_idx: int = 0,
    n_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Diagnose if a fitted surrogate has *collapsed* to a near-constant.

    Useful as a sanity check before trusting Sobol indices: a near-
    constant surrogate has near-zero output variance and any Sobol
    estimate becomes pure noise.

    Returns ``{"y_std": float, "y_range": float, "is_flat": bool}``.
    A surrogate is considered "flat" when its predicted standard
    deviation on a uniform sample is below 1 % of the training
    standard deviation passed in via ``train_y_std`` (callers should
    compare this themselves).
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_inputs))
    Y_pred, _ = surrogate.predict(X)
    if Y_pred.ndim == 1:
        y = Y_pred
    else:
        y = Y_pred[:, output_idx]
    y = np.asarray(y, dtype=float)
    return {
        "y_std": float(np.std(y, ddof=1)),
        "y_range": float(np.ptp(y)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
    }


# ─── Compare predicted vs ground-truth Sobol vectors ─────────────────────
def compare_sensitivity(
    true_st: np.ndarray,
    pred_st: np.ndarray,
) -> Dict[str, float]:
    """Compare two Sobol-ST vectors element-wise.

    Returns a dict with:
      - kendall_tau  : rank correlation
      - spearman_rho : monotone correlation
      - top1_correct : 1.0 if argmax matches, else 0.0
      - top3_jaccard : Jaccard index of top-3 sets
      - L1           : sum |t - p|
      - L2           : sqrt sum (t - p)²
      - argmax_true  : index of the most-important variable in truth
      - argmax_pred  : index of the most-important variable in prediction
    """
    true_st = np.asarray(true_st, dtype=float)
    pred_st = np.asarray(pred_st, dtype=float)
    if true_st.shape != pred_st.shape:
        raise ValueError("true_st and pred_st must have same shape")
    if len(true_st) < 2:
        return {"kendall_tau": float("nan"), "spearman_rho": float("nan"),
                "top1_correct": float("nan"), "top3_jaccard": float("nan"),
                "L1": float("nan"), "L2": float("nan"),
                "argmax_true": -1, "argmax_pred": -1}

    # Replace NaNs in pred with 0 for ranking purposes
    pred_safe = np.where(np.isfinite(pred_st), pred_st, 0.0)

    tau, _ = kendalltau(true_st, pred_safe)
    rho, _ = spearmanr(true_st, pred_safe)

    arg_true = int(np.argmax(true_st))
    arg_pred = int(np.argmax(pred_safe))

    # Top-3 Jaccard overlap (cap at len)
    k = min(3, len(true_st))
    top_true = set(np.argsort(true_st)[::-1][:k].tolist())
    top_pred = set(np.argsort(pred_safe)[::-1][:k].tolist())
    union = top_true | top_pred
    jacc = float(len(top_true & top_pred) / max(len(union), 1))

    return {
        "kendall_tau": float(tau) if np.isfinite(tau) else float("nan"),
        "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
        "top1_correct": float(arg_true == arg_pred),
        "top3_jaccard": jacc,
        "L1": float(np.sum(np.abs(true_st - pred_safe))),
        "L2": float(np.sqrt(np.sum((true_st - pred_safe) ** 2))),
        "argmax_true": arg_true,
        "argmax_pred": arg_pred,
    }


def get_ground_truth_st(case_meta) -> Optional[np.ndarray]:
    """Extract a numpy array of ``sobol_ST`` values in input order.

    Returns None if the case lacks ground-truth Sobol indices.
    """
    if not case_meta.sobol_ST:
        return None
    inputs = case_meta.input_names
    try:
        return np.array([float(case_meta.sobol_ST[name]) for name in inputs])
    except (KeyError, TypeError):
        return None


# ─── Top-level orchestrator ──────────────────────────────────────────────
def run_sensitivity_recovery(
    case_ids: List[str],
    methods: List,
    budget_mult: int = 20,
    n_base: int = 256,
    seed: int = 0,
    output_idx: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """For each (case, method), refit + compute Sobol on surrogate.

    Returns a long-form DataFrame.  Methods are expected to be
    :class:`StaticDesignMethod` instances (or anything with a
    ``run(case, n_train, X_test, seed)`` API) — but here we re-fit
    inside this function to get hold of the surrogate object.

    For sequential methods this falls back to fitting only the
    underlying static-design version (i.e. we use the method's
    sampler + surrogate).
    """
    rows: List[Dict] = []

    for case_id in case_ids:
        case = db.get_case(case_id)
        true_st = get_ground_truth_st(case.meta)
        if true_st is None:
            if verbose:
                print(f"[skip] {case_id}: no ground-truth sobol_ST")
            continue
        d = case.meta.dim_input
        n_train = budget_mult * d

        for method in methods:
            # Get the underlying sampler + surrogate
            # The standard StaticDesignMethod has .sampler and .surrogate
            sampler = getattr(method, "sampler", None)
            surrogate_factory = getattr(method, "surrogate", None)
            if sampler is None or surrogate_factory is None:
                if verbose:
                    print(f"[skip] {method.name}: not a static-design method")
                continue

            try:
                # Sample + evaluate + fit
                X_train = sampler.sample(n_train, d, seed=seed)
                Y_train = case.evaluate(X_train)
                # Drop NaN rows
                mask = ~np.any(np.isnan(Y_train), axis=1)
                X_train, Y_train = X_train[mask], Y_train[mask]
                if len(X_train) < 2:
                    raise RuntimeError("too few valid training samples")
                # Fresh surrogate copy
                surrogate = surrogate_factory.__class__(
                    **surrogate_factory.get_init_kwargs())
                surrogate.fit(X_train, Y_train)

                # Diagnose surrogate collapse to constant
                rr = surrogate_response_range(
                    surrogate, n_inputs=d, output_idx=output_idx, seed=seed,
                )
                train_y_std = float(np.std(Y_train[:, output_idx], ddof=1))
                surrogate_to_train_ratio = (
                    rr["y_std"] / train_y_std if train_y_std > 0 else float("nan")
                )
                # Flag as "flat" if surrogate explains < 5 % of training
                # response variance (very conservative)
                is_flat = (np.isfinite(surrogate_to_train_ratio) and
                           surrogate_to_train_ratio < 0.05)

                # Compute Sobol on surrogate
                pred_st = surrogate_sobol_indices(
                    surrogate, n_inputs=d, n_base=n_base,
                    output_idx=output_idx, seed=seed,
                )
                if pred_st is None:
                    raise RuntimeError("Sobol analyze returned None")

                cmp = compare_sensitivity(true_st, pred_st)
                rows.append({
                    "case_id": case_id,
                    "method": method.name,
                    "output_idx": output_idx,
                    "n_inputs": d,
                    "n_train": len(X_train),
                    "surrogate_y_std": rr["y_std"],
                    "train_y_std": train_y_std,
                    "surr_train_std_ratio": surrogate_to_train_ratio,
                    "is_flat": bool(is_flat),
                    **cmp,
                    # also store the predicted vector for diagnostic plots
                    "pred_st": pred_st.tolist(),
                    "true_st": true_st.tolist(),
                })
                if verbose:
                    flag = " ⚠FLAT" if is_flat else ""
                    print(f"  {case_id:25s} | {method.name:20s} "
                          f"τ={cmp['kendall_tau']:+.2f}  "
                          f"top1={cmp['top1_correct']:.0f}  "
                          f"L1={cmp['L1']:.3f}{flag}")
            except Exception as exc:
                rows.append({
                    "case_id": case_id, "method": method.name,
                    "output_idx": output_idx, "n_inputs": d,
                    "kendall_tau": float("nan"),
                    "spearman_rho": float("nan"),
                    "top1_correct": float("nan"),
                    "top3_jaccard": float("nan"),
                    "L1": float("nan"), "L2": float("nan"),
                    "error": str(exc),
                })
                if verbose:
                    print(f"  {case_id:25s} | {method.name:20s} FAILED: {exc}")

    return pd.DataFrame(rows)


__all__ = [
    "compare_sensitivity",
    "get_ground_truth_st",
    "run_sensitivity_recovery",
    "surrogate_sobol_indices",
]
