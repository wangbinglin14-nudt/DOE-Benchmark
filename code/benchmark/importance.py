"""
benchmark.importance
====================
Variable-importance recovery: how well does a fitted surrogate
recover the true Sobol total-effect indices?

For every case in the doe_testbed library we have ground-truth Sobol'
total-effect indices ($S_T$) stored in :attr:`CaseMeta.sobol_ST`.
A surrogate that gives accurate global predictions might *also* be
useful for variable screening — but this is not implied by low RMSE.

This module computes a *Sobol recovery* score per (method, case)
pair: the Kendall τ between the true $S_T$ vector and the $S_T$
inferred from the trained surrogate via Saltelli / Sobol' Monte Carlo.

For surrogates that don't expose a closed-form Sobol' computation,
we estimate $S_T$ from the surrogate's predictions on a Saltelli
design — this approximates the surrogate's belief about variable
importance.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr

import doe_testbed as db


def _saltelli_design(d: int, n_base: int, seed: int = 0) -> Tuple[np.ndarray, int]:
    """Generate a Saltelli design (n_base*(d+2) rows) without SALib.

    Returns (matrix, total_rows).
    """
    from scipy.stats.qmc import Sobol
    n = n_base
    A = Sobol(d=d, scramble=True, seed=seed).random(n)
    B = Sobol(d=d, scramble=True, seed=seed + 1).random(n)
    rows = []
    rows.append(A)
    rows.append(B)
    for j in range(d):
        AB = A.copy()
        AB[:, j] = B[:, j]
        rows.append(AB)
    return np.vstack(rows), n


def _sobol_indices_from_predictions(
    Y: np.ndarray, n_base: int, d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute first-order and total-effect Sobol indices from a Saltelli design.

    Y : flat array of length n_base*(d+2).  Saltelli layout:
      [0:n]      ← f(A)
      [n:2n]     ← f(B)
      [2n:3n]    ← f(AB_0)
      [3n:4n]    ← f(AB_1)
      ...

    Returns (S1, ST) each of length d.

    Estimator: Saltelli (2010) — Variance-based sensitivity analysis
    of model output: design and estimator for the total sensitivity
    index, *Computer Physics Communications* 181:259–270.
    """
    n = n_base
    yA = Y[:n]
    yB = Y[n:2 * n]
    AB = [Y[(2 + j) * n:(3 + j) * n] for j in range(d)]
    var_y = float(np.var(yA, ddof=1))
    if var_y < 1e-20:
        return np.zeros(d), np.zeros(d)
    S1 = np.zeros(d)
    ST = np.zeros(d)
    for j in range(d):
        # First-order
        S1[j] = float(np.mean(yB * (AB[j] - yA)) / var_y)
        # Total-effect
        ST[j] = float(0.5 * np.mean((yA - AB[j]) ** 2) / var_y)
    return S1, ST


def estimate_sobol_via_surrogate(
    surrogate,
    d: int,
    output_idx: int = 0,
    n_base: int = 1024,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate Sobol indices by querying a fitted surrogate on a Saltelli design.

    The total cost is ``n_base * (d + 2)`` predictions — usually
    cheap because surrogate evaluation is fast.
    """
    X, n = _saltelli_design(d, n_base, seed=seed)
    Y_pred, _ = surrogate.predict(X)
    if Y_pred.ndim == 2:
        Y_pred = Y_pred[:, output_idx]
    return _sobol_indices_from_predictions(Y_pred, n, d)


def true_sobol_ST(case_id: str, output_idx: int = 0) -> Optional[np.ndarray]:
    """Look up stored ground-truth $S_T$ for one case.

    The ``CaseMeta.sobol_ST`` dict is keyed by **input variable name**
    (one entry per input).  The library stores indices for the
    *primary* output only, so ``output_idx`` is currently ignored and
    is kept in the signature for forward compatibility.

    Returns ``None`` when the case has no stored indices.
    """
    try:
        meta = db.get_case_meta(case_id)
    except Exception:
        return None
    if not meta.sobol_ST:
        return None
    return np.array([float(meta.sobol_ST.get(name, 0.0)) for name in meta.input_names])


def compute_importance_recovery(
    surrogate,
    case_id: str,
    output_idx: int = 0,
    n_base: int = 512,
    seed: int = 0,
) -> Dict[str, float]:
    """Compute Kendall τ and Spearman ρ between true and surrogate $S_T$.

    Returns dict with keys: kendall_tau, kendall_p, spearman_rho,
    spearman_p, mae_st, ranking_top_k_overlap.
    """
    true_st = true_sobol_ST(case_id, output_idx=output_idx)
    if true_st is None:
        return {
            "kendall_tau": float("nan"),
            "kendall_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "mae_st": float("nan"),
            "top_k_overlap": float("nan"),
        }
    d = len(true_st)
    _, surrogate_st = estimate_sobol_via_surrogate(
        surrogate, d=d, output_idx=output_idx, n_base=n_base, seed=seed,
    )
    tau, p_tau = kendalltau(true_st, surrogate_st)
    rho, p_rho = spearmanr(true_st, surrogate_st)
    # Top-k overlap (what fraction of true top-k variables are recovered?)
    k = max(1, d // 2)
    true_top = set(np.argsort(true_st)[-k:])
    surr_top = set(np.argsort(surrogate_st)[-k:])
    overlap = len(true_top & surr_top) / k
    return {
        "kendall_tau": float(tau) if not np.isnan(tau) else float("nan"),
        "kendall_p": float(p_tau) if not np.isnan(p_tau) else float("nan"),
        "spearman_rho": float(rho) if not np.isnan(rho) else float("nan"),
        "spearman_p": float(p_rho) if not np.isnan(p_rho) else float("nan"),
        "mae_st": float(np.mean(np.abs(true_st - surrogate_st))),
        "top_k_overlap": float(overlap),
    }


__all__ = [
    "compute_importance_recovery",
    "estimate_sobol_via_surrogate",
    "true_sobol_ST",
]
