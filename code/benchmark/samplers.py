"""
benchmark.samplers
==================
Sampler implementations: produce design matrices in [0,1]^d.

All samplers wrap ``scipy.stats.qmc`` with a uniform interface and
deterministic seeding.  ``OptimisedLHS`` adds the random-CD optimisation
for improved space-filling.

Classical model-based designs
-----------------------------
``DOptimalSampler`` selects ``n`` rows from a candidate pool to maximise
``log |X^T X|`` for a polynomial regression model matrix.  This is a
*model-based* design (the scientifically standard counterpart to the
*model-free* Latin / Sobol / Halton samplers above) and is the most
commonly cited DOE baseline in the engineering DOE literature
(Atkinson, Donev & Tobias, *Optimum Experimental Designs, with SAS*,
2007; Kennard & Stone 1969 KS exchange).  Including it answers the
near-universal reviewer question "where is the D-optimal baseline?"
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.stats.qmc import Halton, LatinHypercube, Sobol

from benchmark.core import Sampler


class RandomSampler(Sampler):
    """Pseudo-random uniform [0,1]^d (baseline; expected to perform worst)."""

    name = "random"

    def sample(self, n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.random(size=(n, d))


class LHSSampler(Sampler):
    """Plain Latin hypercube sampling (no optimisation)."""

    name = "lhs"

    def sample(self, n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
        engine = LatinHypercube(d=d, seed=seed)
        return engine.random(n)


class OptimisedLHSSampler(Sampler):
    """LHS with random-CD optimisation (better space-filling)."""

    name = "lhs_opt"

    def sample(self, n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
        engine = LatinHypercube(d=d, seed=seed, optimization="random-cd")
        return engine.random(n)


class SobolSampler(Sampler):
    """Scrambled Sobol' sequence."""

    name = "sobol"

    def sample(self, n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
        engine = Sobol(d=d, scramble=True, seed=seed)
        # SciPy warns whenever n is not a power of two.  The scrambled
        # Sobol' sequence is still a valid low-discrepancy set for any
        # n; we silence only this specific warning to keep the sweep
        # log clean (n=5d, 15d, 25d, etc. are almost never pow2).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol' points require",
                category=UserWarning,
            )
            return engine.random(n)


class HaltonSampler(Sampler):
    """Scrambled Halton sequence."""

    name = "halton"

    def sample(self, n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
        engine = Halton(d=d, scramble=True, seed=seed)
        return engine.random(n)


def _polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Build a polynomial-regression model matrix up to ``degree``.

    Includes the intercept, all main effects, all two-factor interactions
    (degree >= 2), and pure quadratic terms (degree >= 2).  Cubic and
    higher are not generated because (i) they explode in the column count
    for d >= 8, and (ii) D-optimal designs are most commonly used with
    quadratic models in the engineering DOE practice we are benchmarking
    against.
    """
    n, d = X.shape
    cols = [np.ones(n)]                     # intercept
    cols.extend(X[:, j] for j in range(d))  # main effects
    if degree >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])  # interaction + pure quadratic
    return np.column_stack(cols)


class DOptimalSampler(Sampler):
    r"""Greedy D-optimal design with Fedorov-style point exchange.

    Algorithm (Atkinson--Donev--Tobias 2007, §11.2):

    1.  Generate a candidate pool of ``n_candidates = max(20*n, 200)``
        Sobol' points in :math:`[0,1]^d`.
    2.  Initialise the design with ``n`` random rows from the pool
        (seed-controlled).
    3.  Compute the polynomial-regression information matrix
        :math:`M = X^T X / n` (intercept + main + 2-factor + quadratic).
    4.  **Exchange phase**: while improvements are possible, for each
        design row :math:`i` and each candidate row :math:`k`, compute the
        Fedorov-Mitchell delta
        :math:`\Delta_{ik} = d_k - d_i + d_i d_k - d_{ik}^2`
        where :math:`d_i = x_i^T M^{-1} x_i` is the leverage and
        :math:`d_{ik} = x_i^T M^{-1} x_k` is the cross-leverage.  Swap the
        pair maximising :math:`\Delta_{ik}` (which maximises the ratio of
        the post-swap to pre-swap D-criterion); rebuild :math:`M`.
    5.  Halt when no exchange yields :math:`\Delta > 10^{-6}`, capped at
        ``max_iter = 100``.

    Numerical safeguards
    --------------------
    * Singular :math:`M` is regularised with :math:`\varepsilon I_p`
      where :math:`\varepsilon = 10^{-10} \, \mathrm{trace}(M) / p`.
    * If the candidate pool yields a degenerate model matrix for the
      requested polynomial order (rank-deficient before any exchange),
      we fall back to ``OptimisedLHSSampler`` and emit a UserWarning.

    Cost
    ----
    Per-iteration cost is :math:`O(n \cdot n_c \cdot p)` for ``p``
    polynomial features.  With ``n_c = 20 n`` and ``p = O(d^2)`` this is
    practical up to about :math:`d = 12`; beyond that the polynomial
    column count itself dominates and we cap the order at 1 (linear
    model only) for ``d > 12``.

    Notes
    -----
    The design is fit to a *quadratic* model surface; downstream
    surrogates (GP, RBF, ...) are then trained on the resulting points.
    The mismatch between the design model and the surrogate model is
    intentional and is exactly the comparison the paper studies.
    """

    name = "doptimal"

    def __init__(
        self,
        polynomial_degree: int = 2,
        n_candidates_per_n: int = 30,
        max_iter: int = 100,
        n_starts: int = 4,
        tol: float = 1e-6,
    ) -> None:
        if polynomial_degree not in (1, 2):
            raise ValueError(f"polynomial_degree must be 1 or 2, got {polynomial_degree}")
        self.polynomial_degree = int(polynomial_degree)
        self.n_candidates_per_n = int(n_candidates_per_n)
        self.max_iter = int(max_iter)
        self.n_starts = max(1, int(n_starts))
        self.tol = float(tol)

    def _effective_degree(self, d: int) -> int:
        # For d > 8 a quadratic model has p = 1 + d + d(d+1)/2 columns;
        # at d=10 that is 66 columns and the inner einsum quickly becomes
        # the wall-clock bottleneck.  Drop to a linear model for d > 8 —
        # still a meaningful baseline ("D-optimal for first-order").
        if self.polynomial_degree == 2 and d > 8:
            warnings.warn(
                f"DOptimalSampler: degree downgraded to 1 for d={d} (quadratic "
                f"would need {1 + d + d * (d + 1) // 2} columns).",
                stacklevel=3,
            )
            return 1
        return self.polynomial_degree

    def _scale_budget(self, n: int, d: int) -> tuple[int, int, int]:
        """Adapt n_candidates / n_starts / max_iter to (n, d).

        For large designs the per-iteration cost of the full Fedorov
        exchange grows as O(n * n_c * p), so we shrink the candidate
        pool and the restart count to keep the sampler within a few
        seconds per call.  Caps were tuned so that ``n=1000, d=20``
        finishes within ~10 s on a single core.
        """
        # Cap candidate pool size relative to n
        if n >= 500:
            n_c_per_n = max(self.n_candidates_per_n // 8, 4)
            n_starts = 1
            max_iter = max(self.max_iter // 4, 20)
        elif n >= 200:
            n_c_per_n = max(self.n_candidates_per_n // 4, 6)
            n_starts = max(self.n_starts // 2, 1)
            max_iter = max(self.max_iter // 2, 40)
        else:
            n_c_per_n = self.n_candidates_per_n
            n_starts = self.n_starts
            max_iter = self.max_iter
        return n_c_per_n, n_starts, max_iter

    def sample(self, n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
        n_c_per_n, n_starts, max_iter = self._scale_budget(n, d)

        # ── 1. Candidate pool: union of Sobol and LHS-opt ─────────────
        # Using only Sobol caps the achievable D-criterion at the Sobol
        # pool's geometry.  Mixing in two LHS-opt designs (with different
        # seeds) expands the reachable space-filling configurations
        # without losing Sobol's coverage.  This is the standard
        # ``mixed candidate pool'' recommendation in
        # Atkinson-Donev-Tobias (2007, §11.4).
        n_c = max(n_c_per_n * n, 200)
        n_c_pow2 = int(2 ** np.ceil(np.log2(max(n_c, 4))))
        sobol = Sobol(d=d, scramble=True, seed=seed).random(n_c_pow2)
        _opt = "random-cd" if d > 1 else None  # random-cd requires d >= 2 (column swap)
        lhs1 = LatinHypercube(d=d, seed=seed, optimization=_opt).random(max(n, 64))
        lhs2 = LatinHypercube(d=d, seed=(seed or 0) + 7919, optimization=_opt).random(max(n, 64))
        candidates = np.vstack([sobol, lhs1, lhs2])
        n_c_pow2 = candidates.shape[0]

        # ── 2. Build polynomial features once for the whole pool ─────
        deg = self._effective_degree(d)
        Phi_full = _polynomial_features(candidates, degree=deg)
        p = Phi_full.shape[1]

        if n > n_c_pow2:
            warnings.warn(
                f"DOptimalSampler: requested n={n} exceeds candidate pool {n_c_pow2}; "
                f"using LHS fallback.",
                stacklevel=3,
            )
            return OptimisedLHSSampler().sample(n, d, seed=seed)

        if n < p + 2:
            # Not enough points to fit the model; fall back gracefully.
            warnings.warn(
                f"DOptimalSampler: n={n} is below the {p+2}-row floor needed "
                f"for a stable degree-{deg} D-optimal exchange (d={d}); using LHS.",
                stacklevel=3,
            )
            return OptimisedLHSSampler().sample(n, d, seed=seed)

        def _info_matrix(idx_list):
            X = Phi_full[idx_list, :]
            M = X.T @ X / max(len(idx_list), 1)
            eps = 1e-10 * np.trace(M) / max(p, 1)
            return M + eps * np.eye(p)

        def _logdet(idx_list):
            M = _info_matrix(idx_list)
            sign, logdet = np.linalg.slogdet(M)
            return logdet if sign > 0 else -np.inf

        # ── 3-5. Multi-start Fedorov-Mitchell exchange loop ──────────
        # Each restart begins from a different random subselection of
        # the candidate pool; we keep the design with the largest
        # log|M|.  For n_starts=1 this collapses to a single-start
        # greedy exchange.
        best_idx = None
        best_logdet = -np.inf

        for start in range(n_starts):
            start_rng = np.random.default_rng((seed or 0) * 1024 + start)
            init_idx = list(start_rng.choice(n_c_pow2, size=n, replace=False))
            design_idx = list(init_idx)

            for _ in range(max_iter):
                M = _info_matrix(design_idx)
                try:
                    M_inv = np.linalg.inv(M)
                except np.linalg.LinAlgError:
                    break

                X_design = Phi_full[design_idx, :]                         # (n, p)
                d_design = np.einsum("ij,jk,ik->i", X_design, M_inv, X_design)  # (n,)
                d_cand = np.einsum("ij,jk,ik->i", Phi_full, M_inv, Phi_full)     # (n_c,)
                # Cross leverage matrix: D[i, k] = x_i^T M^{-1} x_k
                D_cross = X_design @ M_inv @ Phi_full.T                    # (n, n_c)

                # Fedorov-Mitchell delta:
                #   Δ_{ik} = d_k - d_i + d_i * d_k - d_{ik}^2
                # Δ > 0  ⇔  swapping (i, k) increases det(M).  We search
                # the full grid for the maximum.  Candidates currently in
                # the design are masked to -inf to forbid self-swaps.
                Delta = (
                    d_cand[None, :]
                    - d_design[:, None]
                    + d_design[:, None] * d_cand[None, :]
                    - D_cross ** 2
                )
                mask_in_design = np.zeros(n_c_pow2, dtype=bool)
                mask_in_design[design_idx] = True
                Delta[:, mask_in_design] = -np.inf

                i_best, k_best = np.unravel_index(int(np.argmax(Delta)), Delta.shape)
                best_delta = float(Delta[i_best, k_best])
                if best_delta <= self.tol:
                    break
                design_idx[i_best] = int(k_best)

            cand_logdet = _logdet(design_idx)
            if cand_logdet > best_logdet:
                best_logdet = cand_logdet
                best_idx = list(design_idx)

        if best_idx is None:
            warnings.warn(
                "DOptimalSampler: every restart failed to converge; falling back to LHS.",
                stacklevel=3,
            )
            return OptimisedLHSSampler().sample(n, d, seed=seed)
        return candidates[best_idx, :]


# ─── Convenience factory ─────────────────────────────────────────────────
SAMPLER_REGISTRY = {
    cls.name: cls for cls in [
        RandomSampler, LHSSampler, OptimisedLHSSampler, SobolSampler, HaltonSampler,
        DOptimalSampler,
    ]
}


def get_sampler(name: str) -> Sampler:
    """Look up a sampler by name."""
    if name not in SAMPLER_REGISTRY:
        raise KeyError(f"Unknown sampler {name!r}; available: {list(SAMPLER_REGISTRY)}")
    return SAMPLER_REGISTRY[name]()
