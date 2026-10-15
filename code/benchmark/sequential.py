"""
benchmark.sequential
====================
Sequential / active-learning DOE methods for the benchmark.

These methods extend the static design pattern with adaptive sampling:
they use the case (oracle) during training to choose where to sample
next, in contrast to :class:`StaticDesignMethod` which fixes the
training design upfront.

Implemented
-----------
- :class:`EGOMethod` — Efficient Global Optimization (Jones et al. 1998),
  also known as Bayesian Optimization with Expected Improvement.
- :class:`AKMCSMethod` — Active Kriging + Monte Carlo Sampling
  (Echard et al. 2011) for failure-probability estimation.
- :class:`MaxVarianceMethod` — Maximum predictive variance (a baseline
  active learner for global accuracy).

All methods inherit :class:`BenchmarkMethod` and produce predictions on
the supplied test set after their adaptive design loop completes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

import doe_testbed as db
from benchmark.core import BenchmarkMethod
from benchmark.samplers import OptimisedLHSSampler
from benchmark.surrogates import GPSurrogate


# ─── EGO / Bayesian Optimisation with Expected Improvement ───────────────
class EGOMethod(BenchmarkMethod):
    """Expected-Improvement Bayesian Optimisation, fitted as a global surrogate.

    Although EGO is canonically an *optimisation* method, here we use the
    same EI-greedy acquisition to grow a training set and then evaluate
    the resulting surrogate's *global* prediction accuracy on the test
    set.  This is the standard way to compare BO methods to space-filling
    methods on a level playing field for prediction quality
    (cf. Picheny et al., 2013; Snoek et al., 2012).

    Parameters
    ----------
    n_init_frac : float
        Fraction of the budget used for the initial space-filling design.
        The remaining points are placed sequentially by maximising EI.
    n_candidates : int
        Number of random candidates to evaluate the acquisition over at
        each step (cheap proxy for full continuous optimisation).
    output_idx : int
        For multi-output cases, which output drives the acquisition.
        Default 0 (the primary output).
    """

    family = "sequential_ego"
    name = "ego_ei"
    # Re-fits a sklearn GaussianProcessRegressor at every adaptive step;
    # peak RSS per worker ~200-500 MB for n_train ~ 250.  "medium"
    # caps concurrency at n_jobs//2 to keep aggregate RSS bounded.
    memory_class = "medium"

    def __init__(
        self,
        n_init_frac: float = 0.5,
        n_candidates: int = 1000,
        output_idx: int = 0,
    ) -> None:
        self.n_init_frac = float(n_init_frac)
        self.n_candidates = int(n_candidates)
        self.output_idx = int(output_idx)

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        import time
        d = case.meta.dim_input
        n_init = max(2 * d, int(np.ceil(self.n_init_frac * n_train)))
        n_init = min(n_init, n_train - 1)  # leave at least 1 sequential point

        # ── Initial space-filling design ──
        t_total = time.perf_counter()
        sampler = OptimisedLHSSampler()
        X = sampler.sample(n_init, d, seed=seed)
        Y = case.evaluate(X)

        rng = np.random.default_rng(seed + 1)

        # ── Sequential EI loop ──
        t_seq = 0.0
        for _step in range(n_train - n_init):
            t0 = time.perf_counter()
            # Candidate pool: random points in [0,1]^d
            X_cand = rng.random((self.n_candidates, d))

            # Fit GP on current data (output of interest)
            mask = ~np.isnan(Y[:, self.output_idx])
            if mask.sum() < 2:
                # Failed evaluations dominate; fall back to random
                X_next = rng.random((1, d))
            else:
                gp = GPSurrogate(n_restarts=2)
                gp.fit(X[mask], Y[mask])
                mu, sigma = gp.predict(X_cand)
                mu_o = mu[:, self.output_idx]
                sigma_o = np.maximum(sigma[:, self.output_idx], 1e-9)
                # EI for *minimisation* of the output of interest.
                # For prediction-accuracy tasks the direction of the
                # objective is irrelevant—EI just keeps the surrogate
                # honest on the high-priority region.  We minimise.
                f_best = float(np.nanmin(Y[mask, self.output_idx]))
                z = (f_best - mu_o) / sigma_o
                ei = (f_best - mu_o) * norm.cdf(z) + sigma_o * norm.pdf(z)
                ei = np.where(sigma_o > 1e-9, ei, 0.0)
                idx = int(np.argmax(ei))
                X_next = X_cand[idx:idx + 1]

            # Evaluate the case at the chosen point
            Y_next = case.evaluate(X_next)
            X = np.vstack([X, X_next])
            Y = np.vstack([Y, Y_next])
            t_seq += time.perf_counter() - t0

        # ── Final fit on all training data + predict on test ──
        mask = ~np.any(np.isnan(Y), axis=1)
        if mask.sum() < 2:
            raise RuntimeError(f"only {mask.sum()} valid samples; cannot fit final GP")
        t0 = time.perf_counter()
        gp = GPSurrogate(n_restarts=3)
        gp.fit(X[mask], Y[mask])
        Y_pred, Y_std = gp.predict(X_test)
        t_final = time.perf_counter() - t0

        info = {
            "n_train_effective": int(mask.sum()),
            "n_train_failed": int((~mask).sum()),
            "n_init": int(n_init),
            "n_sequential": int(n_train - n_init),
            "t_sequential_s": float(t_seq),
            "t_final_fit_s": float(t_final),
            "t_total_method_s": float(time.perf_counter() - t_total),
        }
        return Y_pred, Y_std, info


# ─── Maximum-variance active learning ────────────────────────────────────
class MaxVarianceMethod(BenchmarkMethod):
    """Greedy maximum-predictive-variance active learner.

    A space-filling-by-uncertainty baseline: at each step, evaluate at
    the candidate with the largest GP predictive std.  This is the
    natural counterpart to EI when the goal is *global accuracy* rather
    than optimisation.
    """

    family = "sequential_active_learning"
    name = "max_variance"
    # Same per-step GP refit as EGO; same memory profile.
    memory_class = "medium"

    def __init__(self, n_init_frac: float = 0.5, n_candidates: int = 1000) -> None:
        self.n_init_frac = float(n_init_frac)
        self.n_candidates = int(n_candidates)

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        import time
        d = case.meta.dim_input
        n_init = max(2 * d, int(np.ceil(self.n_init_frac * n_train)))
        n_init = min(n_init, n_train - 1)

        t_total = time.perf_counter()
        sampler = OptimisedLHSSampler()
        X = sampler.sample(n_init, d, seed=seed)
        Y = case.evaluate(X)

        rng = np.random.default_rng(seed + 1)
        for _step in range(n_train - n_init):
            mask = ~np.any(np.isnan(Y), axis=1)
            if mask.sum() < 2:
                X_next = rng.random((1, d))
            else:
                gp = GPSurrogate(n_restarts=2)
                gp.fit(X[mask], Y[mask])
                X_cand = rng.random((self.n_candidates, d))
                _, sigma = gp.predict(X_cand)
                # Sum of std across outputs (fair to multi-output cases)
                score = sigma.sum(axis=1) if sigma.ndim == 2 else sigma
                idx = int(np.argmax(score))
                X_next = X_cand[idx:idx + 1]
            Y_next = case.evaluate(X_next)
            X = np.vstack([X, X_next])
            Y = np.vstack([Y, Y_next])

        mask = ~np.any(np.isnan(Y), axis=1)
        gp = GPSurrogate(n_restarts=3)
        gp.fit(X[mask], Y[mask])
        Y_pred, Y_std = gp.predict(X_test)

        return Y_pred, Y_std, {
            "n_train_effective": int(mask.sum()),
            "n_train_failed": int((~mask).sum()),
            "n_init": int(n_init),
            "n_sequential": int(n_train - n_init),
            "t_total_method_s": float(time.perf_counter() - t_total),
        }


# ─── AK-MCS (failure-probability oriented) ───────────────────────────────
class AKMCSMethod(BenchmarkMethod):
    """Active Kriging + Monte Carlo Sampling (Echard et al. 2011).

    This is a reliability-analysis method: it actively places samples
    near a failure boundary defined by ``output_threshold`` and
    ``direction``.  For benchmarking *global* prediction accuracy, we
    still report standard regression metrics on the test set, but the
    method's strength shows up in failure-probability metrics.

    Parameters
    ----------
    output_idx : int
    threshold : float
    direction : str
        'above' or 'below'
    n_init_frac : float
    """

    family = "sequential_reliability"
    name = "akmcs"
    # AK-MCS uses a much larger candidate pool (default 5,000) than
    # EGO/MaxVar but the GP fit is the same — peak RSS per worker
    # is comparable (~300-700 MB) so the same "medium" cap applies.
    memory_class = "medium"

    def __init__(
        self,
        output_idx: int = 0,
        threshold: float = 0.0,
        direction: str = "above",
        n_init_frac: float = 0.4,
        n_candidates: int = 5000,
    ) -> None:
        self.output_idx = int(output_idx)
        self.threshold = float(threshold)
        if direction not in ("above", "below"):
            raise ValueError("direction must be 'above' or 'below'")
        self.direction = direction
        self.n_init_frac = float(n_init_frac)
        self.n_candidates = int(n_candidates)

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        import time
        d = case.meta.dim_input
        n_init = max(2 * d, int(np.ceil(self.n_init_frac * n_train)))
        n_init = min(n_init, n_train - 1)

        t_total = time.perf_counter()
        sampler = OptimisedLHSSampler()
        X = sampler.sample(n_init, d, seed=seed)
        Y = case.evaluate(X)
        rng = np.random.default_rng(seed + 1)

        for _step in range(n_train - n_init):
            mask = ~np.any(np.isnan(Y), axis=1)
            if mask.sum() < 2:
                X_next = rng.random((1, d))
            else:
                gp = GPSurrogate(n_restarts=2)
                gp.fit(X[mask], Y[mask])
                X_cand = rng.random((self.n_candidates, d))
                mu, sigma = gp.predict(X_cand)
                mu_o = mu[:, self.output_idx]
                sigma_o = np.maximum(sigma[:, self.output_idx], 1e-9)
                # U-function: distance from threshold normalised by std
                if self.direction == "above":
                    u = (mu_o - self.threshold) / sigma_o
                else:
                    u = (self.threshold - mu_o) / sigma_o
                # Pick candidate with smallest |u| (closest to boundary, in std units)
                idx = int(np.argmin(np.abs(u)))
                X_next = X_cand[idx:idx + 1]
            Y_next = case.evaluate(X_next)
            X = np.vstack([X, X_next])
            Y = np.vstack([Y, Y_next])

        mask = ~np.any(np.isnan(Y), axis=1)
        gp = GPSurrogate(n_restarts=3)
        gp.fit(X[mask], Y[mask])
        Y_pred, Y_std = gp.predict(X_test)

        return Y_pred, Y_std, {
            "n_train_effective": int(mask.sum()),
            "n_train_failed": int((~mask).sum()),
            "n_init": int(n_init),
            "n_sequential": int(n_train - n_init),
            "t_total_method_s": float(time.perf_counter() - t_total),
        }
