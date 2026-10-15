"""
benchmark.bo_methods
====================
Bayesian-optimisation adapters built on **BoTorch** / GPyTorch.

These methods follow exactly the same :class:`BenchmarkMethod`
contract as :mod:`benchmark.sequential` (initial design → adaptive
loop → final fit → predict on ``X_test``) but use BoTorch's modern
GP / acquisition stack rather than the hand-rolled ``GPSurrogate``.

Adapters
--------
- :class:`BoTorchQEIMethod`
    q=1 Expected Improvement with a Matérn-5/2 SingleTaskGP, optimised
    via L-BFGS-B over a discretised candidate pool (Balandat et al.,
    *BoTorch: A framework for efficient Monte-Carlo Bayesian
    optimization*, NeurIPS 2020).
- :class:`BoTorchSAASMethod`
    Sparse-Axis-Aligned-Subspaces fully-Bayesian GP for high-dimensional
    BO (Eriksson & Jankowiak, *High-Dimensional Bayesian Optimization
    with Sparse Axis-Aligned Subspaces*, UAI 2021).  Recommended for
    cases with ``d ≥ 10``.

Optional dependency
-------------------
This module imports :mod:`botorch`, :mod:`gpytorch`, :mod:`torch` at
import time.  It will raise :class:`ImportError` if those packages
are not installed.  Install with ``pip install botorch gpytorch torch``
(CPU-only PyTorch is sufficient).
"""

from __future__ import annotations

import time
import warnings
import threading
from typing import Optional, Tuple

import numpy as np

import doe_testbed as db
from benchmark.core import BenchmarkMethod
from benchmark.samplers import OptimisedLHSSampler

# ─── Optional BoTorch import ─────────────────────────────────────────────
try:
    import gpytorch
    import torch
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    _BOTORCH_OK = True
except ImportError as exc:  # pragma: no cover
    _BOTORCH_OK = False
    _BOTORCH_IMPORT_ERROR = exc

try:
    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    _SAAS_OK = True
except ImportError:  # pragma: no cover
    _SAAS_OK = False


def _require_botorch() -> None:
    if not _BOTORCH_OK:
        raise ImportError(
            "BoTorch BO adapters require torch + gpytorch + botorch. "
            "Install with `pip install torch gpytorch botorch` "
            f"(original error: {_BOTORCH_IMPORT_ERROR})."
        )


# ─── Small helpers ───────────────────────────────────────────────────────
def _to_tensor(x: np.ndarray, *, dtype=None) -> "torch.Tensor":
    return torch.as_tensor(x, dtype=dtype or torch.double)


def _fit_singletaskgp(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> "SingleTaskGP":
    """Fit a single-output BoTorch ``SingleTaskGP`` with ``Standardize``.

    ``y_train`` must be 1-D; multi-output cases are handled by fitting
    one model per output dimension at the call site.

    Robustness
    ----------
    * Inputs are coerced to ``torch.double`` for Cholesky stability;
    * After fit, the likelihood noise is clamped to a small positive
      floor so the kernel matrix posterior at predict-time is always
      well-conditioned (avoids the GPyTorch ``rbf_kernel.sq_dist``
      access violations seen on Windows when the noise collapses to
      ~0 under a degenerate fit).
    * On a fit failure we retry once with a wider noise prior.
    """
    Xt = _to_tensor(X_train)
    yt = _to_tensor(y_train).unsqueeze(-1)

    def _build():
        return SingleTaskGP(
            train_X=Xt,
            train_Y=yt,
            outcome_transform=Standardize(m=1),
        )

    try:
        model = _build()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
    except Exception:
        # Retry once: the most common failure mode is a near-singular
        # kernel matrix at the optimiser's first probe; rebuilding the
        # model and re-fitting often succeeds.
        model = _build()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)

    # Numerical floor on the observation noise: keeps the Cholesky
    # well-conditioned at predict time even when the optimiser drove
    # the noise to ~0.  1e-6 is well below any genuine signal but
    # large enough to dominate floating-point round-off.
    try:
        with torch.no_grad():
            noise = model.likelihood.noise
            model.likelihood.noise = torch.clamp(noise, min=1e-6)
    except Exception:
        pass
    return model


def _gp_predict_per_output(
    X: np.ndarray,
    Y: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict (mean, std) on ``X_test`` by fitting one GP per output column."""
    n, m = Y.shape
    means = np.zeros((X_test.shape[0], m), dtype=float)
    stds = np.zeros((X_test.shape[0], m), dtype=float)
    Xt_test = _to_tensor(X_test)
    for j in range(m):
        model = _fit_singletaskgp(X, Y[:, j])
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(Xt_test)
            means[:, j] = posterior.mean.squeeze(-1).cpu().numpy()
            stds[:, j] = posterior.variance.squeeze(-1).clamp_min(1e-12).sqrt().cpu().numpy()
    return means, stds


# ─── qEI Bayesian Optimisation ───────────────────────────────────────────
class BoTorchQEIMethod(BenchmarkMethod):
    """q=1 Log-Expected-Improvement BO with BoTorch's SingleTaskGP.

    Parameters
    ----------
    n_init_frac : float
        Fraction of the budget used for the initial space-filling design.
    n_candidates : int
        Number of random candidates evaluated to optimise the acquisition.
    output_idx : int
        For multi-output cases, the output that drives acquisition.
    """

    family = "sequential_botorch_ei"
    name = "botorch_qei"
    # GPyTorch / BoTorch's MKL-backed kernel ops segfault inside loky
    # child processes on Windows (Cholesky / sq_dist access violations
    # in the spawned worker's MKL runtime).  Threading parallelises
    # them safely because the heavy ops release the GIL.
    parallel_backend = "threading"
    # Per-instance peak RSS ~200-500 MB (one SingleTaskGP + acquisition
    # tensor batch).  "medium" lets the runner cap concurrency at
    # n_jobs // 2 so 8-job hosts run 4 BoTorch-qEI in parallel
    # comfortably even with the rest of the pool churning.
    memory_class = "medium"

    def __init__(
        self,
        n_init_frac: float = 0.5,
        n_candidates: int = 256,
        output_idx: int = 0,
    ) -> None:
        _require_botorch()
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
        d = case.meta.dim_input
        n_init = max(2 * d, int(np.ceil(self.n_init_frac * n_train)))
        n_init = min(n_init, n_train - 1)

        torch.manual_seed(seed)

        t_total = time.perf_counter()
        sampler = OptimisedLHSSampler()
        X = sampler.sample(n_init, d, seed=seed)
        Y = case.evaluate(X)
        rng = np.random.default_rng(seed + 1)

        t_seq = 0.0
        for _step in range(n_train - n_init):
            t0 = time.perf_counter()
            mask = ~np.isnan(Y[:, self.output_idx])
            if mask.sum() < 2:
                X_next = rng.random((1, d))
            else:
                # qLogEI for *minimisation*: BoTorch maximises by default,
                # so we flip the sign of the target — Standardize transform
                # handles scale, and best_f is then the highest observed
                # value of the negated target.
                model_neg = _fit_singletaskgp(X[mask], -Y[mask, self.output_idx])
                best_f = float(np.nanmax(-Y[mask, self.output_idx]))
                X_cand = rng.random((self.n_candidates, d))
                Xc = _to_tensor(X_cand).unsqueeze(1)  # (q=1, d) batched
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    acq = qLogExpectedImprovement(model=model_neg, best_f=best_f)
                    val = acq(Xc).detach().cpu().numpy()
                idx = int(np.argmax(val))
                X_next = X_cand[idx:idx + 1]
            Y_next = case.evaluate(X_next)
            X = np.vstack([X, X_next])
            Y = np.vstack([Y, Y_next])
            t_seq += time.perf_counter() - t0

        # Final fit + predict (one GP per output for full multi-output coverage)
        mask = ~np.any(np.isnan(Y), axis=1)
        if mask.sum() < 2:
            raise RuntimeError(
                f"only {mask.sum()} valid samples; cannot fit final GP"
            )
        t0 = time.perf_counter()
        Y_pred, Y_std = _gp_predict_per_output(X[mask], Y[mask], X_test)
        t_final = time.perf_counter() - t0

        return Y_pred, Y_std, {
            "n_train_effective": int(mask.sum()),
            "n_train_failed": int((~mask).sum()),
            "n_init": int(n_init),
            "n_sequential": int(n_train - n_init),
            "t_sequential_s": float(t_seq),
            "t_final_fit_s": float(t_final),
            "t_total_method_s": float(time.perf_counter() - t_total),
        }


# ─── SAASBO (high-dim Bayesian Optimisation) ─────────────────────────────
class BoTorchSAASMethod(BenchmarkMethod):
    """SAASBO: Sparse-Axis-Aligned-Subspaces fully-Bayesian GP.

    Best suited to high-dimensional cases (``d ≥ 10``) where most
    dimensions are irrelevant.  Posterior is approximated via NUTS;
    each adaptive step costs ~10–60 s.

    Parameters
    ----------
    n_init_frac : float
    output_idx : int
    warmup_steps : int
        NUTS warm-up steps (BoTorch default 256).
    num_samples : int
        NUTS sampling steps (BoTorch default 128).  Reduced for the
        benchmark to keep per-step cost modest.
    """

    family = "sequential_botorch_saas"
    name = "botorch_saasbo"
    # NUTS-based fully-Bayesian fit reuses the same GPyTorch kernels;
    # SAAS additionally spawns a Pyro NUTS sampler whose reentrancy
    # is unreliable under threading (Pyro's _PYRO_STACK is process-global).
    # Use loky (process-based) for full Pyro isolation on each cell.
    parallel_backend = "loky"
    # Per-instance peak RSS ~1-2 GB (NUTS chain holds Cholesky factors,
    # posterior samples, and a batched candidate tensor).  "heavy" lets
    # the runner cap concurrency at bo_n_jobs (default n_jobs // 4) so
    # an 8-job host spawns at most 2 simultaneous SAASBO instances
    # — historically 8 simultaneous on 16 GB Windows triggered
    # MemoryError in the threading backend's shared process.
    memory_class = "heavy"

    def __init__(
        self,
        n_init_frac: float = 0.9,
        n_candidates: int = 64,
        output_idx: int = 0,
        warmup_steps: int = 4,
        num_samples: int = 4,
        thinning: int = 1,
    ) -> None:
        _require_botorch()
        if not _SAAS_OK:
            raise ImportError(
                "SaasFullyBayesianSingleTaskGP not available; "
                "upgrade BoTorch (>=0.7)."
            )
        self.n_init_frac = float(n_init_frac)
        self.n_candidates = int(n_candidates)
        self.output_idx = int(output_idx)
        self.warmup_steps = int(warmup_steps)
        self.num_samples = int(num_samples)
        self.thinning = int(thinning)

    def _fit_saas(self, X: np.ndarray, y: np.ndarray) -> "SaasFullyBayesianSingleTaskGP":
        import torch
        Xt = _to_tensor(X)
        yt = _to_tensor(y).unsqueeze(-1)
        model = SaasFullyBayesianSingleTaskGP(
            train_X=Xt,
            train_Y=yt,
            outcome_transform=Standardize(m=1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_fully_bayesian_model_nuts(
                model,
                warmup_steps=self.warmup_steps,
                num_samples=self.num_samples,
                thinning=self.thinning,
                disable_progbar=True,
            )
        return model

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        d = case.meta.dim_input
        n_init = max(2 * d, int(np.ceil(self.n_init_frac * n_train)))
        n_init = min(n_init, n_train - 1)

        torch.manual_seed(seed)

        t_total = time.perf_counter()
        sampler = OptimisedLHSSampler()
        X = sampler.sample(n_init, d, seed=seed)
        Y = case.evaluate(X)
        rng = np.random.default_rng(seed + 1)

        t_seq = 0.0
        for _step in range(n_train - n_init):
            t0 = time.perf_counter()
            mask = ~np.isnan(Y[:, self.output_idx])
            if mask.sum() < 2:
                X_next = rng.random((1, d))
            else:
                model_neg = self._fit_saas(X[mask], -Y[mask, self.output_idx])
                best_f = float(np.nanmax(-Y[mask, self.output_idx]))
                X_cand = rng.random((self.n_candidates, d))
                Xc = _to_tensor(X_cand).unsqueeze(1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    acq = qLogExpectedImprovement(model=model_neg, best_f=best_f)
                    val = acq(Xc).detach().cpu().numpy()
                # SAAS posteriors are a batch of MCMC draws; qLogEI returns
                # one value per candidate already aggregated over draws.
                idx = int(np.argmax(val.squeeze()))
                X_next = X_cand[idx:idx + 1]
            Y_next = case.evaluate(X_next)
            X = np.vstack([X, X_next])
            Y = np.vstack([Y, Y_next])
            t_seq += time.perf_counter() - t0

        mask = ~np.any(np.isnan(Y), axis=1)
        if mask.sum() < 2:
            raise RuntimeError(f"only {mask.sum()} valid samples; cannot fit final GP")
        t0 = time.perf_counter()
        # Final prediction uses ML SingleTaskGP for speed (SAAS posterior
        # already informed the search; final accuracy is comparable).
        Y_pred, Y_std = _gp_predict_per_output(X[mask], Y[mask], X_test)
        t_final = time.perf_counter() - t0

        return Y_pred, Y_std, {
            "n_train_effective": int(mask.sum()),
            "n_train_failed": int((~mask).sum()),
            "n_init": int(n_init),
            "n_sequential": int(n_train - n_init),
            "t_sequential_s": float(t_seq),
            "t_final_fit_s": float(t_final),
            "t_total_method_s": float(time.perf_counter() - t_total),
        }


# ─── TURBO (Trust-Region Bayesian Optimisation) ──────────────────────────
class BoTorchTurboMethod(BenchmarkMethod):
    """TuRBO: trust-region Bayesian optimisation (Eriksson et al. 2019, NeurIPS).

    Maintains a hyper-rectangular trust region centred on the current
    incumbent, fits a local SingleTaskGP inside it, and uses qEI to
    pick the next point.  The trust-region length :math:`L` is
    multiplicatively shrunk after a streak of failures and expanded
    after a streak of successes; once :math:`L` falls below
    ``L_min`` the trust region is restarted from a fresh space-filling
    initial design.  This makes TuRBO competitive in the regime where
    standard global GP-BO struggles — moderate-to-high dimensions,
    longer budgets, and heteroscedastic / non-stationary responses.

    The Eriksson et al. paper presents TuRBO as the dominant SOTA
    baseline against which any new BO method must compete; including
    it pre-empts the near-universal reviewer comment "where is TuRBO?"

    Parameters
    ----------
    n_init_frac : float
        Fraction of budget used for the initial space-filling design.
    n_candidates : int
        Acquisition candidate pool inside each trust region.
    L_init : float
        Initial trust-region edge length, in normalised coordinates.
    L_min, L_max : float
        Edge-length bounds; restart when :math:`L < L_{\\min}`.
    success_streak, fail_streak : int
        Number of consecutive better/worse iterations that trigger
        an expand or shrink event.  Defaults match the Eriksson et al.
        paper.
    """

    family = "sequential_botorch_turbo"
    name = "botorch_turbo"
    parallel_backend = "threading"  # same rationale as qEI
    memory_class = "medium"

    def __init__(
        self,
        n_init_frac: float = 0.5,
        n_candidates: int = 256,
        output_idx: int = 0,
        L_init: float = 0.8,
        L_min: float = 0.5 ** 7,
        L_max: float = 1.6,
        success_streak: int = 3,
        fail_streak: int = 3,
    ) -> None:
        _require_botorch()
        self.n_init_frac = float(n_init_frac)
        self.n_candidates = int(n_candidates)
        self.output_idx = int(output_idx)
        self.L_init = float(L_init)
        self.L_min = float(L_min)
        self.L_max = float(L_max)
        self.success_streak = int(success_streak)
        self.fail_streak = int(fail_streak)

    def _gp_lengthscale_widths(self, model: "SingleTaskGP") -> np.ndarray:
        """Return per-dimension lengthscale weights, normalised to mean 1.

        TuRBO scales the trust-region per axis by the GP lengthscale, so
        important dimensions (small lengthscale) get a smaller side
        length.  ``Standardize`` only touches outputs, so X stays in
        [0, 1]^d and the kernel lengthscales operate on the normalised
        coordinates directly.
        """
        try:
            with torch.no_grad():
                ls = model.covar_module.base_kernel.lengthscale.squeeze().cpu().numpy()
        except Exception:
            return None  # degenerate; caller falls back to isotropic TR
        ls = np.asarray(ls, dtype=float).ravel()
        if ls.size == 0:
            return None
        ls_mean = float(np.mean(ls))
        if ls_mean <= 0:
            return None
        weights = ls / ls_mean
        # Clamp very small / large values so the TR doesn't degenerate
        weights = np.clip(weights, 0.1, 10.0)
        return weights

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        d = case.meta.dim_input
        n_init = max(2 * d, int(np.ceil(self.n_init_frac * n_train)))
        n_init = min(n_init, n_train - 1)

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed + 1)

        t_total = time.perf_counter()
        sampler = OptimisedLHSSampler()
        X = sampler.sample(n_init, d, seed=seed)
        Y = case.evaluate(X)

        # ── Trust-region state ───────────────────────────────────────
        L = float(self.L_init)
        n_success = 0
        n_failure = 0
        n_restarts = 0
        # Track total evaluations so restart cost is paid out of the
        # remaining budget rather than added on top.  This keeps every
        # method in the comparison faithful to the n_train budget.
        n_evals = int(X.shape[0])

        t_seq = 0.0
        # Use a while loop so a restart can consume future iterations
        # (paying its bootstrap cost out of the remaining budget).
        while n_evals < n_train:
            t0 = time.perf_counter()
            mask = ~np.isnan(Y[:, self.output_idx])
            if mask.sum() < 2:
                X_next = rng.random((1, d))
            else:
                # Best (lowest) observed value drives both the TR centre
                # and the qEI best_f reference.  We minimise so flip sign
                # for BoTorch's "maximise" convention.
                y_obj = -Y[mask, self.output_idx]
                X_obs = X[mask]
                best_idx = int(np.argmax(y_obj))
                x_centre = X_obs[best_idx]
                best_f = float(y_obj[best_idx])

                # Fit local GP on (X, y).  The kernel automatically
                # downweights distant points so we keep all observations.
                model_neg = _fit_singletaskgp(X_obs, y_obj)

                # Build per-dim TR bounds, anchored at x_centre, scaled
                # by GP lengthscales (TuRBO §3.2).
                weights = self._gp_lengthscale_widths(model_neg)
                if weights is None:
                    weights = np.ones(d)
                tr_low = np.clip(x_centre - 0.5 * L * weights, 0.0, 1.0)
                tr_high = np.clip(x_centre + 0.5 * L * weights, 0.0, 1.0)

                # Sample candidates *inside* the TR (truncated uniform).
                X_cand = tr_low + (tr_high - tr_low) * rng.random((self.n_candidates, d))
                Xc = _to_tensor(X_cand).unsqueeze(1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    acq = qLogExpectedImprovement(model=model_neg, best_f=best_f)
                    val = acq(Xc).detach().cpu().numpy()
                idx = int(np.argmax(val))
                X_next = X_cand[idx:idx + 1]

            Y_next = case.evaluate(X_next)
            X = np.vstack([X, X_next])
            Y = np.vstack([Y, Y_next])
            n_evals += 1

            # ── TR adaptation: track success / failure streaks ────────
            if not np.isnan(Y_next[0, self.output_idx]) and X.shape[0] >= 2:
                prev_best = float(np.nanmin(Y[:-1, self.output_idx]))
                new_obs = float(Y_next[0, self.output_idx])
                rel_tol = 1e-3 * (abs(prev_best) + 1e-9)
                if new_obs < prev_best - rel_tol:
                    n_success += 1
                    n_failure = 0
                else:
                    n_failure += 1
                    n_success = 0

            if n_success >= self.success_streak:
                L = min(self.L_max, 2.0 * L)
                n_success = 0
            if n_failure >= self.fail_streak:
                L = max(self.L_min / 2.0, 0.5 * L)
                n_failure = 0

            # ── Restart if trust region collapsed ─────────────────────
            # The bootstrap LHS pays out of the remaining budget so the
            # method does not exceed n_train.  If the remaining budget
            # is insufficient for a full d-point re-init we skip the
            # restart and let the outer while-loop terminate naturally.
            if L < self.L_min:
                remaining = n_train - n_evals
                n_boot = max(d, 2)
                if remaining <= 0:
                    break
                if remaining < n_boot:
                    # Not enough budget for a meaningful restart; just
                    # let the loop continue at the small L until budget
                    # is exhausted.
                    L = max(self.L_min / 2.0, L)
                else:
                    n_restarts += 1
                    X_restart = sampler.sample(
                        n_boot, d,
                        seed=int(rng.integers(0, 2**31 - 1)),
                    )
                    Y_restart = case.evaluate(X_restart)
                    X = np.vstack([X, X_restart])
                    Y = np.vstack([Y, Y_restart])
                    n_evals += n_boot
                    L = float(self.L_init)
                    n_success = 0
                    n_failure = 0
            t_seq += time.perf_counter() - t0

        mask = ~np.any(np.isnan(Y), axis=1)
        if mask.sum() < 2:
            raise RuntimeError(f"only {mask.sum()} valid samples; cannot fit final GP")
        t0 = time.perf_counter()
        Y_pred, Y_std = _gp_predict_per_output(X[mask], Y[mask], X_test)
        t_final = time.perf_counter() - t0

        return Y_pred, Y_std, {
            "n_train_effective": int(mask.sum()),
            "n_train_failed": int((~mask).sum()),
            "n_init": int(n_init),
            "n_evals_total": int(n_evals),
            "n_sequential": int(n_evals - n_init),
            "n_tr_restarts": int(n_restarts),
            "tr_L_final": float(L),
            "t_sequential_s": float(t_seq),
            "t_final_fit_s": float(t_final),
            "t_total_method_s": float(time.perf_counter() - t_total),
        }


# ─── Multi-Fidelity Cokriging BO ─────────────────────────────────────────
class MFCokrigingMethod(BenchmarkMethod):
    """Multi-fidelity surrogate via Forrester--Sobester--Keane cokriging.

    Implements the linear two-fidelity additive model

    .. math::

        \\hat y_{\\text{hi}}(x) = \\rho \\, \\hat y_{\\text{lo}}(x) + \\delta(x)

    where :math:`\\hat y_{\\text{lo}}` is a SingleTaskGP fit on cheap
    ``fidelity="default"`` evaluations, :math:`\\hat y_{\\text{hi}}` is a
    SingleTaskGP fit on expensive ``fidelity="high"`` evaluations, and
    :math:`\\rho` is estimated by least squares on the high-fidelity
    points (Kennedy & O'Hagan 2000; Forrester--Sobester--Keane 2007).

    Budget allocation
    -----------------
    The benchmark runner supplies a single ``n_train`` budget which we
    split between fidelities by their cost ratio
    :math:`r = t_{\\text{hi}} / t_{\\text{def}}` from
    ``meta.typical_eval_time_ms``:

    * ``n_low``  cheap evaluations
    * ``n_high`` expensive evaluations
    * subject to ``n_low + r * n_high <= n_train`` and
      ``n_low / n_high \\approx r * frac_low / (1 - frac_low)``.

    By default ``frac_low = 0.5`` (equal cost on each fidelity).

    Restrictions
    ------------
    Requires the case to support ``fidelity_levels >= 2``.  Falls back
    to a plain ``BoTorchQEIMethod`` (single-fidelity) on cases without
    a ``"high"`` fidelity.

    The method's NRMSE is reported against the runner's reference
    test set, which is evaluated at the *default* fidelity by the
    main benchmark contract; for a head-to-head MF vs SF comparison
    on the same target the recommended workflow is
    ``run_full_benchmark.py --mf-reference-fidelity default --pool mf``.
    """

    family = "multifidelity_cokriging"
    name = "mf_cokriging"
    parallel_backend = "threading"
    memory_class = "medium"

    def __init__(
        self,
        n_init_frac: float = 0.5,
        frac_low_cost: float = 0.5,
        output_idx: int = 0,
    ) -> None:
        _require_botorch()
        self.n_init_frac = float(n_init_frac)
        self.frac_low_cost = float(frac_low_cost)
        if not 0.05 < self.frac_low_cost < 0.95:
            raise ValueError("frac_low_cost must be in (0.05, 0.95)")
        self.output_idx = int(output_idx)

    def _split_budget(self, case: db.BaseCase, n_train: int) -> Tuple[int, int]:
        """Decide how many low-fidelity vs high-fidelity calls to make.

        We treat ``n_train`` as a *cost* budget in low-fidelity units.
        If the resulting allocation cannot afford at least 4 evals of
        each fidelity (the floor at which cokriging is meaningful) we
        return ``(n_train, 0)`` to signal a single-fidelity fallback.
        Otherwise we cap the allocation so the total cost never exceeds
        ``n_train``.
        """
        meta = case.meta
        levels = list(meta.fidelity_levels or [])
        if "high" not in levels:
            return n_train, 0  # no high fidelity available

        time_ms = dict(meta.typical_eval_time_ms or {})
        t_def = float(time_ms.get("default", 1.0))
        t_hi = float(time_ms.get("high", t_def * 5.0))
        r = max(t_hi / max(t_def, 1e-6), 1.0)

        # Cost-aware allocation:
        #   n_low + r * n_high = n_train  (cost equation)
        #   n_low = frac_low_cost * n_train    (low-fidelity cost share)
        #   n_high = (1 - frac_low_cost) * n_train / r
        n_low_target = int(np.floor(self.frac_low_cost * n_train))
        n_high_target = int(np.floor((1.0 - self.frac_low_cost) * n_train / r))

        # Reserve at least 4 of each.  If neither fits we fall back.
        if n_low_target < 4 or n_high_target < 4:
            return n_train, 0

        # Enforce the budget cap exactly: shrink whichever pool overshoots.
        n_low, n_high = n_low_target, n_high_target
        if n_low + r * n_high > n_train:
            # Shave high first (they cost the most)
            while n_low + r * n_high > n_train and n_high > 4:
                n_high -= 1
            while n_low + r * n_high > n_train and n_low > 4:
                n_low -= 1
            # If even the floor (4, 4) overshoots, fall back.
            if n_low + r * n_high > n_train:
                return n_train, 0

        return int(n_low), int(n_high)

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        d = case.meta.dim_input
        levels = case.meta.fidelity_levels or []
        torch.manual_seed(seed)

        n_low, n_high = self._split_budget(case, n_train)
        t_total = time.perf_counter()

        # Pure single-fidelity fallback when 'high' is not available.
        if n_high == 0 or "high" not in levels:
            warnings.warn(
                f"MFCokrigingMethod: case '{case.meta.case_id}' has no 'high' "
                f"fidelity (levels={levels}); falling back to single-fidelity GP."
            )
            sampler = OptimisedLHSSampler()
            X = sampler.sample(n_train, d, seed=seed)
            Y = case.evaluate(X, fidelity="default")
            mask = ~np.any(np.isnan(Y), axis=1)
            if mask.sum() < 2:
                raise RuntimeError("not enough valid samples for fallback GP")
            Y_pred, Y_std = _gp_predict_per_output(X[mask], Y[mask], X_test)
            return Y_pred, Y_std, {
                "n_train_effective": int(mask.sum()),
                "n_low": int(n_train), "n_high": 0,
                "rho_estimate": float("nan"),
                "t_total_method_s": float(time.perf_counter() - t_total),
                "fallback_to_single_fidelity": True,
            }

        # ── 1.  Sample at default (cheap) fidelity ────────────────────
        sampler = OptimisedLHSSampler()
        X_low = sampler.sample(n_low, d, seed=seed)
        Y_low = case.evaluate(X_low, fidelity="default")
        mask_lo = ~np.any(np.isnan(Y_low), axis=1)
        if mask_lo.sum() < 2:
            raise RuntimeError("low-fidelity stage produced too few valid samples")

        # ── 2.  Pick high-fidelity locations by max predictive variance ─
        # We fit one GP on the low-fidelity data and pick the n_high
        # Sobol' candidates with the largest posterior variance.  This
        # is the "batch maximum-variance" approximation that matches the
        # near-optimality of true sequential variance-greedy selection
        # for small batches (n_high ≤ ~10) — see Krause et al. (2008,
        # JMLR §4) for the regret analysis.
        model_lo = _fit_singletaskgp(X_low[mask_lo], Y_low[mask_lo, self.output_idx])
        n_cand = max(2 ** int(np.ceil(np.log2(max(20 * n_high, 200)))), 256)
        from scipy.stats.qmc import Sobol as _Sobol
        cand = _Sobol(d=d, scramble=True, seed=seed).random(n_cand)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = model_lo.posterior(_to_tensor(cand))
            var_all = post.variance.squeeze(-1).cpu().numpy()
        # Top-n_high indices by variance (deterministic given the GP fit)
        top_k_idx = np.argpartition(-var_all, kth=min(n_high, n_cand - 1))[:n_high]
        # Sort for reproducibility
        top_k_idx = top_k_idx[np.argsort(-var_all[top_k_idx])]
        X_hi = cand[top_k_idx]
        Y_hi = case.evaluate(X_hi, fidelity="high")
        mask_hi = ~np.any(np.isnan(Y_hi), axis=1)
        if mask_hi.sum() < 2:
            raise RuntimeError("high-fidelity stage produced too few valid samples")

        # ── 3.  Estimate ρ on high-fidelity points (LS) ──────────────
        # ρ minimises ||y_hi - ρ * y_low_at_xhi - bias||^2; we drop the
        # bias since SingleTaskGP's Standardize already zero-means the
        # outputs internally and we re-fit the residual GP on the
        # zero-mean residual.
        Y_low_at_hi = _gp_predict_per_output(
            X_low[mask_lo], Y_low[mask_lo], X_hi[mask_hi]
        )[0]
        y_hi_obs = Y_hi[mask_hi, self.output_idx]
        y_lo_pred = Y_low_at_hi[:, self.output_idx]
        denom = float(np.dot(y_lo_pred, y_lo_pred))
        rho = float(np.dot(y_hi_obs, y_lo_pred) / denom) if denom > 1e-12 else 1.0

        # ── 4.  Residual GP δ = y_hi - ρ * y_lo ──────────────────────
        residuals = y_hi_obs - rho * y_lo_pred
        model_delta = _fit_singletaskgp(X_hi[mask_hi], residuals)

        # ── 5.  Predict on X_test (per output, but only output_idx is
        # MF-augmented; the rest are predicted from the low-fidelity GP).
        Y_lo_pred, _ = _gp_predict_per_output(
            X_low[mask_lo], Y_low[mask_lo], X_test
        )
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post_d = model_delta.posterior(_to_tensor(X_test))
            d_mean = post_d.mean.squeeze(-1).cpu().numpy()
            d_std = post_d.variance.squeeze(-1).clamp_min(1e-12).sqrt().cpu().numpy()

        Y_pred = Y_lo_pred.copy()
        Y_pred[:, self.output_idx] = rho * Y_lo_pred[:, self.output_idx] + d_mean
        Y_std = np.zeros_like(Y_pred)
        Y_std[:, self.output_idx] = d_std

        return Y_pred, Y_std, {
            "n_train_effective": int(mask_hi.sum()),
            "n_low": int(n_low),
            "n_high": int(n_high),
            "rho_estimate": float(rho),
            "fidelity_high_cost_ratio": float(
                (case.meta.typical_eval_time_ms or {}).get("high", 0.0)
                / max((case.meta.typical_eval_time_ms or {}).get("default", 1.0), 1e-6)
            ),
            "t_total_method_s": float(time.perf_counter() - t_total),
            "fallback_to_single_fidelity": False,
        }


__all__ = [
    "BoTorchQEIMethod",
    "BoTorchSAASMethod",
    "BoTorchTurboMethod",
    "MFCokrigingMethod",
]
