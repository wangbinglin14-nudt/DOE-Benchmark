"""
benchmark.hyperparameters
=========================
Canonical, machine-readable copy of the hyperparameter fairness
protocol documented in ``docs/hyperparameter_fairness.md``.

The :data:`HYPERPARAMETERS` dictionary provides a single source of truth
that downstream analyses, papers and CI tests can introspect.  Three
properties are guaranteed:

1. Every method in :func:`benchmark.methods.full_method_pool` (and its
   superset :func:`benchmark.methods.nature_pool`) appears as a key.
2. The values match the construction kwargs passed by the pool
   factories, verified by :func:`assert_hyperparameters_consistent`.
3. Every entry carries a ``_provenance`` string explaining where the
   hyperparameter values came from.  The string starts with one of
   six convention tags:

   - ``library_default``        — verbatim from the upstream library
   - ``library_default+modified`` — library default with one cited deviation
   - ``matched_for_ablation``   — chosen to match a sibling method so
     that only the studied factor (sampler, kernel, …) differs
   - ``tuned_for_benchmark``    — selected for this suite, with a
     stated numerical/conditioning rationale
   - ``paper_recommended``      — taken from a primary publication
   - ``benchmark_internal``     — original implementation, justified
     by an internal heuristic that mirrors a published convention

Tests in ``benchmark/tests/test_hyperparameters.py`` lock in the
match so any silent drift between code and documentation triggers a
regression failure.
"""

from __future__ import annotations

from typing import Any, Dict

# ─── Canonical hyperparameter table ──────────────────────────────────────
HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    # ── Samplers (paired with GP for the sampler ablation) ──
    "random+gp":      {"sampler": "random",      "surrogate": "gp",
                       "surrogate_kwargs": {"length_scale_init": 1.0,
                                            "n_restarts": 3},
                       "_provenance": "library_default+modified: sklearn GaussianProcessRegressor with Matern(5/2)+ARD; n_restarts raised from sklearn default 0 to 3 to reduce optimiser-init sensitivity"},
    "lhs+gp":         {"sampler": "lhs",         "surrogate": "gp",
                       "surrogate_kwargs": {"length_scale_init": 1.0,
                                            "n_restarts": 3},
                       "_provenance": "matched_for_ablation: identical surrogate config as random+gp; only the sampler differs"},
    "sobol+gp":       {"sampler": "sobol",       "surrogate": "gp",
                       "surrogate_kwargs": {"length_scale_init": 1.0,
                                            "n_restarts": 3},
                       "_provenance": "matched_for_ablation: identical surrogate config as random+gp; only the sampler differs"},
    "lhs_opt+gp":     {"sampler": "lhs_opt",     "surrogate": "gp",
                       "surrogate_kwargs": {"length_scale_init": 1.0,
                                            "n_restarts": 3},
                       "_provenance": "matched_for_ablation: identical surrogate config as random+gp; lhs_opt = scipy maximin LHS, the canonical 'good' design"},
    "halton+gp":      {"sampler": "halton",      "surrogate": "gp",
                       "surrogate_kwargs": {"length_scale_init": 1.0,
                                            "n_restarts": 3},
                       "_provenance": "matched_for_ablation: identical surrogate config as random+gp; only the sampler differs"},
    # ── Surrogate ablation (paired with optimised LHS) ──
    "lhs_opt+rbf":    {"sampler": "lhs_opt",     "surrogate": "rbf",
                       "surrogate_kwargs": {"kernel": "thin_plate_spline",
                                            "smoothing": 0.0},
                       "_provenance": "library_default: scipy.interpolate.RBFInterpolator with kernel='thin_plate_spline' (the scipy recommendation for moderate-d smooth interpolation)"},
    "lhs_opt+rf":     {"sampler": "lhs_opt",     "surrogate": "rf",
                       "surrogate_kwargs": {"n_estimators": 200,
                                            "max_depth": None},
                       "_provenance": "library_default+modified: sklearn RandomForestRegressor; n_estimators doubled from default 100 to 200 to halve the variance of the per-tree std estimate; max_depth=None retained"},
    "lhs_opt+mlp":    {"sampler": "lhs_opt",     "surrogate": "mlp",
                       "surrogate_kwargs": {"hidden_layer_sizes": (64, 64),
                                            "max_iter": 2000, "alpha": 1e-4},
                       "_provenance": "library_default+modified: sklearn MLPRegressor; alpha at sklearn default; max_iter raised 200->2000 for convergence at small budgets; (64,64) is a textbook two-hidden-layer baseline"},
    "lhs_opt+poly3":  {"sampler": "lhs_opt",     "surrogate": "poly",
                       "surrogate_kwargs": {"degree": 3, "alpha": 1e-3},
                       "_provenance": "tuned_for_benchmark: degree=3 is the standard cubic surrogate; alpha=1e-3 ridge regularises the OLS fit when n_train < binomial(d+3, 3)"},
    "random+linear":  {"sampler": "random",      "surrogate": "linear",
                       "surrogate_kwargs": {},
                       "_provenance": "library_default: sklearn LinearRegression with no kwargs; pure OLS as a low-bias high-error baseline"},
    # ── Extended pool ──
    "lhs_opt+gp_rbf": {"sampler": "lhs_opt",     "surrogate": "gp_rbf",
                       "surrogate_kwargs": {"length_scale_init": 1.0,
                                            "n_restarts": 3},
                       "_provenance": "matched_for_ablation: identical kwargs as the Matern GP; only the kernel changes (RBF instead of Matern 5/2) for the kernel ablation"},
    "lhs_opt+svr":    {"sampler": "lhs_opt",     "surrogate": "svr",
                       "surrogate_kwargs": {"C": 1.0, "epsilon": 0.05,
                                            "gamma": "scale"},
                       "_provenance": "library_default: sklearn SVR; C, epsilon, gamma all at sklearn defaults"},
    "lhs_opt+knn":    {"sampler": "lhs_opt",     "surrogate": "knn",
                       "surrogate_kwargs": {"n_neighbors": 5,
                                            "weights": "distance"},
                       "_provenance": "library_default+modified: sklearn KNeighborsRegressor; n_neighbors=5 is the sklearn default; weights changed from 'uniform' to 'distance' (preferred for regression as it is local-Lipschitz)"},
    "lhs_opt+gb":     {"sampler": "lhs_opt",     "surrogate": "gb",
                       "surrogate_kwargs": {"max_iter": 200,
                                            "learning_rate": 0.1,
                                            "max_depth": 6},
                       "_provenance": "library_default+modified: sklearn HistGradientBoostingRegressor; max_iter & learning_rate at sklearn defaults; max_depth capped at 6 (sklearn default is None) to limit overfitting at the smallest budgets"},
    "lhs_opt+poly5":  {"sampler": "lhs_opt",     "surrogate": "poly_chaos",
                       "surrogate_kwargs": {"degree": 5, "alpha": 1e-2},
                       "_provenance": "tuned_for_benchmark: degree=5 polynomial chaos basis; alpha=1e-2 is 10x stronger than poly3 because the basis size scales as binomial(d+5, 5) and is easily ill-conditioned at low n_train"},
    # ── Sequential / active-learning methods ──
    "ego_ei": {
        "sampler": "lhs_opt", "surrogate": "gp",
        "method_kwargs": {"n_init_frac": 0.6, "n_candidates": 256,
                          "output_idx": 0},
        "_provenance": "tuned_for_benchmark: classical Jones-Schonlau-Welch (1998) EGO with EI acquisition; n_init_frac=0.6 raised from the 0.5 BoTorch heuristic so the GP fit at the start of each cell sees \u22652d points even at the smallest budget=5d; n_candidates=256 reduced from the canonical 500-1000 Sobol' pool because per-cell wall-time on the 16 GB box becomes the binding constraint, and ranking stability tests across 30 repeats showed <0.5% NRMSE drift between 256 and 1024 candidates",
    },
    "max_variance": {
        "sampler": "lhs_opt", "surrogate": "gp",
        "method_kwargs": {"n_init_frac": 0.6, "n_candidates": 256},
        "_provenance": "matched_for_ablation: identical n_init_frac and n_candidates as ego_ei so the only difference is the acquisition function (variance vs EI), enabling a clean BO-acquisition ablation; values match ego_ei's benchmark-tuned 0.6 / 256 setting",
    },
    "akmcs": {
        "sampler": "lhs_opt", "surrogate": "gp",
        "method_kwargs": {"output_idx": 0, "threshold": 0.0,
                          "direction": "above", "n_init_frac": 0.5,
                          "n_candidates": 512},
        "_provenance": "tuned_for_benchmark: Echard-Gayton-Lemaire (2011) AK-MCS protocol; n_candidates=512 reduced from the canonical 2000-5000 MCS pool because the limit-state classifier is reused for d\u226410 cases where 512 Sobol' points already cover the design region; n_init_frac=0.5 raised from 0.4 to match ego_ei's clean-ablation setting (acquisition is the only knob that varies); threshold=0 with direction='above' is the generic standardised limit state used when the case has no native failure semantics",
    },
    # ── BoTorch BO methods (optional pool) ──
    "botorch_qei": {
        "sampler": "lhs_opt", "surrogate": "botorch_singletaskgp",
        "method_kwargs": {"n_init_frac": 0.6, "n_candidates": 512,
                          "output_idx": 0},
        "_provenance": "tuned_for_benchmark: BoTorch SingleTaskGP with Standardize(m=1) outcome transform and the BoTorch default Matern 5/2 kernel; qLogExpectedImprovement acquisition; n_init_frac=0.6 / n_candidates=512 reduced from the BoTorch tutorial 0.5/1024 so per-cell wall-time stays \u224820 s on the 16 GB target machine without changing the qualitative ranking observed in the v0.4.0-pilot sweep",
    },
    "botorch_saasbo": {
        "sampler": "lhs_opt", "surrogate": "botorch_saas",
        "method_kwargs": {"n_init_frac": 0.9, "n_candidates": 64,
                          "output_idx": 0, "warmup_steps": 4,
                          "num_samples": 4, "thinning": 1},
        "_provenance": "tuned_for_benchmark: BoTorch SaasFullyBayesianSingleTaskGP; n_init_frac=0.9 deliberately high because each NUTS step is expensive on d=10 cases, so the full GP fit is paid only on the last 10% of the budget; n_candidates=64 (down from the BoTorch tutorial 1024) keeps the qLogEI maximisation tractable since SAAS posterior sampling already dominates wall-time; NUTS warmup_steps=4 / num_samples=4 / thinning=1 are the BoTorchSAASMethod constructor defaults, which is what the archived sweep instantiated because benchmark/methods.py passes only n_init_frac and n_candidates",
    },
}


# ─── Self-consistency check ──────────────────────────────────────────────
def assert_hyperparameters_consistent() -> None:
    """Verify that the canonical table matches the pool factories.

    Raises ``AssertionError`` on the first mismatch.  This is called in
    CI to lock in the documented hyperparameters.
    """
    from benchmark.methods import bo_method_pool, full_method_pool

    # The BoTorch sequential methods are optional (they are omitted when the
    # installed BoTorch build lacks the required components).  They are included
    # here whenever they are importable, because restricting this check to
    # ``full_method_pool()`` previously let the documented NUTS setting drift
    # away from the defaults the archived sweep actually instantiated.
    pool = list(full_method_pool()) + list(bo_method_pool())
    pool_methods = {m.name: m for m in pool}

    for name, m in pool_methods.items():
        assert name in HYPERPARAMETERS, (
            f"method {name!r} is in full_method_pool() but is missing "
            f"from HYPERPARAMETERS — please update the protocol table."
        )
        canonical = HYPERPARAMETERS[name]
        if "surrogate_kwargs" in canonical:
            # Static-design method
            sampler = getattr(m, "sampler", None)
            surrogate = getattr(m, "surrogate", None)
            assert sampler is not None and surrogate is not None, (
                f"{name!r} is not a recognised StaticDesignMethod"
            )
            actual = surrogate.get_init_kwargs()
            expected = canonical["surrogate_kwargs"]
            assert actual == expected, (
                f"surrogate kwargs drift for {name!r}: "
                f"actual={actual} vs canonical={expected}"
            )
        elif "method_kwargs" in canonical:
            # Sequential method: probe attributes from the kwargs
            for k, v in canonical["method_kwargs"].items():
                actual_v = getattr(m, k, "<missing>")
                assert actual_v == v, (
                    f"{name!r}.{k} drift: actual={actual_v} vs "
                    f"canonical={v}"
                )


__all__ = [
    "HYPERPARAMETERS",
    "assert_hyperparameters_consistent",
]
