"""
benchmark.methods
=================
Concrete :class:`BenchmarkMethod` instances and convenience constructors.

Five named pools are provided:

- ``default_method_pool()``    — 10 static-design methods (original pilot pool)
- ``extended_method_pool()``   — 15 methods: default + 5 surrogate families
                                  (4 new + 1 GP-kernel ablation)
- ``full_method_pool()``       — 18 methods: extended + 3 sequential learners
- ``bo_method_pool()``         — 2 BoTorch BO methods (qEI, SAASBO)
- ``nature_pool()``            — 20 methods: full + bo (FROZEN at v0.4.0-prereg)

The pre-registration freeze (paper/preregistration.md §3) requires
``nature_pool()`` to stay at exactly 20 methods.  Two **post-registration**
pools extend the comparison without modifying the headline:

- ``sota_pool()``              — nature + 2 SOTA additions (TURBO, D-optimal)
                                  [robustness / amendment experiment]
- ``mf_pool()``                — multi-fidelity-only methods, applicable to
                                  the 5 cases with ``supports_multifidelity=True``

Each method is identified by a stable string key (e.g. ``"lhs+gp"``)
for downstream analysis.
"""

from __future__ import annotations

from typing import List

from benchmark.core import BenchmarkMethod, StaticDesignMethod
from benchmark.samplers import (
    DOptimalSampler,
    HaltonSampler,
    LHSSampler,
    OptimisedLHSSampler,
    RandomSampler,
    SobolSampler,
)
from benchmark.sequential import AKMCSMethod, EGOMethod, MaxVarianceMethod
from benchmark.surrogates import (
    GPRBFSurrogate,
    GPSurrogate,
    GradientBoostingSurrogate,
    KNNSurrogate,
    LinearSurrogate,
    MLPSurrogate,
    PolyChaosSurrogate,
    PolynomialSurrogate,
    RandomForestSurrogate,
    RBFSurrogate,
    SVRSurrogate,
)


def default_method_pool() -> List[BenchmarkMethod]:
    """The original 10-method pool used in the pilot experiment.

    - 5 sampler ablations × GP (Matérn 5/2)
    - 4 surrogate ablations × OptLHS
    - 1 low-bias baseline (Random + Linear)
    """
    methods: List[BenchmarkMethod] = []

    # ── Sampler ablation: hold surrogate (GP) fixed, vary sampler ──
    methods.append(StaticDesignMethod(RandomSampler(), GPSurrogate(), "random+gp"))
    methods.append(StaticDesignMethod(LHSSampler(), GPSurrogate(), "lhs+gp"))
    methods.append(StaticDesignMethod(SobolSampler(), GPSurrogate(), "sobol+gp"))
    methods.append(StaticDesignMethod(OptimisedLHSSampler(), GPSurrogate(), "lhs_opt+gp"))
    methods.append(StaticDesignMethod(HaltonSampler(), GPSurrogate(), "halton+gp"))

    # ── Surrogate ablation: hold sampler (opt-LHS) fixed, vary surrogate ──
    methods.append(StaticDesignMethod(OptimisedLHSSampler(), RBFSurrogate(), "lhs_opt+rbf"))
    methods.append(StaticDesignMethod(OptimisedLHSSampler(), RandomForestSurrogate(), "lhs_opt+rf"))
    methods.append(StaticDesignMethod(OptimisedLHSSampler(), MLPSurrogate(), "lhs_opt+mlp"))
    methods.append(StaticDesignMethod(OptimisedLHSSampler(), PolynomialSurrogate(degree=3), "lhs_opt+poly3"))

    # ── Baselines ──
    methods.append(StaticDesignMethod(RandomSampler(), LinearSurrogate(), "random+linear"))

    return methods


def extended_method_pool() -> List[BenchmarkMethod]:
    """15-method pool: default pool + 4 new surrogate families + 1 kernel ablation.

    Adds:
    - SVR (Support Vector Regression with RBF kernel)
    - KNN (k-nearest neighbours, distance-weighted)
    - Gradient Boosting (HistGradientBoostingRegressor)
    - PolyChaos (degree-5 polynomial regression with Ridge)
    - GP-RBF (kernel ablation against GP-Matérn)

    Use this pool to study how surrogate choice affects rankings.
    """
    methods = default_method_pool()
    sampler = OptimisedLHSSampler()
    methods.extend([
        StaticDesignMethod(sampler, GPRBFSurrogate(), "lhs_opt+gp_rbf"),
        StaticDesignMethod(sampler, SVRSurrogate(), "lhs_opt+svr"),
        StaticDesignMethod(sampler, KNNSurrogate(), "lhs_opt+knn"),
        StaticDesignMethod(sampler, GradientBoostingSurrogate(), "lhs_opt+gb"),
        StaticDesignMethod(sampler, PolyChaosSurrogate(degree=5), "lhs_opt+poly5"),
    ])
    return methods


def full_method_pool() -> List[BenchmarkMethod]:
    """18-method pool: extended pool + 3 sequential / active-learning methods.

    Adds:
    - EGO with Expected Improvement (Jones et al. 1998)
    - Maximum predictive variance (uncertainty-driven AL)
    - AK-MCS (failure-boundary oriented; uses output_idx=0)

    NOTE: sequential methods take ~10× more wall-clock per run than
    static methods (they fit a GP at every step).  Use a small
    case set or a smaller budget for development.
    """
    methods = extended_method_pool()
    methods.extend([
        EGOMethod(n_init_frac=0.6, n_candidates=256),
        MaxVarianceMethod(n_init_frac=0.6, n_candidates=256),
        # AK-MCS with a generic threshold of 0 (after Y standardisation)
        # — for global accuracy the threshold doesn't materially matter
        AKMCSMethod(output_idx=0, threshold=0.0, direction="above",
                    n_init_frac=0.5, n_candidates=512),
    ])
    return methods


def bo_method_pool() -> List[BenchmarkMethod]:
    """BoTorch-based Bayesian-optimisation methods (qEI + SAASBO).

    Returns an empty list if BoTorch is not installed; this lets callers
    safely union it with another pool without forcing the heavy
    ``torch`` / ``botorch`` dependency on every user.

    Notes
    -----
    SAASBO is **expensive** (NUTS posterior sampling each step,
    ~30–60 s/step).  Use only when ``d ≥ 10`` cases dominate the
    workload.  For routine use prefer ``BoTorchQEIMethod`` alone.
    """
    try:
        from benchmark.bo_methods import BoTorchQEIMethod, BoTorchSAASMethod
    except ImportError:
        return []

    try:
        methods: List[BenchmarkMethod] = [
            BoTorchQEIMethod(n_init_frac=0.6, n_candidates=512),
        ]
    except ImportError:
        return []
    try:
        methods.append(BoTorchSAASMethod(n_init_frac=0.9, n_candidates=64))
    except ImportError:
        # SAAS unavailable in this BoTorch build; skip silently.
        pass
    return methods


def nature_pool() -> List[BenchmarkMethod]:
    """Recommended Nature-grade pool: full pool + BoTorch BO if available.

    Equivalent to ``full_method_pool()`` plus ``bo_method_pool()``.
    Yields the maximum-coverage 18 / 20-method comparison set used in
    the published benchmark.

    **FROZEN at v0.4.0-prereg.**  Adding methods here counts as a
    pre-registration amendment and breaks
    ``benchmark/tests/test_prereg_compliance.py``.  For new methods
    use ``sota_pool()`` (post-registration extension).
    """
    return full_method_pool() + bo_method_pool()


# ─── Post-registration extensions ─────────────────────────────────────────
def sota_pool() -> List[BenchmarkMethod]:
    """Post-registration SOTA pool: nature + TURBO + D-optimal.

    Adds two methods that the pre-registration omitted (the freeze
    happened before TURBO was added to BoTorch's stable API and
    D-optimal was reviewed for our case suite):

    - **TURBO** (Eriksson et al. 2019, NeurIPS) — trust-region BO,
      the canonical SOTA baseline for moderate-to-high-dim BO.
    - **D-optimal** (Atkinson--Donev--Tobias 2007) — classical
      model-based design, the standard answer to
      *"where is your D-optimal baseline?"*

    Sweeps using this pool are **post-registration robustness
    experiments**.  Their results are reported separately in
    Extended Data, with the headline (Discovery 1-3) values still
    coming from the frozen ``nature_pool()`` 30-repeat sweep.
    Cross-link in ``paper/preregistration_amendments.md``.
    """
    pool = nature_pool()
    # Only add SOTA extras when the underlying optional deps are present
    try:
        from benchmark.bo_methods import BoTorchTurboMethod
        pool.append(BoTorchTurboMethod(n_init_frac=0.5, n_candidates=256))
    except ImportError:
        pass
    pool.append(StaticDesignMethod(DOptimalSampler(), GPSurrogate(), "doptimal+gp"))
    return pool


def mf_pool() -> List[BenchmarkMethod]:
    """Multi-fidelity methods.  Applicable only to MF-capable cases
    (``supports_multifidelity=True`` in CaseMeta).

    Returns an empty list if BoTorch is not installed.

    Currently exposes:

    - **MF Cokriging** (Forrester--Sobester--Keane 2007) — linear
      two-fidelity additive cokriging GP, the canonical MF baseline.
    """
    try:
        from benchmark.bo_methods import MFCokrigingMethod
    except ImportError:
        return []
    try:
        return [MFCokrigingMethod(frac_low_cost=0.5)]
    except ImportError:
        return []


# ─── Convenience lookup ────────────────────────────────────────────────────
def list_methods(pool: str = "default") -> List[str]:
    """Return method names for the named pool.

    Parameters
    ----------
    pool : 'default' | 'extended' | 'full' | 'bo' | 'nature' | 'sota' | 'mf'
    """
    if pool == "default":
        return [m.name for m in default_method_pool()]
    if pool == "extended":
        return [m.name for m in extended_method_pool()]
    if pool == "full":
        return [m.name for m in full_method_pool()]
    if pool == "bo":
        return [m.name for m in bo_method_pool()]
    if pool == "nature":
        return [m.name for m in nature_pool()]
    if pool == "sota":
        return [m.name for m in sota_pool()]
    if pool == "mf":
        return [m.name for m in mf_pool()]
    raise ValueError(f"unknown pool {pool!r}")
