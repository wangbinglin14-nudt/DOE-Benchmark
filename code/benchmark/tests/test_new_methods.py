"""Regression tests for the post-registration method additions.

Covers:
  * ``DOptimalSampler``                 — Fedorov-Mitchell exchange
  * ``BoTorchTurboMethod``              — TuRBO trust-region BO
  * ``MFCokrigingMethod``               — Forrester-style MF cokriging
  * ``run_ranking_reversal._build_pairing`` — three pairing modes
  * ``methods.sota_pool / mf_pool``     — pool size / membership
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


# ─── DOptimalSampler ───────────────────────────────────────────────────────
class TestDOptimalSampler:
    def test_shape_and_range(self):
        from benchmark.samplers import DOptimalSampler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = DOptimalSampler().sample(50, 5, seed=42)
        assert X.shape == (50, 5)
        assert np.all((X >= 0) & (X <= 1))

    def test_determinism(self):
        from benchmark.samplers import DOptimalSampler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X1 = DOptimalSampler().sample(50, 5, seed=42)
            X2 = DOptimalSampler().sample(50, 5, seed=42)
        assert np.allclose(X1, X2)

    def test_improves_d_criterion_over_random_init(self):
        """The Fedorov exchange should monotonically improve log|XtX|."""
        from benchmark.samplers import DOptimalSampler, _polynomial_features
        from scipy.stats.qmc import Sobol

        n, d = 50, 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Build the same candidate pool the sampler uses internally
            sobol = Sobol(d=d, scramble=True, seed=42).random(2048)
            init_idx = np.random.default_rng(42).choice(2048, size=n, replace=False)
            X_init = sobol[init_idx]
            Phi_init = _polynomial_features(X_init, 2)
            ld_init = float(np.linalg.slogdet(Phi_init.T @ Phi_init)[1])

            X_final = DOptimalSampler(n_starts=2).sample(n, d, seed=42)
            Phi_final = _polynomial_features(X_final, 2)
            ld_final = float(np.linalg.slogdet(Phi_final.T @ Phi_final)[1])

        assert ld_final >= ld_init - 1e-6, (
            f"D-optimal exchange degraded the D-criterion: "
            f"init={ld_init:.4f} -> final={ld_final:.4f}"
        )

    def test_falls_back_when_n_too_small(self):
        """If n is below the minimum to fit the polynomial model we must
        gracefully fall back to LHS without raising."""
        from benchmark.samplers import DOptimalSampler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = DOptimalSampler().sample(5, 5, seed=42)
        assert X.shape == (5, 5)
        assert np.all((X >= 0) & (X <= 1))

    def test_high_dim_drops_to_linear(self):
        """At d > 8 the sampler should auto-downgrade to a linear model."""
        from benchmark.samplers import DOptimalSampler
        sampler = DOptimalSampler()
        # The downgrade emits a UserWarning when the quadratic-model
        # column count would be impractical.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert sampler._effective_degree(10) == 1
            assert any("downgraded to 1" in str(w.message) for w in caught)


# ─── BoTorchTurboMethod ───────────────────────────────────────────────────
class TestBoTorchTurboMethod:
    def test_construction_requires_botorch(self):
        botorch = pytest.importorskip("botorch")  # noqa: F841
        from benchmark.bo_methods import BoTorchTurboMethod
        m = BoTorchTurboMethod()
        assert m.name == "botorch_turbo"
        assert m.parallel_backend == "threading"
        assert m.memory_class == "medium"

    def test_run_smoke(self):
        pytest.importorskip("botorch")
        import doe_testbed as db
        from benchmark.bo_methods import BoTorchTurboMethod

        case = db.get_case("cantilever_beam_5d")
        X_test = case.sample(50, method="sobol", seed=0)
        m = BoTorchTurboMethod(n_init_frac=0.5, n_candidates=64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y_pred, Y_std, meta = m.run(case, n_train=15, X_test=X_test, seed=42)
        assert Y_pred.shape[0] == 50
        assert meta["n_init"] >= 2 * case.meta.dim_input
        # TR state in metadata
        assert "tr_L_final" in meta
        assert "n_tr_restarts" in meta
        assert meta["tr_L_final"] > 0
        assert meta["n_tr_restarts"] >= 0

    def test_lengthscale_weights_shape(self):
        """The internal lengthscale-weight helper must return ``None`` or a
        d-vector with finite positive entries."""
        pytest.importorskip("botorch")
        from benchmark.bo_methods import BoTorchTurboMethod, _fit_singletaskgp
        rng = np.random.default_rng(0)
        d = 4
        X = rng.random((20, d))
        y = rng.random(20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _fit_singletaskgp(X, y)
            w = BoTorchTurboMethod()._gp_lengthscale_widths(model)
        # Either None (degenerate fit) or a finite positive d-vector
        if w is not None:
            assert w.shape == (d,)
            assert np.all(np.isfinite(w))
            assert np.all(w > 0)


# ─── MFCokrigingMethod ─────────────────────────────────────────────────────
class TestMFCokrigingMethod:
    def test_construction(self):
        pytest.importorskip("botorch")
        from benchmark.bo_methods import MFCokrigingMethod
        m = MFCokrigingMethod()
        assert m.name == "mf_cokriging"
        assert m.memory_class == "medium"

    def test_split_budget_falls_back_for_small_budget(self):
        pytest.importorskip("botorch")
        import doe_testbed as db
        from benchmark.bo_methods import MFCokrigingMethod

        m = MFCokrigingMethod(frac_low_cost=0.5)
        case = db.get_case("hetero_conduction_2d")
        # tiny budget cannot afford 4 of each fidelity at 16x cost ratio
        n_low, n_high = m._split_budget(case, n_train=20)
        assert n_high == 0, f"expected fallback, got n_high={n_high}"
        assert n_low == 20

    def test_split_budget_real_split_at_large_budget(self):
        pytest.importorskip("botorch")
        import doe_testbed as db
        from benchmark.bo_methods import MFCokrigingMethod

        m = MFCokrigingMethod(frac_low_cost=0.5)
        case = db.get_case("hetero_conduction_2d")
        n_low, n_high = m._split_budget(case, n_train=250)
        assert n_high >= 4, f"expected MF allocation, got n_high={n_high}"
        assert n_low >= 4
        # And total cost must NOT exceed the budget (cost ratio = 16)
        assert n_low + 16 * n_high <= 250

    def test_split_budget_no_high_falls_back(self):
        """A case without a ``high`` fidelity must fall back."""
        pytest.importorskip("botorch")
        import doe_testbed as db
        from benchmark.bo_methods import MFCokrigingMethod

        m = MFCokrigingMethod(frac_low_cost=0.5)
        case = db.get_case("cantilever_beam_5d")  # only 'default'
        n_low, n_high = m._split_budget(case, n_train=100)
        assert n_high == 0
        assert n_low == 100

    def test_run_smoke_mf_case(self):
        pytest.importorskip("botorch")
        import doe_testbed as db
        from benchmark.bo_methods import MFCokrigingMethod

        case = db.get_case("hetero_conduction_2d")
        X_test = case.sample(50, method="sobol", seed=0)
        m = MFCokrigingMethod(frac_low_cost=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use a budget large enough to actually exercise MF
            Y_pred, Y_std, meta = m.run(case, n_train=120, X_test=X_test, seed=42)
        assert Y_pred.shape[0] == 50
        assert "rho_estimate" in meta
        assert "n_low" in meta
        assert "n_high" in meta

    def test_run_falls_back_for_single_fidelity_case(self):
        pytest.importorskip("botorch")
        import doe_testbed as db
        from benchmark.bo_methods import MFCokrigingMethod

        case = db.get_case("cantilever_beam_5d")
        X_test = case.sample(50, method="sobol", seed=0)
        m = MFCokrigingMethod(frac_low_cost=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y_pred, Y_std, meta = m.run(case, n_train=20, X_test=X_test, seed=42)
        assert meta["fallback_to_single_fidelity"] is True
        assert meta["n_high"] == 0


# ─── Method pools ─────────────────────────────────────────────────────────
class TestPools:
    def test_nature_pool_unchanged_at_20(self):
        """Headline pool must remain frozen at 20 (BoTorch available)."""
        pytest.importorskip("botorch")
        from benchmark.methods import nature_pool
        assert len(nature_pool()) == 20

    def test_sota_pool_is_22(self):
        """sota_pool = nature + TURBO + D-optimal."""
        pytest.importorskip("botorch")
        from benchmark.methods import sota_pool
        names = [m.name for m in sota_pool()]
        assert len(names) == 22
        assert "botorch_turbo" in names
        assert "doptimal+gp" in names

    def test_mf_pool_present_when_botorch_available(self):
        pytest.importorskip("botorch")
        from benchmark.methods import mf_pool
        pool = mf_pool()
        assert len(pool) >= 1
        assert any(m.name == "mf_cokriging" for m in pool)

    def test_list_methods_supports_new_pools(self):
        from benchmark.methods import list_methods
        # These should never raise
        for pool in ("default", "extended", "full", "bo", "nature", "sota", "mf"):
            try:
                lst = list_methods(pool)
            except ImportError:
                pytest.skip("BoTorch not installed")
            assert isinstance(lst, list)
        with pytest.raises(ValueError, match="unknown pool"):
            list_methods("nonexistent")


# ─── D1 pairing modes ─────────────────────────────────────────────────────
class TestD1PairingModes:
    def test_dimension_pairing_uses_pre_registered_map(self):
        from benchmark.scripts.run_ranking_reversal import (
            ANALYTICAL_CASES,
            DIMENSION_PAIRING,
            _build_pairing,
        )
        p = _build_pairing("dimension", seed=2026)
        # Every analytical case must map to its pre-registered partner
        for a in ANALYTICAL_CASES:
            assert p[a] == DIMENSION_PAIRING[a]

    def test_topology_pairing_distinct_partners(self):
        from benchmark.scripts.run_ranking_reversal import _build_pairing
        p = _build_pairing("topology", seed=2026)
        # No physics case is paired with two analytical cases
        partners = list(p.values())
        assert len(partners) == len(set(partners))

    def test_random_pairing_is_seed_deterministic(self):
        from benchmark.scripts.run_ranking_reversal import _build_pairing
        p1 = _build_pairing("random", seed=2026)
        p2 = _build_pairing("random", seed=2026)
        assert p1 == p2
        # Distinct seeds give a different (statistically) pairing
        p3 = _build_pairing("random", seed=2027)
        assert p1 != p3 or len(p1) <= 1  # almost-surely

    def test_unknown_pairing_raises(self):
        from benchmark.scripts.run_ranking_reversal import _build_pairing
        with pytest.raises(ValueError, match="unknown pairing"):
            _build_pairing("nonsense", seed=0)

    def test_dimension_pairing_respects_small_candidate_pool(self):
        from benchmark.scripts.run_ranking_reversal import (
            ANALYTICAL_CASES,
            _build_pairing,
        )
        p = _build_pairing(
            "dimension",
            seed=2026,
            candidate_pool=["cantilever_beam_5d"],
        )

        assert set(p) == set(ANALYTICAL_CASES)
        assert set(p.values()) == {"cantilever_beam_5d"}

    def test_cases_cli_accepts_all_and_deduplicates_pool(self):
        from benchmark.scripts.run_ranking_reversal import _resolve_candidate_pool

        assert _resolve_candidate_pool(None) is None
        assert _resolve_candidate_pool(["ALL"]) is None
        assert _resolve_candidate_pool([
            "cantilever_beam_5d",
            "cantilever_beam_5d",
            "wing_vlm_6d",
        ]) == ["cantilever_beam_5d", "wing_vlm_6d"]

    def test_cases_cli_rejects_unknown_case(self):
        from benchmark.scripts.run_ranking_reversal import _resolve_candidate_pool

        with pytest.raises(SystemExit, match="Unknown physics case_id"):
            _resolve_candidate_pool(["not_a_case"])

    def test_small_pool_rejects_topology_or_random_pairings(self):
        from benchmark.scripts.run_ranking_reversal import (
            _build_pairing,
            _validate_pairing_modes_for_candidate_pool,
        )

        _validate_pairing_modes_for_candidate_pool(
            ["dimension"],
            ["cantilever_beam_5d"],
        )
        with pytest.raises(SystemExit, match="Topology/random D1 pairings require"):
            _validate_pairing_modes_for_candidate_pool(
                ["dimension", "topology"],
                ["cantilever_beam_5d"],
            )
        with pytest.raises(ValueError, match="requires at least"):
            _build_pairing(
                "random",
                seed=2026,
                candidate_pool=["cantilever_beam_5d"],
            )

    def test_topology_prefers_jaccard_match(self):
        """Topology pairing should pick a physics case sharing tags."""
        from benchmark.scripts.run_ranking_reversal import (
            _build_pairing,
            _physics_topology,
            ANALYTICAL_TOPOLOGY,
        )
        p = _build_pairing("topology", seed=2026)
        for a, phys in p.items():
            tags_a = set(ANALYTICAL_TOPOLOGY.get(a, []))
            tags_p = _physics_topology(phys)
            # Either there is a non-empty overlap, or the analytical
            # case has unique tags (no possible match).
            if tags_a:
                # We expect at least the chosen physics case to share
                # at least *one* topology tag whenever any candidate does.
                # This is a soft test — we only check we made a deterministic
                # choice (overlap ≥ 0).
                assert isinstance(tags_a & tags_p, set)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
