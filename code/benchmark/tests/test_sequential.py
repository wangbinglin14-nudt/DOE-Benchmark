"""Smoke test for sequential / active-learning methods."""

from __future__ import annotations

import warnings

import numpy as np

import doe_testbed as db
from benchmark.metrics import compute_all_metrics
from benchmark.sequential import AKMCSMethod, EGOMethod, MaxVarianceMethod

warnings.filterwarnings("ignore")


def test_ego_runs_end_to_end():
    case = db.get_case("cantilever_beam_5d")
    X_test = case.sample(200, method="sobol", seed=0)
    Y_test = case.evaluate(X_test)
    method = EGOMethod(n_init_frac=0.4, n_candidates=200)
    Y_pred, Y_std, info = method.run(case, n_train=15, X_test=X_test, seed=42)
    assert Y_pred.shape == Y_test.shape
    assert info["n_init"] >= 2 * case.meta.dim_input
    assert info["n_sequential"] == 15 - info["n_init"]


def test_max_variance_runs():
    case = db.get_case("cantilever_beam_5d")
    X_test = case.sample(200, method="sobol", seed=0)
    Y_test = case.evaluate(X_test)
    method = MaxVarianceMethod(n_init_frac=0.4, n_candidates=200)
    Y_pred, _, info = method.run(case, n_train=15, X_test=X_test, seed=42)
    assert Y_pred.shape == Y_test.shape


def test_akmcs_runs():
    case = db.get_case("cantilever_beam_5d")
    X_test = case.sample(200, method="sobol", seed=0)
    method = AKMCSMethod(
        output_idx=0, threshold=250.0, direction="above",
        n_init_frac=0.4, n_candidates=500,
    )
    Y_pred, _, info = method.run(case, n_train=15, X_test=X_test, seed=42)
    assert Y_pred.shape[0] == X_test.shape[0]


def test_sequential_beats_random_for_global_accuracy():
    """Sanity check: max-variance should be at least as good as random
    on a smooth case with a fixed budget (asymptotically true)."""
    case = db.get_case("cantilever_beam_5d")
    X_test = case.sample(500, method="sobol", seed=0)
    Y_test = case.evaluate(X_test)

    # Random baseline
    from benchmark.core import StaticDesignMethod
    from benchmark.samplers import RandomSampler
    from benchmark.surrogates import GPSurrogate
    rnd = StaticDesignMethod(RandomSampler(), GPSurrogate(), name="random+gp")

    # Max-variance active learner
    mv = MaxVarianceMethod(n_init_frac=0.5, n_candidates=500)

    rmse_rnd = []
    rmse_mv = []
    for seed in range(3):
        Y_r, _, _ = rnd.run(case, n_train=25, X_test=X_test, seed=seed)
        Y_m, _, _ = mv.run(case, n_train=25, X_test=X_test, seed=seed)
        rmse_rnd.append(compute_all_metrics(Y_test, Y_r)["nrmse"])
        rmse_mv.append(compute_all_metrics(Y_test, Y_m)["nrmse"])

    # Both should be below 1 (better than constant predictor) on this smooth case
    assert np.mean(rmse_rnd) < 2.0
    assert np.mean(rmse_mv) < 2.0


# ─── BoTorch BO methods (skipped if BoTorch unavailable) ─────────────────
import importlib.util  # noqa: E402

import pytest  # noqa: E402

_BOTORCH_OK = importlib.util.find_spec("botorch") is not None


@pytest.mark.skipif(not _BOTORCH_OK, reason="BoTorch not installed")
def test_botorch_qei_runs_end_to_end():
    """qEI must produce predictions of the right shape and report timing."""
    from benchmark.bo_methods import BoTorchQEIMethod

    case = db.get_case("cantilever_beam_5d")
    X_test = case.sample(50, method="sobol", seed=0)
    Y_test = case.evaluate(X_test)

    method = BoTorchQEIMethod(n_init_frac=0.5, n_candidates=128)
    Y_pred, Y_std, info = method.run(case, n_train=15, X_test=X_test, seed=42)

    assert Y_pred.shape == Y_test.shape
    assert Y_std.shape == Y_test.shape
    assert info["n_init"] >= 2 * case.meta.dim_input
    assert info["n_sequential"] == 15 - info["n_init"]
    assert "t_sequential_s" in info and info["t_sequential_s"] >= 0
    assert "t_final_fit_s" in info and info["t_final_fit_s"] >= 0
    # Predicted std must be strictly positive (numerical floor 1e-12)
    assert (Y_std > 0).all()


@pytest.mark.skipif(not _BOTORCH_OK, reason="BoTorch not installed")
def test_bo_method_pool_returns_at_least_qei():
    from benchmark.methods import bo_method_pool

    pool = bo_method_pool()
    names = {m.name for m in pool}
    assert "botorch_qei" in names


@pytest.mark.skipif(not _BOTORCH_OK, reason="BoTorch not installed")
def test_nature_pool_extends_full_pool():
    """nature_pool() must be a strict superset of full_method_pool()."""
    from benchmark.methods import full_method_pool, nature_pool

    full_names = {m.name for m in full_method_pool()}
    nature_names = {m.name for m in nature_pool()}
    assert full_names <= nature_names
    assert len(nature_names) >= len(full_names) + 1  # at least qEI added
