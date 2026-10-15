"""Tests for the sensitivity_recovery module."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from benchmark.sensitivity_recovery import (
    compare_sensitivity,
    get_ground_truth_st,
    surrogate_response_range,
    surrogate_sobol_indices,
)

warnings.filterwarnings("ignore")


# ─── compare_sensitivity ─────────────────────────────────────────────────
def test_compare_sensitivity_perfect_match():
    true_st = np.array([0.5, 0.3, 0.2])
    pred_st = true_st.copy()
    out = compare_sensitivity(true_st, pred_st)
    assert out["kendall_tau"] == pytest.approx(1.0)
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["top1_correct"] == 1.0
    assert out["top3_jaccard"] == 1.0
    assert out["L1"] == pytest.approx(0.0)
    assert out["L2"] == pytest.approx(0.0)


def test_compare_sensitivity_inverted_ranking():
    true_st = np.array([0.5, 0.3, 0.2])
    pred_st = np.array([0.2, 0.3, 0.5])  # reverse order
    out = compare_sensitivity(true_st, pred_st)
    assert out["kendall_tau"] < 0  # negative correlation
    assert out["top1_correct"] == 0.0  # argmax flipped
    # L1 = |.5-.2| + |.3-.3| + |.2-.5| = 0.6
    assert out["L1"] == pytest.approx(0.6)


def test_compare_sensitivity_handles_nans():
    true_st = np.array([0.5, 0.3, 0.2])
    pred_st = np.array([np.nan, 0.3, 0.2])
    out = compare_sensitivity(true_st, pred_st)
    # NaN replaced with 0; argmax of pred is then index 1
    assert out["argmax_pred"] == 1
    assert out["argmax_true"] == 0
    assert out["top1_correct"] == 0.0


# ─── surrogate_sobol_indices ─────────────────────────────────────────────
def test_surrogate_sobol_known_function():
    """For a known function f(x1,x2,x3) = x1, only x1 should matter."""
    from benchmark.surrogates import GPSurrogate
    rng = np.random.default_rng(0)
    X_train = rng.random((100, 3))
    Y_train = X_train[:, [0]]  # depends only on x1
    surr = GPSurrogate()
    surr.fit(X_train, Y_train)

    st = surrogate_sobol_indices(surr, n_inputs=3, n_base=128, seed=0)
    assert st is not None
    assert st.shape == (3,)
    # x1 should dominate
    assert np.argmax(st) == 0
    # x2 and x3 should have ST close to 0
    assert st[1] < 0.2
    assert st[2] < 0.2


def test_surrogate_sobol_two_active_variables():
    """f(x) = x1 + 2*x2 -- both should be important, but x2 more."""
    from benchmark.surrogates import RBFSurrogate
    rng = np.random.default_rng(1)
    X_train = rng.random((200, 4))
    Y_train = (X_train[:, [0]] + 2 * X_train[:, [1]])
    surr = RBFSurrogate()
    surr.fit(X_train, Y_train)
    st = surrogate_sobol_indices(surr, n_inputs=4, n_base=256, seed=0)
    assert st is not None
    assert st.shape == (4,)
    # x2 should be most important
    assert np.argmax(st[:2]) == 1
    # x3 and x4 should have negligible ST
    assert st[2] < 0.1
    assert st[3] < 0.1


# ─── get_ground_truth_st ─────────────────────────────────────────────────
def test_ground_truth_st_for_real_case():
    import doe_testbed as db
    case = db.get_case("cantilever_beam_5d")
    st = get_ground_truth_st(case.meta)
    assert st is not None
    assert st.shape == (5,)
    # All values should be non-negative
    assert (st >= 0).all()
    # Should sum to roughly the variance fraction
    assert st.sum() > 0


# ─── surrogate_response_range ────────────────────────────────────────────
def test_response_range_detects_constant_surrogate():
    """A surrogate fit on a constant target should have ~0 response std."""
    from benchmark.surrogates import GPSurrogate
    rng = np.random.default_rng(0)
    X = rng.random((50, 3))
    Y = np.ones((50, 1))  # constant target
    s = GPSurrogate()
    s.fit(X, Y)
    rr = surrogate_response_range(s, n_inputs=3, output_idx=0, n_samples=200)
    # std should be near zero
    assert rr["y_std"] < 0.05


def test_response_range_for_active_surrogate():
    """A surrogate fit on a varying target should have non-zero response std."""
    from benchmark.surrogates import RBFSurrogate
    rng = np.random.default_rng(1)
    X = rng.random((100, 3))
    Y = (X[:, 0] + X[:, 1]).reshape(-1, 1)  # genuine variation
    s = RBFSurrogate()
    s.fit(X, Y)
    rr = surrogate_response_range(s, n_inputs=3, output_idx=0, n_samples=500)
    # std should be substantial
    assert rr["y_std"] > 0.1
