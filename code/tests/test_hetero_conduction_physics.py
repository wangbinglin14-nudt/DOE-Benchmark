import numpy as np
import pytest
from pathlib import Path

from doe_testbed.cases.heat_transfer.hetero_conduction_2d import HeterogeneousConduction5D


def test_heterogeneous_conduction_solver_failure_is_not_silent_nan(monkeypatch):
    case = HeterogeneousConduction5D()

    def fail_solve_single(*_args, **_kwargs):
        raise ValueError("simulated heterogeneous-conduction FDM failure")

    monkeypatch.setattr(case, "_solve_single", fail_solve_single)
    x_phys = np.array([[1.0, 10.0, 0.5, 100.0, 20.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_private_physical_input_validator_rejects_before_fdm(monkeypatch):
    case = HeterogeneousConduction5D()
    assert hasattr(case, "_validate_physical_inputs")
    assert "_validate_physical_inputs" in case.meta.solver_details

    with pytest.raises(ValueError, match="2-D array"):
        case._validate_physical_inputs(np.array([1.0, 10.0, 0.5, 100.0, 20.0]))

    def should_not_call_solve_single(*_args, **_kwargs):
        raise AssertionError("_solve_single should not be reached for invalid physical inputs")

    monkeypatch.setattr(case, "_solve_single", should_not_call_solve_single)
    base = np.array([[1.0, 10.0, 0.5, 100.0, 20.0]], dtype=float)
    invalid_values = [
        (0, 0.099),
        (1, 0.999),
        (2, 0.199),
        (3, 59.9),
        (4, -0.1),
        (0, 5.001),
        (1, 50.001),
        (2, 0.801),
        (3, 120.1),
        (4, 30.1),
        (1, np.nan),
    ]

    for col, value in invalid_values:
        x_bad = base.copy()
        x_bad[0, col] = value
        with pytest.raises(ValueError, match="hetero_conduction_2d physical inputs"):
            case._evaluate_core(x_bad)


def test_malformed_heterogeneous_conduction_response_is_not_archived(monkeypatch):
    case = HeterogeneousConduction5D()

    def bad_solve_single(*_args, **_kwargs):
        return np.array([np.nan, -1.0, 50.0])

    monkeypatch.setattr(case, "_solve_single", bad_solve_single)
    x_phys = np.array([[1.0, 10.0, 0.5, 100.0, 20.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_face_flux_uses_harmonic_mean_at_material_jump():
    T = np.array([
        [100.0, 100.0],
        [90.0, 90.0],
    ])
    k_x = np.array([1.0, 100.0])
    h = 0.25

    q_max = HeterogeneousConduction5D._max_face_flux(T, k_x, h)
    harmonic_k = 2.0 * k_x[0] * k_x[1] / (k_x[0] + k_x[1])

    assert abs(q_max - harmonic_k * 10.0 / h) < 1e-12
    assert q_max < 100.0 * 10.0 / h


def test_flux_for_uniform_vertical_conduction_matches_fourier_law():
    T = np.array([
        [100.0, 75.0, 50.0],
        [100.0, 75.0, 50.0],
        [100.0, 75.0, 50.0],
    ])
    k_x = np.array([5.0, 5.0, 5.0])
    h = 0.5

    q_max = HeterogeneousConduction5D._max_face_flux(T, k_x, h)

    assert abs(q_max - 5.0 * 25.0 / h) < 1e-12


def test_declared_boundary_conditions_reduce_to_vertical_conduction_control():
    y = HeterogeneousConduction5D._solve_single(
        k_A=0.1,
        k_B=50.0,
        x_split=0.3,
        T_hot=100.0,
        T_cold=20.0,
        n_grid=25,
    )

    assert np.isclose(y[0], 60.0, rtol=1e-12, atol=1e-12)
    assert np.isclose(y[1], 50.0 * 80.0, rtol=1e-12, atol=1e-10)
    assert np.isclose(y[2], 100.0, rtol=1e-12, atol=1e-12)


def test_x_split_is_inert_for_current_outputs_and_boundary_conditions():
    left_heavy = HeterogeneousConduction5D._solve_single(
        k_A=2.0,
        k_B=10.0,
        x_split=0.2,
        T_hot=90.0,
        T_cold=10.0,
        n_grid=25,
    )
    right_heavy = HeterogeneousConduction5D._solve_single(
        k_A=2.0,
        k_B=10.0,
        x_split=0.8,
        T_hot=90.0,
        T_cold=10.0,
        n_grid=25,
    )

    np.testing.assert_allclose(left_heavy, right_heavy, rtol=1e-12, atol=1e-10)


def test_metadata_declares_nonlinear_response_and_inert_x_split_scope():
    case = HeterogeneousConduction5D()
    text = "\n".join(
        [
            case.__doc__ or "",
            case.meta.governing_equation,
            " ".join(case.meta.scenario_notes.values()),
        ]
    )

    assert case.meta.is_linear is False
    assert "threshold_jump" in case.meta.response_topology
    assert "strong_interaction" not in case.meta.response_topology
    assert case.meta.effective_dim == 4
    assert "x_split" not in case.meta.important_variables
    assert "x_split" in text
    assert "inert" in text


def test_solve_single_keeps_qmax_finite_for_large_conductivity_contrast():
    y = HeterogeneousConduction5D._solve_single(
        k_A=0.1,
        k_B=50.0,
        x_split=0.5,
        T_hot=100.0,
        T_cold=0.0,
        n_grid=25,
    )

    assert np.isfinite(y).all()
    assert y[1] > 0.0
    assert y[1] <= 5000.0 + 1e-6


def test_negative_conductivity_perturbation_is_clipped_to_positive_floor():
    y = HeterogeneousConduction5D._solve_single(
        k_A=-1e-9,
        k_B=10.0,
        x_split=0.5,
        T_hot=80.0,
        T_cold=20.0,
        n_grid=25,
    )

    assert np.isfinite(y).all()


def test_hetero_conduction_audit_keeps_20260608_boundary_honest():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `hetero_conduction_2d` 2026-06-08 official-source boundary tightening:" in audit_text
    assert "current COMSOL heat-transfer theory material" in normalized
    assert "physical meaning of positive thermal conductivity and heat flow from higher to lower temperature" in normalized
    assert "boundary between the implemented harmonic-face heterogeneous conduction control" in normalized
    assert "omitted engineering workflows such as contact resistance" in normalized
    assert "temperature-dependent properties" in normalized
    assert "multidimensional thermal-bridge calibration" in normalized
    assert "do not upgrade the case from analytical control plus targeted regression" in normalized
    assert "No numerical response resweep is needed solely for this source-boundary tightening" in normalized
