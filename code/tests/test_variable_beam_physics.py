import numpy as np
import pytest

from doe_testbed.cases.structures.variable_beam_nd import (
    VariableBeamBase,
    VariableBeam5D,
    VariableBeam10D,
    VariableBeam20D,
)


def test_micro_negative_eigenvalues_are_clipped_to_zero():
    K = np.eye(2)
    M = np.eye(2)
    eigvals = np.array([-1e-14, 4.0])

    sanitized = VariableBeamBase._sanitize_eigenvalues(eigvals, K, M)

    np.testing.assert_array_equal(sanitized, np.array([0.0, 4.0]))


def test_significant_negative_eigenvalues_raise_instead_of_being_hidden():
    K = np.eye(2)
    M = np.eye(2)
    eigvals = np.array([-1e-6, 4.0])

    with pytest.raises(ValueError, match="Negative generalized eigenvalue"):
        VariableBeamBase._sanitize_eigenvalues(eigvals, K, M)


def test_fem_beam_outputs_remain_ordered_and_finite_after_eigenvalue_sanitization():
    case = VariableBeam5D()

    f1, f2, mass = case._fem_beam(np.full(5, 0.08), 5)

    assert np.isfinite([f1, f2, mass]).all()
    assert f1 > 0.0
    assert f2 > f1
    assert mass > 0.0


@pytest.mark.parametrize("case_cls", [VariableBeam5D, VariableBeam10D, VariableBeam20D])
def test_fem_solver_failure_is_not_silent_nan(case_cls, monkeypatch):
    case = case_cls()

    def fail_solve(*_args, **_kwargs):
        raise ValueError("simulated beam eigen-solve failure")

    monkeypatch.setattr(case, "_fem_beam", fail_solve)
    X_phys = np.full((1, case.meta.dim_input), 0.08, dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


@pytest.mark.parametrize("case_cls", [VariableBeam5D, VariableBeam10D, VariableBeam20D])
def test_metadata_marks_input_output_response_as_nonlinear(case_cls):
    case = case_cls()

    assert case.meta.is_linear is False
    assert "sparse_effects" in case.meta.response_topology
    assert hasattr(case, "_validate_physical_inputs")
    assert "named _validate_physical_inputs private physical-input evaluation rejects" in case.meta.solver_details
    assert "before Hermite beam FEM assembly" in case.meta.solver_details


@pytest.mark.parametrize("case_cls", [VariableBeam5D, VariableBeam10D, VariableBeam20D])
def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_fem(
    case_cls,
    monkeypatch,
):
    case = case_cls()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Hermite beam FEM should not be called for invalid input")

    monkeypatch.setattr(case, "_fem_beam", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._validate_physical_inputs(np.full(case.meta.dim_input, 0.08, dtype=float))

    for bad_value in (0.019, 0.151, np.nan):
        x_bad = np.full((1, case.meta.dim_input), 0.08, dtype=float)
        x_bad[0, 0] = bad_value
        with pytest.raises(ValueError, match="segment-height benchmark domain"):
            case._evaluate_core(x_bad)


@pytest.mark.parametrize(
    ("case_cls", "n_segments", "rtol_f2"),
    [
        (VariableBeam5D, 5, 2e-3),
        (VariableBeam10D, 10, 2e-4),
        (VariableBeam20D, 20, 2e-5),
    ],
)
def test_uniform_beam_matches_euler_bernoulli_cantilever_frequencies(
    case_cls,
    n_segments,
    rtol_f2,
):
    case = case_cls()
    h = 0.08

    f1, f2, _ = case._fem_beam(np.full(n_segments, h), n_segments)

    I = case._B * h**3 / 12.0
    A = case._B * h
    freq_scale = np.sqrt(case._E * I / (case._RHO * A)) / (2.0 * np.pi * case._L**2)
    exact_f1 = 1.875104068711961**2 * freq_scale
    exact_f2 = 4.694091132974174**2 * freq_scale

    assert abs(f1 - exact_f1) / exact_f1 < 1e-3
    assert abs(f2 - exact_f2) / exact_f2 < rtol_f2
