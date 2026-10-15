import numpy as np
import pytest

from doe_testbed.cases.structures.cantilever_beam_5d import CantileverBeam5D


def _domain_corners(case):
    lb = np.array([v.lower for v in case.meta.input_variables], dtype=float)
    ub = np.array([v.upper for v in case.meta.input_variables], dtype=float)
    return np.array(np.meshgrid(*zip(lb, ub), indexing="ij")).reshape(5, -1).T


def _evaluate_single(F_kN, L, b, h, E_GPa):
    case = CantileverBeam5D()
    X_phys = np.array([[F_kN, L, b, h, E_GPa]])
    return case.evaluate(case.normalize(X_phys))[0]


def _euler_bernoulli_delta_mm(F_kN, L, b, h, E_GPa):
    F = F_kN * 1e3
    E = E_GPa * 1e9
    I_z = b * h**3 / 12.0
    return F * L**3 / (3.0 * E * I_z) * 1e3


def _shear_delta_mm(F_kN, L, b, h, E_GPa):
    F = F_kN * 1e3
    E = E_GPa * 1e9
    G = E / (2.0 * (1.0 + 0.30))
    kappa = 5.0 / 6.0
    return F * L / (kappa * G * b * h) * 1e3


def test_short_deep_beam_deflection_includes_timoshenko_shear_term():
    F_kN, L, b, h, E_GPa = 50.0, 0.5, 0.3, 0.5, 60.0

    _, delta_mm, _ = _evaluate_single(F_kN, L, b, h, E_GPa)
    bending = _euler_bernoulli_delta_mm(F_kN, L, b, h, E_GPa)
    shear = _shear_delta_mm(F_kN, L, b, h, E_GPa)

    np.testing.assert_allclose(delta_mm, bending + shear, rtol=1e-12)
    assert shear / bending > 0.10


def test_slender_beam_shear_correction_is_small():
    F_kN, L, b, h, E_GPa = 10.0, 3.0, 0.1, 0.1, 200.0

    _, delta_mm, _ = _evaluate_single(F_kN, L, b, h, E_GPa)
    bending = _euler_bernoulli_delta_mm(F_kN, L, b, h, E_GPa)
    shear = _shear_delta_mm(F_kN, L, b, h, E_GPa)

    np.testing.assert_allclose(delta_mm, bending + shear, rtol=1e-12)
    assert shear / bending < 0.01


def test_shear_correction_does_not_change_bending_or_shear_stresses():
    y = _evaluate_single(10.0, 1.5, 0.1, 0.2, 200.0)

    assert np.isclose(y[0], 22.5)
    assert np.isclose(y[2], 0.75)


def test_metadata_marks_input_output_response_as_nonlinear():
    case = CantileverBeam5D()

    assert case.meta.is_linear is False
    assert "strong_interaction" in case.meta.response_topology


def test_declared_output_ranges_cover_domain_corners():
    case = CantileverBeam5D()
    y = case.evaluate(case.normalize(_domain_corners(case)))

    for j, output in enumerate(case.meta.output_variables):
        lo, hi = output.expected_range
        assert float(np.nanmin(y[:, j])) >= lo
        assert float(np.nanmax(y[:, j])) <= hi


def test_private_physical_input_validator_rejects_before_formula_evaluation():
    case = CantileverBeam5D()
    assert hasattr(case, "_validate_physical_inputs")
    assert "_validate_physical_inputs" in case.meta.solver_details

    with pytest.raises(ValueError, match="2-D array"):
        case._validate_physical_inputs(np.array([10.0, 1.0, 0.1, 0.2, 200.0]))

    base = np.array([[10.0, 1.0, 0.1, 0.2, 200.0]], dtype=float)
    invalid_values = [
        (0, 0.99),
        (1, 0.49),
        (2, 0.049),
        (3, 0.099),
        (4, 59.9),
        (0, 50.1),
        (1, 3.01),
        (2, 0.301),
        (3, 0.501),
        (4, 210.1),
        (2, np.nan),
    ]

    for col, value in invalid_values:
        x_bad = base.copy()
        x_bad[0, col] = value
        with pytest.raises(ValueError, match="cantilever_beam_5d physical inputs"):
            case._evaluate_core(x_bad)


def test_nonfinite_or_nonphysical_beam_response_is_not_silent_archive_row():
    case = CantileverBeam5D()

    with pytest.raises(ValueError, match="outside the registered benchmark domain"):
        case._evaluate_core(np.array([[10.0, 1.0, 0.1, 0.0, 200.0]], dtype=float))

    with pytest.raises(ValueError, match="outside the registered benchmark domain"):
        case._evaluate_core(np.array([[-10.0, 1.0, 0.1, 0.2, 200.0]], dtype=float))
