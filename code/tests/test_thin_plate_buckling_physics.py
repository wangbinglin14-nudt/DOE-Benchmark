import numpy as np
import pytest

import doe_testbed as db


def _domain_corners(case):
    lb = np.array([v.lower for v in case.meta.input_variables], dtype=float)
    ub = np.array([v.upper for v in case.meta.input_variables], dtype=float)
    return np.array(np.meshgrid(*zip(lb, ub), indexing="ij")).reshape(5, -1).T


def _case():
    return db.get_case("thin_plate_buckling_5d")


def _evaluate_k(case, a_b, bc):
    x_phys = np.array([[a_b, 0.02, 200.0, 0.3, bc]])
    y = case.evaluate(case.normalize(x_phys))
    return float(y[0, 2])


def _ssss_expected(a_b, h_b, e_gpa, nu):
    k_sigma = min((m / a_b + a_b / m) ** 2 for m in range(1, 21))
    e_pa = e_gpa * 1e9
    h = h_b
    D = e_pa * h**3 / (12.0 * (1.0 - nu**2))
    n_cr = k_sigma * np.pi**2 * D
    sigma_cr = n_cr / h / 1e6
    return np.array([n_cr, sigma_cr, k_sigma])


def test_scsc_short_plate_coefficient_matches_validation_note():
    case = _case()

    k_scsc = _evaluate_k(case, a_b=0.5, bc=1.0)

    assert 13.0 < k_scsc < 14.2


def test_scsc_long_plate_uses_classical_asymptote():
    case = _case()

    for a_b in [1.5, 2.0, 4.0]:
        assert abs(_evaluate_k(case, a_b=a_b, bc=1.0) - 6.97) < 1e-9


def test_ssss_branch_keeps_exact_integer_half_wave_solution():
    case = _case()

    assert abs(_evaluate_k(case, a_b=1.0, bc=0.0) - 4.0) < 1e-9
    assert abs(_evaluate_k(case, a_b=2.0, bc=0.0) - 4.0) < 1e-9


def test_ssss_outputs_match_independent_kirchhoff_formula():
    case = _case()
    x_phys = np.array([[1.5, 0.03, 150.0, 0.3, 0.0]])
    y = case.evaluate(case.normalize(x_phys))

    expected = _ssss_expected(a_b=1.5, h_b=0.03, e_gpa=150.0, nu=0.3)

    np.testing.assert_allclose(y[0], expected, rtol=1e-12, atol=1e-9)


def test_continuous_bc_flag_interpolates_elastic_restraint_surrogate():
    case = _case()

    k_ssss = _evaluate_k(case, a_b=0.5, bc=0.0)
    k_scsc = _evaluate_k(case, a_b=0.5, bc=1.0)
    k_mid = _evaluate_k(case, a_b=0.5, bc=0.5)

    assert abs(k_mid - 0.5 * (k_ssss + k_scsc)) < 1e-9


def test_metadata_marks_input_output_response_as_nonlinear():
    case = _case()

    assert case.meta.is_linear is False
    assert "threshold_jump" in case.meta.response_topology


def test_declared_output_ranges_cover_domain_corners():
    case = _case()
    y = case.evaluate(case.normalize(_domain_corners(case)))

    for j, output in enumerate(case.meta.output_variables):
        lo, hi = output.expected_range
        assert float(np.nanmin(y[:, j])) >= lo
        assert float(np.nanmax(y[:, j])) <= hi


def test_private_physical_input_validator_rejects_before_formula_evaluation():
    case = _case()
    assert hasattr(case, "_validate_physical_inputs")
    assert "_validate_physical_inputs" in case.meta.solver_details

    with pytest.raises(ValueError, match="2-D array"):
        case._validate_physical_inputs(np.array([1.0, 0.02, 200.0, 0.3, 0.0]))

    base = np.array([[1.0, 0.02, 200.0, 0.3, 0.0]], dtype=float)
    invalid_values = [
        (0, 0.499),
        (1, 0.0049),
        (2, 49.9),
        (3, 0.249),
        (4, -0.1),
        (0, 4.001),
        (1, 0.0501),
        (2, 300.1),
        (3, 0.451),
        (4, 1.1),
        (2, np.nan),
    ]

    for col, value in invalid_values:
        x_bad = base.copy()
        x_bad[0, col] = value
        with pytest.raises(ValueError, match="thin_plate_buckling_5d physical inputs"):
            case._evaluate_core(x_bad)


def test_nonfinite_or_nonphysical_buckling_response_is_not_silent_archive_row():
    case = _case()

    with pytest.raises(ValueError, match="outside the registered benchmark domain"):
        case._evaluate_core(np.array([[0.0, 0.02, 200.0, 0.3, 0.0]], dtype=float))

    with pytest.raises(ValueError, match="physical inputs must be finite"):
        case._evaluate_core(np.array([[1.0, 0.02, np.nan, 0.3, 0.0]], dtype=float))
