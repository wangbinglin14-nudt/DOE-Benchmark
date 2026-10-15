from __future__ import annotations

import numpy as np
import pytest


def _case():
    import doe_testbed as db

    return db.get_case("plate_with_hole_5d")


def _expected(a_b, a_w, sx, sy, theta_deg):
    theta = np.deg2rad(theta_deg)
    sx_eff = sx * np.cos(theta) ** 2 + sy * np.sin(theta) ** 2
    sy_eff = sx * np.sin(theta) ** 2 + sy * np.cos(theta) ** 2
    txy_eff = (sx - sy) * np.sin(theta) * np.cos(theta)

    kt_x = 1.0 + 2.0 * a_b
    kt_y = 1.0 + 2.0 / a_b
    kt_shear = 2.0 * (1.0 + a_b)

    ratio = np.clip(2.0 * a_w, 0.0, 0.75)
    k_hw = 1.0 + 0.5 * ratio**3 + 0.3 * ratio**5

    sig_a = sx_eff * kt_x * k_hw - sy_eff
    sig_b = sy_eff * kt_y * k_hw - sx_eff
    sig_shear = abs(txy_eff * kt_shear * k_hw)
    sigma_max = max(abs(sig_a), abs(sig_b)) + sig_shear
    sigma_nom = max(abs(sx_eff), abs(sy_eff), 1.0)
    return np.array([sigma_max, sigma_max / sigma_nom, sigma_max / 2.0])


def test_oblique_loading_includes_inglis_shear_response():
    case = _case()
    x_phys = np.array([[3.0, 0.2, 200.0, 0.0, 45.0]])
    y = case.evaluate(case.normalize(x_phys))[0]

    no_shear_sigma = 100.0 * (1.0 + 2.0 * 3.0) * (
        1.0 + 0.5 * 0.4**3 + 0.3 * 0.4**5
    ) - 100.0
    expected = _expected(3.0, 0.2, 200.0, 0.0, 45.0)

    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)
    assert y[0] > no_shear_sigma


def test_finite_width_correction_caps_heywood_argument():
    case = _case()
    x_phys = np.array([[1.0, 0.45, 200.0, 0.0, 0.0]])
    y = case.evaluate(case.normalize(x_phys))[0]

    expected = _expected(1.0, 0.45, 200.0, 0.0, 0.0)
    uncapped_khw = 1.0 + 0.5 * 0.9**3 + 0.3 * 0.9**5
    capped_khw = 1.0 + 0.5 * 0.75**3 + 0.3 * 0.75**5

    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)
    assert y[0] < 3.0 * 200.0 * uncapped_khw
    assert np.isclose(y[0], 3.0 * 200.0 * capped_khw)


def test_metadata_declares_nonlinear_response_and_theta_importance():
    case = _case()

    assert case.meta.is_linear is False
    assert "strong_interaction" in case.meta.response_topology
    assert "theta_load" in case.meta.important_variables
    assert ("theta_load", "a_over_b") in case.meta.interaction_pairs


def test_private_physical_input_validator_rejects_before_formula_evaluation():
    case = _case()
    assert hasattr(case, "_validate_physical_inputs")
    assert "_validate_physical_inputs" in case.meta.solver_details

    with pytest.raises(ValueError, match="2-D array"):
        case._validate_physical_inputs(np.array([1.0, 0.2, 200.0, 0.0, 0.0]))

    base = np.array([[1.0, 0.2, 200.0, 0.0, 45.0]], dtype=float)
    invalid_values = [
        (0, 0.499),
        (1, 0.099),
        (2, 49.9),
        (3, -50.1),
        (4, -0.1),
        (0, 5.001),
        (1, 0.451),
        (2, 300.1),
        (3, 100.1),
        (4, 90.1),
        (2, np.nan),
    ]

    for col, value in invalid_values:
        x_bad = base.copy()
        x_bad[0, col] = value
        with pytest.raises(ValueError, match="plate_with_hole_5d physical inputs"):
            case._evaluate_core(x_bad)


def test_nonfinite_or_nonphysical_stress_response_is_not_silent_archive_row():
    case = _case()

    with pytest.raises(ValueError, match="outside the registered benchmark domain"):
        case._evaluate_core(np.array([[0.0, 0.2, 200.0, 0.0, 0.0]], dtype=float))

    with pytest.raises(ValueError, match="physical inputs must be finite"):
        case._evaluate_core(np.array([[1.0, 0.2, np.nan, 0.0, 0.0]], dtype=float))
