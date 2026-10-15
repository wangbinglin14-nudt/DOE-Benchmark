import numpy as np
import pytest
from pathlib import Path

from doe_testbed.cases.heat_transfer.fin_heat_sink_4d import FinHeatSink4D


def _domain_corners(case):
    lb = np.array([v.lower for v in case.meta.input_variables], dtype=float)
    ub = np.array([v.upper for v in case.meta.input_variables], dtype=float)
    return np.array(np.meshgrid(*zip(lb, ub), indexing="ij")).reshape(4, -1).T


def test_insulated_tip_efficiency_uses_smooth_small_argument_limit():
    eta_zero = FinHeatSink4D._eta_insulated_tip(0.0)
    eta_small = FinHeatSink4D._eta_insulated_tip(1e-6)
    eta_direct = np.tanh(1e-3) / 1e-3

    assert eta_zero == 1.0
    assert np.isclose(eta_small, 1.0 - (1e-6) ** 2 / 3.0)
    assert np.isclose(FinHeatSink4D._eta_insulated_tip(1e-3), eta_direct)


def test_heat_rate_matches_lateral_area_insulated_tip_formula():
    case = FinHeatSink4D()
    H = 0.04
    t_mm = 1.5
    h = 100.0
    k = 200.0
    W = case._W_FIXED
    dT = case._T_BASE - case._T_INF
    t = t_mm * 1e-3

    y = case.evaluate(case.normalize(np.array([[H, t_mm, h, k]])))[0]

    A_cs = W * t
    P_cs = 2.0 * (W + t)
    m = np.sqrt(h * P_cs / (k * A_cs))
    expected_eta = np.tanh(m * H) / (m * H)
    expected_q = np.sqrt(h * P_cs * k * A_cs) * np.tanh(m * H) * dT

    np.testing.assert_allclose(y[0], expected_q, rtol=1e-12)
    np.testing.assert_allclose(y[1], expected_eta, rtol=1e-12)
    np.testing.assert_allclose(y[2], dT / expected_q, rtol=1e-12)


def test_efficiency_stays_in_physical_range_without_clipping():
    y = FinHeatSink4D().evaluate(
        FinHeatSink4D().normalize(np.array([[0.08, 0.5, 500.0, 100.0]]))
    )[0]

    assert np.isfinite(y).all()
    assert 0.0 < y[1] <= 1.0
    assert y[0] > 0.0


def test_failure_boundary_targets_low_fin_efficiency():
    case = FinHeatSink4D()
    eta_output = case.meta.output_variables[1]
    assert eta_output.name == "eta_fin"
    assert eta_output.failure_threshold == 0.7
    assert eta_output.failure_direction == "below"
    assert "eta_fin < 0.70" in case.meta.failure_condition

    X_phys = np.array(
        [
            [0.01, 3.0, 20.0, 400.0],
            [0.08, 0.5, 500.0, 100.0],
        ],
        dtype=float,
    )
    eta = case.evaluate(case.normalize(X_phys))[:, 1]

    assert eta[0] > eta_output.failure_threshold
    assert eta[1] < eta_output.failure_threshold


def test_metadata_declares_nonlinear_monotone_interaction_response_scope():
    case = FinHeatSink4D()

    assert case.meta.is_linear is False
    assert case.meta.response_topology == ["smooth_monotone", "strong_interaction"]
    assert "H and h_coef strongly coupled" in case.meta.scenario_notes["VS-B"]


def test_declared_output_ranges_cover_domain_corners():
    case = FinHeatSink4D()
    y = case.evaluate(case.normalize(_domain_corners(case)))

    for j, output in enumerate(case.meta.output_variables):
        lo, hi = output.expected_range
        assert float(np.nanmin(y[:, j])) >= lo
        assert float(np.nanmax(y[:, j])) <= hi


def test_private_physical_input_validator_rejects_before_formula_evaluation():
    case = FinHeatSink4D()
    assert hasattr(case, "_validate_physical_inputs")
    assert "_validate_physical_inputs" in case.meta.solver_details

    with pytest.raises(ValueError, match="2-D array"):
        case._validate_physical_inputs(np.array([0.04, 1.5, 100.0, 200.0]))

    base = np.array([[0.04, 1.5, 100.0, 200.0]], dtype=float)
    invalid_values = [
        (0, 0.009),
        (1, 0.499),
        (2, 19.9),
        (3, 99.9),
        (0, 0.081),
        (1, 3.001),
        (2, 500.1),
        (3, 400.1),
        (2, np.nan),
    ]

    for col, value in invalid_values:
        x_bad = base.copy()
        x_bad[0, col] = value
        with pytest.raises(ValueError, match="fin_heat_sink_4d physical inputs"):
            case._evaluate_core(x_bad)


def test_nonfinite_or_nonphysical_fin_response_is_not_silent_archive_row():
    case = FinHeatSink4D()

    with pytest.raises(ValueError, match="outside the registered benchmark domain"):
        case._evaluate_core(np.array([[0.04, 0.0, 100.0, 200.0]], dtype=float))

    with pytest.raises(ValueError, match="physical inputs must be finite"):
        case._evaluate_core(np.array([[0.04, 1.5, np.nan, 200.0]], dtype=float))


def test_fin_audit_keeps_20260608_boundary_honest():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `fin_heat_sink_4d` 2026-06-08 official-source boundary tightening:" in audit_text
    assert "COMSOL fin-design article" in normalized
    assert "same dimensionless fin quantities used here" in normalized
    assert "`m = sqrt(hp/kA)`" in normalized
    assert "`mL` as a practical design parameter" in normalized
    assert "does not validate this benchmark case as a full finned heat-sink assembly" in normalized
    assert "base spreading, contact resistance, radiation, conjugate flow, array interaction" in normalized
    assert "source sharpens the local analytical-control boundary" in normalized
    assert "does not change the evidence tier or require a numerical resweep" in normalized
