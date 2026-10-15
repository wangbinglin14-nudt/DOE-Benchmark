import numpy as np
import pytest
from pathlib import Path

from doe_testbed.cases.heat_transfer.casting_solidification_5d import (
    CastingSolidification5D,
)


def test_casting_solver_failure_is_not_silent_nan(monkeypatch):
    case = CastingSolidification5D()

    def fail_solidify(*_args, **_kwargs):
        raise ValueError("simulated Stefan FDM failure")

    monkeypatch.setattr(case, "_solidify", fail_solidify)
    x_phys = np.array([[700.0, 50.0, 1000.0, 0.8, 0.05]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_private_physical_input_validator_rejects_before_fdm(monkeypatch):
    case = CastingSolidification5D()
    assert hasattr(case, "_validate_physical_inputs")
    assert "_validate_physical_inputs" in case.meta.solver_details

    with pytest.raises(ValueError, match="2-D array"):
        case._validate_physical_inputs(np.array([700.0, 50.0, 1000.0, 0.8, 0.05]))

    def should_not_call_solidify(*_args, **_kwargs):
        raise AssertionError("_solidify should not be reached for invalid physical inputs")

    monkeypatch.setattr(case, "_solidify", should_not_call_solidify)
    base = np.array([[700.0, 50.0, 1000.0, 0.8, 0.05]], dtype=float)
    invalid_values = [
        (0, 624.9),
        (1, 19.9),
        (2, 199.9),
        (3, 0.299),
        (4, 0.009),
        (0, 815.1),
        (1, 100.1),
        (2, 2000.1),
        (3, 1.501),
        (4, 0.101),
        (2, np.nan),
    ]

    for col, value in invalid_values:
        x_bad = base.copy()
        x_bad[0, col] = value
        with pytest.raises(ValueError, match="casting_solidification_5d physical inputs"):
            case._evaluate_core(x_bad)


def test_malformed_casting_response_is_not_archived(monkeypatch):
    case = CastingSolidification5D()

    def bad_solidify(*_args, **_kwargs):
        return np.array([np.nan, -1.0, 500.0])

    monkeypatch.setattr(case, "_solidify", bad_solidify)
    x_phys = np.array([[700.0, 50.0, 1000.0, 0.8, 0.05]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_front_position_interpolates_across_mushy_cell():
    x_front = CastingSolidification5D._front_position_from_enthalpy(
        H=np.array([5.0, 5.0, 15.0, 25.0]),
        H_melt=10.0,
        H_liquid=20.0,
        dx=0.1,
    )

    assert np.isclose(x_front, 0.25)


def test_front_position_handles_initial_mushy_boundary_cell():
    x_front = CastingSolidification5D._front_position_from_enthalpy(
        H=np.array([15.0, 25.0]),
        H_melt=10.0,
        H_liquid=20.0,
        dx=0.1,
    )

    assert np.isclose(x_front, 0.05)


def test_front_position_reaches_full_length_when_all_cells_are_solid():
    x_front = CastingSolidification5D._front_position_from_enthalpy(
        H=np.array([5.0, 5.0, 5.0]),
        H_melt=10.0,
        H_liquid=20.0,
        dx=0.1,
    )

    assert np.isclose(x_front, 0.3)


def test_mean_front_speed_matches_interpolated_solidification_depth():
    case = CastingSolidification5D()
    L = 0.05

    y = case._solidify(
        T_pour=700.0,
        T_mold=50.0,
        h_contact=1500.0,
        k_mold=0.8,
        L=L,
        n_x=20,
    )
    t_solid, v_front, _ = y

    assert np.isfinite(y).all()
    assert t_solid > 0.0
    assert 0.0 < v_front <= L * 1000.0 / t_solid


def test_higher_mold_conductivity_shortens_solidification_time():
    case = CastingSolidification5D()
    common = dict(
        T_pour=700.0,
        T_mold=50.0,
        h_contact=1200.0,
        L=0.04,
        n_x=20,
    )

    low_k = case._solidify(k_mold=0.3, **common)
    high_k = case._solidify(k_mold=1.5, **common)

    assert np.isfinite(low_k).all()
    assert np.isfinite(high_k).all()
    assert high_k[0] < low_k[0]


def test_front_speed_falls_back_to_nonzero_average_when_front_is_unresolved():
    case = CastingSolidification5D()
    L = 0.1

    y = case._solidify(
        T_pour=815.0,
        T_mold=100.0,
        h_contact=200.0,
        k_mold=0.3,
        L=L,
        n_x=20,
    )
    t_solid, v_front, _ = y

    assert np.isfinite(y).all()
    assert t_solid > 0.0
    assert v_front > 0.001
    assert v_front <= L * 1000.0 / t_solid


def test_casting_audit_keeps_20260608_boundary_honest():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `casting_solidification_5d` 2026-06-08 official-source boundary tightening:" in audit_text
    assert "COMSOL phase-change theory and continuous-casting model pages" in normalized
    assert "apparent-heat-capacity methods handle latent heat through an effective heat-capacity formulation" in normalized
    assert "mushy zones and temperature-dependent properties in broader casting simulations" in normalized
    assert "do not validate this compact one-dimensional slab surrogate as a continuous-casting model" in normalized
    assert "alloy-property calibration" in normalized
    assert "fluid-flow/mould-coupled simulation" in normalized
    assert "shrinkage-porosity/hot-cracking predictor" in normalized
    assert "No additional numerical resweep is needed solely for this source-boundary tightening" in normalized
