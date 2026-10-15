from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _domain_corners(case):
    lb = np.array([v.lower for v in case.meta.input_variables], dtype=float)
    ub = np.array([v.upper for v in case.meta.input_variables], dtype=float)
    return np.array(np.meshgrid(*zip(lb, ub), indexing="ij")).reshape(7, -1).T


def _case():
    import doe_testbed as db

    return db.get_case("battery_thermal_7d")


def test_battery_thermal_solver_failure_is_not_silent_nan(monkeypatch):
    case = _case()

    def fail_simulate(*_args, **_kwargs):
        raise ValueError("simulated battery thermal failure")

    monkeypatch.setattr(case, "_simulate", fail_simulate)
    x_phys = np.array([[1.0, 10.0, 120.0, 25.0, 3000.0, 1000.0, 0.06]])

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_malformed_battery_response_is_not_archived(monkeypatch):
    case = _case()

    def bad_simulate(*_args, **_kwargs):
        return np.array([np.nan, 20.0, -1.0])

    monkeypatch.setattr(case, "_simulate", bad_simulate)
    x_phys = np.array([[1.0, 10.0, 120.0, 25.0, 3000.0, 1000.0, 0.06]])

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_delta_t_must_match_peak_rise(monkeypatch):
    case = _case()

    def inconsistent_simulate(*_args, **_kwargs):
        return np.array([30.0, 29.0, 99.0])

    monkeypatch.setattr(case, "_simulate", inconsistent_simulate)
    x_phys = np.array([[1.0, 10.0, 120.0, 25.0, 3000.0, 1000.0, 0.06]])

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_private_physical_input_validator_rejects_before_ode(monkeypatch):
    case = _case()
    assert hasattr(case, "_validate_physical_inputs")
    assert "_validate_physical_inputs" in case.meta.solver_details

    with pytest.raises(ValueError, match="2-D array"):
        case._validate_physical_inputs(np.array([1.0, 10.0, 120.0, 25.0, 3000.0, 1000.0, 0.06]))

    def should_not_call_simulate(*_args, **_kwargs):
        raise AssertionError("_simulate should not be reached for invalid physical inputs")

    monkeypatch.setattr(case, "_simulate", should_not_call_simulate)
    base = np.array([[1.0, 10.0, 120.0, 25.0, 3000.0, 1000.0, 0.06]])
    invalid_values = [
        (0, 0.49),
        (1, 0.99),
        (2, 9.99),
        (3, 9.99),
        (4, 1999.0),
        (5, 799.0),
        (6, 0.009),
        (0, 5.01),
        (1, 50.01),
        (2, 200.01),
        (3, 35.01),
        (4, 8001.0),
        (5, 1201.0),
        (6, 0.101),
        (4, np.nan),
    ]

    for col, value in invalid_values:
        x_bad = base.copy()
        x_bad[0, col] = value
        with pytest.raises(ValueError, match="battery_thermal_7d physical inputs"):
            case._evaluate_core(x_bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"C_rate": 0.0},
        {"R_ref": -1e-3},
        {"h_cool": 0.0},
        {"T_cool": np.nan},
        {"cp": -1000.0},
        {"A_cool": 0.0},
    ],
)
def test_simulate_rejects_inputs_outside_lumped_ode_contract(kwargs):
    case = _case()
    params = {
        "C_rate": 1.0,
        "R_ref": 10e-3,
        "h_cool": 120.0,
        "T_cool": 298.15,
        "Ea_R": 3000.0,
        "cp": 1000.0,
        "A_cool": 0.06,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match="outside the declared"):
        case._simulate(**params)


def test_reversible_entropic_heat_increases_low_crate_temperature():
    case = _case()
    x_phys = np.array([[0.5, 5.0, 120.0, 25.0, 3000.0, 1000.0, 0.06]])
    y_with = case.evaluate(case.normalize(x_phys))[0]

    old_dudt = case._DUDT
    try:
        case._DUDT = 0.0
        y_without = case.evaluate(case.normalize(x_phys))[0]
    finally:
        case._DUDT = old_dudt

    assert y_with[0] > y_without[0]
    assert y_with[2] > y_without[2]


def test_arrhenius_temperature_guard_keeps_extreme_case_finite():
    case = _case()
    x_phys = np.array([
        [5.0, 50.0, 10.0, 10.0, 8000.0, 800.0, 0.01],
        [5.0, 50.0, 200.0, 35.0, 8000.0, 1200.0, 0.10],
    ])
    y = case.evaluate(case.normalize(x_phys))

    assert np.all(np.isfinite(y))
    assert np.all(y[:, 0] >= x_phys[:, 3])
    assert np.all(y[:, 2] >= 0.0)


def test_declared_output_ranges_cover_domain_corners():
    case = _case()
    y = case.evaluate(case.normalize(_domain_corners(case)))

    for j, output in enumerate(case.meta.output_variables):
        lo, hi = output.expected_range
        assert float(np.nanmin(y[:, j])) >= lo
        assert float(np.nanmax(y[:, j])) <= hi


def test_official_source_boundary_keeps_battery_case_lumped_ode_scope():
    case = _case()
    refs = "\n".join(case.meta.physics_references)
    solver_details = case.meta.solver_details
    scenario = case.meta.scenario_notes["VS-G"]
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit_text.split())

    assert "https://www.osti.gov/biblio/5913742" in refs
    assert "NREL Energy Storage Thermal Management" in refs
    assert "NREL Multi-Scale Battery Physics Modeling" in refs
    assert "NASA NESC Calorimetry of Lithium-ion Cells During Thermal Runaway" in refs
    assert "single-node normal-operation thermal surrogate" in solver_details
    assert "not calibrated cell calorimetry" in solver_details
    assert "electrochemical-thermal P2D" in solver_details
    assert "3-D CFD" in solver_details
    assert "thermal-runaway validation" in solver_details
    assert "not thermal-runaway onset or propagation" in scenario
    assert "Additional `battery_thermal_7d` 2026-06-08 official-source re-check" in audit_text
    assert "Additional `battery_thermal_7d` 2026-06-08 official-source boundary tightening" in audit_text
    assert "cell, module, pack, and system levels" in audit_text
    assert "thermal management and characterization work across cell, module, pack, and system levels" in normalized_audit
    assert "energy materials through calorimetry and thermal conductivity" in normalized_audit
    assert "cells and modules through calorimetry and infrared imaging" in normalized_audit
    assert "packs through temperature-variation analysis" in normalized_audit
    assert "full energy-storage systems through interaction with other components" in normalized_audit
    assert "performance, safety, and life analysis" in normalized_audit
    assert "high-fidelity 3-D thermal" in audit_text
    assert "thermal-runaway propagation and calorimetry workflows outside the benchmark" in audit_text
    assert "thermal-runaway calorimetry workflows rather than normal-operation cooling design validation" in normalized_audit
    assert "not calibrated electrochemical-thermal validation" in audit_text
    assert "must not be described as P2D, 3-D CFD, cell-calorimetry" in normalized_audit
    assert "cell/module/pack/system thermal-characterization" in solver_details
    assert "not externally validated cell/module/pack/system thermal-characterization" in normalized_audit
    assert "No numerical response resweep is needed solely for this official-source re-check" in audit_text
