from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _case():
    import doe_testbed as db

    return db.get_case("fixed_bed_reactor_6d")


def test_fixed_bed_reactor_solver_failure_is_not_silent_nan(monkeypatch):
    case = _case()

    def fail_solve_reactor(*_args, **_kwargs):
        raise ValueError("simulated fixed-bed reactor ODE failure")

    monkeypatch.setattr(case, "_solve_reactor", fail_solve_reactor)
    x_phys = np.array([[620.0, 3.0, 90.0, 12.0, -150.0, 9.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_source_input_documentation_matches_k0_log_variable():
    case = _case()
    source = Path("doe_testbed/cases/chemical/fixed_bed_reactor_6d.py").read_text(
        encoding="utf-8"
    )
    references = "\n".join(case.meta.physics_references)

    assert "k0_log" in source
    assert "GHSV_inv" not in source
    assert "k₀ = 1×10⁹" not in source
    assert "Adiabatic energy-balance learning module" in references
    assert "not calibrated fixed-bed-reactor validation data" in references
    assert "COMSOL Packed Bed Reactor documentation" in references
    assert "pellet diffusion, Ergun/pressure-drop" in references
    assert "outside the declared benchmark domain before ODE assembly" in case.meta.solver_details
    assert "intentionally omits packed-bed multigeometry species transport" in case.meta.solver_details
    assert "pellet diffusion/effectiveness factors" in case.meta.solver_details
    assert "Ergun or other pressure-drop equations" in case.meta.solver_details
    assert "radial heat-transfer" in case.meta.solver_details
    assert "catalyst deactivation" in case.meta.solver_details


def test_private_physical_input_contract_rejects_malformed_arrays(monkeypatch):
    case = _case()
    monkeypatch.setattr(
        case,
        "_solve_reactor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ODE solver should not be called for invalid physical input")
        ),
    )

    with pytest.raises(ValueError, match=r"shape \(n, 6\)"):
        case._evaluate_core(np.array([620.0, 3.0, 90.0, 12.0, -150.0, 9.0]))

    with pytest.raises(ValueError, match=r"shape \(n, 6\)"):
        case._evaluate_core(np.array([[620.0, 3.0, 90.0]], dtype=float))


@pytest.mark.parametrize(
    ("column", "bad_value", "name"),
    [
        (0, 550.0 - 1e-5, "T_in"),
        (0, 700.0 + 1e-5, "T_in"),
        (1, 0.5 - 1e-6, "C_A0"),
        (1, 5.0 + 1e-6, "C_A0"),
        (2, 50.0 - 1e-5, "Ea"),
        (2, 150.0 + 1e-5, "Ea"),
        (3, 1.0 - 1e-6, "tau"),
        (3, 20.0 + 1e-6, "tau"),
        (4, -200.0 - 1e-5, "dHr"),
        (4, -50.0 + 1e-5, "dHr"),
        (5, 6.0 - 1e-6, "k0_log"),
        (5, 12.0 + 1e-6, "k0_log"),
    ],
)
def test_private_physical_input_contract_rejects_declared_domain_violations(
    monkeypatch, column: int, bad_value: float, name: str
):
    case = _case()
    monkeypatch.setattr(
        case,
        "_solve_reactor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ODE solver should not be called for invalid physical input")
        ),
    )
    x_phys = np.array([[620.0, 3.0, 90.0, 12.0, -150.0, 9.0]], dtype=float)
    x_phys[0, column] = bad_value

    with pytest.raises(ValueError, match=f"{name!r} is outside"):
        case._evaluate_core(x_phys)


def test_private_physical_input_contract_rejects_nonfinite_values(monkeypatch):
    case = _case()
    monkeypatch.setattr(
        case,
        "_solve_reactor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ODE solver should not be called for invalid physical input")
        ),
    )
    x_phys = np.array([[620.0, 3.0, np.nan, 12.0, -150.0, 9.0]], dtype=float)

    with pytest.raises(ValueError, match="non-finite physical inputs"):
        case._evaluate_core(x_phys)


def test_nonfinite_or_nonphysical_fixed_bed_response_is_not_silent_archive_row():
    case = _case()

    with pytest.raises(ValueError, match="non-finite physical inputs"):
        case._evaluate_core(
            np.array([[620.0, 3.0, np.nan, 12.0, -150.0, 9.0]], dtype=float)
        )

    with pytest.raises(ValueError, match="'dHr' is outside"):
        case._evaluate_core(
            np.array([[620.0, 3.0, 90.0, 12.0, 150.0, 9.0]], dtype=float)
        )


def test_ode_nonfinite_or_nonphysical_trajectory_is_not_silent(monkeypatch):
    import doe_testbed.cases.chemical.fixed_bed_reactor_6d as mod

    case = _case()

    class Solution:
        def __init__(self, y):
            self.success = True
            self.message = ""
            self.y = y

    monkeypatch.setattr(
        mod,
        "solve_ivp",
        lambda *_args, **_kwargs: Solution(
            np.array([[1.0, 0.8, np.nan], [1.0, 1.05, 1.1]], dtype=float)
        ),
    )
    with pytest.raises(RuntimeError, match="non-finite fixed-bed reactor ODE trajectory"):
        case._solve_reactor(620.0, 3.0, 90.0, 12.0, -150.0, 1e9)

    monkeypatch.setattr(
        mod,
        "solve_ivp",
        lambda *_args, **_kwargs: Solution(
            np.array([[1.0, 0.8, 0.9], [1.0, 1.05, 1.1]], dtype=float)
        ),
    )
    with pytest.raises(RuntimeError, match="non-physical fixed-bed reactor ODE trajectory"):
        case._solve_reactor(620.0, 3.0, 90.0, 12.0, -150.0, 1e9)


def test_adiabatic_irreversible_peak_is_outlet_temperature():
    case = _case()
    x_phys = np.array([
        [600.0, 2.0, 80.0, 10.0, -120.0, 8.0],
        [680.0, 5.0, 60.0, 18.0, -200.0, 10.0],
        [560.0, 1.0, 140.0, 4.0, -50.0, 7.0],
    ])
    y = case.evaluate(case.normalize(x_phys))

    assert np.all(np.isfinite(y))
    assert np.allclose(y[:, 1], y[:, 2], rtol=1e-10, atol=1e-10)
    assert np.all(y[:, 1] >= x_phys[:, 0])


def test_temperature_rise_matches_adiabatic_energy_balance():
    case = _case()
    x_phys = np.array([[620.0, 3.0, 90.0, 12.0, -150.0, 9.0]])
    y = case.evaluate(case.normalize(x_phys))[0]
    x_a, t_max, t_out = y

    rho = case._rho
    cp = case._Cp
    expected_rise = (-x_phys[0, 4]) * x_phys[0, 1] * x_a / (rho * cp)

    assert np.isclose(t_max, t_out, rtol=1e-10, atol=1e-10)
    assert np.isclose(t_out - x_phys[0, 0], expected_rise, rtol=0.02, atol=1e-3)


def test_arrhenius_guard_keeps_extreme_trials_finite():
    case = _case()
    x_phys = np.array([
        [550.0, 5.0, 150.0, 20.0, -200.0, 12.0],
        [700.0, 5.0, 50.0, 20.0, -200.0, 12.0],
    ])
    y = case.evaluate(case.normalize(x_phys))

    assert np.all(np.isfinite(y))
    assert np.all((0.0 <= y[:, 0]) & (y[:, 0] <= 1.0))


def test_fixed_bed_audit_keeps_20260608_boundary_honest():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `fixed_bed_reactor_6d` 2026-06-08 official-source boundary tightening:" in audit_text
    assert "University of Michigan/Fogler adiabatic-reactor lecture notes" in normalized
    assert "adiabatic `T = f(X)` energy-balance closure" in normalized
    assert "coupled mole and energy balances rather than a single isothermal rate law" in normalized
    assert "do not validate this compact dimensionless surrogate as calibrated packed-bed catalyst design" in normalized
    assert "pellet diffusion/effectiveness-factor modelling" in normalized
    assert "Ergun pressure drop" in normalized
    assert "radial heat-transfer" in normalized
    assert "deactivation, multiphase hydrotreating, or plant reactor operation" in normalized
    assert "No numerical response resweep is needed solely for this source-boundary tightening" in normalized


def test_fixed_bed_audit_records_packed_bed_multiphysics_boundary():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `fixed_bed_reactor_6d` 2026-06-08 packed-bed workflow boundary note:" in audit_text
    assert "COMSOL Packed Bed Reactor documentation" in normalized
    assert "multigeometry species-transport" in normalized
    assert "pellet-scale diffusion" in normalized
    assert "pressure-drop modelling" in normalized
    assert "not calibration or validation data" in normalized
    assert "remains a 1-D adiabatic PFR-style ODE" in normalized
    assert "No numerical response resweep is needed solely for this boundary-reference addition" in normalized
