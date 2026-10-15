import numpy as np
import pytest
from scipy.linalg import expm
from pathlib import Path

from doe_testbed.cases.chemical.membrane_reactor_5d import MembraneReactor5D


def test_membrane_reactor_solver_failure_is_not_silent_nan(monkeypatch):
    case = MembraneReactor5D()

    def fail_reactor(*_args, **_kwargs):
        raise ValueError("simulated membrane reactor failure")

    monkeypatch.setattr(case, "_reactor", fail_reactor)
    x_phys = np.array([[2.0, 1.0, 2.0, 20.0, 0.03]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_nonfinite_or_nonphysical_membrane_response_is_not_silent_archive_row():
    case = MembraneReactor5D()

    with pytest.raises(ValueError, match="non-finite"):
        case._evaluate_core(np.array([[2.0, 1.0, np.nan, 20.0, 0.03]], dtype=float))

    with pytest.raises(ValueError, match="C_B0_frac"):
        case._evaluate_core(np.array([[2.0, 1.0, 2.0, 20.0, -0.1]], dtype=float))


@pytest.mark.parametrize(
    "x_phys, message",
    [
        (np.array([2.0, 1.0, 2.0, 20.0, 0.03], dtype=float), "shape"),
        (np.array([[2.0, 1.0, 2.0, 20.0]], dtype=float), "shape"),
        (np.array([[2.0, np.inf, 2.0, 20.0, 0.03]], dtype=float), "non-finite"),
    ],
)
def test_private_physical_input_contract_rejects_malformed_and_nonfinite(x_phys, message):
    case = MembraneReactor5D()

    with pytest.raises(ValueError, match=message):
        case._evaluate_core(x_phys)


@pytest.mark.parametrize(
    "column, bad_value",
    [
        (0, 0.1 - 1e-6),
        (0, 10.0 + 1e-5),
        (1, 0.01 - 1e-7),
        (1, 5.0 + 1e-6),
        (2, 0.5 - 1e-6),
        (2, 20.0 + 1e-5),
        (3, 1.0 - 1e-6),
        (3, 100.0 + 1e-4),
        (4, -1e-6),
        (4, 0.1 + 1e-6),
    ],
)
def test_private_physical_input_contract_rejects_declared_domain_violations(column, bad_value):
    case = MembraneReactor5D()
    x_phys = np.array([[2.0, 1.0, 2.0, 20.0, 0.03]], dtype=float)
    x_phys[0, column] = bad_value
    var_name = case.meta.input_variables[column].name

    with pytest.raises(ValueError, match=var_name):
        case._evaluate_core(x_phys)


def test_invalid_private_inputs_do_not_reach_ode_solver(monkeypatch):
    case = MembraneReactor5D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("ODE path should not be called for invalid private input")

    monkeypatch.setattr(case, "_reactor", fail_if_called)
    x_phys = np.array([[2.0, 1.0, 2.0, 20.0, 0.1 + 1e-6]], dtype=float)

    with pytest.raises(ValueError, match="outside the declared benchmark domain"):
        case._evaluate_core(x_phys)


def test_ode_solve_failures_are_not_silent_pseudo_results(monkeypatch):
    import doe_testbed.cases.chemical.membrane_reactor_5d as mod

    class FailedSolution:
        success = False
        message = "simulated solve failure"
        y = np.full((3, 1), np.nan)

    monkeypatch.setattr(mod, "solve_ivp", lambda *_args, **_kwargs: FailedSolution())

    with pytest.raises(RuntimeError, match="membrane reactor ODE solve failed"):
        MembraneReactor5D._reactor(Da=2.0, Da_m=1.0, K_eq=2.0, Pe=20.0, C_B0=0.03)


def test_no_membrane_baseline_solve_failure_is_not_silent_pseudo_eta(monkeypatch):
    import doe_testbed.cases.chemical.membrane_reactor_5d as mod

    class Solution:
        def __init__(self, success, y, message=""):
            self.success = success
            self.y = y
            self.message = message

    calls = {"n": 0}

    def fake_solve_ivp(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return Solution(True, np.array([[0.5], [0.4], [0.2]], dtype=float))
        return Solution(False, np.full((3, 1), np.nan), "simulated baseline failure")

    monkeypatch.setattr(mod, "solve_ivp", fake_solve_ivp)

    with pytest.raises(RuntimeError, match="no-membrane baseline ODE solve failed"):
        MembraneReactor5D._reactor(Da=2.0, Da_m=1.0, K_eq=2.0, Pe=20.0, C_B0=0.03)


def _legacy_static_eta(Da, Da_m, K_eq, Pe, C_B0):
    y = MembraneReactor5D._reactor(Da, Da_m, K_eq, Pe, C_B0)
    k_r_ratio = 1.0 / max(K_eq, 1e-6)
    c_b_theory = 1.0 / (1.0 + k_r_ratio)
    return float(np.clip(1.0 - y[1] / max(c_b_theory, 0.01), 0, 1))


def test_membrane_permeation_increases_conversion_and_lowers_product_exit():
    weak_membrane = MembraneReactor5D._reactor(Da=2.0, Da_m=0.01, K_eq=2.0, Pe=20.0, C_B0=0.0)
    strong_membrane = MembraneReactor5D._reactor(Da=2.0, Da_m=5.0, K_eq=2.0, Pe=20.0, C_B0=0.0)

    assert strong_membrane[0] > weak_membrane[0]
    assert strong_membrane[1] < weak_membrane[1]


def test_eta_mem_is_integrated_flux_ratio_not_static_equilibrium_shortfall():
    params = {"Da": 2.0, "Da_m": 1.0, "K_eq": 2.0, "Pe": 20.0, "C_B0": 0.0}
    eta = MembraneReactor5D._reactor(**params)[2]
    legacy_eta = _legacy_static_eta(**params)

    assert 0.0 <= eta <= 1.0
    assert abs(eta - legacy_eta) > 0.05


def test_reactor_matches_constant_coefficient_matrix_exponential_control():
    Da = 2.0
    Da_m = 1.0
    K_eq = 2.0
    C_B0 = 0.03
    k_r_ratio = 1.0 / K_eq

    with_membrane = np.array(
        [
            [-Da, Da * k_r_ratio, 0.0],
            [Da, -(Da * k_r_ratio + Da_m), 0.0],
            [0.0, Da_m, 0.0],
        ],
        dtype=float,
    )
    no_membrane = np.array(
        [
            [-Da, Da * k_r_ratio, 0.0],
            [Da, -Da * k_r_ratio, 0.0],
            [0.0, Da_m, 0.0],
        ],
        dtype=float,
    )
    y0 = np.array([1.0, C_B0, 0.0], dtype=float)
    y_with = expm(with_membrane) @ y0
    y_no = expm(no_membrane) @ y0
    expected = np.array(
        [
            1.0 - y_with[0],
            y_with[1],
            np.clip(y_with[2] / max(y_no[2], 1e-12), 0.0, 1.0),
        ],
        dtype=float,
    )

    observed = MembraneReactor5D._reactor(
        Da=Da,
        Da_m=Da_m,
        K_eq=K_eq,
        Pe=20.0,
        C_B0=C_B0,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-10)


def test_membrane_utilisation_drops_when_permeation_depletes_product_along_reactor():
    weak = MembraneReactor5D._reactor(Da=2.0, Da_m=0.01, K_eq=2.0, Pe=20.0, C_B0=0.0)[2]
    strong = MembraneReactor5D._reactor(Da=2.0, Da_m=5.0, K_eq=2.0, Pe=20.0, C_B0=0.0)[2]

    assert weak > 0.95
    assert strong < weak


def test_pe_remains_declared_inert_variable_in_plug_flow_limit():
    low_pe = MembraneReactor5D._reactor(Da=2.0, Da_m=1.0, K_eq=2.0, Pe=1.0, C_B0=0.03)
    high_pe = MembraneReactor5D._reactor(Da=2.0, Da_m=1.0, K_eq=2.0, Pe=100.0, C_B0=0.03)

    np.testing.assert_allclose(low_pe, high_pe, rtol=0.0, atol=0.0)


def test_metadata_declares_pe_inert_and_response_nonlinear_scope():
    case = MembraneReactor5D()
    text = "\n".join(
        [
            case.meta.governing_equation,
            case.meta.solver_details,
            " ".join(case.meta.scenario_notes.values()),
        ]
    )

    assert case.meta.is_linear is False
    assert "strong_interaction" in case.meta.response_topology
    assert "Pe → ∞" in text
    assert "zero index for Pe" in text
    assert "Private physical-input calls are rejected" in case.meta.solver_details
    assert case.meta.effective_dim == 3
    assert "Pe" not in case.meta.important_variables
    references = "\n".join(case.meta.physics_references)
    details = case.meta.solver_details
    assert "University of Michigan Elements of Chemical Reaction Engineering" in references
    assert "COMSOL Hydrodealkylation in a Membrane Reactor documentation" in references
    assert "Darcy-type pressure-driven membrane flux" in references
    assert "species-selective membrane balances" in references
    assert "not calibration data" in references
    assert "finite-Pe axial dispersion" in details
    assert "mass-transfer-film/concentration-polarisation correlations" in details
    assert "membrane selectivity or material calibration" in details
    assert "pressure-driven Darcy flux" in details
    assert "shell-side pressure/composition coupling" in details
    assert "thermodynamic/property packages" in details
    assert "fouling/aging" in details


def test_membrane_reactor_audit_keeps_20260608_boundary_honest():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `membrane_reactor_5d` 2026-06-08 official-source boundary tightening:" in audit_text
    assert "University of Michigan/Fogler membrane-reactor module" in normalized
    assert "combine reaction with separation" in normalized
    assert "remove a product through the membrane" in normalized
    assert "push equilibrium conversion higher in line with Le Chatelier's principle" in normalized
    assert "does not validate this v1 `Pe -> infinity` plug-flow surrogate" in normalized
    assert "membrane selectivity data" in normalized
    assert "axial dispersion, mass-transfer polarisation" in normalized
    assert "catalyst deactivation, membrane aging/fouling" in normalized
    assert "source sharpens the boundary; it does not upgrade the evidence tier" in normalized
    assert "No numerical response resweep is needed solely for this source-boundary tightening" in normalized


def test_membrane_reactor_audit_records_comsol_workflow_boundary():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `membrane_reactor_5d` 2026-06-08 membrane-workflow boundary note:" in audit_text
    assert "COMSOL Hydrodealkylation in a Membrane Reactor documentation" in normalized
    assert "Thermodynamics and Chemistry interfaces" in normalized
    assert "plug-flow reactor equations" in normalized
    assert "Darcy-law hydrogen flux through the membrane" in normalized
    assert "reactor/shell pressure coupling" in normalized
    assert "not validation data" in normalized
    assert "remains a dimensionless Pe-infinity plug-flow ODE" in normalized
    assert "No numerical response resweep is needed solely for this membrane-workflow boundary addition" in normalized
