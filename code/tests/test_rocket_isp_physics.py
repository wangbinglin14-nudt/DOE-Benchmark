import numpy as np
import pytest
from scipy.optimize import brentq

from doe_testbed.cases.chemical.rocket_isp_4d import RocketIsp4D


def test_rocket_isp_surrogate_failure_is_not_silent_nan(monkeypatch):
    case = RocketIsp4D()

    def fail_isp_calc(*_args, **_kwargs):
        raise ValueError("simulated rocket Isp failure")

    monkeypatch.setattr(case, "_isp_calc", fail_isp_calc)
    x_phys = np.array([[5.5, 70.0, 40.0, 200.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_nonfinite_or_nonphysical_rocket_response_is_not_silent_archive_row():
    case = RocketIsp4D()

    with pytest.raises(ValueError, match="non-finite"):
        case._evaluate_core(np.array([[5.5, 70.0, np.nan, 200.0]], dtype=float))

    with pytest.raises(ValueError, match="epsilon"):
        case._evaluate_core(np.array([[5.5, 70.0, 0.5, 200.0]], dtype=float))


@pytest.mark.parametrize(
    "x_phys, message",
    [
        (np.array([5.5, 70.0, 40.0, 200.0], dtype=float), "shape"),
        (np.array([[5.5, 70.0, 40.0]], dtype=float), "shape"),
        (np.array([[5.5, np.inf, 40.0, 200.0]], dtype=float), "non-finite"),
    ],
)
def test_private_physical_input_contract_rejects_malformed_and_nonfinite(x_phys, message):
    case = RocketIsp4D()

    with pytest.raises(ValueError, match=message):
        case._evaluate_core(x_phys)


@pytest.mark.parametrize(
    "column, bad_value",
    [
        (0, 2.0 - 1e-6),
        (0, 8.0 + 1e-6),
        (1, 5.0 - 1e-6),
        (1, 200.0 + 1e-4),
        (2, 5.0 - 1e-6),
        (2, 100.0 + 1e-4),
        (3, 80.0 - 1e-6),
        (3, 300.0 + 1e-4),
    ],
)
def test_private_physical_input_contract_rejects_declared_domain_violations(column, bad_value):
    case = RocketIsp4D()
    x_phys = np.array([[5.5, 70.0, 40.0, 200.0]], dtype=float)
    x_phys[0, column] = bad_value
    var_name = case.meta.input_variables[column].name

    with pytest.raises(ValueError, match=var_name):
        case._evaluate_core(x_phys)


def test_invalid_private_inputs_do_not_reach_surrogate_solver(monkeypatch):
    case = RocketIsp4D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("surrogate/nozzle path should not be called for invalid private input")

    monkeypatch.setattr(case, "_isp_calc", fail_if_called)
    x_phys = np.array([[5.5, 70.0, 100.0 + 1e-4, 200.0]], dtype=float)

    with pytest.raises(ValueError, match="outside the declared benchmark domain"):
        case._evaluate_core(x_phys)


def test_nozzle_area_mach_solve_failure_is_not_silent_fallback(monkeypatch):
    import scipy.optimize

    def fail_brentq(*_args, **_kwargs):
        raise ValueError("simulated area-Mach failure")

    monkeypatch.setattr(scipy.optimize, "brentq", fail_brentq)

    with pytest.raises(ValueError, match="area-Mach solve failed"):
        RocketIsp4D._isp_calc(5.8, P_c_bar=70.0, epsilon=40.0, T0=200.0)


def _area_ratio_to_mach(epsilon, gamma):
    def residual(Ma):
        return (
            ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * Ma**2))
            ** ((gamma + 1) / (2 * (gamma - 1)))
            / Ma
        ) - epsilon

    return brentq(residual, 1.000001, 50.0, xtol=1e-10)


def _vacuum_thrust_coefficient(epsilon, gamma):
    Ma_e = _area_ratio_to_mach(epsilon, gamma)
    p_e_over_p_c = (1 + (gamma - 1) / 2 * Ma_e**2) ** (-gamma / (gamma - 1))
    inner = (
        2.0
        * gamma**2
        / (gamma - 1.0)
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        * (1.0 - p_e_over_p_c ** ((gamma - 1.0) / gamma))
    )
    return np.sqrt(inner) + epsilon * p_e_over_p_c


def _effective_gamma_from_mr(MR):
    phi = np.clip(8.0 / max(MR, 0.1), 0.2, 3.0)
    excess_h2_frac = max(0, 1.0 - 1.0 / phi)
    return float(np.clip(1.15 + 0.05 * excess_h2_frac, 1.1, 1.35))


def test_isp_peak_is_fuel_rich_not_stoichiometric():
    """LOX/LH2 vacuum Isp should peak fuel-rich, not at MR=8 stoichiometric."""
    mixture_ratios = np.linspace(4.0, 8.0, 17)
    isp_values = np.array(
        [RocketIsp4D._isp_calc(MR, P_c_bar=70.0, epsilon=40.0, T0=200.0)[0] for MR in mixture_ratios]
    )

    peak_mr = mixture_ratios[int(np.argmax(isp_values))]

    assert 5.0 <= peak_mr <= 6.0
    assert RocketIsp4D._isp_calc(5.5, 70.0, 40.0, 200.0)[0] > RocketIsp4D._isp_calc(
        8.0, 70.0, 40.0, 200.0
    )[0]


def test_chamber_temperature_peak_is_fuel_rich():
    t_fuel_rich = RocketIsp4D._isp_calc(5.8, P_c_bar=70.0, epsilon=40.0, T0=200.0)[2]
    t_stoichiometric = RocketIsp4D._isp_calc(8.0, P_c_bar=70.0, epsilon=40.0, T0=200.0)[2]

    assert t_fuel_rich > t_stoichiometric


def test_initial_temperature_adds_chamber_enthalpy():
    cold = RocketIsp4D._isp_calc(6.0, P_c_bar=70.0, epsilon=40.0, T0=90.0)
    warm = RocketIsp4D._isp_calc(6.0, P_c_bar=70.0, epsilon=40.0, T0=300.0)

    assert warm[2] > cold[2]
    assert warm[1] > cold[1]


def test_high_expansion_solution_remains_finite_and_physical():
    isp, c_star, t_chamber = RocketIsp4D._isp_calc(6.0, P_c_bar=20.0, epsilon=100.0, T0=200.0)

    assert np.isfinite([isp, c_star, t_chamber]).all()
    assert 300.0 < isp < 480.0
    assert 1800.0 < c_star < 2500.0
    assert 2000.0 < t_chamber < 3700.0


def test_declared_domain_corners_remain_inside_expected_output_ranges():
    case = RocketIsp4D()
    corners = np.array(
        [
            [mr, pc, epsilon, t0]
            for mr in (2.0, 8.0)
            for pc in (5.0, 200.0)
            for epsilon in (5.0, 100.0)
            for t0 in (80.0, 300.0)
        ],
        dtype=float,
    )

    Y = case._evaluate_core(corners)

    for j, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        assert np.all(Y[:, j] >= lower)
        assert np.all(Y[:, j] <= upper)


def test_out_of_expected_response_envelope_is_not_silent_archive_row(monkeypatch):
    case = RocketIsp4D()

    def out_of_envelope(*_args, **_kwargs):
        return np.array([999.0, 1900.0, 3000.0], dtype=float)

    monkeypatch.setattr(case, "_isp_calc", out_of_envelope)

    with pytest.raises(RuntimeError, match="outside the declared expected response range"):
        case._evaluate_core(np.array([[5.5, 70.0, 40.0, 200.0]], dtype=float))


def test_isp_matches_independent_vacuum_nozzle_equation():
    MR = 5.8
    epsilon = 40.0
    isp, c_star, _t_chamber = RocketIsp4D._isp_calc(
        MR, P_c_bar=70.0, epsilon=epsilon, T0=200.0
    )

    gamma = _effective_gamma_from_mr(MR)
    cf_vac = _vacuum_thrust_coefficient(epsilon, gamma)
    expected_isp = c_star * cf_vac / RocketIsp4D._G0

    np.testing.assert_allclose(isp, expected_isp, rtol=1e-10, atol=1e-10)


def test_metadata_and_validation_material_declare_not_cea_scope():
    from pathlib import Path

    case = RocketIsp4D()
    validation_text = Path("tests/_cross_validation/rocket_isp_4d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path("doe_testbed/cases/chemical/rocket_isp_4d.py").read_text(
        encoding="utf-8"
    )

    assert "low-order LOX/LH2 thermochemical surrogate" in validation_text
    assert "not NASA CEA" in validation_text
    assert "5 % case-level CEA validation" in validation_text
    assert "not NASA CEA Gibbs-minimisation thermochemistry" in case.meta.governing_equation
    assert "finite-rate chemistry" in case.meta.governing_equation
    assert "no NASA CEA" in case.meta.solver_details
    assert "no finite-rate chemistry" in case.meta.solver_details
    assert "no dissociation/equilibrium/frozen expansion chemistry" in case.meta.solver_details
    assert "no nozzle losses" in case.meta.solver_details
    assert "no two-phase flow" in case.meta.solver_details
    assert "no film cooling" in case.meta.solver_details
    assert "no injector/chamber/engine-cycle design" in case.meta.solver_details
    assert "no claimed case-level CEA accuracy" in case.meta.solver_details
    assert "Private physical-input calls are rejected" in case.meta.solver_details
    assert "outputs are rejected" in case.meta.solver_details
    assert "does not call NASA CEA" in source_text
    assert "Target accuracy" not in validation_text


def test_metadata_pins_official_nasa_boundary_sources():
    refs = "\n".join(RocketIsp4D().meta.physics_references)

    assert "nasa.gov/glenn/research/chemical-equilibrium-with-applications/cea-history" in refs
    assert "ntrs.nasa.gov/api/citations/19960044559/downloads/19960044559.pdf" in refs
    assert "grc.nasa.gov/www/k-12/airplane/isentrop.html" in refs
    assert "www1.grc.nasa.gov/beginners-guide-to-aeronautics/nozzle-design" in refs
