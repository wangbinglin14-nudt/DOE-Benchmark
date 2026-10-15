"""Physics regression tests for ``supersonic_nozzle_5d``.

These tests pin the normal-shock branch that is not covered by generic
range checks.  In particular, the exit Mach number after an in-nozzle
shock must be obtained from the downstream subsonic area-Mach relation,
not by empirical blending.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from doe_testbed.cases.chemical.supersonic_nozzle_5d import SupersonicNozzle5D


def test_nozzle_surrogate_failure_is_not_silent_nan(monkeypatch):
    case = SupersonicNozzle5D()

    def fail_nozzle(*_args, **_kwargs):
        raise ValueError("simulated nozzle failure")

    monkeypatch.setattr(case, "_nozzle", fail_nozzle)
    x_phys = np.array([[10.0, 0.1, 1.4, 1200.0, 0.5]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)


def test_area_mach_solver_failure_is_not_silent_fallback():
    case = SupersonicNozzle5D()

    with pytest.raises(ValueError, match="invalid area-Mach inputs"):
        case._area_ratio_to_Ma(0.5, 1.4, supersonic=True)

    with pytest.raises(ValueError, match="invalid area-Mach inputs"):
        case._area_ratio_to_Ma(2.0, 1.0, supersonic=False)


def test_nonfinite_or_nonphysical_nozzle_response_is_not_silent_archive_row():
    case = SupersonicNozzle5D()

    with pytest.raises(ValueError, match="non-finite"):
        case._evaluate_core(np.array([[10.0, 0.1, np.nan, 1200.0, 0.5]], dtype=float))

    with pytest.raises(ValueError, match="epsilon"):
        case._evaluate_core(np.array([[0.5, 0.1, 1.4, 1200.0, 0.5]], dtype=float))


@pytest.mark.parametrize(
    "x_phys, message",
    [
        (np.array([10.0, 0.1, 1.4, 1200.0, 0.5], dtype=float), "shape"),
        (np.array([[10.0, 0.1, 1.4, 1200.0]], dtype=float), "shape"),
        (np.array([[10.0, np.inf, 1.4, 1200.0, 0.5]], dtype=float), "non-finite"),
    ],
)
def test_private_physical_input_contract_rejects_malformed_and_nonfinite(x_phys, message):
    case = SupersonicNozzle5D()

    with pytest.raises(ValueError, match=message):
        case._evaluate_core(x_phys)


@pytest.mark.parametrize(
    "column, bad_value",
    [
        (0, 2.0 - 1e-6),
        (0, 20.0 + 1e-5),
        (1, 0.01 - 1e-7),
        (1, 1.0 + 1e-6),
        (2, 1.2 - 1e-6),
        (2, 1.67 + 1e-6),
        (3, 300.0 - 1e-4),
        (3, 2000.0 + 1e-3),
        (4, -1e-6),
        (4, 1.0 + 1e-6),
    ],
)
def test_private_physical_input_contract_rejects_declared_domain_violations(column, bad_value):
    case = SupersonicNozzle5D()
    x_phys = np.array([[10.0, 0.1, 1.4, 1200.0, 0.5]], dtype=float)
    x_phys[0, column] = bad_value
    var_name = case.meta.input_variables[column].name

    with pytest.raises(ValueError, match=var_name):
        case._evaluate_core(x_phys)


def test_invalid_private_inputs_do_not_reach_nozzle_solver(monkeypatch):
    case = SupersonicNozzle5D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("nozzle path should not be called for invalid private input")

    monkeypatch.setattr(case, "_nozzle", fail_if_called)
    x_phys = np.array([[10.0, 0.1, 1.4, 1200.0, 1.0 + 1e-6]], dtype=float)

    with pytest.raises(ValueError, match="outside the declared benchmark domain"):
        case._evaluate_core(x_phys)


def _area_mach(Ma: float, gamma: float) -> float:
    return (
        ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * Ma**2))
        ** ((gamma + 1) / (2 * (gamma - 1)))
        / Ma
    )


def _domain_corners(case: SupersonicNozzle5D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_in_nozzle_shock_exit_mach_uses_downstream_subsonic_branch():
    case = SupersonicNozzle5D()
    epsilon = 10.0
    gamma = 1.4
    T0 = 1200.0
    shock_pos = 0.5

    Ma_design = case._area_ratio_to_Ma(epsilon, gamma, supersonic=True)
    Ma2_design_sq = (
        ((gamma - 1) * Ma_design**2 + 2)
        / (2 * gamma * Ma_design**2 - (gamma - 1))
    )
    Ma2_design = float(np.sqrt(Ma2_design_sq))
    p02_ratio = case._normal_shock_p0_ratio(Ma_design, gamma)
    p_after_shock = (
        (1 + (gamma - 1) / 2 * Ma2_design**2) ** (-gamma / (gamma - 1))
        * p02_ratio
    )

    # Choose a back pressure above the shock-at-exit pressure so the
    # implementation enters the in-nozzle shock branch.
    p_back_ratio = min(0.95, p_after_shock * 1.8)
    Y = case._nozzle(epsilon, p_back_ratio, gamma, T0, shock_pos)
    Ma_exit, T_exit, eta = Y

    A_shock = 1.0 + (epsilon - 1.0) * shock_pos
    Ma_shock = case._area_ratio_to_Ma(A_shock, gamma, supersonic=True)
    Ma2s_sq = (
        ((gamma - 1) * Ma_shock**2 + 2)
        / (2 * gamma * Ma_shock**2 - (gamma - 1))
    )
    Ma2s = float(np.sqrt(Ma2s_sq))
    expected_area_ratio = (epsilon / A_shock) * _area_mach(Ma2s, gamma)
    expected_Ma_exit = case._area_ratio_to_Ma(
        expected_area_ratio, gamma, supersonic=False
    )

    assert Ma_exit < 1.0
    assert np.isclose(Ma_exit, expected_Ma_exit, rtol=1e-10, atol=1e-10)
    assert np.isclose(T_exit, T0 / (1 + (gamma - 1) / 2 * Ma_exit**2))
    assert np.isclose(eta, case._normal_shock_p0_ratio(Ma_shock, gamma))


def test_flow_regime_boundaries_use_exact_critical_pressure_ratios():
    case = SupersonicNozzle5D()
    epsilon = 6.0
    gamma = 1.4
    T0 = 1000.0
    shock_pos = 0.5

    Ma_design = case._area_ratio_to_Ma(epsilon, gamma, supersonic=True)
    p_exit_design = (1 + (gamma - 1) / 2 * Ma_design**2) ** (
        -gamma / (gamma - 1)
    )
    Ma2_sq = (
        ((gamma - 1) * Ma_design**2 + 2)
        / (2 * gamma * Ma_design**2 - (gamma - 1))
    )
    Ma2 = float(np.sqrt(Ma2_sq))
    p02_ratio = case._normal_shock_p0_ratio(Ma_design, gamma)
    p_after_shock = (
        (1 + (gamma - 1) / 2 * Ma2**2) ** (-gamma / (gamma - 1))
        * p02_ratio
    )

    just_below_design = case._nozzle(
        epsilon, np.nextafter(p_exit_design, 0.0), gamma, T0, shock_pos
    )
    just_above_design = case._nozzle(
        epsilon, np.nextafter(p_exit_design, 1.0), gamma, T0, shock_pos
    )
    at_shock_exit = case._nozzle(epsilon, p_after_shock, gamma, T0, shock_pos)
    just_above_shock_exit = case._nozzle(
        epsilon, np.nextafter(p_after_shock, 1.0), gamma, T0, shock_pos
    )

    assert np.isclose(just_below_design[0], Ma_design)
    assert just_above_design[0] < 1.0
    assert np.isclose(at_shock_exit[0], Ma2)
    assert just_above_shock_exit[0] < 1.0
    assert not np.isclose(just_above_shock_exit[0], Ma2)


def test_declared_output_ranges_cover_domain_corners():
    case = SupersonicNozzle5D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper
