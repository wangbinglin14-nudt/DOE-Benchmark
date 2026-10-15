from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.chemical.laminar_flame_speed_5d import LaminarFlameSpeed5D


def _domain_corners(case):
    lb = np.array([v.lower for v in case.meta.input_variables], dtype=float)
    ub = np.array([v.upper for v in case.meta.input_variables], dtype=float)
    return np.array(np.meshgrid(*zip(lb, ub), indexing="ij")).reshape(5, -1).T


def _case():
    import doe_testbed as db

    return db.get_case("laminar_flame_speed_5d")


def test_low_fidelity_flame_speed_failure_is_not_silent_nan(monkeypatch):
    case = _case()

    def fail_gulder(*_args, **_kwargs):
        raise ValueError("simulated low-fidelity flame failure")

    monkeypatch.setattr(case, "_gulder_formula", fail_gulder)
    x_phys = np.array([[1.0, 300.0, 1.0, 1.0, 0.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys, fidelity="low")


@pytest.mark.parametrize(
    "x_phys",
    [
        np.array([[1.0, 300.0, -1.0, 1.0, 0.0]], dtype=float),
        np.array([[1.0, 300.0, 1.0, 1.2, 0.0]], dtype=float),
        np.array([[1.0, 300.0, 1.0, 1.0, 1.0]], dtype=float),
        np.array([[1.0, np.nan, 1.0, 1.0, 0.0]], dtype=float),
    ],
)
def test_invalid_low_fidelity_inputs_raise_before_archive_rows(x_phys):
    case = _case()

    with pytest.raises(ValueError):
        case._evaluate_core(x_phys, fidelity="low")


@pytest.mark.parametrize(
    "x_phys, message",
    [
        (np.array([1.0, 300.0, 1.0, 1.0, 0.0], dtype=float), "shape"),
        (np.array([[1.0, 300.0, 1.0, 1.0]], dtype=float), "shape"),
        (np.array([[1.0, np.inf, 1.0, 1.0, 0.0]], dtype=float), "non-finite"),
    ],
)
def test_private_physical_input_contract_rejects_malformed_and_nonfinite(x_phys, message):
    case = _case()

    with pytest.raises(ValueError, match=message):
        case._evaluate_core(x_phys, fidelity="low")


@pytest.mark.parametrize(
    "column, bad_value",
    [
        (0, 0.6 - 1e-6),
        (0, 1.4 + 1e-6),
        (1, 300.0 - 1e-4),
        (1, 500.0 + 1e-4),
        (2, 0.5 - 1e-6),
        (2, 20.0 + 1e-5),
        (3, -1e-6),
        (3, 1.0 + 1e-6),
        (4, -1e-6),
        (4, 0.2 + 1e-6),
    ],
)
def test_private_physical_input_contract_rejects_declared_domain_violations(column, bad_value):
    case = _case()
    x_phys = np.array([[1.0, 300.0, 1.0, 1.0, 0.0]], dtype=float)
    x_phys[0, column] = bad_value
    var_name = case.meta.input_variables[column].name

    with pytest.raises(ValueError, match=var_name):
        case._evaluate_core(x_phys, fidelity="low")


@pytest.mark.parametrize("fidelity", ["low", "high"])
def test_invalid_private_inputs_do_not_reach_flame_solvers(monkeypatch, fidelity):
    case = _case()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("solver path should not be called for invalid private input")

    monkeypatch.setattr(case, "_gulder_formula", fail_if_called)
    monkeypatch.setattr(case, "_cantera_flame", fail_if_called)
    x_phys = np.array([[0.6 - 1e-6, 300.0, 1.0, 1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="outside the declared benchmark domain"):
        case._evaluate_core(x_phys, fidelity=fidelity)


@pytest.mark.parametrize(
    "args",
    [
        (1.0, 300.0, -1.0, 1.0, 0.0),
        (1.0, 300.0, 1.0, -0.1, 0.0),
        (1.0, 300.0, 1.0, 1.1, 0.0),
        (1.0, 300.0, 1.0, 1.0, 1.0),
    ],
)
def test_gulder_formula_rejects_inputs_outside_contract(args):
    with pytest.raises(ValueError, match="outside the declared"):
        LaminarFlameSpeed5D._gulder_formula(*args)


def test_high_fidelity_cantera_failure_falls_back_to_low_fidelity(monkeypatch):
    case = _case()

    def fail_cantera(*_args, **_kwargs):
        raise RuntimeError("simulated cantera failure")

    monkeypatch.setattr(case, "_cantera_flame", fail_cantera)
    x_phys = np.array([[1.0, 300.0, 1.0, 1.0, 0.0]], dtype=float)

    y_high = case._evaluate_core(x_phys, fidelity="high")
    y_low = case._evaluate_core(x_phys, fidelity="low")

    assert np.allclose(y_high, y_low)


def test_low_fidelity_pure_hydrogen_stp_speed_matches_reference_band():
    case = _case()
    x_phys = np.array([[1.0, 300.0, 1.0, 0.0, 0.0]])
    y = case.evaluate(case.normalize(x_phys), fidelity="low")

    sl = y[0, 0]
    assert 200.0 <= sl <= 300.0


def test_low_fidelity_methane_stp_reference_preserved():
    case = _case()
    x_phys = np.array([[1.0, 300.0, 1.0, 1.0, 0.0]])
    y = case.evaluate(case.normalize(x_phys), fidelity="low")

    sl = y[0, 0]
    assert 25.0 < sl < 50.0


def test_low_fidelity_adiabatic_flame_temperature_uses_absolute_reference():
    case = _case()
    methane = case.evaluate(
        case.normalize(np.array([[1.0, 300.0, 1.0, 1.0, 0.0]])),
        fidelity="low",
    )[0]
    hydrogen = case.evaluate(
        case.normalize(np.array([[1.0, 300.0, 1.0, 0.0, 0.0]])),
        fidelity="low",
    )[0]

    assert 2150.0 <= methane[1] <= 2300.0
    assert 2300.0 <= hydrogen[1] <= 2500.0
    assert methane[2] < 8.0
    assert hydrogen[2] < 8.5


def test_dilution_preserves_fuel_to_oxidizer_ratio():
    base = {
        "CH4": 0.095,
        "O2": 0.190,
        "N2": 0.715,
    }
    diluted = LaminarFlameSpeed5D._apply_co2_dilution(base, 0.2)

    assert np.isclose(sum(diluted.values()), 1.0)
    assert np.isclose(diluted["CO2"], 0.2)
    assert np.isclose(diluted["CH4"] / diluted["O2"], base["CH4"] / base["O2"])
    assert np.isclose(diluted["O2"] / diluted["N2"], base["O2"] / base["N2"])


def test_low_fidelity_temperature_and_expansion_shapes_are_finite():
    case = _case()
    x_phys = np.array([
        [0.6, 300.0, 1.0, 0.0, 0.0],
        [1.05, 300.0, 1.0, 0.0, 0.0],
        [1.4, 300.0, 1.0, 0.0, 0.0],
    ])
    y = case.evaluate(case.normalize(x_phys), fidelity="low")

    assert np.all(np.isfinite(y))
    assert y[1, 1] > y[0, 1]
    assert y[1, 1] > y[2, 1]
    assert np.all((3.0 <= y[:, 2]) & (y[:, 2] <= 9.0))


def test_declared_output_ranges_cover_low_fidelity_domain_corners():
    case = _case()
    y = case.evaluate(case.normalize(_domain_corners(case)), fidelity="low")

    for j, output in enumerate(case.meta.output_variables):
        lo, hi = output.expected_range
        assert float(np.nanmin(y[:, j])) >= lo
        assert float(np.nanmax(y[:, j])) <= hi


def test_metadata_and_validation_material_declare_default_low_fidelity_scope():
    case = _case()
    validation_text = Path("tests/_cross_validation/laminar_flame_speed_5d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path("doe_testbed/cases/chemical/laminar_flame_speed_5d.py").read_text(
        encoding="utf-8"
    )

    assert "default/low fidelity is a Gülder-style" in validation_text
    assert "not a 10 % validation against detailed chemistry" in validation_text
    assert "headline default rows use low fidelity" in validation_text
    assert "formal default fidelity is\nthe fast Gülder-style" in source_text
    assert "Default/low fidelity" in case.meta.solver_details
    assert "optional high fidelity" in case.meta.solver_details
    assert "headline sweeps use default=low" in case.meta.solver_details
    assert "Target accuracy" not in validation_text
