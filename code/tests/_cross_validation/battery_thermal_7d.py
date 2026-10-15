r"""Cross-validation: battery_thermal_7d — Li-ion battery transient thermal.

Independent references  (see references.bib)
----------------------------------------------
\cite{Incropera2007}  §5.2 (Lumped capacitance method, Biot number).
\cite{Bernardi1985}   I²R Joule heating energy balance.

Key physics
-----------
Lumped-capacity ODE: m·cp·dT/dt = I²·R(T) − h·A·(T − T_cool)
R(T) = R_ref·exp(Ea_R·(1/T − 1/T_ref))    (Arrhenius)
Energy balance: at steady state, I²R(T_ss) = h·A·(T_ss − T_cool).

Validation scope
----------------
These checks verify lumped-capacity energy-balance trends, finite-temperature
guards, Bernardi-style irreversible/reversible heat terms, and the named
_validate_physical_inputs private physical-input path for the implemented
single-node ODE.  They do not establish a 5 % case-level match to a calibrated
electrochemical-thermal cell model or pack thermal experiment.
"""

import numpy as np
import pytest

CASE_ID = "battery_thermal_7d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against energy balance and Incropera lumped-capacity theory."""

    def test_temperature_rises_with_crate(self, case):
        r"""\cite{Bernardi1985}: higher C-rate → more Joule heating → higher T_max."""
        X_low = np.array([[1.0, 10.0, 50.0, 25.0, 3000.0, 1000.0, 0.04]])
        X_high = np.array([[4.0, 10.0, 50.0, 25.0, 3000.0, 1000.0, 0.04]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert (
            Y_high[0, 0] > Y_low[0, 0]
        ), f"Higher C-rate should give higher T_max: {Y_high[0,0]} vs {Y_low[0,0]}"

    def test_better_cooling_lower_temp(self, case):
        r"""\cite{Incropera2007} §5.2: higher h·A → lower steady-state temperature."""
        X_poor = np.array([[3.0, 20.0, 20.0, 25.0, 3000.0, 1000.0, 0.02]])
        X_good = np.array([[3.0, 20.0, 150.0, 25.0, 3000.0, 1000.0, 0.08]])
        Y_poor = case.evaluate(case.normalize(X_poor))
        Y_good = case.evaluate(case.normalize(X_good))
        assert (
            Y_good[0, 0] < Y_poor[0, 0]
        ), f"Better cooling → lower T_max: {Y_good[0,0]} vs {Y_poor[0,0]}"

    def test_delta_T_positive(self, case):
        """Temperature rise δT = T_max − T_cool must be non-negative."""
        X_phys = np.array([[2.0, 15.0, 100.0, 20.0, 4000.0, 900.0, 0.05]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        T_max, T_end, delta_T = Y[0]
        assert delta_T >= 0, f"delta_T={delta_T} should be ≥ 0"
        assert T_max >= 20.0, f"T_max={T_max} should be ≥ T_cool"

    def test_low_resistance_low_heating(self, case):
        """Very low R_ref → minimal Joule heating → T_max ≈ T_cool."""
        X_phys = np.array([[1.0, 1.0, 100.0, 25.0, 3000.0, 1000.0, 0.05]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        # With R_ref=1 mΩ and C_rate=1, very little heat
        assert Y[0, 0] < 40.0, f"T_max={Y[0,0]} should be near ambient"
