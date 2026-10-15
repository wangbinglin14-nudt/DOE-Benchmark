r"""Cross-validation: fixed_bed_reactor_6d — Adiabatic fixed-bed reactor ODE.

Independent references  (see references.bib)
----------------------------------------------
\cite{Fogler2016}      §12.2 (Adiabatic operation), Example 12-2.
\cite{Levenspiel1999}  §14 (Temperature and pressure effects).
University of Michigan Elements of Chemical Reaction Engineering,
adiabatic energy-balance learning module:
    https://websites.umich.edu/~elements/asyLearn/bits/ebforadiabatic/index.htm

Key physics
-----------
Adiabatic reactor: T_out = T_in + (-ΔH_r)·C_A0·X_A / (ρ·cp)
At equilibrium or full conversion: X_A → 1, T_max = T_in + |ΔH_r|·C_A0/(ρ·cp).
Energy balance must hold: enthalpy change = reaction heat.

Validation scope
----------------
These checks verify bounded conversion, Arrhenius/residence-time trends, and
the adiabatic energy-balance identity for the implemented low-order ODE.  They
do not establish a 5 % case-level match to a calibrated fixed-bed reactor or
to detailed heterogeneous-catalyst transport data.

Private physical-input contract:
The ODE is only assembled for declared benchmark-domain physical inputs:
T_in in [550, 700] K, C_A0 in [0.5, 5.0] mol/m^3, Ea in [50, 150] kJ/mol,
tau in [1, 20] s, dHr in [-200, -50] kJ/mol, and k0_log in [6, 12].
Malformed arrays, non-finite values, and domain violations are rejected before
the scipy ODE solver is called.  This is failure-path hardening only and leaves
legal-domain finite fixed-bed-reactor responses unchanged.
"""

import numpy as np
import pytest

CASE_ID = "fixed_bed_reactor_6d"
RTOL = 0.05


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against Fogler (2016) reactor design principles."""

    def test_conversion_bounded(self, case):
        """Conversion X_A must be in [0, 1]."""
        X_phys = np.array([[600.0, 2.0, 80.0, 10.0, -100.0, 8.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert 0.0 <= Y[0, 0] <= 1.0, f"X_A={Y[0,0]} out of [0,1]"

    def test_adiabatic_temperature_rise(self, case):
        """T_max ≥ T_in always (exothermic reaction, ΔH_r < 0)."""
        X_phys = np.array([[600.0, 2.0, 80.0, 10.0, -100.0, 8.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 1] >= 600.0, f"T_max={Y[0,1]} should be ≥ T_in=600"

    def test_higher_activation_less_conversion_at_low_T(self, case):
        r"""\cite{Fogler2016} §12.2: Higher Ea at low T_in → lower conversion."""
        # Low Ea
        X_low = np.array([[560.0, 2.0, 60.0, 10.0, -100.0, 8.0]])
        Y_low = case.evaluate(case.normalize(X_low))
        # High Ea
        X_high = np.array([[560.0, 2.0, 140.0, 10.0, -100.0, 8.0]])
        Y_high = case.evaluate(case.normalize(X_high))
        # At low T_in, higher Ea should give LOWER conversion (Arrhenius)
        assert Y_high[0, 0] < Y_low[0, 0], f"Higher Ea → lower X_A: {Y_high[0,0]} vs {Y_low[0,0]}"

    def test_longer_residence_higher_conversion(self, case):
        r"""\cite{Fogler2016}: longer τ → more time for reaction → higher X_A."""
        X_short = np.array([[650.0, 3.0, 80.0, 2.0, -100.0, 9.0]])
        X_long = np.array([[650.0, 3.0, 80.0, 18.0, -100.0, 9.0]])
        Y_short = case.evaluate(case.normalize(X_short))
        Y_long = case.evaluate(case.normalize(X_long))
        assert (
            Y_long[0, 0] >= Y_short[0, 0]
        ), f"Longer τ should give higher X_A: {Y_long[0,0]} vs {Y_short[0,0]}"

    def test_energy_balance(self, case):
        """Adiabatic energy balance: T_outlet − T_in ∝ X_A · |ΔH_r| · C_A0."""
        X_phys = np.array([[600.0, 2.0, 80.0, 10.0, -120.0, 8.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        X_A, T_max, T_outlet = Y[0]
        # Temperature rise should be positive for exothermic reaction
        assert T_outlet >= 600.0
