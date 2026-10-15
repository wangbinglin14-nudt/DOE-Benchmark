r"""Cross-validation: laminar_flame_speed_5d — Premixed flame speed.

Independent references  (see references.bib)
----------------------------------------------
\cite{Gulder1984}  Semi-empirical laminar burning velocity formula.
\cite{Law2006}     Table 7.1 (CH\textsubscript{4}/air $S_L \approx 36$ cm/s at STP).

Key relation (Gülder formula)
-----------------------------
SL = W · φ^η · exp(−ξ·(φ − σ)²) · (T/T_ref)^α · (P/P_ref)^β
with tabulated constants for methane and propane.

At φ=1.0, T=300 K, P=1 atm (pure methane): SL ≈ 36 cm/s (Law Table 7.1).

Scope: default/low fidelity is a Gülder-style semi-empirical trend surrogate.
These checks validate reference bands and monotone pressure/temperature/
equivalence-ratio trends, not a 10 % validation against detailed chemistry.
The optional high-fidelity path uses Cantera FreeFlame when available, but
headline default rows use low fidelity.

Private physical-input contract
-------------------------------
The private `_evaluate_core(X_phys, ...)` interface rejects malformed arrays,
non-finite values, and physical inputs outside the declared benchmark domain
before either the Gülder surrogate or optional Cantera `FreeFlame` path is
called. This prevents invalid direct calls from being archived as fallback
surrogate rows during large Linux sweeps; legal-domain finite responses are
unchanged by this runtime-contract hardening.
"""

import numpy as np
import pytest

CASE_ID = "laminar_flame_speed_5d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against Law (2006) and Gülder (1984) reference values."""

    def test_stoichiometric_methane_stp(self, case):
        r"""\cite{Law2006} Table 7.1: CH4/air at φ=1.0, T=300K, P=1atm → SL ≈ 36 cm/s."""
        # phi=1.0, T=300K, P=1atm, fuel_frac=1.0(methane), diluent=0
        X_phys = np.array([[1.0, 300.0, 1.0, 1.0, 0.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        SL = Y[0, 0]
        assert 25 < SL < 50, f"SL={SL} cm/s, expected ≈ 36 (Law 2006)"

    def test_lean_mixture_slower(self, case):
        r"""\cite{Gulder1984}: lean mixtures (φ < 1) burn slower than stoichiometric."""
        X_stoich = np.array([[1.0, 300.0, 1.0, 1.0, 0.0]])
        X_lean = np.array([[0.7, 300.0, 1.0, 1.0, 0.0]])
        Y_stoich = case.evaluate(case.normalize(X_stoich))
        Y_lean = case.evaluate(case.normalize(X_lean))
        assert (
            Y_lean[0, 0] < Y_stoich[0, 0]
        ), f"Lean SL={Y_lean[0,0]} should be < stoich SL={Y_stoich[0,0]}"

    def test_higher_pressure_slower(self, case):
        r"""\cite{Law2006} §7.3: SL decreases with pressure for hydrocarbons (β < 0)."""
        X_1atm = np.array([[1.0, 300.0, 1.0, 1.0, 0.0]])
        X_10atm = np.array([[1.0, 300.0, 10.0, 1.0, 0.0]])
        Y_1 = case.evaluate(case.normalize(X_1atm))
        Y_10 = case.evaluate(case.normalize(X_10atm))
        assert (
            Y_10[0, 0] < Y_1[0, 0]
        ), f"Higher P should slow flame: SL(10atm)={Y_10[0,0]} vs SL(1atm)={Y_1[0,0]}"

    def test_higher_temperature_faster(self, case):
        r"""\cite{Gulder1984}: preheated mixture burns faster (T^α, α > 0)."""
        X_cold = np.array([[1.0, 300.0, 1.0, 1.0, 0.0]])
        X_hot = np.array([[1.0, 450.0, 1.0, 1.0, 0.0]])
        Y_cold = case.evaluate(case.normalize(X_cold))
        Y_hot = case.evaluate(case.normalize(X_hot))
        assert (
            Y_hot[0, 0] > Y_cold[0, 0]
        ), f"Preheated should burn faster: SL(450K)={Y_hot[0,0]} vs SL(300K)={Y_cold[0,0]}"
