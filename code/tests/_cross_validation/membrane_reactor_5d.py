"""Cross-validation: membrane_reactor_5d — Membrane reactor ODE model.

Independent references  (see references.bib)
----------------------------------------------
\\cite{Fogler2016}      §6.7 (membrane reactors, selective product removal).
\\cite{Levenspiel1999}  §6.5 (equilibrium-limited reactions).

Key physics:
  A ⇌ B reversible reaction with membrane removing product B.
  Higher Da → more reaction → higher X_A.
  Higher Da_m → more product removal → shifts equilibrium → higher X_A.
  At K_eq → ∞, reaction is irreversible; membrane effect diminishes.

Validation scope
----------------
These checks verify reversible plug-flow reaction/permeation trends and
bounded outputs for the implemented Pe -> infinity ODE surrogate.  They do not
establish a 5 % case-level match to a calibrated membrane reactor, axial-
dispersion BVP, or concentration-polarisation experiment.

Private physical-input contract
-------------------------------
The private `_evaluate_core(X_phys, ...)` interface rejects malformed arrays,
non-finite values, and physical inputs outside the declared benchmark domain
before the ODE solver is called. This prevents invalid direct calls from being
archived as fallback or clipped pseudo-results during large Linux sweeps; legal
finite-domain ODE responses are unchanged by this reliability hardening.
"""

import numpy as np
import pytest

CASE_ID = "membrane_reactor_5d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against \\cite{Fogler2016} membrane reactor principles."""

    def test_higher_da_higher_conversion(self, case):
        """\\cite{Fogler2016} §6.7: higher Da → more residence time → higher X_A."""
        X_low = np.array([[0.5, 0.5, 5.0, 10.0, 0.0]])
        X_high = np.array([[8.0, 0.5, 5.0, 10.0, 0.0]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert Y_high[0, 0] > Y_low[0, 0], f"Higher Da → higher X_A: {Y_high[0,0]} vs {Y_low[0,0]}"

    def test_membrane_shifts_equilibrium(self, case):
        """\\cite{Fogler2016}: product removal via membrane increases X_A."""
        X_no_mem = np.array([[2.0, 0.01, 2.0, 20.0, 0.0]])  # low Da_m
        X_mem = np.array([[2.0, 3.0, 2.0, 20.0, 0.0]])  # high Da_m
        Y_no = case.evaluate(case.normalize(X_no_mem))
        Y_mem = case.evaluate(case.normalize(X_mem))
        assert (
            Y_mem[0, 0] >= Y_no[0, 0]
        ), f"Membrane should increase X_A: {Y_mem[0,0]} vs {Y_no[0,0]}"

    def test_conversion_bounded(self, case):
        """X_A must be in [0, 1]."""
        X_phys = np.array([[5.0, 2.0, 10.0, 50.0, 0.05]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert 0 <= Y[0, 0] <= 1.0, f"X_A={Y[0,0]} out of [0,1]"
