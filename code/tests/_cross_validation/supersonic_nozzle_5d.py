r"""Cross-validation: supersonic_nozzle_5d — Converging–diverging nozzle flow.

Independent references  (see references.bib)
----------------------------------------------
\cite{Anderson2010}  Table A.1 (Isentropic flow), Table A.2 (Normal shock).
\cite{NACATN1135}    Equations, Tables, and Charts for Compressible Flow.

Key relations (γ = 1.4)
-----------------------
A/A* = (1/Ma)·[(2/(γ+1))·(1+(γ-1)/2·Ma²)]^((γ+1)/(2(γ-1)))
T/T0 = 1 / (1 + (γ-1)/2 · Ma²)

Validation scope:
These checks are analytical controls for the implemented quasi-one-dimensional
isentropic/normal-shock nozzle surrogate.  They verify area-Mach and
temperature-ratio identities for ideal-gas relations, not calibrated external
validation of viscous nozzle flow, boundary layers, separation, or
geometry-specific CFD/experiment.

Private physical-input contract:
The private `_evaluate_core(X_phys, ...)` interface rejects malformed arrays,
non-finite values, and physical inputs outside the declared benchmark domain
before area-Mach or nozzle-regime solves are called. This prevents invalid
direct calls from being archived as fallback pseudo-results during large
Linux sweeps; legal finite-domain responses are unchanged by this
runtime-contract hardening.
"""

import numpy as np
import pytest

CASE_ID = "supersonic_nozzle_5d"
RTOL_DESIGN = 0.01


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


def _isentropic_area_ratio(Ma, gamma):
    """Compute A/A* from isentropic relations."""
    g = gamma
    return (1.0 / Ma) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * Ma**2)) ** ((g + 1) / (2 * (g - 1)))


class TestCrossValidation:
    """Verify against Anderson (2010) isentropic flow tables."""

    def test_design_supersonic_ma3(self, case):
        r"""\cite{Anderson2010} Table A.1: ε ≈ 4.235, γ = 1.4 → Ma = 3.0.

        At design condition (very low back pressure), nozzle is fully expanded.
        T_exit = T0/(1+0.2·9) = T0/2.8, η = 1.0.
        """
        gamma = 1.4
        epsilon = _isentropic_area_ratio(3.0, gamma)  # ≈ 4.2346
        T0 = 1000.0
        X_phys = np.array([[epsilon, 0.01, gamma, T0, 0.5]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert abs(Y[0, 0] - 3.0) < 0.05, f"Ma_exit={Y[0,0]}, expected 3.0"
        T_expected = T0 / (1 + 0.2 * 9)
        assert abs(Y[0, 1] - T_expected) < 10.0, f"T_exit={Y[0,1]}, expected {T_expected:.1f}"
        assert abs(Y[0, 2] - 1.0) < 0.02, f"eta={Y[0,2]}, expected 1.0"

    def test_design_supersonic_ma35(self, case):
        r"""\cite{Anderson2010} Table A.1: ε for Ma=3.5, γ=1.4 → ε ≈ 6.79.

        T_exit = T0/(1+0.2·12.25) = T0/3.45.
        Design p_exit_ratio ≈ 0.0131 > p_back_min=0.01, so fully expanded.
        """
        gamma = 1.4
        epsilon = _isentropic_area_ratio(3.5, gamma)
        T0 = 2000.0
        X_phys = np.array([[epsilon, 0.01, gamma, T0, 0.5]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert abs(Y[0, 0] - 3.5) < 0.10, f"Ma_exit={Y[0,0]}, expected 3.5"
        T_expected = T0 / (1 + 0.2 * 3.5**2)
        assert abs(Y[0, 1] - T_expected) < 20.0

    def test_subsonic_regime(self, case):
        """Near-closed nozzle (p_back ≈ p0): flow is subsonic, Ma < 1."""
        X_phys = np.array([[5.0, 0.99, 1.4, 500.0, 0.5]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 0] < 1.0, "Subsonic regime expected"
        # eta may not be exactly 1 due to internal shock treatment
        assert Y[0, 2] > 0.1, f"Subsonic eta={Y[0,2]}, should be > 0"
