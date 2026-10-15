r"""Cross-validation: cantilever_beam_5d — shear-corrected rectangular cantilever beam.

Independent references  (see references.bib)
----------------------------------------------
\cite{Shigley2020}  Table A-9, Case 1 (cantilever, point load at tip).
\cite{Roark2002}    Table 8.1, Case 1a.

Formulae (analytical control)
-----------------------------
σ_max = 6·F·L / (b·h²)                  [Pa]
δ_max = F·L³ / (3·E·I_z) + F·L/(κ·G·A),  I_z = b·h³/12, κ = 5/6  [m]
τ_max = 3·F / (2·b·h)                    [Pa]

Validation scope:
These tests are formula-regression controls for the implemented
shear-corrected rectangular cantilever equations.  They verify dimensional
consistency, textbook beam identities, and the named _validate_physical_inputs
private physical-input path for the declared ideal beam model, not calibrated
external validation of arbitrary beam hardware or nonlinear structural
response.
"""

import numpy as np
import pytest

CASE_ID = "cantilever_beam_5d"
RTOL = 1e-6  # machine-precision match expected (both sides are the same formula)


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


def _analytical(F_kN, L, b, h, E_GPa):
    """Hand-compute expected outputs from shear-corrected beam theory."""
    F = F_kN * 1e3  # kN → N
    E = E_GPa * 1e9  # GPa → Pa
    nu = 0.30
    kappa = 5.0 / 6.0
    G = E / (2.0 * (1.0 + nu))
    A = b * h
    I_z = b * h**3 / 12.0  # m⁴
    sigma = F * L * (h / 2) / I_z * 1e-6  # MPa
    delta = (F * L**3 / (3 * E * I_z) + F * L / (kappa * G * A)) * 1e3  # mm
    tau = 3 * F / (2 * b * h) * 1e-6  # MPa
    return np.array([sigma, delta, tau])


# ── Test vectors ──────────────────────────────────────────────────────────

# Point 1 – mid-range values
#   Shigley Table A-9: σ = M·c/I, δ = P·L³/(3EI), τ = 3V/(2A)
P1_PHYS = np.array([[10.0, 1.5, 0.10, 0.20, 200.0]])
P1_EXPECTED = _analytical(10.0, 1.5, 0.10, 0.20, 200.0)

# Point 2 – different corner of the design space
P2_PHYS = np.array([[25.0, 2.0, 0.15, 0.30, 120.0]])
P2_EXPECTED = _analytical(25.0, 2.0, 0.15, 0.30, 120.0)

# Point 3 – near lower bounds (high stress, large deflection)
P3_PHYS = np.array([[40.0, 2.5, 0.06, 0.12, 70.0]])
P3_EXPECTED = _analytical(40.0, 2.5, 0.06, 0.12, 70.0)


class TestCrossValidation:
    """Verify case outputs against hand-computed shear-corrected beam results."""

    def test_mid_range(self, case):
        r"""\cite{Shigley2020} Table A-9 Case 1 — mid-range inputs."""
        X_norm = case.normalize(P1_PHYS)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0], P1_EXPECTED, rtol=RTOL)

    def test_corner(self, case):
        r"""\cite{Roark2002} Table 8.1 Case 1a — large cross-section."""
        X_norm = case.normalize(P2_PHYS)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0], P2_EXPECTED, rtol=RTOL)

    def test_high_stress(self, case):
        """Reference: same formula, small cross-section → high stress."""
        X_norm = case.normalize(P3_PHYS)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0], P3_EXPECTED, rtol=RTOL)

    def test_known_numerical_values(self, case):
        r"""Hand-computed numerical check \cite{Shigley2020} §4-2.

        F=10 kN, L=1.5 m, b=0.10 m, h=0.20 m, E=200 GPa
        I_z = 0.10·0.20³/12 = 6.6667e-5 m⁴
        σ = 10000·1.5·0.10 / 6.6667e-5 = 22.5 MPa
        δ_b = 10000·1.5³ / (3·200e9·6.6667e-5) = 0.8438 mm
        δ_s = 10000·1.5 / ((5/6)·G·0.10·0.20) = 0.0117 mm
        τ = 3·10000 / (2·0.10·0.20) = 0.75 MPa
        """
        X_norm = case.normalize(P1_PHYS)
        Y = case.evaluate(X_norm)
        assert abs(Y[0, 0] - 22.5) < 0.01
        assert abs(Y[0, 1] - P1_EXPECTED[1]) < 0.001
        assert abs(Y[0, 2] - 0.75) < 0.001
