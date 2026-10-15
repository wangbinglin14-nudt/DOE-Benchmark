r"""Cross-validation: plate_with_hole_5d — Stress concentration at an elliptical hole.

Independent references  (see references.bib)
----------------------------------------------
\cite{Pilkey2008}   Chart 4.1 (elliptical hole, finite-width correction).
\cite{Kirsch1898}   Circular hole in infinite plate: Kt = 3.

Formulae
--------
Inglis SCF (infinite plate):
  Kt_x = 1 + 2·(a/b)         (remote tension perpendicular to major axis)
  Kt_y = 1 + 2·(b/a)         (remote tension perpendicular to minor axis)
Heywood finite-width correction (Pilkey Table 4-1):
  K_hw = 1 + 0.5·(2a/W)³ + 0.3·(2a/W)⁵
Biaxial superposition:
  σ_A = σ_x · Kt_x · K_hw − σ_y    (major-axis end, θ = π/2)
  σ_B = σ_y · Kt_y · K_hw − σ_x    (minor-axis end, θ = 0)
  σ_shear = |τ_xy| · 2(1 + a/b) · K_hw
  σ_max = max(|σ_A|, |σ_B|) + σ_shear

Validation scope:
These checks are analytical controls for the implemented Inglis/Kirsch plus
Heywood finite-width surrogate.  They verify stress-concentration identities
superposition logic, and the named _validate_physical_inputs private
physical-input path for the declared formula, not calibrated external validation
of finite-geometry plate tests, plasticity, crack initiation, or arbitrary
notch design.
"""

import numpy as np
import pytest

CASE_ID = "plate_with_hole_5d"
RTOL = 1e-6  # machine-precision for analytical formula


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


def _analytical(a_b, a_W, sx, sy, theta_deg):
    """Hand-compute expected outputs from Inglis + Heywood theory."""
    theta = np.deg2rad(theta_deg)
    sx_eff = sx * np.cos(theta) ** 2 + sy * np.sin(theta) ** 2
    sy_eff = sx * np.sin(theta) ** 2 + sy * np.cos(theta) ** 2
    txy_eff = (sx - sy) * np.sin(theta) * np.cos(theta)

    Kt_x = 1.0 + 2.0 * a_b
    Kt_y = 1.0 + 2.0 / a_b
    Kt_shear = 2.0 * (1.0 + a_b)

    ratio = np.clip(2.0 * a_W, 0.0, 0.75)
    K_hw = 1.0 + 0.5 * ratio**3 + 0.3 * ratio**5

    sig_A = sx_eff * Kt_x * K_hw - sy_eff
    sig_B = sy_eff * Kt_y * K_hw - sx_eff
    sig_shear = abs(txy_eff * Kt_shear * K_hw)
    sigma_max = max(abs(sig_A), abs(sig_B)) + sig_shear
    sigma_max = max(sigma_max, 0.1)

    sigma_nom = max(abs(sx_eff), abs(sy_eff), 1.0)
    Kt = sigma_max / sigma_nom
    tau_max = sigma_max / 2.0

    return np.array([sigma_max, Kt, tau_max])


# ── Test vectors ──────────────────────────────────────────────────────────

# Point 1: circular hole, uniaxial tension → Kt ≈ 3 (Kirsch)
P1_PHYS = np.array([[1.0, 0.1, 200.0, 0.0, 0.0]])
P1_EXPECTED = _analytical(1.0, 0.1, 200.0, 0.0, 0.0)

# Point 2: elliptical hole (a/b = 2), biaxial loading
P2_PHYS = np.array([[2.0, 0.2, 150.0, 50.0, 0.0]])
P2_EXPECTED = _analytical(2.0, 0.2, 150.0, 50.0, 0.0)

# Point 3: high aspect ratio, angled loading
P3_PHYS = np.array([[3.0, 0.3, 100.0, -20.0, 45.0]])
P3_EXPECTED = _analytical(3.0, 0.3, 100.0, -20.0, 45.0)


class TestCrossValidation:
    """Verify against Kirsch (1898) and Pilkey (2008) references."""

    def test_kirsch_circular_hole(self, case):
        r"""\cite{Kirsch1898}: circular hole, uniaxial → Kt ≈ 3.

        Small a/W = 0.1 so finite-width correction is negligible.
        """
        X_norm = case.normalize(P1_PHYS)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0], P1_EXPECTED, rtol=RTOL)
        # Kt should be very close to 3.0 (Kirsch exact)
        assert abs(Y[0, 1] - 3.0) < 0.1, f"Kt={Y[0,1]}, expected ≈ 3.0"

    def test_elliptical_biaxial(self, case):
        r"""\cite{Pilkey2008} Chart 4.1: elliptical hole, biaxial load."""
        X_norm = case.normalize(P2_PHYS)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0], P2_EXPECTED, rtol=RTOL)

    def test_angled_loading(self, case):
        """Angled loading direction (θ = 45°) with negative σ_y."""
        X_norm = case.normalize(P3_PHYS)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0], P3_EXPECTED, rtol=RTOL)
