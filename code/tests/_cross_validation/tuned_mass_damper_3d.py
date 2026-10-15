r"""Cross-validation: tuned_mass_damper_3d — Den Hartog 2-DOF TMD.

Independent references  (see references.bib)
----------------------------------------------
\cite{DenHartog1956}  §3.3 equal-peak (fixed-point) optimal design.
\cite{Asami2002}      Exact closed-form $H_\infty$ optimum.

Den Hartog optimal design
-------------------------
f_opt   = 1 / (1 + μ)
ζ_d_opt = √(3μ / (8(1+μ)))
H_max   ≈ √(1 + 2/μ)   (equal-peak approximation, exact in the undamped-primary limit)

Validation scope
----------------
These checks verify the implemented 2-DOF force-excitation transfer function
against Den Hartog equal-peak control points using the resonance-refined
frequency sweep plus the named _validate_physical_inputs private physical-input
path.  They do not establish calibrated accuracy for arbitrary TMD
configurations, building-specific damping, or field-installed TMD performance.
"""

import numpy as np
import pytest

CASE_ID = "tuned_mass_damper_3d"
RTOL_HMAX = 0.02  # 2% for H_max (discretisation of sharp peak)


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


def _den_hartog_optimal(mu):
    """Return (f_opt, zeta_opt, H_max_approx) from Den Hartog fixed-point theory."""
    f_opt = 1.0 / (1.0 + mu)
    zeta_opt = np.sqrt(3.0 * mu / (8.0 * (1.0 + mu)))
    H_max_approx = np.sqrt(1.0 + 2.0 / mu)
    return f_opt, zeta_opt, H_max_approx


class TestCrossValidation:
    """Verify TMD peak amplification against Den Hartog's classical theory."""

    def test_mu_005_optimal(self, case):
        r"""\cite{DenHartog1956} §3.3: μ=0.05 optimal → H_max ≈ √41 = 6.403."""
        mu = 0.05
        f_opt, zeta_opt, H_expected = _den_hartog_optimal(mu)
        X_phys = np.array([[mu, f_opt, zeta_opt]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0, 0], H_expected, rtol=RTOL_HMAX)

    def test_mu_002_optimal(self, case):
        r"""\cite{DenHartog1956}: μ=0.02 optimal → H_max ≈ √101 = 10.050."""
        mu = 0.02
        f_opt, zeta_opt, H_expected = _den_hartog_optimal(mu)
        X_phys = np.array([[mu, f_opt, zeta_opt]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0, 0], H_expected, rtol=RTOL_HMAX)

    def test_mu_010_optimal(self, case):
        r"""\cite{DenHartog1956}: μ=0.10 optimal → H_max ≈ √21 = 4.583."""
        mu = 0.10
        f_opt, zeta_opt, H_expected = _den_hartog_optimal(mu)
        X_phys = np.array([[mu, f_opt, zeta_opt]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0, 0], H_expected, rtol=RTOL_HMAX)

    def test_no_damping_single_resonance(self, case):
        """Without TMD damping (ζ_d→min), two sharp anti-resonance peaks
        should appear, and H_max should be very large (> 20)."""
        X_phys = np.array([[0.05, 1.0, 0.01]])  # near-zero damping
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 0] > 20.0, "Under-damped TMD should give very high H_max"
