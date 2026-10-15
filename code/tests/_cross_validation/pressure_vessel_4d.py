r"""Cross-validation: pressure_vessel_4d — Thin-walled cylindrical pressure vessel.

Independent references  (see references.bib)
----------------------------------------------
\cite{Shigley2020}  §3-14, Thin-walled pressure vessels.
\cite{Boresi2003}   §4.5 (von Mises criterion for biaxial stress state).
Air Force Stress Analysis Manual §8.3.1, Simple Thin Pressure Vessels:
    https://engineeringlibrary.org/reference/simple-thin-pressure-vessels-air-force-stress-manual
NASA/TM-2015-218216, thin pressure-vessel pipe formulation:
    https://ntrs.nasa.gov/api/citations/20160000456/downloads/20160000456.pdf
ASME BPVC Section VIII, Division 1 official product page:
    https://www.asme.org/codes-standards/find-codes-standards/bpvc-viii-1-bpvc-section-viii-rules-construction-pressure-vessels-division-1
ASME Boiler and Pressure Vessel Certification overview:
    https://www.asme.org/certification-accreditation/Boiler-and-Pressure-Vessel-Certification

Formulae (analytical control)
-----------------------------
Hoop stress:   σ_θ  = P·R / t
Axial stress:  σ_z  = P·R / (2t)
Von Mises:     σ_VM = √(σ_θ² − σ_θ·σ_z + σ_z²) = P·R/t · √3/2
Safety factor: FS   = σ_y / σ_VM
Mass/length:   m/L  = ρ_steel · π · [(R+t)² − R²]    (ρ_steel = 7850 kg/m³)

Validation scope:
These checks are analytical controls for the implemented strength-only
thin-wall surrogate.  They validate membrane-stress and von-Mises identities,
and the public Air Force/NASA anchors support the same thin-wall formulae.
They do not provide external case-level validation, calibration, or a full
ASME BPVC design check; this is not a full ASME design/certification model.
The targeted physics suite also recomputes the hoop and axial membrane
stresses separately before assembling the biaxial von-Mises identity, so a
future edit cannot silently preserve the simplified sqrt(3)/2 coefficient
while changing the underlying principal-stress interpretation.
The ASME certification overview is cited only to make the omitted certification,
quality-control-system, inspection, Certification Mark (for example U/UM/U2/U3),
and code-stamp scope explicit; it is not validation data for the analytical
response surface.
The current ASME pages also sharpen omitted scopes that are far beyond this
benchmark: Section VIII Division 1 covers vessels at internal or external
pressure above 15 psig, class-specific material and fabrication-method rules,
and appendices for nondestructive examination and inspection acceptance.
The certification page also lists pressure-relief-device programs such as
UV/UD in addition to vessel scopes. None of those workflows are modelled here.
Thick-wall Lame/radial stress gradients, end-cap and support discontinuities,
nozzle/opening local stresses, external pressure buckling, weld efficiency,
corrosion allowance, fatigue, fracture, hydrostatic testing, inspection,
certification, material allowables, and fabrication rules are outside this
benchmark case.

Private physical-input contract:
The analytical formula is only evaluated for declared benchmark-domain
physical inputs: P in [1, 20] MPa, R in [0.5, 2.0] m, t in [0.005, 0.05] m,
and sigma_y in [200, 600] MPa, with finite two-dimensional input arrays.
Malformed arrays, non-finite values, and domain violations are rejected before
the thin-wall formula is entered.  This is failure-path hardening only; it
does not change legal-domain finite responses.
"""

import numpy as np
import pytest

CASE_ID = "pressure_vessel_4d"
RTOL = 1e-10  # machine-precision expected
RHO_STEEL = 7850.0
SQRT3_2 = np.sqrt(3.0) / 2.0


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


def _analytical(P, R, t, sigma_y):
    """Hand-compute expected outputs from thin-wall + von Mises theory."""
    sigma_VM = P * R / t * SQRT3_2  # MPa
    FS = sigma_y / sigma_VM  # -
    W_per_L = RHO_STEEL * np.pi * ((R + t) ** 2 - R**2)  # kg/m, mass per length
    return np.array([FS, W_per_L, sigma_VM])


# ── Test vectors (physical units: P[MPa], R[m], t[m], σ_y[MPa]) ──────────

P1_PHYS = np.array([[10.0, 0.5, 0.02, 350.0]])
P1_EXPECTED = _analytical(10.0, 0.5, 0.02, 350.0)

P2_PHYS = np.array([[5.0, 1.0, 0.03, 400.0]])
P2_EXPECTED = _analytical(5.0, 1.0, 0.03, 400.0)

P3_PHYS = np.array([[15.0, 1.5, 0.01, 250.0]])
P3_EXPECTED = _analytical(15.0, 1.5, 0.01, 250.0)


class TestCrossValidation:
    """Verify case outputs against hand-computed thin-wall + von Mises results."""

    def test_mid_range(self, case):
        r"""\cite{Shigley2020} §3-14 — standard pressure vessel dimensions."""
        X_norm = case.normalize(P1_PHYS)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0], P1_EXPECTED, rtol=RTOL)

    def test_large_radius(self, case):
        r"""\cite{Boresi2003} §4.5 — large radius, moderate pressure."""
        X_norm = case.normalize(P2_PHYS)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0], P2_EXPECTED, rtol=RTOL)

    def test_high_pressure_thin_wall(self, case):
        """Extreme case: high P, very thin wall → low safety factor."""
        X_norm = case.normalize(P3_PHYS)
        Y = case.evaluate(X_norm)
        np.testing.assert_allclose(Y[0], P3_EXPECTED, rtol=RTOL)

    def test_known_numerical_values(self, case):
        r"""Hand-computed numerical check \cite{Shigley2020} §3-14.

        P=10 MPa, R=0.5 m, t=0.02 m, σ_y=350 MPa
        σ_VM = 10·0.5/0.02 · √3/2 = 250·0.86603 = 216.506 MPa
        FS = 350/216.506 = 1.6166
        m/L = 7850·π·(0.52²−0.5²) = 7850·π·0.0204 = 503.07 kg/m
        """
        X_norm = case.normalize(P1_PHYS)
        Y = case.evaluate(X_norm)
        assert abs(Y[0, 2] - 216.50635) < 0.001  # σ_VM
        assert abs(Y[0, 0] - 1.6166) < 0.001  # FS
