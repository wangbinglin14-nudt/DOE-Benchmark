r"""Cross-validation: wing_vlm_6d - Prandtl lifting-line finite-wing surrogate.

Independent references  (see references.bib)
----------------------------------------------
\\cite{Anderson2010}    §5 (Prandtl lifting-line theory, finite wings).
\\cite{KatzPlotkin2001} §12 (numerical lifting-line, sweep correction).
NASA Glenn induced-drag material supports the public
``CDi = CL^2/(pi AR e)`` convention; NASA historical material provides
Prandtl-Glauert rule context, not swept-wing or transonic validation.
NASA Glenn winglet and drag-factor pages are used only to clarify that
induced drag is one drag component and that profile/skin-friction, form,
wave, viscosity, and compressibility drag mechanisms are omitted from this
induced-drag-only response.
MIT AVL documentation is used only as a boundary source for what a true
rigid-aircraft vortex-lattice/AVL lifting-surface model would include; this
implementation does not create an AVL lattice, body/control-surface model,
slender-body model, or Trefftz-plane analysis.
The official MIT AVL User Primer reinforces the same omitted-workflow
boundary with the concrete user-facing setup of run cases, trimming,
control deflections, and output inspection inside the AVL toolchain.
NASA NTRS vortex-lattice preliminary-design material is used as additional
boundary evidence for planform, twist, camber, flap/control-surface, and
Trefftz-plane design workflows inside a broader conceptual-aircraft design
context that are not implemented by this PLLT case.
NASA OpenVSP VSPAERO basics is used as current boundary evidence for
potential-flow aerodynamic analysis built around OpenVSP geometries and
DegenGeom-style thin-surface representations; this benchmark does not run
OpenVSP/VSPAERO or review pressure/wake solutions.  The same NASA page
documents the VSPAERO v3.45 / solver v7 transition, which is treated as
tool-version/workflow boundary evidence only, not as a stable numerical
reference dataset for the registered PLLT response surface.

Public source anchors:
  NASA Glenn Induced Drag Coefficient:
    https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/
  NASA Glenn Winglets:
    https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/winglets/
  NASA Glenn Factors that Affect Drag:
    https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/factors-that-affect-drag/
  NASA History SP-4219, Prandtl-Glauert background:
    https://www.nasa.gov/history/SP-4219/Chapter3.html
  MIT AVL documentation:
    https://web.mit.edu/drela/Public/web/avl/avl_doc.txt
  MIT AVL User Primer:
    https://web.mit.edu/drela/Public/web/avl/AVL_User_Primer.pdf
  NASA NTRS vortex-lattice preliminary-design applications:
    https://ntrs.nasa.gov/citations/19760021082
    https://ntrs.nasa.gov/api/citations/19760021082/downloads/19760021082.pdf
  NASA SP-405 collected volume containing the vortex-lattice material:
    https://ntrs.nasa.gov/citations/19760021075
  NASA OpenVSP VSPAERO basics:
    https://www.nasa.gov/reference/openvsp-vspaero-basics/

Key analytical check: untwisted rectangular finite wing (TR=1, sweep=0, twist=0):
  CL = a₀·α / (1 + a₀/(π·AR)),  a₀ = 2π   (Prandtl)
  CDi = CL² / (π·AR·e),  e = 1 for elliptic loading
  Oswald e is high but not exactly 1 for the rectangular TR=1 geometry.
  The implementation computes induced drag with the denominator-free PLLT
  identity CDi = π·AR·Σ n A_n², then derives e.  This avoids silently
  converting twisted near-zero-lift states into CDi=0/e=1.
  The separate trivial zero-loading state (alpha=0 and twist=0) is pinned as
  CL=0, CDi=0, e=1.0.  This is a finite benchmark-output convention for the
  exactly unloaded wing, not a claim that Oswald efficiency is physically
  measurable at zero lift.

Validation scope: qualitative finite-wing trend and identity controls for a
Prandtl lifting-line/Fourier surrogate.  The retained historical case_id is
not evidence of a full VLM, swept-wing CFD, or externally calibrated
case-specific 5% accuracy claim.
The small-sample scenario is a benchmark-use rationale, not a pre-result
claim that a fixed training size is sufficient.

The public NASA/MIT anchors are formula and boundary evidence only. They are
not external validation data for the PLLT/Fourier surrogate, swept-wing VLM
accuracy, OpenVSP/VSPAERO aerodynamic analysis, flap/control-surface modelling,
multi-lifting-surface design, profile/parasite drag, stall, panel-method, or
CFD behaviour.
OpenVSP/VSPAERO version-transition notes likewise cannot be used as calibration
or validation evidence for this case.
The NASA Glenn winglet and drag-factor anchors are also boundary evidence
only; they do not make ``CDi`` a complete aircraft drag coefficient and do not
validate winglet, profile-drag, wave-drag, or viscous-flow behaviour.

Formula-domain guard: the low-level PLLT helper is defined for finite
positive aspect ratio and taper ratio, subsonic non-negative Mach numbers
for the Prandtl-Glauert correction, and a valid Fourier discretisation.  It
raises explicit errors for out-of-domain helper calls rather than fabricating
archiveable finite-wing coefficients outside the declared surrogate domain.
The named `_validate_physical_inputs` private path rejects malformed arrays
and AR/TR/sweep/alpha/Ma/twist values outside the declared benchmark domain
before PLLT/Fourier assembly, so direct private calls cannot extrapolate
beyond the registered case bounds.
"""

import numpy as np
import pytest

CASE_ID = "wing_vlm_6d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    r"""Verify against \\cite{Anderson2010} Prandtl lifting-line theory."""

    def test_rectangular_wing_lift_slope_order(self, case):
        """Prandtl finite-wing lift-slope estimate at low α.

        AR=10, TR=1, sweep=0, α=4°, Ma=0.1, twist=0.
        a₀=2π, CL_theory = 2π·(4π/180) / (1 + 2/(10)) = 0.4389/1.2 ≈ 0.366
        """
        alpha_deg = 4.0
        AR = 10.0
        a0 = 2 * np.pi
        CL_theory = a0 * np.deg2rad(alpha_deg) / (1 + a0 / (np.pi * AR))
        X_phys = np.array([[AR, 1.0, 0.0, alpha_deg, 0.1, 0.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert abs(Y[0, 0] - CL_theory) / CL_theory < 0.10, f"CL={Y[0,0]}, theory={CL_theory:.4f}"

    def test_oswald_high_for_rectangular_untwisted_wing(self, case):
        """Rectangular untwisted wing should have high but not exact-unity e."""
        X_phys = np.array([[10.0, 1.0, 0.0, 4.0, 0.1, 0.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 2] > 0.85, f"e={Y[0,2]}, expected high e for an untwisted rectangular wing"

    def test_higher_alpha_higher_cl(self, case):
        """Higher α → higher CL (before stall)."""
        X_low = np.array([[8.0, 0.5, 10.0, 2.0, 0.3, 0.0]])
        X_high = np.array([[8.0, 0.5, 10.0, 6.0, 0.3, 0.0]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert Y_high[0, 0] > Y_low[0, 0]

    def test_cdi_scales_with_cl_squared(self, case):
        """CDi ∝ CL²: doubling α (small) should roughly quadruple CDi."""
        X1 = np.array([[10.0, 0.8, 5.0, 2.0, 0.2, 0.0]])
        X2 = np.array([[10.0, 0.8, 5.0, 4.0, 0.2, 0.0]])
        Y1 = case.evaluate(case.normalize(X1))
        Y2 = case.evaluate(case.normalize(X2))
        # CL ∝ α → CDi ∝ α² → CDi ratio ≈ 4
        ratio = Y2[0, 1] / max(Y1[0, 1], 1e-10)
        assert 2.0 < ratio < 6.0, f"CDi ratio={ratio}, expected ≈ 4"
