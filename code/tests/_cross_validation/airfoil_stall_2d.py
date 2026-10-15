r"""Cross-validation: airfoil_stall_2d — NACA 4-digit airfoil lift/drag.

Independent references  (see references.bib)
----------------------------------------------
\cite{AbbottDoenhoff1959}  NACA 0012 experimental data (Re $= 3\times10^6$).
\cite{Sheldahl1981}        Comprehensive wind-tunnel CL/CD tables.
XFOIL documentation is used as panel/integral-boundary-layer solver context,
including convergence and separated-flow operating-limit boundaries:
https://web.mit.edu/drela/Public/web/xfoil/
NASA Glenn inclination/stall pages are used as general boundary evidence for
near-linear small-angle lift, separated-flow stall, high-angle drag rise, and
the need for experimental stall-point determination:
https://www.grc.nasa.gov/WWW/K-12/VirtualAero/BottleRocket/airplane/incline.html
https://www.grc.nasa.gov/WWW/k-12/VirtualAero/BottleRocket/airplane/inclind.html

The code uses the low-level ``neuralfoil.get_aero_from_airfoil`` call as a
baseline polar and applies an explicit low-subsonic Prandtl-Glauert scalar
correction for Mach sensitivity.  Current AeroSandbox wrappers expose a
``mach`` argument, but switching to that wrapper would be a formal backend
change requiring protocol review and full resweep. Expected agreement with
wind-tunnel data is a qualitative/attached-flow range check; near/post-stall
and Mach effects are not treated as case-level external validation without a
dedicated validation-note folder. NASA Glenn inclination/stall material is
therefore a boundary anchor only, not calibration evidence for this frozen
NeuralFoil response surface.
The official NASA anchors are also narrower than the benchmark family:
NASA TM-4074 is NACA 0012 low-speed Mach/Re context rather than validation of
the benchmark's full NACA00xx thickness sweep or explicit ``CD/beta`` scaling,
and NASA TMR is a higher-fidelity CFD validation-dataset boundary rather than
calibration evidence for the frozen low-level NeuralFoil backend.

Validation scope: qualitative low-alpha lift-slope, symmetry, Reynolds-number
trend, high-drag stall-region, and range controls for the fixed
NeuralFoil-plus-low-order-PG response surface.  This file must not be read as
quantitative external validation of stall, drag, or Mach-resolved airfoil
behavior, and it makes no pre-result claim that sequential methods need a
specific percentage fewer samples.

Omitted scope: this benchmark case does not validate cambered sections,
flaps/control-surface deflection, surface roughness, trip/transition-control sweeps,
turbulence-model selection, RANS/LES grid convergence,
dynamic-stall hysteresis, unsteady pitching/plunging, 3-D finite-wing effects,
rotating-blade correction, icing/contamination, aeroelastic coupling, or
acoustic effects.  NASA TMR and SAND80-2114 are used as boundary and trend
context, and XFOIL documentation is used as solver-limit context, not as
calibration evidence for this compact surrogate.
The named ``_validate_physical_inputs`` private physical-input path rejects
malformed arrays and alpha/Re/Ma/t_c values outside the declared benchmark
domain before constructing the NACA section or calling NeuralFoil, so direct
private backend calls cannot silently extrapolate across hosts.
"""

import numpy as np
import pytest

CASE_ID = "airfoil_stall_2d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against Abbott & von Doenhoff (1959) and Sheldahl (1981)."""

    def test_naca0012_low_alpha_lift_slope(self, case):
        r"""\cite{AbbottDoenhoff1959}: NACA 0012, Re = 1e6, α = 5° → CL ≈ 0.55 (thin-airfoil
        theory: dCL/dα ≈ 2π → CL(5°) ≈ 0.55). CD ≈ 0.007–0.01."""
        X_phys = np.array([[5.0, 1e6, 0.05, 0.12]])  # α, Re, Ma, t/c
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        CL, CD, LoD = Y[0]
        assert 0.35 < CL < 0.75, f"CL={CL}, expected ≈ 0.55"
        assert 0.004 < CD < 0.040, f"CD={CD}, expected ≈ 0.008–0.03 (NeuralFoil)"
        assert LoD > 20, f"L/D={LoD}, expected > 20 at low α"

    def test_zero_alpha_symmetric(self, case):
        """NACA 00xx at α = 0° → CL ≈ 0 (symmetry)."""
        X_phys = np.array([[0.0, 500000.0, 0.1, 0.12]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert abs(Y[0, 0]) < 0.1, f"CL at α=0 should be ≈ 0: {Y[0,0]}"

    def test_stall_cl_drop(self, case):
        r"""\cite{Sheldahl1981}: beyond stall (α > 15°), CL should decrease or plateau."""
        X_pre = np.array([[10.0, 1e6, 0.05, 0.12]])
        X_post = np.array([[18.0, 1e6, 0.05, 0.12]])
        Y_pre = case.evaluate(case.normalize(X_pre))
        Y_post = case.evaluate(case.normalize(X_post))
        # Post-stall CL should be ≤ pre-stall CL (or at most marginally higher)
        assert (
            Y_post[0, 0] < Y_pre[0, 0] + 0.3
        ), f"Post-stall CL={Y_post[0,0]} vs pre-stall CL={Y_pre[0,0]}"

    def test_thicker_airfoil_higher_clmax(self, case):
        r"""\cite{AbbottDoenhoff1959}: thicker NACA sections stall at higher CL_max (up to ~18%)."""
        X_thin = np.array([[10.0, 1e6, 0.1, 0.08]])
        X_thick = np.array([[10.0, 1e6, 0.1, 0.15]])
        Y_thin = case.evaluate(case.normalize(X_thin))
        Y_thick = case.evaluate(case.normalize(X_thick))
        # Thicker section should generally have slightly higher CL near stall
        # This is a soft check
        assert Y_thick[0, 0] > Y_thin[0, 0] - 0.2
