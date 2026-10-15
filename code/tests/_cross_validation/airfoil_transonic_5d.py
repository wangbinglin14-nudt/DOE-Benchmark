r"""Cross-validation: airfoil_transonic_5d — Transonic drag divergence.

Independent references  (see references.bib)
----------------------------------------------
\cite{Anderson2010}  §11.6 (Drag-divergence Mach number).
\cite{Harris1981}    Experimental transonic CL/CD for NACA 0012.
XFOIL documentation is used as panel/integral-boundary-layer solver context,
including convergence and separated-flow operating-limit boundaries:
https://web.mit.edu/drela/Public/web/xfoil/

Public source anchors:
  NASA TM-100527, high-Reynolds-number transonic NACA 0012 tests:
    https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19880009187.pdf
  NASA History SP-4219, transonic drag-divergence / sound-barrier context:
    https://www.nasa.gov/history/SP-4219/Chapter3.html
  NASA Glenn similarity parameters / Mach number and compressibility context:
    https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/similarity-parameters-2/
  NASA Langley Turbulence Modeling Resource, NACA 0012 validation:
    https://turbmodels.larc.nasa.gov/naca0012_val.html
  NASA NTRS, Turbulence Simulations of Transonic Flows over an NACA-0012 Airfoil:
    https://ntrs.nasa.gov/citations/20220017612
    https://ntrs.nasa.gov/citations/20220018292

Code uses the low-level ``neuralfoil.get_aero_from_airfoil`` call for a
baseline polar and explicit engineering Prandtl-Glauert / wave-drag
corrections for Mach effects. Current AeroSandbox wrappers expose a ``mach``
argument, but switching to the wrapper would be a formal backend change
requiring protocol review and full resweep. These checks are qualitative
range/trend controls, not a claim of 10 % Mach-resolved transonic CFD accuracy.
This is not Mach-resolved NeuralFoil, Euler, RANS, or wind-tunnel validation;
it is not a derivative M_dd solver, not a Korn-relation check, not buffet-onset
modelling, and not shock/separation validation. The CD Sobol table is not a claim that every
case-level important variable is CD-dominant; alpha, camber, and Reynolds
number are retained for the multi-output CL/CL_CD response contract.

The NASA anchors are transonic-test, drag-divergence-phenomenology, and
high-fidelity validation-dataset context only. XFOIL documentation is
tool-limit context only. They are not external validation data for this
NeuralFoil-baseline plus explicit Prandtl-Glauert/wave-drag surrogate,
derivative M_dd, Korn relation, buffet, shock/boundary-layer separation,
grid-converged RANS, or wind-tunnel-calibrated prediction. They are also not
validation of the benchmark's camber/thickness sweep or the compact critical-
Mach / fourth-power wave-drag trigger applied outside the specific NACA 0012
tunnel configurations.
NASA Glenn similarity-parameter material supports the Mach/compressibility
and shock-wave boundary, but it does not calibrate the critical-Mach trigger
or validate the compact drag-rise formula.
The 2022 NASA transonic-turbulence/buffet NACA-0012 material is boundary
evidence for a much richer validation workflow: pre/post-buffet operating
conditions, URANS, DDES, WMLES, FUN3D, shock-location comparisons, and
wind-tunnel data.  Those workflows are not implemented by this algebraic
response surface and do not validate its compact wave-drag trigger.

The named `_validate_physical_inputs` private path rejects malformed arrays
and Ma/alpha/tc/fc/Re values outside the registered benchmark domain before
constructing the NACA section or importing/calling NeuralFoil. This is
failure-path hardening for direct private calls, not a change to legal-domain
finite responses.
"""

import numpy as np
import pytest

CASE_ID = "airfoil_transonic_5d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against Anderson (2010) and Harris (1981) transonic data."""

    def test_drag_divergence_exists(self, case):
        r"""\cite{Anderson2010} §11.6: CD should increase sharply above Ma ≈ 0.7–0.8."""
        # Low Mach
        X_low = np.array([[0.5, 2.0, 0.12, 0.02, 3e6]])
        Y_low = case.evaluate(case.normalize(X_low))
        # High Mach (transonic)
        X_high = np.array([[0.82, 2.0, 0.12, 0.02, 3e6]])
        Y_high = case.evaluate(case.normalize(X_high))
        assert (
            Y_high[0, 0] > Y_low[0, 0]
        ), f"CD should increase in transonic: CD(0.82)={Y_high[0,0]} vs CD(0.5)={Y_low[0,0]}"

    def test_subsonic_low_drag(self, case):
        r"""\cite{Harris1981}: at Ma=0.5, α=2°, NACA 0012, CD ≈ 0.006–0.01."""
        X_phys = np.array([[0.5, 2.0, 0.12, 0.0, 5e6]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 0] < 0.03, f"Subsonic CD={Y[0,0]}, expected < 0.03"

    def test_thinner_airfoil_higher_mdd(self, case):
        r"""\cite{Anderson2010}: thinner sections have higher drag-divergence Mach number.
        At Ma=0.8, thin section should have lower CD than thick section."""
        X_thin = np.array([[0.8, 2.0, 0.09, 0.02, 5e6]])
        X_thick = np.array([[0.8, 2.0, 0.18, 0.02, 5e6]])
        Y_thin = case.evaluate(case.normalize(X_thin))
        Y_thick = case.evaluate(case.normalize(X_thick))
        assert (
            Y_thin[0, 0] < Y_thick[0, 0]
        ), f"Thinner should have lower transonic CD: {Y_thin[0,0]} vs {Y_thick[0,0]}"
