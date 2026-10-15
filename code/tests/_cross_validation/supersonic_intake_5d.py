r"""Cross-validation: supersonic_intake_5d - inviscid shock-system surrogate.

Independent references  (see references.bib)
----------------------------------------------
\cite{Anderson2010}  Table A.2 (Normal shock), §9.3 (Oblique shock).
\cite{NACATN1135}    Equations, Tables, and Charts for Compressible Flow.
NASA Glenn oblique/normal-shock pages and the NASA SUPIN 2012/2024 technical
memoranda provide public equation and external-compression inlet-design
context.
NASA Glenn's inlet overview also supports the performance framing used here:
supersonic inlet flow is slowed before the compressor, pressure recovery is
an inlet performance index, and spillage/capture losses are real inlet effects
outside this compact shock-system surrogate.
NASA Glenn's inlet-performance page sharpens that production boundary: a good
inlet must produce high pressure recovery, low spillage drag, and low
distortion. This benchmark does not model those compressor-face propulsion
metrics directly; it only tracks a compact shock-system pressure-recovery
surrogate plus exit Mach and static-compression outputs.
NASA Glenn's inlet-performance model page adds a more concrete wind-tunnel
boundary: engine airflow is varied while pressure recovery, spillage drag, and
distortion are measured or visualized. Those airflow-sweep and rake-style
measurement workflows are outside this registered shock cascade.
Current NASA Glenn inlets/nozzles project and design-software pages are used
only as boundary evidence for omitted CFD, wind-tunnel, propulsion-system
integration, real-time test-display, digital engine/aircraft architecture,
SUPIN grid-generation, and engine-face distortion workflows. They do not
validate this two-ramp algebraic shock surrogate.

Key verifiable property: for attached weak oblique shocks, the intake total
pressure recovery is the product of each oblique-shock total-pressure ratio
and the terminal normal-shock total-pressure ratio.  A direct normal shock
at the freestream Mach number is a useful lower-recovery reference for
moderate external-compression ramps.

Validation scope: qualitative shock-loss trends, normal-shock reference
values, and oblique-shock identity controls for an inviscid two-ramp
surrogate.  This file must not be read as 5% external validation of a
complete inlet, CFD, bleed, shock-train, unstart, or off-design model.
Shock-detachment penalties and the terminal-shock branch create finite
response jumps, so the benchmark topology is explicitly discontinuous rather
than globally smooth.
The oblique-shock helper deliberately selects the weak attached-shock branch;
strong-shock branch selection and back-pressure-driven shock-position solves
are production inlet/design questions outside this benchmark case.
It also does not validate shock/boundary-layer interaction, cowl-lip geometry,
capture-area or spillage behaviour, spillage drag, inlet buzz, compressor-face distortion,
diffuser area scheduling, inlet flow-rate accounting, inlet drag build-up,
surface-grid or CFD-grid output, inlet-engine matching, variable geometry,
strong-shock branch selection, back-pressure-driven shock-position solves,
3-D sidewall/corner flow, X-59 inlet/nozzle CFD, wind-tunnel inlet-distortion
testing, engine-face distortion or operability assessment, propulsion-system
integration, real-time test displays, digital engine/aircraft architecture,
wind-tunnel engine-airflow sweeps, pressure-rake recovery/distortion
measurements,
calorically imperfect shocks, vibrational-energy
effects, real-gas chemistry, wind-tunnel calibration, or a SUPIN-style inlet
design calculation. The gamma input is a constant-gamma gas-property stressor,
not a high-temperature thermochemical model. The 2024 SUPIN technical memorandum
is boundary evidence for what production inlet-design tooling records
(geometry, flowpath components, pressure recovery, flow rates, drag, and
surface/CFD-grid outputs), not validation data for this compact two-ramp benchmark.
The VS-F screening note distinguishes the stored eta_pr Sobol table from the
full five-input shock-cascade interaction structure.

Formula-domain guard: the public NASA Glenn/NACA shock equations used here
are supersonic constant-gamma relations.  The implementation therefore
rejects non-finite helper inputs, upstream Ma <= 1, non-positive compression
deflections, or gamma <= 1 instead of silently extrapolating those formulas
outside their stated domain.
The named `_validate_physical_inputs` private physical-input path rejects malformed arrays and
Ma_in/delta1/delta2/gamma/alpha_off values outside the registered benchmark
domain before oblique-shock cascade evaluation, so direct private calls cannot
extrapolate beyond the declared two-ramp intake bounds.
"""

import numpy as np
import pytest

CASE_ID = "supersonic_intake_5d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


def _normal_shock_p0_ratio(Ma, gamma):
    """Total pressure ratio across a normal shock (Anderson Table A.2)."""
    g = gamma
    t1 = ((g + 1) * Ma**2 / ((g - 1) * Ma**2 + 2)) ** (g / (g - 1))
    t2 = ((g + 1) / (2 * g * Ma**2 - (g - 1))) ** (1 / (g - 1))
    return t1 * t2


class TestCrossValidation:
    r"""Verify against \cite{Anderson2010} normal/oblique shock tables."""

    def test_monotone_eta_vs_mach(self, case):
        """η_pr should decrease with increasing Ma_in (stronger shocks)."""
        Y_list = []
        for ma in [2.0, 3.0, 4.0]:
            X_phys = np.array([[ma, 10.0, 10.0, 1.4, 0.0]])
            X_norm = case.normalize(X_phys)
            Y = case.evaluate(X_norm)
            Y_list.append(Y[0, 0])
        assert Y_list[0] > Y_list[1] > Y_list[2], f"eta_pr should decrease with Ma: {Y_list}"

    def test_ma2_reasonable_eta(self, case):
        """At Ma=2.0 with moderate ramps, η_pr should be > 0.9.

        Anderson2010 Table A.2: normal shock at Ma=2 gives p02/p01 = 0.7209.
        With oblique pre-compression (two 10° ramps), η should be higher.
        """
        X_phys = np.array([[2.0, 10.0, 10.0, 1.4, 0.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 0] > 0.85, f"eta_pr={Y[0,0]}, expected > 0.85 for Ma=2.0"

    def test_ma3_eta_bounded(self, case):
        """At Ma=3.0, oblique + normal shock system; η_pr < 1 always."""
        X_phys = np.array([[3.0, 12.0, 12.0, 1.4, 0.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert 0.3 < Y[0, 0] < 1.0
        assert Y[0, 1] < 3.0, "Ma_exit should be subsonic or low supersonic"

    def test_zero_alpha_symmetry(self, case):
        """α_off = 0 should give same result as baseline (no spillage)."""
        base = np.array([[2.5, 10.0, 10.0, 1.4, 0.0]])
        perturbed = np.array([[2.5, 10.0, 10.0, 1.4, 0.0]])
        Y_base = case.evaluate(case.normalize(base))
        Y_pert = case.evaluate(case.normalize(perturbed))
        np.testing.assert_allclose(Y_base, Y_pert, rtol=1e-10)
