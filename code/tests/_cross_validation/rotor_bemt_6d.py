r"""Cross-validation: rotor_bemt_6d — Helicopter rotor hover (BEMT).

Independent references  (see references.bib)
----------------------------------------------
\\cite{Leishman2006}  §3 (BEMT in hover), §2.4 (Prandtl tip-loss factor).
\\cite{Johnson2013}   §3 (hover performance, figure of merit).
NREL AeroDyn Theory Manual: BEM induction/tip-loss implementation context.
NASA NDARC documentation: rotorcraft performance coefficient context.
Current NASA NDARC release 1.19 documentation: current rotorcraft
conceptual-design workflow boundary context, not component-map equivalence for
this compact case.
NREL AeroDyn User's Guide: wind-turbine BEM implementation and production-tool
scope context, not helicopter-hover or 15-station grid-convergence validation.
NREL AeroDyn v15 / OpenFAST documentation: production BEMT option, Prandtl
tip-loss, hub-loss, induction-tolerance, and airfoil-table workflow context.
OpenFAST online AeroDyn documentation: Wake_Mod/BEM_Mod, DBEMT_Mod, Skew_Mod,
UA_Mod, tower-potential/tower-shadow/tower-aero options, airfoil files, and
driver/output-file contracts define production-tool boundaries not implemented
by this compact case; when AeroDyn is used in standalone mode, an additional
driver input file is required. The current OpenFAST introduction also frames
AeroDyn as a time-domain wind-turbine aerodynamics module inside a coupled
aero-elastic simulation environment, which is outside this hover-only
surrogate. The current OpenFAST v5.0.0 online manual further lists BEM
algorithm options, dynamic-wake/dynamic-inflow options, OLAF free-vortex-wake
options, unsteady-airfoil options, blade/airfoil input sections, and
hub/nacelle/tower/output sections that remain outside this case. The NREL
AeroDyn overview page is likewise used only as current
production-module context for horizontal-axis wind-turbine aero-elastic
simulation, wake/induction models, blade airfoil aerodynamics, and tower
influence/drag boundaries. Public HVAB data products likewise include validation-oriented run
lists, explicit data-use recommendations for combining performance/wake/
acoustic products, and uncertainty-aware performance products not provided by
this compact benchmark. The official performance-data page publishes primary
rotor performance files with mean values and uncertainty estimates for
critical data runs; the benchmark does not ingest, fit, or compare those files.
NASA HVAB blade-set and hovering-rotor CFD-validation reports are used only as
boundary evidence that real validation workflows include geometry/test-article
definition, deformation/transition/wake/airloads/performance/acoustic products,
and explicit data comparisons not present in this benchmark.
NASA 1.7 m rotor performance data addendum: public CT/CP/FM reporting context.
NASA Ames Hover Validation and Acoustic Baseline data: public hover/acoustic/
wake measurement context, not calibration data for this compact surrogate.
NASA UAM proprotor test-validation material is additional boundary evidence
for static-hover, axial-forward-flight, and acoustic/aerodynamic validation
workflows; it is not calibration data for this six-input hover BEMT surrogate.
NASA TTR/699 numerical-data addenda are used as current proprotor
coefficient-definition and wind-tunnel test-program boundary context only:
conversion/airplane/helicopter-mode data and vertical-climb axial-flow
conditions are not ingested, fitted, or compared by this hover-only case.

Public source anchors:
  NREL AeroDyn Theory Manual, NREL/TP-500-36881:
    https://www.nrel.gov/docs/fy05osti/36881.pdf
  NREL AeroDyn v15 User's Guide and Theory Manual for OpenFAST:
    https://www.nrel.gov/docs/libraries/wind-docs/aerodyn-manual.pdf
  NASA NDARC Design and Analysis of Rotorcraft, NASA/TP-2015-218751:
    https://ntrs.nasa.gov/citations/20170000454
  NASA Ames NDARC reports and papers / NDARC release 1.19:
    https://rotorcraft.arc.nasa.gov/ndarc/index.php/reports-and-papers
    https://ntrs.nasa.gov/citations/20250010468
  NASA 1.7 m rotor performance test-data addendum:
    https://ntrs.nasa.gov/citations/20260000902
  NASA Ames Hover Validation and Acoustic Baseline rotor data:
    https://rotorcraft.arc.nasa.gov/HVAB/
  NASA Ames HVAB data-use recommendations:
    https://rotorcraft.arc.nasa.gov/HVAB/generalInformation.html
  NASA Ames HVAB performance data:
    https://rotorcraft.arc.nasa.gov/HVAB/performanceData.html
  NASA HVAB blade-set definition:
    https://ntrs.nasa.gov/citations/20205002153
  NASA CFD validation using HVAB data:
    https://ntrs.nasa.gov/citations/20230004378
  NASA UAM proprotor test validation:
    https://ntrs.nasa.gov/citations/20220015637
  NASA TTR/699 full-scale proprotor numerical-data addendum:
    https://ntrs.nasa.gov/citations/20260001349
  NASA TTR/699 vertical-climb numerical-data addendum:
    https://ntrs.nasa.gov/citations/20260001346
  OpenFAST AeroDyn input-file documentation:
    https://openfast.readthedocs.io/en/main/source/user/aerodyn/index.html
    https://openfast.readthedocs.io/en/main/source/user/aerodyn/input.html
    https://openfast.readthedocs.io/en/v4.0.5/source/user/aerodyn/input.html
  OpenFAST AeroDyn current introduction:
    https://openfast.readthedocs.io/en/dev/source/user/aerodyn/introduction.html
  NREL AeroDyn overview:
    https://www.nrel.gov/wind/nwtc/aerodyn
    https://www.nrel.gov/wind/nwtc/aerodyn

Key physics:
  Ideal hover FM = C_T^{3/2} / (√2 · C_P)   (momentum theory limit)
  Non-ideal hover power should not be below the actuator-disk ideal.
  Prandtl tip-loss: F = 2/pi arccos(exp(-B/2 * (1-r)/(r sin(phi)))).
  The implemented inflow is solved station-by-station as a scalar nonlinear
  residual, not accepted after a fixed number of unconverged fixed-point
  iterations.
  Typical FM for good rotors: 0.7–0.8.
  CT increases with collective θ₀.
  Negative linear twist changes radial loading; in this compact uniform-chord
  surrogate, excessive washout can reduce thrust and FM.
  More blades → higher solidity → different FM trade-off.

Scope: qualitative BEMT hover controls and nondimensional identities only.
The finite-station inflow/tip-loss implementation is not externally
calibrated, so this file must not be read as a 5% case-level validation.
It is not an AeroDyn, NDARC, free-wake, dynamic-inflow, unsteady-airfoil,
dynamic-stall, Reynolds/Mach table, structural, acoustic, or rotor-specific
calibration validation. It is not AeroDyn/OpenFAST-equivalent and does not
implement production BEMT residual-equation controls, Wake_Mod/BEM_Mod,
DBEMT_Mod, Skew_Mod, UA_Mod, tower-influence/shadow/aero options, NacelleDrag,
OpenFAST driver/input/output-file contracts, OpenFAST standalone
driver/input/output-file contracts, configurable Prandtl hub/tip-loss switches,
axial/tangential induction drag-term options, induction-tolerance settings, or
OLAF free-vortex-wake options, unsteady-airfoil options, or an airfoil-table
workflow. It also does
not model forward flight, rotor trim, ground effect,
tip-Mach compressibility, blade-vortex interaction, or calibration to NASA
HVAB / hover-test data, including the primary performance spreadsheets with
mean values and uncertainty estimates, blade-set geometry matching,
deformed-geometry comparison, transition-state comparison, hovering-rotor CFD
validation, or airload/wake/acoustic validation.
It is also not NASA UAM proprotor test-validation, axial-forward-flight
validation, or acoustic/aerodynamic proprotor validation.
It is not TTR/699 full-scale proprotor wind-tunnel validation,
conversion-mode validation, airplane-mode validation, helicopter-mode
validation, or vertical-climb data calibration.
Small-sample scenario notes are benchmark hypotheses, not pre-result sample
count claims.
The named `_validate_physical_inputs` private path rejects malformed arrays
and values outside the declared R/Omega/c_bar/theta_0/theta_tw/B benchmark
domain before blade-count rounding or BEMT solving, so invalid rows cannot be
hidden by integer blade-count clamping. Finite but out-of-contract CT/CP/FM
responses are also rejected against the declared archive envelope before any
row can be written as a successful benchmark result; this is a data-integrity
contract, not an external calibration claim.

These public anchors are used as BEMT equation, coefficient-definition,
tool-scope, and public validation-dataset context only. They are not evidence
that this compact six-input surrogate is AeroDyn-equivalent, NDARC-equivalent,
OpenFAST-equivalent, grid-converged, two-blade validated, or calibrated
against NASA HVAB or any other hover-test dataset.
Even the official HVAB data-use page expects users to choose recommended data
combinations and interpret uncertainty-aware products, which is outside this
compact benchmark workflow.
The official HVAB performance-data page reinforces the same boundary because
mean/uncertainty performance products exist but are not part of this registered
response surface.
The NASA HVAB blade-set and CFD-validation reports reinforce that same
boundary: a true validation workflow can depend on exact test-article geometry,
instrumentation, transition state, blade deformation, airload, wake,
performance, and acoustic comparisons, while this benchmark keeps only a
compact finite-station hover BEMT response.
The NASA UAM proprotor test-validation source reinforces the same boundary:
real rotor validation workflows can combine acoustic and aerodynamic
measurements across static-hover and axial-forward-flight tests, but this
benchmark does not ingest, fit, or compare those data.
The NASA TTR/699 numerical-data addenda reinforce the coefficient and
test-program boundary: full-scale conversion/airplane/helicopter-mode and
vertical-climb data exist, but they are not part of this registered response
surface.
"""

import numpy as np
import pytest

CASE_ID = "rotor_bemt_6d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    r"""Verify against \cite{Leishman2006} BEMT hover theory."""

    def test_higher_collective_higher_ct(self, case):
        r"""\cite{Leishman2006} §3: higher θ₀ → higher CT."""
        X_low = np.array([[2.0, 300.0, 0.1, 8.0, -3.0, 4.0]])
        X_high = np.array([[2.0, 300.0, 0.1, 16.0, -3.0, 4.0]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert Y_high[0, 0] > Y_low[0, 0], f"Higher θ₀ → higher CT: {Y_high[0,0]} vs {Y_low[0,0]}"

    def test_fm_bounded(self, case):
        r"""\cite{Johnson2013}: figure of merit 0 < FM < 1."""
        X_phys = np.array([[3.0, 400.0, 0.1, 12.0, -5.0, 4.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert 0 < Y[0, 2] < 1.0, f"FM={Y[0,2]}, expected in (0, 1)"

    def test_negative_twist_changes_loading_and_excessive_washout_degrades_fm(self, case):
        r"""\cite{Leishman2006}: twist changes radial loading; excessive
        washout can degrade useful hover thrust and FM in this compact model."""
        X_no_tw = np.array([[3.0, 350.0, 0.12, 10.0, 0.0, 4.0]])
        X_twist = np.array([[3.0, 350.0, 0.12, 10.0, -4.0, 4.0]])
        X_over_twist = np.array([[3.0, 350.0, 0.12, 10.0, -10.0, 4.0]])
        Y_no = case.evaluate(case.normalize(X_no_tw))
        Y_tw = case.evaluate(case.normalize(X_twist))
        Y_over = case.evaluate(case.normalize(X_over_twist))
        assert not np.isclose(Y_tw[0, 0], Y_no[0, 0]), (
            f"Twist should alter loading: CT_tw={Y_tw[0,0]:.4f} vs "
            f"CT_no={Y_no[0,0]:.4f}"
        )
        assert Y_over[0, 0] < Y_no[0, 0], f"Over-twist CT={Y_over[0,0]:.4f} vs no-twist CT={Y_no[0,0]:.4f}"
        assert Y_over[0, 2] < Y_tw[0, 2], f"Over-twist FM={Y_over[0,2]:.4f} vs moderate twist FM={Y_tw[0,2]:.4f}"

    def test_power_is_not_below_ideal_actuator_disk_power(self, case):
        X_phys = np.array([[3.0, 350.0, 0.12, 12.0, -5.0, 4.0]])
        Y = case.evaluate(case.normalize(X_phys))
        CT, CP, FM = Y[0]
        assert CP >= CT**1.5 / np.sqrt(2.0)
        assert FM < 1.0

    def test_prandtl_tip_loss_formula_uses_inflow_angle(self, case):
        r"""Prandtl tip-loss denominator uses r sin(phi), not r lambda."""
        r_bar = np.array([0.35, 0.70, 0.95])
        lam_i = np.array([0.055, 0.045, 0.035])
        B = 4
        observed = case._prandtl_tip_loss(r_bar, lam_i, B)
        phi = np.arctan(np.abs(lam_i) / r_bar)
        f = B / 2.0 * (1.0 - r_bar) / (r_bar * np.sin(phi))
        expected = 2.0 / np.pi * np.arccos(np.exp(-f))
        assert np.allclose(observed, expected, rtol=1e-12, atol=1e-12)
