# Cross-Validation: `rotor_bemt_6d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `rotor_bemt_6d` |
| Formula class | Finite-station hover BEMT plausibility/control surrogate |
| Evidence tier | P2 plausibility-only |
| Validation scope | Local physics identities, trend checks, and source-boundary controls for the implemented compact hover BEMT surrogate. This is not external calibration or production-tool equivalence. |
| Response-surface status | No response-surface change; this note documents the current implementation and review boundary only. |

This Markdown note is a reviewer-facing companion to
`tests/_cross_validation/rotor_bemt_6d.py` and
`tests/test_rotor_bemt_physics.py`. It improves traceability without changing
`case.evaluate(X_norm)`, the case registry, source formulas, or archived
response contracts.

## 2. Implemented Formula Class

The case implements a compact nondimensional hover blade-element/momentum
balance on a fixed radial grid. For each nondimensional station `r_bar`, the
local inflow is solved as a scalar residual before CT/CP/FM integration.

Governing checks pinned by tests:

```text
dCT_momentum = 4 F lambda_i^2 r_bar dr_bar
dCT_blade    = 0.5 sigma CL r_bar^2 dr_bar
F            = 2/pi arccos(exp(-B/2 * (1 - r_bar) / (r_bar sin(phi))))
FM           = CT^(3/2) / (sqrt(2) CP), for positive CT
FM           = 0 for non-positive useful thrust
```

The power coefficient includes a non-ideal hover induced-power floor plus
profile power, so successful positive-thrust rows cannot report power below
the ideal actuator-disk momentum limit. Because the implemented floor is
`CP >= kappa * CT^(3/2) / sqrt(2)` with `kappa = 1.10`, every positive-thrust
row must also satisfy `FM <= 1/kappa`; the regression suite pins this as an
implementation-control bound for the current surrogate.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| NREL AeroDyn Theory Manual, NREL/TP-500-36881: https://www.nrel.gov/docs/fy05osti/36881.pdf | BEM induction and Prandtl tip-loss implementation context. | Helicopter-hover calibration, 15-station grid-convergence validation, or proof that this case is AeroDyn-equivalent. |
| NREL AeroDyn v15 User's Guide and Theory Manual for OpenFAST: https://www.nrel.gov/docs/libraries/wind-docs/aerodyn-manual.pdf | Production BEMT option and wind-turbine AeroDyn scope context. | OpenFAST-equivalent simulation, induction-tolerance equivalence, or airfoil-table workflow validation. |
| NREL AeroDyn overview: https://www.nrel.gov/wind/nwtc/aerodyn | Current production-module context for horizontal-axis wind-turbine aero-elastic simulation, wake/induction modelling, blade airfoil aerodynamics, and tower influence/drag workflows. | Helicopter-hover calibration or proof that this compact finite-station case is AeroDyn/OpenFAST-equivalent. |
| OpenFAST AeroDyn current user-guide index and input documentation: https://openfast.readthedocs.io/en/main/source/user/aerodyn/index.html, https://openfast.readthedocs.io/en/main/source/user/aerodyn/input.html, and https://openfast.readthedocs.io/en/v4.0.5/source/user/aerodyn/input.html | Boundary evidence for Wake_Mod/BEM_Mod, DBEMT_Mod, Skew_Mod, UA_Mod, BEM algorithm options, dynamic-wake/dynamic-inflow options, OLAF free-vortex-wake options, unsteady-airfoil options, tower/nacelle sections, modelling considerations, and driver/input/output-file workflows omitted here. | Validation that this compact hover-only surrogate implements OpenFAST contracts. |
| OpenFAST AeroDyn introduction: https://openfast.readthedocs.io/en/dev/source/user/aerodyn/introduction.html | Boundary evidence that AeroDyn is a time-domain wind-turbine aerodynamics module in a coupled aero-elastic environment. | Evidence for rotorcraft hover-test calibration or OpenFAST-equivalent behavior. |
| NASA NDARC documentation: https://ntrs.nasa.gov/citations/20170000454 | Rotorcraft conceptual-design coefficient context. | NDARC-equivalent component maps, sizing workflow, mission analysis, or calibrated rotor performance. |
| NASA Ames NDARC reports and papers / NDARC release 1.19: https://rotorcraft.arc.nasa.gov/ndarc/index.php/reports-and-papers and https://ntrs.nasa.gov/citations/20250010468 | Current rotorcraft conceptual-design workflow boundary context. | NDARC-equivalent component maps, mission/trim validation, sizing workflow validation, or proof that this six-input hover surrogate matches NDARC. |
| NASA 1.7 m rotor performance addendum: https://ntrs.nasa.gov/citations/20260000902 | Public CT/CP/FM reporting and hover-test context. | Dataset fitting, uncertainty ingestion, or validation of the registered response surface. |
| NASA Ames HVAB main page: https://rotorcraft.arc.nasa.gov/HVAB/ | Public hover validation/acoustic/wake-data context. | HVAB-calibrated rotor performance. |
| NASA Ames HVAB data-use recommendations: https://rotorcraft.arc.nasa.gov/HVAB/generalInformation.html | Boundary evidence that HVAB products require explicit run/data selection and uncertainty-aware use. | Automatic validation of this benchmark case. |
| NASA Ames HVAB performance data: https://rotorcraft.arc.nasa.gov/HVAB/performanceData.html | Evidence that primary performance files with mean values and uncertainty estimates exist. | Ingestion, fitting, or comparison by this registered case. |
| NASA HVAB blade-set definition: https://ntrs.nasa.gov/citations/20205002153 | Boundary evidence that real hover-validation datasets depend on explicit rotor geometry and test-article definition. | Geometry matching, blade-set calibration, or HVAB validation by this compact surrogate. |
| NASA CFD validation using HVAB data: https://ntrs.nasa.gov/citations/20230004378 | Boundary evidence that real hovering-rotor validation can include performance, airloads, acoustics, wake, transition, and deformed-geometry comparisons. | CFD validation, transition/deformation comparison, or ingestion of HVAB data by this benchmark. |
| NASA UAM proprotor test validation: https://ntrs.nasa.gov/citations/20220015637 | Boundary evidence for static-hover, axial-forward-flight, and acoustic/aerodynamic proprotor validation workflows. | Calibration of this compact hover BEMT surrogate or evidence of proprotor/acoustic/forward-flight validation. |
| NASA TTR/699 full-scale proprotor numerical-data addendum: https://ntrs.nasa.gov/citations/20260001349 | Coefficient-definition context and boundary evidence for full-scale proprotor wind-tunnel conversion/airplane/helicopter-mode data products. | Calibration of this hover-only six-input BEMT surrogate or evidence of TTR/699-equivalent conversion-mode performance. |
| NASA TTR/699 vertical-climb numerical-data addendum: https://ntrs.nasa.gov/citations/20260001346 | Boundary evidence that near-hover/vertical-climb proprotor validation data require explicit axial-flow wind-tunnel conditions and test-program metadata. | Validation of the fixed 15-station hover BEMT response surface or ingestion of vertical-climb data. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not AeroDyn-equivalent, not
OpenFAST-equivalent, not NDARC-equivalent, and not HVAB-calibrated. It omits
hub loss, tangential induction, dynamic inflow, free wake, unsteady airfoil
physics, dynamic stall, Reynolds/Mach-resolved airfoil tables, forward flight,
rotor trim, ground effect, tip-Mach compressibility, blade-vortex interaction,
structural dynamics, acoustics, tower/nacelle effects, production AeroDyn
driver/input/output workflows, OLAF free-vortex-wake workflows, and
rotor-specific calibration. It does not perform HVAB blade-set geometry
matching, hovering-rotor CFD validation, transition-state comparison,
deformed-geometry comparison, or airload/wake/acoustic validation. It is also
not NASA UAM proprotor test-validation
or acoustic/aerodynamic proprotor validation. It is not TTR/699 full-scale
proprotor wind-tunnel validation, conversion-mode validation, airplane-mode
validation, helicopter-mode validation, or vertical-climb data calibration.

The documented equations and source anchors support local BEMT plausibility,
nondimensional identities, and production-tool boundary honesty. They do not
support a strong external case-validation claim unless a future pass adds a
registered validation-note folder with case-specific hover-test or high-fidelity
reference comparisons.
