"""
Case A05: Rotor hover performance (Blade Element Momentum Theory, BEMT)
=======================================================================

Engineering background
----------------------
Helicopter rotor hover-performance analysis is foundational to rotor design.
Blade Element Momentum Theory (BEMT) is a standard low-order engineering
framework for rotor aerodynamics, coupling blade-element theory (2-D airfoil
aerodynamics) with momentum theory (stream-tube model) for preliminary-design
trend analysis.
This benchmark implementation is intentionally compact: it omits hub loss,
tangential induction, dynamic stall, Reynolds/Mach-resolved airfoil tables,
unsteady airfoil aerodynamics, free-wake or dynamic-inflow modelling,
skewed-wake correction, tower influence/shadow, nacelle/tower drag, forward
flight, rotor trim, ground effect, tip-Mach compressibility, blade-vortex
interaction, structural dynamics, acoustic constraints, and calibration to a
particular rotor or hover-test dataset.  OpenFAST/AeroDyn production workflows
also include a time-domain wind-turbine aerodynamics module, coupled
aero-elastic simulation, BEM/dynamic-wake/free-vortex/unsteady-airfoil option
families, primary input files, separate blade/airfoil files, standalone driver
inputs, and output-file specifications that this compact benchmark does not
emulate.  Public HVAB data products likewise include
validation-oriented run lists, explicit data-use recommendations for combining
hover/performance/wake/acoustic products, and uncertainty-aware performance
products that are outside this surrogate.  The official HVAB performance-data
page also publishes primary rotor performance files with mean values and
uncertainty estimates for critical runs; those files are not ingested, fitted,
or compared by this benchmark.  Current NASA HVAB blade-set and hovering-rotor
CFD-validation material further shows that real validation workflows include
precise rotor geometry, instrumentation, transition, deformation, airloads,
wake, performance, and acoustic datasets; none of those data products are
ingested, fitted, or compared here.  NASA UAM proprotor test-validation material is
used only as additional boundary evidence for static-hover,
axial-forward-flight, and acoustic/aerodynamic proprotor validation workflows
outside this case.  Recent NASA TTR/699 numerical-data addenda are used only
to sharpen the
coefficient-definition and test-program boundary: they document full-scale
proprotor wind-tunnel data, conversion/airplane/helicopter-mode runs, and
vertical-climb conditions simulated by axial flow, not a calibration of this
compact hover-only BEMT surrogate.
The two-blade corner
of the input domain is retained as a
benchmark stress point, but Prandtl tip-loss accuracy is weaker for very low
blade counts than for the better-conditioned multi-blade cases.  The 15-station
radial discretization is a low-order benchmark choice, not a grid-convergence
claim or a rotor-specific design setting.

Core BEMT equations (solved independently at each radial station):
    Annular hover momentum balance:
      dCT = 4 F lambda_i^2 r_bar dr_bar
    Blade element thrust:
      dCT = 0.5 sigma CL r_bar^2 dr_bar
    Station nonlinear solve:
      lambda_i - sqrt(max(sigma CL(lambda_i) r_bar, 0) / (8 F(lambda_i))) = 0
    Shaft-power coefficient:
      CP = max(CP_blade-ind, kappa CT^(3/2)/sqrt(2)) + CP_profile, kappa = 1.10

Input variables
---------------
R        Rotor radius        [0.5, 5.0]  m      Dimensional scale, cancels in CT/CP/FM
Omega    Rotational speed     [100, 600]  RPM    Dimensional scale, cancels in CT/CP/FM
c_bar    Mean chord ratio     [0.05, 0.20] -     Solidity σ = Bc/(πR)
theta_0  Collective pitch     [5, 20]     deg    Root pitch angle
theta_tw Twist rate           [-10, 0]    deg/R  Washout (negative = linear twist)
B        Number of blades     [2, 6]      -      Discrete integer (rounded to 2–6)

Output responses
----------------
CT       Thrust coefficient   -   CT = T/(ρ·(ΩR)²·πR²)
CP       Power coefficient    -   CP = P/(ρ·(ΩR)³·πR²), with non-ideal induced-power floor
FM       Figure of merit      -   FM = CT^1.5 / (√2·CP), closer to 1 is better

DOE validation value
--------------------
VS-B (effect estimation):
    theta_0 has the strongest main effect on CT; theta_tw affects FM
    (twist distribution influences lift efficiency).  Solidity (via c_bar)
    interacts strongly with theta_0 (high solidity + high pitch → stall).

Scale-invariance note:
    The outputs are nondimensional hover coefficients CT, CP, and FM.
    For fixed nondimensional geometry and pitch distribution, both R and
    Omega cancel from the coefficient definitions.  Use dimensional thrust
    or power outputs, not this case, for RPM-robustness studies.

References
----------
1. Leishman, J. G. (2006). Principles of Helicopter Aerodynamics, 2nd ed.
   Cambridge University Press. ISBN 978-0-521-85860-1.
2. Johnson, W. (2013). Rotorcraft Aeromechanics. Cambridge.
3. Moriarty, P. J. & Hansen, A. C. (2005). AeroDyn Theory Manual.
   NREL/TP-500-36881. https://www.nrel.gov/docs/fy05osti/36881.pdf
4. Johnson, W. (2016). NDARC NASA Design and Analysis of Rotorcraft.
   NASA/TP-2015-218751. https://ntrs.nasa.gov/citations/20170000454
   Current NDARC release 1.19 documentation:
   https://rotorcraft.arc.nasa.gov/ndarc/index.php/reports-and-papers
   https://ntrs.nasa.gov/citations/20250010468
5. Moriarty, P. J. & Hansen, A. C. (2005). AeroDyn User's Guide and
   Theory Manual, NREL/TP-500-36881. Wind-turbine BEM implementation context,
   not helicopter-hover or grid-convergence validation for this surrogate.
   https://www.nrel.gov/docs/fy05osti/36881.pdf
6. NREL. AeroDyn v15 User's Guide and Theory Manual for OpenFAST.
   https://www.nrel.gov/docs/libraries/wind-docs/aerodyn-manual.pdf
7. Litherland, B. L. et al. (2026). Addendum to the U.S. Army/NASA
   1.7-Meter-Diameter Rotor Performance Test Data, NASA/TM-20260000902.
   https://ntrs.nasa.gov/citations/20260000902
8. NASA Ames. Hover Validation and Acoustic Baseline rotor test data.
   https://rotorcraft.arc.nasa.gov/HVAB/
9. NASA Ames. Hover Validation and Acoustic Baseline data-use recommendations.
   https://rotorcraft.arc.nasa.gov/HVAB/generalInformation.html
10. NASA Ames. Hover Validation and Acoustic Baseline performance data.
   https://rotorcraft.arc.nasa.gov/HVAB/performanceData.html
11. NASA NTRS. Hover Validation and Acoustic Baseline blade set.
   https://ntrs.nasa.gov/citations/20205002153
12. NASA NTRS. CFD Validation of a Hovering Rotor using HVAB Data.
   https://ntrs.nasa.gov/citations/20230004378
13. NASA NTRS. Test Validation of Acoustically and Aerodynamically Optimized
   UAM Proprotor Designs.
   https://ntrs.nasa.gov/citations/20220015637
14. NASA NTRS. Wind Tunnel Performance Tests of a Full-Scale Proprotor on the
   Tiltrotor Test Rig Addendum: Numerical Data.
   https://ntrs.nasa.gov/citations/20260001349
15. NASA NTRS. Vertical Climb Testing of the Model 699 Proprotor on the
   Tiltrotor Test Rig Addendum: Numerical Data.
   https://ntrs.nasa.gov/citations/20260001346
16. NREL. OpenFAST AeroDyn online documentation, v4.0.5 / current v5.0.0 docs.
   https://openfast.readthedocs.io/en/main/source/user/aerodyn/index.html
   https://openfast.readthedocs.io/en/main/source/user/aerodyn/input.html
   https://openfast.readthedocs.io/en/v4.0.5/source/user/aerodyn/input.html
   https://openfast.readthedocs.io/en/dev/source/user/aerodyn/introduction.html
17. NREL/NLR AeroDyn overview page, current production-module context.
   https://www.nrel.gov/wind/nwtc/aerodyn
   https://www.nlr.gov/wind/nwtc/aerodyn
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class RotorBEMT6D(BaseCase):
    """Rotor hover performance via BEMT (6D → 3D)."""

    meta = CaseMeta(
        case_id="rotor_bemt_6d",
        name="Rotor hover performance (BEMT)",
        version="1.0.0",
        domain="aerodynamics",
        tags=["rotor", "bemt", "helicopter", "hover", "blade_element", "momentum", "6d"],
        dim_input=6,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="R",
                symbol="R",
                description="Rotor radius (dimensional scale; cancels from CT/CP/FM)",
                unit="m",
                lower=0.5,
                upper=5.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="Omega",
                symbol=r"\Omega",
                description="Rotational speed (dimensional scale; cancels from CT/CP/FM)",
                unit="RPM",
                lower=100.0,
                upper=600.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="c_bar",
                symbol=r"\bar{c}/R",
                description="Mean chord-to-radius ratio (affects solidity σ)",
                unit="-",
                lower=0.05,
                upper=0.20,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="theta_0",
                symbol=r"\theta_0",
                description="Root collective pitch angle",
                unit="deg",
                lower=5.0,
                upper=20.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="theta_tw",
                symbol=r"\theta_{tw}",
                description="Linear twist rate (negative = washout)",
                unit="deg/R",
                lower=-10.0,
                upper=0.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="B",
                symbol="B",
                description="Number of blades (rounded to integer)",
                unit="-",
                lower=2.0,
                upper=6.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="CT",
                symbol="C_T",
                description="Thrust coefficient (negative in non-lifting hover-failure regimes)",
                unit="-",
                is_primary=True,
                # Uniform sampling covers under-pitch / excessive-washout cases
                # whose blade-element integral produces no useful hover thrust.
                expected_range=(-0.025, 0.10),
            ),
            OutputVariable(
                name="CP",
                symbol="C_P",
                description="Power coefficient (induced + profile drag)",
                unit="-",
                is_primary=True,
                expected_range=(0.0, 0.025),
            ),
            OutputVariable(
                name="FM",
                symbol="FM",
                description="Figure of merit (hover efficiency; 0 when CT <= 0); ideal limit FM = 1",
                unit="-",
                is_primary=True,
                expected_range=(0.0, 0.90),
                failure_threshold=0.60,
                failure_direction="below",
            ),
        ],
        governing_equation=(
            "BEMT (coupled blade-element / momentum):\n"
            "  Hover momentum: dCT = 4F·lambda_i^2·r_bar·dr_bar\n"
            "  Blade element: dCT = 0.5·sigma·CL·r_bar^2·dr_bar\n"
            "  Station solve: lambda_i - sqrt(max(sigma·CL(lambda_i)·r_bar,0)/(8F(lambda_i))) = 0\n"
            "  CP = max(CP_blade_ind, kappa·CT^(3/2)/√2) + CP_profile, kappa=1.10\n"
            "  FM = CT^(3/2) / (√2·CP)"
        ),
        pde_type="ode_system",
        spatial_dim=1,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Leishman, J. G. (2006). Principles of Helicopter Aerodynamics, 2nd ed. Cambridge.",
            "Johnson, W. (2013). Rotorcraft Aeromechanics. Cambridge.",
            "Moriarty, P. J. & Hansen, A. C. (2005). AeroDyn Theory Manual. "
            "NREL/TP-500-36881, https://www.nrel.gov/docs/fy05osti/36881.pdf",
            "NASA NTRS, Design and Analysis of Rotorcraft (NDARC) input documentation, "
            "https://ntrs.nasa.gov/citations/20170000454",
            "NASA Ames NDARC reports and papers page and NASA/CR-20250010468 "
            "NDARC release 1.19 theory/input/dictionary documentation; "
            "current conceptual-design workflow boundary context only, not "
            "NDARC-equivalent component maps or mission/trim validation for "
            "this six-input hover BEMT surrogate. "
            "https://rotorcraft.arc.nasa.gov/ndarc/index.php/reports-and-papers; "
            "https://ntrs.nasa.gov/citations/20250010468",
            "NREL, AeroDyn User's Guide and Theory Manual, NREL/TP-500-36881, "
            "horizontal-axis wind-turbine BEM/tip-loss implementation context "
            "and production-tool scope boundaries, not helicopter-hover or "
            "15-station grid-convergence validation, "
            "https://www.nrel.gov/docs/fy05osti/36881.pdf",
            "NREL, AeroDyn v15 User's Guide and Theory Manual for OpenFAST, "
            "BEMT option, Prandtl tip-loss, hub-loss, induction-tolerance, "
            "and airfoil-table input context, "
            "https://www.nrel.gov/docs/libraries/wind-docs/aerodyn-manual.pdf",
            "NASA NTRS, Addendum to the U.S. Army/NASA 1.7-Meter-Diameter Rotor "
            "Performance Test Data, NASA/TM-20260000902, hover-performance "
            "coefficient and figure-of-merit context, https://ntrs.nasa.gov/citations/20260000902",
            "NASA Ames, Hover Validation and Acoustic Baseline rotor test data, "
            "public hover-performance/acoustic/wake measurement context, "
            "https://rotorcraft.arc.nasa.gov/HVAB/",
            "NASA Ames, Hover Validation and Acoustic Baseline data-use "
            "recommendations, public guidance on recommended data combinations, "
            "run selection, and uncertainty-aware use of HVAB products; "
            "boundary/context evidence only, not direct calibration data for "
            "this compact surrogate, "
            "https://rotorcraft.arc.nasa.gov/HVAB/generalInformation.html",
            "NASA Ames, Hover Validation and Acoustic Baseline performance data, "
            "primary rotor performance files with mean values and uncertainty "
            "estimates for critical runs; validation-dataset boundary evidence "
            "only because these files are not ingested, fitted, or compared here, "
            "https://rotorcraft.arc.nasa.gov/HVAB/performanceData.html",
            "NASA NTRS, Hover Validation and Acoustic Baseline blade-set "
            "definition, NASA/TM-2020-5002153; rotor geometry, blade-set, "
            "and test-article definition boundary evidence only, not geometry "
            "matching or HVAB validation for this compact surrogate, "
            "https://ntrs.nasa.gov/citations/20205002153",
            "NASA NTRS, CFD Validation of a Hovering Rotor using HVAB Data, "
            "NASA/TM-20230004378; performance, airloads, acoustics, wake, "
            "transition, and deformed-geometry validation-workflow boundary "
            "context only, not CFD validation or data ingestion by this case, "
            "https://ntrs.nasa.gov/citations/20230004378",
            "NASA NTRS, Test Validation of Acoustically and Aerodynamically Optimized "
            "UAM Proprotor Designs, static-hover / axial-forward-flight and "
            "acoustic/aerodynamic validation-workflow boundary context, not "
            "calibration of this compact hover BEMT surrogate, "
            "https://ntrs.nasa.gov/citations/20220015637",
            "NASA NTRS, Wind Tunnel Performance Tests of a Full-Scale Proprotor "
            "on the Tiltrotor Test Rig Addendum: Numerical Data, "
            "NASA/TP-20260001349; full-scale TTR/699 proprotor coefficient "
            "and conversion/airplane/helicopter-mode data context, not "
            "calibration or validation of this compact six-input hover BEMT "
            "surrogate, https://ntrs.nasa.gov/citations/20260001349",
            "NASA NTRS, Vertical Climb Testing of the Model 699 Proprotor on "
            "the Tiltrotor Test Rig Addendum: Numerical Data, "
            "NASA/TP-20260001346; vertical-climb/near-hover axial-flow data "
            "context and test-program boundary evidence, not finite-station "
            "BEMT response-surface calibration, "
            "https://ntrs.nasa.gov/citations/20260001346",
            "NREL OpenFAST AeroDyn online documentation, v4.0.5 and current "
            "v5.0.0 development documentation, production-model boundary context for "
            "time-domain wind-turbine aerodynamics, coupled aero-elastic "
            "simulation, Wake_Mod/BEM_Mod, DBEMT_Mod, Skew_Mod, UA_Mod, "
            "BEM algorithm options, dynamic-wake/dynamic-inflow options, "
            "OLAF free-vortex-wake options, unsteady-airfoil options, "
            "TwrPotent, TwrShadow, TwrAero, NacelleDrag, primary input files, "
            "airfoil and blade files, standalone driver inputs, and output files, "
            "https://openfast.readthedocs.io/en/main/source/user/aerodyn/index.html; "
            "https://openfast.readthedocs.io/en/main/source/user/aerodyn/input.html; "
            "https://openfast.readthedocs.io/en/v4.0.5/source/user/aerodyn/input.html; "
            "https://openfast.readthedocs.io/en/dev/source/user/aerodyn/introduction.html",
            "NREL/NLR AeroDyn overview page, current production-module context "
            "for horizontal-axis wind-turbine aero-elastic simulation, wake/"
            "induction modelling, blade airfoil aerodynamics, and tower "
            "influence/drag boundaries not implemented by this helicopter-hover "
            "BEMT surrogate. https://www.nrel.gov/wind/nwtc/aerodyn; "
            "https://www.nlr.gov/wind/nwtc/aerodyn",
        ],
        solver_library="numpy (BEMT stationwise nonlinear inflow solver)",
        solver_details=(
            "15 radial stations, stationwise bisection solve for the hover "
            "inflow residual (tol=1e-10, max 80 bisection steps); "
            "the radial grid is a deliberately low-order benchmark "
            "discretization and is not validated as a grid-converged "
            "AeroDyn/OpenFAST, NDARC, or HVAB-equivalent rotor model; "
            "shaft-power coefficient uses sectional torque with a non-ideal hover "
            "momentum-theory induced-power floor plus profile power so FM cannot "
            "exceed the ideal actuator-disk limit by construction; private physical-input "
            "evaluation rejects malformed arrays and values outside the declared "
            "R/Omega/c_bar/theta_0/theta_tw/B domain before rounding the blade count "
            "or solving BEMT; finite but out-of-contract CT/CP/FM responses outside "
            "the declared archive envelope are rejected before benchmark row archival; "
            "no hub-loss, tangential-induction, dynamic-inflow, "
            "free-wake, unsteady-airfoil, dynamic-stall, skewed-wake-correction, "
            "production BEM/DBEMT mode selection, induction-tolerance control, "
            "configurable Prandtl hub/tip-loss switches, axial/tangential induction "
            "drag-term options, OLAF free-vortex-wake options, "
            "Reynolds/Mach-resolved airfoil-table, tower-influence, tower-shadow, "
            "tower-aerodynamic, nacelle-drag, structural-dynamics, forward-flight, "
            "rotor-trim, ground-effect, tip-Mach-compressibility, "
            "blade-vortex-interaction, acoustic, grid-convergence, hover-test-dataset "
            "calibration, HVAB mean/uncertainty performance-data ingestion, "
            "HVAB blade-set geometry matching, hovering-rotor CFD validation, "
            "deformed-geometry or transition-state comparison, "
            "NASA UAM proprotor test-validation, static-hover/axial-forward-flight "
            "test comparison, acoustic/aerodynamic proprotor validation, "
            "rotor-specific calibration model, AeroDyn/OpenFAST-equivalent "
            "BEMT residual-equation controls, time-domain wind-turbine aero-elastic "
            "simulation contract, OpenFAST driver/input/output-file "
            "contract, standalone AeroDyn driver/input/output-file contract, "
            "or production airfoil-table workflow"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 3.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_unimodal", "strong_interaction", "discontinuous"],
        effective_dim=4,
        important_variables=["theta_0", "c_bar", "theta_tw", "B"],
        interaction_pairs=[("theta_0", "c_bar"), ("theta_0", "theta_tw"), ("c_bar", "B")],
        has_failure_boundary=True,
        failure_condition="FM < 0.60 (poor hover efficiency relative to ideal momentum-theory power)",
        sobol_S1={
            "R": 0.000,
            "Omega": 0.000,
            "c_bar": 0.167,
            "theta_0": 0.271,
            "theta_tw": 0.042,
            "B": 0.136,
        },
        sobol_ST={
            "R": 0.000,
            "Omega": 0.000,
            "c_bar": 0.402,
            "theta_0": 0.512,
            "theta_tw": 0.125,
            "B": 0.354,
        },
        sobol_computation="salib_saltelli_n2048_seed42_CT_output",
        applicable_scenarios=["VS-B", "VS-C", "VS-D"],
        scenario_notes={
            "VS-B": "theta_0 has the strongest main effect; interacts with c_bar (high solidity + high pitch → stall).",
            "VS-C": "R and Omega are deliberate scale variables for nondimensional-coefficient invariance checks.",
            "VS-D": (
                "Compact 4-active-variable hover-efficiency response with threshold-like "
                "low-FM regions; use convergence curves to test small-sample design stability, "
                "without pre-claiming a required sample count. B=2 is retained as a "
                "stress-test corner where simple Prandtl tip-loss accuracy is weaker."
            ),
        },
        recommended_n_train={"default": 60},
        n_saturation=250,
        noise_factor_candidates=[],
        control_factor_candidates=["c_bar", "theta_0", "theta_tw", "B"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    _N_RADIAL = 15  # number of radial stations
    _RHO = 1.225  # sea-level air density kg/m³
    _A0 = 2 * np.pi  # thin-airfoil lift slope /rad
    _CD0 = 0.012  # airfoil zero-lift drag coefficient
    _INDUCED_POWER_FACTOR = 1.10  # non-ideal hover induced-power factor
    _INFLOW_LO = 1e-6
    _INFLOW_HI = 0.5
    _INFLOW_RESIDUAL_TOL = 1e-10
    _INFLOW_MAX_BISECTION = 80

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        """Return validated physical inputs or reject private-domain violations."""
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "rotor_bemt_6d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        R_col = X_phys[:, 0]
        Omega_rpm_col = X_phys[:, 1]
        c_bar_col = X_phys[:, 2]
        theta_0_col = X_phys[:, 3]
        theta_tw_col = X_phys[:, 4]
        B_col = X_phys[:, 5]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(R_col < 0.5 - tol)
            or np.any(R_col > 5.0 + tol)
            or np.any(Omega_rpm_col < 100.0 - tol)
            or np.any(Omega_rpm_col > 600.0 + tol)
            or np.any(c_bar_col < 0.05 - tol)
            or np.any(c_bar_col > 0.20 + tol)
            or np.any(theta_0_col < 5.0 - tol)
            or np.any(theta_0_col > 20.0 + tol)
            or np.any(theta_tw_col < -10.0 - tol)
            or np.any(theta_tw_col > 0.0 + tol)
            or np.any(B_col < 2.0 - tol)
            or np.any(B_col > 6.0 + tol)
        ):
            raise ValueError(
                "rotor_bemt_6d physical inputs must stay inside the declared "
                "R/Omega/c_bar/theta_0/theta_tw/B benchmark domain before BEMT evaluation"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                R = float(X_phys[i, 0])
                Omega = float(X_phys[i, 1]) * np.pi / 30.0  # RPM → rad/s
                c_bar = float(X_phys[i, 2])
                theta_0 = np.deg2rad(float(X_phys[i, 3]))
                theta_tw = np.deg2rad(float(X_phys[i, 4]))
                B = int(round(float(X_phys[i, 5])))
                B = max(2, min(6, B))
                y_i = np.asarray(self._bemt(R, Omega, c_bar, theta_0, theta_tw, B), dtype=float)
                if y_i.shape != (3,) or not np.all(np.isfinite(y_i)):
                    raise FloatingPointError("non-finite or malformed rotor BEMT response")
                ct, cp, fm = y_i
                if cp <= 0.0 or fm < 0.0 or fm > 1.0:
                    raise FloatingPointError("non-physical rotor BEMT response")
                output_ranges = np.asarray(
                    [var.expected_range for var in self.meta.output_variables],
                    dtype=float,
                )
                below = y_i < output_ranges[:, 0]
                above = y_i > output_ranges[:, 1]
                if np.any(below | above):
                    raise FloatingPointError(
                        "out-of-contract rotor BEMT response outside declared "
                        "CT/CP/FM archive envelope"
                    )
                Y[i] = y_i
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "rotor_bemt_6d BEMT hover solve failed for physical "
                    f"input {x_text}; refusing silent NaN output because the "
                    "formal benchmark response surface requires finite CT, "
                    "positive CP, FM coefficients inside [0, 1], and CT/CP/FM "
                    "inside the declared archive envelope."
                ) from None
        return Y

    @staticmethod
    def _prandtl_tip_loss(r_bar: np.ndarray, lam_i: np.ndarray, B: int) -> np.ndarray:
        """Prandtl tip-loss factor for hover BEMT.

        The standard normalized form is
        ``F = 2/pi * arccos(exp(-B/2 * (1-r)/(r sin(phi))))`` with
        ``phi = atan(lambda/r)`` in hover.
        """
        r_safe = np.maximum(np.asarray(r_bar, dtype=float), 1e-6)
        lam_safe = np.maximum(np.abs(np.asarray(lam_i, dtype=float)), 1e-8)
        phi = np.arctan(lam_safe / r_safe)
        sin_phi = np.maximum(np.sin(phi), 1e-8)
        f = float(B) / 2.0 * (1.0 - r_safe) / np.maximum(r_safe * sin_phi, 1e-8)
        F = 2.0 / np.pi * np.arccos(np.exp(-np.maximum(f, 0.0)))
        return np.clip(F, 0.01, 1.0)

    @classmethod
    def _station_target_inflow(cls, r_bar: float, sigma: float, theta: float, lam_i: float, B: int) -> float:
        phi = np.arctan(lam_i / (r_bar + 1e-8))
        alpha_b = theta - phi
        CL_b = np.clip(cls._A0 * alpha_b, -1.5, 1.5)
        F = cls._prandtl_tip_loss(np.array([r_bar]), np.array([lam_i]), B)[0]
        return float(np.sqrt(max(sigma * CL_b * r_bar, 0.0) / (8.0 * F)))

    @classmethod
    def _solve_station_inflow(cls, r_bar: float, sigma: float, theta: float, B: int) -> float:
        """Solve the scalar hover BEMT inflow residual for one radial station."""

        def residual(lam_i: float) -> float:
            return lam_i - cls._station_target_inflow(r_bar, sigma, theta, lam_i, B)

        lo = cls._INFLOW_LO
        hi = cls._INFLOW_HI
        r_lo = residual(lo)
        if not np.isfinite(r_lo):
            raise ValueError("non-finite low-end inflow residual")
        if r_lo >= 0.0:
            return lo

        r_hi = residual(hi)
        if not np.isfinite(r_hi):
            raise ValueError("non-finite high-end inflow residual")
        if r_hi <= 0.0:
            raise ValueError(
                "BEMT station inflow root is not bracketed inside the "
                f"declared search interval [{lo}, {hi}]"
            )

        lower = lo
        upper = hi
        root = hi
        for _ in range(cls._INFLOW_MAX_BISECTION):
            root = 0.5 * (lower + upper)
            r_mid = residual(root)
            if not np.isfinite(r_mid):
                raise ValueError("non-finite bisection inflow residual")
            if abs(r_mid) <= cls._INFLOW_RESIDUAL_TOL or (upper - lower) <= cls._INFLOW_RESIDUAL_TOL:
                return root
            if r_mid > 0.0:
                upper = root
            else:
                lower = root

        final_residual = residual(root)
        if abs(final_residual) > 1e-8:
            raise ValueError(f"BEMT station inflow solve did not converge; residual={final_residual!r}")
        return root

    def _bemt(self, R, Omega, c_bar, theta_0, theta_tw, B):
        """BEMT with non-uniform inflow and Prandtl tip-loss correction."""
        if (
            not np.all(np.isfinite([R, Omega, c_bar, theta_0, theta_tw, B]))
            or R <= 0.0
            or Omega <= 0.0
            or c_bar <= 0.0
            or B < 2
        ):
            raise ValueError("invalid rotor BEMT physical input")
        rho = self._RHO
        a0 = self._A0
        cd0 = self._CD0

        # Radial stations (cosine distribution, avoiding tip)
        r_bar = np.linspace(0.05, 0.99, self._N_RADIAL)  # r/R
        r = r_bar * R
        dr = np.gradient(r)

        # Geometric pitch distribution
        c = c_bar * R * np.ones_like(r)  # uniform chord
        theta = theta_0 + theta_tw * r_bar  # linear twist

        # Rotor solidity
        sigma = B * c / (np.pi * R)

        lam_c = 0.0  # climb velocity ratio (= 0 in hover)

        # Stationwise nonlinear solve of the hover equilibrium:
        # 4F lambda^2 r = 0.5 sigma CL(lambda) r^2.
        lam_i = np.array(
            [
                self._solve_station_inflow(float(rb), float(sig), float(th), B)
                for rb, sig, th in zip(r_bar, sigma, theta)
            ],
            dtype=float,
        )

        # Final aerodynamic coefficient integration
        lam = lam_i + lam_c
        phi = np.arctan(lam / (r_bar + 1e-8))
        alpha_b = theta - phi
        CL_b = np.clip(a0 * alpha_b, -1.5, 1.5)
        F = self._prandtl_tip_loss(r_bar, lam_i, B)

        # Normalised thrust coefficient (integration).  Negative CT is kept as
        # a real non-lifting failure response instead of being floored into a
        # fake weak positive-thrust archive row.
        # dCT = ½σ(CL cosφ - CD sinφ) r² dr (normalised, V_tip = ΩR)
        CT = 0.5 * np.trapezoid(sigma * (CL_b * np.cos(phi) - cd0 * np.sin(phi)) * r_bar**2, r_bar)
        CT = float(CT)

        # Shaft-power coefficient.  The blade-element tangential-force integral
        # can undercut the actuator-disk ideal in this simplified finite-station
        # model.  Enforce a non-ideal induced-power floor for useful positive
        # thrust, then add profile power.  For CT <= 0, the compact hover
        # surrogate reports FM = 0 and retains only non-negative shaft power.
        CP_induced_sectional = 0.5 * np.trapezoid(sigma * CL_b * np.sin(phi) * r_bar**3, r_bar)
        CP_profile = 0.5 * np.trapezoid(sigma * cd0 * np.cos(phi) * r_bar**3, r_bar)
        CT_useful = max(CT, 0.0)
        CP_ideal_hover = CT_useful**1.5 / np.sqrt(2.0)
        CP_induced = max(CP_induced_sectional, self._INDUCED_POWER_FACTOR * CP_ideal_hover, 0.0)
        CP = CP_induced + CP_profile
        CP = max(CP, 1e-6)
        if not np.isfinite(CT) or not np.isfinite(CP) or CP <= 0.0:
            raise ValueError("non-physical rotor CT/CP response")

        # Figure of merit
        if CT <= 0.0:
            FM = 0.0
        else:
            FM = CT**1.5 / (np.sqrt(2.0) * CP)
        if not np.isfinite(FM) or FM < 0.0 or FM > 1.0:
            raise ValueError("non-physical rotor figure of merit")

        return np.array([CT, CP, FM])
