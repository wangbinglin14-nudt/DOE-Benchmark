"""
Case M01: Aircraft conceptual design (Raymer-style surrogate)
==============================================================

Engineering background
----------------------
During the conceptual-design phase of an aircraft, aerodynamic, weight,
and performance estimates must be produced rapidly with limited data to
screen the design space.  The methods of Raymer (2012) and Roskam (1997)
are the most widely used semi-empirical approaches in industry.

This case applies Raymer-style conceptual-design scalings to compute the
aerodynamic efficiency (L/D), structural efficiency (empty-weight fraction),
and range performance (Breguet range equation) of a subsonic civil transport.
It is a low-order benchmark surrogate for constant-cruise trend testing, not
a full aircraft sizing workflow or mission/trajectory fuel-burn analysis.
It does not close payload, take-off weight, wing loading, thrust loading,
constraint diagrams, climb/descent/loiter/diversion reserves, take-off/landing
field length, engine lapse, aeroelasticity, stability/control, certification,
cost, noise, or emissions constraints.  It also does not compute cruise lift
coefficient from dynamic pressure, wing area, and weight; does not model
compressibility drag rise, wave drag, drag-divergence Mach number, buffet, or
supercritical-airfoil design; and does not couple propulsion performance to
altitude or Mach.  It also does not run NASA-style aircraft synthesis,
mission-analysis, or optimisation software such as FLOPS/GASP/Aviary/OpenMDAO,
nor a parametric geometry, DegenGeom, potential-flow, or VSPAERO workflow such
as OpenVSP.
Current OpenMDAO Aviary documentation is used only as boundary evidence for
broader workflows: Aviary's core geometry, mass, aerodynamic, propulsion, and
mission subsystems, its conventional tube-and-wing empirical equations, its
phase_info/trajectory machinery, and its ability to connect external
subsystems or external tools for medium- and high-fidelity analysis are all
outside this constant-cruise algebraic benchmark.
The Aviary features/functionality page is likewise boundary evidence that
mission setup is phase-based and that minimum Aviary models include geometry,
mass, aerodynamics, propulsion, and mission subsystems; this benchmark does
not instantiate that coupled subsystem/mission machinery.
NASA FLOPS software-catalog material is likewise used only as boundary
evidence for a full aircraft synthesis system with weights, aerodynamics,
propulsion data, mission performance, takeoff/landing, and program-control
modules outside this benchmark.
NASA GASP and Aviary comparison-to-FLOPS material is used in the same boundary
role: GASP/FLOPS-style geometry, mass/weight, aerodynamics, propulsion,
mission, takeoff/landing, performance, economic, and optimization workflows are
not instantiated or calibrated by this compact algebraic case.

Physics model
-------------
1. Aerodynamic model (drag polar):
   CDmin estimated from wetted area and skin friction
   Induced drag: CDi = CL²/(π·AR·e)
   L/D_max = √(π·AR·e/(4·CDmin))

2. Structural weight estimation (Raymer-style low-order scaling):
   We/WTO (empty weight / take-off weight) as a simplified power law of
   design parameters.  This is a benchmark trend model, not a verbatim
   implementation of a Raymer/Roskam sizing table.

3. Breguet range equation (jet):
   R = (V/(g·SFC))·(L/D)·ln(W_start/W_end), where SFC is mass TSFC
   in kg/(N·s) after conversion from the input kg/(N·h)
   The model represents the constant-speed cruise segment and does not
   compute climb, descent, maneuver, wind/weather, alternate, or reserve
   mission segments beyond the fixed 5 % cruise-fuel deduction below.  Wing
   reference area and cruise-altitude density are retained as DOE stress-test
   inputs but are not used by the current constant-cruise algebraic surrogate.
   Mach number enters only through the assumed cruise speed, not through a
   cruise-trim, transonic drag-rise, or engine-lapse model.

Input variables
---------------
AR        Wing aspect ratio          [7, 14]      -       Key aerodynamic-efficiency parameter
SW        Reference wing area        [100, 400]   m²      Aircraft size
lambda_   Taper ratio                [0.2, 0.7]   -       Lift-distribution efficiency
tc        Thickness-to-chord ratio   [0.10, 0.18] -       Structure / aerodynamics trade-off
SFC       Thrust-specific fuel cons. [0.05, 0.08] kg/(N·h) Engine efficiency (modern turbofan)
W_frac    Fuel-weight fraction       [0.2, 0.45]  -       Range-determining factor
rho_alt   Cruise-altitude air density [0.3, 0.8]  kg/m³   Flight condition
Ma_cruise Cruise Mach number         [0.6, 0.9]   -       Speed–drag trade-off

Output responses
----------------
range_km  Maximum range            km     Primary output (commercial value)
L_D_max   Maximum lift-to-drag ratio -     Aerodynamic efficiency
We_frac   Empty-weight fraction      -     Structural efficiency indicator

DOE validation value
--------------------
VS-B (effect estimation):
    AR is the strongest main effect on L/D, while the stored range_km Sobol
    metadata is dominated by W_frac, SFC, and Ma_cruise through the Breguet
    product.  Geometry still enters range through L/D.

VS-E (high-dimensional adaptability):
    8-D problem; 6 of 8 variables contribute (SW and rho_alt are unused
    in this simplified model) — suitable for evaluating high-dimensional
    DOE methods.

References
----------
1. Raymer, D. P. (2012). Aircraft Design: A Conceptual Approach, 5th ed.
   AIAA Education Series.
2. Roskam, J. (1997). Airplane Design, 8 vols. DAR Corporation.
3. NASA Glenn. Range - Constant Velocity.
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-constant-velocity/
4. NASA Glenn. Range Summary.
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-summary/
5. MIT Unified Engineering. Breguet range equation.
   https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node98.html
6. NASA Glenn. Induced Drag Coefficient.
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/
7. NASA Glenn. Mach Number.
   https://www.grc.nasa.gov/www/BGH/mach.html
8. NASA Glenn. What is Drag?
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/what-is-drag/
9. NASA Software Catalog. Aviary.
   https://software.nasa.gov/software/LEW-20538-1
10. NASA. Aviary: A New NASA Software Platform for Aircraft Modelling.
    https://www.nasa.gov/aeronautics/aviary-software-overview/
11. OpenMDAO Aviary documentation.
    https://openmdao.github.io/Aviary/
12. NASA NTRS. Aviary: An Open-Source Multidisciplinary Design,
    Analysis, and Optimization Tool for Modeling Aircraft With Analytic
    Gradients.
    https://ntrs.nasa.gov/citations/20230015940
13. OpenMDAO Aviary external-subsystems documentation.
    https://openmdao.github.io/Aviary/user_guide/using_external_subsystems.html
14. OpenMDAO Aviary functionality overview and FAQ.
    https://openmdao.github.io/Aviary/theory_guide/intro.html
    https://openmdao.github.io/Aviary/user_guide/features_and_functionalities.html
    https://openmdao.github.io/Aviary/misc_resources/FAQ.html
15. NASA. OpenVSP VSPAERO Basics.
    https://www.nasa.gov/reference/openvsp-vspaero-basics/
16. NASA Software Catalog. Flight Optimization System (FLOPS) Software v.9.
    https://software.nasa.gov/software/LAR-18934-1
17. NASA NTRS. The Flight Optimization System Weights Estimation Method.
    https://ntrs.nasa.gov/citations/20170005851
18. NASA NTRS. GASP - General Aviation Synthesis Program, Volume 1:
    Main program, Part 1: Theoretical development.
    https://ntrs.nasa.gov/citations/19810010562
19. OpenMDAO Aviary, Comparison to FLOPS.
    https://openmdao.github.io/Aviary/misc_resources/comparison_to_flops.html

The public NASA/MIT anchors support only the Breguet range, range-scope,
Mach/drag, and induced-drag formula context used by this compact surrogate.
They are not external validation data for the Raymer-style empty-weight power
law, a full aircraft sizing workflow, transonic performance, cruise trim, or
mission fuel-burn modelling.  FLOPS/GASP/Aviary/OpenMDAO and OpenVSP/VSPAERO
references are boundary evidence for omitted full synthesis, mission,
takeoff/landing, propulsion, optimisation, parametric-geometry, DegenGeom, and
potential-flow aerodynamic-analysis workflows, pressure/wake-solution review,
custom command-line/input-setup workflows, and OpenMDAO-based mission/design/
optimization workflows, including Aviary phase_info, trajectory, and external
subsystem-builder workflows. NASA NTRS Aviary material further frames real
Aviary analyses as multidisciplinary design/analysis/optimization with
analytic gradients and tightly coupled aircraft/subsystem design, not
calibration evidence for this algebraic benchmark case. Current Aviary
documentation also states that its geometry, mass, and aerodynamic calculations
are empirical conventional tube-and-wing relationships, that mission analysis
uses trajectory phases, and that external tools/subsystems can be connected for
medium- and high-fidelity analysis; none of those workflows are run here.
The Aviary features/functionality page is used in the same boundary role: it
describes minimum subsystem coverage and mission-phase setup for Aviary models,
not validation evidence for this compact constant-cruise response surface.
NASA GASP documentation and the Aviary comparison-to-FLOPS page sharpen the
same omitted-workflow boundary: real GASP/FLOPS/Aviary analyses integrate
technical disciplines, parametric studies, empirical mass/geometry/aerodynamic
relations, mission and performance modules, takeoff/landing, engine decks, and
optimization/trajectory machinery, whereas this benchmark only evaluates the
declared Breguet/parabolic-drag-polar/empty-weight trend equations.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class AircraftConceptual8D(BaseCase):
    """Aircraft conceptual design (8D → 3D, Raymer-style low-order surrogate)."""

    meta = CaseMeta(
        case_id="aircraft_conceptual_8d",
        name="Aircraft conceptual design (Raymer-style low-order surrogate)",
        version="1.0.0",
        domain="multidisciplinary",
        tags=[
            "aircraft",
            "conceptual_design",
            "raymer",
            "aerodynamics",
            "weight",
            "range",
            "breguet",
            "8d",
        ],
        dim_input=8,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="AR",
                symbol="AR",
                description="Wing aspect ratio",
                unit="-",
                lower=7.0,
                upper=14.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="SW",
                symbol="S_W",
                description="Wing reference area",
                unit="m²",
                lower=100.0,
                upper=400.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="lam",
                symbol=r"\lambda",
                description="Wing taper ratio",
                unit="-",
                lower=0.2,
                upper=0.7,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="tc",
                symbol="t/c",
                description="Wing thickness-to-chord ratio",
                unit="-",
                lower=0.10,
                upper=0.18,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="SFC",
                symbol="SFC",
                description="Thrust-specific fuel consumption (modern turbofan 0.05–0.08)",
                unit="kg/(N·h)",
                lower=0.05,
                upper=0.08,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="W_frac",
                symbol="W_f/W_{TO}",
                description="Fuel-weight fraction",
                unit="-",
                lower=0.20,
                upper=0.45,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="rho_alt",
                symbol=r"\rho_{alt}",
                description="Cruise-altitude air density",
                unit="kg/m³",
                lower=0.3,
                upper=0.8,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="Ma_cruise",
                symbol="Ma_c",
                description="Cruise Mach number",
                unit="-",
                lower=0.60,
                upper=0.90,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="range_km",
                symbol="R",
                description="Breguet maximum range",
                unit="km",
                is_primary=True,
                expected_range=(2000.0, 27000.0),
            ),
            OutputVariable(
                name="L_D_max",
                symbol="(L/D)_{max}",
                description="Maximum lift-to-drag ratio",
                unit="-",
                is_primary=True,
                expected_range=(10.0, 25.0),
            ),
            OutputVariable(
                name="We_frac",
                symbol="W_e/W_{TO}",
                description="Empty-weight fraction (structural efficiency)",
                unit="-",
                is_primary=False,
                expected_range=(0.40, 0.65),
            ),
        ],
        governing_equation=(
            "Raymer-style conceptual-design surrogate:\n"
            "  CDmin = Cf·(Swet/Sref) (skin friction + form-drag correction)\n"
            "  L/D_max = √(π·AR·e/4/CDmin)\n"
            "  Breguet: R = (V_cruise/(g·SFC))·(L/D)·ln(1/(1−0.95·W_frac))\n"
            "  Empty weight: simplified Raymer-style power-law scaling"
        ),
        pde_type="algebraic",
        spatial_dim=0,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Raymer, D. P. (2012). Aircraft Design, 5th ed. AIAA Education Series.",
            "Roskam, J. (1997). Airplane Design. DAR Corporation.",
            "NASA Glenn, Range - Constant Velocity, https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-constant-velocity/",
            "NASA Glenn, Range Summary, https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-summary/",
            "MIT Unified Engineering, Breguet range equation, https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node98.html",
            "NASA Glenn, Induced Drag Coefficient, https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/",
            "NASA Glenn, Mach Number, https://www.grc.nasa.gov/www/BGH/mach.html",
            "NASA Glenn, What is Drag?, https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/what-is-drag/",
            "NASA Software Catalog, Aviary, https://software.nasa.gov/software/LEW-20538-1",
            "NASA Software Catalog, Flight Optimization System (FLOPS) Software v.9. "
            "Public boundary evidence that FLOPS is a multidisciplinary system "
            "for conceptual/preliminary design and evaluation of advanced "
            "aircraft concepts with weights, aerodynamics, propulsion data "
            "scaling/interpolation, mission performance, takeoff and landing, "
            "and program-control modules; not validation data for this "
            "constant-cruise algebraic surrogate. "
            "https://software.nasa.gov/software/LAR-18934-1",
            "NASA NTRS, The Flight Optimization System Weights Estimation Method. "
            "Public boundary evidence for NASA aircraft-synthesis weight "
            "estimation scope; not a source for calibrating the benchmark's "
            "simplified Raymer-style empty-weight power law. "
            "https://ntrs.nasa.gov/citations/20170005851",
            "NASA NTRS, GASP - General Aviation Synthesis Program, Volume 1: "
            "Main program, Part 1: Theoretical development. Public boundary "
            "evidence that GASP integrates preliminary-design technical "
            "disciplines, aircraft sizing, parametric studies, performance, "
            "economics, and configuration trade studies; not validation data "
            "for this compact Breguet/Raymer-style algebraic surrogate. "
            "https://ntrs.nasa.gov/citations/19810010562",
            "NASA, Aviary software overview. Public boundary evidence that "
            "Aviary creates virtual aircraft models from user-provided inputs, "
            "can link with other codes/tools, and publishes documentation via "
            "the OpenMDAO project site; not calibration evidence for this "
            "compact algebraic surrogate. "
            "https://www.nasa.gov/aeronautics/aviary-software-overview/",
            "OpenMDAO Aviary documentation. Public boundary evidence that "
            "Aviary is an aircraft analysis, design, and optimization tool "
            "built on top of OpenMDAO, with broader mission/vehicle coupling "
            "workflows outside this compact algebraic surrogate. "
            "https://openmdao.github.io/Aviary/",
            "NASA NTRS, Aviary MDAO/analytic-gradient paper. Public boundary "
            "evidence that Aviary supports multidisciplinary aircraft design, "
            "analysis, optimization, analytic gradients, and tightly coupled "
            "aircraft/subsystem design outside this compact constant-cruise "
            "surrogate. https://ntrs.nasa.gov/citations/20230015940",
            "OpenMDAO Aviary external-subsystems documentation. Public boundary "
            "evidence that Aviary can integrate modular OpenMDAO subsystems "
            "across aircraft trajectories; this benchmark does not instantiate "
            "that workflow. "
            "https://openmdao.github.io/Aviary/user_guide/using_external_subsystems.html",
            "OpenMDAO Aviary functionality overview. Public boundary evidence "
            "that Aviary core subsystems include empirical geometry, mass, "
            "aerodynamic, propulsion, and mission models for conventional "
            "tube-and-wing aircraft, and recommend external subsystems for "
            "novel configurations. "
            "https://openmdao.github.io/Aviary/theory_guide/intro.html",
            "OpenMDAO Aviary features and functionalities. Public boundary "
            "evidence that Aviary models need geometry, mass, aerodynamics, "
            "propulsion, and mission subsystems, and define missions through "
            "user-selected phase options; this benchmark does not instantiate "
            "that coupled subsystem/mission-phase machinery. "
            "https://openmdao.github.io/Aviary/user_guide/features_and_functionalities.html",
            "OpenMDAO Aviary FAQ. Public boundary evidence that external "
            "tools/subsystems such as NPSS, pyCycle, OpenAeroStruct, VSPAero, "
            "ANOPP, and electrical modeling can be connected to Aviary for "
            "medium- and high-fidelity analysis/optimization; not used here. "
            "https://openmdao.github.io/Aviary/misc_resources/FAQ.html",
            "OpenMDAO Aviary comparison to FLOPS. Public boundary evidence "
            "that Aviary's FLOPS/GASP-derived workflows include geometry, "
            "mass, empirical and tabular aerodynamics, external aerodynamic "
            "models, propulsion engine decks, mission equations of motion, "
            "reserve/off-design missions, takeoff/landing, and OpenMDAO "
            "optimization/analytic-derivative machinery outside this benchmark. "
            "https://openmdao.github.io/Aviary/misc_resources/comparison_to_flops.html",
            "NASA, OpenVSP VSPAERO Basics. Public boundary evidence that "
            "VSPAERO is an OpenVSP-based potential-flow aerodynamics tool using "
            "the DegenGeom thin-surface representation, with pressure/wake "
            "solution review and custom-input/command-line workflows outside "
            "this benchmark. https://www.nasa.gov/reference/openvsp-vspaero-basics/",
        ],
        solver_library="numpy (Raymer-style algebraic surrogate)",
        solver_details=(
            "Simplified power-law weight scaling + analytical parabolic-drag-polar/"
            "Breguet constant-cruise model; not a calibrated full aircraft sizing "
            "code, constraint-analysis workflow, payload/MTOW closure, or "
            "mission/trajectory fuel-burn model; named _validate_physical_inputs "
            "private physical-input evaluation rejects malformed arrays and "
            "AR/SW/lam/tc/SFC/W_frac/rho_alt/Ma_cruise values outside the "
            "declared benchmark domain before computing the Breguet/Raymer-style "
            "response; non-finite, non-physical, or out-of-contract "
            "range_km/L_D_max/We_frac responses are rejected before dispatch "
            "can archive a row; no cruise CL/dynamic-pressure "
            "trim closure with SW/rho_alt, no wing-loading or thrust-loading "
            "constraint diagram, no wind/weather mission correction, no "
            "compressibility drag rise, wave drag, "
            "drag-divergence Mach number, buffet, supercritical-airfoil design, "
            "altitude/Mach-dependent engine lapse, FLOPS/GASP/Aviary/OpenMDAO "
            "aircraft synthesis, mission analysis, or optimisation, or "
            "FLOPS weights/aerodynamics/propulsion-data/mission/takeoff-landing/"
            "program-control module workflow, GASP integrated preliminary-design/"
            "parametric-study/performance/economics/configuration-trade workflow, "
            "OpenVSP/VSPAERO-style parametric geometry, DegenGeom, "
            "potential-flow, pressure/wake-solution review, custom command-line "
            "input setup, and aerodynamic-analysis workflow; no Aviary "
            "phase_info, trajectory, mission-phase, external-subsystem, or "
            "subsystem-builder workflow; no OpenMDAO/Aviary analytic-gradient "
            "MDAO, tightly coupled aircraft/subsystem design, modular subsystem "
            "integration across trajectories, empirical Aviary geometry/mass/"
            "aerodynamic/propulsion/mission core-subsystem workflow, "
            "Aviary five-subsystem model and mission-phase setup workflow, "
            "conventional tube-and-wing empirical-equation workflow, novel-"
            "configuration external-subsystem workflow, or external-tool "
            "coupling to NPSS, pyCycle, OpenAeroStruct, VSPAero, ANOPP, or "
            "electrical modeling, and no Aviary comparison-to-FLOPS workflow "
            "covering empirical/tabular aerodynamics, external aerodynamic "
            "models, engine decks, equations of motion, reserve/off-design "
            "missions, takeoff/landing, or OpenMDAO optimization"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.1},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_monotone", "strong_interaction"],
        effective_dim=6,
        important_variables=["AR", "W_frac", "SFC", "Ma_cruise", "tc", "lam"],
        interaction_pairs=[
            ("AR", "lam"),
            ("AR", "tc"),
            ("AR", "SFC"),
            ("AR", "W_frac"),
            ("AR", "Ma_cruise"),
            ("lam", "SFC"),
            ("lam", "W_frac"),
            ("lam", "Ma_cruise"),
            ("tc", "SFC"),
            ("tc", "W_frac"),
            ("tc", "Ma_cruise"),
            ("SFC", "W_frac"),
            ("SFC", "Ma_cruise"),
            ("W_frac", "Ma_cruise"),
        ],
        has_failure_boundary=False,
        sobol_S1={
            "AR": 0.057,
            "SW": 0.000,
            "lam": 0.004,
            "tc": 0.009,
            "SFC": 0.161,
            "W_frac": 0.620,
            "rho_alt": 0.000,
            "Ma_cruise": 0.117,
        },
        sobol_ST={
            "AR": 0.063,
            "SW": 0.000,
            "lam": 0.004,
            "tc": 0.010,
            "SFC": 0.174,
            "W_frac": 0.648,
            "rho_alt": 0.000,
            "Ma_cruise": 0.128,
        },
        sobol_computation="salib_saltelli_n2048_seed42_range_km_output",
        applicable_scenarios=["VS-B", "VS-E"],
        scenario_notes={
            "VS-B": (
                "AR is the strongest main effect on L/D. For the stored range_km "
                "Sobol metadata, W_frac dominates, followed by SFC and Ma_cruise; "
                "geometry enters range through L/D."
            ),
            "VS-E": "8-D multidisciplinary problem; 6 of 8 variables contribute (SW and rho_alt unused) — high-dimensional method test.",
        },
        recommended_n_train={"default": 80},
        n_saturation=400,
        noise_factor_candidates=["SFC", "rho_alt"],
        control_factor_candidates=["AR", "SW", "lam", "tc", "W_frac", "Ma_cruise"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "aircraft_conceptual_8d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        AR = X_phys[:, 0]
        SW = X_phys[:, 1]
        lam = X_phys[:, 2]
        tc = X_phys[:, 3]
        SFC = X_phys[:, 4]
        W_frac = X_phys[:, 5]
        rho_alt = X_phys[:, 6]
        Ma_cruise = X_phys[:, 7]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(AR < 7.0 - tol)
            or np.any(AR > 14.0 + tol)
            or np.any(SW < 100.0 - tol)
            or np.any(SW > 400.0 + tol)
            or np.any(lam < 0.2 - tol)
            or np.any(lam > 0.7 + tol)
            or np.any(tc < 0.10 - tol)
            or np.any(tc > 0.18 + tol)
            or np.any(SFC < 0.05 - tol)
            or np.any(SFC > 0.08 + tol)
            or np.any(W_frac < 0.20 - tol)
            or np.any(W_frac > 0.45 + tol)
            or np.any(rho_alt < 0.3 - tol)
            or np.any(rho_alt > 0.8 + tol)
            or np.any(Ma_cruise < 0.60 - tol)
            or np.any(Ma_cruise > 0.90 + tol)
        ):
            raise ValueError(
                "aircraft_conceptual_8d physical inputs must stay inside the "
                "declared AR/SW/lam/tc/SFC/W_frac/rho_alt/Ma_cruise benchmark "
                "domain before Breguet/Raymer-style response assembly"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                y_i = self._design(
                    AR=float(X_phys[i, 0]),
                    SW=float(X_phys[i, 1]),
                    lam=float(X_phys[i, 2]),
                    tc=float(X_phys[i, 3]),
                    SFC=float(X_phys[i, 4]),
                    W_frac=float(X_phys[i, 5]),
                    rho_alt=float(X_phys[i, 6]),
                    Ma_cruise=float(X_phys[i, 7]),
                )
                if y_i.shape != (3,) or not np.all(np.isfinite(y_i)):
                    raise FloatingPointError(
                        "non-finite or malformed conceptual-design response"
                    )
                range_km, l_d_max, we_frac = y_i
                if range_km <= 0.0 or l_d_max <= 0.0 or not (0.0 < we_frac < 1.0):
                    raise FloatingPointError(
                        "non-physical conceptual-design response"
                    )
                output_ranges = np.asarray(
                    [output.expected_range for output in self.meta.output_variables],
                    dtype=float,
                )
                below = y_i < (output_ranges[:, 0] - 1e-10)
                above = y_i > (output_ranges[:, 1] + 1e-10)
                if np.any(below) or np.any(above):
                    raise FloatingPointError(
                        "out-of-contract conceptual-design response"
                    )
                Y[i] = y_i
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "aircraft_conceptual_8d Breguet/Raymer-style surrogate "
                    f"calculation failed for physical input {x_text}; "
                    "refusing silent NaN or non-physical output because the "
                    "formal benchmark response surface requires finite positive "
                    "range, positive L/D, and empty-weight-fraction values "
                    "strictly between 0 and 1 and inside the declared output "
                    "ranges."
                ) from None
        return Y

    @staticmethod
    def _design(AR, SW, lam, tc, SFC, W_frac, rho_alt, Ma_cruise):
        """Raymer-style low-order subsonic transport surrogate.

        Units:
          SFC input [kg/(N·h)] → converted to [kg/(N·s)] = SFC/3600 (for Breguet)
          V_cruise = Ma × a_cruise; at ~11 km cruise altitude a ≈ 295–300 m/s
        """
        if (
            not np.all(np.isfinite([AR, SW, lam, tc, SFC, W_frac, rho_alt, Ma_cruise]))
            or not (7.0 <= AR <= 14.0)
            or not (100.0 <= SW <= 400.0)
            or not (0.2 <= lam <= 0.7)
            or not (0.10 <= tc <= 0.18)
            or not (0.05 <= SFC <= 0.08)
            or not (0.20 <= W_frac <= 0.45)
            or not (0.3 <= rho_alt <= 0.8)
            or not (0.60 <= Ma_cruise <= 0.90)
        ):
            raise ValueError(
                "invalid aircraft_conceptual_8d input outside the declared "
                "Breguet/Raymer-style surrogate contract"
            )
        g = 9.80665

        # Cruise speed of sound (high-altitude ISA ~11 km: T = 216.65 K)
        a_cruise = 295.1  # np.sqrt(1.4 * 287 * 216.65)
        V_cruise = Ma_cruise * a_cruise

        # ── Aerodynamics: zero-lift drag + induced drag ─────────────────────
        # Wetted-area ratio (Raymer Ch. 12, subsonic jet empirical value 5–6)
        Swet_Sref = 5.5
        # Skin friction (turbulent flat plate, Re ~ 5e7)
        Cf = 0.0025
        CDmin = Cf * Swet_Sref * (1.0 + tc * 1.5 + 100.0 * tc**4)

        # Oswald efficiency (taper ratio + thickness correction)
        e = 1.0 / (1.0 + 0.03 * AR * (1.0 - lam) ** 2 + 0.01 * AR * tc / 0.12)
        e = float(e)
        if not np.isfinite(e) or not (0.0 < e < 1.0):
            raise ValueError("non-physical aircraft Oswald efficiency factor")

        # Maximum L/D: L/D_max = 0.5·√(π·AR·e / CDmin)
        L_D_max = 0.5 * np.sqrt(np.pi * AR * e / CDmin)
        if not np.isfinite(L_D_max) or L_D_max <= 0.0:
            raise ValueError("non-physical aircraft lift-to-drag response")

        # ── Empty-weight fraction: Raymer-style low-order trend surrogate ──────
        We_frac = 0.55 * (Ma_cruise**0.05) * (AR ** (-0.03)) * ((tc / 0.14) ** 0.08)
        We_frac = float(We_frac)
        if not np.isfinite(We_frac) or not (0.0 < We_frac < 1.0):
            raise ValueError("non-physical aircraft empty-weight fraction")

        # ── Breguet range: R = (V/(g·SFC))·(L/D)·ln(W_i/W_f) ─────────────
        # Deduct 5 % reserve fuel
        W_fuel_eff = W_frac * 0.95
        We_eff = 1.0 - W_fuel_eff  # Final weight / initial weight
        SFC_SI = SFC / 3600.0  # kg/(N·s)

        R_breguet = (V_cruise / (g * SFC_SI)) * L_D_max * np.log(1.0 / We_eff)
        range_km = R_breguet / 1000.0
        if not np.isfinite(range_km) or range_km <= 0.0:
            raise ValueError("non-physical aircraft Breguet range response")

        return np.array([range_km, L_D_max, We_frac])
