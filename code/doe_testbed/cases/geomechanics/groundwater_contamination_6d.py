"""
Case G02: Groundwater contaminant transport (advection–dispersion equation)
============================================================================

Engineering background
----------------------
Groundwater contamination (chemical spills, landfill leachate, underground
nuclear-waste storage) is a major environmental engineering problem.
Contaminant migration through porous media is controlled by:
  · Advection (transport with groundwater flow)
  · Dispersion (hydrodynamic dispersion)
  · Adsorption (solid–liquid partitioning)
  · Degradation (biochemical / radioactive decay)

This case uses an Ogata-Banks-style 1-D advection-dispersion surrogate with
linear retardation and separable first-order decay attenuation, computing the
concentration profile at a given time and the contaminant-front arrival time.

The implementation is a compact screening surrogate. It does not model 2-D/3-D
flow, heterogeneous aquifers, transient groundwater flow, source depletion,
pulsed or finite-duration releases, site boundary conditions, wells or
pumping/injection stresses, MODFLOW-coupled flow, spatially variable hydraulic
gradients, recharge/evapotranspiration, variable water table, transverse or
vertical dispersion tensors, nonlinear or nonequilibrium sorption, dual-domain
mass transfer, multispecies reaction networks, density-dependent flow,
biodegradation kinetics beyond a first-order half-life, geochemical speciation,
uncertainty in source strength, numerical mesh/time-step convergence, or
calibration against field plume data or a full reactive-transport simulator.
Current USGS groundwater-vulnerability guidance also makes clear that
transport assessment depends on system boundaries, flow-pattern uncertainty,
and pumping stresses; this benchmark intentionally freezes that broader
setting to a one-dimensional constant-average-velocity surrogate with fixed
source conditions. Current USGS principal-aquifer quality guidance further
states that irrigation, pumping, artificial recharge, and drainage can alter
groundwater flow and groundwater quality, so this compact case must not be read
as a site-specific vulnerability or remediation model.
Official USGS MODFLOW 6 GWT documentation is used only as omitted-scope
context for three-dimensional, flow-coupled, finite-difference transport with
advection/dispersion, sorption, immobile-domain transfer, decay/production, and
source/sink mixing; the official documentation also includes advanced-package
transport through streams, lakes, multi-aquifer wells, the unsaturated zone,
and Water Mover-connected packages. This benchmark does not run MODFLOW 6.
USGS PHREEQC documentation is used only as a formula-context anchor for the
advection-reaction-dispersion equation and the role of hydrodynamic dispersion;
this benchmark does not run PHREEQC or model geochemical speciation.
The official USGS MT3D-USGS software page is likewise used only to define the
boundary of a full MODFLOW-linked solute-transport simulator, including
unsaturated-zone transport, stream/lake exchange, interspecies/parent-daughter
reactions, pump-and-treat recirculation, transient mass-storage accounting, and
separate mobile/immobile-domain Kd specification; the official software page
also states MODFLOW 6 flow support for structured grids only.
The official MODFLOW 6 overview is likewise broader than the frozen benchmark:
it frames MODFLOW 6 as a modular groundwater-simulation platform with coupled
GWF/GWT packages whose flow and transport models can operate simultaneously,
including unstructured-grid and advanced-package workflows, while the benchmark
remains a one-dimensional constant-velocity transport surrogate and does not
reproduce package-level flow/transport coupling, simultaneous GWF/GWT
operation, unstructured-grid transport support, mover routing, or advanced
boundary-package workflows.
Current USGS MODFLOW 6 overview material also documents a broader simulation
platform than this benchmark, including groundwater flow, groundwater
transport, groundwater energy, and particle-tracking model types plus
hydrologic stress packages such as wells, recharge, rivers, drains, lakes,
multi-aquifer wells, evapotranspiration, storage, and mover workflows. Those
platform features are boundary evidence only; this case does not run GWF, GWT,
GWE, PRT, stress-package routing, particle tracking, or heat/energy transport.
Current EPA Superfund groundwater-remedy and technology pages are used only as
remediation-workflow boundary evidence. They describe remedy selection,
exposure-pathway protection, plume containment, pump-and-treat, in situ
treatment, vertical engineered barriers, monitored natural attenuation,
institutional controls, alternative water supply, and capture-zone analysis.
This benchmark does not choose remedies, evaluate exposure pathways, simulate
extraction/injection wells, assess hydraulic containment, or validate monitored
natural attenuation or cleanup performance.
The concentration formula is smooth for fixed coefficients, but the benchmark
front metrics are threshold-derived: ``x_front`` is selected from a finite
spatial grid and ``t_arrive_10m`` from a finite logarithmic time scan, so the
multi-output response includes threshold/discrete jumps.
The reflection term is evaluated with the mathematically equivalent
``erfcx(A2) * exp(v*x/D - A2**2)`` form so legal low-dispersion points do not
depend on an arbitrary ``exp(v*x/D)`` overflow cap.

Governing equation (1-D ADE)
----------------------------
  R·∂C/∂t = D·∂²C/∂x² − v·∂C/∂x − λ·C

where:
  R = 1 + (ρ_b·K_d)/(1000·n)  (K_d in L/kg; R > 1 means adsorption retards transport)
  D = α_L·v + D_m·τ     (hydrodynamic dispersion coefficient)
  v = K_h·i/n            (Darcy velocity / porosity = true pore velocity)
  λ = ln(2)/t_half        (first-order decay rate)

Input variables
---------------
v_pore      True pore velocity     [0.001, 1.0]  m/d     Set by hydraulic gradient
alpha_L     Longitudinal dispersivity [0.01, 5.0] m       Scale-dependent
K_d         Partition coefficient   [0.0, 10.0]   L/kg    Adsorption strength
rho_b       Dry bulk density        [1400, 1900]  kg/m³   Soil density
t_half      Degradation half-life   [10, 1000]    d       Contaminant decay rate
n_por       Porosity                [0.2, 0.5]    -       Medium parameter

Output responses
----------------
C_max_frac  Max relative concentration after 1 yr (C/C0)  -   Contamination-level indicator
x_front     Front position after 1 yr (C/C0 = 0.01)       m   Pollution spread extent
t_arrive    Front arrival time at x = 10 m                 d   Early-warning time

DOE validation value
--------------------
VS-A (space filling):
    The plume-front metrics vary dramatically across 3 orders of magnitude of
    v_pore (0.001 → 1 m/d); the advantage of log-scale sampling is most
    apparent in x_front and t_arrive.  C_max_frac is a source-including
    profile maximum and is therefore dominated by decay/retardation in this
    fixed-boundary surrogate.

VS-G (sequential / reliability):
    Find the parameter combinations for which the front exceeds x = 10 m
    within 1 year (high-risk region).

References
----------
1. Ogata, A. & Banks, R. B. (1961). A Solution of the Differential Equation
   of Longitudinal Dispersion in Porous Media. USGS PP-411-A.
   https://pubs.usgs.gov/pp/0411a/report.pdf
2. Domenico, P. A. & Schwartz, F. W. (1997). Physical and Chemical Hydrogeology,
   2nd ed. Wiley.
3. USGS Circular 1224. Concepts of groundwater flow and Darcy velocity.
   https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html
4. USGS. MT3D-USGS groundwater solute-transport model documentation.
   https://pubs.usgs.gov/tm/06/a53/tm6a53.pdf
5. USGS. MT3D-USGS: Groundwater Solute Transport Simulator for MODFLOW.
   https://www.usgs.gov/software/mt3d-usgs-groundwater-solute-transport-simulator-modflow
6. USGS. MT3DMS: Model use, calibration, and validation.
   https://www.usgs.gov/publications/mt3dms-model-use-calibration-and-validation
7. USGS Techniques and Methods 6-A61. Documentation for the MODFLOW 6
   Groundwater Transport Model.
   https://pubs.usgs.gov/publication/tm6A61
8. USGS PHREEQC Version 1 manual, advection-reaction-dispersion equation.
   https://wwwbrr.cr.usgs.gov/projects/GWC_coupled/phreeqc.v1/html/phqc_69.htm
9. USGS. MODFLOW 6: USGS Modular Hydrologic Model.
   https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model
10. USGS. Groundwater Quality in Principal Aquifers of the United States, 1991-2010.
   https://www.usgs.gov/programs/environmental-health-program/science/groundwater-quality-principal-aquifers-united-states
11. USGS MODFLOW 6 Description of Input and Output, advanced stress-package
    and model-type boundary context.
    https://modflow6.readthedocs.io/en/latest/_mf6io/index.html
12. EPA. How Superfund Addresses Groundwater Contamination.
    https://www.epa.gov/superfund/how-superfund-addresses-groundwater-contamination
13. EPA. Groundwater Technologies.
    https://www.epa.gov/superfund/groundwater-technologies
14. EPA. Selecting a Groundwater Remedy.
    https://www.epa.gov/superfund/selecting-groundwater-remedy
15. EPA. A Systematic Approach for Evaluation of Capture Zones at Pump and
    Treat Systems.
    https://cfpub.epa.gov/si/si_public_record_report.cfm?Lab=NRMRL&dirEntryId=187788
16. EPA. Ground Water Modeling Research.
    https://www.epa.gov/land-research/ground-water-modeling-research
17. EPA. Remediation Technologies for Cleaning Up Contaminated Sites.
    https://www.epa.gov/remedytech/remediation-technologies-cleaning-contaminated-sites
"""

import numpy as np
from scipy.special import erfc, erfcx

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class GroundwaterContamination6D(BaseCase):
    """Groundwater contaminant transport (Ogata–Banks-style ADE surrogate), 6D → 3D."""

    meta = CaseMeta(
        case_id="groundwater_contamination_6d",
        name="Groundwater contaminant transport (1-D ADE, Ogata–Banks-style surrogate)",
        version="1.0.0",
        domain="geomechanics",
        tags=[
            "groundwater",
            "contaminant_transport",
            "ade",
            "dispersion",
            "adsorption",
            "6d",
            "analytical",
        ],
        dim_input=6,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="v_pore",
                symbol="v",
                description="True pore velocity (hydraulic gradient / porosity)",
                unit="m/d",
                lower=0.001,
                upper=1.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="alpha_L",
                symbol=r"\alpha_L",
                description="Longitudinal dispersivity",
                unit="m",
                lower=0.01,
                upper=5.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="K_d",
                symbol="K_d",
                description="Linear partition coefficient (adsorption)",
                unit="L/kg",
                lower=0.0,
                upper=10.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="rho_b",
                symbol=r"\rho_b",
                description="Soil dry bulk density",
                unit="kg/m³",
                lower=1400.0,
                upper=1900.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="t_half",
                symbol="t_{1/2}",
                description="Contaminant degradation half-life",
                unit="d",
                lower=10.0,
                upper=1000.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="n_por",
                symbol="n",
                description="Soil porosity",
                unit="-",
                lower=0.2,
                upper=0.5,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="C_max_frac",
                symbol="C_{max}/C_0",
                description="Maximum relative concentration after 1 year (source-including profile maximum)",
                unit="-",
                is_primary=True,
                expected_range=(0.0, 1.0),
            ),
            OutputVariable(
                name="x_front",
                symbol="x_{front}",
                description="Front position after 1 year (at C/C0 = 0.01)",
                unit="m",
                is_primary=False,
                expected_range=(0.0, 500.0),
                failure_threshold=10.0,
                failure_direction="above",
            ),
            OutputVariable(
                name="t_arrive_10m",
                symbol="t_{10m}",
                description="Front arrival time at x = 10 m",
                unit="d",
                is_primary=False,
                expected_range=(1.0, 10000.0),
            ),
        ],
        governing_equation=(
            "1-D ADE (advection–dispersion–reaction):\n"
            "  R·∂C/∂t = D·∂²C/∂x² − v·∂C/∂x − λ·C\n"
            "  Ogata–Banks-style erfc solution with adsorption R and separable decay attenuation\n"
            "  D = α_L·v + D_m·τ, R = 1 + ρ_b·K_d/(1000·n) for K_d in L/kg"
        ),
        pde_type="parabolic",
        spatial_dim=1,
        is_steady=False,
        # The ADE is linear in concentration for fixed transport coefficients,
        # but this benchmark's response map is nonlinear in the design inputs
        # through erfc terms, R(K_d, rho_b, n), D(alpha_L, v), and lambda(t_half).
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Ogata, A. & Banks, R. B. (1961). USGS PP-411-A. https://pubs.usgs.gov/pp/0411a/report.pdf",
            "USGS Circular 1224. Ground-water flow, Darcy's law, and average linear velocity. https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html",
            "USGS Techniques and Methods 6-A53. MT3D-USGS groundwater solute-transport "
            "model documentation. https://pubs.usgs.gov/tm/06/a53/tm6a53.pdf",
            "USGS. MT3D-USGS: Groundwater Solute Transport Simulator for MODFLOW. "
            "https://www.usgs.gov/software/mt3d-usgs-groundwater-solute-transport-simulator-modflow",
            "USGS. MT3DMS: Model use, calibration, and validation. "
            "https://www.usgs.gov/publications/mt3dms-model-use-calibration-and-validation",
            "USGS Techniques and Methods 6-A61. Documentation for the MODFLOW 6 Groundwater Transport Model. "
            "https://pubs.usgs.gov/publication/tm6A61",
            "USGS PHREEQC Version 1 manual, advection-reaction-dispersion equation "
            "and longitudinal hydrodynamic dispersion coefficient context only; "
            "not PHREEQC-equivalent reactive-transport chemistry or simulator validation. "
            "https://wwwbrr.cr.usgs.gov/projects/GWC_coupled/phreeqc.v1/html/phqc_69.htm",
            "USGS. MODFLOW 6: USGS Modular Hydrologic Model. "
            "Modular coupled groundwater-flow/transport platform context only; "
            "not validation data for this one-dimensional surrogate. "
            "https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model",
            "USGS MODFLOW 6 Description of Input and Output. Boundary evidence "
            "for omitted GWF/GWT/GWE/PRT model types and stress packages such "
            "as wells, recharge, rivers, drains, lakes, multi-aquifer wells, "
            "evapotranspiration, storage, mover routing, particle tracking, "
            "and heat/energy transport. "
            "https://modflow6.readthedocs.io/en/latest/_mf6io/index.html",
            "USGS. Groundwater Quality in Principal Aquifers of the United States, 1991-2010. "
            "Irrigation, pumping, artificial recharge, and drainage can alter groundwater flow and water quality; "
            "used here only as site-vulnerability / pumping-boundary context. "
            "https://www.usgs.gov/programs/environmental-health-program/science/groundwater-quality-principal-aquifers-united-states",
            "EPA. How Superfund Addresses Groundwater Contamination. Boundary evidence "
            "for omitted Superfund exposure-pathway protection, plume containment, "
            "pump-and-treat, in situ treatment, vertical engineered barriers, "
            "monitored natural attenuation, institutional controls, and alternative "
            "water-supply remedy workflows; not validation data for this analytical "
            "transport surrogate. "
            "https://www.epa.gov/superfund/how-superfund-addresses-groundwater-contamination",
            "EPA. Groundwater Technologies. Boundary evidence for omitted "
            "monitored-natural-attenuation evaluation, active/less-active treatment "
            "selection, site-characterization, cleanup-goal, and remedy-performance "
            "guidance workflows. https://www.epa.gov/superfund/groundwater-technologies",
            "EPA. Selecting a Groundwater Remedy. Boundary evidence for omitted "
            "Superfund remedy-selection and decision-document workflows, including "
            "MNA policy and groundwater cleanup decision guidance. "
            "https://www.epa.gov/superfund/selecting-groundwater-remedy",
            "EPA. A Systematic Approach for Evaluation of Capture Zones at Pump and "
            "Treat Systems. Boundary evidence for omitted three-dimensional hydraulic "
            "containment, extraction-well/drain capture-zone, converging-lines-of-"
            "evidence, field-data, and groundwater-modeling analyses. "
            "https://cfpub.epa.gov/si/si_public_record_report.cfm?Lab=NRMRL&dirEntryId=187788",
            "EPA. Ground Water Modeling Research. Boundary evidence for omitted "
            "contaminated-groundwater characterization, monitoring, treatment, "
            "source/plume remediation models, and site-support modeling tools. "
            "https://www.epa.gov/land-research/ground-water-modeling-research",
            "EPA. Remediation Technologies for Cleaning Up Contaminated Sites. "
            "Boundary evidence for omitted contaminated-site technology guides, "
            "including monitored natural attenuation, pump and treat, permeable "
            "reactive barriers, in situ chemical oxidation/reduction, in situ "
            "thermal treatment, phytotechnologies, soil vapor extraction/air "
            "sparging, solidification/stabilization, vapor intrusion mitigation, "
            "and vertical engineered barriers. "
            "https://www.epa.gov/remedytech/remediation-technologies-cleaning-contaminated-sites",
            "Domenico, P. A. & Schwartz, F. W. (1997). Physical and Chemical Hydrogeology, 2nd ed. Wiley.",
        ],
        solver_library="scipy.special (Ogata–Banks-style analytical surrogate)",
        solver_details=(
            "Ogata-Banks erfc solution with linear-retardation and separable "
            "first-order-decay attenuation; reflection term evaluated with "
            "the stable erfcx identity rather than an arbitrary exp(v*x/D) "
            "cap; analytical low-order transport "
            "surrogate, not the exact decaying constant-source ADE solution and "
            "not a calibrated site-specific reactive-transport model; no 2-D/3-D "
            "flow, heterogeneous aquifer, transient groundwater flow, source "
            "depletion, finite-duration release, site-specific boundary condition, "
            "well/pumping stress, MODFLOW-coupled flow, spatially variable hydraulic "
            "gradient, recharge/evapotranspiration, variable water table, transverse "
            "or vertical dispersion tensor, nonlinear or nonequilibrium sorption, "
            "dual-domain mass transfer, immobile-domain storage, matrix diffusion, "
            "multispecies reaction network, "
            "density-dependent flow, biodegradation kinetics beyond first-order "
            "half-life, geochemical speciation, source-strength uncertainty, "
            "numerical mesh/time-step convergence, field-plume calibration, "
            "PHREEQC-equivalent reactive-transport chemistry, "
            "unsaturated-zone transport, stream/lake solute exchange, "
            "interspecies or parent-daughter reaction packages, pump-and-treat "
            "recirculation, transient mass-storage budgets, separate "
            "mobile/immobile-domain Kd specification, advanced stress-package "
            "transport, multi-aquifer-well transport, Water Mover-connected "
            "package transport, source/sink mixing, direct solute mass loading, or "
            "MT3DMS/MT3D-USGS/MODFLOW 6 GWT package-level transport simulation; "
            "the official MODFLOW 6 platform overview likewise frames a broader "
            "modular GWF/GWT package ecosystem than this frozen one-dimensional "
            "constant-velocity surrogate implements, including simultaneous "
            "GWF/GWT operation, unstructured-grid transport support, and "
            "advanced-package coupling; current MODFLOW 6 input/output "
            "documentation also exposes GWF/GWT/GWE/PRT model types plus "
            "well/recharge/river/drain/lake/multi-aquifer-well/"
            "evapotranspiration/storage stress packages, mover routing, "
            "particle tracking, and heat/energy transport that this benchmark "
            "does not run; "
            "current USGS groundwater-vulnerability guidance likewise treats "
            "system boundaries, flow-pattern uncertainty, and pumping stresses "
            "as transport controls that are intentionally frozen out of this "
            "one-dimensional constant-average-velocity surrogate; current USGS "
            "principal-aquifer quality guidance likewise notes that irrigation, "
            "pumping, artificial recharge, and drainage can alter groundwater "
            "flow and water quality, which is outside this frozen transport setting; "
            "current EPA Superfund groundwater-remedy and technology guidance likewise "
            "describes exposure-pathway protection, plume containment, pump-and-treat, "
            "in situ treatment, vertical engineered barriers, monitored natural "
            "attenuation, institutional controls, alternative water supply, remedy "
            "selection, capture-zone analysis, converging lines of evidence, and "
            "groundwater characterization/monitoring/treatment workflows; EPA "
            "remedy-technology guides likewise cover monitored natural attenuation, "
            "pump and treat, permeable reactive barriers, in situ chemical "
            "oxidation/reduction, in situ thermal treatment, phytotechnologies, "
            "soil vapor extraction/air sparging, solidification/stabilization, "
            "vapor intrusion mitigation, and vertical engineered barriers that "
            "this benchmark does not implement; "
            "named _validate_physical_inputs private physical-input evaluation "
            "rejects malformed arrays and "
            "v_pore/alpha_L/K_d/rho_b/t_half/n_por values outside the declared "
            "benchmark domain before Ogata-Banks surrogate evaluation; "
            "non-finite or out-of-contract C_max_frac/x_front/t_arrive_10m "
            "responses are rejected before dispatch can archive a row"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 1.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=[
            "smooth_monotone",
            "strong_interaction",
            "sparse_effects",
            "heterogeneous",
            "threshold_jump",
            "discontinuous",
        ],
        effective_dim=6,
        important_variables=["v_pore", "alpha_L", "K_d", "rho_b", "t_half", "n_por"],
        interaction_pairs=[
            ("v_pore", "alpha_L"),
            ("v_pore", "K_d"),
            ("v_pore", "t_half"),
            ("v_pore", "n_por"),
            ("alpha_L", "K_d"),
            ("alpha_L", "t_half"),
            ("K_d", "rho_b"),
            ("K_d", "n_por"),
            ("K_d", "t_half"),
            ("rho_b", "n_por"),
            ("rho_b", "t_half"),
            ("t_half", "n_por"),
        ],
        has_failure_boundary=True,
        failure_condition="x_front > 10 m after 1 year (1% concentration front reaches the monitoring point)",
        sobol_S1={
            "v_pore": 0.000,
            "alpha_L": 0.000,
            "K_d": 0.312,
            "rho_b": 0.001,
            "t_half": 0.581,
            "n_por": 0.023,
        },
        sobol_ST={
            "v_pore": 0.000,
            "alpha_L": 0.000,
            "K_d": 0.402,
            "rho_b": 0.003,
            "t_half": 0.691,
            "n_por": 0.028,
        },
        sobol_computation="salib_saltelli_n2048_seed42_C_max_frac_output",
        applicable_scenarios=["VS-A", "VS-C", "VS-G"],
        scenario_notes={
            "VS-A": (
                "v_pore spans 3 orders of magnitude; log-scale sampling advantage "
                "is pronounced for x_front and t_arrive. C_max_frac is a "
                "source-including maximum and is dominated by decay/retardation."
            ),
            "VS-C": (
                "Surrogate-accuracy stress case for an Ogata-Banks-style erfc "
                "front with linear retardation and separable decay attenuation; "
                "appropriate for DOE method stress testing but not a calibrated "
                "site-specific reactive-transport, MT3DMS-style multispecies transport, "
                "or exact decaying constant-source ADE validation."
            ),
            "VS-G": "Find the high-risk parameter domain where x = 10 m is exceeded within 1 year.",
        },
        recommended_n_train={"default": 60},
        n_saturation=300,
        noise_factor_candidates=["K_d", "rho_b", "n_por"],
        control_factor_candidates=["v_pore", "alpha_L", "t_half"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "groundwater_contamination_6d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        v_pore = X_phys[:, 0]
        alpha_L = X_phys[:, 1]
        K_d = X_phys[:, 2]
        rho_b = X_phys[:, 3]
        t_half = X_phys[:, 4]
        n_por = X_phys[:, 5]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(v_pore < 0.001 * (1.0 - tol))
            or np.any(v_pore > 1.0 * (1.0 + tol))
            or np.any(alpha_L < 0.01 * (1.0 - tol))
            or np.any(alpha_L > 5.0 * (1.0 + tol))
            or np.any(K_d < 0.0 - tol)
            or np.any(K_d > 10.0 + tol)
            or np.any(rho_b < 1400.0 - tol)
            or np.any(rho_b > 1900.0 + tol)
            or np.any(t_half < 10.0 * (1.0 - tol))
            or np.any(t_half > 1000.0 * (1.0 + tol))
            or np.any(n_por < 0.2 - tol)
            or np.any(n_por > 0.5 + tol)
        ):
            raise ValueError(
                "groundwater_contamination_6d physical inputs must stay inside the declared "
                "v_pore/alpha_L/K_d/rho_b/t_half/n_por benchmark domain before "
                "Ogata-Banks surrogate evaluation"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                Y[i] = self._ogata_banks(
                    v=float(X_phys[i, 0]),
                    alpha_L=float(X_phys[i, 1]),
                    K_d=float(X_phys[i, 2]),
                    rho_b=float(X_phys[i, 3]),
                    t_half=float(X_phys[i, 4]),
                    n_por=float(X_phys[i, 5]),
                )
                c_max, x_front, t_arrive = Y[i]
                if (
                    Y[i].shape != (self.meta.dim_output,)
                    or not np.all(np.isfinite(Y[i]))
                    or c_max < 0.0
                    or c_max > 1.0
                    or x_front < 0.0
                    or x_front > 500.0
                    or t_arrive < 1.0
                    or t_arrive > 10000.0
                ):
                    raise RuntimeError("non-finite or out-of-contract groundwater transport response")
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "groundwater_contamination_6d Ogata-Banks surrogate failed for "
                    f"physical input {x_text}; refusing silent NaN or out-of-contract output because "
                    "the formal benchmark response surface requires a finite "
                    "transport calculation inside the declared output ranges."
                ) from None
        return Y

    @staticmethod
    def _ogata_banks(v, alpha_L, K_d, rho_b, t_half, n_por):
        """Ogata–Banks-style 1-D ADE surrogate with adsorption and decay.

        C(x,t)/C0 = 0.5·[erfc((Rx−vt)/(2√(DRt)))              # advection term
                        + exp(vx/D)·erfc((Rx+vt)/(2√(DRt)))]   # reflection term (small)
        × exp(−λt/R)                                             # separable decay attenuation
        """
        D_m = 1e-9 * 86400  # molecular diffusion coefficient m²/d (≈ 10⁻⁹ m²/s)
        tau = 0.7  # tortuosity factor
        D = alpha_L * v + D_m * tau

        R = 1.0 + rho_b * K_d / (1000.0 * n_por)  # retardation factor (K_d in L/kg)
        lam = np.log(2.0) / max(t_half, 1e-3)  # decay rate 1/d

        t_eval = 365.0  # 1 year = 365 days
        x_eval = np.linspace(0.0, 500.0, 500)

        def conc(x, t):
            return GroundwaterContamination6D._relative_concentration(
                x=x,
                t=t,
                v=v,
                D=D,
                R=R,
                lam=lam,
            )

        # Concentration profile after 1 year
        C_profile = np.array([conc(x, t_eval) for x in x_eval])
        C_max_frac = float(np.max(C_profile))

        # Front position (C = 0.01)
        idx_front = np.where(C_profile > 0.01)[0]
        x_front = float(x_eval[idx_front[-1]]) if len(idx_front) > 0 else 0.0

        # Arrival time at x = 10 m: first threshold crossing.  Do not
        # require the final-time concentration to remain above threshold;
        # with first-order decay the concentration can arrive early and then
        # fall below the threshold again.
        t_arr = 10000.0
        for t_try in np.logspace(0, 4, 200):
            if conc(10.0, t_try) > 0.01:
                t_arr = float(t_try)
                break

        return np.array([C_max_frac, x_front, t_arr])

    @staticmethod
    def _relative_concentration(x, t, v, D, R, lam):
        """Relative concentration for the implemented Ogata-Banks surrogate."""
        if t <= 0:
            return 0.0
        if D <= 0 or R <= 0:
            raise ValueError("groundwater transport coefficients must be positive")
        A1 = (R * x - v * t) / (2.0 * np.sqrt(D * R * t) + 1e-30)
        A2 = (R * x + v * t) / (2.0 * np.sqrt(D * R * t) + 1e-30)
        vx_D = v * x / D
        reflection = erfcx(A2) * np.exp(vx_D - A2 * A2)
        c = 0.5 * (erfc(A1) + reflection)
        c *= np.exp(-lam * t / R)
        if not np.isfinite(c):
            raise FloatingPointError("non-finite groundwater relative concentration")
        tol = 1e-12
        if c < -tol or c > 1.0 + tol:
            raise FloatingPointError(
                "groundwater relative concentration outside physical [0, 1] envelope"
            )
        return float(min(1.0, max(0.0, c)))
