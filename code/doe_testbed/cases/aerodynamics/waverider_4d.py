"""
Case A06: Hypersonic waverider lift-to-drag ratio
=================================================

Engineering background
----------------------
A waverider is a high-lift-to-drag hypersonic vehicle concept
(Ma > 5).  Its lower surface is designed to ride on the generated shock
wave, helping retain high pressure on the windward surface and improving
compression lift.  Real integrated vehicles still depend on propulsion,
base drag, viscous effects, shock attachment, off-design operation,
aerothermal protection, real-gas chemistry, vibrational excitation,
dissociation, ionization, chemical/thermal nonequilibrium, rarefied-flow
effects, leading-edge bluntness, trim/control moments, stability, and
packaging constraints.
NASA X-43A/Hyper-X and NASA hypersonic-facility material are used here
only as boundary evidence for integrated scramjet flight, CFD/wind-tunnel
workflows, thermal protection, and high-enthalpy testing requirements.
NASA Glenn Mach-number material is used only as boundary evidence that
Ma > 5 is hypersonic and that shock and high-temperature gas effects become
central; it does not validate the fixed-gamma modified-Newtonian formulas.
NASA Glenn's 1x1 Supersonic Wind Tunnel material is likewise boundary
evidence for benchmark-quality CFD-validation experiments in supersonic and
hypersonic fluid mechanics, not validation evidence for this algebraic case.
NASA Ames arc-jet and thermal-protection material is boundary evidence that
real hypersonic/re-entry qualification includes convective/radiative heating,
boundary-layer thermal loads, material thermal response, and ablation/TPS
testing, none of which are modelled by this L/D surrogate.
NASA Hypersonic Technology Project material is used only as current NASA
program-scope boundary evidence for system-level design/analysis/validation,
propulsion, vehicle technologies, and high-temperature durable materials; it
does not validate this four-input algebraic L/D response surface.

This case uses an analytical approach based on the modified Newtonian
impact theory to compute the vehicle’s L/D and study how geometric
parameters and flight conditions affect aerodynamic efficiency.

Physics model
-------------
Simplified waverider aerodynamic model based on modified Newtonian
impact theory:

1. Lower-surface pressure coefficient (windward impact surface):
       C_{p,lower} = C_{p,max} · sin²δ
   where δ = θ_c + α is the effective deflection angle of the lower
   surface relative to the freestream.
   C_{p,max}(Ma) is given by the calorically-perfect Rayleigh-Pitot
   formula (post-shock stagnation total pressure) at fixed γ.

2. Upper surface is in the leeward shadow: C_{p,upper} = 0 (Newtonian
   shadowing).

3. Lift / wave-drag decomposition:
       C_L    = (C_{p,lower} − C_{p,upper}) · cosδ · f_plan
       C_{D,w}= (C_{p,lower} − C_{p,upper}) · sinδ · f_plan
   f_plan = sin(φ), the wetted-area planform-projection ratio.

4. The pressure model uses a calorically-perfect-gas Rayleigh-Pitot
   coefficient with fixed γ = 1.4; real-gas variable-specific-heat,
   vibrational excitation, dissociation, ionization, and chemical/thermal
   nonequilibrium effects are explicitly out of scope.

5. Friction drag uses a hypersonic turbulent skin-friction surrogate
   (van-Driest-inspired Mach correction, order-of-magnitude only):
       C_f ≈ 0.0015 · (1 + 0.02·Ma²)^(−0.3)
       C_{D,f} = C_f · S_wet/S_ref, S_wet/S_ref ≈ 2/sin(φ)

6. Volumetric efficiency: V^{2/3}/S_ref, geometric-similarity model:
       η_v ≈ 0.22 · tan(θ_c)^{2/3} · sin(φ)^{−1/3}

Note: this case is a **simplified surrogate model** for DOE trend
analysis and cannot replace real CFD / wind-tunnel data for waverider
engineering design.  It does not solve an osculating-cone or
streamline-tracing waverider design, shock fitting, shock-attachment
verification, leading-edge bluntness, boundary-layer heating, base drag,
propulsion integration, trim/control moments, static/dynamic stability,
rarefied-flow effects, real-gas chemistry, vibrational excitation,
dissociation, ionization, chemical/thermal nonequilibrium, or thermal
protection problem.

Input variables
---------------
Ma       Flight Mach number   [5, 12]   -     Hypersonic freestream
theta_c  Cone half-angle      [5, 20]   deg   Determines lower-surface inclination
alpha    Angle of attack      [0, 5]    deg   Incremental deflection
phi      Waverider half-span  [30, 90]  deg   Determines planform-projection ratio f_plan

Output responses
----------------
L_D      Lift-to-drag ratio   -   Primary output (aerodynamic efficiency)
CL       Lift coefficient     -   Dimensionless lift (ref S_ref)
eta_v    Volumetric efficiency -   V^(2/3)/S_ref (payload / aero-area metric)

DOE validation value
--------------------
VS-B (effect estimation):
    Geometry dominates the response: theta_c, alpha, and phi set the
    effective deflection, projected lifting area, and wetted-area friction
    penalty.  Ma is retained as a weak hypersonic trend variable through
    Cpmax saturation and the empirical skin-friction correction.

References
----------
1. Anderson, J. D. (1990). Modern Compressible Flow, 2nd ed. McGraw-Hill.
2. Bowcutt, K. G., Anderson, J. D. & Capriotti, D. (1987).
   Viscous Optimized Hypersonic Waveriders. AIAA Paper 87-0272.
3. NASA TP-3559 (1996). Aerodynamic Characteristics of Waverider-Derived
   Hypersonic Cruise Configurations.
   https://ntrs.nasa.gov/api/citations/19960045290/downloads/19960045290.pdf
   Public NTRS metadata describes a design Mach number of 4.0 and
   experimental/limited-CFD data over Mach 1.6-4.63, below this case's
   registered Mach 5-12 domain; it is therefore scope and order-of-magnitude
   context, not validation of the full benchmark domain.
4. NASA CR-182076 (1988). Aerodynamic Preliminary Analysis System II.
   https://ntrs.nasa.gov/api/citations/19910013767/downloads/19910013767.pdf
5. NACA Report 1135. Equations, Tables, and Charts for Compressible Flow.
   https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf
6. NASA Glenn. Normal Shock Wave Equations.
   https://www.grc.nasa.gov/WWW/BGH/normal.html
7. NASA Glenn. Real Gas Effects.
   https://www.grc.nasa.gov/www/BGH/realgas.html
8. NASA Glenn. Mach Number.
   https://www.grc.nasa.gov/www/BGH/mach.html
9. NASA Glenn. Hypersonic Missions.
   https://www.grc.nasa.gov/WWW/BGH/hmission.html
10. NASA Glenn. Re-Entry Aircraft.
   https://www.grc.nasa.gov/www/BGH/hihyper.html
11. NASA. X-43A Hyper-X.
   https://www.nasa.gov/reference/x-43a/
12. NASA. CFD Image of X-43A at Mach 7 Test Condition.
   https://www.nasa.gov/image-article/cfd-image-of-x-43a-mach-7-test-condition/
13. NASA Glenn. Hypersonic Tunnel Facility.
   https://www.nasa.gov/centers-and-facilities/glenn/hypersonic-tunnel-facility/
14. NASA Glenn. 1x1 Supersonic Wind Tunnel.
   https://www.nasa.gov/centers-and-facilities/glenn/1x1-supersonic-wind-tunnel/
15. NASA Ames. Interaction Heating Facility / Arc Jet Complex.
   https://www.nasa.gov/ames/arc-jet-complex/interaction-heating-facility/
16. NASA. Fully Implicit Ablation and Thermal Response Program (FIAT).
   https://software.nasa.gov/software/ARC-14294-1
17. NASA. Hypersonic Technology Project.
   https://www.nasa.gov/directorates/armd/aavp/ht/
18. NASA. Hypersonic Research Topics.
   https://www.nasa.gov/directorates/armd/aavp/ht/hypersonic-research-topics/

The public NASA anchors support waverider order-of-magnitude context,
modified-Newtonian/impact-method context, and omitted real-gas physics
boundaries such as variable specific heats, vibrational excitation,
dissociation, ionization, and chemical/thermal nonequilibrium. The
current NASA Glenn hypersonic-missions page is especially relevant to this
registered Mach 5-12 box because it states that Mach 5-7 design work must
evaluate real-gas effects. NASA Glenn Mach-number material independently
anchors the Ma > 5 hypersonic classification and the shock/high-temperature
gas-effect boundary. NASA TP-3559 is especially useful for integrated-configuration
caveats, but its public Mach range is below this benchmark case's Mach 5-12
input box. NASA X-43A/Hyper-X sources and the NASA Glenn Hypersonic Tunnel
Facility source further show that real hypersonic vehicles and tests involve
airframe-integrated scramjet propulsion, additional thermal protection,
CFD/engine-operating analysis, hypersonic aerodynamic flight data, and
Mach 5-7 true-enthalpy ground-test capability. The NASA Glenn 1x1 Supersonic
Wind Tunnel source sharpens the omitted validation workflow by identifying
benchmark-quality experiments for CFD code validation in supersonic and
hypersonic fluid mechanics, with continuous Mach 1.3-6.0 operation. These
NASA Ames arc-jet and FIAT anchors sharpen the TPS omitted-scope boundary:
real aerothermal qualification evaluates convective/radiative heating,
surface heat flux, material thermal response, and ablation recession. These
NASA Hypersonic Technology Project anchors sharpen the current program-level
boundary: serious hypersonic work involves system-level design, analysis, and
validation plus propulsion, vehicle technologies, and high-temperature durable
materials. These
anchors are not external
validation data for this fixed-gamma
modified-Newtonian surrogate, its empirical friction term, full Mach-domain
response, shock-attached design accuracy, aerothermal/TPS prediction,
CFD/wind-tunnel/arc-jet calibration, or integrated-vehicle performance.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class Waverider4D(BaseCase):
    """Hypersonic waverider lift-to-drag ratio (4D → 3D, modified Newtonian impact theory)."""

    meta = CaseMeta(
        case_id="waverider_4d",
        name="Hypersonic waverider L/D (modified Newtonian impact theory)",
        version="1.0.0",
        domain="aerodynamics",
        tags=[
            "waverider",
            "hypersonic",
            "newtonian_impact",
            "rayleigh_pitot",
            "lift_drag",
            "4d",
            "high_speed",
        ],
        dim_input=4,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="Ma",
                symbol="Ma_\\infty",
                description="Flight Mach number",
                unit="-",
                lower=5.0,
                upper=12.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="theta_c",
                symbol=r"\theta_c",
                description="Cone half-angle",
                unit="deg",
                lower=5.0,
                upper=20.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="alpha",
                symbol=r"\alpha",
                description="Off-design angle of attack / attitude perturbation",
                unit="deg",
                lower=0.0,
                upper=5.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="phi",
                symbol=r"\phi_w",
                description="Waverider half-span angle (affects lift-area ratio)",
                unit="deg",
                lower=30.0,
                upper=90.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="L_D",
                symbol="L/D",
                description="Lift-to-drag ratio",
                unit="-",
                is_primary=True,
                expected_range=(0.5, 10.0),
            ),
            OutputVariable(
                name="CL",
                symbol="C_L",
                description="Lift coefficient",
                unit="-",
                is_primary=False,
                expected_range=(0.0, 0.5),
            ),
            OutputVariable(
                name="eta_v",
                symbol=r"\eta_V",
                description="Volumetric efficiency V^{2/3}/S_ref",
                unit="-",
                is_primary=False,
                expected_range=(0.01, 0.15),
            ),
        ],
        governing_equation=(
            "Modified Newtonian impact theory:\n"
            "  C_{p,lower} = C_{p,max}(Ma)·sin²(θ_c+α)\n"
            "  C_L = C_{p,lower}·cosδ·sinφ\n"
            "  C_D = C_{p,lower}·sinδ·sinφ + C_f·2/sinφ\n"
            "C_{p,max} from Rayleigh pitot formula (Anderson 3rd, Ch.14)."
        ),
        pde_type="algebraic",
        spatial_dim=2,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Anderson, J. D. (2019). Hypersonic and High-Temperature Gas Dynamics, 3rd ed. AIAA, Chap. 14.",
            "Bowcutt, K. G. et al. (1987). Viscous Optimized Hypersonic Waveriders. AIAA-87-0272.",
            "NASA TP-3559 (1996). Aerodynamic Characteristics of Waverider-Derived Hypersonic Cruise Configurations, "
            "NTRS public metadata: design Mach 4.0 and data over Mach 1.6-4.63, used as "
            "integrated-configuration/order-of-magnitude context below this case's Mach 5-12 domain, "
            "https://ntrs.nasa.gov/citations/19960045290, "
            "PDF: https://ntrs.nasa.gov/api/citations/19960045290/downloads/19960045290.pdf",
            "NASA CR-182076 (1988). Aerodynamic Preliminary Analysis System II, "
            "modified Newtonian hypersonic methods, https://ntrs.nasa.gov/api/citations/19910013767/downloads/19910013767.pdf",
            "NACA Report 1135, Equations, Tables, and Charts for Compressible Flow, "
            "normal-shock/Rayleigh-Pitot compressible-flow context, "
            "https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf",
            "NASA Glenn, Normal Shock Wave Equations, calorically perfect/imperfect gas shock context, "
            "https://www.grc.nasa.gov/WWW/BGH/normal.html",
            "NASA Glenn, Hypersonic Aerodynamics background, https://www.grc.nasa.gov/www/BGH/index.html",
            "NASA Glenn, Real Gas Effects, hypersonic chemistry/thermal excitation context, "
            "https://www.grc.nasa.gov/www/BGH/realgas.html",
            "NASA Glenn, Mach Number, Ma > 5 hypersonic classification and "
            "shock/high-temperature gas-effect boundary context, "
            "https://www.grc.nasa.gov/www/BGH/mach.html",
            "NASA Glenn, Hypersonic Missions, Mach 5-7 mission-design context and "
            "explicit note that real-gas effects must be evaluated in design, "
            "https://www.grc.nasa.gov/WWW/BGH/hmission.html",
            "NASA Glenn, Re-Entry Aircraft, hypersonic shock/plasma/TPS/high-drag boundary context, "
            "https://www.grc.nasa.gov/www/BGH/hihyper.html",
            "NASA, X-43A Hyper-X, integrated scramjet flight, thermal-protection, and "
            "hypersonic aerodynamic-data boundary context, https://www.nasa.gov/reference/x-43a/",
            "NASA, CFD Image of X-43A at Mach 7 Test Condition, engine-operating CFD "
            "boundary context, https://www.nasa.gov/image-article/cfd-image-of-x-43a-mach-7-test-condition/",
            "NASA Glenn, Hypersonic Tunnel Facility, Mach 5-7 true-enthalpy "
            "ground-test boundary context, "
            "https://www.nasa.gov/centers-and-facilities/glenn/hypersonic-tunnel-facility/",
            "NASA Glenn, 1x1 Supersonic Wind Tunnel, benchmark-quality "
            "supersonic/hypersonic CFD-validation experiment boundary context, "
            "continuous Mach 1.3-6.0 operation, "
            "https://www.nasa.gov/centers-and-facilities/glenn/1x1-supersonic-wind-tunnel/",
            "NASA Ames, Interaction Heating Facility / Arc Jet Complex, "
            "convective/radiative heating and thermal-protection-system test "
            "boundary context, not validation of this aerodynamic L/D surrogate, "
            "https://www.nasa.gov/ames/arc-jet-complex/interaction-heating-facility/",
            "NASA, Fully Implicit Ablation and Thermal Response Program (FIAT), "
            "thermal-response and ablation-analysis software boundary context, "
            "not a material-response model implemented by this case, "
            "https://software.nasa.gov/software/ARC-14294-1",
            "NASA Hypersonic Technology Project, program-scope boundary context "
            "for reusable airbreathing hypersonic-flight research organized "
            "around system-level design/analysis/validation, propulsion, "
            "vehicle technologies, and high-temperature durable materials, "
            "not validation of this four-input algebraic L/D surrogate, "
            "https://www.nasa.gov/directorates/armd/aavp/ht/; "
            "research topics: https://www.nasa.gov/directorates/armd/aavp/ht/hypersonic-research-topics/",
        ],
        solver_library="numpy (modified Newtonian analytical formulas)",
        solver_details=(
            "Rayleigh pitot formula + Newtonian impact + empirical "
            "van-Driest-inspired skin-friction surrogate with fixed gamma=1.4 "
            "calorically-perfect-gas Cpmax; no osculating-cone or "
            "streamline-tracing design, shock fitting, shock-attachment "
            "verification, leading-edge bluntness, boundary-layer heating, "
            "base drag, propulsion integration, trim/control moments, variable "
            "specific heats, vibrational excitation, dissociation, ionization, "
            "chemical/thermal nonequilibrium, nonequilibrium chemistry, real-gas chemistry, "
            "NASA Glenn Mach-number shock/high-temperature gas-effect workflow calibration, "
            "plasma effects, blunt-body/re-entry high-drag flight, rarefied-flow effects, "
            "static/dynamic stability, TPS, airframe-integrated scramjet propulsion, "
            "high-enthalpy facility simulation, benchmark-quality CFD-validation "
            "experiments, arc-jet thermal-protection testing, material thermal-response "
            "or ablation analysis, NASA Hypersonic Technology Project system-level "
            "design/analysis/validation, propulsion, vehicle-technologies, or "
            "high-temperature durable-materials workflow, CFD, "
            "or wind-tunnel calibration; named _validate_physical_inputs "
            "private physical-input evaluation rejects "
            "malformed arrays and Ma/theta_c/alpha/phi values outside the declared "
            "benchmark domain before modified-Newtonian formula evaluation; "
            "non-finite, non-positive, or out-of-contract L_D/CL/eta_v "
            "responses are rejected before dispatch can archive a row"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 2.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_unimodal", "strong_interaction"],
        effective_dim=4,
        important_variables=["Ma", "theta_c", "alpha", "phi"],
        interaction_pairs=[
            ("Ma", "theta_c"),
            ("Ma", "alpha"),
            ("Ma", "phi"),
            ("theta_c", "alpha"),
            ("theta_c", "phi"),
            ("alpha", "phi"),
        ],
        has_failure_boundary=False,
        sobol_S1={"Ma": 0.005, "theta_c": 0.696, "alpha": 0.054, "phi": 0.116},
        sobol_ST={"Ma": 0.009, "theta_c": 0.815, "alpha": 0.099, "phi": 0.212},
        sobol_computation="salib_saltelli_n2048_seed42_L_D_output",
        applicable_scenarios=["VS-B", "VS-C", "VS-D"],
        scenario_notes={
            "VS-B": (
                "Geometry dominates: theta_c, alpha, and phi drive the deflection, "
                "planform, and wetted-area terms. Ma is a weak hypersonic trend "
                "variable through Cpmax saturation and empirical friction scaling, "
                "not the dominant interaction driver in this surrogate."
            ),
            "VS-C": (
                "Surrogate-accuracy stressor for a smooth but strongly coupled "
                "modified-Newtonian-plus-friction response; the case tests trend "
                "capture, not CFD- or wind-tunnel-calibrated prediction accuracy, "
                "shock-attached design, shock-attachment verification, "
                "real-gas/aerothermal analysis, fixed-gamma calorically-perfect "
                "gas rather than variable-specific-heat, vibrational excitation, "
                "dissociation, ionization, or chemical/thermal nonequilibrium chemistry, "
                "rarefied-flow analysis, "
                "trim/control stability, or full integrated-vehicle performance."
            ),
            "VS-D": (
                "Compact 4-D small-sample case with geometry-dominated interactions "
                "and an L/D tradeoff in theta_c; use convergence curves without "
                "pre-claiming a required sample count."
            ),
        },
        recommended_n_train={"default": 40},
        n_saturation=200,
        noise_factor_candidates=["alpha"],
        control_factor_candidates=["Ma", "theta_c", "phi"],
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "waverider_4d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        Ma_raw = X_phys[:, 0]
        theta_c_raw = X_phys[:, 1]
        alpha_raw = X_phys[:, 2]
        phi_raw = X_phys[:, 3]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(Ma_raw < 5.0 - tol)
            or np.any(Ma_raw > 12.0 + tol)
            or np.any(theta_c_raw < 5.0 - tol)
            or np.any(theta_c_raw > 20.0 + tol)
            or np.any(alpha_raw < 0.0 - tol)
            or np.any(alpha_raw > 5.0 + tol)
            or np.any(phi_raw < 30.0 - tol)
            or np.any(phi_raw > 90.0 + tol)
        ):
            raise ValueError(
                "waverider_4d physical inputs must stay inside the declared "
                "Ma/theta_c/alpha/phi benchmark domain before modified-Newtonian evaluation"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        Ma = X_phys[:, 0]
        theta_c = np.deg2rad(X_phys[:, 1])
        alpha = np.deg2rad(X_phys[:, 2])
        phi = np.deg2rad(X_phys[:, 3])
        Y = self._waverider_perf(Ma, theta_c, alpha, phi)
        if (
            Y.shape != (X_phys.shape[0], self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y <= 0.0)
        ):
            raise RuntimeError(
                "waverider_4d modified-Newtonian solve produced an invalid "
                "response array; refusing silent non-finite, non-positive, "
                "or malformed outputs in the formal benchmark archive."
            )
        output_ranges = np.asarray(
            [output.expected_range for output in self.meta.output_variables],
            dtype=float,
        )
        tol = 1e-10
        below = Y < (output_ranges[:, 0] - tol)
        above = Y > (output_ranges[:, 1] + tol)
        if np.any(below) or np.any(above):
            raise RuntimeError(
                "waverider_4d modified-Newtonian solve produced an "
                "out-of-contract L_D/CL/eta_v response outside the declared "
                "output ranges; refusing silent archive contamination."
            )
        return Y

    # ── Rayleigh pitot formula: C_{p,max} for modified Newtonian theory ────────
    @staticmethod
    def _cp_max(Ma: np.ndarray, gamma: float = 1.4) -> np.ndarray:
        """C_{p,max}(Ma) for modified Newtonian theory.

        p_{0,2}/p_1 = ((γ+1)²·Ma² / (4γ·Ma²−2(γ−1)))^(γ/(γ−1))
                    · (1−γ+2γ·Ma²)/(γ+1)
        C_{p,max} = 2/(γ·Ma²) · (p_{0,2}/p_1 − 1)
        For γ = 1.4, this Rayleigh-pitot modified-Newtonian value
        asymptotes to about 1.839 as Ma → ∞.  The classical Newtonian
        limiting coefficient 2 is a separate idealised upper bound.
        """
        Ma = np.asarray(Ma, dtype=float)
        if gamma <= 1.0 or not np.isfinite(gamma):
            raise ValueError("invalid gamma for Rayleigh-pitot Cpmax calculation")
        if np.any(~np.isfinite(Ma)) or np.any(Ma <= 1.0):
            raise ValueError(
                "Rayleigh-pitot Cpmax contract requires finite supersonic Mach numbers"
            )
        g = gamma
        term1 = ((g + 1) ** 2 * Ma**2 / (4 * g * Ma**2 - 2 * (g - 1))) ** (g / (g - 1))
        term2 = (1 - g + 2 * g * Ma**2) / (g + 1)
        p02_p1 = term1 * term2
        return 2.0 / (g * Ma**2) * (p02_p1 - 1.0)

    def _waverider_perf(
        self,
        Ma: np.ndarray,
        theta_c: np.ndarray,
        alpha: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """Vectorised waverider performance (modified Newtonian + turbulent friction)."""
        if (
            np.any(~np.isfinite(Ma))
            or np.any(~np.isfinite(theta_c))
            or np.any(~np.isfinite(alpha))
            or np.any(~np.isfinite(phi))
            or np.any(Ma <= 1.0)
        ):
            raise RuntimeError(
                "waverider_4d modified-Newtonian inputs are outside the finite "
                "supersonic Rayleigh-Pitot contract."
            )
        Cp_max = self._cp_max(Ma)

        # ── Lower-surface (windward) pressure coefficient ────────────────────
        delta = theta_c + alpha  # effective deflection angle
        sin_d = np.sin(delta)
        Cp_lower = Cp_max * sin_d**2  # Newtonian impact
        Cp_upper = 0.0  # leeward shadow

        # ── Planform factor: larger φ → larger lower-surface area ───────────────
        f_plan = np.sin(phi)

        # ── Lift and wave-drag coefficients ─────────────────────────────────────
        CL = (Cp_lower - Cp_upper) * np.cos(delta) * f_plan
        CD_wave = (Cp_lower - Cp_upper) * sin_d * f_plan

        # ── Friction drag coefficient (empirical hypersonic skin-friction scale) ──────
        # Re fixed at typical flight-altitude (~30 km) order 5e6–1e7;
        # high Ma/aerodynamic-heating trends are represented by the
        # (1+0.02·Ma²)^-0.3 correction. This is not a full van Driest II
        # boundary-layer calculation.
        Cf = 0.0015 * (1.0 + 0.02 * Ma**2) ** (-0.3)
        # Wetted-area ratio: lower f_plan + upper ~f_plan + sides ≈ 2/(sin φ).
        # The declared domain has phi >= 30 deg, so sin(phi) is safely positive.
        # Treat any violation as a contract failure instead of clipping it into
        # a seemingly valid aerodynamic row.
        sin_phi = np.sin(phi)
        if np.any(~np.isfinite(sin_phi)) or np.any(sin_phi <= 0.0):
            raise RuntimeError(
                "waverider_4d modified-Newtonian geometry produced a non-positive "
                "sin(phi); refusing silent wetted-area clipping."
            )
        S_wet_ratio = 2.0 / sin_phi
        CD_fric = Cf * S_wet_ratio

        CD_total = CD_wave + CD_fric
        if np.any(~np.isfinite(CD_total)) or np.any(CD_total <= 0.0):
            raise RuntimeError(
                "waverider_4d modified-Newtonian solve produced non-positive "
                "total drag; refusing silent L/D denominator clipping."
            )

        L_D = CL / CD_total

        # ── Volumetric efficiency η_v = V^(2/3)/S_ref ─────────────────────
        # For wedge-cone waverider: V ∝ tan(θ_c)·sin(φ)·L³, S_ref ∝ sin(φ)·L²,
        # so η_v ∝ tan(θ_c)^{2/3}·sin(φ)^{-1/3}.  The coefficient 0.22 is a
        # surrogate scale factor chosen to keep V^(2/3)/S_ref in the published
        # waverider order-of-magnitude range; it is not a CFD-calibrated constant.
        eta_v = 0.22 * np.tan(theta_c) ** (2.0 / 3.0) * np.sin(phi) ** (-1.0 / 3.0)

        return np.column_stack([L_D, CL, eta_v])
