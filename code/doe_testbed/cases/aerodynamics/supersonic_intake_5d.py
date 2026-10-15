"""
Case A04: Supersonic intake performance surrogate (oblique shock system)
========================================================================

Engineering background
----------------------
A supersonic intake decelerates the freestream from supersonic to subsonic
speed as required by the engine; its performance directly affects thrust
and fuel consumption.  External-compression (wedge) intakes are the most
common configuration: the flow passes through several oblique shocks plus
a terminal normal shock to complete compression.

This case analyses a low-order 2-D external-compression intake surrogate.
Supersonic flow over two compression wedges generates weak attached oblique
shocks that progressively decelerate the flow; a terminal normal shock (or
sonic condition) transitions the flow to subsonic in the diffuser.  The model
does not include viscous boundary layers, bleed, cowl-lip geometry, shock
train dynamics, unstart, inlet buzz, compressor-face distortion,
mass-capture/spillage, spillage drag, area scheduling,
diffuser boundary-layer growth, inlet-engine matching,
variable geometry, strong-shock branch selection from back pressure,
three-dimensional sidewall/corner effects, real-gas chemistry, or
CFD/wind-tunnel-calibrated off-design behaviour.  NASA Glenn X-59
inlet/nozzle project material is used only as boundary evidence for CFD,
wind-tunnel, propulsion-system integration, real-time test-display, digital
engine/aircraft architecture, and engine-face distortion workflows outside
this algebraic shock surrogate.
NASA Glenn inlet-performance model material is likewise boundary evidence:
it frames pressure recovery, spillage drag, and distortion through wind-tunnel
engine-airflow sweeps and rake-style measurements, not through this compact
two-ramp shock cascade.

Physics model (oblique-shock relations)
---------------------------------------
Exact inviscid shock relations are used for the attached-shock segments:

Oblique-shock parameters: freestream Ma₁, wedge half-angle δ
  → shock angle β (from the θ–β–M relation)
Post-shock Mach number Ma₂ (reduced)
Total-pressure ratio: p02/p01 = [(γ+1)Ma₁²sin²β/((γ−1)Ma₁²sin²β+2)]^(γ/(γ−1))
       × [(γ+1)/(2γMa₁²sin²β−(γ−1))]^(1/(γ−1))

Total pressure recovery:
  η = ∏_i p0_{i+1}/p0_i  (product of total-pressure ratios across all shocks)

The implementation uses calorically perfect, constant-γ shock equations at
each sample.  Sweeping γ over 1.30–1.40 is a low-order gas-property stressor,
not a calorically imperfect / vibrational-energy / thermochemical real-gas
calculation.  The low-level shock helpers are intentionally defined only for
finite supersonic upstream Mach numbers, positive compression deflections,
and γ > 1; formula-domain violations raise explicit errors instead of
creating archiveable surrogate rows.

Input variables
---------------
Ma_in    Freestream Mach number  [2.0, 5.0]  -     Freestream condition
delta1   1st wedge half-angle    [5, 15]     deg   Geometric design variable
delta2   2nd wedge half-angle    [5, 20]     deg   Geometric design variable
gamma    Specific heat ratio     [1.3, 1.4]  -     Constant-gamma gas-property stressor
alpha    AoA offset              [-3, 3]     deg   Intake off-design angle

Output responses
----------------
eta_pr   Total pressure recovery  -   Primary output (closer to 1 is better)
Ma_exit  Diffuser-entry Mach no.  -   Subsonic exit Mach number
p_ratio  Static pressure ratio    -   Flow compression ratio

DOE validation value
--------------------
VS-C (surrogate accuracy):
    η vs Ma_in exhibits a mild discontinuity at the two-shock / three-shock
    mode switch; surrogate accuracy degrades there.
    Shock detachment and the terminal-shock/no-terminal-shock branch also
    create finite response jumps, so this case is a discontinuous topology
    stressor rather than a globally smooth intake map.

VS-G (sequential):
    Locate the feasible-domain boundary where η > 0.95 in (Ma, δ1, δ2) space.

References
----------
1. Anderson, J. D. (1990). Modern Compressible Flow, 2nd ed. McGraw-Hill.
2. Seddon, J. & Goldsmith, E. L. (1999). Intake Aerodynamics, 2nd ed.
   AIAA Education Series.
3. NASA Glenn. Oblique Shock Relations.
   https://www.grc.nasa.gov/www/k-12/airplane/oblique.html
4. NASA Glenn. Normal Shock Relations.
   https://www.grc.nasa.gov/WWW/BGH/normal.html
5. NASA Glenn. Inlets.
   https://www.grc.nasa.gov/WWW/k-12/airplane/inlet.html
6. NASA/TM-2012-217660. Design and Analysis Tool for External-Compression
   Supersonic Inlets (SUPIN).
   https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120013600.pdf
7. NASA/TM-20240008586. SUPIN: A Tool for the Aerodynamic Design and
   Analysis of Supersonic Inlets.
   https://ntrs.nasa.gov/citations/20240008586
8. NASA Glenn Inlets and Nozzles project pages and X-59 inlet-distortion
   material, used as current boundary evidence for CFD, wind-tunnel,
   propulsion-system integration, real-time test displays, digital
   engine/aircraft architecture, and engine-face distortion workflows not
   implemented here.
   https://www.nasa.gov/glenn/research/inlets-and-nozzles/support-of-flight-demonstration-projects/
   https://www.nasa.gov/glenn/research/inlets-and-nozzles/design-analysis-software/
   https://ntrs.nasa.gov/citations/20220010443
9. NASA Glenn. Inlet Performance Model.
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/inlet-performance-model/
   Legacy NASA Glenn inlet-performance page:
   https://www.grc.nasa.gov/www/k-12/BGP/inleth.html
"""

import numpy as np
from scipy.optimize import brentq

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class SupersonicIntake5D(BaseCase):
    """Supersonic external-compression intake surrogate, 5D→3D."""

    meta = CaseMeta(
        case_id="supersonic_intake_5d",
        name="Supersonic external-compression intake (oblique-shock TPR)",
        version="1.0.0",
        domain="aerodynamics",
        tags=[
            "supersonic",
            "intake",
            "shock",
            "oblique_shock",
            "total_pressure_recovery",
            "5d",
            "compressible",
        ],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="Ma_in",
                symbol="Ma_\\infty",
                description="Freestream Mach number",
                unit="-",
                lower=2.0,
                upper=5.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="delta1",
                symbol=r"\delta_1",
                description="1st compression-wedge half-angle",
                unit="deg",
                lower=5.0,
                upper=15.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="delta2",
                symbol=r"\delta_2",
                description="2nd compression-wedge half-angle",
                unit="deg",
                lower=5.0,
                upper=20.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="gamma",
                symbol=r"\gamma",
                description="Specific heat ratio for constant-gamma shock relations",
                unit="-",
                lower=1.30,
                upper=1.40,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="alpha_off",
                symbol=r"\alpha_{off}",
                description="Angle-of-attack offset (off-design)",
                unit="deg",
                lower=-3.0,
                upper=3.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="eta_pr",
                symbol=r"\eta_{pr}",
                description="Total pressure recovery (product across all shocks)",
                unit="-",
                is_primary=True,
                # In failure region (shock detachment, severe over-deflection) eta_pr can drop to 0.1;
                # this is a feasible-domain identification target for DOE.
                expected_range=(0.0, 1.0),
                failure_threshold=0.80,
                failure_direction="below",
            ),
            OutputVariable(
                name="Ma_exit",
                symbol="Ma_{exit}",
                description="Subsonic Mach number after terminal shock",
                unit="-",
                is_primary=False,
                expected_range=(0.3, 1.0),
            ),
            OutputVariable(
                name="p_ratio",
                symbol="p_2/p_1",
                description="Static pressure ratio (flow compression ratio)",
                unit="-",
                is_primary=False,
                expected_range=(1.0, 200.0),
            ),
        ],
        governing_equation=(
            "Two-ramp inviscid external-compression intake surrogate:\n"
            "  tan(δ)=2cotβ·(Ma²sin²β−1)/(Ma²(γ+cos2β)+2)\n"
            "  Equivalent denominator: (γ+1)Ma²−2(Ma²sin²β−1)\n"
            "  Weak attached oblique shocks + terminal normal shock; cumulative total-pressure ratio\n"
            "  Strong-shock branch/back-pressure selection is outside the benchmark scope\n"
            "  Calorically perfect constant-gamma shock equations; gamma sweep is a gas-property stressor\n"
            "  Detached-shock/off-design fallback is a surrogate penalty, not CFD or inlet-unstart modelling"
        ),
        pde_type="algebraic",
        spatial_dim=1,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Anderson, J. D. (1990). Modern Compressible Flow, 2nd ed. McGraw-Hill.",
            "Seddon, J. & Goldsmith, E. L. (1999). Intake Aerodynamics, 2nd ed. AIAA.",
            "NASA Glenn, Oblique Shock Relations, https://www.grc.nasa.gov/www/k-12/airplane/oblique.html",
            "NASA Glenn, Normal Shock Relations, https://www.grc.nasa.gov/WWW/BGH/normal.html",
            "NASA Glenn, Inlets, supersonic inlet deceleration, pressure-recovery index, and "
            "spillage/capture context, https://www.grc.nasa.gov/WWW/k-12/airplane/inlet.html",
            "NASA Glenn, Inlet Performance, pressure recovery / spillage drag / distortion context, "
            "https://www.grc.nasa.gov/www/k-12/airplane/inleth.html; "
            "legacy BGP mirror: https://www.grc.nasa.gov/www/k-12/BGP/inleth.html",
            "NASA Glenn, Inlet Performance Model, wind-tunnel engine-airflow "
            "sweep, spillage-drag, pressure-recovery, and distortion-rake "
            "measurement context; boundary evidence only, not validation of "
            "this algebraic shock cascade, "
            "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/inlet-performance-model/",
            "NACA Report 1135, Equations, Tables, and Charts for Compressible Flow, "
            "https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf",
            "NASA Glenn, calorically perfect versus imperfect normal-shock context, "
            "https://www.grc.nasa.gov/WWW/BGH/normal.html",
            "Slater, J. W. (2012). Design and Analysis Tool for External-Compression Supersonic Inlets, "
            "NASA/TM-2012-217660, https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120013600.pdf",
            "Slater, J. W. (2024). SUPIN: A Tool for the Aerodynamic Design and Analysis of Supersonic "
            "Inlets, NASA/TM-20240008586, https://ntrs.nasa.gov/citations/20240008586",
            "NASA Glenn, Support of Flight Demonstration Projects - Inlets and Nozzles, "
            "current X-59 boundary evidence for extensive inlet/nozzle CFD, wind-tunnel "
            "testing, real-time displays, propulsion-system integration, and digital "
            "engine/aircraft architecture workflows not implemented by this surrogate, "
            "https://www.nasa.gov/glenn/research/inlets-and-nozzles/support-of-flight-demonstration-projects/",
            "NASA Glenn, Design and Analysis Software - Inlets and Nozzles, current SUPIN "
            "software boundary evidence for supersonic-inlet aerodynamic design/analysis "
            "and three-dimensional structured volume grids for higher-fidelity CFD, "
            "https://www.nasa.gov/glenn/research/inlets-and-nozzles/design-analysis-software/",
            "NASA/TM-20220010443, High Speed Inlet Distortion Test for the X-59 Low Boom "
            "Flight Demonstrator in the NASA Glenn 8- by 6-Foot Supersonic Wind Tunnel, "
            "engine-face distortion and inlet/engine operability boundary context, "
            "https://ntrs.nasa.gov/citations/20220010443",
        ],
        solver_library="scipy.optimize (θ–β–M equation numerical solve)",
        solver_details=(
            "brentq for weak oblique-shock angle; inviscid Rankine-Hugoniot "
            "shock relations; no viscous boundary-layer or shock/boundary-layer "
            "interaction model, bleed schedule, cowl-lip geometry, capture-area "
            "or spillage model, spillage-drag build-up, shock-train dynamics, "
            "inlet buzz, unstart dynamics, compressor-face distortion metrics, "
            "wind-tunnel engine-airflow sweep, pressure-rake recovery/distortion "
            "measurement, "
            "diffuser area schedule, inlet flow-rate accounting, inlet drag build-up, surface-grid or "
            "CFD-grid output, inlet-engine matching, variable geometry, "
            "strong-shock branch selection, back-pressure-driven shock-position solve, "
            "3-D sidewall/corner flow, "
            "X-59 inlet/nozzle CFD, wind-tunnel inlet-distortion testing, "
            "engine-face distortion or operability assessment, propulsion-system "
            "integration, real-time test displays, digital engine/aircraft "
            "architecture integration, "
            "calorically imperfect shock iteration, vibrational-energy model, "
            "real-gas chemistry, CFD calibration, wind-tunnel calibration, or "
            "SUPIN-style inlet design calculation"
            "; low-level shock helpers reject formula-domain violations "
            "(non-finite inputs, Ma <= 1, non-positive compression angle, "
            "or gamma <= 1) rather than extrapolating outside the declared "
            "supersonic constant-gamma relation; private physical-input evaluation "
            "rejects malformed arrays and Ma_in/delta1/delta2/gamma/alpha_off "
            "values outside the declared benchmark domain before oblique-shock "
            "cascade evaluation"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 1.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=[
            "smooth_unimodal",
            "threshold_jump",
            "strong_interaction",
            "discontinuous",
        ],
        effective_dim=5,
        important_variables=["Ma_in", "delta1", "delta2", "gamma", "alpha_off"],
        interaction_pairs=[
            ("Ma_in", "delta1"),
            ("Ma_in", "delta2"),
            ("Ma_in", "gamma"),
            ("Ma_in", "alpha_off"),
            ("delta1", "delta2"),
            ("delta1", "gamma"),
            ("delta1", "alpha_off"),
            ("delta2", "gamma"),
            ("delta2", "alpha_off"),
            ("gamma", "alpha_off"),
        ],
        has_failure_boundary=True,
        failure_condition="eta_pr < 0.80 or shock detachment (delta > max wedge angle)",
        sobol_S1={
            "Ma_in": 0.948,
            "delta1": 0.012,
            "delta2": 0.024,
            "gamma": 0.007,
            "alpha_off": 0.005,
        },
        sobol_ST={
            "Ma_in": 0.955,
            "delta1": 0.014,
            "delta2": 0.028,
            "gamma": 0.008,
            "alpha_off": 0.005,
        },
        sobol_computation="salib_saltelli_n2048_seed42_eta_pr_output",
        applicable_scenarios=["VS-C", "VS-F", "VS-G"],
        scenario_notes={
            "VS-C": "eta_pr has a flat optimum near the best wedge angles; surrogate accuracy test.",
            "VS-F": (
                "For the stored eta_pr Sobol metadata, Ma_in dominates. The full five-input "
                "importance list is retained because wedge angles, gamma, and off-design angle "
                "enter the attached/detached shock cascade and metadata interaction structure."
            ),
            "VS-G": "Locate the feasible domain where eta_pr > 0.95 in (Ma, δ1, δ2) space.",
        },
        recommended_n_train={"default": 60},
        n_saturation=300,
        noise_factor_candidates=["gamma", "alpha_off"],
        control_factor_candidates=["Ma_in", "delta1", "delta2"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        """Return validated physical inputs or reject private-domain violations."""
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "supersonic_intake_5d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        Ma_in = X_phys[:, 0]
        delta1 = X_phys[:, 1]
        delta2 = X_phys[:, 2]
        gamma = X_phys[:, 3]
        alpha_off = X_phys[:, 4]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(Ma_in < 2.0 - tol)
            or np.any(Ma_in > 5.0 + tol)
            or np.any(delta1 < 5.0 - tol)
            or np.any(delta1 > 15.0 + tol)
            or np.any(delta2 < 5.0 - tol)
            or np.any(delta2 > 20.0 + tol)
            or np.any(gamma < 1.30 - tol)
            or np.any(gamma > 1.40 + tol)
            or np.any(alpha_off < -3.0 - tol)
            or np.any(alpha_off > 3.0 + tol)
        ):
            raise ValueError(
                "supersonic_intake_5d physical inputs must stay inside the declared "
                "Ma_in/delta1/delta2/gamma/alpha_off benchmark domain before "
                "oblique-shock cascade evaluation"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                response = self._intake_perf(
                    Ma_in=float(X_phys[i, 0]),
                    delta1=float(X_phys[i, 1]),
                    delta2=float(X_phys[i, 2]),
                    gamma=float(X_phys[i, 3]),
                    alpha_off=float(X_phys[i, 4]),
                )
                if (
                    response.shape != (3,)
                    or not np.all(np.isfinite(response))
                    or not (0.0 <= response[0] <= 1.0)
                    or not (0.0 < response[1] <= 1.0)
                    or not (1.0 <= response[2] <= 200.0)
                ):
                    raise ValueError(
                        "non-finite or non-physical intake response "
                        f"eta_pr/Ma_exit/p_ratio={np.array2string(response, precision=6, separator=', ')}"
                    )
                Y[i] = response
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "supersonic_intake_5d oblique-shock intake solve failed "
                    f"for physical input {x_text}; refusing silent NaN output "
                    "because the formal benchmark response surface requires "
                    "finite shock-system recovery values."
                ) from None
        return Y

    @staticmethod
    def _oblique_shock(Ma1, delta_deg, gamma):
        """Oblique-shock calculation (weak-shock solution, θ–β–M relation).

        Returns (Ma2, p02/p01, p2/p1) or None (shock detachment).

        Algorithm: the θ–β–M curve on [β_min, π/2] rises then falls; a
        solution exists only when the peak ≥ tan(δ).  The weak-shock root
        lies in [β_min, β_peak]; found by coarse sweep for the first sign
        change, then brentq refinement.
        """
        if (
            not np.isfinite(Ma1)
            or not np.isfinite(delta_deg)
            or not np.isfinite(gamma)
            or Ma1 <= 1.0
            or delta_deg <= 0.0
            or gamma <= 1.0
        ):
            raise ValueError(
                "oblique-shock relation requires finite Ma1 > 1, "
                "compression deflection delta_deg > 0, and gamma > 1"
            )
        delta = np.deg2rad(delta_deg)
        g = gamma

        # θ–β–M equation (Anderson 1990, Eq. 4.17)
        def theta_beta_M(beta):
            sin2 = np.sin(beta) ** 2
            return 2.0 / np.tan(beta) * (Ma1**2 * sin2 - 1.0) / (
                Ma1**2 * (g + np.cos(2 * beta)) + 2.0
            ) - np.tan(delta)

        beta_min = np.arcsin(1.0 / Ma1) + 1e-6  # Mach angle (zero deflection)
        beta_max = np.pi / 2.0 - 1e-6  # normal shock

        # Coarse sweep: find first interval where tbm changes from negative to positive
        betas = np.linspace(beta_min, beta_max, 120)
        tvals = np.array([theta_beta_M(b) for b in betas])
        # If tbm < 0 throughout, δ exceeds the max deflection angle → shock detachment
        if np.max(tvals) < 0:
            return None

        # Take first sign change (weak-shock side)
        sign_change = np.where((tvals[:-1] < 0) & (tvals[1:] >= 0))[0]
        if len(sign_change) == 0:
            return None
        idx = sign_change[0]
        try:
            beta_sol = brentq(theta_beta_M, betas[idx], betas[idx + 1], xtol=1e-10)
        except Exception:
            return None

        Mn1 = Ma1 * np.sin(beta_sol)
        # Post-shock normal Mach number
        Mn2_sq = ((g - 1) * Mn1**2 + 2) / (2 * g * Mn1**2 - (g - 1))
        Mn2 = np.sqrt(max(Mn2_sq, 0.0))
        # Post-shock Mach number
        Ma2 = Mn2 / np.sin(max(beta_sol - delta, 1e-6))

        # Total-pressure ratio (isentropic relation)
        p02_p01 = (((g + 1) * Mn1**2) / ((g - 1) * Mn1**2 + 2)) ** (g / (g - 1)) * (
            (g + 1) / (2 * g * Mn1**2 - (g - 1))
        ) ** (1.0 / (g - 1))

        # Static-pressure ratio
        p2_p1 = (2 * g * Mn1**2 - (g - 1)) / (g + 1)

        return Ma2, p02_p01, p2_p1

    @staticmethod
    def _normal_shock(Ma1, gamma):
        """Normal shock (Ma1 > 1 → subsonic)."""
        if (
            not np.isfinite(Ma1)
            or not np.isfinite(gamma)
            or Ma1 <= 1.0
            or gamma <= 1.0
        ):
            raise ValueError("normal-shock relation requires finite Ma1 > 1 and gamma > 1")
        g = gamma
        Mn2_sq = ((g - 1) * Ma1**2 + 2) / (2 * g * Ma1**2 - (g - 1))
        Mn2 = np.sqrt(max(Mn2_sq, 0.0))
        p02_p01 = (((g + 1) * Ma1**2) / ((g - 1) * Ma1**2 + 2)) ** (g / (g - 1)) * (
            (g + 1) / (2 * g * Ma1**2 - (g - 1))
        ) ** (1.0 / (g - 1))
        p2_p1 = (2 * g * Ma1**2 - (g - 1)) / (g + 1)
        return Mn2, p02_p01, p2_p1

    def _intake_perf(self, Ma_in, delta1, delta2, gamma, alpha_off):
        """Two-stage oblique shock + terminal normal shock intake performance."""
        if not np.all(np.isfinite([Ma_in, delta1, delta2, gamma, alpha_off])):
            raise ValueError("intake inputs must be finite")
        if Ma_in <= 1.0 or gamma <= 1.0:
            raise ValueError("intake shock surrogate requires Ma_in > 1 and gamma > 1")
        g = gamma
        # AoA offset treated as effective correction to delta1
        delta1_eff = delta1 + alpha_off
        if delta1_eff <= 0.0 or delta2 <= 0.0:
            raise ValueError("intake compression-wedge deflections must be positive")

        # 1st oblique shock
        res1 = self._oblique_shock(Ma_in, delta1_eff, g)
        if res1 is None:
            return np.array([0.3, 0.5, 2.0])  # detached shock, very poor performance
        Ma2, p02_p01_1, p2_p1_1 = res1

        # 2nd oblique shock (wedge angle delta2 relative to post-1st-shock flow)
        delta2_eff = max(delta2, 1.0)
        if Ma2 <= 1.0:
            return np.array([0.5, 0.8, p2_p1_1])
        res2 = self._oblique_shock(Ma2, delta2_eff, g)
        if res2 is None:
            # 2nd-stage detachment → degrades to single-stage + normal shock
            Ma3 = Ma2
            p02_p01_2 = 1.0
            p2_p1_2 = 1.0
        else:
            Ma3, p02_p01_2, p2_p1_2 = res2

        # Terminal normal shock
        if Ma3 <= 1.0:
            Ma_exit = Ma3
            p02_p01_3 = 1.0
            p2_p1_3 = 1.0
        else:
            Ma_exit, p02_p01_3, p2_p1_3 = self._normal_shock(Ma3, g)

        eta_pr = p02_p01_1 * p02_p01_2 * p02_p01_3
        p_total_ratio = p2_p1_1 * p2_p1_2 * p2_p1_3

        return np.array([float(eta_pr), float(Ma_exit), float(p_total_ratio)])
