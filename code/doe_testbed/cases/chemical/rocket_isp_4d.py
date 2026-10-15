"""
Case C03: Rocket engine specific impulse (thermodynamic calculation)
====================================================================

Engineering background
----------------------
Specific impulse (Isp) is the core performance metric of a rocket engine,
defined as the impulse per unit weight of propellant (unit: s):
  Isp = F / (ṁ · g₀) = V_e / g₀

Isp is determined by propellant combination and combustion conditions.
High-fidelity design normally uses chemical-equilibrium thermodynamic
calculations such as NASA CEA:
  · High Isp: LOX/LH₂ (450 s) > LOX/kerosene (340 s) > solid (250–300 s)
  · Isp increases slightly with chamber pressure (more complete expansion)
  · Mixture ratio (O/F) has an optimum (slightly fuel-rich)

This benchmark case uses a simplified LOX/LH2 thermochemical surrogate plus
isentropic nozzle equations to study the effect of combustion conditions on
Isp. It does not call NASA CEA and should not be used as a CEA-level
thermochemistry model.

Physics model
-------------
Adiabatic combustion chamber + isentropic nozzle expansion, represented by a
low-order thermochemical surrogate:

1. Given chamber temperature T_c, chamber pressure P_c, nozzle exit
   pressure P_e, and fuel/oxidiser mixture ratio MR.

2. Empirical LOX/LH2 correlations approximate chamber temperature, effective
   molecular weight, and γ; Gibbs-free-energy minimisation is not performed.

3. Nozzle performance:
   c* = characteristic velocity = P_c·A*/ṁ
   C_F = thrust coefficient (isentropic expansion + exit pressure)
   Isp = c* · C_F / g₀

The nozzle calculation follows ideal isentropic area-Mach and thrust-
coefficient relations.

Input variables
---------------
MR          Mixture ratio (O/F mass ratio) [2.0, 8.0]   -    Oxidiser-to-fuel ratio
P_c         Chamber pressure               [5, 200]     bar  Higher pressure → more efficient (log)
epsilon     Nozzle area ratio (expansion)   [5, 100]     -    Determines expansion degree (log)
T_initial   Propellant initial temperature  [80, 300]    K    LH₂ ∼20 K / LOX ∼90 K

Output responses
----------------
Isp         Vacuum specific impulse         s      Primary output (key performance metric)
c_star      Characteristic velocity         m/s    Combustion-efficiency indicator
T_chamber   Chamber temperature             K      Thermodynamic-limit indicator

DOE validation value
--------------------
VS-C (surrogate accuracy):
    Isp vs MR has a flat optimum near the ideal mixture ratio; descent
    is asymmetric (fuel-rich side drops slower) — a surrogate test.

References
----------
1. Sutton, G. P. & Biblarz, O. (2010). Rocket Propulsion Elements, 8th ed. Wiley.
2. McBride, B. J. & Gordon, S. (1996). Computer Program for Calculation of
   Complex Chemical Equilibrium Compositions and Applications. NASA RP-1311.
3. NASA Glenn Research Center. Chemical Equilibrium with Applications (CEA)
   history and documentation trail.
4. NASA Glenn Research Center. Isentropic flow and nozzle-design equations.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class RocketIsp4D(BaseCase):
    """Rocket engine specific impulse (LOX/LH₂ propellant, 4D → 3D).

    Simplified LOX/LH2 thermochemical surrogate plus ideal nozzle equations.
    This is not a NASA CEA replacement.
    """

    # LOX/LH2 thermochemical data
    _G0 = 9.80665  # m/s², standard gravity

    meta = CaseMeta(
        case_id="rocket_isp_4d",
        name="Rocket engine Isp (LOX/LH₂ thermodynamic calculation)",
        version="1.0.0",
        domain="chemical_process",
        tags=["rocket", "propulsion", "specific_impulse", "LOX_LH2", "thermodynamics", "4d"],
        dim_input=4,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="MR",
                symbol="MR",
                description="Mixture ratio (oxidiser/fuel mass ratio)",
                unit="-",
                lower=2.0,
                upper=8.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="P_c",
                symbol="P_c",
                description="Chamber pressure",
                unit="bar",
                lower=5.0,
                upper=200.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="epsilon",
                symbol=r"\varepsilon",
                description="Nozzle area expansion ratio (exit area / throat area)",
                unit="-",
                lower=5.0,
                upper=100.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="T_initial",
                symbol="T_0",
                description="Propellant initial temperature (affects enthalpy)",
                unit="K",
                lower=80.0,
                upper=300.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="Isp",
                symbol="I_{sp}",
                description="Vacuum specific impulse",
                unit="s",
                is_primary=True,
                expected_range=(250.0, 480.0),
            ),
            OutputVariable(
                name="c_star",
                symbol="c^*",
                description="Characteristic velocity (combustion-efficiency indicator)",
                unit="m/s",
                is_primary=False,
                expected_range=(1500.0, 2500.0),
            ),
            OutputVariable(
                name="T_chamber",
                symbol="T_c",
                description="Chamber adiabatic flame temperature",
                unit="K",
                is_primary=False,
                expected_range=(2000.0, 3700.0),
            ),
        ],
        governing_equation=(
            "LOX/LH2 thermochemical surrogate + ideal nozzle equations:\n"
            "  Adiabatic chamber: empirical MR/T0/P_c correlations → T_c, γ_eff, M_eff\n"
            "  Isentropic nozzle: c*=√(γRT_c/M)/(γ·(2/(γ+1))^((γ+1)/(2γ−2)))\n"
            "  Isp = c*·Cf/g0; Cf determined by area ratio and pressure ratio\n"
            "  Scope: trend/identity surrogate, not NASA CEA Gibbs-minimisation thermochemistry, "
            "finite-rate chemistry, equilibrium/frozen expansion, or engine-cycle design"
        ),
        pde_type="algebraic",
        spatial_dim=0,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Sutton, G. P. & Biblarz, O. (2010). Rocket Propulsion Elements, 8th ed. Wiley.",
            "McBride, B. J. & Gordon, S. (1996). NASA RP-1311.",
            "NASA Glenn Research Center. Chemical Equilibrium with Applications (CEA) "
            "history: https://www.nasa.gov/glenn/research/chemical-equilibrium-with-applications/cea-history/",
            "NASA RP-1311 Part II PDF: https://ntrs.nasa.gov/api/citations/19960044559/downloads/19960044559.pdf",
            "NASA Glenn isentropic-flow equations: https://www.grc.nasa.gov/www/k-12/airplane/isentrop.html",
            "NASA Glenn nozzle-design overview: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/nozzle-design/",
        ],
        solver_library="numpy (LOX/LH2 thermochemical surrogate + isentropic nozzle)",
        solver_details=(
            "Empirical chamber-temperature/effective-gamma/effective-molecular-weight "
            "correlations plus area-Mach nozzle solve; no NASA CEA, no Gibbs-free-energy "
            "minimisation, no finite-rate chemistry, no dissociation/equilibrium/frozen "
            "expansion chemistry, no nozzle losses, no two-phase flow, no film cooling, no "
            "injector/chamber/engine-cycle design, and no claimed case-level CEA accuracy. "
            "Private physical-input calls are rejected before the surrogate/nozzle solve if "
            "they are malformed, non-finite, or outside the declared benchmark domain; "
            "outputs are rejected if they fall outside the declared expected physical ranges."
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 1.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_unimodal"],
        effective_dim=3,
        important_variables=["MR", "P_c", "epsilon"],
        interaction_pairs=[("MR", "P_c")],
        has_failure_boundary=False,
        sobol_S1={"MR": 0.279, "P_c": 0.001, "epsilon": 0.622, "T_initial": 0.068},
        sobol_ST={"MR": 0.306, "P_c": 0.001, "epsilon": 0.623, "T_initial": 0.095},
        sobol_computation="salib_saltelli_n2048_seed42_Isp_output",
        applicable_scenarios=["VS-C", "VS-D"],
        scenario_notes={
            "VS-C": "Isp has a flat optimum near MR ≈ 5.5 (LOX/LH2).",
        },
        recommended_n_train={"default": 30},
        n_saturation=150,
        noise_factor_candidates=["T_initial"],
        control_factor_candidates=["MR", "P_c", "epsilon"],
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> None:
        """Reject malformed or out-of-domain private physical inputs."""
        X = np.asarray(X_phys, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.meta.dim_input:
            raise ValueError(
                "rocket_isp_4d expects a physical input array with "
                f"shape (n, {self.meta.dim_input}); got {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "rocket_isp_4d received non-finite physical inputs; refusing "
                "to extrapolate the LOX/LH2 thermochemical surrogate."
            )

        tol = 1e-10
        for j, var in enumerate(self.meta.input_variables):
            lower = float(var.lower)
            upper = float(var.upper)
            scale = max(abs(lower), abs(upper), 1.0)
            bad = (X[:, j] < lower - tol * scale) | (X[:, j] > upper + tol * scale)
            if np.any(bad):
                raise ValueError(
                    "rocket_isp_4d physical input "
                    f"{var.name!r} is outside the declared benchmark domain "
                    f"[{lower}, {upper}]."
                )

    def _validate_response_contract(self, Y: np.ndarray, n: int) -> None:
        """Reject malformed or out-of-envelope rocket-performance outputs."""
        if (
            Y.shape != (n, self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y[:, 0] <= 0.0)
            or np.any(Y[:, 1] <= 0.0)
            or np.any(Y[:, 2] <= 0.0)
        ):
            raise RuntimeError(
                "rocket_isp_4d produced a malformed, non-finite, or "
                "non-physical rocket-performance response; refusing silent "
                "archive rows"
            )

        for j, output in enumerate(self.meta.output_variables):
            if output.expected_range is None:
                continue
            lower, upper = map(float, output.expected_range)
            scale = max(abs(lower), abs(upper), 1.0)
            tol = 1e-9 * scale
            bad = (Y[:, j] < lower - tol) | (Y[:, j] > upper + tol)
            if np.any(bad):
                observed = Y[bad, j]
                raise RuntimeError(
                    "rocket_isp_4d produced output "
                    f"{output.name!r} outside the declared expected response range "
                    f"[{lower}, {upper}]; observed min={float(np.min(observed))}, "
                    f"max={float(np.max(observed))}. Refusing silent archive rows."
                )

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        self._validate_physical_inputs(X_phys)
        X_phys = np.asarray(X_phys, dtype=float)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                Y[i] = self._isp_calc(
                    MR=float(X_phys[i, 0]),
                    P_c_bar=float(X_phys[i, 1]),
                    epsilon=float(X_phys[i, 2]),
                    T0=float(X_phys[i, 3]),
                )
            except Exception as exc:
                raise RuntimeError(
                    "rocket_isp_4d refusing silent NaN after rocket Isp "
                    f"evaluation failure for MR={X_phys[i, 0]!r}, "
                    f"P_c_bar={X_phys[i, 1]!r}, epsilon={X_phys[i, 2]!r}, "
                    f"T_initial={X_phys[i, 3]!r}"
                ) from exc
        self._validate_response_contract(Y, n)
        return Y

    @staticmethod
    def _isp_calc(MR, P_c_bar, epsilon, T0):
        """LOX/LH₂ simplified thermodynamic calculation (Sutton & Biblarz method).

        Assumes complete combustion; products are H₂O + excess H₂ (fuel-rich)
        or H₂O + O₂ (fuel-lean).  Uses NASA polynomial approximations for
        thermodynamic properties.

        LOX/LH₂ stoichiometric mixture ratio: MR_stoic = 8.0 (mass).
        Actual optimum MR ≈ 5.5–6.0 (slightly fuel-rich to maximise Isp).
        """
        if (
            not np.all(np.isfinite([MR, P_c_bar, epsilon, T0]))
            or MR <= 0.0
            or P_c_bar <= 0.0
            or epsilon < 1.0
            or T0 <= 0.0
        ):
            raise ValueError(
                "invalid rocket_isp_4d input outside the declared physical contract"
            )
        R_univ = 8314.0  # J/(kmol·K)
        g0 = 9.80665

        # Adiabatic flame temperature estimate (LOX/LH₂ empirical correlation)
        # Stoichiometric MR_s = 8 (mass O2/H2)
        # Adjustment based on equivalence ratio
        phi = 8.0 / max(MR, 0.1)  # equivalence ratio (fuel-rich phi > 1)
        phi = np.clip(phi, 0.2, 3.0)

        # Adiabatic flame temperature (approximate formula, interpolated from Sutton Table 5-5)
        # H2/O2 T_ad in the MR = 5–8 range is 3200–3500 K.  The
        # performance optimum for LOX/LH2 is fuel-rich rather than
        # stoichiometric because excess hydrogen lowers the exhaust
        # molecular weight; use a broad peak near MR ~= 5.8.
        T_ad_ref = 3500.0
        phi_opt = 8.0 / 5.8
        mixture_factor = np.exp(-0.6 * (phi - phi_opt) ** 2)
        # Warmer propellants carry more sensible enthalpy into the chamber.
        enthalpy_factor = 1.0 + 0.05 * (T0 - 90.0) / 210.0
        eta_T = mixture_factor * enthalpy_factor
        T_c = T_ad_ref * np.clip(eta_T, 0.58, 1.03)
        # Minor pressure effect on T_c
        T_c *= 1.0 + 0.01 * np.log10(P_c_bar / 10.0)
        T_c = float(np.clip(T_c, 2000.0, 3700.0))

        # Effective molecular weight (H2O + H2 mixture approximation)
        # M_H2O = 18, M_H2 = 2
        # Stoichiometric: all H2O; fuel-rich: contains H2
        excess_H2_frac = max(0, 1.0 - 1.0 / phi)  # excess H2 fraction proxy
        # The hydrogen-rich plume contains H2O, H2, OH, and dissociation
        # products.  A full CEA calculation is outside this L0 case, so use a
        # moderated composition factor rather than treating all excess fuel as
        # free H2; otherwise the Isp optimum shifts to unrealistically low MR.
        h2_composition_factor = 0.3
        M_eff = (
            18.0 * (1 - excess_H2_frac * h2_composition_factor)
            + 2.0 * excess_H2_frac * h2_composition_factor
        )
        M_eff = float(np.clip(M_eff, 8.0, 18.0))

        # Specific-heat ratio (approximate)
        gamma = 1.15 + 0.05 * excess_H2_frac
        gamma = float(np.clip(gamma, 1.1, 1.35))

        # Characteristic velocity c* (isentropic, calorically-perfect-gas approximation)
        R_eff = R_univ / M_eff  # J/(kg·K)
        c_star = np.sqrt(gamma * R_eff * T_c) / (
            gamma * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * gamma - 2))
        )
        c_star = float(np.clip(c_star, 1000.0, 2600.0))

        # ── Isentropic nozzle expansion: area ratio → exit Mach ───────────────────────
        # A/A* = (1/Ma)·((2/(γ+1))·(1+(γ-1)/2·Ma²))^((γ+1)/(2(γ-1)))
        from scipy.optimize import brentq

        def area_ratio_eq(Ma):
            return (
                ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * Ma**2))
                ** ((gamma + 1) / (2 * (gamma - 1)))
                / Ma
            ) - epsilon

        try:
            Ma_e = brentq(area_ratio_eq, 1.000001, 50.0, xtol=1e-6)
        except Exception:
            raise ValueError(
                "rocket_isp_4d area-Mach solve failed for the declared nozzle "
                f"contract: epsilon={epsilon!r}, gamma={gamma!r}"
            )

        # ── Exit pressure ratio (dimensionless) r_p = P_e/P_c (isentropic) ──────────────
        r_p = (1.0 + (gamma - 1) / 2 * Ma_e**2) ** (-gamma / (gamma - 1))

        # ── Vacuum thrust coefficient (Sutton & Biblarz 2010, Eq. 3-30) ────────────
        #   C_F,vac = sqrt( 2γ²/(γ-1) · (2/(γ+1))^((γ+1)/(γ-1)) · (1 - r_p^((γ-1)/γ)) )
        #           + ε · r_p
        inner = (
            2.0
            * gamma**2
            / (gamma - 1.0)
            * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
            * (1.0 - r_p ** ((gamma - 1.0) / gamma))
        )
        Cf_vac = np.sqrt(max(inner, 0.0)) + epsilon * r_p

        # ── Specific impulse ───────────────────────────────────────────────────────
        Isp = c_star * Cf_vac / g0
        response = np.array([float(Isp), float(c_star), float(T_c)])
        if (
            not np.all(np.isfinite(response))
            or response[0] <= 0.0
            or response[1] <= 0.0
            or response[2] <= 0.0
        ):
            raise ValueError("non-physical rocket-performance response")

        return response
