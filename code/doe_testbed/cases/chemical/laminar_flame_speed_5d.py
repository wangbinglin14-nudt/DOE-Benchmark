"""
Case C02: Laminar premixed burning velocity (Gulder low-fidelity + Cantera high-fidelity)
=========================================================================================

Engineering background
----------------------
Laminar burning velocity (SL) is a fundamental combustion-design parameter:
  · Engine combustion-chamber design (in-cylinder ignition and propagation speed)
  · Gas-turbine premixed-flame stability (flashback and blow-off prevention)
  · Explosion safety assessment (hazard level of flammable mixtures)

SL is a strongly nonlinear function of mixture composition, pressure, and
temperature:
  · Equivalence ratio φ ≈ 1.0–1.1 gives maximum SL; too rich or too lean
    reduces it
  · Higher pressure generally lowers SL (P^n, n < 0)
  · Higher initial temperature raises SL (Arrhenius-type)

This case exposes a multi-fidelity interface.  The formal default fidelity is
the fast Gülder-style semi-empirical surrogate; optional high fidelity calls
Cantera FreeFlame with GRI-Mech 3.0 when Cantera is installed.  Headline
benchmark rows therefore use the low-fidelity response unless high fidelity
is explicitly requested.

Physics model
-------------
High-fidelity path: 1-D premixed planar flame equations (conservation +
chemical source):
  ρu·du/dx = −dP/dx  (momentum)
  ρu·dY_k/dx = −d(ρV_kY_k)/dx + ω_k·W_k  (species)
  ρu·c_p·dT/dx = dp/dx − Σ h_k·ω_k·W_k + d(λdT/dx)/dx  (energy)

Cantera solves this 1-D BVP with detailed chemical kinetics.  The default
path instead evaluates the Gülder formula
SL = W·φ^η·exp[-ξ(φ−σ)^2]·(T/T_ref)^α·(P/P_ref)^β with empirical blending and
CO2 dilution factors; it is a trend surrogate, not detailed chemistry.

Input variables
---------------
phi         Equivalence ratio          [0.6, 1.4]  -    Combustion stoichiometry
T_init      Unburnt mixture temperature [300, 500]  K    Preheating effect
P_init      Combustion pressure        [0.5, 20]   atm  Operating pressure
fuel_frac   CH₄ fraction (CH₄/H₂ blend) [0, 1]    -    Fuel type (0 = pure H₂, 1 = pure CH₄)
diluent_frac CO₂ dilution ratio        [0, 0.2]    -    Exhaust-gas recirculation or CO₂ blending

Output responses
----------------
SL          Laminar burning velocity   cm/s   Primary output
T_ad        Adiabatic flame temperature K     Thermodynamic indicator
sigma_exp   Expansion ratio (ρ_u/ρ_b)  -     Combustion-rate indicator

DOE validation value
--------------------
VS-C (surrogate accuracy):
    SL peaks near φ = 1.0–1.1 and drops steeply for φ < 0.7 or φ > 1.3
    — a classic test of surrogate accuracy near an extremum.

VS-G (sequential optimisation):
    Finding the maximum-SL parameter point (φ, T_init, P optimum) is a
    sequential-optimisation test target.

Note: Cantera evaluations take ∼3 s each (L3 high-cost case); multi-
fidelity is needed (low-fidelity: simplified formula; high-fidelity:
Cantera).

References
----------
1. Turns, S. R. (2012). An Introduction to Combustion, 3rd ed. McGraw-Hill.
2. Law, C. K. (2006). Combustion Physics. Cambridge.
3. Smith, G. P. et al. (2000). GRI-Mech 3.0. http://combustion.berkeley.edu/gri_mech/
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class LaminarFlameSpeed5D(BaseCase):
    """Laminar burning velocity (5D → 3D).

    Default/low mode uses the Gülder semi-empirical formula. Optional high
    mode uses Cantera FreeFlame when available.
    """

    meta = CaseMeta(
        case_id="laminar_flame_speed_5d",
        name="Laminar premixed burning velocity (Gülder default + optional Cantera)",
        version="1.0.0",
        domain="chemical_process",
        tags=[
            "combustion",
            "flame_speed",
            "cantera",
            "premixed",
            "5d",
            "high_cost",
            "multifidelity",
        ],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="phi",
                symbol=r"\phi",
                description="Equivalence ratio (1.0 = stoichiometric)",
                unit="-",
                lower=0.6,
                upper=1.4,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="T_init",
                symbol="T_u",
                description="Unburnt mixture initial temperature",
                unit="K",
                lower=300.0,
                upper=500.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="P_atm",
                symbol="P",
                description="Combustion pressure",
                unit="atm",
                lower=0.5,
                upper=20.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="fuel_frac",
                symbol=r"\chi_{CH4}",
                description="CH₄ volume fraction (remainder H₂); 0 = pure H₂, 1 = pure CH₄",
                unit="-",
                lower=0.0,
                upper=1.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="diluent",
                symbol=r"\chi_{CO2}",
                description="CO₂ dilution ratio (exhaust-gas recirculation)",
                unit="-",
                lower=0.0,
                upper=0.2,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="SL",
                symbol="S_L",
                description="Laminar burning velocity",
                unit="cm/s",
                is_primary=True,
                expected_range=(1.0, 500.0),
                failure_threshold=5.0,
                failure_direction="below",
            ),
            OutputVariable(
                name="T_ad",
                symbol="T_{ad}",
                description="Adiabatic flame temperature",
                unit="K",
                is_primary=False,
                expected_range=(1000.0, 2800.0),
            ),
            OutputVariable(
                name="sigma_exp",
                symbol=r"\sigma",
                description="Expansion ratio (unburnt/burnt density ratio)",
                unit="-",
                is_primary=False,
                expected_range=(3.0, 9.0),
            ),
        ],
        governing_equation=(
            "1-D premixed planar flame (BVP):\n"
            "  ρu·dY_k/dx = −d(ρV_kY_k)/dx + ω_k·W_k\n"
            "  ρu·c_p·dT/dx = d(λdT/dx)/dx − Σh_k·ω_k·W_k\n"
            "Low-fidelity: Gülder formula SL=W·φⁿ·exp(−ξ·(φ−σ)²)·(T/T_ref)^α·(P/P_ref)^β"
        ),
        pde_type="ode_system",
        spatial_dim=1,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Turns, S. R. (2012). An Introduction to Combustion, 3rd ed. McGraw-Hill.",
            "Law, C. K. (2006). Combustion Physics. Cambridge.",
            "Smith, G. P. et al. (2000). GRI-Mech 3.0. UC Berkeley.",
        ],
        solver_library="cantera (high-fidelity) / Gülder formula (low-fidelity)",
        solver_details=(
            "Default/low fidelity: Gülder semi-empirical trend surrogate, <1 ms/eval; "
            "optional high fidelity: Cantera FreeFlame, GRI-Mech 3.0, ~3 s/eval. "
            "Formal headline sweeps use default=low unless high fidelity is explicitly requested. "
            "Private physical-input calls are rejected before either solver path if they are "
            "malformed, non-finite, or outside the declared benchmark domain."
        ),
        fidelity_levels=["default", "low", "high"],
        # 'default' aliases to 'low' (Gülder formula) — see _evaluate_core.
        # Duplicating the key lets registry.list_cases(max_eval_time_ms=...),
        # CLI `coverage`, and check_implementation's timing limit pick up the
        # correct figure without a special-case lookup.
        typical_eval_time_ms={"default": 0.5, "low": 0.5, "high": 3000.0},
        cost_level="L3",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=True,
        response_topology=["smooth_unimodal", "strong_interaction"],
        effective_dim=5,
        important_variables=["P_atm", "fuel_frac", "T_init", "phi", "diluent"],
        interaction_pairs=[("phi", "T_init"), ("phi", "P_atm"), ("fuel_frac", "phi")],
        has_failure_boundary=True,
        failure_condition="SL < 5 cm/s (flame cannot propagate stably; near extinction limit)",
        sobol_S1={
            "phi": 0.078,
            "T_init": 0.152,
            "P_atm": 0.402,
            "fuel_frac": 0.165,
            "diluent": 0.069,
        },
        sobol_ST={
            "phi": 0.113,
            "T_init": 0.195,
            "P_atm": 0.495,
            "fuel_frac": 0.216,
            "diluent": 0.103,
        },
        sobol_computation="salib_saltelli_n1024_seed42_SL_output",
        applicable_scenarios=["VS-C", "VS-G"],
        scenario_notes={
            "VS-C": "SL peaks at φ = 1.0–1.1 with steep drop on both sides; surrogate-accuracy challenge.",
            "VS-G": "Sequential optimisation to find the maximum-SL parameter point.",
        },
        recommended_n_train={"low": 100, "high": 40},
        n_saturation=400,
        noise_factor_candidates=["diluent"],
        control_factor_candidates=["phi", "T_init", "P_atm", "fuel_frac"],
        reference_solution_available=True,
        reference_grid_size=1000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> None:
        """Reject malformed or out-of-domain private physical inputs."""
        X = np.asarray(X_phys, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.meta.dim_input:
            raise ValueError(
                "laminar_flame_speed_5d expects a physical input array with "
                f"shape (n, {self.meta.dim_input}); got {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "laminar_flame_speed_5d received non-finite physical inputs; "
                "refusing to extrapolate the premixed-flame surrogate."
            )

        tol = 1e-10
        for j, var in enumerate(self.meta.input_variables):
            lower = float(var.lower)
            upper = float(var.upper)
            scale = max(abs(lower), abs(upper), 1.0)
            bad = (X[:, j] < lower - tol * scale) | (X[:, j] > upper + tol * scale)
            if np.any(bad):
                raise ValueError(
                    "laminar_flame_speed_5d physical input "
                    f"{var.name!r} is outside the declared benchmark domain "
                    f"[{lower}, {upper}]."
                )

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        # 'default' is equivalent to 'low' (Gülder formula)
        if fidelity == "default":
            fidelity = "low"
        self._validate_physical_inputs(X_phys)
        X_phys = np.asarray(X_phys, dtype=float)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            phi = float(X_phys[i, 0])
            T_init = float(X_phys[i, 1])
            P_atm = float(X_phys[i, 2])
            fuel_frac = float(X_phys[i, 3])  # CH4 fraction
            diluent = float(X_phys[i, 4])  # CO2 dilution

            if fidelity == "high":
                try:
                    Y[i] = self._cantera_flame(phi, T_init, P_atm, fuel_frac, diluent)
                except Exception:
                    Y[i] = self._gulder_formula(phi, T_init, P_atm, fuel_frac, diluent)
            else:
                try:
                    Y[i] = self._gulder_formula(phi, T_init, P_atm, fuel_frac, diluent)
                except Exception as exc:
                    raise RuntimeError(
                        "laminar_flame_speed_5d refusing silent NaN after "
                        "low-fidelity flame-speed evaluation failure for "
                        f"phi={phi!r}, T_init={T_init!r}, P_atm={P_atm!r}, "
                        f"fuel_frac={fuel_frac!r}, diluent={diluent!r}"
                    ) from exc
        if (
            Y.shape != (n, self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y[:, 0] <= 0.0)
            or np.any(Y[:, 1] <= 0.0)
            or np.any(Y[:, 2] <= 0.0)
        ):
            raise RuntimeError(
                "laminar_flame_speed_5d produced a malformed, non-finite, or "
                "non-physical flame response; refusing silent archive rows"
            )
        return Y

    @staticmethod
    def _gulder_formula(phi, T_init, P_atm, fuel_frac, diluent):
        """Gülder (1984) semi-empirical formula for laminar burning velocity.

        CH₄ params: W=0.422, η=0.15, ξ=5.18, σ=1.075, α=2.105, β=−0.504
        H₂  params: W=2.30,  η=0.0,  ξ=2.00, σ=1.0,   α=1.75,  β=−0.20
        Blend: linear interpolation (approximate).
        """
        if (
            not np.all(np.isfinite([phi, T_init, P_atm, fuel_frac, diluent]))
            or phi <= 0.0
            or T_init <= 0.0
            or P_atm <= 0.0
            or fuel_frac < 0.0
            or fuel_frac > 1.0
            or diluent < 0.0
            or diluent >= 1.0
        ):
            raise ValueError(
                "invalid laminar_flame_speed_5d input outside the declared "
                "premixed-flame surrogate contract"
            )
        # CH4 Gülder parameters
        W_CH4, eta_CH4, xi_CH4 = 0.422, 0.15, 5.18
        sigma_CH4, alpha_CH4, beta_CH4 = 1.075, 2.105, -0.504

        # H2 parameters tuned to the 200-300 cm/s pure-H2 STP peak range
        # reported in standard premixed-flame references.
        W_H2, eta_H2, xi_H2 = 2.30, 0.0, 2.00
        sigma_H2, alpha_H2, beta_H2 = 1.0, 1.75, -0.20

        # Blended parameters (linear interpolation)
        W = fuel_frac * W_CH4 + (1 - fuel_frac) * W_H2
        eta = fuel_frac * eta_CH4 + (1 - fuel_frac) * eta_H2
        xi = fuel_frac * xi_CH4 + (1 - fuel_frac) * xi_H2
        sig = fuel_frac * sigma_CH4 + (1 - fuel_frac) * sigma_H2
        alp = fuel_frac * alpha_CH4 + (1 - fuel_frac) * alpha_H2
        bet = fuel_frac * beta_CH4 + (1 - fuel_frac) * beta_H2

        # Reference conditions T_ref = 298 K, P_ref = 1 atm
        T_ref = 298.0
        P_ref = 1.0

        SL_0 = W * phi**eta * np.exp(-xi * (phi - sig) ** 2)
        SL = SL_0 * (T_init / T_ref) ** alp * (P_atm / P_ref) ** bet
        SL *= 1 - 2.5 * diluent  # CO2 dilution effect (approximate)
        SL = float(SL * 100)  # m/s → cm/s
        if not np.isfinite(SL) or SL <= 0.0:
            raise ValueError("non-physical low-fidelity laminar flame speed")

        # Adiabatic flame temperature (empirical estimate, low-fidelity only).
        # A smooth quadratic shape avoids the nonphysical kink produced by the
        # earlier min(phi, 1)/phi scaling and mimics high-temperature dissociation.
        phi_core = min(phi, 1.4)
        if phi_core <= 1.05:
            scale_factor = 1.0 - 0.4 * (phi_core - 1.05) ** 2
        else:
            scale_factor = 1.0 - 0.6 * (phi_core - 1.05) ** 2
        # Reference adiabatic flame temperatures for near-stoichiometric
        # methane-air and hydrogen-air mixtures at 298 K, 1 atm are treated
        # as absolute temperatures, not as temperature rises.  Convert them
        # to a heat-release temperature rise before adding the actual
        # unburnt-mixture temperature.
        T_ref = 298.0
        T_ad_ref = fuel_frac * 2220.0 + (1.0 - fuel_frac) * 2380.0
        delta_T_ref = max(T_ad_ref - T_ref, 0.0)
        T_ad = T_init + delta_T_ref * scale_factor * (1.0 - 0.35 * diluent)
        T_ad = float(T_ad)
        if not np.isfinite(T_ad) or T_ad <= T_init:
            raise ValueError("non-physical low-fidelity adiabatic flame temperature")

        # Expansion ratio (ideal-gas isobaric): sigma = (T_b/T_u) * (W_u/W_b).
        W_ratio = 1.0 + 0.05 * (phi - 1.0) if phi > 1.0 else 1.0 - 0.03 * (1.0 - phi)
        sigma_exp = (T_ad / T_init) * W_ratio
        sigma_exp = float(sigma_exp)
        if not np.isfinite(sigma_exp) or sigma_exp <= 1.0:
            raise ValueError("non-physical low-fidelity expansion ratio")

        return np.array([SL, T_ad, sigma_exp])

    @staticmethod
    def _apply_co2_dilution(mole_fractions, diluent):
        """Add CO2 while preserving the premixed fuel/oxidizer equivalence ratio."""
        diluent = float(np.clip(diluent, 0.0, 0.95))
        x_orig = {k: float(v) for k, v in mole_fractions.items() if float(v) > 0.0}
        scale = 1.0 - diluent
        x_new = {k: v * scale for k, v in x_orig.items()}
        x_new["CO2"] = x_new.get("CO2", 0.0) + diluent
        total = sum(x_new.values())
        return {k: v / total for k, v in x_new.items()}

    @staticmethod
    def _cantera_flame(phi, T_init, P_atm, fuel_frac, diluent):
        """Compute laminar burning velocity using Cantera (high-fidelity)."""
        import cantera as ct

        P = P_atm * ct.one_atm

        # Fuel mixture
        fuel_str = f"CH4:{fuel_frac}, H2:{1-fuel_frac}"

        gas = ct.Solution("gri30.yaml")
        gas.TP = T_init, P
        gas.set_equivalence_ratio(phi, fuel_str, "O2:1, N2:3.76")

        # Add CO2 dilution without changing the fuel/oxidizer equivalence ratio.
        if diluent > 0.0:
            gas.TPX = T_init, P, LaminarFlameSpeed5D._apply_co2_dilution(gas.mole_fraction_dict(), diluent)

        domain_width = 0.05 / np.sqrt(P_atm)
        f = ct.FreeFlame(gas, width=domain_width)
        f.set_refine_criteria(ratio=3, slope=0.08, curve=0.12)
        try:
            f.solve(loglevel=0, auto=True)
        except ct.CanteraError:
            f.set_refine_criteria(ratio=4, slope=0.15, curve=0.25)
            f.solve(loglevel=0, auto=True)

        SL = float(f.velocity[0] * 100)  # m/s → cm/s
        T_ad = float(f.T[-1])
        rho_u = float(f.density[0])
        rho_b = float(f.density[-1])
        sigma_exp = rho_u / rho_b

        return np.array([SL, T_ad, sigma_exp])
