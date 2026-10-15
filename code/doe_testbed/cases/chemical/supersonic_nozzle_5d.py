"""
Case C05: Supersonic nozzle internal flow (1-D isentropic + normal shock)
=========================================================================

Engineering background
----------------------
Converging–diverging nozzles (Laval nozzles) are the core component of
rocket engines, supersonic wind tunnels, and industrial jet devices.
Nozzle aerodynamic performance directly determines:
  · Rocket thrust efficiency (degree of isentropic expansion)
  · Wind-tunnel test-section uniformity
  · Ramjet inlet efficiency

Three flow regimes can occur inside the nozzle:
  1. Fully subsonic flow (low pressure ratio)
  2. Supersonic design condition (no shock)
  3. Normal shock inside the nozzle (over-expansion / shock location)

This case computes the nozzle exit aerodynamic parameters and the normal-
shock location (if present), studying the relationship among area ratio,
pressure ratio, and shock position.

Governing equations (1-D isentropic + normal-shock relations)
-------------------------------------------------------------
Isentropic area–Mach relation:
  A/A* = (1/Ma)·((2/(γ+1))·(1+(γ−1)/2·Ma²))^((γ+1)/(2(γ−1)))

Normal-shock relations (Anderson Table A-2):
  Ma₂² = (2 + (γ−1)·Ma₁²) / (2γ·Ma₁² − (γ−1))
  p0₂/p0₁ = [(γ+1)Ma₁²/((γ−1)Ma₁²+2)]^(γ/(γ−1))·[(γ+1)/(2γMa₁²−(γ−1))]^(1/(γ−1))

Input variables
---------------
epsilon       Nozzle area ratio (exit/throat)   [2, 20]    -     Determines design Ma
P_back_ratio  Back-pressure ratio (p_b/p_0)     [0.01, 1]  -     Determines flow regime
gamma         Specific-heat ratio                [1.2, 1.67] -    Gas type
T0            Stagnation (total) temperature     [300, 2000] K    Gas temperature
shock_pos_frac Shock position fraction x_s/L     [0, 1]      -     Shock-location selector when an internal shock exists

Output responses
----------------
Ma_exit       Exit Mach number                 -          Primary output
T_exit        Exit static temperature           K
eta_nozzle    Nozzle efficiency (actual/ideal thrust ratio) -

DOE validation value
--------------------
VS-C (surrogate accuracy):
    Flow-regime transitions (subsonic → supersonic, shock → no shock) cause
    discontinuities in the response surface; surrogate accuracy degrades
    at regime boundaries — a classic discontinuity challenge.

VS-G (sequential optimisation):
    Locate the parameter-domain boundary of fully supersonic (design) flow.

References
----------
1. Anderson, J. D. (1990). Modern Compressible Flow, 2nd ed. McGraw-Hill.
2. Shapiro, A. H. (1953). Dynamics and Thermodynamics of Compressible
   Fluid Flow. Ronald Press.
"""

import numpy as np
from scipy.optimize import brentq

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class SupersonicNozzle5D(BaseCase):
    """Supersonic nozzle internal flow (5D → 3D, isentropic + normal-shock relations)."""

    meta = CaseMeta(
        case_id="supersonic_nozzle_5d",
        name="Supersonic converging–diverging nozzle flow (isentropic + normal shock)",
        version="1.0.0",
        domain="chemical_process",
        tags=["nozzle", "supersonic", "isentropic", "shock", "compressible", "5d"],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="epsilon",
                symbol=r"\varepsilon",
                description="Nozzle area expansion ratio (exit/throat)",
                unit="-",
                lower=2.0,
                upper=20.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="p_back_ratio",
                symbol="p_b/p_0",
                description="Back-pressure to stagnation-pressure ratio",
                unit="-",
                lower=0.01,
                upper=1.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="gamma",
                symbol=r"\gamma",
                description="Specific-heat ratio of the gas",
                unit="-",
                lower=1.2,
                upper=1.67,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="T0",
                symbol="T_0",
                description="Stagnation (total) temperature",
                unit="K",
                lower=300.0,
                upper=2000.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="shock_pos_frac",
                symbol="x_s/L",
                description="Shock position fraction (0 = throat, 1 = exit); effective only when shock exists",
                unit="-",
                lower=0.0,
                upper=1.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="Ma_exit",
                symbol="Ma_e",
                description="Nozzle exit Mach number",
                unit="-",
                is_primary=True,
                expected_range=(0.01, 6.0),
            ),
            OutputVariable(
                name="T_exit",
                symbol="T_e",
                description="Nozzle exit static temperature",
                unit="K",
                is_primary=False,
                expected_range=(50.0, 2000.0),
            ),
            OutputVariable(
                name="eta_nozzle",
                symbol=r"\eta_{nozzle}",
                description="Nozzle total-pressure recovery (with shock: p₀₂/p₀₁; design/subsonic = 1)",
                unit="-",
                is_primary=False,
                # Over-expanded with in-nozzle shock: Ma_shock can reach 5+, p02/p01 drops to 0.06;
                # design condition (no shock): eta = 1.  Covers the full physical range.
                expected_range=(0.05, 1.0),
                failure_threshold=0.80,
                failure_direction="below",
            ),
        ],
        governing_equation=(
            "1-D isentropic flow + normal-shock relations:\n"
            "  Area-ratio equation: A/A* = f(Ma, γ)\n"
            "  Normal shock: Ma₂ = f(Ma₁, γ), p0₂/p0₁ = f(Ma₁, γ)\n"
            "  Flow-regime determination: back-pressure ratio decides shock existence"
        ),
        pde_type="algebraic",
        spatial_dim=1,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Anderson, J. D. (1990). Modern Compressible Flow, 2nd ed. McGraw-Hill.",
            "Shapiro, A. H. (1953). Dynamics of Compressible Fluid Flow. Ronald Press.",
        ],
        solver_library="scipy.optimize (isentropic area-ratio equation solver)",
        solver_details=(
            "brentq for area-ratio equation; exact isentropic + normal-shock "
            "relations; post-shock exit Mach solved on downstream subsonic branch. "
            "Private physical-input calls are rejected before area-Mach/nozzle "
            "solves if they are malformed, non-finite, or outside the declared "
            "benchmark domain."
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 1.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["threshold_jump", "discontinuous"],
        effective_dim=3,
        important_variables=["epsilon", "p_back_ratio", "gamma"],
        interaction_pairs=[("epsilon", "p_back_ratio")],
        has_failure_boundary=True,
        failure_condition="Normal shock inside nozzle (over-expanded state; eta_nozzle significantly below 1)",
        sobol_S1={
            "epsilon": 0.211,
            "p_back_ratio": 0.194,
            "gamma": 0.000,
            "T0": 0.000,
            "shock_pos_frac": 0.015,
        },
        sobol_ST={
            "epsilon": 0.790,
            "p_back_ratio": 0.745,
            "gamma": 0.187,
            "T0": 0.000,
            "shock_pos_frac": 0.033,
        },
        sobol_computation="salib_saltelli_n2048_seed42_Ma_exit_output",
        applicable_scenarios=["VS-C", "VS-F", "VS-G"],
        scenario_notes={
            "VS-C": "Response surface discontinuity at flow-regime transitions; surrogate-accuracy challenge.",
            "VS-G": "Locate the parameter domain of fully supersonic (design) flow.",
        },
        recommended_n_train={"default": 60},
        n_saturation=300,
        noise_factor_candidates=["gamma"],
        control_factor_candidates=["epsilon", "p_back_ratio", "T0"],
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> None:
        """Reject malformed or out-of-domain private physical inputs."""
        X = np.asarray(X_phys, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.meta.dim_input:
            raise ValueError(
                "supersonic_nozzle_5d expects a physical input array with "
                f"shape (n, {self.meta.dim_input}); got {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "supersonic_nozzle_5d received non-finite physical inputs; "
                "refusing to extrapolate the 1-D nozzle surrogate."
            )

        tol = 1e-10
        for j, var in enumerate(self.meta.input_variables):
            lower = float(var.lower)
            upper = float(var.upper)
            scale = max(abs(lower), abs(upper), 1.0)
            bad = (X[:, j] < lower - tol * scale) | (X[:, j] > upper + tol * scale)
            if np.any(bad):
                raise ValueError(
                    "supersonic_nozzle_5d physical input "
                    f"{var.name!r} is outside the declared benchmark domain "
                    f"[{lower}, {upper}]."
                )

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        self._validate_physical_inputs(X_phys)
        X_phys = np.asarray(X_phys, dtype=float)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                Y[i] = self._nozzle(
                    epsilon=float(X_phys[i, 0]),
                    p_back_ratio=float(X_phys[i, 1]),
                    gamma=float(X_phys[i, 2]),
                    T0=float(X_phys[i, 3]),
                    shock_pos=float(X_phys[i, 4]),
                )
            except Exception as exc:
                raise RuntimeError(
                    "supersonic_nozzle_5d refusing silent NaN after nozzle "
                    f"evaluation failure for epsilon={X_phys[i, 0]!r}, "
                    f"p_back_ratio={X_phys[i, 1]!r}, gamma={X_phys[i, 2]!r}, "
                    f"T0={X_phys[i, 3]!r}, shock_pos_frac={X_phys[i, 4]!r}"
                ) from exc
        if (
            Y.shape != (n, self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y[:, 0] <= 0.0)
            or np.any(Y[:, 1] <= 0.0)
            or np.any(Y[:, 2] <= 0.0)
            or np.any(Y[:, 2] > 1.0)
        ):
            raise RuntimeError(
                "supersonic_nozzle_5d produced a malformed, non-finite, or "
                "non-physical nozzle response; refusing silent archive rows"
            )
        return Y

    @staticmethod
    def _area_ratio_to_Ma(AR, gamma, supersonic=True):
        """Solve for the Mach number corresponding to area ratio AR."""
        if not np.all(np.isfinite([AR, gamma])) or AR < 1.0 or gamma <= 1.0:
            raise ValueError(f"invalid area-Mach inputs: AR={AR!r}, gamma={gamma!r}")

        def f(Ma):
            return (
                ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * Ma**2))
                ** ((gamma + 1) / (2 * (gamma - 1)))
                / Ma
            ) - AR

        if supersonic:
            return brentq(f, 1.000001, 50.0, xtol=1e-8)
        else:
            return brentq(f, 0.01, 0.999, xtol=1e-8)

    @staticmethod
    def _normal_shock_p0_ratio(Ma1, gamma):
        g = gamma
        t1 = ((g + 1) * Ma1**2 / ((g - 1) * Ma1**2 + 2)) ** (g / (g - 1))
        t2 = ((g + 1) / (2 * g * Ma1**2 - (g - 1))) ** (1 / (g - 1))
        return t1 * t2

    def _nozzle(self, epsilon, p_back_ratio, gamma, T0, shock_pos):
        g = gamma
        if (
            not np.all(np.isfinite([epsilon, p_back_ratio, gamma, T0, shock_pos]))
            or epsilon < 1.0
            or p_back_ratio <= 0.0
            or p_back_ratio > 1.0
            or gamma <= 1.0
            or T0 <= 0.0
            or shock_pos < 0.0
            or shock_pos > 1.0
        ):
            raise ValueError(
                "invalid supersonic-nozzle input outside the declared physical "
                "contract"
            )
        # Design Mach number (supersonic solution)
        Ma_design = self._area_ratio_to_Ma(epsilon, g, supersonic=True)
        # Design back-pressure ratio (fully supersonic, no shock)
        p_exit_design = (1 + (g - 1) / 2 * Ma_design**2) ** (-g / (g - 1))

        # Back-pressure ratio for normal shock at exit
        Ma2_sq = ((g - 1) * Ma_design**2 + 2) / (2 * g * Ma_design**2 - (g - 1))
        Ma2 = np.sqrt(max(Ma2_sq, 0.0))
        p02_ratio = self._normal_shock_p0_ratio(Ma_design, g)
        p_after_shock = (1 + (g - 1) / 2 * Ma2**2) ** (-g / (g - 1)) * p02_ratio

        # Determine flow regime
        if p_back_ratio >= (1 - 1e-4):
            # Fully subsonic
            Ma_e = self._area_ratio_to_Ma(epsilon, g, supersonic=False)
            eta = 1.0
        elif p_back_ratio <= p_exit_design:
            # Supersonic design condition or over-expansion
            Ma_e = Ma_design
            eta = 1.0
        elif p_back_ratio <= p_after_shock:
            # Shock at exit or external
            # Exact boundary condition for a normal shock at the exit plane.
            Ma_e = Ma2
            eta = float(p02_ratio)
        else:
            # Shock inside nozzle (position approximated by shock_pos)
            # Find the corresponding Ma_shock
            A_shock = 1.0 + (epsilon - 1.0) * shock_pos
            Ma_shock = self._area_ratio_to_Ma(A_shock, g, supersonic=True)
            Ma_shock = max(Ma_shock, 1.001)
            # Post-shock Mach number
            Ma2s_sq = ((g - 1) * Ma_shock**2 + 2) / (2 * g * Ma_shock**2 - (g - 1))
            Ma2s = np.sqrt(max(Ma2s_sq, 0.0))
            # Downstream of a normal shock the flow remains isentropic on
            # a new, lower-total-pressure subsonic branch.  The downstream
            # critical area A*_2 is fixed by (A_shock, Ma2s), so the exit
            # area ratio is:
            #   A_e/A*_2 = (A_e/A_shock) * (A_shock/A*_2)
            #             = (epsilon/A_shock) * F(Ma2s)
            # where F(M) is the isentropic area-Mach function.  Solving
            # this subsonic branch preserves one-dimensional mass
            # conservation and avoids empirical Mach blending.
            A_ratio_downstream_shock = (
                ((2 / (g + 1)) * (1 + (g - 1) / 2 * Ma2s**2))
                ** ((g + 1) / (2 * (g - 1)))
                / Ma2s
            )
            A_ratio_exit_downstream = (epsilon / A_shock) * A_ratio_downstream_shock
            Ma_e = self._area_ratio_to_Ma(A_ratio_exit_downstream, g, supersonic=False)
            eta = float(self._normal_shock_p0_ratio(Ma_shock, g))

        Ma_e = float(Ma_e)
        T_e = T0 / (1 + (g - 1) / 2 * Ma_e**2)
        T_e = float(T_e)
        eta = float(eta)
        if (
            not np.all(np.isfinite([Ma_e, T_e, eta]))
            or Ma_e <= 0.0
            or T_e <= 0.0
            or T_e > T0
            or eta <= 0.0
            or eta > 1.0
        ):
            raise ValueError(
                "non-physical supersonic-nozzle response from isentropic/shock "
                "relations"
            )

        return np.array([Ma_e, T_e, eta])
