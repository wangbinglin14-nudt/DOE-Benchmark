"""
Case T05: Casting solidification (Stefan phase-change problem)
===============================================================

Engineering background
----------------------
Casting is one of the most important forming processes in manufacturing.
During solidification of a casting:
  · Molten metal solidifies progressively from outside to inside, forming
    a solid–liquid interface (solidification front)
  · Latent-heat release delays interior solidification, creating a
    temperature plateau
  · Solidification rate determines microstructure and defects (shrinkage
    porosity, hot cracks)

The Stefan problem (moving-boundary problem) is the core mathematical
description of solidification heat transfer:
  Solid-phase conduction + liquid-phase conduction + interface condition
  (Stefan condition)

This case solves a 1-D slab casting solidification process with explicit
FDM, computing the complete solidification time, solidification-front
velocity history, and approximate final stress distribution.

Governing equations (1-D Stefan problem)
-----------------------------------------
Solid region (0 ≤ x ≤ s(t)):
  ρ_s·c_{p,s}·∂T/∂t = k_s·∂²T/∂x²

Liquid region (s(t) ≤ x ≤ L):
  ρ_l·c_{p,l}·∂T/∂t = k_l·∂²T/∂x²

Interface condition (Stefan condition):
  ρ_s·L_f·ds/dt = k_s·(∂T/∂x)|_{s⁻} − k_l·(∂T/∂x)|_{s⁺}

Initial condition: T(x, 0) = T_pour (pouring temperature)
Boundary condition: x = 0: −k_s·∂T/∂x = h_eff·(T − T_mold), with
  h_eff = (1/h_contact + ell_mold/k_mold)^−1
as a low-order series resistance for interface contact plus near-wall
mould conduction.

Input variables
---------------
T_pour    Pouring temperature       [T_melt+10, T_melt+200] °C  Superheat
T_mold    Mould initial temperature [20, 100]               °C  Room temp. or preheat
h_contact Casting–mould interface HTC [200, 2000]            W/m²K Interface heat transfer
k_mold    Mould thermal conductivity [0.3, 1.5]             W/mK  Mould material
L_half    Casting half-thickness     [0.01, 0.10]           m     Casting dimension

Output responses
----------------
t_solid   Complete solidification time  s      Primary output (casting cycle time)
v_front   Mean solidification-front speed mm/s Microstructure prediction indicator
T_center  Centre peak temperature        °C    Internal temperature peak

DOE validation value
--------------------
VS-A (space filling):
    t_solid exhibits a steep gradient along h_contact (drastic change when
    interface thermal resistance dominates) — a typical case of response-
    surface spatial heterogeneity.

VS-G (sequential):
    Find the feasible parameter domain where t_solid < a given process-
    time limit.

References
----------
1. Carslaw, H. S. & Jaeger, J. C. (1959). Conduction of Heat in Solids, 2nd ed. Oxford.
2. Shamsundar, N. & Sparrow, E. M. (1975). Analysis of multidimensional
   conduction phase change via the enthalpy model.
   ASME J. Heat Transfer, 97(3):333-340.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class CastingSolidification5D(BaseCase):
    """1-D casting solidification (Stefan phase-change problem, FDM), 5D → 3D."""

    # Casting (aluminium alloy A356) fixed thermophysical properties
    _T_MELT = 615.0  # °C, solidification temperature
    _RHO_S = 2680.0  # kg/m³ (solid phase)
    _RHO_L = 2560.0  # kg/m³ (liquid phase)
    _CP_S = 963.0  # J/(kg·K)
    _CP_L = 1080.0
    _K_S = 155.0  # W/(m·K)
    _K_L = 90.0
    _L_F = 3.97e5  # J/kg, latent heat
    _MOLD_RESISTANCE_LENGTH = 1.0e-3  # m, near-surface mold thermal length

    meta = CaseMeta(
        case_id="casting_solidification_5d",
        name="1-D casting solidification (Stefan phase-change problem)",
        version="1.0.0",
        domain="heat_transfer",
        tags=[
            "casting",
            "solidification",
            "stefan_problem",
            "phase_change",
            "fdm",
            "5d",
            "transient",
        ],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="T_pour",
                symbol="T_{pour}",
                description="Aluminium-alloy pouring temperature (superheat above solidification temperature)",
                unit="°C",
                lower=625.0,
                upper=815.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="T_mold",
                symbol="T_{mold}",
                description="Mould initial temperature",
                unit="°C",
                lower=20.0,
                upper=100.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="h_contact",
                symbol="h_c",
                description="Casting–mould interface heat-transfer coefficient",
                unit="W/(m²·K)",
                lower=200.0,
                upper=2000.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="k_mold",
                symbol="k_{mold}",
                description="Mould thermal conductivity",
                unit="W/(m·K)",
                lower=0.3,
                upper=1.5,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="L_half",
                symbol="L_{half}",
                description="Casting half-thickness (single-side cooling)",
                unit="m",
                lower=0.01,
                upper=0.10,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="t_solid",
                symbol="t_{sol}",
                description="Complete solidification time (thin wall + high interface HTC can be < 10 s)",
                unit="s",
                is_primary=True,
                # L_half=0.01 m + h=2000 W/m²K combination: Chvorinov scale < 10 s
                expected_range=(1.0, 10000.0),
            ),
            OutputVariable(
                name="v_front",
                symbol="v_{front}",
                description="Mean solidification-front advance speed",
                unit="mm/s",
                is_primary=False,
                expected_range=(0.001, 10.0),
            ),
            OutputVariable(
                name="T_center_peak",
                symbol="T_{c,peak}",
                description="Casting centre peak temperature",
                unit="°C",
                is_primary=False,
                expected_range=(615.0, 820.0),
            ),
        ],
        governing_equation=(
            "1-D Stefan problem (enthalpy-method FDM):\n"
            "  ρ·∂H/∂t = k·∂²T/∂x²\n"
            "  H(T) = c_{p,s}·T,  T < T_melt\n"
            "       = c_{p,s}·T_melt + L_f·f_s,  T = T_melt (latent heat)\n"
            "  BC: −k·∂T/∂x|_0 = h_eff·(T − T_mold), "
            "h_eff = (1/h_c + ell_mold/k_mold)^−1"
        ),
        pde_type="parabolic",
        spatial_dim=1,
        is_steady=False,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Carslaw, H. S. & Jaeger, J. C. (1959). Conduction of Heat in Solids. Oxford.",
            "Shamsundar, N. & Sparrow, E. M. (1975). ASME J. Heat Transfer, 97(3):333-340.",
        ],
        solver_library="numpy (enthalpy-method explicit FDM)",
        solver_details=(
            "Private _validate_physical_inputs rejects malformed, non-finite, "
            "or out-of-domain T_pour/T_mold/h_contact/k_mold/L_half physical "
            "arrays before enthalpy-method FDM assembly; 20 spatial nodes "
            "(default) / 100 (high); adaptive time step; Fourier stability "
            "control; mushy-zone interpolation for front tracking"
        ),
        fidelity_levels=["default", "high"],
        typical_eval_time_ms={"default": 60.0, "high": 200.0},
        cost_level="L1",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=True,
        response_topology=["smooth_unimodal", "strong_interaction", "heterogeneous"],
        effective_dim=5,
        important_variables=["L_half", "h_contact", "T_pour"],
        interaction_pairs=[("h_contact", "L_half"), ("T_pour", "h_contact")],
        has_failure_boundary=False,
        sobol_S1={
            "T_pour": 0.002,
            "T_mold": 0.000,
            "h_contact": 0.016,
            "k_mold": 0.000,
            "L_half": 0.957,
        },
        sobol_ST={
            "T_pour": 0.003,
            "T_mold": 0.001,
            "h_contact": 0.043,
            "k_mold": 0.000,
            "L_half": 0.984,
        },
        sobol_computation="salib_saltelli_n1024_seed42_t_solid_output",
        applicable_scenarios=["VS-A", "VS-C"],
        scenario_notes={
            "VS-A": "t_solid has a steep gradient along h_contact (interface resistance dominated); strong heterogeneity.",
        },
        recommended_n_train={"default": 60, "high": 200},
        n_saturation=300,
        noise_factor_candidates=["T_mold", "k_mold"],
        control_factor_candidates=["T_pour", "h_contact", "L_half"],
        reference_solution_available=True,
        reference_grid_size=2000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "casting_solidification_5d physical inputs must be a 2-D array "
                f"with {self.meta.dim_input} columns"
            )
        if not np.all(np.isfinite(X_phys)):
            raise ValueError("casting_solidification_5d physical inputs must be finite")

        lower = np.array([v.lower for v in self.meta.input_variables], dtype=float)
        upper = np.array([v.upper for v in self.meta.input_variables], dtype=float)
        tol = 1e-10
        outside = (X_phys < lower - tol) | (X_phys > upper + tol)
        if np.any(outside):
            bad_rows, bad_cols = np.where(outside)
            row = int(bad_rows[0])
            col = int(bad_cols[0])
            var = self.meta.input_variables[col]
            raise ValueError(
                "casting_solidification_5d physical inputs outside the "
                "registered benchmark domain for "
                "T_pour/T_mold/h_contact/k_mold/L_half; "
                f"row {row}, {var.name}={X_phys[row, col]!r} not in "
                f"[{var.lower}, {var.upper}] {var.unit}"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n_x = 20 if fidelity == "default" else 100
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                Y[i] = self._solidify(
                    T_pour=float(X_phys[i, 0]),
                    T_mold=float(X_phys[i, 1]),
                    h_contact=float(X_phys[i, 2]),
                    k_mold=float(X_phys[i, 3]),
                    L=float(X_phys[i, 4]),
                    n_x=n_x,
                )
                if (
                    Y[i].shape != (self.meta.dim_output,)
                    or not np.all(np.isfinite(Y[i]))
                    or Y[i, 0] <= 0.0
                    or Y[i, 1] <= 0.0
                    or Y[i, 2] < self._T_MELT
                    or Y[i, 2] > float(X_phys[i, 0]) + 1e-7
                ):
                    raise ValueError(
                        "malformed, non-finite, or non-physical casting "
                        "solidification response "
                        f"t_solid/v_front/T_center_peak={np.array2string(Y[i], precision=6, separator=', ')}"
                    )
            except Exception as exc:
                raise RuntimeError(
                    "casting_solidification_5d refusing silent NaN after "
                    f"Stefan FDM evaluation failure for T_pour={X_phys[i, 0]!r}, "
                    f"T_mold={X_phys[i, 1]!r}, h_contact={X_phys[i, 2]!r}, "
                    f"k_mold={X_phys[i, 3]!r}, L_half={X_phys[i, 4]!r}, "
                    f"fidelity={fidelity!r}"
                ) from exc
        return Y

    def _solidify(self, T_pour, T_mold, h_contact, k_mold, L, n_x=20):
        """Enthalpy-method FDM for 1-D Stefan problem (vectorised, adaptive exit)."""
        T_m = self._T_MELT
        rho_s = self._RHO_S
        rho_l = self._RHO_L
        cp_s = self._CP_S
        cp_l = self._CP_L
        k_s = self._K_S
        k_l = self._K_L
        Lf = self._L_F

        dx = L / n_x
        h_eff = 1.0 / (
            1.0 / max(h_contact, 1e-12)
            + self._MOLD_RESISTANCE_LENGTH / max(k_mold, 1e-12)
        )
        alpha_max = max(k_s / (rho_s * cp_s), k_l / (rho_l * cp_l))
        dt = 0.45 * dx**2 / alpha_max  # CFL stability time step

        # Estimate max solidification time.  The previous Chvorinov-style
        # cap was too tight and often became the returned t_solid itself,
        # suppressing the intended dependence on superheat, contact HTC, and
        # mould conductivity.  Use a conservative diffusive upper bound while
        # keeping the benchmark runtime finite.
        t_max = min(80.0 * (L / 2) ** 2 / alpha_max, 10000.0)

        H_melt = rho_s * cp_s * T_m
        H_liquid = H_melt + rho_s * Lf

        def T_to_H(Tf):
            # Three-segment: solid sensible / phase-change latent / liquid sensible + superheat
            # For T_pour > T_m initial states, liquid superheat energy must be included in
            # initial H; otherwise the superheat contribution to solidification time is lost.
            return np.where(
                Tf < T_m,
                rho_s * cp_s * Tf,
                H_liquid + rho_l * cp_l * (Tf - T_m),
            )

        def H_to_T(Hf):
            T_out = np.where(Hf < H_melt, Hf / (rho_s * cp_s), T_m)
            T_out = np.where(Hf > H_liquid, T_m + (Hf - H_liquid) / (rho_l * cp_l), T_out)
            return T_out

        H = T_to_H(np.full(n_x, T_pour))
        t = 0.0
        t_solid = None
        T_center_peak = float(T_pour)
        front_x0 = front_t0 = 0.0
        front_x1 = front_t1 = 0.0

        # Check convergence every check_interval steps to reduce overhead
        check_interval = max(1, int(0.5 / dt))  # check roughly every 0.5 s

        step = 0
        while t < t_max:
            T = H_to_T(H)

            # Update centre peak temperature (last node = casting centre)
            T_c = float(T[-1])
            if T_c > T_center_peak:
                T_center_peak = T_c

            # Conductivity field (interpolation in mushy zone)
            frac_s = np.clip((H_liquid - H) / (rho_s * Lf), 0.0, 1.0)
            k_field = k_s * frac_s + k_l * (1.0 - frac_s)

            # Vectorised FDM
            k_e = (k_field[1:] + k_field[:-1]) * 0.5
            flux = k_e * np.diff(T) / dx
            dH = np.empty(n_x)
            dH[1:-1] = dt / dx * (flux[1:] - flux[:-1])
            dH[0] = dt / dx * (h_eff * (T_mold - T[0]) + flux[0])
            dH[-1] = -dt / dx * flux[-1]

            H += dH
            t += dt
            step += 1

            if step % check_interval == 0:
                new_x = self._front_position_from_enthalpy(H, H_melt, H_liquid, dx)
                if new_x > front_x1 + 1e-12:
                    front_x0, front_t0 = front_x1, front_t1
                    front_x1, front_t1 = new_x, t

                if np.all(H < H_melt + rho_s * Lf * 0.02):
                    t_solid = t
                    break

        if t_solid is None:
            t_solid = t_max

        front_x_final = max(front_x1, self._front_position_from_enthalpy(H, H_melt, H_liquid, dx))
        if front_x_final > 0.0:
            v_front = min(front_x_final, L) * 1000.0 / max(t_solid, 1e-6)
        else:
            v_front = L * 1000.0 / max(t_solid, 1.0)

        return np.array([t_solid, v_front, T_center_peak])

    @staticmethod
    def _front_position_from_enthalpy(H, H_melt, H_liquid, dx):
        """Return a continuous solidification-front position from an enthalpy field."""
        H_arr = np.asarray(H, dtype=float)
        n = H_arr.size
        solid_prefix = 0
        while solid_prefix < n and H_arr[solid_prefix] < H_melt:
            solid_prefix += 1

        if solid_prefix >= n:
            return n * dx

        H_node = H_arr[solid_prefix]
        if H_melt <= H_node <= H_liquid:
            latent_span = max(H_liquid - H_melt, 1e-30)
            local_solid_fraction = np.clip((H_liquid - H_node) / latent_span, 0.0, 1.0)
            return (solid_prefix + local_solid_fraction) * dx

        return solid_prefix * dx
