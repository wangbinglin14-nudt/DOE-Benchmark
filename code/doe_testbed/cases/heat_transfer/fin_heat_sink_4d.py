"""
Case T02: Rectangular fin heat-sink thermal resistance
=======================================================

Engineering background
----------------------
Fin heat sinks are core components in electronics cooling, heat exchangers,
and engine thermal management.  Heat-transfer analysis of rectangular fins
is fundamental to thermal design.

This case computes the heat dissipation, fin efficiency, and thermal
resistance of a single rectangular uniform-cross-section straight fin,
studying the effects of fin geometry and operating conditions on cooling
performance.

Fin heat-transfer equation (1-D steady state)
---------------------------------------------
Considering lateral convective dissipation, the temperature distribution
along the fin height satisfies:

  d²T/dx² − m²·(T − T_∞) = 0

where m² = h·P / (k·A), P = 2(W+t) (cross-section perimeter), A = W·t (area)

Boundary conditions:
  x = 0 (fin base): T = T_base
  x = H (fin tip):  adiabatic, dT/dx = 0

Exact solution:
  θ(x) = θ_b · cosh(m(H−x)) / cosh(mH)
  q_fin = θ_b · √(h·P·k·A) · tanh(mH)

Fin efficiency: η_fin = q_fin / (h·P·H·θ_b) = tanh(mH)/(mH)

Input variables
---------------
H       Fin height         [0.01, 0.08] m      Cooling area
W       Fin width          [0.01, 0.10] m      Heat-exchanger width (fixed = 0.05 m simplified)
t       Fin thickness      [0.5, 3.0]   mm     Manufacturing / weight constraint
h_coef  Surface HTC        [20, 500]    W/m²K  Convection condition (natural / forced)
k_fin   Fin thermal cond.  [100, 400]   W/mK   Aluminium (200) / copper (400)

Output responses
----------------
q_fin    Single-fin heat dissipation   W      Primary output
eta_fin  Fin efficiency                -      Fin utilisation (0–1)
R_fin    Fin thermal resistance        K/W    Thermal resistance (lower is better)

DOE validation value
--------------------
VS-B (effect estimation):
    H and h_coef are strongly coupled via m = √(h·P/(k·A)); a typical
    nonlinear-interaction case.

VS-H (robust Taguchi):
    h_coef has ±50 % uncertainty under natural-convection conditions (noise
    factor); H and t are control factors (fin design parameters).

References
----------
1. Incropera, F. P. et al. (2007). Fundamentals of Heat and Mass Transfer,
   6th ed. Wiley. Section 3.6.
2. Kraus, A. D., Aziz, A. & Welty, J. (2001). Extended Surface Heat Transfer.
   Wiley.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class FinHeatSink4D(BaseCase):
    """Rectangular fin heat sink (4D → 3D, analytical 1-D fin equation).

    Note: W is fixed at 0.05 m (typical fin width); effective inputs = 4.
    """

    _W_FIXED = 0.05  # m, fin width (fixed)
    _T_BASE = 80.0  # °C, fin-base temperature
    _T_INF = 25.0  # °C, ambient temperature

    meta = CaseMeta(
        case_id="fin_heat_sink_4d",
        name="Rectangular fin heat-sink thermal resistance (1-D fin equation)",
        version="1.0.0",
        domain="heat_transfer",
        tags=["fin", "heat_sink", "cooling", "convection", "4d", "analytical", "electronics"],
        dim_input=4,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="H",
                symbol="H",
                description="Fin height (cooling length)",
                unit="m",
                lower=0.01,
                upper=0.08,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="t",
                symbol="t",
                description="Fin thickness",
                unit="mm",
                lower=0.5,
                upper=3.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="h_coef",
                symbol="h",
                description="Surface convective heat-transfer coefficient",
                unit="W/(m²·K)",
                lower=20.0,
                upper=500.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="k_fin",
                symbol="k_{fin}",
                description="Fin material thermal conductivity",
                unit="W/(m·K)",
                lower=100.0,
                upper=400.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="q_fin",
                symbol="q_{fin}",
                description="Single-fin heat dissipation (fixed base temperature)",
                unit="W",
                is_primary=True,
                expected_range=(0.0, 100.0),
            ),
            OutputVariable(
                name="eta_fin",
                symbol=r"\eta_{fin}",
                description="Fin efficiency (actual / ideal heat dissipation ratio)",
                unit="-",
                is_primary=True,
                expected_range=(0.08, 1.0),
                failure_threshold=0.7,
                failure_direction="below",
            ),
            OutputVariable(
                name="R_fin",
                symbol="R_{fin}",
                description="Fin thermal resistance",
                unit="K/W",
                is_primary=False,
                expected_range=(0.1, 100.0),
            ),
        ],
        governing_equation=(
            "1-D fin equation: d²θ/dx² − m²θ = 0, θ = T − T_∞\n"
            "m² = h·P/(k·A), P = 2(W+t), A = W·t\n"
            "Adiabatic tip: θ(x) = θ_b·cosh(m(H−x))/cosh(mH)\n"
            "η_fin = tanh(mH)/(mH)"
        ),
        pde_type="ode_system",
        spatial_dim=1,
        is_steady=True,
        # The fin ODE is linear in theta for fixed coefficients, but the
        # benchmark response is nonlinear in the inputs through sqrt(hP/kA),
        # tanh(mH), and the geometry products P=2(W+t), A=W*t.
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Incropera, F. P., DeWitt, D. P., Bergman, T. L. & Lavine, A. S. (2007). "
            "Fundamentals of Heat and Mass Transfer, 6th ed. Wiley. §3.6 (Extended Surfaces).",
            "Kraus, A. D. et al. (2001). Extended Surface Heat Transfer. Wiley.",
        ],
        solver_library="numpy (exact analytical formula)",
        solver_details=(
            "Private _validate_physical_inputs rejects malformed, non-finite, "
            "or out-of-domain H/t/h_coef/k_fin physical arrays before exact "
            "insulated-tip fin formula evaluation; lateral wetted area only; "
            "stable tanh(mH)/(mH) small-argument expansion"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.05},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_monotone", "strong_interaction"],
        effective_dim=4,
        important_variables=["H", "h_coef", "t", "k_fin"],
        interaction_pairs=[("H", "h_coef"), ("t", "h_coef"), ("H", "k_fin")],
        has_failure_boundary=True,
        failure_condition="eta_fin < 0.70 (poor fin utilisation / excessive temperature drop along the fin)",
        sobol_S1={"H": 0.102, "t": 0.044, "h_coef": 0.745, "k_fin": 0.026},
        sobol_ST={"H": 0.143, "t": 0.092, "h_coef": 0.819, "k_fin": 0.055},
        sobol_computation="salib_saltelli_n2048_seed42_q_fin_output",
        applicable_scenarios=["VS-B", "VS-C", "VS-D", "VS-H"],
        scenario_notes={
            "VS-B": "H and h_coef strongly coupled via m = √(hP/kA); CCD necessary.",
            "VS-H": "h_coef is a noise factor (natural-convection uncertainty); find fin designs insensitive to h.",
        },
        recommended_n_train={"default": 30},
        n_saturation=150,
        noise_factor_candidates=["h_coef"],
        control_factor_candidates=["H", "t", "k_fin"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "fin_heat_sink_4d physical inputs must be a 2-D array with "
                f"{self.meta.dim_input} columns"
            )
        if not np.all(np.isfinite(X_phys)):
            raise ValueError("fin_heat_sink_4d physical inputs must be finite")

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
                "fin_heat_sink_4d physical inputs outside the registered "
                "benchmark domain for H/t/h_coef/k_fin; "
                f"row {row}, {var.name}={X_phys[row, col]!r} not in "
                f"[{var.lower}, {var.upper}] {var.unit}"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        W = self._W_FIXED
        dT = self._T_BASE - self._T_INF

        for i in range(n):
            H = float(X_phys[i, 0])
            t = float(X_phys[i, 1]) * 1e-3  # mm → m
            h = float(X_phys[i, 2])
            k = float(X_phys[i, 3])
            if (
                not np.all(np.isfinite([H, t, h, k]))
                or H <= 0.0
                or t <= 0.0
                or h <= 0.0
                or k <= 0.0
            ):
                raise RuntimeError(
                    "fin_heat_sink_4d produced a malformed, non-finite, or "
                    "non-physical fin response; refusing silent archive rows"
                )

            # Cross-section parameters
            A_cs = W * t  # m² (cross-sectional area)
            P_cs = 2 * (W + t)  # m  (cross-section perimeter)

            # Fin parameter m
            m = np.sqrt(h * P_cs / (k * A_cs + 1e-30))
            mH = m * H

            # Fin efficiency for an insulated tip.  The series branch avoids a
            # hard max/clip at mH ≈ 0 while preserving eta -> 1 smoothly.
            eta = self._eta_insulated_tip(mH)

            # Heat dissipation (adiabatic tip BC)
            q = np.sqrt(h * P_cs * k * A_cs) * np.tanh(mH) * dT
            q = float(q)

            # Thermal resistance
            R = dT / max(q, 1e-10)

            Y[i] = [q, eta, R]
        if (
            Y.shape != (n, self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y[:, 0] <= 0.0)
            or np.any(Y[:, 1] <= 0.0)
            or np.any(Y[:, 1] > 1.0)
            or np.any(Y[:, 2] <= 0.0)
        ):
            raise RuntimeError(
                "fin_heat_sink_4d produced a malformed, non-finite, or "
                "non-physical fin response; refusing silent archive rows"
            )
        return Y

    @staticmethod
    def _eta_insulated_tip(mH):
        z = float(mH)
        if abs(z) < 1e-4:
            z2 = z * z
            return 1.0 - z2 / 3.0 + 2.0 * z2 * z2 / 15.0
        return float(np.tanh(z) / z)
