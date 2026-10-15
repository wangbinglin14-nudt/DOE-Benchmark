"""
Case S02: Thin-plate buckling critical load
===========================================

Engineering background
----------------------
Thin-plate buckling is a core failure mode in aerospace structures (skins,
bulkheads), building steel structures (webs, cover plates), and ship
structures (shell plating).

The critical buckling stress of a uniformly compressed thin rectangular
plate is determined by:
  · Geometry: aspect ratio a/b, thickness h
  · Boundary conditions: support type on each edge (simply supported / clamped)
  · Material: Young’s modulus E, Poisson’s ratio ν

This case uses the exact analytical solution from Kirchhoff–Love plate
theory for a plate with all four edges simply supported under uniaxial
compression, with an analytical–numerical hybrid for mixed BCs (two short
edges clamped, two long edges simply supported).

Governing equation (Kirchhoff–Love plate)
-----------------------------------------
  D·∇⁴w = N_x·∂²w/∂x²
  D = Eh³ / (12(1−ν²))  (plate bending stiffness)

Exact solution for SSSS plate under uniaxial compression
(Timoshenko & Gere, 1961, §9.2):
  N_{x,cr} = k_σ · π²D / b²
  k_σ = min_m (m·b/a + n·a/(m·b))²         (n = 1 half-wave across b)
      = min_m (m/λ + λ/m)²,  with λ = a/b

BC correction: clamping the two unloaded longitudinal edges increases k_σ.
The SCSC branch uses a Bulson/Timoshenko table-fit approximation for short
plates and the classical long-plate asymptote k_σ ≈ 6.97.

Input variables
---------------
a_over_b   Aspect ratio a/b      [0.5, 4.0]  -    Determines buckling half-wave number
h_over_b   Thickness-to-width     [0.005, 0.05] -  Determines bending stiffness
E          Young’s modulus        [50, 300]   GPa  Material property
nu         Poisson’s ratio        [0.25, 0.45] -   Affects D = Eh³/12(1−ν²)
bc_flag    Boundary condition     [0, 1]      -    0 = SSSS, 1 = SCSC (continuous interpolation)

Output responses
----------------
N_cr       Critical buckling line load   N/m   Primary output
sigma_cr   Critical buckling stress      MPa   For comparison with allowable stress
k_sigma    Buckling coefficient          -     Dimensionless buckling load

DOE validation value
--------------------
VS-A (space-filling):
    k_σ vs a/b has local minima near integer a/b = 1, 2, 3 — a classic
    “periodic-trough” response surface; tests sampling coverage of local
    features.

VS-G (sequential):
    Find the feasible domain where sigma_cr > sigma_applied (given).

References
----------
1. Timoshenko, S. P. & Gere, J. M. (1961). Theory of Elastic Stability,
   2nd ed. McGraw-Hill. Chapter 9.
2. Bulson, P. S. (1970). The Stability of Flat Plates. Chatto & Windus.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class ThinPlateBuckling5D(BaseCase):
    """Thin-plate critical buckling load (Kirchhoff–Love theory), 5D → 3D."""

    meta = CaseMeta(
        case_id="thin_plate_buckling_5d",
        name="Rectangular thin-plate compressive buckling critical load",
        version="1.0.0",
        domain="structures",
        tags=[
            "buckling",
            "thin_plate",
            "kirchhoff",
            "stability",
            "critical_load",
            "5d",
            "analytical",
        ],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="a_over_b",
                symbol="a/b",
                description="Plate aspect ratio (loaded-direction length / width)",
                unit="-",
                lower=0.5,
                upper=4.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="h_over_b",
                symbol="h/b",
                description="Plate thickness-to-width ratio",
                unit="-",
                lower=0.005,
                upper=0.05,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="E",
                symbol="E",
                description="Young's modulus",
                unit="GPa",
                lower=50.0,
                upper=300.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="nu",
                symbol=r"\nu",
                description="Poisson's ratio",
                unit="-",
                lower=0.25,
                upper=0.45,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="bc_flag",
                symbol="BC",
                description="Boundary condition (0 = SSSS simply supported, 1 = SCSC two long edges clamped)",
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
                name="N_cr",
                symbol="N_{cr}",
                description="Critical buckling line load",
                unit="N/m",
                is_primary=True,
                expected_range=(0.0, 5.5e8),
                failure_threshold=None,
            ),
            OutputVariable(
                name="sigma_cr",
                symbol=r"\sigma_{cr}",
                description="Critical buckling stress",
                unit="MPa",
                is_primary=False,
                expected_range=(0.0, 11000.0),
            ),
            OutputVariable(
                name="k_sigma",
                symbol=r"k_\sigma",
                description="Dimensionless buckling coefficient (incl. BC correction)",
                unit="-",
                is_primary=False,
                expected_range=(3.0, 15.0),
            ),
        ],
        governing_equation=(
            "Kirchhoff–Love plate buckling: D·∇⁴w = Nx·∂²w/∂x²\n"
            "SSSS: Nx_cr = k_σ·π²D/b²,\n"
            "  k_σ = min_m[(m·b/a + a/(m·b))²] (m = 1, 2, …)\n"
            "SCSC: Bulson/Timoshenko table-fit for a/b < 1.4; "
            "long-plate asymptote k_σ ≈ 6.97 for a/b ≥ 1.4"
        ),
        pde_type="algebraic",
        spatial_dim=2,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Timoshenko, S. P. & Gere, J. M. (1961). Theory of Elastic Stability, 2nd ed. McGraw-Hill.",
            "Bulson, P. S. (1970). The Stability of Flat Plates. Chatto & Windus.",
        ],
        solver_library="numpy (analytical SSSS; table-fit SCSC approximation)",
        solver_details=(
            "Private _validate_physical_inputs rejects malformed, non-finite, "
            "or out-of-domain a_over_b/h_over_b/E/nu/bc_flag physical arrays "
            "before Kirchhoff thin-plate buckling formula evaluation; "
            "Sweep integer half-wave m = 1..20 for SSSS minimum k_σ; "
            "SCSC uses short-plate table-fit and long-plate asymptote. "
            "The integer mode search creates physical cusp points at modal transitions."
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.1},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["threshold_jump", "smooth_unimodal"],
        effective_dim=4,
        important_variables=["h_over_b", "E", "a_over_b", "nu"],
        interaction_pairs=[("h_over_b", "E"), ("a_over_b", "bc_flag")],
        has_failure_boundary=False,
        failure_condition="sigma_cr < sigma_applied (applied stress exceeds critical buckling stress; user-defined threshold)",
        sobol_S1={"a_over_b": 0.002, "h_over_b": 0.738, "E": 0.120, "nu": 0.002, "bc_flag": 0.001},
        sobol_ST={"a_over_b": 0.013, "h_over_b": 0.871, "E": 0.245, "nu": 0.004, "bc_flag": 0.007},
        sobol_computation="salib_saltelli_n2048_seed42_N_cr_output",
        applicable_scenarios=["VS-A", "VS-C", "VS-F"],
        scenario_notes={
            "VS-A": "k_σ vs a/b has periodic troughs at integer a/b = 1, 2, 3; tests sampling coverage.",
            "VS-F": "h_over_b and E dominate (ST > 0.5); a/b and nu near zero — easy screening verification.",
        },
        recommended_n_train={"default": 40},
        n_saturation=200,
        noise_factor_candidates=["nu"],
        control_factor_candidates=["a_over_b", "h_over_b", "E", "bc_flag"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "thin_plate_buckling_5d physical inputs must be a 2-D array "
                f"with {self.meta.dim_input} columns"
            )
        if not np.all(np.isfinite(X_phys)):
            raise ValueError("thin_plate_buckling_5d physical inputs must be finite")

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
                "thin_plate_buckling_5d physical inputs outside the registered "
                "benchmark domain for a_over_b/h_over_b/E/nu/bc_flag; "
                f"row {row}, {var.name}={X_phys[row, col]!r} not in "
                f"[{var.lower}, {var.upper}] {var.unit}"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            a_b = float(X_phys[i, 0])
            h_b = float(X_phys[i, 1])
            E = float(X_phys[i, 2]) * 1e9  # GPa → Pa
            nu = float(X_phys[i, 3])
            bc = float(X_phys[i, 4])  # 0=SSSS, 1=SCSC
            if (
                not np.all(np.isfinite([a_b, h_b, E, nu, bc]))
                or a_b <= 0.0
                or h_b <= 0.0
                or E <= 0.0
                or abs(nu) >= 1.0
                or bc < 0.0
                or bc > 1.0
            ):
                raise RuntimeError(
                    "thin_plate_buckling_5d produced a malformed, non-finite, "
                    "or non-physical buckling response; refusing silent archive rows"
                )

            b = 1.0  # unit width (normalised)
            h = h_b * b
            a = a_b * b

            # Plate bending stiffness
            D = E * h**3 / (12.0 * (1.0 - nu**2))

            # Optimal half-wave number (m = 1..20), k_σ = (m/n + n·(a/b)²/m)², n = 1 (uniaxial)
            k_min = np.inf
            for m in range(1, 21):
                k = (m * b / a + a / (m * b)) ** 2
                if k < k_min:
                    k_min = k
            k_ssss = k_min

            # SCSC branch: loaded edges simply supported, unloaded longitudinal
            # edges clamped.  For continuous bc_flag values, interpret the
            # interpolation as an elastic rotational-restraint surrogate.
            if a_b >= 1.4:
                k_scsc = 6.97
            else:
                k_scsc = 21.03 * a_b**2 - 46.96 * a_b + 32.22
                k_scsc = float(np.clip(k_scsc, 6.97, 15.0))
            k_sigma = k_ssss * (1.0 - bc) + k_scsc * bc

            # Critical load
            N_cr = k_sigma * np.pi**2 * D / b**2  # N/m
            sigma_cr = N_cr / h / 1e6  # MPa

            Y[i] = [N_cr, sigma_cr, k_sigma]
        if (
            Y.shape != (n, self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y[:, 0] <= 0.0)
            or np.any(Y[:, 1] <= 0.0)
            or np.any(Y[:, 2] <= 0.0)
        ):
            raise RuntimeError(
                "thin_plate_buckling_5d produced a malformed, non-finite, "
                "or non-physical buckling response; refusing silent archive rows"
            )
        return Y
