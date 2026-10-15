"""
Case S05: Stress concentration around an elliptical hole in a finite-width plate
================================================================================

Engineering background
----------------------
Holes in engineering structures (lightening holes, fastener holes, process
holes) are stress-concentration sources that may initiate fatigue cracks.
Kirsch (1898) gave the exact elastic solution for an infinite plate with
a circular hole under uniaxial tension — a classical result in elasticity.

For an elliptical hole (Inglis solution, 1913), the stress-concentration
factor under a remote tensile stress perpendicular to the major axis is:
  Kt = 1 + 2·(a/b)
where a is the semi-major axis and b the semi-minor axis.  The stress
concentrates at the major-axis ends; the crack-like limit is recovered as
a/b becomes large.

This case computes stress concentration in a finite-width plate with an
elliptical hole, including a finite-width correction factor, and studies
the effects of hole geometry, plate width, and biaxial loading on the
peak stress.

Physics model
-------------
Inglis (1913) solution — elliptical hole, infinite plate:
  Kt_x = 1 + 2·(a/b)   (load perpendicular to major axis)
  Kt_y = 1 + 2·(b/a)   (load perpendicular to minor axis)

Biaxial superposition at the two hole-edge end points (the textbook
Kirsch–Inglis formulas, not a naive additive superposition):
  Point A, major-axis end (θ = π/2):  σ_A = σ_x · Kt_x − σ_y
  Point B, minor-axis end (θ = 0):    σ_B = σ_y · Kt_y − σ_x
  Oblique loading also induces τ_xy; Inglis' pure-shear response adds
  Kt_shear = 2·(1 + a/b).
  σ_max = max(|σ_A|, |σ_B|) + |τ_xy|·Kt_shear

In the circular-hole limit Kt_x = Kt_y = 3, this recovers Kirsch (1898):
σ_A = 3σ_x − σ_y.

Finite-width correction (Pilkey 2008, Chart 4.1, Heywood polynomial):
  K_Heywood(a/W) = 1 + 0.5·(2a/W)³ + 0.3·(2a/W)⁵
  K_finite_x = Kt_x · K_Heywood,  K_finite_y = Kt_y · K_Heywood
  (W is the full plate width; a/W is the hole radius-to-half-width
  ratio, so the argument of K_Heywood is 2a/W; the empirical argument is
  capped at 0.75 to stay inside the conservative fit domain.)

Input variables
---------------
a_over_b    Ellipse axis ratio a/b     [0.5, 5.0]  -     Determines Kt
a_over_W    Hole radius / plate half-width [0.1, 0.45] -  Finite-width correction
sigma_x     X-direction nominal stress [50, 300]   MPa   Primary load
sigma_y     Y-direction nominal stress [-50, 100]  MPa   Biaxial load
theta_load  Load direction angle       [0, 90]     deg   Ellipse axis vs load direction

Output responses
----------------
sigma_max   Peak hole-edge stress      MPa   Primary output (fatigue-critical)
Kt          Stress concentration factor -    σ_max / σ_nom
tau_max     Maximum shear stress       MPa   For failure criterion

DOE validation value
--------------------
VS-C (surrogate accuracy):
    sigma_max is nearly linear in a_over_b at large axis ratios
    (Inglis: 1 + 2a/b), but the finite-width correction introduces
    nonlinearity — a moderate-difficulty surrogate test.

VS-G (sequential / reliability):
    Locate the safe domain where sigma_max < material allowable stress.

References
----------
1. Kirsch, E. G. (1898). Die Theorie der Elastizität und die Bedürfnisse der
   Festigkeitslehre. Zeitschrift des Vereines deutscher Ingenieure, 42(29):797–807.
2. Inglis, C. E. (1913). Stresses in a plate due to the presence of cracks
   and sharp corners. Transactions of the Institution of Naval Architects,
   55:219–241.
3. Pilkey, W. D. & Pilkey, D. F. (2008). Peterson's Stress Concentration
   Factors, 3rd ed. Wiley, New York. ISBN 978-0-470-04824-5. §4.3 (Elliptical
   holes) and Chart 4.1 (Circular hole in a finite-width plate).
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class PlateWithHole5D(BaseCase):
    """Stress concentration in a finite-width plate with elliptical hole (5D → 3D)."""

    meta = CaseMeta(
        case_id="plate_with_hole_5d",
        name="Finite-width plate with elliptical hole SCF (Inglis–Kirsch)",
        version="1.0.0",
        domain="structures",
        tags=[
            "stress_concentration",
            "plate",
            "hole",
            "kirsch",
            "inglis",
            "fatigue",
            "5d",
            "analytical",
        ],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="a_over_b",
                symbol="a/b",
                description="Ellipse axis ratio (a = major semi-axis; Kt_x assumes sigma_x is perpendicular to it)",
                unit="-",
                lower=0.5,
                upper=5.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="a_over_W",
                symbol="a/W",
                description="Hole radius to plate half-width ratio (finite-width correction)",
                unit="-",
                lower=0.1,
                upper=0.45,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="sigma_x",
                symbol=r"\sigma_x",
                description="X-direction nominal stress, taken perpendicular to the ellipse major axis at theta_load = 0",
                unit="MPa",
                lower=50.0,
                upper=300.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="sigma_y",
                symbol=r"\sigma_y",
                description="Y-direction (transverse) nominal stress",
                unit="MPa",
                lower=-50.0,
                upper=100.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="theta_load",
                symbol=r"\theta_{load}",
                description="Rotation angle of the nominal stress axes relative to the ellipse axes",
                unit="deg",
                lower=0.0,
                upper=90.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="sigma_max",
                symbol=r"\sigma_{max}",
                description="Peak hole-edge stress",
                unit="MPa",
                is_primary=True,
                expected_range=(50.0, 5000.0),
                failure_threshold=500.0,
                failure_direction="above",
            ),
            OutputVariable(
                name="Kt",
                symbol="K_t",
                description="Stress concentration factor",
                unit="-",
                is_primary=True,
                expected_range=(1.0, 25.0),
            ),
            OutputVariable(
                name="tau_max",
                symbol=r"\tau_{max}",
                description="Peak hole-edge shear stress",
                unit="MPa",
                is_primary=False,
                expected_range=(0.0, 2500.0),
            ),
        ],
        governing_equation=(
            "Inglis solution (elliptical hole, infinite plate):\n"
            "  Kt_x = 1 + 2·(a/b)  (sigma_x perpendicular to major axis)\n"
            "  Kt_y = 1 + 2·(b/a) = 1 + 2/(a/b)  (sigma_y perpendicular to minor axis)\n"
            "  Finite-width correction (Pilkey 2008, Chart 4.1, Heywood polynomial):\n"
            "    K_Heywood(a/W) = 1 + 0.5·clip(2a/W,0,0.75)³ + "
            "0.3·clip(2a/W,0,0.75)⁵\n"
            "  Biaxial superposition at hole-edge end points:\n"
            "    Point A (major-axis end): σ_A = σ_x · Kt_x · K_Heywood − σ_y\n"
            "    Point B (minor-axis end): σ_B = σ_y · Kt_y · K_Heywood − σ_x\n"
            "    Oblique-load shear term: |τ_xy| · 2(1+a/b) · K_Heywood\n"
            "    σ_max = max(|σ_A|, |σ_B|) + shear term\n"
            "  (Circular-hole limit Kt_x = Kt_y = 3 recovers Kirsch's 3σ_x − σ_y.)"
        ),
        pde_type="algebraic",
        spatial_dim=2,
        is_steady=True,
        # The underlying Kirsch/Inglis elasticity solution is linear in the
        # applied stresses for fixed geometry, but the benchmark response is
        # nonlinear in geometry and load angle through SCFs, finite-width
        # correction, max/absolute-value selection, and stress-axis rotation.
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Kirsch, E. G. (1898). Die Theorie der Elastizität und die Bedürfnisse "
            "der Festigkeitslehre. Z. VDI, 42(29):797–807.",
            "Inglis, C. E. (1913). Stresses in a plate due to the presence of cracks "
            "and sharp corners. Trans. Inst. Naval Architects, 55:219–241.",
            "Pilkey, W. D. & Pilkey, D. F. (2008). Peterson's Stress Concentration "
            "Factors, 3rd ed. Wiley. ISBN 978-0-470-04824-5, §4.3 + Chart 4.1.",
        ],
        solver_library="numpy (Inglis analytical solution + Heywood finite-width correction)",
        solver_details=(
            "Private _validate_physical_inputs rejects malformed, non-finite, "
            "or out-of-domain a_over_b/a_over_W/sigma_x/sigma_y/theta_load "
            "physical arrays before Inglis/Kirsch stress-concentration formula "
            "evaluation; exact Inglis analytical formula + Heywood polynomial "
            "correction + biaxial superposition"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.1},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_monotone", "strong_interaction"],
        effective_dim=5,
        important_variables=["theta_load", "a_over_b", "sigma_x", "a_over_W", "sigma_y"],
        interaction_pairs=[
            ("a_over_b", "sigma_x"),
            ("sigma_x", "sigma_y"),
            ("a_over_b", "a_over_W"),
            ("theta_load", "a_over_b"),
            ("theta_load", "sigma_x"),
            ("theta_load", "sigma_y"),
        ],
        has_failure_boundary=True,
        failure_condition="sigma_max > 500 MPa (exceeds typical aluminium-alloy allowable stress)",
        sobol_S1={
            "a_over_b": 0.151,
            "a_over_W": 0.033,
            "sigma_x": 0.213,
            "sigma_y": 0.015,
            "theta_load": 0.280,
        },
        sobol_ST={
            "a_over_b": 0.374,
            "a_over_W": 0.049,
            "sigma_x": 0.357,
            "sigma_y": 0.048,
            "theta_load": 0.547,
        },
        sobol_computation="salib_saltelli_n2048_seed42_sigma_max_output",
        applicable_scenarios=["VS-C", "VS-G"],
        scenario_notes={
            "VS-C": "sigma_max is nearly linear (Inglis) but with nonlinear finite-width correction.",
            "VS-G": "Locate the safe domain where sigma_max < allowable stress.",
        },
        recommended_n_train={"default": 40},
        n_saturation=200,
        noise_factor_candidates=["sigma_y"],
        control_factor_candidates=["a_over_b", "a_over_W", "sigma_x", "theta_load"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "plate_with_hole_5d physical inputs must be a 2-D array with "
                f"{self.meta.dim_input} columns"
            )
        if not np.all(np.isfinite(X_phys)):
            raise ValueError("plate_with_hole_5d physical inputs must be finite")

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
                "plate_with_hole_5d physical inputs outside the registered "
                "benchmark domain for a_over_b/a_over_W/sigma_x/sigma_y/theta_load; "
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
            a_W = float(X_phys[i, 1])
            sx = float(X_phys[i, 2])
            sy = float(X_phys[i, 3])
            theta_deg = float(X_phys[i, 4])
            if not np.all(np.isfinite([a_b, a_W, sx, sy, theta_deg])) or a_b <= 0.0:
                raise RuntimeError(
                    "plate_with_hole_5d produced a malformed, non-finite, or "
                    "non-physical stress-concentration response; refusing silent "
                    "archive rows"
                )
            theta = np.deg2rad(theta_deg)

            # Load-angle rotation (transform applied direction to ellipse coordinate system)
            # σ_x_eff = σx·cos²θ + σy·sin²θ,  σ_y_eff = σx·sin²θ + σy·cos²θ
            sx_eff = sx * np.cos(theta) ** 2 + sy * np.sin(theta) ** 2
            sy_eff = sx * np.sin(theta) ** 2 + sy * np.cos(theta) ** 2
            txy_eff = (sx - sy) * np.sin(theta) * np.cos(theta)

            # Inglis SCF (each direction independent).  The 1+2a/b term is
            # for remote tension perpendicular to the major axis.
            Kt_x = 1.0 + 2.0 * a_b
            Kt_y = 1.0 + 2.0 / a_b
            Kt_shear = 2.0 * (1.0 + a_b)  # pure-shear response

            # Heywood finite-width correction (Pilkey 2008, Table 4-1, classic polynomial)
            #   K_finite = Kt_inf × (1 + 0.5·(2a/W)³ + 0.3·(2a/W)⁵)
            ratio = 2.0 * a_W  # 2a/W
            ratio = float(np.clip(ratio, 0.0, 0.75))
            K_hw = 1.0 + 0.5 * ratio**3 + 0.3 * ratio**5

            K_finite_x = Kt_x * K_hw
            K_finite_y = Kt_y * K_hw
            K_finite_shear = Kt_shear * K_hw

            # Peak hole-edge stress (biaxial superposition, max of major/minor axis endpoints)
            # For an elliptical hole under combined σ_x, σ_y:
            #   Point A (major-axis end, θ = π/2): σ_θθ = σ_x·Kt_x − σ_y
            #   Point B (minor-axis end, θ = 0):   σ_θθ = σ_y·Kt_y − σ_x
            # Circular-hole limit (Kt_x = Kt_y = 3) recovers Kirsch result.
            sig_A = sx_eff * K_finite_x - sy_eff
            sig_B = sy_eff * K_finite_y - sx_eff
            sig_shear = abs(txy_eff * K_finite_shear)
            sigma_max = max(abs(sig_A), abs(sig_B)) + sig_shear
            sigma_max = max(sigma_max, 0.1)

            # Nominal stress (for Kt calculation)
            sigma_nom = max(abs(sx_eff), abs(sy_eff), 1.0)
            Kt = sigma_max / sigma_nom

            # Maximum shear stress (Tresca)
            tau_max = sigma_max / 2.0

            Y[i] = [sigma_max, Kt, tau_max]
        if (
            Y.shape != (n, self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y[:, 0] <= 0.0)
            or np.any(Y[:, 1] <= 0.0)
            or np.any(Y[:, 2] < 0.0)
        ):
            raise RuntimeError(
                "plate_with_hole_5d produced a malformed, non-finite, or "
                "non-physical stress-concentration response; refusing silent "
                "archive rows"
            )
        return Y
