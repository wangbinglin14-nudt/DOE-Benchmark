"""
Case T04: Multi-layer composite wall steady-state heat conduction (scalable dimension)
=======================================================================================

Engineering background
----------------------
Building envelopes (external walls, roofs), industrial furnace linings,
and pipe insulation layers are typically composed of multiple layers of
different materials, each contributing an independent thermal resistance
to the overall heat-transfer performance.

Steady-state conduction through a multi-layer wall is the simplest and
clearest series thermal-resistance model:
  R_total = Σ R_i = Σ (δ_i / k_i)  (sum of per-layer resistances)

This is an ideal benchmark for high-dimensional DOE validation (VS-E):
  · Dimension = number of layers N (can be 4 / 8 / 16)
  · Per-layer resistances add independently — no interaction effects
    (pure main-effect model)
  · Effective dimension = N (all layers matter, but thicker / lower-k
    layers contribute more)
  · Complementary to the variable-beam case (S03): one has sparse effects,
    the other dense effects

This “pure main-effect” model is an important control group for method
evaluation: Resolution-III designs should perform well here, while failing
on strong-interaction problems (e.g. fixed-bed reactor).

Governing equation
------------------
N-layer planar-wall steady-state conduction (equal cross-sectional area):
  q = (T_hot − T_cold) / R_total
  R_total = Σ δ_i / k_i  (m²·K/W, unit-area thermal resistance)
  T_i(x) = T(x_0) − q·(x − x_0) / k_i  (linear temperature profile per layer)

Per-layer parameters: thickness δ_i (m), thermal conductivity k_i (W/(m·K))

Input variables (N layers; each k_i is a design variable, δ_i fixed)
---------------------------------------------------------------------
k_i (i = 1..N)  Per-layer thermal conductivity  [0.05, 50] W/(m·K)  log-uniform

Fixed parameters: equal layer thickness δ = 0.05 m, T_hot = 80 °C, T_cold = 20 °C

Output responses
----------------
R_total   Total thermal resistance   m²·K/W   Primary output (higher = better insulation)
q_flux    Heat-flux density           W/m²     Heat-transfer intensity
T_mid     Mid-layer interface temperature  °C  Representative interface temperature

DOE validation value
--------------------
VS-E (high-dimensional adaptability):
    N = 4/8/16 layers adjustable; all layers contribute (no sparse effects).
    Contrasts with S03 (root segments dominate) and G01 (diagonal blocks
    dominate), representing three different high-dimensional effect structures.

VS-F (variable screening):
    All variables have non-zero Sobol ST, but lower-k layers contribute more.
    Tests whether screening methods correctly identify the rule “lower-
    conductivity layers are more important”.

References
----------
1. Incropera, F. P. et al. (2007). Fundamentals of Heat and Mass Transfer,
   6th ed. Wiley. Section 3.1.
2. Bergman, T. L. et al. (2011). Introduction to Heat Transfer, 5th ed. Wiley.
3. MIT Unified Engineering, Thermal Resistance Circuits.
   https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node118.html
4. NPTEL Heat Transfer lecture material, conduction through composite walls.
   https://archive.nptel.ac.in/content/storage2/courses/112101097/download/lecture-4.pdf
5. NASA Passive Thermal Control Engineering Guidebook, used only as boundary
   evidence for broader thermal-path, hardware-integration, and
   test/analysis workflows outside this compact series-resistance benchmark.
   https://ntrs.nasa.gov/citations/20230013900
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


def _make_multilayer_meta(N: int) -> CaseMeta:
    return CaseMeta(
        case_id=f"multilayer_wall_{N}d",
        name=f"Multi-layer wall steady-state conduction ({N} layers, scalable)",
        version="1.0.0",
        domain="heat_transfer",
        tags=[
            "multilayer",
            "wall",
            "conduction",
            "thermal_resistance",
            f"{N}d",
            "scalable",
            "series_resistance",
        ],
        dim_input=N,
        dim_output=3,
        input_variables=[
            InputVariable(
                name=f"k{i+1}",
                symbol=f"k_{i+1}",
                description=f"Layer {i+1} thermal conductivity",
                unit="W/(m·K)",
                lower=0.05,
                upper=50.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            )
            for i in range(N)
        ],
        output_variables=[
            OutputVariable(
                name="R_total",
                symbol="R_{total}",
                description="Total thermal resistance (per unit area)",
                unit="m²·K/W",
                is_primary=True,
                expected_range=(0.004, 20.0),
            ),
            OutputVariable(
                name="q_flux",
                symbol="q''",
                description="Steady-state heat-flux density",
                unit="W/m²",
                is_primary=False,
                expected_range=(3.0, 15000.0),
            ),
            OutputVariable(
                name="T_mid",
                symbol="T_{mid}",
                description="Mid-layer interface temperature (after layer N//2)",
                unit="°C",
                is_primary=False,
                expected_range=(20.0, 80.0),
            ),
        ],
        governing_equation=(
            "N-layer planar-wall steady-state conduction:\n"
            "  R_total = Σ_i δ_i/k_i  (series thermal resistance)\n"
            "  q = ΔT/R_total\n"
            "  Linear temperature profile per layer (no heat source)"
        ),
        pde_type="algebraic",
        spatial_dim=1,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Incropera, F. P., DeWitt, D. P., Bergman, T. L. & Lavine, A. S. (2007). "
            "Fundamentals of Heat and Mass Transfer, 6th ed. Wiley. §3.1 (The Plane Wall).",
            "MIT Unified Engineering, Thermal Resistance Circuits. "
            "https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node118.html",
            "NPTEL Heat Transfer, conduction through composite walls. "
            "https://archive.nptel.ac.in/content/storage2/courses/112101097/download/lecture-4.pdf",
            "NASA Passive Thermal Control Engineering Guidebook, boundary context for "
            "broader thermal-path, hardware-integration, and test/analysis workflows "
            "outside this compact series-resistance benchmark. "
            "https://ntrs.nasa.gov/citations/20230013900",
        ],
        solver_library="numpy (series resistance analytical calculation)",
        solver_details=(
            "Direct unit-area series-resistance summation for a fixed-thickness "
            "1-D composite wall; no numerical error and no numerical "
            "discretization error. Analytical "
            "control for Fourier-law identities, not a calibrated material-stack, "
            "thermal-bridge, moisture, contact-resistance, radiation/convection, "
            "2-D/3-D building-envelope, spacecraft passive-thermal hardware/"
            "integration/test-analysis workflow, or code-compliance design model; named "
            "_validate_physical_inputs private physical-input evaluation rejects "
            "malformed arrays and layer conductivities outside the declared "
            "benchmark domain before series-resistance assembly."
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.02},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_monotone"],
        effective_dim=N,
        important_variables=[f"k{i+1}" for i in range(N)],
        interaction_pairs=[],  # pure main effects, no interactions
        has_failure_boundary=False,
        sobol_S1={f"k{i+1}": round(1.0 / N, 3) for i in range(N)},
        sobol_ST={f"k{i+1}": round(1.0 / N, 3) for i in range(N)},
        sobol_computation=(
            "analytic_series_resistance_control; equal 1/N indices are a "
            "balanced-main-effect metadata convention for the primary resistance "
            "control, not a fresh Saltelli estimate for every nonlinear output"
        ),
        applicable_scenarios=["VS-E", "VS-F"],
        scenario_notes={
            "VS-E": f"{N}-D pure main-effect model, effective dim = {N}; "
            "represents the non-sparse high-dimensional challenge (contrast with G01's sparse structure).",
            "VS-F": "All variables have non-zero ST; lower-k layers contribute more. "
            "Tests whether screening methods can identify non-equal but non-zero effects.",
        },
        recommended_n_train={"default": N * 3},
        n_saturation=N * 20,
        noise_factor_candidates=[],
        control_factor_candidates=[f"k{i+1}" for i in range(N)],
        reference_solution_available=True,
        reference_grid_size=3000,
        scalable_dimension=True,
        scalable_dimension_range=(4, 16),
    )


class MultilayerWallBase(BaseCase):
    """Multi-layer composite wall steady-state conduction base class."""

    _DELTA_PER_LAYER = 0.05  # m, per-layer thickness (fixed)
    _T_HOT = 80.0  # °C
    _T_COLD = 20.0  # °C

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                f"{self.meta.case_id} expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        lower = 0.05 * (1.0 - tol)
        upper = 50.0 * (1.0 + tol)
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(X_phys < lower)
            or np.any(X_phys > upper)
        ):
            raise ValueError(
                f"{self.meta.case_id} physical inputs must stay inside the "
                "declared layer-conductivity benchmark domain before "
                "series-resistance assembly"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        N = self.meta.dim_input
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        delta = self._DELTA_PER_LAYER
        dT = self._T_HOT - self._T_COLD

        for i in range(n):
            k_arr = X_phys[i]  # per-layer conductivities
            if not np.all(np.isfinite(k_arr)) or np.any(k_arr <= 0.0):
                raise RuntimeError(
                    f"{self.meta.case_id} produced a malformed, non-finite, or "
                    "non-physical series-resistance response; refusing silent "
                    "archive rows"
                )
            R_layers = delta / k_arr  # per-layer thermal resistances
            R_total = float(np.sum(R_layers))
            q_flux = dT / R_total if R_total > 0 else 0.0

            # Mid-layer interface temperature
            mid = N // 2
            T_mid = self._T_HOT - q_flux * float(np.sum(R_layers[:mid]))

            Y[i] = [R_total, q_flux, T_mid]
        if (
            Y.shape != (n, self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y[:, 0] <= 0.0)
            or np.any(Y[:, 1] <= 0.0)
            or np.any(Y[:, 2] < min(self._T_HOT, self._T_COLD))
            or np.any(Y[:, 2] > max(self._T_HOT, self._T_COLD))
        ):
            raise RuntimeError(
                f"{self.meta.case_id} produced a malformed, non-finite, or "
                "non-physical series-resistance response; refusing silent "
                "archive rows"
            )
        return Y


@register_case
class MultilayerWall8D(MultilayerWallBase):
    meta = _make_multilayer_meta(8)


@register_case
class MultilayerWall4D(MultilayerWallBase):
    meta = _make_multilayer_meta(4)


@register_case
class MultilayerWall16D(MultilayerWallBase):
    meta = _make_multilayer_meta(16)
