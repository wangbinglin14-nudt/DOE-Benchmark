"""
Case S01: Uniform rectangular cantilever beam — bending stress and deflection
=============================================================================

Engineering background
----------------------
A uniform rectangular cantilever beam is one of the most classical
benchmark problems in structural engineering.  One end is clamped; a
concentrated transverse force F and torque T are applied at the free end.
The critical cross-section (clamped end) develops bending normal stress
and shear stress; the free end exhibits transverse deflection and twist.

The problem appears in many engineering contexts:
  · Wing root loads (landing-gear bay)
  · Robotic arm holding a payload
  · Factory crane beam under moving loads

Response-surface features
-------------------------
Maximum bending stress:  σ_max = M_max · (h/2) / I_z = 6FL / (b·h²)

where:
  · F — tip concentrated force (transverse)
  · L — beam length
  · b — cross-section width
  · h — cross-section height

Note σ_max is inversely proportional to b and to h².
Thus b and h exhibit a **strong multiplicative interaction** (b·h² in
the denominator):
  · Increasing b reduces stress; increasing h is even more effective (h²)
  · Studying main effects only and ignoring interaction overestimates
    the relative importance of h

Maximum deflection:  δ_max = FL³ / (3EI_z) + FL/(κGA)
with κ = 5/6 and G = E/(2(1+ν)).
  · F and L affect deflection far more than the cross-section
  · h is more effective than b (h³ effect)

This gives the case:
  1. Analytically computable Sobol indices (ground truth for screening)
  2. Clear two-factor interaction (b × h²)
  3. Different variable-importance rankings for different outputs (σ vs δ)

Governing equations
-------------------
Euler–Bernoulli bending stress with Timoshenko shear-deflection correction
(small-deflection, uniform section, isotropic elastic):

  Bending moment:  M(x) = F·(L − x),  x ∈ [0, L] (x = 0 is clamped end)
  Section moment:  I_z = bh³/12
  Stress:          σ_max = M_max·(h/2)/I_z = 6FL/(bh²)
  Deflection:      δ_max = FL³/(3EI_z) + FL/(κGA)
  Shear stress:    τ_max = 3F/(2bh) (max at neutral axis, rectangular section)

Failure criterion (von Mises):
  σ_VM = √(σ_max² + 3·τ_max²) ≤ σ_yield / FS,  FS = 1.5

Input variables
---------------
F       Transverse tip force   [1, 50]    kN    Primary driver, linearly affects stress
L       Beam length            [0.5, 3.0] m     Cubic effect on deflection
b       Section width          [0.05, 0.3] m    Strong multiplicative interaction with h
h       Section height         [0.1, 0.5] m     Quadratic effect on stress, cubic on deflection
E       Young’s modulus        [60, 210]  GPa   Only affects deflection (linear), not stress

Output responses
----------------
sigma_max  Maximum bending stress      MPa   Critical-section outer fibre
delta_max  Maximum tip deflection      mm    Structural stiffness metric
tau_max    Neutral-axis shear stress   MPa   Rectangular section τ = 3F/(2bh)

DOE validation value
--------------------
VS-B (effect estimation):
    σ_max = 6FL/(bh²) has exactly known Sobol indices (analytically
    computable).  Tests whether resolution IV / V factorial designs
    correctly identify the b×h interaction without aliasing it to main
    effects (a common failure mode of resolution-III designs).

VS-D (small-sample efficiency):
    5-D problem; only 3 variables truly affect σ_max (F, b, h; L and E
    have little influence).  Can methods distinguish important from
    secondary variables at small sample sizes (N = 8–15)?

VS-H (robust design):
    F (load) is a Taguchi noise factor (±20 % uncertainty in practice);
    b and h are control factors.  Robust optimum: minimise variance of
    σ_max under F fluctuation while keeping the mean sufficiently low.

Sobol indices (σ_max output, SALib Saltelli n = 4096, uniform inputs)
---------------------------------------------------------------------
E has zero contribution to σ_max (absent from the formula).  h dominates
via the h⁻² effect (ST ≈ 0.67); F and b contribute similarly (ST ≈ 0.31);
L is secondary (ST ≈ 0.18).  See sobol_S1 / sobol_ST metadata fields.

References
----------
1. Timoshenko, S. P. & Goodier, J. N. (1951). Theory of Elasticity.
   McGraw-Hill, New York.
2. Young, W. C. & Budynas, R. G. (2002). Roark’s Formulas for Stress and
   Strain, 7th ed. McGraw-Hill.
3. Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer.
   Wiley, ISBN: 978-0-470-05997-5.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class CantileverBeam5D(BaseCase):
    """Uniform rectangular cantilever beam bending stress and deflection (5-D in, 3-D out).

    Analytical beam theory with Euler–Bernoulli bending stress and
    Timoshenko shear-deflection correction.
    Contains a strong multiplicative two-factor interaction (b×h²)
    and a sparse-effect structure (E has no influence on stress).

    Applicable to: DOE effect-estimation validation (VS-B),
    small-sample efficiency (VS-D), robust Taguchi design (VS-H).
    """

    meta = CaseMeta(
        case_id="cantilever_beam_5d",
        name="Rectangular cantilever beam bending stress and deflection (multiplicative interaction)",
        version="1.0.0",
        domain="structures",
        tags=["beam", "bending", "analytical", "interaction", "sparse", "taguchi"],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="F",
                symbol="F",
                description="Transverse tip concentrated force",
                unit="kN",
                lower=1.0,
                upper=50.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",  # can serve as Taguchi noise factor
            ),
            InputVariable(
                name="L",
                symbol="L",
                description="Beam length (clamped end to load point)",
                unit="m",
                lower=0.5,
                upper=3.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="b",
                symbol="b",
                description="Rectangular cross-section width",
                unit="m",
                lower=0.05,
                upper=0.30,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="h",
                symbol="h",
                description="Rectangular cross-section height",
                unit="m",
                lower=0.10,
                upper=0.50,
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
                lower=60.0,
                upper=210.0,
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
                description="Maximum bending stress at clamped end",
                unit="MPa",
                is_primary=True,
                expected_range=(0.0, 1800.0),
                failure_threshold=250.0,  # typical structural steel yield / safety factor
                failure_direction="above",
            ),
            OutputVariable(
                name="delta_max",
                symbol=r"\delta_{max}",
                description="Maximum tip transverse deflection",
                unit="mm",
                is_primary=True,
                expected_range=(0.0, 1810.0),
                failure_threshold=20.0,  # L/150 typical deflection limit (L_max=3m → ~20mm)
                failure_direction="above",
            ),
            OutputVariable(
                name="tau_max",
                symbol=r"\tau_{max}",
                description="Maximum shear stress at neutral axis (clamped end)",
                unit="MPa",
                is_primary=False,
                expected_range=(0.0, 100.0),
            ),
        ],
        governing_equation=(
            "Euler–Bernoulli bending + Timoshenko shear-deflection correction:\n"
            "  σ_max = 6FL/(bh²)  [MPa]\n"
            "  δ_max = FL³/(3EI_z) + FL/(κGA), κ=5/6, ν=0.30 [mm]\n"
            "  τ_max = 3F/(2bh)    [MPa]"
        ),
        pde_type="analytical",
        spatial_dim=1,
        is_steady=True,
        is_linear=False,
        physics_references=[
            "Timoshenko, S. P. & Goodier, J. N. (1951). Theory of Elasticity. McGraw-Hill.",
            "Young, W. C. & Budynas, R. G. (2002). Roark's Formulas for Stress and Strain, 7th ed. McGraw-Hill.",
        ],
        solver_library="numpy (analytical formulas, no numerical solve)",
        solver_details=(
            "Private _validate_physical_inputs rejects malformed, non-finite, "
            "or out-of-domain F/L/b/h/E physical arrays before direct analytical "
            "beam formula evaluation; Euler-Bernoulli bending stress with "
            "Timoshenko rectangular-section shear correction for deflection"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.01},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,  # analytical gradient not implemented
        supports_multifidelity=False,
        response_topology=["smooth_monotone", "strong_interaction"],
        effective_dim=4,  # E has no effect on sigma (effective dimension)
        important_variables=["F", "L", "h", "b"],
        interaction_pairs=[("b", "h"), ("F", "L")],
        has_failure_boundary=True,
        failure_condition="sigma_max > 250 MPa (exceeds typical structural-steel allowable stress)",
        # Sobol indices (sigma_max output, uniform inputs, SALib Saltelli n=4096)
        # h dominates (h² nonlinearity); F and b contribute similarly; L secondary; E = 0 (absent from σ formula)
        sobol_S1={"F": 0.099, "L": 0.048, "b": 0.091, "h": 0.339, "E": 0.000},
        sobol_ST={"F": 0.310, "L": 0.181, "b": 0.308, "h": 0.673, "E": 0.000},
        sobol_computation="salib_saltelli_n4096",
        applicable_scenarios=["VS-B", "VS-D", "VS-F", "VS-H"],
        scenario_notes={
            "VS-B": "σ = 6FL/bh² has exactly known effect structure: F, L are pure main effects; "
            "b and h have multiplicative interaction (denominator bh²). "
            "Resolution-III designs alias b×h interaction into main effects.",
            "VS-D": "In this 5-D problem, E contributes zero to stress (sparse effects); "
            "a standard test for identifying redundant variables at small sample sizes.",
            "VS-F": "E has ST ≈ 0; F/L/b/h all have significant ST. "
            "Screening methods should distinguish 'important' from 'unimportant' variables with N ≤ 20.",
            "VS-H": "F is the noise factor (±20 % load uncertainty); "
            "b/h are control factors — find optimal cross-section dimensions robust to load fluctuation.",
        },
        recommended_n_train={"default": 30},
        n_saturation=100,
        noise_factor_candidates=["F"],
        control_factor_candidates=["b", "h", "L"],
        reference_solution_available=True,
        reference_grid_size=10000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "cantilever_beam_5d physical inputs must be a 2-D array with "
                f"{self.meta.dim_input} columns"
            )
        if not np.all(np.isfinite(X_phys)):
            raise ValueError("cantilever_beam_5d physical inputs must be finite")

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
                "cantilever_beam_5d physical inputs outside the registered "
                "benchmark domain for F/L/b/h/E; "
                f"row {row}, {var.name}={X_phys[row, col]!r} not in "
                f"[{var.lower}, {var.upper}] {var.unit}"
            )
        return X_phys

    def _evaluate_core(
        self,
        X_phys: np.ndarray,
        fidelity: str = "default",
        **kwargs,
    ) -> np.ndarray:
        """Analytical beam calculation with shear-corrected tip deflection.

        Parameters
        ----------
        X_phys : ndarray, shape (n, 5)
            Columns: [F(kN), L(m), b(m), h(m), E(GPa)]

        Returns
        -------
        Y : ndarray, shape (n, 3)
            Columns: [sigma_max(MPa), delta_max(mm), tau_max(MPa)]
        """
        X_phys = self._validate_physical_inputs(X_phys)
        F = X_phys[:, 0] * 1e3  # kN → N
        L = X_phys[:, 1]  # m
        b = X_phys[:, 2]  # m
        h = X_phys[:, 3]  # m
        E = X_phys[:, 4] * 1e9  # GPa → Pa

        # Section moment of inertia I_z = bh³/12 (m⁴)
        I_z = b * h**3 / 12.0

        # Maximum bending moment (clamped end)
        M_max = F * L  # N·m

        # Maximum bending stress (Pa → MPa)
        sigma_max = M_max * (h / 2.0) / I_z  # Pa
        sigma_max_MPa = sigma_max * 1e-6  # MPa

        # Maximum tip deflection (m → mm): bending + Timoshenko shear term.
        nu = 0.30
        kappa = 5.0 / 6.0
        G = E / (2.0 * (1.0 + nu))
        A = b * h
        delta_bending = (F * L**3) / (3.0 * E * I_z)  # m
        delta_shear = F * L / (kappa * G * A)  # m
        delta_max = delta_bending + delta_shear
        delta_max_mm = delta_max * 1e3  # mm

        # Maximum neutral-axis shear stress (rectangular section: τ_max = 3F/(2A) = 3F/(2bh))
        tau_max = 3.0 * F / (2.0 * b * h)  # Pa
        tau_max_MPa = tau_max * 1e-6  # MPa

        Y = np.stack([sigma_max_MPa, delta_max_mm, tau_max_MPa], axis=1)
        if (
            Y.shape != (len(X_phys), self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y <= 0.0)
        ):
            raise RuntimeError(
                "cantilever_beam_5d produced a malformed, non-finite, or "
                "non-physical response array; refusing silent archive rows "
                "because the analytical beam formulas require positive "
                "stress and deflection outputs over the declared domain."
            )
        return Y
