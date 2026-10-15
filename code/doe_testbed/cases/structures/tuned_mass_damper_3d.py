"""
Case S08: Tuned mass damper (TMD) optimal design
=================================================

Engineering background
----------------------
A tuned mass damper (TMD) is a secondary mass–spring–damper system
attached to a primary structure to suppress resonant vibrations.
Applications include high-rise buildings (Taipei 101), long-span
bridges, and precision machinery.

The classical design problem (Den Hartog, 1956) is to choose the
TMD parameters (mass ratio, frequency ratio, damping ratio) so that
the peak dynamic amplification factor of the primary structure is
minimised across all excitation frequencies.

Governing equations (2-DOF forced vibration)
--------------------------------------------
Primary structure: mass M, stiffness K, natural frequency ω_n = √(K/M)
TMD: mass m_d = μ·M, stiffness k_d, damping c_d

Frequency ratio:  f = ω_d / ω_n = √(k_d/m_d) / ω_n
Damping ratio:    ζ_d = c_d / (2·m_d·ω_d)
Mass ratio:       μ = m_d / M

Steady-state displacement amplification of the primary mass under
harmonic force excitation F₀·exp(iΩt) (primary structure itself
undamped), expressed as |X_1/X_st| with X_st = F_0/K:

  H(r) = √(A² + B²) / √(C² + D²)

where r = Ω/ω_n is the excitation frequency ratio, and:
  A = f² − r²
  B = 2·ζ_d·f·r
  C = (f² − r²)·(1 − r²) − μ·r²·f²
  D = 2·ζ_d·f·r·(1 − r² − μ·r²)

This is the exact 2-DOF transfer function (see Rana & Soong 1998,
eq. 1a, or Den Hartog 1956, §3.3).  We evaluate it on a resonance-refined
frequency sweep r ∈ [0.01, 4.0] and extract the peak H_max.

Den Hartog optimal design (for the equal-peak criterion):
  f_opt    = 1 / (1 + μ)
  ζ_opt    = √[3μ / (8 (1 + μ))]
  H_max,opt = √(1 + 2/μ)

Outputs: H_max (peak amplification), Δf (−3 dB bandwidth),
         F_ratio = max_r(μ r² |X_2/X_st|) (peak TMD inertia force).

Input variables
---------------
mu       Mass ratio m_d/M         [0.005, 0.10]  -  Design (control)
f_ratio  Frequency ratio ω_d/ω_n  [0.70, 1.30]   -  Noise (manuf. tolerance)
zeta_d   TMD damping ratio         [0.01, 0.30]   -  Design (control)

Output responses
----------------
H_max      Peak displacement amplification factor  -
BW         Bandwidth (frequency range where H < H_max/√2)  -
F_ratio    Peak TMD force / static force (F_TMD_max / F_0)  -

DOE validation value
--------------------
VS-A (space-filling):
    The H(r) response has sharp resonance peaks that move with
    f_ratio; the response surface is highly heterogeneous in the
    (f_ratio, zeta_d) plane — a demanding test for space-filling.

VS-C (surrogate accuracy):
    The double-peak structure near f_ratio ≈ 1 creates a narrow
    saddle point that is difficult to capture with sparse samples.

VS-H (robust Taguchi):
    In practice, f_ratio drifts due to manufacturing tolerance
    (noise factor); the design goal is to choose (mu, zeta_d)
    so that H_max remains low despite f_ratio uncertainty.

References
----------
1. Den Hartog, J. P. (1956). Mechanical Vibrations, 4th ed. McGraw-Hill.
   Chapter 3, §3.3.
2. Rana, R. & Soong, T. T. (1998). Parametric study and simplified design
   of tuned mass dampers. Engineering Structures, 20(3), 193–204.
   DOI:10.1016/S0141-0296(97)00078-3.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class TunedMassDamper3D(BaseCase):
    """Tuned mass damper (TMD) 2-DOF vibration control, 3D → 3D."""

    meta = CaseMeta(
        case_id="tuned_mass_damper_3d",
        name="Tuned mass damper peak amplification (Den Hartog 2-DOF model)",
        version="1.0.0",
        domain="structures",
        tags=[
            "vibration",
            "tmd",
            "tuned_mass_damper",
            "resonance",
            "2dof",
            "3d",
            "analytical",
            "den_hartog",
        ],
        dim_input=3,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="mu",
                symbol=r"\mu",
                description="Mass ratio (TMD mass / primary mass)",
                unit="-",
                lower=0.005,
                upper=0.10,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="f_ratio",
                symbol="f",
                description="Frequency ratio (TMD frequency / primary frequency)",
                unit="-",
                lower=0.70,
                upper=1.30,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="zeta_d",
                symbol=r"\zeta_d",
                description="TMD damping ratio",
                unit="-",
                lower=0.01,
                upper=0.30,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="H_max",
                symbol="H_{max}",
                description="Peak displacement amplification factor (over all frequencies)",
                unit="-",
                is_primary=True,
                expected_range=(1.0, 100.0),
                failure_threshold=10.0,
                failure_direction="above",
            ),
            OutputVariable(
                name="BW",
                symbol=r"\Delta f",
                description="Half-power (-3 dB) bandwidth: width of the frequency band where H > H_max/sqrt(2)",
                unit="-",
                is_primary=True,
                expected_range=(0.0, 1.0),
            ),
            OutputVariable(
                name="F_ratio",
                symbol="F_{TMD}/F_0",
                description="Peak TMD force ratio (max TMD spring+damper force / excitation amplitude)",
                unit="-",
                is_primary=False,
                expected_range=(0.0, 50.0),
            ),
        ],
        governing_equation=(
            "2-DOF forced vibration (harmonic excitation):\n"
            "  H(r) = |X_1/X_st| (transfer function of primary mass)\n"
            "  r = Omega/omega_n (excitation frequency ratio)\n"
            "  H_max = max_r H(r)\n"
            "  Den Hartog optimal: f_opt = 1/(1+mu), zeta_opt = sqrt(3*mu/(8*(1+mu)))"
        ),
        pde_type="algebraic",
        spatial_dim=0,
        is_steady=True,
        # The 2-DOF equations are linear in displacement for fixed TMD
        # parameters, but the benchmark response is nonlinear in mu, f_ratio,
        # and zeta_d because it is a rational transfer-function peak over r.
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Den Hartog, J. P. (1956). Mechanical Vibrations, 4th ed. "
            "McGraw-Hill. Chapter 3, Section 3.3.",
            "Rana, R. & Soong, T. T. (1998). Parametric study and simplified "
            "design of tuned mass dampers. Eng. Struct., 20(3), 193-204. "
            "DOI:10.1016/S0141-0296(97)00078-3.",
        ],
        solver_library="numpy (analytical transfer function, frequency sweep)",
        solver_details=(
            "Private _validate_physical_inputs rejects malformed, non-finite, "
            "or out-of-domain mu/f_ratio/zeta_d physical arrays before "
            "Den Hartog 2-DOF transfer-function evaluation; "
            "Resonance-refined sweep r = 0.01..4.0 with dense sampling over "
            "0.5..1.5; exact 2-DOF transfer function with one protected "
            "denominator reused for primary response and TMD force"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.2},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_unimodal", "strong_interaction", "heterogeneous"],
        effective_dim=3,
        important_variables=["mu", "zeta_d", "f_ratio"],
        interaction_pairs=[("f_ratio", "zeta_d"), ("mu", "f_ratio")],
        has_failure_boundary=True,
        failure_condition="H_max > 10 (excessive vibration amplification; TMD mis-tuned or under-damped)",
        sobol_S1={"mu": 0.334, "f_ratio": 0.260, "zeta_d": 0.277},
        sobol_ST={"mu": 0.420, "f_ratio": 0.372, "zeta_d": 0.384},
        sobol_computation="salib_saltelli_n2048_seed42_H_max_output",
        applicable_scenarios=["VS-A", "VS-C", "VS-H"],
        scenario_notes={
            "VS-A": "H(r) has sharp double resonance peaks that shift with f_ratio; "
            "highly heterogeneous — space-filling designs must cover the narrow saddle region.",
            "VS-C": "The saddle point near f_ratio ~ 1 is a narrow nonlinear feature; "
            "surrogate accuracy is challenging with sparse samples.",
            "VS-H": "f_ratio is the noise factor (manufacturing tolerance); "
            "robust design: choose (mu, zeta_d) to minimise worst-case H_max "
            "under f_ratio uncertainty.",
        },
        recommended_n_train={"default": 30},
        n_saturation=120,
        noise_factor_candidates=["f_ratio"],
        control_factor_candidates=["mu", "zeta_d"],
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    # Number of frequency points in the resonance-sensitive core region.
    _N_FREQ_CORE = 1200
    _DEN_EPS = 1e-15

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "tuned_mass_damper_3d physical inputs must be a 2-D array "
                f"with {self.meta.dim_input} columns"
            )
        if not np.all(np.isfinite(X_phys)):
            raise ValueError("tuned_mass_damper_3d physical inputs must be finite")

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
                "tuned_mass_damper_3d physical inputs outside the registered "
                "benchmark domain for mu/f_ratio/zeta_d; "
                f"row {row}, {var.name}={X_phys[row, col]!r} not in "
                f"[{var.lower}, {var.upper}] {var.unit}"
            )
        return X_phys

    @classmethod
    def _frequency_grid(cls) -> np.ndarray:
        """Frequency-ratio grid, refined where TMD double peaks live."""
        return np.unique(
            np.concatenate([
                np.linspace(0.01, 0.5, 160, endpoint=False),
                np.linspace(0.5, 1.5, cls._N_FREQ_CORE, endpoint=False),
                np.linspace(1.5, 4.0, 320),
            ])
        )

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                Y[i] = self._tmd_response(
                    mu=float(X_phys[i, 0]),
                    f=float(X_phys[i, 1]),
                    zeta_d=float(X_phys[i, 2]),
                )
                if (
                    Y[i].shape != (self.meta.dim_output,)
                    or not np.all(np.isfinite(Y[i]))
                    or Y[i, 0] <= 0.0
                    or Y[i, 1] < 0.0
                    or Y[i, 2] < 0.0
                ):
                    raise ValueError(
                        "malformed, non-finite, or non-physical TMD response "
                        f"H_max/BW/F_ratio={np.array2string(Y[i], precision=6, separator=', ')}"
                    )
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "tuned_mass_damper_3d Den Hartog 2-DOF frequency-response "
                    f"solve failed for physical input {x_text}; refusing silent "
                    "NaN output because the formal benchmark response surface "
                    "requires finite amplification, bandwidth, and force-ratio "
                    "values."
                ) from None
        return Y

    def _tmd_response(self, mu: float, f: float, zeta_d: float) -> np.ndarray:
        """Compute peak amplification, bandwidth, and TMD force ratio.

        Uses the exact 2-DOF transfer function for a primary undamped SDOF
        structure with an attached TMD (mass ratio mu, frequency ratio f,
        damping ratio zeta_d) under harmonic force excitation.

        Transfer function (primary mass displacement / static displacement):
            H(r) = sqrt(A^2 + B^2) / sqrt(C^2 + D^2)
        where r = Omega/omega_n:
            A = f^2 - r^2
            B = 2*zeta_d*f*r
            C = (f^2 - r^2)*(1 - r^2) - mu*r^2*f^2
            D = 2*zeta_d*f*r*(1 - r^2 - mu*r^2)
        """
        r = self._frequency_grid()

        A = f**2 - r**2
        B = 2.0 * zeta_d * f * r
        C = (f**2 - r**2) * (1.0 - r**2) - mu * r**2 * f**2
        D = 2.0 * zeta_d * f * r * (1.0 - r**2 - mu * r**2)

        num = np.sqrt(A**2 + B**2)
        den = np.maximum(np.sqrt(C**2 + D**2), self._DEN_EPS)
        H = num / den

        # Peak amplification
        H_max = float(np.max(H))
        H_max = min(H_max, 100.0)

        # Bandwidth: frequency range where H > H_max / sqrt(2)
        threshold = H_max / np.sqrt(2.0)
        above = H > threshold
        if np.any(above) and not above[0] and not above[-1]:
            r_above = r[above]
            BW = float(r_above[-1] - r_above[0])
        else:
            BW = 0.0

        # TMD force ratio: |F_TMD / F_0| = m_d * Omega^2 * |X_2| / F_0
        # In nondimensional form with r = Omega/omega_n, F_0 = k * X_st:
        #   F_TMD/F_0 = mu * r^2 * |X_2 / X_st|
        # From the TMD equation: X_2/X_1 = (f^2 + i*B)/(A + i*B)
        # Therefore |X_2/X_st| = |X_2/X_1| * |X_1/X_st|
        #                      = sqrt(f^4 + B^2) / sqrt(A^2 + B^2)
        #                        * sqrt(A^2 + B^2) / sqrt(C^2 + D^2)
        #                      = sqrt(f^4 + B^2) / sqrt(C^2 + D^2)
        X2_over_Xst = np.sqrt(f**4 + B**2) / den
        F_ratio_curve = mu * r**2 * X2_over_Xst
        F_ratio = float(np.max(F_ratio_curve))
        F_ratio = min(F_ratio, 50.0)

        return np.array([H_max, BW, F_ratio])
