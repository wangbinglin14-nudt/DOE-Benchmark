"""
Case S03: Variable-section cantilever beam natural frequency (scalable, N segments)
===================================================================================

Engineering background
----------------------
The natural frequency of engineering structures (bridges, aircraft wings,
building floor slabs) is a core parameter in structural dynamics design.
Natural frequencies must be far from external excitation frequencies to
avoid resonance.

Real engineering structures usually have variable cross-sections:
  · Aircraft wings taper from root to tip (load-transfer requirement)
  · Bridge piers vary from base to top

This case simulates the fundamental natural frequency of an N-segment
variable-section cantilever beam (root clamped, tip free), with each
segment’s section height as an independent design variable.

This is a typical **high-dimensional scalable problem**: dimension = N
(can be 5 / 10 / 20).  Physically, segments near the root contribute
more to the first mode frequency (larger stiffness contribution), while
tip segments contribute less — a natural source of high-dimensional
sparse effects.

Physics model (FEM Euler–Bernoulli beam)
----------------------------------------
The cantilever is equally divided into N segments, each uniform with a
rectangular cross-section (width b = fixed, height h_i varies).

Bending stiffness per segment:  EI_i = E · b · h_i³ / 12

Hermite beam elements are used to assemble the global stiffness matrix K
and mass matrix M; solve the eigenvalue problem  K·φ = ω²·M·φ  and
extract the smallest eigenvalue ω_1.

Input variables (N segment heights)
-----------------------------------
h_i (i = 1..N)  Section height of each segment  [0.02, 0.15] m
Default N = 10; h_root near clamped end, h_tip near free end.

Fixed parameters: total length L = 2 m, width b = 0.1 m, E = 200 GPa
(steel), ρ = 7850 kg/m³.

Output responses
----------------
f_1     First natural frequency     Hz   Primary output (first bending mode)
f_2     Second natural frequency    Hz
mass    Total beam mass              kg

DOE validation value
--------------------
VS-E (high-dimensional adaptability):
    Dimension N = 5/10/20 is adjustable; effective dim ≈ N/2 (root
    segments more important).  A standard scalable benchmark for
    high-dimensional DOE methods.

VS-F (variable screening):
    Sobol ST: h_1 (root) >> h_N (tip).
    Screening methods should identify root segments as more important.

References
----------
1. Cook, R. D. et al. (2002). Concepts and Applications of Finite Element
   Analysis, 4th ed. Wiley. Chapter 11.
2. Inman, D. J. (2014). Engineering Vibration, 4th ed. Pearson.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case

_VBEAM_S1 = {
    5: {"h1": 0.400, "h2": 0.229, "h3": 0.086, "h4": 0.013, "h5": 0.075},
    10: {
        "h1": 0.244,
        "h2": 0.176,
        "h3": 0.134,
        "h4": 0.083,
        "h5": 0.047,
        "h6": 0.023,
        "h7": 0.009,
        "h8": 0.002,
        "h9": 0.014,
        "h10": 0.040,
    },
    20: {
        "h1": 0.140,
        "h2": 0.126,
        "h3": 0.105,
        "h4": 0.078,
        "h5": 0.057,
        "h6": 0.056,
        "h7": 0.041,
        "h8": 0.035,
        "h9": 0.025,
        "h10": 0.010,
        "h11": 0.010,
        "h12": 0.002,
        "h13": 0.003,
        "h14": 0.001,
        "h15": 0.003,
        "h16": 0.003,
        "h17": 0.003,
        "h18": 0.008,
        "h19": 0.014,
        "h20": 0.011,
    },
}
_VBEAM_ST = {
    5: {"h1": 0.576, "h2": 0.370, "h3": 0.183, "h4": 0.034, "h5": 0.094},
    10: {
        "h1": 0.378,
        "h2": 0.310,
        "h3": 0.234,
        "h4": 0.163,
        "h5": 0.101,
        "h6": 0.047,
        "h7": 0.017,
        "h8": 0.007,
        "h9": 0.018,
        "h10": 0.044,
    },
    20: {
        "h1": 0.238,
        "h2": 0.218,
        "h3": 0.167,
        "h4": 0.147,
        "h5": 0.131,
        "h6": 0.106,
        "h7": 0.079,
        "h8": 0.065,
        "h9": 0.040,
        "h10": 0.024,
        "h11": 0.015,
        "h12": 0.008,
        "h13": 0.003,
        "h14": 0.002,
        "h15": 0.002,
        "h16": 0.003,
        "h17": 0.005,
        "h18": 0.008,
        "h19": 0.012,
        "h20": 0.017,
    },
}


def _make_variable_beam_meta(N: int) -> CaseMeta:
    return CaseMeta(
        case_id=f"variable_beam_{N}d",
        name=f"Variable-section cantilever beam natural frequency ({N} segments, scalable)",
        version="1.0.0",
        domain="structures",
        tags=[
            "beam",
            "vibration",
            "fem",
            "variable_section",
            f"{N}d",
            "scalable",
            "natural_frequency",
        ],
        dim_input=N,
        dim_output=3,
        input_variables=[
            InputVariable(
                name=f"h{i+1}",
                symbol=f"h_{i+1}",
                description=f"Section height of segment {i+1} ({'root' if i==0 else 'tip' if i==N-1 else 'middle'})",
                unit="m",
                lower=0.02,
                upper=0.15,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            )
            for i in range(N)
        ],
        output_variables=[
            OutputVariable(
                name="f1",
                symbol="f_1",
                description="First natural frequency (bending)",
                unit="Hz",
                is_primary=True,
                expected_range=(0.5, 200.0),
            ),
            OutputVariable(
                name="f2",
                symbol="f_2",
                description="Second natural frequency",
                unit="Hz",
                is_primary=False,
                expected_range=(1.0, 2000.0),
            ),
            OutputVariable(
                name="mass",
                symbol="m",
                description="Total beam mass ρ·b·L·h̄ (h̄ = mean section height)",
                unit="kg",
                is_primary=False,
                # L=2m, b=0.1m, ρ=7850 kg/m³, h∈[0.02,0.15]
                # → mass ∈ [7850·2·0.1·0.02, 7850·2·0.1·0.15] = [31.4, 235.5]
                expected_range=(30.0, 250.0),
            ),
        ],
        governing_equation=(
            "Euler–Bernoulli beam FEM:\n"
            "  K·φ = ω²·M·φ (generalised eigenvalue problem)\n"
            "  EI_i = E·b·h_i³/12 (segment stiffness)\n"
            "  f_n = ω_n / (2π)"
        ),
        pde_type="algebraic",
        spatial_dim=1,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Cook, R. D. et al. (2002). Finite Element Analysis, 4th ed. Wiley. Ch.11.",
            "Inman, D. J. (2014). Engineering Vibration, 4th ed. Pearson.",
        ],
        solver_library="scipy.linalg.eigh (generalised symmetric eigenproblem)",
        solver_details=f"{N}-segment Hermite beam elements, 2N = {2*N} active DOF "
        f"(clamped-end constraint removes v_0, θ_0); scipy.linalg.eigh "
        f"with subset_by_index=[0,1] extracts the two lowest modes; "
        f"piecewise-constant segment mass/stiffness; named "
        f"_validate_physical_inputs private physical-input evaluation rejects "
        f"malformed arrays and segment heights outside the declared benchmark "
        f"domain before Hermite beam FEM assembly.",
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 1.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_monotone", "sparse_effects"],
        effective_dim=max(N // 2, 3),
        important_variables=[f"h{i+1}" for i in range(min(N // 2 + 1, N))],
        interaction_pairs=[("h1", "h2"), ("h1", f"h{min(3,N)}")],
        has_failure_boundary=False,
        sobol_S1=_VBEAM_S1.get(N),
        sobol_ST=_VBEAM_ST.get(N),
        sobol_computation="salib_saltelli_n2048_seed42_f1_output",
        applicable_scenarios=["VS-E", "VS-F"],
        scenario_notes={
            "VS-E": f"{N}-D scalable problem, effective dim ≈ {N//2}; root segment h1 has the largest influence.",
            "VS-F": "Sobol ST: h_1 (root) >> h_N (tip); tests screening for spatial-position effects.",
        },
        recommended_n_train={"default": N * 5},
        n_saturation=N * 30,
        noise_factor_candidates=[],
        control_factor_candidates=[f"h{i+1}" for i in range(N)],
        reference_solution_available=True,
        reference_grid_size=3000,
        scalable_dimension=True,
        scalable_dimension_range=(5, 20),
    )


class VariableBeamBase(BaseCase):
    """Variable-section cantilever beam natural frequency base class (FEM, Hermite elements)."""

    # Fixed material and geometry parameters
    _E = 200e9  # Pa (steel)
    _RHO = 7850.0  # kg/m³
    _B = 0.1  # m width (fixed)
    _L = 2.0  # m total beam length

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                f"{self.meta.case_id} expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(X_phys < 0.02 - tol)
            or np.any(X_phys > 0.15 + tol)
        ):
            raise ValueError(
                f"{self.meta.case_id} physical inputs must stay inside the "
                "declared segment-height benchmark domain before Hermite beam "
                "FEM assembly"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        N = self.meta.dim_input
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                Y[i] = self._fem_beam(X_phys[i], N)
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    f"{self.meta.case_id} Hermite beam FEM eigen solve failed "
                    f"for physical input {x_text}; refusing silent NaN output "
                    "because the formal benchmark response surface requires "
                    "finite natural frequencies and mass."
                ) from None
        return Y

    def _fem_beam(self, h_arr, N):
        """Hermite beam-element FEM, solve for natural frequencies.

        N-segment cantilever, 1 Hermite element per segment (4 DOF each:
        v1, θ1, v2, θ2).  Total DOF = (N+1)×2; clamped end (v1, θ1) = 0
        → effective DOF = N×2.
        """
        from scipy.linalg import eigh

        E = self._E
        b = self._B
        rho = self._RHO
        L = self._L
        le = L / N  # segment length

        total_dof = (N + 1) * 2
        K_global = np.zeros((total_dof, total_dof))
        M_global = np.zeros((total_dof, total_dof))

        for seg in range(N):
            h = float(h_arr[seg])
            I = b * h**3 / 12.0
            A = b * h
            EI = E * I
            rhoA = rho * A

            # Hermite element stiffness matrix (4×4)
            k = (
                EI
                / le**3
                * np.array(
                    [
                        [12, 6 * le, -12, 6 * le],
                        [6 * le, 4 * le**2, -6 * le, 2 * le**2],
                        [-12, -6 * le, 12, -6 * le],
                        [6 * le, 2 * le**2, -6 * le, 4 * le**2],
                    ]
                )
            )
            # Consistent mass matrix
            m = (
                rhoA
                * le
                / 420.0
                * np.array(
                    [
                        [156, 22 * le, 54, -13 * le],
                        [22 * le, 4 * le**2, 13 * le, -3 * le**2],
                        [54, 13 * le, 156, -22 * le],
                        [-13 * le, -3 * le**2, -22 * le, 4 * le**2],
                    ]
                )
            )

            # Assemble (each segment from seg*2 to (seg+2)*2)
            dofs = [seg * 2, seg * 2 + 1, (seg + 1) * 2, (seg + 1) * 2 + 1]
            for a_i, a in enumerate(dofs):
                for b_i, b_d in enumerate(dofs):
                    K_global[a, b_d] += k[a_i, b_i]
                    M_global[a, b_d] += m[a_i, b_i]

        # Apply clamped BC (remove DOF 0 and 1)
        free = list(range(2, total_dof))
        K_red = K_global[np.ix_(free, free)]
        M_red = M_global[np.ix_(free, free)]

        # Solve eigenvalue problem
        eigvals = eigh(K_red, M_red, eigvals_only=True, subset_by_index=[0, 1])
        eigvals = self._sanitize_eigenvalues(eigvals, K_red, M_red)
        omega = np.sqrt(eigvals)
        freqs = omega / (2 * np.pi)

        # Total mass
        mass = rho * b * np.sum(h_arr) * (L / N)

        return np.array([freqs[0], freqs[1], mass])

    @staticmethod
    def _sanitize_eigenvalues(eigvals, K_red, M_red):
        eigvals = np.asarray(eigvals, dtype=float)
        if not np.all(np.isfinite(eigvals)):
            raise ValueError("Non-finite generalized eigenvalue in variable beam solve")

        scale = max(
            1.0,
            float(np.linalg.norm(K_red, ord=np.inf))
            / max(float(np.linalg.norm(M_red, ord=np.inf)), 1e-300),
        )
        tol = 1000.0 * np.finfo(float).eps * scale
        min_eig = float(np.min(eigvals))
        if min_eig < -tol:
            raise ValueError(f"Negative generalized eigenvalue {min_eig:.6e} below tolerance {tol:.6e}")

        return np.where(eigvals < 0.0, 0.0, eigvals)


# Register default 10-D version
@register_case
class VariableBeam10D(VariableBeamBase):
    meta = _make_variable_beam_meta(10)


@register_case
class VariableBeam5D(VariableBeamBase):
    meta = _make_variable_beam_meta(5)


@register_case
class VariableBeam20D(VariableBeamBase):
    meta = _make_variable_beam_meta(20)
