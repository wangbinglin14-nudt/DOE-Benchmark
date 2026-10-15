"""
Case T01: Heterogeneous 2-D steady-state heat conduction
=========================================================

Engineering background
----------------------
Heat conduction in heterogeneous media arises widely in:
  · Electronic-component packaging (substrates, copper interconnects, and
    moulding compounds with different thermal conductivities)
  · Building-wall thermal-bridge analysis (steel reinforcement crossing
    insulation layers creates localised high-conductivity paths)
  · Aerospace thermal-protection systems (ceramic insulation + metallic
    structural layer with graded conductivity)

This case simulates a square computational domain (1 m × 1 m) divided into
a left low-conductivity region (region A) and a right high-conductivity
region (region B), with an interface along the x-direction.
Geometric parameter x_split controls the interface position; k_A and k_B
are the thermal conductivities of the two sides.  With the current
bottom/top Dirichlet and left/right adiabatic boundary conditions, the
temperature solution is the one-dimensional vertical profile; x_split is
retained as an intentionally inert geometric parameter in v1.0.0.
Boundary conditions: bottom Dirichlet (T = T_hot), top Dirichlet
(T = T_cold), left and right adiabatic (Neumann, ∂T/∂n = 0).

Response-surface characteristics
---------------------------------
Key outputs: area-weighted mean temperature T_mid at x = 0.5 (mid-line)
and maximum heat-flux density q_max.

Response-surface features:
  · T_mid is controlled by the imposed bottom/top boundary temperatures.
  · q_max is controlled by the larger local conductivity and the imposed
    temperature difference.
  · x_split is inert for the current local-output definitions because the
    exact solution is vertical and both material regions remain present over
    the declared input domain.  It is kept as a ground-truth zero-effect
    geometry variable for screening checks.

Governing equation
------------------
2-D steady-state heat-conduction equation (no internal heat source):

  ∇·(k(x,y)·∇T) = 0,  (x,y) ∈ Ω = [0,1]²

where k(x,y) is the heterogeneous conductivity field:
  k(x,y) = k_A,  x < x_split
  k(x,y) = k_B,  x ≥ x_split

Boundary conditions:
  T(x, 0) = T_hot    (bottom, high-temperature Dirichlet)
  T(x, 1) = T_cold   (top, low-temperature Dirichlet)
  ∂T/∂x(0, y) = 0   (left, adiabatic Neumann)
  ∂T/∂x(1, y) = 0   (right, adiabatic Neumann)

Numerical method: FDM on a uniform structured N×N grid.
Fidelity levels:
  'default' → 25×25 grid (~5 ms)
  'high'    → 60×60 grid (~80 ms, for multi-fidelity studies)

This problem has an analytical benchmark under the declared boundary
conditions:
  T(x,y) = T_hot − (T_hot − T_cold) y
  T_mid = (T_hot + T_cold)/2
  q_max = max(k_A, k_B) (T_hot − T_cold)
provided both materials are present.  The FDM solution is used to exercise
heterogeneous-coefficient assembly and face-consistent flux extraction.

Input variables
---------------
k_A       Left-region conductivity   [0.1, 5.0]   W/(m·K)   Insulation material
k_B       Right-region conductivity  [1.0, 50.0]  W/(m·K)   Conductive material (10–500× k_A)
x_split   Interface x-coordinate     [0.2, 0.8]   m         Controls relative region area
T_hot     Bottom temperature         [60, 120]    °C        Heat-source temperature
T_cold    Top temperature            [0, 30]      °C        Cold-side temperature

Output responses
----------------
T_mid      Mid-line (y = 0.5) area-weighted mean temperature   °C
q_max      Maximum heat-flux density (entire domain)           W/m²
T_hot_surf Bottom mid-point (x = 0.5, y = 0) temperature      °C   BC consistency check

DOE validation value
--------------------
VS-A (space filling):
    When k_B/k_A is large, the local heat-flux field is highly heterogeneous
    even though the temperature field is one-dimensional.  The case tests
    whether designs cover the high-conductivity flux regime.

VS-F (variable screening):
    x_split is a declared geometric input but inert for the current boundary
    conditions and local outputs.  Screening methods should identify it as a
    zero-effect variable.

VS-C (surrogate accuracy):
    5-D problem with two parameters of different orders of magnitude
    (k_A vs k_B); the log-scale k_B makes the parameter space highly
    anisotropic in linear coordinates — a good case for testing surrogate-
    model adaptability to non-uniform parameter scales.

References
----------
1. Carslaw, H. S. & Jaeger, J. C. (1959). Conduction of Heat in Solids, 2nd ed.
   Oxford University Press.
2. Incropera, F. P. et al. (2007). Fundamentals of Heat and Mass Transfer, 6th ed.
   Wiley, ISBN: 978-0-471-45728-2.
3. LeVeque, R. J. (2007). Finite Difference Methods for Ordinary and Partial
   Differential Equations. SIAM, DOI: 10.1137/1.9780898717839.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class HeterogeneousConduction5D(BaseCase):
    """Heterogeneous 2-D steady-state heat conduction (dual-material, vertical thermal bridge).

    5-D input → 3-D output (T_mid, q_max, T_hot_surf).

    Numerical method: FDM, default 25×25 grid; high-fidelity 60×60.
    Supports multi-fidelity (fidelity='default'|'high').
    """

    meta = CaseMeta(
        case_id="hetero_conduction_2d",
        name="Heterogeneous 2-D steady-state heat conduction (thermal-bridge effect)",
        version="1.0.0",
        domain="heat_transfer",
        tags=[
            "heat_conduction",
            "heterogeneous",
            "fdm",
            "2d",
            "steady_state",
            "thermal_bridge",
            "interaction",
        ],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="k_A",
                symbol="k_A",
                description="Left-region thermal conductivity",
                unit="W/(m·K)",
                lower=0.1,
                upper=5.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="k_B",
                symbol="k_B",
                description="Right-region thermal conductivity (can be much larger than k_A)",
                unit="W/(m·K)",
                lower=1.0,
                upper=50.0,
                log_scale=True,  # log normalisation
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="x_split",
                symbol="x_{split}",
                description="Interface x-coordinate (0 = left boundary, 1 = right boundary)",
                unit="m",
                lower=0.2,
                upper=0.8,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="T_hot",
                symbol="T_{hot}",
                description="Bottom (hot-side) temperature boundary condition",
                unit="°C",
                lower=60.0,
                upper=120.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",  # in practice, heat-source temperature often fluctuates
            ),
            InputVariable(
                name="T_cold",
                symbol="T_{cold}",
                description="Top (cold-side) temperature boundary condition",
                unit="°C",
                lower=0.0,
                upper=30.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="T_mid",
                symbol="T_{mid}",
                description="Area-weighted mean temperature at the domain mid-line (y = 0.5)",
                unit="°C",
                is_primary=True,
                expected_range=(0.0, 120.0),
            ),
            OutputVariable(
                name="q_max",
                symbol="q_{max}",
                description="Maximum local heat-flux density in the domain",
                unit="W/m²",
                is_primary=True,
                expected_range=(0.0, 6500.0),
            ),
            OutputVariable(
                name="T_hot_surf",
                symbol="T_{hot,surf}",
                description="Bottom mid-point (x = 0.5, y = 0) temperature (BC verification)",
                unit="°C",
                is_primary=False,
                expected_range=(0.0, 120.0),
            ),
        ],
        governing_equation=(
            "∇·(k(x,y)·∇T) = 0  [2-D steady state]\n"
            "k(x,y) = k_A (x < x_split), k_B (x ≥ x_split)\n"
            "BC: T(x,0) = T_hot, T(x,1) = T_cold, left/right adiabatic"
        ),
        pde_type="elliptic",
        spatial_dim=2,
        is_steady=True,
        # The fixed-coefficient conduction PDE is linear in temperature, but
        # q_max is a piecewise product of conductivity and temperature drop;
        # the benchmark input-output response is therefore not globally linear.
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Carslaw, H. S. & Jaeger, J. C. (1959). Conduction of Heat in Solids, 2nd ed. Oxford.",
            "Incropera, F. P. et al. (2007). Fundamentals of Heat and Mass Transfer, 6th ed. Wiley.",
            "LeVeque, R. J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM.",
        ],
        solver_library="scipy.sparse.linalg (FDM sparse linear system)",
        solver_details=(
            "Private _validate_physical_inputs rejects malformed, non-finite, "
            "or out-of-domain k_A/k_B/x_split/T_hot/T_cold physical arrays "
            "before heterogeneous-conduction FDM assembly; "
            "Uniform orthogonal FDM grid, default 25×25; high-fidelity 60×60; "
            "harmonic-mean k at material interfaces ensures heat-flux continuity; "
            "q_max is extracted from face-consistent fluxes rather than cross-interface central gradients"
        ),
        fidelity_levels=["default", "high"],
        typical_eval_time_ms={"default": 5.0, "high": 80.0},
        cost_level="L1",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=True,
        response_topology=["smooth_monotone", "threshold_jump", "heterogeneous"],
        effective_dim=4,
        important_variables=["T_hot", "T_cold", "k_A", "k_B"],
        interaction_pairs=[("T_hot", "k_A"), ("T_hot", "k_B"), ("T_cold", "k_A"), ("T_cold", "k_B")],
        has_failure_boundary=False,
        sobol_S1={"k_A": 0.000, "k_B": 0.000, "x_split": 0.000, "T_hot": 0.800, "T_cold": 0.200},
        sobol_ST={"k_A": 0.000, "k_B": 0.000, "x_split": 0.000, "T_hot": 0.800, "T_cold": 0.200},
        sobol_computation="salib_saltelli_n1024_seed42_T_mid_output",
        applicable_scenarios=["VS-A", "VS-C", "VS-F"],
        scenario_notes={
            "VS-A": "When k_B/k_A is large, local heat flux concentrates in the high-conductivity region. "
            "Space-filling methods should cover the high-flux, high-k regime.",
            "VS-C": "k_B (log scale) + k_A (linear scale) creates an anisotropic parameter space; "
            "tests surrogate-model adaptability to multi-scale parameter spaces.",
            "VS-F": "x_split is a retained inert geometry variable under the current boundary conditions and local outputs; "
            "screening methods should assign it essentially zero importance.",
        },
        recommended_n_train={"default": 60, "high": 200},
        n_saturation=250,
        noise_factor_candidates=["T_hot", "T_cold"],
        control_factor_candidates=["k_A", "k_B", "x_split"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    # ── Core simulation ───────────────────────────────────────────────────────

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "hetero_conduction_2d physical inputs must be a 2-D array with "
                f"{self.meta.dim_input} columns"
            )
        if not np.all(np.isfinite(X_phys)):
            raise ValueError("hetero_conduction_2d physical inputs must be finite")

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
                "hetero_conduction_2d physical inputs outside the registered "
                "benchmark domain for k_A/k_B/x_split/T_hot/T_cold; "
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
        """FDM solution of 2-D steady-state heat conduction.

        Parameters
        ----------
        X_phys : ndarray, shape (n, 5)
            Column order: [k_A, k_B, x_split, T_hot, T_cold]
        """
        X_phys = self._validate_physical_inputs(X_phys)
        n_grid = 25 if fidelity == "default" else 60
        n_samples = X_phys.shape[0]
        Y = np.full((n_samples, 3), np.nan)

        for i in range(n_samples):
            try:
                Y[i] = self._solve_single(
                    k_A=X_phys[i, 0],
                    k_B=X_phys[i, 1],
                    x_split=X_phys[i, 2],
                    T_hot=X_phys[i, 3],
                    T_cold=X_phys[i, 4],
                    n_grid=n_grid,
                )
                if (
                    Y[i].shape != (self.meta.dim_output,)
                    or not np.all(np.isfinite(Y[i]))
                    or Y[i, 0] < float(X_phys[i, 4]) - 1e-8
                    or Y[i, 0] > float(X_phys[i, 3]) + 1e-8
                    or Y[i, 1] < 0.0
                    or not np.isclose(Y[i, 2], float(X_phys[i, 3]), rtol=1e-10, atol=1e-8)
                ):
                    raise ValueError(
                        "malformed, non-finite, or non-physical heterogeneous "
                        "conduction response "
                        f"T_mid/q_max/T_hot_surf={np.array2string(Y[i], precision=6, separator=', ')}"
                    )
            except Exception as exc:
                raise RuntimeError(
                    "hetero_conduction_2d refusing silent NaN after "
                    f"heterogeneous-conduction FDM failure for k_A={X_phys[i, 0]!r}, "
                    f"k_B={X_phys[i, 1]!r}, x_split={X_phys[i, 2]!r}, "
                    f"T_hot={X_phys[i, 3]!r}, T_cold={X_phys[i, 4]!r}, "
                    f"fidelity={fidelity!r}"
                ) from exc
        return Y

    @staticmethod
    def _solve_single(
        k_A: float,
        k_B: float,
        x_split: float,
        T_hot: float,
        T_cold: float,
        n_grid: int = 25,
    ) -> np.ndarray:
        """FDM solution for a single parameter point.

        Numerical scheme:
          Heterogeneous scheme (harmonic-mean k at interface):
            Between the i-th and (i+1)-th nodes in the x-direction,
            use the harmonic-mean conductivity k_{i+1/2} = 2 k_i k_{i+1} / (k_i + k_{i+1}).
          This ensures normal heat-flux continuity (critical for heterogeneous-conductivity problems).

          Discrete equation (interior nodes):
            k_{i-1/2} T_{i-1,j} + k_{i+1/2} T_{i+1,j}
          + k_{j-1/2} T_{i,j-1} + k_{j+1/2} T_{i,j+1}
          - (k_{i-1/2} + k_{i+1/2} + k_{j-1/2} + k_{j+1/2}) T_{i,j} = 0

          where k_{i±1/2} is the x-direction harmonic mean, k_{j±1/2} the y-direction harmonic mean.
          The y-direction k is uniform (no sub-regions), so k_{j±1/2} = k(x_i).
        """
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve

        N = n_grid  # nodes per direction (including boundaries)
        h = 1.0 / (N - 1)  # grid spacing

        k_A = max(float(k_A), 1e-6)
        k_B = max(float(k_B), 1e-6)

        # Generate k field (x-direction)
        x_nodes = np.linspace(0, 1, N)
        k_x = np.where(x_nodes < x_split, k_A, k_B)  # shape (N,)

        # Generate 2-D k field: k[i, j] = k_x[i] (independent of y)
        # Node numbering: node(i,j) = i*N + j, i = x-direction, j = y-direction
        n_total = N * N
        A_mat = lil_matrix((n_total, n_total))
        b_vec = np.zeros(n_total)

        def idx(i, j):
            return i * N + j

        for i in range(N):
            for j in range(N):
                node = idx(i, j)

                # Top boundary (j = N-1): T = T_cold
                if j == N - 1:
                    A_mat[node, node] = 1.0
                    b_vec[node] = T_cold
                    continue
                # Bottom boundary (j = 0): T = T_hot
                if j == 0:
                    A_mat[node, node] = 1.0
                    b_vec[node] = T_hot
                    continue
                # Left boundary (i = 0): ∂T/∂x = 0 → T[0,j] = T[1,j]
                if i == 0:
                    A_mat[node, node] = 1.0
                    A_mat[node, idx(1, j)] = -1.0
                    continue
                # Right boundary (i = N-1): ∂T/∂x = 0 → T[N-1,j] = T[N-2,j]
                if i == N - 1:
                    A_mat[node, node] = 1.0
                    A_mat[node, idx(N - 2, j)] = -1.0
                    continue

                # Interior node: FDM discretisation
                # x-direction harmonic-mean k
                k_xm = 2 * k_x[i - 1] * k_x[i] / (k_x[i - 1] + k_x[i] + 1e-30)
                k_xp = 2 * k_x[i] * k_x[i + 1] / (k_x[i] + k_x[i + 1] + 1e-30)
                # y-direction k is uniform (equals k_x[i])
                k_ym = k_x[i]
                k_yp = k_x[i]

                coeff = -(k_xm + k_xp + k_ym + k_yp)
                A_mat[node, node] = coeff
                A_mat[node, idx(i - 1, j)] = k_xm
                A_mat[node, idx(i + 1, j)] = k_xp
                A_mat[node, idx(i, j - 1)] = k_ym
                A_mat[node, idx(i, j + 1)] = k_yp

        T = spsolve(A_mat.tocsr(), b_vec).reshape(N, N)

        # ── Extract output quantities ────────────────────────────────────────────
        # T_mid: mean temperature at y = 0.5 (j = N//2) across all x-nodes
        j_mid = N // 2
        T_mid = float(np.mean(T[:, j_mid]))

        # q_max: maximum heat-flux density from face-consistent Fourier fluxes.
        # This avoids applying a central-difference gradient across a material
        # jump, where the physical temperature gradient is discontinuous.
        q_max = HeterogeneousConduction5D._max_face_flux(T, k_x, h)

        # T_hot_surf: (x = 0.5, y = 0) = T[N//2, 0]
        T_hot_surf = float(T[N // 2, 0])

        return np.array([T_mid, q_max, T_hot_surf])

    @staticmethod
    def _max_face_flux(T: np.ndarray, k_x: np.ndarray, h: float) -> float:
        """Maximum heat-flux magnitude using conductivity defined on faces."""
        k_x = np.maximum(np.asarray(k_x, dtype=float), 1e-12)

        k_face_x = 2.0 * k_x[:-1] * k_x[1:] / (k_x[:-1] + k_x[1:])
        qx_faces = -k_face_x[:, None] * (T[1:, :] - T[:-1, :]) / h

        qy_faces = -k_x[:, None] * (T[:, 1:] - T[:, :-1]) / h

        return float(max(np.max(np.abs(qx_faces)), np.max(np.abs(qy_faces))))
