r"""
Case S04: Ten-bar planar truss (nodal displacements and stresses)
=================================================================

Engineering background
----------------------
The ten-bar truss is the most classical planar truss teaching example in
structural engineering.  It consists of 10 bars whose cross-sectional
areas are design variables.  Under given loads, nodal displacements and
bar stresses are computed.

This is a standard case for high-dimensional DOE validation (VS-E) and
variable screening (VS-F):
  · 10 design variables (10 bar cross-sectional areas), effective dim ≈ 3–4
  · Primary load-carrying bars (diagonal braces + bottom chord) contribute
    most to displacements
  · Secondary bars (vertical members) contribute little

Structural geometry (three-bay cantilever 10-bar truss)
-------------------------------------------------------
        4   5   6
        ●───●───●           (nodes 4, 5, 6 are on the top chord y = 9.144 m)
        │\  │\  │
        │ \ │ \ │           All bay widths = bay height = 9.144 m (30 ft)
        │  \│  \│
        ●───●───●───●       (nodes 1, 2, 3 are on the bottom chord y = 0)
        1   2   3   7       (node 7: lower-right pin support)

    Supports (1-based node indexing used in the diagram):
      node 4: upper-left pin  (fully restrained)
      node 7: lower-right pin (fully restrained)

    Node coordinates (m):
      1:(0,0), 2:(9.144,0), 3:(18.288,0), 7:(27.432,0)       bottom row
      4:(0,9.144), 5:(9.144,9.144), 6:(18.288,9.144)         top row

    Loads: Fy = −2×10⁵ N applied at node 2 and node 3 (vertical, gravity sense).

    Bar list (cross-sectional areas A1..A10 are the 10 design variables):
      A1:  1–2   — bottom chord (left bay)
      A2:  2–3   — bottom chord (centre bay)
      A3:  3–7   — bottom chord (right bay, to support)
      A4:  4–5   — top chord (left bay)
      A5:  5–6   — top chord (right bay)
      A6:  1–5   — diagonal: lower-left → upper-centre
      A7:  2–5   — vertical: lower-centre → upper-centre
      A8:  2–6   — diagonal: lower-centre → upper-right
      A9:  3–6   — vertical: lower-right → upper-right
      A10: 4–2   — diagonal: upper-left (support) → lower-centre

    In the 0-based indexing used by `_NODES` and `_ELEMENTS` below, node
    i (1-based) maps to `_NODES[i-1]`.

Input variables (10 bar cross-sectional areas)
----------------------------------------------
A_i (i = 1..10)  Bar cross-sectional area  [0.645, 25.8] cm²

Output responses
----------------
u_max    Maximum nodal horizontal displacement  mm  Primary output
v_max    Maximum nodal vertical displacement    mm  Primary output (usually the key design constraint)
mass     Total structural mass                  kg

DOE validation value
--------------------
VS-E (high-dimensional adaptability): 10-D standard truss, effective dim ≈ 4.
VS-F (variable screening):
    Bottom-chord bars (A2, A3) and main diagonals (A8, A10) contribute most
    to vertical displacement in the current FEM/Sobol metadata; secondary
    bars such as A4, A5, and A9 have small influence.

References
----------
1. Topping, B. H. V. (1983). Shape optimization of skeletal structures.
   J. Struct. Eng., 109(8).
2. Arora, J. S. (2004). Introduction to Optimum Design, 2nd ed. Elsevier.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class TenBarTruss10D(BaseCase):
    """Ten-bar planar truss (10D → 3D, FEM static analysis)."""

    # Structural geometry (6 active nodes + 1 support node, in metres)
    _NODES = np.array(
        [
            [0.0, 0.0],  # 0: lower-left
            [9.144, 0.0],  # 1: lower-centre
            [18.288, 0.0],  # 2: lower-right (loaded node)
            [0.0, 9.144],  # 3: upper-left (pin support)
            [9.144, 9.144],  # 4: upper-centre
            [18.288, 9.144],  # 5: upper-right
            [27.432, 0.0],  # 6: right-end pin support
        ]
    )
    # Bar connectivity (node pairs)
    _ELEMENTS = [
        (0, 1),
        (1, 2),
        (2, 6),  # bottom chord A1, A2, A3
        (3, 4),
        (4, 5),  # top chord A4, A5
        (0, 4),
        (1, 4),
        (1, 5),
        (2, 5),  # diagonals A6, A7, A8, A9
        (3, 1),  # diagonal A10
    ]
    # Fixed supports (node 3: upper-left pin, node 6: lower-right pin)
    _SUPPORTS = {3: [True, True], 6: [True, True]}
    # External loads (node index, force [Fx, Fy], unit N)
    _LOADS = {1: [0.0, -2e5], 2: [0.0, -2e5]}
    _E = 68.9e9  # Pa (aluminium alloy)
    _RHO = 2700.0  # kg/m³

    meta = CaseMeta(
        case_id="ten_bar_truss_10d",
        name="Ten-bar planar truss (nodal displacements and stresses)",
        version="1.0.0",
        domain="structures",
        tags=[
            "truss",
            "10bar",
            "structural_fem",
            "10d",
            "displacement",
            "stress",
            "high_dimension",
        ],
        dim_input=10,
        dim_output=3,
        input_variables=[
            InputVariable(
                name=f"A{i+1}",
                symbol=f"A_{i+1}",
                description=f"Cross-sectional area of bar {i+1}",
                unit="cm²",
                lower=0.645,
                upper=25.8,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            )
            for i in range(10)
        ],
        output_variables=[
            OutputVariable(
                name="u_max",
                symbol="u_{max}",
                description="Maximum nodal horizontal displacement (absolute value)",
                unit="mm",
                is_primary=False,
                # Thin-bar samples (A ≈ 0.1 in²) can produce metre-scale displacements;
                # this is precisely the DOE screening purpose (identify failure region).
                expected_range=(0.0, 2500.0),
            ),
            OutputVariable(
                name="v_max",
                symbol="v_{max}",
                description="Maximum nodal vertical displacement (absolute value, usually downward)",
                unit="mm",
                is_primary=True,
                # Failure threshold 50 mm is an engineering constraint; exceeding it
                # is treated as structural failure.  expected_range covers the full
                # physical output interval to accommodate failure samples.
                expected_range=(0.0, 10000.0),
                failure_threshold=50.0,
                failure_direction="above",
            ),
            OutputVariable(
                name="mass",
                symbol="m",
                description="Total truss mass",
                unit="kg",
                is_primary=False,
                expected_range=(17.0, 800.0),
            ),
        ],
        governing_equation=(
            "Truss FEM: K·u = F\n"
            "  K_e = (EA/L)·[1,−1;−1,1] (each bar's local stiffness)\n"
            "  Global stiffness K = Σ Tᵀ·K_e·T (coordinate transformation)"
        ),
        pde_type="algebraic",
        spatial_dim=2,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Topping, B. H. V. (1983). Shape optimization of skeletal structures. J. Struct. Eng.",
            "Arora, J. S. (2004). Introduction to Optimum Design, 2nd ed. Elsevier.",
        ],
        solver_library="numpy.linalg (FEM direct solve)",
        solver_details=(
            "Private _validate_physical_inputs rejects malformed, non-finite, "
            "or out-of-domain A1-A10 physical arrays before direct-stiffness "
            "FEM assembly; planar truss FEM, 14×14 global stiffness matrix, "
            "constrained DOF elimination to a 10×10 free-DOF direct solve"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.5},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_monotone", "sparse_effects"],
        effective_dim=4,
        important_variables=["A2", "A3", "A10", "A8"],
        interaction_pairs=[],  # purely additive on v_max (ST ≈ S1 for all bars)
        has_failure_boundary=True,
        failure_condition="v_max > 50 mm (deflection exceeds limit)",
        sobol_S1={
            "A1": 0.000,
            "A2": 0.316,
            "A3": 0.312,
            "A4": 0.008,
            "A5": 0.009,
            "A6": 0.000,
            "A7": 0.000,
            "A8": 0.070,
            "A9": 0.009,
            "A10": 0.280,
        },
        sobol_ST={
            "A1": 0.000,
            "A2": 0.313,
            "A3": 0.313,
            "A4": 0.009,
            "A5": 0.009,
            "A6": 0.000,
            "A7": 0.000,
            "A8": 0.070,
            "A9": 0.009,
            "A10": 0.278,
        },
        sobol_computation="salib_saltelli_n8192_seed42_v_max_output",
        applicable_scenarios=["VS-E", "VS-F"],
        scenario_notes={
            "VS-E": "10-D standard truss, effective dim ≈ 4–5; baseline test for high-dimensional methods.",
            "VS-F": "Bottom-chord and diagonal bars have large ST; vertical secondary bars have small ST; screening methods should identify them.",
        },
        recommended_n_train={"default": 80},
        n_saturation=400,
        noise_factor_candidates=[],
        control_factor_candidates=[f"A{i+1}" for i in range(10)],
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "ten_bar_truss_10d physical inputs must be a 2-D array with "
                f"{self.meta.dim_input} columns"
            )
        if not np.all(np.isfinite(X_phys)):
            raise ValueError("ten_bar_truss_10d physical inputs must be finite")

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
                "ten_bar_truss_10d physical inputs outside the registered "
                "benchmark domain for A1-A10; "
                f"row {row}, {var.name}={X_phys[row, col]!r} not in "
                f"[{var.lower}, {var.upper}] {var.unit}"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                A_arr = X_phys[i] * 1e-4  # cm² → m²
                Y[i] = self._truss_fem(A_arr)
                if (
                    Y[i].shape != (self.meta.dim_output,)
                    or not np.all(np.isfinite(Y[i]))
                    or Y[i, 0] < 0.0
                    or Y[i, 1] < 0.0
                    or Y[i, 2] <= 0.0
                ):
                    raise ValueError(
                        "malformed, non-finite, or non-physical ten-bar truss "
                        f"response u_max/v_max/mass={np.array2string(Y[i], precision=6, separator=', ')}"
                    )
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "ten_bar_truss_10d direct-stiffness truss FEM solve "
                    f"failed for physical input {x_text}; refusing silent "
                    "NaN output because the formal benchmark response "
                    "surface requires finite displacements and mass."
                ) from None
        return Y

    def _truss_fem(self, A_arr):
        """Planar truss finite-element analysis (direct stiffness method)."""
        nodes = self._NODES
        elems = self._ELEMENTS
        n_nodes = len(nodes)
        n_dof = n_nodes * 2

        K = np.zeros((n_dof, n_dof))

        lengths = []
        for ei, (ni, nj) in enumerate(elems):
            dx = nodes[nj, 0] - nodes[ni, 0]
            dy = nodes[nj, 1] - nodes[ni, 1]
            L = np.sqrt(dx**2 + dy**2)
            lengths.append(L)
            cx = dx / L
            cy = dy / L
            EA_L = self._E * A_arr[ei] / L
            # Local stiffness → global coordinates
            k_local = EA_L * np.outer(
                [cx, cy, -cx, -cy],
                [cx, cy, -cx, -cy],
            )
            dofs = [ni * 2, ni * 2 + 1, nj * 2, nj * 2 + 1]
            for a_i, a in enumerate(dofs):
                for b_i, b in enumerate(dofs):
                    K[a, b] += k_local[a_i, b_i]

        # External load vector
        F = np.zeros(n_dof)
        for node_idx, (fx, fy) in self._LOADS.items():
            F[node_idx * 2] += fx
            F[node_idx * 2 + 1] += fy

        fixed_dofs = self._fixed_dofs(n_dof)
        u = self._solve_constrained_system(K, F, fixed_dofs)
        u_disp = u.reshape(n_nodes, 2)

        # Exclude support nodes
        free_nodes = [i for i in range(n_nodes) if i not in self._SUPPORTS]
        u_free = u_disp[free_nodes]

        u_max = float(np.max(np.abs(u_free[:, 0]))) * 1000.0  # m→mm
        v_max = float(np.max(np.abs(u_free[:, 1]))) * 1000.0

        # Total mass
        mass = sum(self._RHO * A_arr[ei] * lengths[ei] for ei in range(len(elems)))

        return np.array([u_max, v_max, float(mass)])

    @classmethod
    def _fixed_dofs(cls, n_dof):
        fixed = []
        for node_idx, dirs in cls._SUPPORTS.items():
            for j, is_fixed in enumerate(dirs):
                if is_fixed:
                    dof = node_idx * 2 + j
                    if dof < n_dof:
                        fixed.append(dof)
        return np.array(sorted(fixed), dtype=int)

    @staticmethod
    def _solve_constrained_system(K, F, fixed_dofs):
        """Solve K u = F by removing fixed DOF and reconstructing full u."""
        n_dof = K.shape[0]
        fixed_mask = np.zeros(n_dof, dtype=bool)
        fixed_mask[np.asarray(fixed_dofs, dtype=int)] = True
        free_dofs = np.flatnonzero(~fixed_mask)

        u = np.zeros(n_dof)
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        u[free_dofs] = np.linalg.solve(K_ff, F_f)
        return u
