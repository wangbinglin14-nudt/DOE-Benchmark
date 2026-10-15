"""
Case G01: 2-D oil-reservoir multi-block seepage (steady state)
===============================================================

Engineering background
----------------------
Oil-reservoir exploitation is a core petroleum-engineering problem.
The permeability distribution directly determines fluid-flow paths and
recovery factor.  Real reservoir permeability is highly heterogeneous:
  · Fracture-zone permeability can be 3 orders of magnitude higher than
    the rock matrix
  · Mudstone interlayer permeability is 2–4 orders of magnitude lower
    than sandstone

In reservoir simulation (history matching) and development optimisation,
geologists need to determine which blocks’ permeabilities have a decisive
impact on total production (or injection–production pressure drop) — a
typical high-dimensional variable-screening engineering scenario.

This case simplifies a 2-D reservoir into 3×3 = 9 homogeneous blocks, each
with an independent permeability K_i.  The left and right boundaries impose
fixed injection and production pressures, while the top and bottom are closed.
Under steady-state flow the pressure field, total boundary conductance, and
effective Darcy-equivalent permeability are solved.

The implementation is a compact single-phase pressure-boundary conductance
surrogate. It does not model wells, Peaceman well indices, well controls,
multiphase relative permeability, capillary pressure, compressibility,
porosity/storage, gravity segregation, 3-D stratigraphy, fracture networks,
time-dependent saturation fronts, aquifer support, compositional/thermal/EOR
processes, black-oil/compositional simulator decks, surface facilities,
CO2 storage physics, history matching, field calibration, upscaling studies,
standard or multisegment wells, local grid refinement, deck parsing, restart
files, polymer/solvent/H2/CO2STORE options, geothermal/energy equations, or
geomechanical coupling, Eclipse-format summary/restart output,
OPM Python/PYACTION/UDQ/ACTIONX workflows, Newton/NLDD nonlinear solves,
CPR/multigrid linear solvers, or adaptive timestep controls.
Current USGS porous-media-flow guidance also makes clear that real subsurface
flow paths depend on external boundaries, recharge/discharge structure, and
pumping stresses; this benchmark intentionally freezes that broader setting to
left/right Dirichlet pressure boundaries with closed top/bottom boundaries.
Current USGS recharge/discharge framing also makes clear that well pumping is
one discharge pathway, while seepage from floodwater, streams, and irrigation
can act as recharge. This compact surrogate intentionally omits those
distributed source/sink and recharge/discharge forcings.
SPE10, MRST, OPM Flow, and Peaceman are cited as
scope/context anchors only; this case does not execute MRST/OPM Flow, read
industry-standard reservoir decks, solve wells/facilities, or perform
SPE10-style upscaling comparison. Current official MRST/OPM pages also make
clear that real workflows include ECLIPSE deck input, corner-point or
unstructured/fractured grids, wells/facilities, group/network controls,
restart files, black-oil/compositional/thermal/EOR variants, and adjoint or
automatic-differentiation tooling. Current OPM Flow manual pages additionally
expose release-specific reference manuals and Python/PYACTION documentation.
This compact surrogate does not emulate those simulator, scripting, nonlinear
solver, restart, or workflow contracts.

Physics model (Darcy’s law)
---------------------------
Governing equation (2-D steady-state incompressible seepage):

  ∇·(K(x,y)/μ · ∇p) = 0

where K(x,y) is the block-wise homogeneous permeability field, μ is the
dynamic viscosity.

Boundary conditions:
  Left (x = 0):  p = p_inj   (injection pressure, set to 10 MPa)
  Right (x = 1): p = p_prod  (production pressure, set to 1 MPa)
  Top / bottom (y = 0, y = 1): ∂p/∂y = 0  (closed, no-flow)

Numerical method:
  FDM, uniform 18×18 grid at default fidelity (6 nodes per block);
  high fidelity uses 72×72 grid (24 nodes per block).
  Interface permeability handled with harmonic mean.
  Sparse linear system solved with scipy.sparse.linalg.spsolve.

Screening / interaction design
------------------------------
The 3×3 permeability field is intentionally heterogeneous and strongly
coupled through the elliptic Darcy solve.  Current Sobol metadata treats all
9 blocks as active, with the central block and neighbouring flow-path blocks
ranking highest.  The case is therefore a compact interaction/screening
stress test, not a single-channel reservoir model.

Input variables (9 permeability blocks)
---------------------------------------
K_ij   Block (i, j) permeability (i = row, j = col, row-major from top-left)
       Range: [1, 1000] mD (millidarcy), log-scale distribution

Block numbering (3×3 grid):
  [K11 K12 K13]   Left column:   K11, K21, K31
  [K21 K22 K23]   Centre column: K12, K22, K32
  [K31 K32 K33]   Right column:  K13, K23, K33

All blocks participate in the elliptic solve; the central and adjacent
flow-path blocks rank highest in the current Sobol metadata. This is a
compact permeability-screening surrogate, not a calibrated field reservoir
simulation or a claim that one preferential channel alone controls production.

Output responses
----------------
Q_total   Single-phase boundary conductance       m³/(day·MPa)  Primary output
dp_max    Maximum pressure gradient in flow field  MPa/m         Fracture-detection indicator
eff_perm  Effective (Darcy-equivalent) permeability mD           Overall seepage capacity

DOE validation value
--------------------
VS-F (variable screening):
    All 9 permeability variables are active; the central block and adjacent
    flow-path blocks rank highest
    because they couple several pressure-gradient paths.  This tests whether
    benchmark methods recover the declared all-active permeability ranking
    under the preregistered small-budget design.

VS-E (high-dimensional adaptability):
    9-D problem, extensible to 16-D (4×4) or 25-D (5×5), serving as a
    progressive benchmark for high-dimensional DOE methods.

References
----------
1. Dake, L. P. (1978). Fundamentals of Reservoir Engineering.
   Elsevier, Amsterdam. ISBN 978-0-444-41830-2.
2. Bear, J. (1972). Dynamics of Fluids in Porous Media.
   Elsevier, New York.
3. Aziz, K. & Settari, A. (1979). Petroleum Reservoir Simulation.
   Applied Science Publishers.
4. Christie, M. A. & Blunt, M. J. (2001). Tenth SPE Comparative Solution
   Project: A Comparison of Upscaling Techniques.
   SPE Reservoir Eval. Eng. 4(4):308-317. DOI: 10.2118/72469-PA
   https://doi.org/10.2118/72469-PA
   https://www.ipt.ntnu.no/~kleppe/pub/SPE-COMPARATIVE/papers/tenth.pdf
5. Lie, K.-A. (2019). An Introduction to Reservoir Simulation Using
   MATLAB/GNU Octave: User Guide for the MATLAB Reservoir Simulation Toolbox
   (MRST). https://www.sintef.no/projectweb/mrst/
   Current SINTEF MRST software page: https://www.sintef.no/en/software/mrst/
   Official MRST source repository: https://github.com/SINTEF-AppliedCompSci/MRST
   MRST core module: https://www.sintef.no/projectweb/mrst/modules/mrst-core/
   MRST deckformat module: https://www.sintef.no/projectweb/mrst/modules/deckformat/
6. Open Porous Media. OPM Flow reservoir simulator documentation.
   https://opm-project.org/
   https://opm-project.org/?page_id=19
   OPM Flow reference manual page (current 2025.10 manual and Python/PYACTION
   documentation entry point): https://opm-project.org/?page_id=955
   OPM PyAction and UDQ/ACTIONX documentation:
   https://opm-project.org/?page_id=1454
   OPM Flow linear-solver workflow boundary:
   https://opm-project.org/?page_id=1798
   https://github.com/OPM/opm-simulators
7. Peaceman, D. W. (1978). Interpretation of Well-Block Pressures in
   Numerical Reservoir Simulation. SPE Journal. DOI:10.2118/6893-PA.
   https://doi.org/10.2118/6893-PA
8. U.S. Geological Survey Circular 1224. Darcy's Law / hydraulic
   conductivity background for porous-media groundwater flow.
   https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html
9. U.S. Geological Survey. Groundwater recharge and discharge overview.
   https://www.usgs.gov/media/images/groundwater-recharge-and-discharge
"""

from typing import Any

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case

# Block labels (3×3 grid, row-major)
_BLOCK_NAMES = [f"K{r+1}{c+1}" for r in range(3) for c in range(3)]


@register_case
class OilReservoir9D(BaseCase):
    """2-D oil reservoir 3×3-block seepage (9-D, steady-state Darcy).

    9-D input (9 block permeabilities, log-scale) → 3-D output.
    All 9 block permeabilities are active in the current Sobol metadata.
    Compact high-dimensional permeability-screening surrogate.
    """

    meta = CaseMeta(
        case_id="oil_reservoir_9d",
        name="2-D oil reservoir 3×3-block seepage (permeability screening)",
        version="1.0.0",
        domain="geomechanics",
        tags=[
            "reservoir",
            "darcy",
            "porous_media",
            "screening",
            "high_dimension",
            "fdm",
            "heterogeneous",
            "2d",
        ],
        dim_input=9,
        dim_output=3,
        input_variables=[
            InputVariable(
                name=name,
                symbol=f"K_{{{name[1:]}}}",
                description=f"Permeability of block ({name[1]},{name[2]})",
                unit="mD",
                lower=1.0,
                upper=1000.0,
                log_scale=True,  # permeability spans 3 orders of magnitude; log distribution
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            )
            for name in _BLOCK_NAMES
        ],
        output_variables=[
            OutputVariable(
                name="Q_total",
                symbol="Q_{total}",
                description=(
                    "Single-phase Darcy boundary conductance normalised by "
                    "injection-production pressure drop and unit-depth domain"
                ),
                unit="m³/(day·MPa)",
                is_primary=True,
                expected_range=(0.0, 500.0),
            ),
            OutputVariable(
                name="dp_max",
                symbol=r"|\nabla p|_{max}",
                description="Maximum pressure gradient in flow field (identifies main channel; heterogeneous paths can reach ~100 MPa/m)",
                unit="MPa/m",
                is_primary=False,
                # average gradient ~9 MPa/m; high-K sandwiched between low-K can reach 8–10× average
                expected_range=(0.0, 100.0),
            ),
            OutputVariable(
                name="eff_perm",
                symbol="K_{eff}",
                description="Effective Darcy permeability (back-calculated from total flow)",
                unit="mD",
                is_primary=False,
                expected_range=(0.0, 1000.0),
            ),
        ],
        governing_equation=(
            "2-D steady-state incompressible Darcy seepage:\n"
            "  ∇·(K(x,y)/μ · ∇p) = 0\n"
            "  BC: p(0,y) = p_inj = 10 MPa, p(1,y) = p_prod = 1 MPa\n"
            "      ∂p/∂y|_{y=0,1} = 0 (top/bottom closed boundaries)\n"
            "  FDM solution; harmonic-mean permeability at block interfaces"
        ),
        pde_type="elliptic",
        spatial_dim=2,
        is_steady=True,
        # Fixed-coefficient Darcy flow is linear in pressure, but the
        # benchmark response is nonlinear in the permeability inputs because
        # K enters the sparse operator and interface harmonic means.
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Dake, L. P. (1978). Fundamentals of Reservoir Engineering. Elsevier.",
            "Bear, J. (1972). Dynamics of Fluids in Porous Media. Elsevier.",
            "Aziz, K. & Settari, A. (1979). Petroleum Reservoir Simulation. Applied Science Publishers.",
            "Christie, M. A. & Blunt, M. J. (2001). Tenth SPE Comparative Solution Project: "
            "A Comparison of Upscaling Techniques. SPE Reservoir Eval. Eng. 4(4):308-317. "
            "https://doi.org/10.2118/72469-PA; "
            "https://www.ipt.ntnu.no/~kleppe/pub/SPE-COMPARATIVE/papers/tenth.pdf",
            "Lie, K.-A. (2019). An Introduction to Reservoir Simulation Using MATLAB/GNU Octave: "
            "User Guide for the MATLAB Reservoir Simulation Toolbox (MRST). "
            "https://www.sintef.no/projectweb/mrst/; "
            "SINTEF MRST software page. https://www.sintef.no/en/software/mrst/; "
            "official MRST source repository. https://github.com/SINTEF-AppliedCompSci/MRST; "
            "MRST core module. https://www.sintef.no/projectweb/mrst/modules/mrst-core/; "
            "MRST deckformat module. https://www.sintef.no/projectweb/mrst/modules/deckformat/",
            "Open Porous Media. OPM Flow reservoir simulator documentation. "
            "https://opm-project.org/; "
            "https://opm-project.org/?page_id=19; "
            "OPM Flow manual page, current 2025.10 reference manual and "
            "Python/PYACTION documentation entry point. "
            "https://opm-project.org/?page_id=955; "
            "OPM PyAction and UDQ/ACTIONX documentation. "
            "https://opm-project.org/?page_id=1454; "
            "OPM Flow linear-solver workflow boundary. "
            "https://opm-project.org/?page_id=1798; "
            "OPM simulators source repository. https://github.com/OPM/opm-simulators",
            "Peaceman, D. W. (1978). Interpretation of Well-Block Pressures in "
            "Numerical Reservoir Simulation. SPE Journal. DOI:10.2118/6893-PA. "
            "https://doi.org/10.2118/6893-PA",
            "U.S. Geological Survey Circular 1224, Darcy's Law and hydraulic-conductivity "
            "groundwater-flow background. "
            "https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html",
            "U.S. Geological Survey. Groundwater recharge and discharge overview. "
            "https://www.usgs.gov/media/images/groundwater-recharge-and-discharge",
        ],
        solver_library="scipy.sparse.linalg (FDM sparse linear system)",
        solver_details=(
            "Uniform 18×18 grid at default fidelity (6 nodes per block) "
            "and 72×72 grid at high fidelity (24 nodes per block); "
            "harmonic-mean interface permeability; invalid/non-positive permeability "
            "inputs, malformed sparse pressure solves, pressure fields outside the "
            "Dirichlet-boundary envelope, non-positive conductance, and non-finite "
            "or out-of-contract Q_total/dp_max/eff_perm outputs raise explicit "
            "failures rather than entering sweep archives; "
            "named _validate_physical_inputs private physical-input evaluation "
            "rejects malformed arrays and K11-K33 permeability values outside the "
            "declared benchmark domain before sparse Darcy assembly; "
            "no wells, Peaceman well indices, "
            "well controls, multiphase relative permeability, capillary pressure, "
            "compressibility, porosity/storage, gravity segregation, 3-D stratigraphy, "
            "fracture networks, time-dependent saturation front, aquifer support, "
            "black-oil formulation, compositional/thermal/EOR process, CO2 storage "
            "physics, industry-standard simulator deck, ECLIPSE-deck parsing, "
            "surface-facility constraint, standard or multisegment wells, local "
            "grid refinement, restart-file workflows, polymer/surfactant option, "
            "solvent model, H2 storage, geothermal/energy equation, history "
            "matching, field calibration, upscaling study, "
            "geomechanical coupling, Eclipse-format summary/restart output, "
            "OPM Python/PYACTION/UDQ/ACTIONX workflows, Newton/NLDD nonlinear "
            "solvers, CPR/multigrid linear solvers, or adaptive timestep "
            "controls; OPM-style linear-solver failure and timestep-cut retry "
            "workflows are likewise outside this steady linear Darcy surrogate; "
            "official MRST/OPM capability boundaries "
            "such as corner-point/unstructured/fractured grids, adjoint/automatic-"
            "differentiation tooling, group/network controls, and Eclipse/restart "
            "workflows, plus MRST-core boundary-condition/source-term/well "
            "workflow objects, are scope/context anchors only; this benchmark "
            "intentionally freezes those richer control spaces to fixed left/right "
            "pressure boundaries, closed top/bottom boundaries, and no wells or "
            "distributed source/sink terms; "
            "SPE10, MRST, OPM Flow, and "
            "Peaceman references are not executed or used as validation data for "
            "this compact pressure-boundary conductance surrogate; current USGS "
            "groundwater-flow guidance likewise treats external boundaries, recharge/"
            "discharge structure, and pumping-well stresses as flow-path controls "
            "that are intentionally frozen out of this left/right-pressure-boundary "
            "surrogate, and the current USGS recharge/discharge framing likewise "
            "treats well pumping as discharge plus floodwater/stream/irrigation "
            "seepage as recharge that this compact surrogate omits as distributed "
            "source/sink forcing"
        ),
        fidelity_levels=["default", "high"],
        typical_eval_time_ms={"default": 10.0, "high": 80.0},
        cost_level="L1",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=True,
        response_topology=["smooth_monotone", "strong_interaction"],
        effective_dim=9,  # all 9 conductivity blocks active (ST >= 0.05 for Q_total)
        important_variables=_BLOCK_NAMES,  # all blocks are active; see stored ST values and VS-F note
        interaction_pairs=[
            ("K11", "K12"),
            ("K12", "K13"),
            ("K21", "K22"),
            ("K22", "K23"),
            ("K31", "K32"),
            ("K32", "K33"),
            ("K11", "K21"),
            ("K21", "K31"),
            ("K12", "K22"),
            ("K22", "K32"),
            ("K13", "K23"),
            ("K23", "K33"),
            ("K11", "K22"),
            ("K13", "K22"),
            ("K22", "K31"),
            ("K22", "K33"),
        ],
        has_failure_boundary=False,
        sobol_S1={
            "K11": 0.061,
            "K12": 0.045,
            "K13": 0.024,
            "K21": 0.075,
            "K22": 0.113,
            "K23": 0.030,
            "K31": 0.026,
            "K32": 0.052,
            "K33": 0.059,
        },
        sobol_ST={
            "K11": 0.142,
            "K12": 0.169,
            "K13": 0.133,
            "K21": 0.233,
            "K22": 0.301,
            "K23": 0.161,
            "K31": 0.157,
            "K32": 0.190,
            "K33": 0.174,
        },
        sobol_computation="salib_saltelli_n1024_seed42_Q_total_output",
        applicable_scenarios=["VS-E", "VS-F"],
        scenario_notes={
            "VS-E": "Compact 9-D all-active permeability-screening problem, extensible to 16-D (4×4) and 25-D (5×5), "
            "for testing surrogate-accuracy degradation in high dimensions (curse-of-dimensionality quantification).",
            "VS-F": "All 9 variables active (ST > 0.13); central block K22 dominates (ST ≈ 0.30), "
            "K21 second (ST ≈ 0.23); interaction ratio 0.71 (strong block-to-block coupling). "
            "Tests whether benchmark methods recover the declared all-active permeability ranking "
            "under the preregistered small-budget design; no pre-result ranking-success claim.",
        },
        recommended_n_train={"default": 60, "high": 200},
        n_saturation=400,
        noise_factor_candidates=[],
        control_factor_candidates=_BLOCK_NAMES,
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    _P_INJ = 10.0  # MPa
    _P_PROD = 1.0  # MPa
    _MU = 1.0e-3  # Pa·s; fixed single-phase viscosity for the conductance surrogate
    _N_PER_BLOCK = 6  # nodes per block (default fidelity)

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "oil_reservoir_9d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(X_phys < 1.0 * (1.0 - tol))
            or np.any(X_phys > 1000.0 * (1.0 + tol))
        ):
            raise ValueError(
                "oil_reservoir_9d physical inputs must stay inside the declared "
                "K11-K33 permeability benchmark domain before sparse Darcy assembly"
            )
        return X_phys

    def _evaluate_core(
        self,
        X_phys: np.ndarray,
        fidelity: str = "default",
        **kwargs,
    ) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n_per_block = self._N_PER_BLOCK if fidelity == "default" else 24
        n_samples = X_phys.shape[0]
        Y = np.full((n_samples, 3), np.nan)
        for i in range(n_samples):
            try:
                K_blocks = X_phys[i]  # shape (9,), mD
                Y[i] = self._solve_reservoir(K_blocks, n_per_block)
                if Y[i].shape != (self.meta.dim_output,) or not np.all(np.isfinite(Y[i])):
                    raise RuntimeError("non-finite reservoir conductance response")
                output_ranges = np.asarray(
                    [output.expected_range for output in self.meta.output_variables],
                    dtype=float,
                )
                below = Y[i] < (output_ranges[:, 0] - 1e-10)
                above = Y[i] > (output_ranges[:, 1] + 1e-10)
                if np.any(below) or np.any(above):
                    raise RuntimeError(
                        "out-of-contract reservoir conductance response outside "
                        "the declared Q_total/dp_max/eff_perm output ranges"
                    )
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "oil_reservoir_9d sparse Darcy reservoir solve failed for "
                    f"physical input {x_text}; refusing silent NaN output because "
                    "the formal benchmark response surface requires a finite "
                    "single-phase conductance calculation."
                ) from None
        return Y

    def _solve_reservoir(
        self,
        K_blocks: np.ndarray,
        n_per_block: int = _N_PER_BLOCK,
        *,
        return_diagnostics: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """FDM solution of 2-D steady-state Darcy seepage.

        K_blocks : shape (9,), row-major, mD
        Returns: [Q_total, dp_max, eff_perm].  With return_diagnostics=True,
        also returns internal face-flux diagnostics for conservation tests;
        the public benchmark response vector is unchanged.
        """
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve

        K_blocks = np.asarray(K_blocks, dtype=float)
        if K_blocks.shape != (self.meta.dim_input,):
            raise ValueError(
                "oil_reservoir_9d expects exactly 9 permeability blocks; "
                f"got shape {K_blocks.shape!r}"
            )
        if not np.all(np.isfinite(K_blocks)) or np.any(K_blocks <= 0.0):
            raise ValueError("oil_reservoir_9d permeability inputs must be finite and positive")
        if int(n_per_block) != n_per_block or n_per_block < 2:
            raise ValueError("oil_reservoir_9d n_per_block must be an integer >= 2")
        n_per_block = int(n_per_block)

        # mD → m² conversion: 1 mD = 9.869e-16 m²
        K_SI = K_blocks * 9.869e-16  # m²

        N_block = 3  # blocks per row/column
        N = N_block * n_per_block  # total node count
        h = 1.0 / (N - 1)  # spatial step (normalised domain [0,1]×[0,1])

        # 2-D permeability field K[iy, ix], aligned with p_field[iy, ix].
        # Block (block_r, block_c) maps to row/column node ranges.
        K_field = np.zeros((N, N))
        for block_r in range(N_block):
            for block_c in range(N_block):
                idx = block_r * N_block + block_c
                i_start = block_r * n_per_block
                i_end = (block_r + 1) * n_per_block
                j_start = block_c * n_per_block
                j_end = (block_c + 1) * n_per_block
                K_field[i_start:i_end, j_start:j_end] = K_SI[idx]

        # Build sparse system (pressure p[i,j], node index i*N+j, i = y-dir, j = x-dir)
        # Note: injection at j = 0 (x = 0), production at j = N-1 (x = 1)
        n_total = N * N
        A_mat = lil_matrix((n_total, n_total))
        b_vec = np.zeros(n_total)

        def node(iy, ix):
            return iy * N + ix

        p_inj = self._P_INJ * 1e6  # Pa
        p_prod = self._P_PROD * 1e6  # Pa

        for iy in range(N):
            for ix in range(N):
                nd = node(iy, ix)

                # Left boundary (ix=0): p = p_inj
                if ix == 0:
                    A_mat[nd, nd] = 1.0
                    b_vec[nd] = p_inj
                    continue
                # Right boundary (ix=N-1): p = p_prod
                if ix == N - 1:
                    A_mat[nd, nd] = 1.0
                    b_vec[nd] = p_prod
                    continue
                # Top/bottom boundary (iy=0 or N-1): Neumann ∂p/∂y = 0
                if iy == 0:
                    A_mat[nd, nd] = 1.0
                    A_mat[nd, node(1, ix)] = -1.0
                    continue
                if iy == N - 1:
                    A_mat[nd, nd] = 1.0
                    A_mat[nd, node(N - 2, ix)] = -1.0
                    continue

                # Interior node: FDM, harmonic-mean permeability
                Kxm = (
                    2
                    * K_field[iy, ix - 1]
                    * K_field[iy, ix]
                    / (K_field[iy, ix - 1] + K_field[iy, ix] + 1e-40)
                )
                Kxp = (
                    2
                    * K_field[iy, ix]
                    * K_field[iy, ix + 1]
                    / (K_field[iy, ix] + K_field[iy, ix + 1] + 1e-40)
                )
                Kym = (
                    2
                    * K_field[iy - 1, ix]
                    * K_field[iy, ix]
                    / (K_field[iy - 1, ix] + K_field[iy, ix] + 1e-40)
                )
                Kyp = (
                    2
                    * K_field[iy, ix]
                    * K_field[iy + 1, ix]
                    / (K_field[iy, ix] + K_field[iy + 1, ix] + 1e-40)
                )

                coeff = -(Kxm + Kxp + Kym + Kyp)
                A_mat[nd, nd] = coeff
                A_mat[nd, node(iy, ix - 1)] = Kxm
                A_mat[nd, node(iy, ix + 1)] = Kxp
                A_mat[nd, node(iy - 1, ix)] = Kym
                A_mat[nd, node(iy + 1, ix)] = Kyp

        p_flat = np.asarray(spsolve(A_mat.tocsr(), b_vec), dtype=float)
        if p_flat.shape != (n_total,) or not np.all(np.isfinite(p_flat)):
            raise ValueError("non-finite or malformed Darcy pressure solution")
        pressure_tol = 1e-8 * max(abs(p_inj), abs(p_prod), 1.0)
        if np.min(p_flat) < min(p_prod, p_inj) - pressure_tol or np.max(p_flat) > max(p_prod, p_inj) + pressure_tol:
            raise ValueError("Darcy pressure solution violates the imposed pressure-boundary envelope")
        p_field = p_flat.reshape(N, N)  # shape (N_y, N_x)

        # ── Extract output quantities ───────────────────────────────────────
        # Total flow.  For heterogeneous nodal FDM fields, a single boundary
        # face can carry a small discretisation bias.  Average the Darcy flux
        # over all internal vertical faces using the same harmonic interface
        # permeability as the pressure operator.
        Q_faces = self._internal_vertical_face_fluxes(K_field, p_field, h)
        if not np.all(np.isfinite(Q_faces)) or np.any(Q_faces < 0.0):
            raise ValueError("non-physical Darcy face fluxes")
        Q_total_SI = float(np.mean(Q_faces))  # m²/s (per unit depth)
        # Convert to m³/(day·MPa) conductance per unit depth and normalise
        dp_total = p_inj - p_prod
        Q_norm = Q_total_SI / dp_total * 1e6 * 86400  # m³/(day·MPa)
        if not np.isfinite(Q_norm) or Q_norm <= 0.0:
            raise ValueError("non-physical non-positive Darcy boundary conductance")

        # Maximum pressure gradient (MPa/m)
        dp_dx = (p_field[:, 2:] - p_field[:, :-2]) / (2 * h)
        dp_dy = (p_field[2:, :] - p_field[:-2, :]) / (2 * h)
        grad_p = np.sqrt(dp_dx[1:-1, :] ** 2 + dp_dy[:, 1:-1] ** 2)
        dp_max_MPa = float(np.max(np.abs(grad_p))) * 1e-6
        if not np.isfinite(dp_max_MPa) or dp_max_MPa <= 0.0:
            raise ValueError("non-physical Darcy pressure-gradient response")

        # Effective permeability (back-calculated from Q, 1-D Darcy equivalent)
        # Q_norm = K_eff / μ · 1/L (m²/s per MPa) → K_eff
        # Q_norm [m³/(day·MPa)] → Q_SI [m²/s] = Q_norm / 86400 / 1e6
        Q_SI_per_MPa = Q_norm / 86400 / 1e6
        K_eff_SI = Q_SI_per_MPa * self._MU * 1.0  # L=1m
        K_eff_mD = float(K_eff_SI / 9.869e-16)
        k_min = float(np.min(K_blocks))
        k_max = float(np.max(K_blocks))
        tol = 1e-8 * max(1.0, k_max)
        if not np.isfinite(K_eff_mD) or K_eff_mD < k_min - tol or K_eff_mD > k_max + tol:
            raise ValueError(
                "non-physical effective permeability outside the input "
                f"permeability envelope: K_eff={K_eff_mD!r}, range=[{k_min!r}, {k_max!r}]"
            )
        K_eff_mD = min(max(K_eff_mD, k_min), k_max)

        response = np.array([Q_norm, dp_max_MPa, K_eff_mD], dtype=float)
        if not np.all(np.isfinite(response)):
            raise ValueError("non-finite Darcy reservoir response")
        if return_diagnostics:
            diagnostics = {
                "Q_faces_SI": Q_faces.copy(),
                "Q_total_SI": Q_total_SI,
                "face_flux_relative_spread": float(
                    (np.max(Q_faces) - np.min(Q_faces)) / Q_total_SI
                ),
                "grid_nodes": N,
                "n_per_block": n_per_block,
            }
            return response, diagnostics
        return response

    def _internal_vertical_face_fluxes(
        self,
        K_field: np.ndarray,
        p_field: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """Return integrated Darcy fluxes across internal vertical faces."""
        Kx_faces = (
            2.0
            * K_field[:, :-1]
            * K_field[:, 1:]
            / (K_field[:, :-1] + K_field[:, 1:] + 1e-40)
        )
        dp_dx_faces = (p_field[:, 1:] - p_field[:, :-1]) / h
        Q_face_density = Kx_faces / self._MU * np.abs(dp_dx_faces)  # m/s
        return np.trapezoid(Q_face_density, dx=h, axis=0)  # m²/s per unit depth
