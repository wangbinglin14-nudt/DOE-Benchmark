"""
Case G03: Dam foundation seepage uplift pressure (2-D steady-state seepage)
============================================================================

Engineering background
----------------------
The safety and stability of concrete gravity dams are governed by the uplift
pressure from foundation seepage.  Uplift pressure is the upward pressure
exerted by seeping water on the dam base, directly reducing the effective
self-weight of the dam and increasing the overturning moment risk.

Dam-foundation seepage analysis is an essential part of dam design
(Chinese hydraulic-engineering standard SL 319-2018):
  · Upstream water level H₁ (reservoir side), downstream level H₂ (tailwater)
  · Permeability of the foundation rock and the grout curtain (seepage barrier)
  · Drainage holes (engineering measure to reduce uplift)

The standard is cited for engineering context.  The implemented case is a
Darcy/FDM benchmark surrogate with simplified curtain and drain boundary
conditions, not a code-compliant dam-foundation design calculation.
It does not model seepage discharge, exit gradients, piping/internal erosion,
unsaturated flow, transient reservoir operation, anisotropy, fractures/joints,
three-dimensional abutment flow, seepage velocities, full pore-pressure-field
reporting, flow-net construction, piezometer or instrumentation interpretation,
permeability-uncertainty sensitivity studies,
mesh-convergence evidence, drainage-gallery hydraulics, collector-pipe / header
layouts, relief-well systems, relief-well spacing/sizing, filter design,
rehabilitation planning, or structural sliding/overturning stability checks.
Current USACE/ASDSO seepage-control guidance is used only as boundary evidence
for omitted monitoring, inspection, maintenance/rehabilitation, filter-design,
and remediation-selection workflows, not as validation of this fixed-grid
uplift-pressure surrogate.
Current USACE relief-well guidance also makes explicit that real drain-relief
systems involve siting, spacing, hydraulic design, construction quality,
development/pump-testing, operation, and maintenance workflows that this
single-node pressure-relief surrogate does not emulate.  The 2025 USACE ERDC
relief-well current-practice report is likewise used only as boundary evidence
for relief-well maintenance/repair and seepage-remediation workflows outside
this benchmark.
Current FERC risk-informed dam-safety and potential-failure-mode material is
likewise boundary evidence for omitted loading-likelihood, system-response,
failure-consequence, PFM/PFMA, surveillance, monitoring, inspection, and
risk-management workflows, not validation of this three-output uplift surrogate.
FEMA P-1032 is used in the same boundary role for omitted seepage/internal-
erosion evaluation, flow/pressure/water-quality monitoring, performance-
monitoring, life-cycle monitoring, emergency response, and remediation
workflows; none of those workflows are represented by the three-output
uplift-pressure response surface.

This case solves a 2-D steady-state Darcy seepage surrogate with FDM,
computing the uplift-pressure distribution along the dam base and the total
uplift force and midpoint uplift-pressure intensity.

Governing equations
-------------------
Darcy seepage (Laplace equation, homogeneous assumption):
  ∂²H/∂x² + ∂²H/∂z² = 0  (homogeneous foundation)
  Heterogeneous: ∂/∂x(K_x·∂H/∂x) + ∂/∂z(K_z·∂H/∂z) = 0

Boundary conditions:
  Upstream face (x = 0): H = H1 (total head)
  Downstream face (x = L): H = H2 (tailwater head)
  Top (z = 0, dam base): impermeable (∂H/∂z = 0), except for one
      simplified drain node with pressure-relief head when that head is
      below the no-drain natural head at the drain location
  Bottom (z = −D, bedrock): impermeable (∂H/∂z = 0)

Curtain treatment: low-permeability grout curtain alters the head distribution.
Drain treatment: a single pressure-relief surrogate, not a detailed drain-hole
network.  The relief node is active only when it reduces local head; otherwise
it remains inactive so the simplified drain cannot become a recharge source.

Input variables
---------------
H1          Upstream head          [20, 100]   m      Reservoir level
H2          Downstream head        [0, 10]     m      Tailwater level
K_base      Foundation permeability [1e-7, 1e-4] m/s  Bedrock condition
K_curtain   Curtain permeability   [1e-9, 1e-7] m/s   Seepage-barrier effectiveness
D_base      Foundation depth       [10, 50]    m      Geological condition
drain_eff   Drain-hole efficiency  [0, 0.8]    -      Drainage effectiveness

Output responses
----------------
U_total     Total uplift force     MN/m        Primary output (stability design load)
U_center    Uplift at dam-base midpoint MPa    Typical monitoring point
eta_uplift  Uplift reduction factor -           Design-code parameter

DOE validation value
--------------------
VS-B (effect estimation):
    H1 and K_curtain exhibit strong interaction (better curtain → smaller H1
    influence).

VS-C (surrogate accuracy):
    FDM heterogeneous seepage; smooth within each drain-activation regime but
    with a threshold at drain_eff = 0 because the simplified drain node switches
    from impermeable to prescribed-head relief; moderate-difficulty surrogate
    test.

References
----------
1. Ministry of Water Resources, PRC. (2018). SL 319-2018 Design Code for
   Concrete Gravity Dams.
2. Bear, J. (1972). Dynamics of Fluids in Porous Media. Elsevier.
3. USACE EM 1110-2-2200. Gravity Dam Design.
   https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2200.pdf
4. USBR Design Standards No. 13, Chapter 8. Seepage.
   https://www.usbr.gov/tsc/techreferences/designstandards-datacollectionguides/finalds-pdfs/DS13-8.pdf
5. USACE EM 1110-2-1901. Seepage Analysis and Control for Dams.
   https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-1901.pdf
6. Association of State Dam Safety Officials summary for USACE EM 1110-2-1901,
   used as public boundary evidence for seepage-control topics such as
   monitoring, inspection, filters, drains, piping, and erosion that are not
   implemented in this uplift-only surrogate.
   https://damsafety.org/content/seepage-analysis-and-control-dams
7. USACE EM 1110-2-1914. Design, Construction, and Maintenance of
   Relief Wells.
   https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM%201110-2-1914_FINAL%2020250307.pdf
   USACE Digital Library historical record:
   https://usace.contentdm.oclc.org/digital/collection/p16021coll9/id/2595/
8. USACE ERDC/GSL TR-25-5. Relief Wells for Dams and Levees: History
   and Current Practice.
   https://www.erdc.usace.army.mil/Media/Publication-Notices/Article/4139695/usace-relief-wells-for-dams-and-levees-history-and-current-practice/
9. FERC Engineering Guidelines, Chapter 3, Gravity Dams.
   https://www.ferc.gov/sites/default/files/2020-04/chap3.pdf
10. FERC Risk-Informed Decision Making for dam safety.
   https://www.ferc.gov/dam-safety-and-inspections/risk-informed-decision-making-ridm
11. FERC Potential Failure Modes and DSPMP/PFMA dam-safety pages.
   https://www.ferc.gov/dam-safety-and-inspections/potential-failure-modes-pfms
   https://www.ferc.gov/dam-safety-and-inspections/dam-safety-performance-monitoring-program-dspmp-and-potential-failure
12. FEMA P-1032. Evaluation and Monitoring of Seepage and Internal Erosion.
    https://www.fema.gov/sites/default/files/2020-08/fema_p1032_eval_monitoring_seepage_internal_erosion.pdf
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class DamSeepage6D(BaseCase):
    """Dam foundation seepage uplift pressure (6D → 3D, 2-D FDM)."""

    _L_BASE = 30.0  # dam base width, m (fixed)
    _N_X = 20  # horizontal node count
    _N_Z = 15  # vertical node count
    _CURTAIN_X = 5.0  # curtain position (5 m from upstream)
    _CURTAIN_W = 1.0  # curtain thickness (1 m)
    _DRAIN_X = 8.0  # drain-hole position (8 m from upstream)

    meta = CaseMeta(
        case_id="dam_seepage_6d",
        name="Concrete gravity dam foundation seepage uplift (2-D FDM)",
        version="1.0.0",
        domain="geomechanics",
        tags=["dam", "seepage", "uplift_pressure", "darcy", "fdm", "6d", "hydraulic_engineering"],
        dim_input=6,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="H1",
                symbol="H_1",
                description="Upstream total head (relative to dam-base elevation)",
                unit="m",
                lower=20.0,
                upper=100.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="H2",
                symbol="H_2",
                description="Downstream head (tailwater level)",
                unit="m",
                lower=0.0,
                upper=10.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="K_base",
                symbol="K_{base}",
                description="Foundation rock permeability",
                unit="m/s",
                lower=1e-7,
                upper=1e-4,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="K_curtain",
                symbol="K_{curt}",
                description="Grout-curtain permeability",
                unit="m/s",
                lower=1e-9,
                upper=1e-7,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="D_base",
                symbol="D_{base}",
                description="Foundation depth (computational domain height L_z)",
                unit="m",
                lower=10.0,
                upper=50.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="drain_eff",
                symbol=r"\eta_{drain}",
                description=(
                    "Benchmark pressure-relief factor for the fixed-head drain "
                    "surrogate (0 = inactive drain, 0.8 = strong prescribed-head relief; "
                    "not a design-code drain-efficiency recommendation and not a "
                    "calibrated drain-hole-network parameter)"
                ),
                unit="-",
                lower=0.0,
                upper=0.8,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="U_total",
                symbol="U",
                description="Total uplift force per unit dam width",
                unit="MN/m",
                is_primary=True,
                expected_range=(0.0, 50.0),
            ),
            OutputVariable(
                name="U_center",
                symbol="u_c",
                description="Uplift pressure intensity at dam-base midpoint",
                unit="MPa",
                is_primary=False,
                expected_range=(0.0, 1.0),
            ),
            OutputVariable(
                name="eta_uplift",
                symbol=r"\eta_u",
                description="Uplift reduction factor (actual / no seepage barrier)",
                unit="-",
                is_primary=False,
                expected_range=(0.1, 1.0),
            ),
        ],
        governing_equation=(
            "Darcy seepage (Laplace equation, heterogeneous):\n"
            "  ∇·(K(x,z)·∇H) = 0,  (x,z) ∈ Ω\n"
            "  K = K_curtain in the grout-curtain column, K_base elsewhere\n"
            "BC: H = H1 upstream, H = H2 downstream,\n"
            "    top and bottom impermeable except a simplified drain node\n"
            "    with H_d = H2 + (1-drain_eff)·(H1-H2)/2 when H_d is below\n"
            "    the no-drain natural head at the drain node,\n"
            "    bedrock bottom impermeable (∂H/∂n = 0).\n"
            "Uplift: u(x) = γ_w · H(x, z = 0)"
        ),
        pde_type="elliptic",
        spatial_dim=2,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Bear, J. (1972). Dynamics of Fluids in Porous Media. Elsevier.",
            "Ministry of Water Resources, PRC. (2018). SL 319-2018 Design Code for Concrete Gravity Dams.",
            "USACE EM 1110-2-2200. Gravity Dam Design. "
            "https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2200.pdf",
            "USBR Design Standards No. 13, Chapter 8. Seepage. "
            "https://www.usbr.gov/tsc/techreferences/designstandards-datacollectionguides/finalds-pdfs/DS13-8.pdf",
            "USACE EM 1110-2-1901. Seepage Analysis and Control for Dams. "
            "https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-1901.pdf",
            "Association of State Dam Safety Officials summary for USACE EM 1110-2-1901, "
            "used as public boundary evidence for seepage-control monitoring, inspection, "
            "maintenance/rehabilitation, filters, drains, piping, and erosion topics omitted "
            "from this uplift-only surrogate. "
            "https://damsafety.org/content/seepage-analysis-and-control-dams",
            "USACE EM 1110-2-1914. Design, Construction, and Maintenance of Relief Wells. "
            "Current USACE publications-index PDF plus Digital Library record are used "
            "only as relief-well workflow boundary evidence, not validation data for "
            "the single-node pressure-relief surrogate. "
            "https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM%201110-2-1914_FINAL%2020250307.pdf; "
            "https://usace.contentdm.oclc.org/digital/collection/p16021coll9/id/2595/",
            "USACE ERDC/GSL TR-25-5. Relief Wells for Dams and Levees: History and Current Practice. "
            "Current official boundary evidence for relief-well maintenance/repair, pressure-relief, "
            "seepage-remediation, and well-system practice that is not represented by "
            "the benchmark's single pressure-relief node. "
            "https://www.erdc.usace.army.mil/Media/Publication-Notices/Article/4139695/"
            "usace-relief-wells-for-dams-and-levees-history-and-current-practice/",
            "FERC Engineering Guidelines for the Evaluation of Hydropower Projects, Chapter 3, Gravity Dams. "
            "https://www.ferc.gov/sites/default/files/2020-04/chap3.pdf",
            "FERC Risk-Informed Decision Making for dam safety, boundary evidence "
            "for omitted loading likelihood, system response, failure consequence, "
            "and risk-management workflows. "
            "https://www.ferc.gov/dam-safety-and-inspections/risk-informed-decision-making-ridm",
            "FERC Potential Failure Modes and DSPMP/PFMA dam-safety pages, "
            "boundary evidence for omitted potential-failure-mode descriptions, "
            "surveillance, performance monitoring, inspection, instrumentation, "
            "and risk-reduction workflows. "
            "https://www.ferc.gov/dam-safety-and-inspections/potential-failure-modes-pfms; "
            "https://www.ferc.gov/dam-safety-and-inspections/dam-safety-performance-monitoring-program-dspmp-and-potential-failure",
            "FEMA P-1032, Evaluation and Monitoring of Seepage and Internal Erosion. "
            "Public boundary evidence for omitted seepage/internal-erosion evaluation, "
            "visual observation, geophysical investigation, seepage collection, "
            "flow-rate/pressure/water-quality monitoring, PFM identification, "
            "life-cycle monitoring, emergency response, and remediation workflows; "
            "not validation data for this fixed-grid uplift-only surrogate. "
            "https://www.fema.gov/sites/default/files/2020-08/"
            "fema_p1032_eval_monitoring_seepage_internal_erosion.pdf",
        ],
        solver_library="scipy.sparse.linalg (FDM sparse system)",
        solver_details=(
            "Uniform 20×15 grid (nx×nz), harmonic-mean interface permeability, "
            "grid-aligned grout-curtain mask from the declared curtain position/width "
            "replaces K_base, one drain node uses a prescribed "
            "reduced head as a pressure-relief surrogate only when it is below "
            "the no-drain natural head at that node, preventing recharge-like "
            "uplift increases from the simplified drain; no seepage-discharge, "
            "exit-gradient, piping/internal-erosion, unsaturated, transient, "
            "anisotropic, fracture/joint, 3-D abutment-flow, drainage-gallery, "
            "collector-pipe/header layout, "
            "flow-net construction, piezometer/instrumentation interpretation, "
            "permeability-uncertainty sensitivity study, mesh-convergence evidence, "
            "seepage-velocity field, full pore-pressure-field reporting, "
            "relief-well design/construction/maintenance, spacing/sizing, "
            "relief-well siting, development/pump-testing, operation/maintenance, "
            "relief-well maintenance/repair current-practice review, "
            "filter-design, rehabilitation-planning, "
            "FEMA P-1032 seepage/internal-erosion evaluation, monitoring, "
            "emergency-response, remediation, life-cycle monitoring, flow-rate/"
            "pressure/water-quality measurement, "
            "dam-safety monitoring/inspection program, maintenance/rehabilitation workflow, "
            "seepage-control remediation selection, "
            "risk-informed dam-safety likelihood/loading/system-response/"
            "consequence assessment, potential-failure-mode analysis, "
            "or structural-stability calculation; named _validate_physical_inputs "
            "private physical-input evaluation "
            "rejects malformed arrays and H1/H2/K_base/K_curtain/D_base/drain_eff "
            "values outside the declared benchmark domain before sparse FDM assembly"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 10.0},
        cost_level="L1",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=[
            "smooth_monotone",
            "strong_interaction",
            "threshold_jump",
            "discontinuous",
        ],
        effective_dim=5,
        important_variables=["H1", "K_base", "H2", "K_curtain", "drain_eff"],
        interaction_pairs=[
            ("H1", "K_curtain"),
            ("H2", "drain_eff"),
            ("K_base", "K_curtain"),
            ("K_curtain", "D_base"),
            ("drain_eff", "H1"),
        ],
        has_failure_boundary=False,
        sobol_S1={
            "H1": 0.701,
            "H2": 0.086,
            "K_base": 0.070,
            "K_curtain": 0.029,
            "D_base": 0.003,
            "drain_eff": 0.046,
        },
        sobol_ST={
            "H1": 0.736,
            "H2": 0.088,
            "K_base": 0.115,
            "K_curtain": 0.071,
            "D_base": 0.011,
            "drain_eff": 0.051,
        },
        sobol_computation="salib_saltelli_n1024_seed42_U_total_output",
        applicable_scenarios=["VS-B", "VS-C"],
        scenario_notes={
            "VS-B": "H1 and K_curtain strongly interact (better curtain → smaller H1 influence).",
            "VS-C": (
                "FDM heterogeneous uplift surrogate with smooth response within each "
                "drain-activation regime and a threshold where the single drain node "
                "switches between impermeable and prescribed-head relief only if the "
                "relief head is below the no-drain natural head; "
                "moderate surrogate difficulty, not a calibrated seepage-discharge, "
                "exit-gradient, piping, or dam-stability design model."
            ),
        },
        recommended_n_train={"default": 60},
        n_saturation=300,
        noise_factor_candidates=["H2", "K_base"],
        control_factor_candidates=["H1", "K_curtain", "D_base", "drain_eff"],
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "dam_seepage_6d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        H1 = X_phys[:, 0]
        H2 = X_phys[:, 1]
        K_base = X_phys[:, 2]
        K_curtain = X_phys[:, 3]
        D_base = X_phys[:, 4]
        drain_eff = X_phys[:, 5]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(H1 < 20.0 - tol)
            or np.any(H1 > 100.0 + tol)
            or np.any(H2 < 0.0 - tol)
            or np.any(H2 > 10.0 + tol)
            or np.any(K_base < 1e-7 * (1.0 - tol))
            or np.any(K_base > 1e-4 * (1.0 + tol))
            or np.any(K_curtain < 1e-9 * (1.0 - tol))
            or np.any(K_curtain > 1e-7 * (1.0 + tol))
            or np.any(D_base < 10.0 - tol)
            or np.any(D_base > 50.0 + tol)
            or np.any(drain_eff < 0.0 - tol)
            or np.any(drain_eff > 0.8 + tol)
        ):
            raise ValueError(
                "dam_seepage_6d physical inputs must stay inside the declared "
                "H1/H2/K_base/K_curtain/D_base/drain_eff benchmark domain before "
                "sparse FDM seepage evaluation"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                Y[i] = self._seepage(
                    H1=float(X_phys[i, 0]),
                    H2=float(X_phys[i, 1]),
                    K_base=float(X_phys[i, 2]),
                    K_curtain=float(X_phys[i, 3]),
                    D_base=float(X_phys[i, 4]),
                    drain_eff=float(X_phys[i, 5]),
                )
                if (
                    Y[i].shape != (self.meta.dim_output,)
                    or not np.all(np.isfinite(Y[i]))
                    or Y[i, 0] <= 0.0
                    or Y[i, 1] < 0.0
                    or not (0.0 < Y[i, 2] <= 1.0)
                ):
                    raise RuntimeError("nonphysical dam-seepage uplift response")
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "dam_seepage_6d sparse FDM seepage solve failed for physical "
                    f"input {x_text}; refusing silent NaN or nonphysical output "
                    "because the formal benchmark response surface requires a "
                    "finite positive Darcy/FDM uplift calculation."
                ) from None
        return Y

    def _drain_relief_head(self, H1: float, H2: float, drain_eff: float) -> float:
        return H2 + (1.0 - drain_eff) * (H1 - H2) * 0.5

    def _base_head_profile(
        self,
        H1,
        H2,
        K_base,
        K_curtain,
        D_base,
        drain_eff,
        *,
        apply_drain: bool | None = None,
    ):
        """Return x nodes and the solved dam-base head profile."""
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve

        Lx = self._L_BASE
        Lz = D_base  # foundation depth
        nx = self._N_X
        nz = self._N_Z
        dx = Lx / (nx - 1)
        dz = Lz / (nz - 1)
        drain_col = max(1, int(self._DRAIN_X / dx))

        if apply_drain is None:
            apply_drain = False
            if drain_eff > 0.0:
                x0, H0 = self._base_head_profile(
                    H1,
                    H2,
                    K_base,
                    K_curtain,
                    D_base,
                    0.0,
                    apply_drain=False,
                )
                h_relief = self._drain_relief_head(H1, H2, drain_eff)
                apply_drain = h_relief < H0[drain_col] - 1e-12

        # Permeability field (uniform isotropic, except at the declared curtain).
        K_field = np.full((nz, nx), K_base)
        x_nodes = np.linspace(0.0, Lx, nx)
        curtain_half_width = 0.5 * max(self._CURTAIN_W, dx)
        curtain_mask = np.abs(x_nodes - self._CURTAIN_X) <= curtain_half_width
        curtain_mask[0] = False
        curtain_mask[-1] = False
        if not np.any(curtain_mask):
            curtain_mask[max(1, min(nx - 2, int(round(self._CURTAIN_X / dx))))] = True
        K_field[:, curtain_mask] = K_curtain

        n_tot = nx * nz
        A = lil_matrix((n_tot, n_tot))
        b = np.zeros(n_tot)

        def idx(iz, ix):
            return iz * nx + ix

        for iz in range(nz):
            for ix in range(nx):
                nd = idx(iz, ix)
                # Upstream boundary (ix=0): H = H1
                if ix == 0:
                    A[nd, nd] = 1.0
                    b[nd] = H1
                    continue
                # Downstream boundary (ix=nx-1): H = H2
                if ix == nx - 1:
                    A[nd, nd] = 1.0
                    b[nd] = H2
                    continue
                # Top face (iz=0, dam base): Neumann (impermeable or drain)
                if iz == 0:
                    # Drain-hole simplification: apply reduced head at drain
                    # location only when it is lower than the no-drain natural
                    # head, so the pressure-relief surrogate cannot recharge
                    # the foundation in strong-curtain cases.
                    if ix == drain_col and apply_drain:
                        h_drain = self._drain_relief_head(H1, H2, drain_eff)
                        A[nd, nd] = 1.0
                        b[nd] = h_drain
                        continue
                    # Impermeable: zero normal flux, equivalent to H[iz=0]=H[iz=1]
                    A[nd, nd] = 1.0
                    A[nd, idx(1, ix)] = -1.0
                    continue
                # Bottom face (iz=nz-1): Neumann
                if iz == nz - 1:
                    A[nd, nd] = 1.0
                    A[nd, idx(nz - 2, ix)] = -1.0
                    continue

                # Interior node: harmonic-mean permeability
                def K_x(iz_, ix1, ix2):
                    k1 = K_field[iz_, min(ix1, nx - 1)]
                    k2 = K_field[iz_, min(ix2, nx - 1)]
                    return 2 * k1 * k2 / (k1 + k2 + 1e-40)

                def K_z(iz1, iz2, ix_):
                    k1 = K_field[min(iz1, nz - 1), ix_]
                    k2 = K_field[min(iz2, nz - 1), ix_]
                    return 2 * k1 * k2 / (k1 + k2 + 1e-40)

                kxm = K_x(iz, ix - 1, ix) / dx**2
                kxp = K_x(iz, ix, ix + 1) / dx**2
                kzm = K_z(iz - 1, iz, ix) / dz**2
                kzp = K_z(iz, iz + 1, ix) / dz**2

                A[nd, nd] = -(kxm + kxp + kzm + kzp)
                A[nd, idx(iz, ix - 1)] = kxm
                A[nd, idx(iz, ix + 1)] = kxp
                A[nd, idx(iz - 1, ix)] = kzm
                A[nd, idx(iz + 1, ix)] = kzp

        H_flat = spsolve(A.tocsr(), b)
        H_field = H_flat.reshape(nz, nx)

        x_nodes = np.linspace(0.0, Lx, nx)
        return x_nodes, H_field[0, :]

    def _seepage(self, H1, H2, K_base, K_curtain, D_base, drain_eff):
        """2-D FDM steady-state seepage; extract dam-base uplift pressure."""
        Lx = self._L_BASE
        nx = self._N_X
        dx = Lx / (nx - 1)
        x_nodes, H_base = self._base_head_profile(H1, H2, K_base, K_curtain, D_base, drain_eff)

        # Uplift distribution (u = γ_w · H, kPa; γ_w = 9.81 kN/m³)
        gamma_w = 9.81  # kN/m³
        uplift_kPa = gamma_w * H_base  # kPa

        # Total uplift force (integration, MN/m)
        U_total = float(np.trapezoid(uplift_kPa, dx=dx) / 1000.0)  # kN/m → MN/m

        # Uplift at the exact dam-base midpoint (MPa).  The default even
        # node count has no node at Lx/2, so interpolate instead of taking
        # the nearest grid node.
        U_center = float(np.interp(0.5 * Lx, x_nodes, uplift_kPa) / 1000.0)

        # Uplift reduction factor (relative to no seepage barrier: linear H1→H2)
        u_no_curtain = gamma_w * (H1 + H2) / 2.0 * Lx / 1000.0  # MN/m
        eta = U_total / max(u_no_curtain, 1e-6)
        eta = float(np.clip(eta, 0.05, 1.0))

        return np.array([U_total, U_center, eta])
