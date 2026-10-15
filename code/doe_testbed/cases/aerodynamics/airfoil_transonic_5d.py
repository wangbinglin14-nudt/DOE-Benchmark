"""
Case A02: Transonic airfoil drag-divergence surrogate
=====================================================

Engineering background
----------------------
Transonic flight (Ma = 0.75–0.85 for airliners, 0.8–1.2 for fighters) is
the primary cruise regime.  When local flow reaches sonic speed (Ma_local = 1),
a shock forms on the airfoil surface, causing a sharp increase in wave
drag — known as *drag divergence*.

Drag-divergence Mach number (M_dd) engineering definition:
  dCD/dMa = 0.1 (i.e. the rate of CD rise with Ma equals 0.1).
This benchmark does not solve that derivative definition for each airfoil.
Instead, it imposes a compact critical-Mach/wave-drag trigger that creates a
drag-rise elbow suitable for DOE stress testing.  It also does not implement
Korn's drag-divergence relation, buffet onset, or shock/boundary-layer
separation physics.

A core airfoil-design objective: delay drag divergence so that cruise Ma
can be as high as possible while maintaining adequate lift.

This case studies the lift-drag characteristics of NACA 4-digit airfoils
with an engineering drag-divergence surrogate.  The installed NeuralFoil API
used here is the low-level ``neuralfoil.get_aero_from_airfoil`` call, which
supplies the baseline airfoil polar as a function of airfoil shape, angle of
attack, and Reynolds number; Mach effects are applied explicitly through
Prandtl-Glauert scaling and an empirical wave-drag term.  Current
AeroSandbox wrappers expose a ``mach`` keyword, but switching this frozen
benchmark response surface to that wrapper would require protocol review and
full resweep.  The case is not a high-fidelity transonic CFD/RANS replacement.

Physics model
-------------
Baseline polar + compressibility/wave-drag surrogate:
  · NeuralFoil baseline polar for NACA 4-digit airfoil geometry, alpha, and Re
  · Prandtl-Glauert lift/profile-drag scaling, beta = sqrt(1 - Ma^2)
  · Empirical wave drag: CD_wave = 20 (Ma - Mcr)^4 for Ma > Mcr
  · Approximate critical Mach number decreases with thickness ratio

The explicit Prandtl-Glauert factor is a subcritical compressibility trend
correction.  At the high end of the declared domain (Ma up to 0.88), the
wave-drag term is only an engineering trigger for the drag-divergence regime,
not a substitute for shock-resolving Euler/RANS, wind-tunnel calibration, or
NeuralFoil Mach-resolved inference.
NASA Glenn similarity-parameter material is used only as boundary evidence:
Mach number controls compressibility similarity, and near/transonic flows can
develop shock waves that strongly affect drag.  That supports the case's
Mach-driven stress-test role, but it is not calibration of the compact
critical-Mach or fourth-power wave-drag trigger.

Response-surface features
-------------------------
CD vs Ma shows a clear “elbow” around Ma ≈ 0.7–0.85:
  · Ma < M_dd: CD changes slowly (profile drag dominates)
  · Ma > M_dd: CD surges (shock drag + shock-induced separation)

The transition from mild-gradient to steep-gradient is the core
transonic DOE challenge:
  · Uniform designs waste samples in the non-divergence region
  · Sequential designs are evaluated on whether they allocate samples near M_dd

Input variables
---------------
Ma      Flight Mach number   [0.3, 0.88]  -     Primary driver (drag-divergence trigger)
alpha   Angle of attack      [-2, 6]      deg   Determines lift demand
tc      Thickness ratio t/c  [0.08, 0.18] -     Affects M_dd (thinner airfoil → higher M_dd)
fc      Max camber f/c       [0.0, 0.06]  -     Affects zero-lift slope
Re      Reynolds number      [1e6, 1e7]   -     Affects BL state (secondary)

Output responses
----------------
CD      Drag coefficient     -   Primary output (including wave drag)
CL      Lift coefficient     -   Operating lift
CL_CD   Lift-to-drag ratio   -   Aerodynamic efficiency

DOE validation value
--------------------
VS-A (space-filling):
    CD is highly heterogeneous in the Ma–alpha plane (small gradient in
    sub-divergence region, large gradient in divergence region).

VS-C (surrogate accuracy):
    The sharp wave-drag rise near Ma ≈ 0.8 degrades surrogate accuracy;
    a standard test for GP / RBF methods in gradient-transition regions.

VS-G (sequential / adaptive):
    Target: build an accurate CD–Ma divergence curve with minimal samples.
    Tests whether EGR (expected gradient reduction) and similar infill
    criteria allocate samples near the drag-divergence region.

References
----------
1. Drela, M. (1989). XFOIL: An Analysis and Design System for Low
   Reynolds Number Airfoils. Springer. DOI:10.1007/978-3-642-84010-4_1
2. Harris, C. D. (1981). Two-Dimensional Aerodynamic Characteristics of
   the NACA 0012 Airfoil in the Langley 8-foot Transonic Pressure Tunnel.
   NASA TM-81927.
   https://ntrs.nasa.gov/search.jsp?R=19810014503
   https://ntrs.nasa.gov/citations/19810014503
3. Sharpe, P. D. & Hansman, R. J. (2025). NeuralFoil: An Airfoil Aerodynamics
   Analysis Tool Using Physics-Informed Machine Learning. arXiv:2503.16323.
   https://arxiv.org/abs/2503.16323
4. AeroSandbox Airfoil API documentation, ``get_aero_from_neuralfoil`` wrapper.
   https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/index.html
5. NASA TM-100527. High Reynolds number transonic tests on a NACA 0012
   airfoil in the Langley 0.3-meter transonic cryogenic tunnel.
   https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19880009187.pdf
6. NASA History SP-4219. Transonic drag-divergence / sound-barrier context.
   https://www.nasa.gov/history/SP-4219/Chapter3.html
7. NASA Glenn Beginner's Guide to Aeronautics, similarity parameters / Mach
   number and compressibility context.
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/similarity-parameters-2/
8. NASA Langley Turbulence Modeling Resource. 2D NACA 0012 airfoil validation.
   https://turbmodels.larc.nasa.gov/naca0012_val.html
9. XFOIL online documentation, convergence and separated-flow operating
   limits for panel/integral-boundary-layer calculations.
   https://web.mit.edu/drela/Public/web/xfoil/
10. NASA NTRS 2022 transonic NACA 0012 turbulence/buffet simulations.
   https://ntrs.nasa.gov/citations/20220017612 and
   https://ntrs.nasa.gov/citations/20220018292

The public NASA anchors support transonic NACA 0012 test-data context,
drag-divergence physics, and what high-fidelity RANS validation datasets look
like. They are not external validation data for this NeuralFoil-baseline plus
explicit Prandtl-Glauert/wave-drag surrogate, nor for derivative M_dd, Korn,
buffet, shock/boundary-layer-separation, grid-converged RANS, or wind-tunnel
calibrated prediction.
The current NTRS transonic-buffet material sharpens that boundary by showing
that pre/post-buffet NACA 0012 analysis uses URANS, DDES, WMLES, NASA FUN3D,
shock-location comparisons, and wind-tunnel conditions; none of those are
implemented here.
They are also narrower than the benchmark input family: the official tunnel
data are for specific NACA 0012 sections and test conditions, not validation
of this benchmark's camber/thickness sweep or the compact critical-Mach /
fourth-power wave-drag trigger applied across the registered domain.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class AirfoilTransonic5D(BaseCase):
    """Transonic airfoil drag characteristics (with shock-drag divergence), 5D→3D."""

    meta = CaseMeta(
        case_id="airfoil_transonic_5d",
        name="Transonic airfoil drag (shock-drag divergence)",
        version="1.0.0",
        domain="aerodynamics",
        tags=["airfoil", "transonic", "shock", "wave_drag", "drag_divergence", "5d", "neuralfoil"],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="Ma",
                symbol="Ma",
                description="Flight Mach number",
                unit="-",
                lower=0.3,
                upper=0.88,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="alpha",
                symbol=r"\alpha",
                description="Angle of attack",
                unit="deg",
                lower=-2.0,
                upper=6.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="tc",
                symbol="t/c",
                description="Thickness-to-chord ratio",
                unit="-",
                lower=0.08,
                upper=0.18,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="fc",
                symbol="f/c",
                description="Maximum camber ratio",
                unit="-",
                lower=0.0,
                upper=0.06,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="Re",
                symbol="Re",
                description="Reynolds number",
                unit="-",
                lower=1e6,
                upper=1e7,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="noise",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="CD",
                symbol="C_D",
                description="Drag coefficient (including wave drag)",
                unit="-",
                is_primary=True,
                expected_range=(0.003, 0.15),
                failure_threshold=0.05,
                failure_direction="above",
            ),
            OutputVariable(
                name="CL",
                symbol="C_L",
                description="Lift coefficient",
                unit="-",
                is_primary=False,
                expected_range=(-0.5, 3.0),
            ),
            OutputVariable(
                name="CL_CD",
                symbol="C_L/C_D",
                description="Lift-to-drag ratio (aero efficiency; can exceed 150 at high CL, low drag)",
                unit="-",
                is_primary=False,
                # thin airfoil + high lift + subsonic low wave-drag combo can push CL/CD > 150
                expected_range=(-60.0, 250.0),
            ),
        ],
        governing_equation=(
            "NeuralFoil baseline polar + explicit engineering transonic corrections:\n"
            "  beta_pg = sqrt(max(1 - Ma^2, 0.01))\n"
            "  CL = CL_NeuralFoil(alpha, Re, NACA4) / beta_pg\n"
            "  CD = CD_NeuralFoil(alpha, Re, NACA4) / beta_pg + CD_wave\n"
            "  Mcr = 0.75 - 0.06*(t/c)/0.12; CD_wave = 20*(Ma - Mcr)^4 for Ma > Mcr\n"
            "Scope: drag-divergence DOE surrogate, not direct RANS/SA transonic CFD"
        ),
        pde_type="algebraic",
        spatial_dim=2,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Drela, M. (1989). XFOIL. Springer. DOI:10.1007/978-3-642-84010-4_1",
            "Harris, C. D. (1981). NACA 0012 Transonic. NASA TM-81927.",
            "Sharpe, P. D. & Hansman, R. J. (2025). NeuralFoil: An Airfoil Aerodynamics "
            "Analysis Tool Using Physics-Informed Machine Learning. arXiv:2503.16323, "
            "https://arxiv.org/abs/2503.16323",
            "AeroSandbox Airfoil API documentation, get_aero_from_neuralfoil wrapper, "
            "https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/index.html",
            "NASA NTRS, High Reynolds number transonic tests on a NACA 0012 airfoil in "
            "the Langley 0.3-meter transonic cryogenic tunnel, NASA-TM-100527, "
            "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19880009187.pdf",
            "NASA NTRS, Harris (1981), Two-Dimensional Aerodynamic Characteristics of "
            "the NACA 0012 Airfoil in the Langley 8-Foot Transonic Pressure Tunnel, "
            "NASA-TM-81927. https://ntrs.nasa.gov/search.jsp?R=19810014503; "
            "stable citation metadata: https://ntrs.nasa.gov/citations/19810014503",
            "NASA History, Research in Supersonic Flight and the Breaking of the Sound Barrier, "
            "drag-divergence discussion, https://www.nasa.gov/history/SP-4219/Chapter3.html",
            "NASA Glenn Beginner's Guide to Aeronautics, similarity parameters / Mach number "
            "and compressibility context, "
            "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/similarity-parameters-2/",
            "NASA Langley Turbulence Modeling Resource, 2D NACA 0012 Airfoil Validation, "
            "RANS validation-dataset context, https://turbmodels.larc.nasa.gov/naca0012_val.html",
            "XFOIL online documentation, convergence and separated-flow operating "
            "limits for panel/integral-boundary-layer calculations, "
            "https://web.mit.edu/drela/Public/web/xfoil/",
            "NASA NTRS, Turbulence Simulations of Transonic Flows over an "
            "NACA-0012 Airfoil, URANS/DDES/WMLES/FUN3D and pre/post-buffet "
            "boundary context, https://ntrs.nasa.gov/citations/20220017612; "
            "presentation metadata: https://ntrs.nasa.gov/citations/20220018292",
        ],
        solver_library="neuralfoil + numpy",
        solver_details=(
            "Low-level neuralfoil.get_aero_from_airfoil large model for baseline airfoil "
            "polar; Mach effects are explicit Prandtl-Glauert and empirical wave-drag "
            "corrections, not a direct Mach-resolved NeuralFoil, Euler, or RANS/SA "
            "transonic CFD solve. AeroSandbox's Airfoil.get_aero_from_neuralfoil "
            "wrapper exposes a mach argument in current documentation, but the formal "
            "response surface is pinned to the low-level NeuralFoil call plus explicit "
            "PG/wave-drag corrections; changing that backend is a response-surface "
            "change requiring protocol review and full resweep. Private physical-input "
            "evaluation rejects values outside the declared transonic airfoil domain "
            "before constructing the NACA section or calling NeuralFoil. No derivative-based "
            "M_dd solution, Korn relation, buffet onset, or shock/boundary-layer "
            "separation model. Official NASA transonic NACA 0012 tunnel data are "
            "single-section context rather than calibration of this benchmark's "
            "camber/thickness sweep or compact Mcr/fourth-power wave-drag trigger. "
            "NASA Glenn similarity-parameter material supports the Mach/compressibility/"
            "shock-wave boundary but is not calibration of the critical-Mach trigger. "
            "NASA transonic-buffet URANS/DDES/WMLES/FUN3D material is omitted-"
            "workflow boundary evidence, not validation of this algebraic "
            "response surface or its shock-location/buffet behavior."
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 1.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["threshold_jump", "heterogeneous", "smooth_unimodal"],
        effective_dim=5,
        important_variables=["Ma", "alpha", "tc", "fc", "Re"],
        interaction_pairs=[("Ma", "tc"), ("Ma", "alpha")],
        has_failure_boundary=True,
        failure_condition="CD > 0.05 (beyond acceptable range after drag divergence)",
        sobol_S1={"Ma": 0.883, "alpha": 0.013, "tc": 0.024, "fc": 0.000, "Re": 0.004},
        sobol_ST={"Ma": 0.949, "alpha": 0.017, "tc": 0.093, "fc": 0.004, "Re": 0.007},
        sobol_computation=(
            "salib_saltelli_n2048_seed42_CD_output; CD Sobol indices are dominated by Ma "
            "and tc, while important_variables/effective_dim record the multi-output "
            "backend-input union because alpha, camber, and Re affect CL and/or CL_CD"
        ),
        applicable_scenarios=["VS-A", "VS-C", "VS-F", "VS-G"],
        scenario_notes={
            "VS-A": "CD is highly heterogeneous in the Ma–tc plane; divergence-region gradient far exceeds subsonic region.",
            "VS-C": "Sharp wave-drag rise near Ma≈0.8 degrades surrogate accuracy.",
            "VS-F": (
                "For the stored CD Sobol metadata, Ma and tc dominate. The case-level "
                "important-variable list is a multi-output union because alpha, fc, "
                "and Re also affect CL and/or CL_CD."
            ),
            "VS-G": "Sequential modelling target: test whether designs capture the CD vs Ma divergence curve efficiently.",
        },
        recommended_n_train={"default": 100},
        n_saturation=400,
        noise_factor_candidates=["Re"],
        control_factor_candidates=["Ma", "alpha", "tc", "fc"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        """Return validated physical inputs or reject private-domain violations."""
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "airfoil_transonic_5d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        Ma = X_phys[:, 0]
        alpha = X_phys[:, 1]
        tc = X_phys[:, 2]
        fc = X_phys[:, 3]
        Re = X_phys[:, 4]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(Ma < 0.3 - tol)
            or np.any(Ma > 0.88 + tol)
            or np.any(alpha < -2.0 - tol)
            or np.any(alpha > 6.0 + tol)
            or np.any(tc < 0.08 - tol)
            or np.any(tc > 0.18 + tol)
            or np.any(fc < 0.0 - tol)
            or np.any(fc > 0.06 + tol)
            or np.any(Re < 1e6 * (1.0 - tol))
            or np.any(Re > 1e7 * (1.0 + tol))
        ):
            raise ValueError(
                "airfoil_transonic_5d physical inputs must stay inside the declared "
                "Ma/alpha/tc/fc/Re benchmark domain before NeuralFoil evaluation"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        Ma = X_phys[:, 0]
        alpha = X_phys[:, 1]
        tc = X_phys[:, 2]
        fc = X_phys[:, 3]
        Re = X_phys[:, 4]
        try:
            import neuralfoil as nf
        except ImportError:
            raise ImportError("pip install neuralfoil")

        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)

        import aerosandbox as asb

        for i in range(n):
            try:
                naca_name = self._naca4_name(fc=float(fc[i]), tc=float(tc[i]))
                af = asb.Airfoil(naca_name)
                # The frozen low-level NeuralFoil airfoil API has no Mach
                # argument here, so compressibility is handled by the explicit
                # P-G/wave-drag corrections below.
                beta_pg = np.sqrt(max(1.0 - Ma[i] ** 2, 0.01))
                alpha_eq = float(alpha[i])
                aero = nf.get_aero_from_airfoil(
                    airfoil=af,
                    alpha=alpha_eq,
                    Re=float(Re[i]),
                    model_size="large",
                )
                CL = float(np.squeeze(aero["CL"])) / beta_pg  # P-G lift correction
                # CD includes profile drag (P-G corrected) + wave drag (Ma effect approx)
                CD0 = float(np.squeeze(aero["CD"]))
                if not np.isfinite(CL) or not np.isfinite(CD0) or CD0 <= 0.0:
                    raise RuntimeError(
                        f"non-finite or non-positive NeuralFoil baseline polar: CL={CL}, CD0={CD0}"
                    )
                # Shock drag (simplified model: surges when Ma > Mcr)
                Mcr = 0.75 - 0.06 * tc[i] / 0.12  # critical Ma (approx)
                if Ma[i] > Mcr:
                    CD_wave = 20 * (Ma[i] - Mcr) ** 4
                else:
                    CD_wave = 0.0
                CD = CD0 / beta_pg + CD_wave
                if CD <= 0.0:
                    raise RuntimeError(f"non-positive corrected drag coefficient: CD={CD}")
                if not np.all(np.isfinite([CD, CL, CL / CD])):
                    raise RuntimeError("non-finite transonic airfoil response")
                Y[i, 0] = CD
                Y[i, 1] = CL
                Y[i, 2] = CL / CD
            except Exception as exc:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "airfoil_transonic_5d NeuralFoil/AeroSandbox backend failed for "
                    f"physical input {x_text}; refusing silent NaN output because "
                    "the formal benchmark response surface requires this backend."
                ) from exc
        return Y

    @staticmethod
    def _naca4_name(fc: float, tc: float) -> str:
        """Return a standard NACA 4-digit name for the surrogate airfoil.

        The benchmark varies maximum camber and fixes its position at 0.4c.
        For zero camber the position digit is physically irrelevant; use the
        standard symmetric-airfoil spelling ``00xx`` rather than ``04xx``.
        """
        m_d = int(round(fc * 100))
        t_d = int(round(tc * 100))
        p_d = 0 if m_d == 0 else 4
        return f"naca{m_d}{p_d}{t_d:02d}"
