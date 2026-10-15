"""
Case A03: Wing aerodynamic characteristics (Prandtl lifting-line, Fourier form)
==============================================================================

Engineering background
----------------------
Preliminary aerodynamic analysis of a tapered wing during aircraft
concept design.  This case uses the **Glauert Fourier-series form of
Prandtl's lifting-line theory** (a lower-order finite-wing method, not VLM) to
compute lift, induced drag, and Oswald efficiency.  It provides low-order
conceptual-design trend-screening data at far lower cost than VLM or RANS
CFD, but it is not a calibrated VLM/CFD replacement.
It reports induced-drag-only finite-wing coefficients; profile/parasite drag,
viscous stall, camber and zero-lift-angle effects, dihedral, winglets,
fuselage interference, ground effect, and true swept lifting-surface physics
are outside this surrogate.

This case computes the lift coefficient, induced drag, and Oswald
efficiency factor of a trapezoidal wing at a given flight condition,
studying interaction effects between wing geometry and flight parameters.
The lift response is monotone in angle of attack over most fixed-geometry
slices, but the multi-output response is not globally monotone because
induced drag scales with CL² and Oswald efficiency becomes ill-conditioned
near zero lift.

Note
----
The ``case_id`` and file name retain the historical "vlm" label for
compatibility, but the actual solver is PLLT (Prandtl lifting-line
theory, Glauert Fourier method), not AVL/VLM.  AVL-style vortex-lattice
methods represent rigid-aircraft thin lifting surfaces with horseshoe-vortex
lattices and are best suited to small angles of attack/sideslip; AVL also
supports slender-body modelling, control deflections, and Trefftz-plane
induced-drag analysis inside a broader conceptual-aircraft workflow. This
benchmark does not construct a lifting-surface lattice, control-surface
model, body model, or Trefftz-plane AVL analysis.

Governing equations (PLLT + Glauert Fourier series)
---------------------------------------------------
With the transform y = (b/2)·cos(θ), the circulation is expanded as an
odd Fourier series:
  Γ(θ) = 2b·V∞·Σ Aₙ·sin(nθ)
and the lifting-line equation is enforced at N collocation points,
yielding an N×N linear system for Aₙ.  Then
  CL  = π·AR·A₁
  CDi = π·AR·Σₙ n·Aₙ²
      = CL²/(π·AR)·(1 + δ), δ = Σₙ≥₂ n·(Aₙ/A₁)² when A₁ ≠ 0
  e   = CL²/(π·AR·CDi)
At exactly zero lift with nonzero twist loading, CDi remains finite and e
approaches zero; at the trivial zero-loading state CL = CDi = 0, e is reported
as 1.0 as a finite-output convention for the benchmark.

Prandtl–Glauert compressibility correction:
  a0 → a0/β, β = √(1 − Ma²); A₁ ∝ 1/β ⇒ CL = CL_incomp/β

Sweep correction (conceptual normal-flow scaling, not a swept-wing VLM):
  CL_sw = CL · cos²(Λ)
The sweep term is a deliberately simple normal-flow trend factor.  It should
not be used as evidence for swept-wing vortex-lattice, panel-method, or CFD
accuracy.

Input variables
---------------
AR      Aspect ratio          [5, 14]    -     Main effect: lift slope, induced drag
TR      Taper ratio           [0.2, 1.0] -     Affects lift distribution (Oswald eff.)
sweep   Quarter-chord sweep   [0, 35]    deg   High-subsonic performance
alpha   Angle of attack       [0, 8]     deg   Directly controls CL
Ma      Flight Mach number    [0.1, 0.7] -     Compressibility correction
twist   Tip twist angle       [-5, 5]    deg   Washout; affects lift distribution

Output responses
----------------
CL      Lift coefficient              -
CDi     Induced drag coefficient      -
e       Oswald efficiency factor       -

DOE validation value
--------------------
VS-B: For the CL headline output, alpha dominates and tip twist is second;
      sweep, Ma, and AR are secondary; TR mainly affects lift distribution/e.
VS-C: The low-order PLLT surrogate is smooth and inexpensive to sample.

References
----------
1. Katz, J. & Plotkin, A. (2001). Low-Speed Aerodynamics, 2nd ed. Cambridge.
   §8.1 (Finite Wing: The Lifting-Line Model), §8.3 (The Effect of Aspect Ratio).
2. Anderson, J. D. (2010). Fundamentals of Aerodynamics, 5th ed. McGraw-Hill.
   §5.3 (Prandtl's Classical Lifting-Line Theory) — Glauert transform
   and the formula CDi = CL²/(π·e·AR).
3. NASA Glenn. Induced Drag Coefficient.
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/
4. NASA Glenn. Winglets / induced-drag context.
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/winglets/
5. NASA Glenn. Factors that affect drag.
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/factors-that-affect-drag/
6. NASA History SP-4219. Prandtl-Glauert compressibility-rule background.
   https://www.nasa.gov/history/SP-4219/Chapter3.html
7. Drela, M. & Youngren, H. AVL documentation.
   https://web.mit.edu/drela/Public/web/avl/avl_doc.txt
8. Drela, M. & Youngren, H. AVL User Primer.
   https://web.mit.edu/drela/Public/web/avl/AVL_User_Primer.pdf
9. NASA NTRS. Applications of vortex lattice theory to preliminary
   aerodynamic design.
   https://ntrs.nasa.gov/citations/19760021082
   PDF: https://ntrs.nasa.gov/api/citations/19760021082/downloads/19760021082.pdf
   Collected SP-405 volume: https://ntrs.nasa.gov/citations/19760021075
10. NASA OpenVSP VSPAERO basics.
   https://www.nasa.gov/reference/openvsp-vspaero-basics/

The public NASA/MIT anchors support only the induced-drag identity,
Prandtl-Glauert background, and the boundary definition of a true AVL/VLM
lifting-surface or OpenVSP/VSPAERO potential-flow workflow.  They are not
external validation data for this PLLT/Fourier surrogate, swept-wing VLM
accuracy, OpenVSP/VSPAERO analysis, profile/parasite drag, stall, panel-method,
or CFD behaviour. The OpenVSP/VSPAERO page is a current tool/workflow boundary
source, including the documented VSPAERO v3.45 / solver v7 transition, not a
stable numerical-reference dataset for this benchmark response surface.
NASA Glenn winglet and drag-factor pages are used only to clarify that
induced drag is one drag component and that profile/skin-friction, form,
wave, viscosity, and compressibility effects belong outside this
induced-drag-only response.
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class WingVLM6D(BaseCase):
    """Historical VLM-labelled wing case solved by a PLLT/Fourier surrogate."""

    meta = CaseMeta(
        case_id="wing_vlm_6d",
        name="Wing PLLT aerodynamic characteristics (historical VLM label)",
        version="1.0.0",
        domain="aerodynamics",
        tags=["wing", "vlm", "pllt", "aerodynamics", "aspect_ratio", "induced_drag", "lifting_line", "6d"],
        dim_input=6,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="AR",
                symbol="AR",
                description="Wing aspect ratio",
                unit="-",
                lower=5.0,
                upper=14.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="TR",
                symbol=r"\lambda",
                description="Taper ratio",
                unit="-",
                lower=0.2,
                upper=1.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="sweep",
                symbol=r"\Lambda_{c/4}",
                description="Quarter-chord sweep angle",
                unit="deg",
                lower=0.0,
                upper=35.0,
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
                lower=0.0,
                upper=8.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="Ma",
                symbol="Ma",
                description="Flight Mach number",
                unit="-",
                lower=0.1,
                upper=0.70,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="twist",
                symbol=r"\theta_{tip}",
                description="Tip twist angle (negative = washout)",
                unit="deg",
                lower=-5.0,
                upper=5.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="CL",
                symbol="C_L",
                description="Wing lift coefficient (can be negative at α=0 with large washout)",
                unit="-",
                is_primary=True,
                # α_root ∈ [0°, 8°] with tip_twist ∈ [-5°, 5°]; effective AoA can fall below zero-lift angle → CL < 0
                expected_range=(-0.3, 1.5),
            ),
            OutputVariable(
                name="CDi",
                symbol="C_{Di}",
                description="Induced drag coefficient",
                unit="-",
                is_primary=True,
                # CDi = CL^2/(pi*AR*e); AR=5, CL=1.27, e=0.7 -> CDi~0.15
                expected_range=(0.0, 0.2),
            ),
            OutputVariable(
                name="e",
                symbol="e",
                description="Oswald efficiency factor (ill-conditioned when A_1→0; see docstring)",
                unit="-",
                is_primary=False,
                # Near zero-lift with nonzero twist loading, the denominator-free
                # PLLT CDi formula stays finite and e -> 0.  At the trivial
                # zero-loading state CL = CDi = 0, e = 1 is a finite-output
                # benchmark convention.
                expected_range=(0.0, 1.0),
            ),
        ],
        governing_equation=(
            "Prandtl Lifting-Line Theory (PLLT):\n"
            "  Γ(θ)=2b·V∞·ΣAₙ·sin(nθ), station equation:\n"
            "  ΣAₙ[sin(nθ)+n·μ·sin(nθ)/sin(θ)]=μ·α_eff(θ), μ=c·a0/(4b)\n"
            "  CL=π·AR·A₁, CDi=π·AR·ΣnAₙ²=CL²/(π·AR)·(1+δ), e=CL²/(π·AR·CDi)\n"
            "Prandtl–Glauert correction: a0→a0/√(1−Ma²), CL→CL_incomp/√(1−Ma²)"
        ),
        pde_type="integral",
        spatial_dim=3,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Katz, J. & Plotkin, A. (2001). Low-Speed Aerodynamics, 2nd ed. Cambridge.",
            "Anderson, J. D. (2010). Fundamentals of Aerodynamics, 5th ed. McGraw-Hill, "
            "Prandtl lifting-line theory and induced-drag relation.",
            "NASA Glenn, Induced Drag Coefficient, "
            "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/",
            "NASA Glenn, Winglets / induced-drag context, "
            "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/winglets/",
            "NASA Glenn, Factors that Affect Drag, omitted profile/skin-friction, "
            "form, wave, viscosity, and compressibility drag context, "
            "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/factors-that-affect-drag/",
            "NASA History SP-4219, Prandtl-Glauert compressibility-rule background, "
            "https://www.nasa.gov/history/SP-4219/Chapter3.html",
            "Drela, M. & Youngren, H. AVL documentation, Vortex-Lattice Modeling "
            "Principles, rigid-aircraft thin-lifting-surface/small-angle scope, "
            "horseshoe-vortex lattices, slender-body modelling, and "
            "Trefftz-plane induced-drag analysis within a broader conceptual-"
            "aircraft workflow. "
            "https://web.mit.edu/drela/Public/web/avl/avl_doc.txt; "
            "AVL User Primer. "
            "https://web.mit.edu/drela/Public/web/avl/AVL_User_Primer.pdf",
            "NASA NTRS, Applications of vortex lattice theory to preliminary aerodynamic "
            "design, rigid-aircraft VLM/Trefftz-plane planform, twist, camber, flap, "
            "and control-surface design context, "
            "specific paper: https://ntrs.nasa.gov/citations/19760021082, "
            "PDF: https://ntrs.nasa.gov/api/citations/19760021082/downloads/19760021082.pdf, "
            "SP-405 collection: https://ntrs.nasa.gov/citations/19760021075",
            "NASA OpenVSP VSPAERO basics, potential-flow aerodynamics and "
            "DegenGeom/OpenVSP geometry workflow boundary context, including "
            "the documented VSPAERO v3.45 / solver v7 transition as "
            "workflow/version boundary evidence rather than validation data, "
            "https://www.nasa.gov/reference/openvsp-vspaero-basics/",
        ],
        solver_library="numpy (self-implemented Prandtl lifting-line + P-G correction)",
        solver_details=(
            "PLLT with 40 Fourier terms, Prandtl-Glauert subsonic "
            "compressibility correction, conceptual cos^2 sweep scaling, and "
            "induced-drag-only output; no profile/parasite drag, viscous stall, "
            "NASA Glenn drag-factor or winglet workflow calibration, "
            "camber/zero-lift-angle, dihedral, winglet, fuselage-interference, "
            "ground-effect, true swept lifting-surface, AVL/VLM horseshoe-vortex "
            "lattice, control-surface/body model, Trefftz-plane post-processing, "
            "flap/control-surface design, multi-lifting-surface VLM design, "
            "OpenVSP/VSPAERO geometry/DegenGeom potential-flow workflow, "
            "VSPAERO version-transition workflow, "
            "panel-method, or CFD model; low-level PLLT helper rejects "
            "non-finite inputs, AR <= 0, TR <= 0, non-subsonic or negative Mach, "
            "and invalid Fourier-term counts rather than extrapolating the "
            "declared finite-wing/P-G surrogate outside its formula domain; "
            "private physical-input evaluation rejects malformed arrays and "
            "AR/TR/sweep/alpha/Ma/twist values outside the declared benchmark "
            "domain before PLLT/Fourier assembly; non-finite or out-of-contract "
            "CL/CDi/e responses are rejected before dispatch can archive a row"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 1.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_unimodal", "strong_interaction"],
        effective_dim=6,
        important_variables=["AR", "alpha", "TR", "sweep", "twist", "Ma"],
        interaction_pairs=[
            ("AR", "TR"),
            ("AR", "sweep"),
            ("AR", "alpha"),
            ("AR", "Ma"),
            ("AR", "twist"),
            ("TR", "alpha"),
            ("TR", "Ma"),
            ("TR", "twist"),
            ("sweep", "alpha"),
            ("sweep", "Ma"),
            ("sweep", "twist"),
            ("alpha", "Ma"),
            ("alpha", "twist"),
            ("Ma", "twist"),
        ],
        has_failure_boundary=False,
        sobol_S1={
            "AR": 0.008,
            "TR": 0.000,
            "sweep": 0.028,
            "alpha": 0.716,
            "Ma": 0.012,
            "twist": 0.214,
        },
        sobol_ST={
            "AR": 0.012,
            "TR": 0.000,
            "sweep": 0.040,
            "alpha": 0.732,
            "Ma": 0.018,
            "twist": 0.220,
        },
        sobol_computation="salib_saltelli_n2048_CL_output_pllt_mu_fixed",
        applicable_scenarios=["VS-B", "VS-C", "VS-D"],
        scenario_notes={
            "VS-B": (
                "For the CL output used by the stored Sobol metadata, alpha dominates and "
                "tip twist is second; sweep, Ma, and AR are secondary. TR has near-zero "
                "CL effect in this metadata but can affect lift distribution and e."
            ),
            "VS-C": (
                "The low-order PLLT surrogate is smooth and inexpensive to sample; "
                "accuracy claims are limited to finite-wing trend/identity checks, "
                "not profile-drag, stall, or swept-wing validation."
            ),
            "VS-D": (
                "Compact 6-D conceptual-wing surrogate with a dominant alpha/twist "
                "lift response and secondary geometry/compressibility effects; use "
                "small-sample convergence curves without pre-claiming a sufficient N."
            ),
        },
        recommended_n_train={"default": 50},
        n_saturation=200,
        noise_factor_candidates=["Ma"],
        control_factor_candidates=["AR", "TR", "sweep", "alpha", "twist"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        """Return validated physical inputs or reject private-domain violations."""
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "wing_vlm_6d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        AR = X_phys[:, 0]
        TR = X_phys[:, 1]
        sweep = X_phys[:, 2]
        alpha = X_phys[:, 3]
        Ma = X_phys[:, 4]
        twist = X_phys[:, 5]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(AR < 5.0 - tol)
            or np.any(AR > 14.0 + tol)
            or np.any(TR < 0.2 - tol)
            or np.any(TR > 1.0 + tol)
            or np.any(sweep < 0.0 - tol)
            or np.any(sweep > 35.0 + tol)
            or np.any(alpha < 0.0 - tol)
            or np.any(alpha > 8.0 + tol)
            or np.any(Ma < 0.1 - tol)
            or np.any(Ma > 0.70 + tol)
            or np.any(twist < -5.0 - tol)
            or np.any(twist > 5.0 + tol)
        ):
            raise ValueError(
                "wing_vlm_6d physical inputs must stay inside the declared "
                "AR/TR/sweep/alpha/Ma/twist benchmark domain before PLLT evaluation"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                response = self._lifting_line(
                    AR=float(X_phys[i, 0]),
                    TR=float(X_phys[i, 1]),
                    sweep_deg=float(X_phys[i, 2]),
                    alpha_deg=float(X_phys[i, 3]),
                    Ma=float(X_phys[i, 4]),
                    twist_deg=float(X_phys[i, 5]),
                )
                if (
                    response.shape != (3,)
                    or not np.all(np.isfinite(response))
                    or response[0] < -0.3
                    or response[0] > 1.5
                    or response[1] < 0.0
                    or response[1] > 0.2
                    or not (0.0 <= response[2] <= 1.0)
                ):
                    raise ValueError(
                        "non-finite or non-physical PLLT response "
                        f"CL/CDi/e={np.array2string(response, precision=6, separator=', ')}"
                    )
                Y[i] = response
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "wing_vlm_6d PLLT/Fourier wing solve failed for "
                    f"physical input {x_text}; refusing silent NaN output "
                    "because the formal benchmark response surface requires "
                    "finite finite-wing lift and induced-drag values inside "
                    "the declared output ranges."
                ) from None
        return Y

    @staticmethod
    def _lifting_line(AR, TR, sweep_deg, alpha_deg, Ma, twist_deg, N_terms=40):
        """Prandtl Lifting-Line Theory (Fourier series method).

        Valid range: low-speed straight wings (sweep < 35°), Ma < 0.7
        (P-G correction valid).

        Algorithm:
          Discretise the semi-span into N theta stations.
          Assume circulation Γ(θ) = V∞·c(θ)·Σ Aₙ·sin(nθ).
          Enforce Kutta–Joukowski at each station.
          Solve the N×N linear system for Aₙ.
          Derive CL and CDi.

        Parameters
        ----------
        AR, TR, sweep_deg, alpha_deg, Ma, twist_deg : see meta
        N_terms : number of Fourier terms (default 40)

        Returns
        -------
        [CL, CDi, e]
        """
        if not np.all(np.isfinite([AR, TR, sweep_deg, alpha_deg, Ma, twist_deg])):
            raise ValueError("PLLT helper inputs must be finite.")
        if AR <= 0.0:
            raise ValueError("PLLT helper requires AR > 0.")
        if TR <= 0.0:
            raise ValueError("PLLT helper requires TR > 0.")
        if Ma < 0.0 or Ma >= 1.0:
            raise ValueError("Prandtl-Glauert correction requires 0 <= Ma < 1.")
        if not np.isfinite(N_terms) or int(N_terms) != N_terms or N_terms < 2:
            raise ValueError("PLLT helper requires an integer N_terms >= 2.")

        # Prandtl-Glauert compressibility correction factor.  The declared
        # benchmark domain has Ma <= 0.7, so no numerical floor is needed.
        beta_arg = 1.0 - Ma**2
        if not np.isfinite(beta_arg) or beta_arg <= 0.0:
            raise ValueError(
                "Prandtl-Glauert correction requires finite subsonic Mach; "
                f"received Ma={Ma!r}."
            )
        beta = np.sqrt(beta_arg)

        # Reference parameters
        S_ref = 1.0
        b = np.sqrt(AR * S_ref)
        c_root = 2.0 * S_ref / (b * (1.0 + TR))
        c_tip = c_root * TR

        # 2-D lift slope (thin-airfoil theory + Prandtl–Glauert compressibility correction)
        # P-G correction divides the incompressible section lift slope a0 = 2π by β,
        # embedded in the PLLT matrix; the resulting A1 already contains the 1/β factor,
        # so CL = π·AR·A1 (no extra β multiplication).
        a0 = 2.0 * np.pi / beta  # /rad, P-G corrected section lift slope

        # Theta stations (avoid 0 and π)
        N = N_terms
        theta = np.linspace(np.pi / (2 * N), np.pi * (1 - 1 / (2 * N)), N)
        eta = np.cos(theta)  # cosine spanwise stations, eta = y / (b/2)

        c_at_eta = c_root + (c_tip - c_root) * np.abs(eta)
        # Geometric AoA with linear tip twist. Negative twist is washout, so
        # the local tip angle of attack is lower than the root angle.
        alpha_geom = np.deg2rad(alpha_deg) + np.deg2rad(twist_deg) * np.abs(eta)

        # Fourier terms (all terms; even terms contribute zero by symmetry)
        n_arr = np.arange(1, N + 1)

        # Build linear system  (Katz & Plotkin Eq. 8.27; Anderson Eq. 5.45)
        # Σ Aₙ sin(nθ)[1 + μ·n/sin(θ)] = μ·α_eff(θ)
        # where μ = c·a₀/(4b),  b = full wingspan
        mu = c_at_eta * a0 / (4.0 * b)
        sin_n_theta = np.sin(n_arr[None, :] * theta[:, None])  # (N, N)
        LHS = sin_n_theta * (1.0 + mu[:, None] * n_arr[None, :] / np.sin(theta)[:, None])
        RHS = mu * alpha_geom

        # Solve for coefficients Aₙ.  A singular collocation matrix would mean
        # this surrogate is undefined at the requested point, so do not replace
        # the model with a least-squares approximation.
        try:
            A_n = np.linalg.solve(LHS, RHS)
        except np.linalg.LinAlgError as exc:
            raise ValueError("PLLT Fourier collocation matrix is singular.") from exc
        if not np.all(np.isfinite(A_n)):
            raise ValueError("PLLT Fourier solve produced non-finite coefficients.")

        # Lift coefficient (standard PLLT formula)
        # a0 = 2π/β embeds P-G correction into the matrix; A1 ∝ 1/β,
        # so CL = π·AR·A1 ≡ CL_incomp/β, increasing with Ma. ✓
        A1 = A_n[0]
        CL = np.pi * AR * A1  # PLLT + P-G: CL ∝ 1/β, correctly increases with Ma

        # Induced drag coefficient.  The direct PLLT identity avoids division
        # by A1, so twisted zero-lift states keep their finite induced drag
        # instead of being silently converted to CDi=0 and e=1.
        CDi = np.pi * AR * np.sum(n_arr * A_n**2)
        if not np.isfinite(CDi) or CDi < 0.0:
            raise ValueError(f"non-physical PLLT induced drag CDi={CDi!r}.")
        if CDi > 0.0:
            e = CL**2 / (np.pi * AR * CDi)
        else:
            e = 1.0
        if not np.isfinite(e):
            raise ValueError(f"non-finite PLLT Oswald efficiency e={e!r}.")
        e = min(max(float(e), 0.0), 1.0)

        # Conceptual sweep correction: normal-flow lift scaling, not a swept-wing
        # lifting-surface/VLM solve.
        sweep_rad = np.deg2rad(sweep_deg)
        CL_sw = CL * np.cos(sweep_rad) ** 2
        CDi_sw = CDi * np.cos(sweep_rad) ** 4

        return np.array([CL_sw, CDi_sw, e])
