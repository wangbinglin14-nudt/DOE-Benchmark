"""
Case A01: Low-speed airfoil lift-drag polar (with stall region)
===============================================================

Engineering background
----------------------
Airfoils are the fundamental building block of aerodynamic design for
aircraft, UAVs, and wind-turbine blades.  In the attached low-subsonic
regime, the lift coefficient C_L increases roughly linearly with angle of
attack α.
When α exceeds the stall angle (approx. 12°–18°, depending on Re and
airfoil shape), the boundary layer separates massively, causing C_L to
roll over or plateau and drag coefficient C_D to surge — this is *stall*.

The high-drag stall region is both a safety-relevant boundary and a classic DOE challenge:
  · Pre-stall: smooth response surface, moderate gradients
  · Near stall: C_L gradient with respect to α changes abruptly (threshold)
  · Post-stall: highly nonlinear; poorly space-filling designs miss this region

Governing equations
-------------------
The formal benchmark backend is NeuralFoil, a physics-informed neural-network
airfoil-polar surrogate used here through its low-level airfoil API.  It is
treated as the fixed response-surface backend, not as a case-specific XFOIL,
RANS, or wind-tunnel validation claim; the current
AeroSandbox/NeuralFoil path has a noticeable first-call initialization cost
and then evaluates at a few milliseconds per sample in this wrapper.

Underlying physics (XFoil):
  · Potential flow: panel method solving the Laplace equation ∇²φ = 0
  · Boundary layer: integral von Kármán equation (closure from Drela 1989)
  · Stall detection: critical dissipation angle / wake shape factor

The analytical thin-airfoil + stall model is retained only as a diagnostic
helper.  It is not used by ``evaluate`` because mixing NeuralFoil and
analytical fallback rows would change the benchmark response surface.
The installed low-level ``neuralfoil.get_aero_from_airfoil`` call is used as
the baseline polar; Mach dependence is imposed by an explicit
Prandtl-Glauert scalar correction (``1 / sqrt(1 - Ma^2)``) on both CL and CD
over Ma <= 0.40.  Current AeroSandbox wrappers expose a ``mach`` keyword, but
switching this formal benchmark backend would change the archived response
surface and would require a deliberate protocol revision plus full resweep.
This is a low-subsonic/mild-compressibility benchmark stressor, not a
Mach-resolved CFD or wind-tunnel validation.
The public NASA/NACA0012 anchors are narrower than the benchmark family:
NASA TM-4074 is single-section NACA 0012 low-speed Mach/Reynolds context,
while NASA TMR is a validation-dataset boundary for higher-fidelity CFD
workflows rather than calibration of the frozen low-level NeuralFoil backend
or the full NACA00xx thickness sweep.
NASA Glenn inclination/stall pages are used only as general boundary evidence:
they support the small-angle near-linear lift trend, boundary-layer separation
and abrupt lift/drag changes at high inclination, and the fact that stall-point
prediction normally needs carefully matched wind-tunnel testing rather than a
compact surrogate assertion.
This case also intentionally omits camber, flaps/control-surface deflection,
surface roughness, explicit trip/transition controls, turbulence-model
selection, RANS/LES grid-convergence studies, dynamic-stall hysteresis,
unsteady pitching/plunging motion, 3-D finite-wing effects, rotating-blade
corrections, icing/contamination, aeroelastic coupling, and acoustic effects.

Input variables
---------------
alpha   Angle of attack   α    [-5, 20] deg         Primary driver; stall threshold along this dim
Re      Reynolds number   Re   [5e4, 5e6] (log)     Affects stall angle and transition
Ma      Mach number       Ma   [0.0, 0.40]          Low-subsonic/mild-compressibility stressor
t_c     Thickness ratio   t/c  [0.06, 0.18]         Affects post-stall behaviour (thicker = milder)

Output responses
----------------
CL      Lift coefficient   C_L  dimensionless   Primary output; drops sharply post-stall
CD      Drag coefficient   C_D  dimensionless   Surges post-stall
LoD     Lift-to-drag ratio L/D  dimensionless   = C_L / C_D, overall aerodynamic efficiency

DOE validation value
--------------------
VS-A (space-filling):
    The stall region (α ≈ 12°–18°) has very steep gradients — a classic
    spatially heterogeneous case.  Poorly space-filling designs leave
    large gaps in this region, inflating surrogate error there.

VS-C (surrogate accuracy):
    The lift curve’s nonlinearity and drag surge vary strongly with α (heterogeneous
    covariance), making it a stringent test for GP / RBF / NN surrogates.
    Recommended training set N = 60–100.

VS-G (sequential / adaptive design):
    The high-drag stall boundary α_stall(Re, t/c) is a high-dimensional
    implicit surface, a typical target for sequential methods.  This case
    tests whether adaptive sampling locates that boundary more efficiently
    than space-filling designs; no pre-result sample-saving claim is made.

VS-F (variable screening):
    α is the most important variable (S1 ≈ 0.65, ST ≈ 0.89), followed by
    Re (ST ≈ 0.31); t/c is secondary and Ma is essentially inert
    (ST ≈ 0.001) in the Ma ≤ 0.4 low-speed regime.  Clear importance
    hierarchy for testing Morris / Sobol screening.

Known Sobol indices (uniform / loguniform inputs, SALib Saltelli N_base=2048, seed=42)
--------------------------------------------------------------------------------------
C_L output (see `meta.sobol_S1` / `meta.sobol_ST`):
    alpha : S1=0.651, ST=0.888 (dominant; large S1→ST gap = strong alpha–Re interaction)
    Re    : S1=0.101, ST=0.306
    t_c   : S1=0.015, ST=0.085
    Ma    : S1=0.000, ST=0.001  (essentially inert in the low-speed range Ma ≤ 0.4)

References
----------
1. Drela, M. (1989). XFOIL: An Analysis and Design System for Low Reynolds
   Number Airfoils. Low Reynolds Number Aerodynamics, Springer, pp.1-12.
   DOI: 10.1007/978-3-642-84010-4_1
2. Sharpe, P. D. & Hansman, R. J. (2025). NeuralFoil: An Airfoil Aerodynamics Analysis Tool Using Physics-Informed Machine Learning. arXiv:2503.16323.
   https://arxiv.org/abs/2503.16323
3. AeroSandbox Airfoil API documentation, ``get_aero_from_neuralfoil`` wrapper.
   https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/index.html
4. NASA Langley Turbulence Modeling Resource, NACA 0012 validation case.
   https://turbmodels.larc.nasa.gov/naca0012_val.html
5. Sheldahl, R. E. & Klimas, P. C. (1981). Aerodynamic Characteristics of
   Seven Symmetrical Airfoil Sections Through 180-Degree Angle of Attack,
   SAND80-2114. https://www.osti.gov/servlets/purl/6548367
6. XFOIL online documentation, convergence and separated-flow operating
   limits for panel/integral-boundary-layer calculations.
   https://web.mit.edu/drela/Public/web/xfoil/

"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class AirfoilStall2D(BaseCase):
    """Low-speed airfoil lift-drag polar (with stall region).

    4-D input → 3-D output (C_L, C_D, L/D).

    The benchmark solver requires NeuralFoil.  The analytical thin-airfoil
    approximation is retained as a diagnostic helper but is not used by the
    public ``evaluate`` path, so a missing or failing backend is a launch
    error instead of a silent response-surface change.
    """

    meta = CaseMeta(
        case_id="airfoil_stall_2d",
        name="Low-speed airfoil lift-drag polar (with stall)",
        version="1.0.0",
        domain="aerodynamics",
        tags=["airfoil", "low_speed", "stall", "threshold", "naca", "boundary_layer"],
        dim_input=4,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="alpha",
                symbol=r"\alpha",
                description="Angle of attack",
                unit="deg",
                lower=-5.0,
                upper=20.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="Re",
                symbol="Re",
                description="Chord-based Reynolds number",
                unit="-",
                lower=5e4,
                upper=5e6,
                log_scale=True,  # spans two orders of magnitude, log-normalised
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="Ma",
                symbol="Ma",
                description="Freestream Mach number (low-subsonic/mild-compressibility stressor)",
                unit="-",
                lower=0.0,
                upper=0.40,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="t_c",
                symbol="t/c",
                description="Maximum thickness-to-chord ratio",
                unit="-",
                lower=0.06,
                upper=0.18,
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
                description="Lift coefficient",
                unit="-",
                is_primary=True,
                expected_range=(-0.8, 2.5),
            ),
            OutputVariable(
                name="CD",
                symbol="C_D",
                description="Drag coefficient (can reach 0.3+ in deep stall / thin airfoil at low Re)",
                unit="-",
                is_primary=False,
                # Low Re + high AoA / thick airfoil stall region CD >= 0.2; upper bound widened to 0.6
                expected_range=(0.003, 0.6),
                failure_threshold=0.05,
                failure_direction="above",  # high-drag deep-stall / separated-flow region
            ),
            OutputVariable(
                name="LoD",
                symbol="L/D",
                description="Lift-to-drag ratio (may go negative post-stall as C_L reverses)",
                unit="-",
                is_primary=False,
                # In deep stall (α>α_stall) C_L reverses, L/D can drop to -25
                expected_range=(-90.0, 180.0),
            ),
        ],
        governing_equation=(
            "NeuralFoil baseline airfoil polar "
            "+ explicit low-subsonic Prandtl-Glauert scalar correction"
        ),
        pde_type="integral",
        spatial_dim=2,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=True,
        physics_references=[
            "Drela, M. (1989). XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils. "
            "Low Reynolds Number Aerodynamics, Springer, DOI:10.1007/978-3-642-84010-4_1",
            "Sharpe, P. D. & Hansman, R. J. (2025). NeuralFoil: An Airfoil Aerodynamics Analysis "
            "Tool Using Physics-Informed Machine Learning. arXiv:2503.16323, "
            "https://arxiv.org/abs/2503.16323",
            "AeroSandbox Airfoil API documentation, get_aero_from_neuralfoil wrapper, "
            "https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/index.html",
            "NASA TM-4074, Effects of Independent Variation of Mach and Reynolds Numbers on the "
            "Low-Speed Aerodynamic Characteristics of the NACA 0012 Airfoil Section, "
            "https://ntrs.nasa.gov/citations/19880019495",
            "NASA Langley Turbulence Modeling Resource, 2D NACA 0012 Airfoil Validation, "
            "https://turbmodels.larc.nasa.gov/naca0012_val.html",
            "NASA Turbulence Modeling Resource, transition/turbulence validation context, "
            "https://www.nasa.gov/nasa-turbulence-modeling-resource/",
            "NASA Glenn Beginner's Guide to Aeronautics, inclination effects on lift and drag, "
            "https://www.grc.nasa.gov/WWW/K-12/VirtualAero/BottleRocket/airplane/incline.html "
            "and https://www.grc.nasa.gov/WWW/k-12/VirtualAero/BottleRocket/airplane/inclind.html",
            "Sheldahl, R. E. & Klimas, P. C. (1981). Aerodynamic Characteristics of "
            "Seven Symmetrical Airfoil Sections Through 180-Degree Angle of Attack, "
            "SAND80-2114. https://www.osti.gov/servlets/purl/6548367",
            "XFOIL online documentation, convergence and separated-flow operating "
            "limits for panel/integral-boundary-layer calculations, "
            "https://web.mit.edu/drela/Public/web/xfoil/",
        ],
        solver_library="neuralfoil",
        solver_details=(
            "Low-level neuralfoil.get_aero_from_airfoil 'large' model; airfoil: "
            "NACA 4-digit symmetric family. Mach dependence is imposed outside "
            "NeuralFoil by multiplying CL and CD by 1/sqrt(1-Ma^2); this low-order "
            "correction is a compressibility stress factor, not Mach-resolved "
            "airfoil validation. AeroSandbox's Airfoil.get_aero_from_neuralfoil "
            "wrapper exposes a mach argument in current documentation, but the "
            "formal response surface is pinned to the low-level NeuralFoil call "
            "plus explicit PG scaling; changing that backend is a response-surface "
            "change requiring protocol review and full resweep. Named "
            "_validate_physical_inputs private physical-input "
            "evaluation rejects values outside the declared low-speed airfoil domain "
            "before constructing the NACA section or calling NeuralFoil. "
            "The analytical thin-airfoil + stall model is diagnostic-only and is not used "
            "by the public evaluate path; missing neuralfoil/aerosandbox or per-point "
            "backend failures are launch/runtime errors because the two backends differ "
            "significantly in the stall region (α > 12°). Public NASA TM-4074 "
            "and NASA TMR anchors are single-section NACA 0012 context and "
            "validation-dataset boundaries, not calibration of the benchmark's "
            "full NACA00xx thickness sweep, explicit CD/beta scaling, or low-level "
            "NeuralFoil backend. NASA Glenn inclination/stall pages are general "
            "boundary evidence for near-linear small-angle lift, separated-flow "
            "stall, high-angle drag rise, and experimental stall-point sensitivity; "
            "they are not calibration evidence for this NeuralFoil response surface. "
            "No cambered airfoils, "
            "flaps/control surfaces, roughness/trip/transition-control sweep, "
            "turbulence-model selection, RANS/LES grid-convergence validation, "
            "dynamic-stall hysteresis, unsteady pitching/plunging motion, 3-D "
            "finite-wing effect, rotating-blade correction, icing/contamination, "
            "aeroelastic coupling, or acoustic model is included."
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 15.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_unimodal", "threshold_jump", "heterogeneous"],
        effective_dim=4,
        important_variables=["alpha", "Re", "t_c", "Ma"],
        interaction_pairs=[("alpha", "Re"), ("alpha", "t_c")],
        has_failure_boundary=True,
        failure_condition="C_D > 0.05 (high-drag separated-flow / deep-stall region)",
        sobol_S1={"alpha": 0.651, "Re": 0.101, "Ma": 0.000, "t_c": 0.015},
        sobol_ST={"alpha": 0.888, "Re": 0.306, "Ma": 0.001, "t_c": 0.085},
        sobol_computation="salib_saltelli_n2048_seed42_CL_output",
        applicable_scenarios=["VS-A", "VS-C", "VS-D", "VS-F", "VS-G"],
        scenario_notes={
            "VS-A": "Stall region (α≈12°–18°) concentrates steep gradients; strong spatial "
            "heterogeneity amplifies deficiencies of poorly uniform designs.",
            "VS-C": "Local nonlinearity of C_L (sharp post-stall drop) challenges surrogate "
            "accuracy; recommend N=60–100; GP typically overestimates uncertainty in stall zone.",
            "VS-D": "Classic small-sample 4-D case: recommend N=30–80 convergence curves; "
            "stall region is easily missed by small-sample methods.",
            "VS-F": "Clear importance hierarchy (alpha >> Re > t_c > Ma); "
            "Morris/Sobol screening should rank correctly with N ≤ 30.",
            "VS-G": "Stall boundary α_stall(Re, t_c) is a 3-D implicit surface; "
            "use this case to test whether sequential methods locate the high-drag boundary "
            "more efficiently than space-filling designs.",
        },
        recommended_n_train={"default": 80},
        n_saturation=300,
        noise_factor_candidates=["Re", "Ma"],
        control_factor_candidates=["alpha", "t_c"],
        reference_solution_available=True,
        reference_grid_size=10000,
    )

    # ── Core simulation ───────────────────────────────────────────────────────────

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "airfoil_stall_2d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        alpha = X_phys[:, 0]
        Re = X_phys[:, 1]
        Ma = X_phys[:, 2]
        t_c = X_phys[:, 3]
        tol = 1e-10
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(alpha < -5.0 - tol)
            or np.any(alpha > 20.0 + tol)
            or np.any(Re < 5e4 * (1.0 - tol))
            or np.any(Re > 5e6 * (1.0 + tol))
            or np.any(Ma < 0.0 - tol)
            or np.any(Ma > 0.40 + tol)
            or np.any(t_c < 0.06 - tol)
            or np.any(t_c > 0.18 + tol)
        ):
            raise ValueError(
                "airfoil_stall_2d physical inputs must stay inside the declared "
                "alpha/Re/Ma/t_c benchmark domain before NeuralFoil evaluation"
            )
        return X_phys

    def _evaluate_core(
        self,
        X_phys: np.ndarray,
        fidelity: str = "default",
        **kwargs,
    ) -> np.ndarray:
        """Evaluate with the formal NeuralFoil backend.

        Parameters
        ----------
        X_phys : ndarray, shape (n, 4)
            Columns: [alpha(deg), Re(-), Ma(-), t_c(-)]
        """
        return self._eval_neuralfoil(X_phys)

    def _eval_neuralfoil(self, X_phys: np.ndarray) -> np.ndarray:
        """NeuralFoil backend (preferred; sub-ms, engineering accuracy)."""
        X_phys = self._validate_physical_inputs(X_phys)

        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)

        alpha = X_phys[:, 0]  # deg
        Re = X_phys[:, 1]
        Ma = X_phys[:, 2]
        t_c = X_phys[:, 3]

        import neuralfoil as nf
        import aerosandbox as asb

        for i in range(n):
            try:
                t_digits = int(round(float(t_c[i]) * 100))
                af = asb.Airfoil(f"naca00{t_digits:02d}")
                beta_pg = np.sqrt(max(1.0 - Ma[i] ** 2, 0.01))
                aero = nf.get_aero_from_airfoil(
                    airfoil=af,
                    alpha=float(alpha[i]),
                    Re=float(Re[i]),
                    model_size="large",
                )
                CL = float(np.squeeze(aero["CL"])) / beta_pg
                CD = float(np.squeeze(aero["CD"])) / beta_pg
                if not np.isfinite(CL) or not np.isfinite(CD) or CD <= 0.0:
                    raise ValueError(
                        f"non-finite or non-positive NeuralFoil polar: CL={CL}, CD={CD}"
                    )
                LoD = CL / CD if CD > 1e-8 else 0.0
                if not np.isfinite(LoD):
                    raise ValueError(f"non-finite lift-to-drag ratio: LoD={LoD}")
                Y[i] = [CL, CD, LoD]
            except Exception as exc:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "airfoil_stall_2d NeuralFoil backend failed for physical input "
                    f"{x_text}; refusing analytical fallback because it would change "
                    "the formal benchmark response surface."
                ) from exc
        return Y

    def _eval_analytical(self, X_phys: np.ndarray) -> np.ndarray:
        """Diagnostic analytical approximation backend.

        Model components:
        1. Thin-airfoil lift-slope: C_L0 = 2π · α(rad) / β
           where β = √(1 − Ma²) is the Prandtl–Glauert compressibility factor
        2. Reynolds correction: lift slope reduced at low Re
           f_Re = 1 − exp(−Re / 2e5)
        3. Stall model: for α > α_stall a Gaussian window simulates post-stall lift drop
           α_stall = 13 + (t/c − 0.12)*30 + log10(Re/1e6)*2.5 (deg)
        4. Zero-lift angle: assumes symmetric airfoil (NACA 00XX), C_L0 from pure AoA
        5. Drag: parabolic polar C_D = C_D0 + C_L²/(π·e·AR), AR≈6 (reference wing)
           C_D0 depends on thickness and Re
        """
        alpha = X_phys[:, 0]  # deg
        Re = X_phys[:, 1]
        Ma = X_phys[:, 2]
        t_c = X_phys[:, 3]

        alpha_rad = np.radians(alpha)
        beta = np.sqrt(np.maximum(1.0 - Ma**2, 0.01))

        # ── Lift slope ─────────────────────────────────────────────────────────────
        CL_alpha = 2.0 * np.pi / beta  # theoretical lift slope /rad
        f_Re = 1.0 - np.exp(-Re / 2e5)  # low-Re correction
        CL_linear = CL_alpha * alpha_rad * f_Re

        # ── Stall angle (varies with Re and thickness) ───────────────────────────
        alpha_stall = (
            13.0
            + (t_c - 0.12) * 30.0  # thicker airfoil stalls later
            + np.log10(np.maximum(Re, 1e4) / 1e6) * 2.5  # higher Re stalls later
        )  # deg

        # Post-stall lift drop: Gaussian decay window
        overshoot = np.maximum(alpha - alpha_stall, 0.0)
        stall_factor = np.exp(-0.3 * overshoot**2)
        CL = CL_linear * stall_factor

        # ── Drag (parabolic polar) ─────────────────────────────────────────────────
        CD0 = (
            0.006
            + 0.025 * (t_c - 0.12) ** 2  # thickness term
            + 0.003 / (np.log10(np.maximum(Re, 1e4)) - 3.0 + 1e-3)  # Re friction term
            + 0.004 * Ma**2 / beta  # wave drag
        )
        AR_eff = 6.0  # effective aspect ratio (planar wing ref)
        e = 0.85  # Oswald efficiency
        CL_i = CL**2 / (np.pi * e * AR_eff)  # induced drag
        # Post-stall drag surge
        post_stall_CD = 0.15 * (1.0 - stall_factor) * np.maximum(alpha - alpha_stall, 0) ** 0.5
        CD = CD0 + CL_i + post_stall_CD
        CD = np.maximum(CD, 0.003)

        LoD = np.where(CD > 1e-8, CL / CD, 0.0)
        return np.stack([CL, CD, LoD], axis=1)
