"""
Case M02: Thin-walled cylindrical pressure-vessel safety factor
================================================================

Engineering background
----------------------
Pressure vessels are critical safety equipment in the chemical, energy,
and aerospace industries — e.g. gas storage tanks, boilers, aircraft
cabins, and rocket propellant tanks.  Failure (burst or leakage) can
have catastrophic consequences.

Engineering design typically requires checking strength, stability,
fabrication, inspection, and code-specific allowable-stress rules.  This
benchmark case intentionally isolates the closed-form strength part:
  (1) The maximum equivalent membrane stress must not exceed the material
      reference stress by the chosen safety-factor threshold.
  (2) Structural mass should be minimised while satisfying that strength
      margin.
Buckling, thick-wall Lame stress gradients, radial stress, end-cap and support
discontinuities, local nozzle/opening stresses, weld efficiency, corrosion
allowance, fatigue, fracture, fabrication, inspection, hydrostatic testing,
pressure-relief-device certification, external-pressure vessel rules, and full ASME code-compliance rules are outside this surrogate.  This
intentionally differs from ASME BPVC design/certification workflows, which
include construction, fabrication, inspection, testing, certification,
material allowables, weld-joint efficiency, corrosion allowance, openings,
material-class rules, fabrication methods, assembly, erection, examination,
and pressure-relief-device scopes.

This case establishes a multi-constraint optimisation problem with
an analytical **strength constraint**.  The “failure / safe” boundary is an
implicit hyper-surface in the 4-D parameter space, making it a typical test
scenario for sequential DOE methods (especially reliability analysis).

Engineering model
-----------------
Thin-walled closed cylindrical vessel under internal pressure P:

Basic parameters:
  R — inner radius
  t — wall thickness
  L — cylinder length
  P — design working pressure
  σ_y — material reference/yield stress used to define the safety margin

Stress calculation (thin-wall theory, t/R <= 0.1 conventional upper limit,
with the boundary treated as a benchmark-domain limit rather than a
high-fidelity pressure-vessel design guarantee):
  Hoop stress (principal stress 1): σ_θ = P·R / t
  Axial stress (principal stress 2): σ_a = P·R / (2t)
  Von Mises equivalent stress: σ_VM = √(σ_θ² − σ_θ·σ_a + σ_a²)
                                    = P·R/t · √3/2

Simplified design criterion:
  Strength safety factor: FS_strength = σ_y / σ_VM  (> 1.5 is safe)
  Mass-per-length objective: m/L = ρ·π·((R+t)²−R²)

This is a thin-wall membrane-stress surrogate for DOE/reliability testing,
not a claim of ASME BPVC Section VIII compliance.

Failure boundary:
  FS_strength = 1.5 (strength safety-factor threshold)

  This boundary is an implicit hyper-surface in (P, R, t, σ_y) space:
  P·R / (t·σ_y) = 2/(1.5·√3) ≈ 0.770

  This is a classic analytical-control setting for sequential reliability
  analysis: the benchmark tests whether a method can estimate P(FS < 1.5)
  efficiently relative to uniform sampling.

Input variables
---------------
P       Design internal pressure  [1.0, 20.0]   MPa   Load (natural noise factor)
R       Vessel inner radius       [0.5, 2.0]    m     Geometric design variable
t       Wall thickness            [0.005, 0.05] m     Geometric design variable (t/R <= 0.1 over the declared domain)
sigma_y Material reference stress [200, 600]    MPa   Material selection

Output responses
----------------
FS       Strength safety factor   -      FS = σ_y/σ_VM = σ_y·t/(P·R·√3/2)
W_per_L  Mass per unit length     kg/m   m/L = ρ·π·((R+t)²−R²), with 2πRtρ as the thin-wall approximation
sigma_VM Von Mises stress         MPa    Equivalent stress

DOE validation value
--------------------
VS-G (sequential / reliability):
    Failure boundary FS = 1.5 is an analytical implicit surface in 4-D.
    Sequential reliability methods (ALM, EGRA) should estimate the
    failure probability P_f = P(FS < 1.5) accurately with far fewer
    samples than uniform Monte Carlo.

VS-D (small-sample efficiency):
    4-D problem; the response surface is smooth and monotone (FS is
    monotone w.r.t. each variable); this is an intentionally simple
    analytical baseline for small-sample DOE checks.

VS-H (robust Taguchi design):
    P is the noise factor (in-service pressure fluctuates ±10 %);
    R, t, and σ_y are control/design factors (geometry plus material
    selection).
    Robust-design objective: choose geometry/material controls so that
    FS remains > 1.5 under P fluctuations.

Sensitivity metadata (FS output, uniform-distribution inputs)
-------------------------------------------------------------
FS = σ_y·t·2/(P·R·√3), so pressure and radius enter inversely while
wall thickness and material reference stress enter directly.  With the
current linear-uniform input ranges, the Sobol ranking is not the equal
log-space heuristic; the stored metadata uses a Saltelli computation for
the declared linear input distribution.

References
----------
1. ASME BPVC Section VIII, Division 1 product page. ASME.
   Design-code context only; this benchmark does not implement full BPVC
   design, fabrication, inspection, testing, certification, material, or
   code-stamp rules.
2. Shigley, J. E. et al. (2011). Mechanical Engineering Design, 9th ed.
   McGraw-Hill. ISBN 978-0-07-352928-8. Chapter 14.
3. Bichon, B. J. et al. (2008). Efficient Global Reliability Analysis for
   Nonlinear Implicit Performance Functions. AIAA J. 46(10):2459-2468.
   DOI: 10.2514/1.34321
4. Air Force Flight Dynamics Laboratory (1986). Stress Analysis Manual,
   Section 8.3.1, Simple Thin Pressure Vessels.
   https://engineeringlibrary.org/reference/simple-thin-pressure-vessels-air-force-stress-manual
5. NASA/TM-2015-218216. Thin pressure-vessel pipe formulation; gives
   sigma_theta = p r / t and sigma_L = p r / (2t), with radial stress
   neglected for thin-walled pipe.
   https://ntrs.nasa.gov/api/citations/20160000456/downloads/20160000456.pdf
6. ASME BPVC Section VIII, Division 1 product page.
   https://www.asme.org/codes-standards/find-codes-standards/bpvc-viii-1-bpvc-section-viii-rules-construction-pressure-vessels-division-1
7. ASME Boiler and Pressure Vessel Certification overview.
   https://www.asme.org/certification-accreditation/Boiler-and-Pressure-Vessel-Certification
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class PressureVessel4D(BaseCase):
    """Thin-walled cylindrical pressure-vessel safety factor (analytical).

    4-D input → 3-D output (safety factor, weight per unit length, Von Mises stress).

    The failure boundary (FS < 1.5) is an analytical hyper-surface in the
    parameter space — a standard test for sequential reliability methods.
    """

    meta = CaseMeta(
        case_id="pressure_vessel_4d",
        name="Thin-walled cylindrical pressure-vessel safety factor",
        version="1.0.0",
        domain="multidisciplinary",
        tags=[
            "pressure_vessel",
            "reliability",
            "analytical",
            "failure_boundary",
            "structural",
            "safety_factor",
            "taguchi",
        ],
        dim_input=4,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="P",
                symbol="P",
                description="Design working internal pressure",
                unit="MPa",
                lower=1.0,
                upper=20.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",  # P is a Taguchi noise factor (fluctuates in service)
            ),
            InputVariable(
                name="R",
                symbol="R",
                description="Vessel inner-wall radius",
                unit="m",
                lower=0.5,
                upper=2.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="t",
                symbol="t",
                description="Vessel wall thickness",
                unit="m",
                lower=0.005,
                upper=0.05,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="sigma_y",
                symbol=r"\sigma_y",
                description="Material reference/yield stress used to define the safety margin",
                unit="MPa",
                lower=200.0,
                upper=600.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",  # material selection
            ),
        ],
        output_variables=[
            OutputVariable(
                name="FS",
                symbol="FS",
                description="Strength safety factor (Von Mises criterion)",
                unit="-",
                is_primary=True,
                expected_range=(0.02, 70.0),
                failure_threshold=1.5,
                failure_direction="below",  # FS < 1.5 → failure
            ),
            OutputVariable(
                name="W_per_L",
                symbol="m/L",
                description="Vessel mass per unit length (lightweight indicator)",
                unit="kg/m",
                is_primary=False,
                # W/L = rho*pi*((R+t)^2-R^2), rho=7850 kg/m^3
                # R=2.0,t=0.05 -> ~5000 kg/m; R=0.5,t=0.005 -> ~124 kg/m
                expected_range=(0.0, 5000.0),
            ),
            OutputVariable(
                name="sigma_VM",
                symbol=r"\sigma_{VM}",
                description="Von Mises equivalent stress (thin-wall theory)",
                unit="MPa",
                is_primary=False,
                expected_range=(0.0, 7000.0),
                failure_threshold=None,
            ),
        ],
        governing_equation=(
            "Thin-walled cylindrical pressure-vessel membrane-stress surrogate:\n"
            "  Hoop stress: σ_θ = P·R / t\n"
            "  Axial stress: σ_a = P·R / (2t)\n"
            "  Von Mises: σ_VM = P·R/t · √3/2\n"
            "  Safety factor: FS = σ_y / σ_VM\n"
            "  Mass per unit length: m/L = ρ·π·((R+t)²−R²) ≈ 2πRt·ρ  [ρ = 7850 kg/m³]\n"
            "  Scope: strength-only thin-wall surrogate, not a full ASME BPVC, "
            "thick-wall, local-stress, fatigue/fracture, or buckling design check"
        ),
        pde_type="analytical",
        spatial_dim=2,
        is_steady=True,
        is_linear=False,
        physics_references=[
            "ASME BPVC Section VIII, Division 1 product page. "
            "Design-code context only; full BPVC design, fabrication, "
            "inspection, testing, certification, material, and code-stamp "
            "rules are not modelled, including external-pressure coverage, "
            "material-class requirements, fabrication-method rules, assembly, "
            "erection, examination, and "
            "nondestructive-examination / inspection acceptance appendices. "
            "https://www.asme.org/codes-standards/find-codes-standards/"
            "bpvc-viii-1-bpvc-section-viii-rules-construction-pressure-vessels-division-1",
            "ASME Boiler and Pressure Vessel Certification overview. "
            "Certification/quality-control-system context only; this analytical "
            "surrogate does not implement ASME authorization, inspections, "
            "Certification Mark workflows (for example U/UM/U2/U3 pressure-"
            "vessel scopes and UV/UD pressure-relief-device scopes), or code-stamp workflows. "
            "https://www.asme.org/certification-accreditation/"
            "Boiler-and-Pressure-Vessel-Certification",
            "Air Force Flight Dynamics Laboratory (1986). Stress Analysis Manual, "
            "Section 8.3.1 Simple Thin Pressure Vessels. "
            "https://engineeringlibrary.org/reference/simple-thin-pressure-vessels-air-force-stress-manual",
            "NASA/TM-2015-218216. Thin pressure-vessel pipe formulation "
            "with sigma_theta = p r / t and sigma_L = p r / (2t), radial "
            "stress neglected for thin-walled pipe. "
            "https://ntrs.nasa.gov/api/citations/20160000456/downloads/20160000456.pdf",
            "Shigley, J. E. et al. (2011). Mechanical Engineering Design, 9th ed. McGraw-Hill.",
            "Bichon, B. J. et al. (2008). Efficient Global Reliability Analysis. "
            "AIAA J. 46(10):2459-2468. DOI: 10.2514/1.34321",
        ],
        solver_library="numpy (analytical formulae)",
        solver_details=(
            "Direct strength-only thin-wall membrane-stress calculation; no "
            "thick-wall Lame/radial-stress gradient, end-cap/support "
            "discontinuity, nozzle/opening local-stress model, weld efficiency, "
            "corrosion allowance, fatigue, fracture, external-pressure buckling, "
            "hydrostatic-test, fabrication/inspection, material-class rule set, "
            "pressure-relief-device certification, nondestructive examination, "
            "inspection-acceptance appendices, or full ASME code check"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.01},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_monotone", "strong_interaction"],
        effective_dim=4,
        important_variables=["P", "R", "t", "sigma_y"],
        interaction_pairs=[
            ("P", "R"),
            ("P", "t"),
            ("P", "sigma_y"),
            ("R", "t"),
            ("R", "sigma_y"),
            ("t", "sigma_y"),
        ],
        has_failure_boundary=True,
        failure_condition=(
            "FS < 1.5 (equivalent to σ_VM > σ_y/1.5, " "i.e. P·R/t > 2·σ_y/(1.5·√3))"
        ),
        # Sobol indices (FS output, uniform-distribution inputs, SALib Saltelli n=4096)
        # P dominates FS (widest range: [1, 20], i.e. 20×); σ_y contributes least
        # Note: if inputs were log-uniform, all four indices would be equal (~0.25); this case uses linear normalisation
        sobol_S1={"P": 0.352, "R": 0.189, "t": 0.073, "sigma_y": 0.023},
        sobol_ST={"P": 0.692, "R": 0.468, "t": 0.236, "sigma_y": 0.101},
        sobol_computation="salib_saltelli_n4096",
        applicable_scenarios=["VS-D", "VS-G", "VS-H"],
        scenario_notes={
            "VS-D": "4-D smooth monotone problem; intentionally simple analytical baseline "
            "for small-sample efficiency checks before moving to harder cases.",
            "VS-G": "Failure boundary FS = 1.5 is an analytical hyper-surface; "
            "standard test for sequential reliability methods (ALM / EGRA):\n"
            "the experiment tests whether P_f = P(FS < 1.5) is estimated "
            "efficiently relative to the preregistered MCS reference.",
            "VS-H": "P is a Taguchi noise factor (±10 % fluctuation); "
            "R, t, and sigma_y are control/design factors; "
            "robust-design objective: ensure FS > 1.5 under P fluctuations.",
        },
        recommended_n_train={"default": 20},
        n_saturation=80,
        noise_factor_candidates=["P"],
        control_factor_candidates=["R", "t", "sigma_y"],
        reference_solution_available=True,
        reference_grid_size=10000,
    )

    _RHO_STEEL = 7850.0  # kg/m³, steel density
    _SQRT3_2 = np.sqrt(3.0) / 2.0  # Von Mises coefficient

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> None:
        """Reject malformed or out-of-domain private physical inputs."""
        X = np.asarray(X_phys, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.meta.dim_input:
            raise ValueError(
                "pressure_vessel_4d expects a physical input array with "
                f"shape (n, {self.meta.dim_input}); got {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "pressure_vessel_4d received non-finite physical inputs; "
                "refusing to extrapolate the thin-wall strength surrogate."
            )

        tol = 1e-10
        for j, var in enumerate(self.meta.input_variables):
            lower = float(var.lower)
            upper = float(var.upper)
            scale = max(abs(lower), abs(upper), 1.0)
            below = X[:, j] < lower - tol * scale
            above = X[:, j] > upper + tol * scale
            if np.any(below | above):
                raise ValueError(
                    "pressure_vessel_4d physical input "
                    f"{var.name!r} is outside the declared benchmark domain "
                    f"[{lower}, {upper}]."
                )

    def _evaluate_core(
        self,
        X_phys: np.ndarray,
        fidelity: str = "default",
        **kwargs,
    ) -> np.ndarray:
        """Thin-walled pressure-vessel analytical calculation.

        Parameters
        ----------
        X_phys : ndarray, shape (n, 4)
            Column order: [P(MPa), R(m), t(m), sigma_y(MPa)]

        Returns
        -------
        Y : ndarray, shape (n, 3)
            Column order: [FS(-), W_per_L(kg/m), sigma_VM(MPa)]
        """
        self._validate_physical_inputs(X_phys)
        X_phys = np.asarray(X_phys, dtype=float)

        P = X_phys[:, 0]  # MPa
        R = X_phys[:, 1]  # m
        t = X_phys[:, 2]  # m
        sigma_y = X_phys[:, 3]  # MPa

        # Von Mises stress (thin-wall theory, MPa)
        # σ_VM = P·R/t · √3/2
        sigma_VM = P * R / t * self._SQRT3_2

        # Safety factor
        FS = sigma_y / sigma_VM

        # Mass per unit length: m/L = ρ·π·((R+t)²−R²) ≈ 2πRtρ  [kg/m]
        W_per_L = self._RHO_STEEL * np.pi * ((R + t) ** 2 - R**2)

        Y = np.stack([FS, W_per_L, sigma_VM], axis=1)
        if Y.shape != (X_phys.shape[0], self.meta.dim_output) or not np.all(np.isfinite(Y)):
            raise RuntimeError(
                "pressure_vessel_4d analytical strength calculation produced "
                "an invalid response array; refusing silent non-finite or "
                "malformed outputs in the formal benchmark archive."
            )
        if not np.all((FS > 0.0) & (W_per_L > 0.0) & (sigma_VM > 0.0)):
            raise RuntimeError(
                "pressure_vessel_4d analytical strength calculation produced "
                "a non-physical response array; refusing silent non-positive "
                "FS, mass-per-length, or von-Mises stress outputs in the "
                "formal benchmark archive."
            )
        return Y
