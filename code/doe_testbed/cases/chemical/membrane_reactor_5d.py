"""
Case C04: Membrane reactor concentration polarisation (diffusion–reaction 1-D steady state)
============================================================================================

Engineering background
----------------------
Membrane reactors combine chemical reaction with membrane separation and are
widely used for industrial dehydrogenation (propane → propylene), methane
reforming (hydrogen production), etc.  Selective permeation through the
membrane continuously removes product, breaking the thermodynamic equilibrium
limit.

Key challenge: **concentration polarisation**
  Reactant is consumed at the membrane surface, lowering the local
  concentration below the bulk value and creating a concentration-gradient
  layer (diffusion boundary layer) that reduces the effective permeation
  driving force.

This case simulates the steady-state concentration distribution inside a
tubular membrane reactor, considering a first-order reversible reaction
A ⇌ B and selective permeation of species B through the membrane.

Governing equations (1-D convection–reaction–permeation, Pe → ∞ limit)
----------------------------------------------------------------------
The present implementation uses the plug-flow (pure-convection) limit of
the full diffusion–reaction–permeation equations.  Axial dispersion is
neglected (i.e. Pe → ∞) so the dimensionless concentration balance
reduces to a linear first-order ODE system:

  dC_A/dz = −Da·C_A + (Da/K_eq)·C_B        (reaction, net forward)
  dC_B/dz = +Da·C_A − (Da/K_eq)·C_B − Da_m·C_B  (reaction − permeation)
  with    z ∈ [0, 1],  C_A(0) = 1,  C_B(0) = C_B0_frac.

Boundary conditions: z = 0: C_A = 1 (normalised), C_B = C_B0_frac.

Input variables
---------------
Da        Damköhler number (reaction/convection)         [0.1, 10]   -   Reaction intensity
Da_m      Membrane Damköhler number (permeation/conv.)   [0.01, 5]   -   Permeation intensity
K_eq      Equilibrium constant                           [0.5, 20]   -   Thermodynamic limit
Pe        Peclet number (reserved for future extension)  [1, 100]    -   Currently unused; Pe → ∞ limit
C_B0_frac Inlet product / reactant concentration ratio   [0, 0.1]    -   Feed purity

.. note::
   ``Pe`` is declared so that the case's input dimensionality stays fixed
   at 5 for downstream DOE scripts, but it does not appear in the
   governing equations above.  Its Sobol index is therefore exactly zero
   — this is a **ground-truth** zero-effect variable, useful for
   validating variable-screening methods (VS-F).  A follow-up release
   may add axial dispersion  +(1/Pe)·d²C/dz²  via a two-point BVP solver,
   at which point ``Pe`` will become physically active.

Output responses
----------------
X_A_exit  Exit conversion                -          Primary output (exceeding equilibrium = success)
C_B_exit  Exit product concentration     mol/m³     Product yield indicator
eta_mem   Membrane utilisation efficiency -          Membrane performance indicator
            (actual permeation / no-membrane-profile ideal permeation)

DOE validation value
--------------------
VS-B (effect estimation):
    Da and Da_m exhibit strong interaction (fast reaction but slow membrane
    removal → conversion is limited).

References
----------
1. Sanchez Marcano, J. G. & Tsotsis, T. T. (2002). Catalytic Membranes and
   Membrane Reactors. Wiley-VCH.
2. Dittmeyer, R. & Caro, J. (2008). In: Handbook of Heterogeneous Catalysis, Wiley.
3. University of Michigan Elements of Chemical Reaction Engineering membrane
   reactor module. Public teaching anchor for product-removal/Le Chatelier
   trends, not calibrated membrane-reactor validation data.
   https://websites.umich.edu/~elements/web_mod/membrane/intro.htm
4. COMSOL Hydrodealkylation in a Membrane Reactor documentation. Public
   workflow-boundary reference for omitted property packages, Darcy-type
   cross-membrane flux, pressure coupling, and species-selective membrane
   balances, not validation of this compact plug-flow ODE.
   https://doc.comsol.com/6.4/doc/com.comsol.help.models.chem.membrane_hda/membrane_hda.html
"""

import numpy as np
from scipy.integrate import solve_ivp

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class MembraneReactor5D(BaseCase):
    """Tubular membrane reactor with concentration polarisation (5D → 3D, 1-D ODE)."""

    meta = CaseMeta(
        case_id="membrane_reactor_5d",
        name="Membrane reactor concentration polarisation (diffusion–reaction–permeation 1-D ODE)",
        version="1.0.0",
        domain="chemical_process",
        tags=[
            "membrane_reactor",
            "concentration_polarization",
            "ode",
            "separation",
            "5d",
            "reaction_engineering",
        ],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="Da",
                symbol="Da",
                description="Damköhler number (forward reaction intensity)",
                unit="-",
                lower=0.1,
                upper=10.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="Da_m",
                symbol="Da_m",
                description="Membrane Damköhler number (permeation intensity)",
                unit="-",
                lower=0.01,
                upper=5.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="K_eq",
                symbol="K_{eq}",
                description="Reaction equilibrium constant (K_eq = k_f/k_r)",
                unit="-",
                lower=0.5,
                upper=20.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="Pe",
                symbol="Pe",
                description=(
                    "Peclet number (convection/diffusion).  Currently reserved: "
                    "the solver uses the Pe → ∞ plug-flow limit, so Pe has zero "
                    "effect on the outputs (ground-truth inert variable for VS-F)."
                ),
                unit="-",
                lower=1.0,
                upper=100.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="C_B0_frac",
                symbol="C_{B0}/C_{A0}",
                description="Inlet product-to-reactant concentration ratio",
                unit="-",
                lower=0.0,
                upper=0.1,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="X_A_exit",
                symbol="X_{A,exit}",
                description="Exit conversion of reactant A",
                unit="-",
                is_primary=True,
                expected_range=(0.0, 1.0),
            ),
            OutputVariable(
                name="C_B_exit",
                symbol="C_{B,exit}",
                description="Exit concentration of product B (normalised)",
                unit="-",
                is_primary=False,
                expected_range=(0.0, 1.5),
            ),
            OutputVariable(
                name="eta_mem",
                symbol=r"\eta_{mem}",
                description=(
                    "Membrane permeation efficiency: actual integrated B removal divided by "
                    "the ideal removal from the no-membrane PFR concentration profile"
                ),
                unit="-",
                is_primary=False,
                expected_range=(0.0, 1.0),
            ),
        ],
        governing_equation=(
            "1-D tubular membrane reactor, plug-flow (Pe → ∞) limit:\n"
            "  dC_A/dz = −Da·C_A + (Da/K_eq)·C_B\n"
            "  dC_B/dz = Da·C_A − (Da/K_eq)·C_B − Da_m·C_B\n"
            "  dQ_B/dz = Da_m·C_B,  eta_mem = Q_B / Q_B,no-membrane-profile\n"
            "  Normalised: C_A(0) = 1, C_B(0) = C_B0_frac.\n"
            "  Note: Pe is declared as a 5th input for dimensionality but does\n"
            "  not enter the equations in v1.0.0 (see class docstring)."
        ),
        pde_type="ode_system",
        spatial_dim=1,
        is_steady=True,
        # The concentration ODE is linear for fixed Da/Da_m/K_eq, but the
        # benchmark response map is nonlinear in those inputs through the
        # coefficient matrix, equilibrium ratio, and eta_mem normalisation.
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Sanchez Marcano, J. G. & Tsotsis, T. T. (2002). Catalytic Membranes. Wiley-VCH.",
            "University of Michigan Elements of Chemical Reaction Engineering, "
            "membrane reactor module. Public teaching anchor for product "
            "removal and Le Chatelier equilibrium-shift trends, not calibrated "
            "membrane-reactor validation data. "
            "https://websites.umich.edu/~elements/web_mod/membrane/intro.htm",
            "COMSOL Hydrodealkylation in a Membrane Reactor documentation. "
            "Public boundary reference for omitted thermodynamic/property "
            "packages, Darcy-type pressure-driven membrane flux, shell/reactor "
            "pressure coupling, species-selective membrane balances, and "
            "industrial membrane-reactor workflow scope; not calibration data "
            "for this compact plug-flow ODE surrogate. "
            "https://doc.comsol.com/6.4/doc/com.comsol.help.models.chem.membrane_hda/membrane_hda.html",
        ],
        solver_library="scipy.integrate.solve_ivp",
        solver_details=(
            "RK45, rtol=1e-8, linear ODE system with integrated permeation "
            "normalised by a no-membrane PFR baseline. Private physical-input "
            "calls are rejected before the ODE solver if they are malformed, "
            "non-finite, or outside the declared benchmark domain. The v1 "
            "surrogate intentionally omits finite-Pe axial dispersion, "
            "mass-transfer-film/concentration-polarisation correlations, "
            "membrane selectivity or material calibration, pressure-driven "
            "Darcy flux, shell-side pressure/composition coupling, "
            "thermodynamic/property packages, deactivation, fouling/aging, "
            "and industrial membrane-reactor workflow controls."
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 2.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_monotone", "strong_interaction"],
        effective_dim=3,
        important_variables=["Da", "Da_m", "K_eq"],
        interaction_pairs=[("Da", "Da_m"), ("Da", "K_eq")],
        has_failure_boundary=False,
        sobol_S1={"Da": 0.809, "Da_m": 0.014, "K_eq": 0.097, "Pe": 0.000, "C_B0_frac": 0.000},
        sobol_ST={"Da": 0.879, "Da_m": 0.035, "K_eq": 0.160, "Pe": 0.000, "C_B0_frac": 0.001},
        sobol_computation="salib_saltelli_n2048_seed42_X_A_exit_output",
        applicable_scenarios=["VS-B", "VS-C", "VS-F"],
        scenario_notes={
            "VS-B": "Da and Da_m exhibit strong interaction (fast-reaction / slow-permeation bottleneck).",
            "VS-F": "Pe is a reserved ground-truth inert variable (Pe → ∞ plug-flow limit); "
            "screening methods (Morris / Sobol) must return an essentially zero index for Pe.",
        },
        recommended_n_train={"default": 50},
        n_saturation=200,
        noise_factor_candidates=["Pe", "C_B0_frac"],
        control_factor_candidates=["Da", "Da_m", "K_eq"],
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> None:
        """Reject malformed or out-of-domain private physical inputs."""
        X = np.asarray(X_phys, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.meta.dim_input:
            raise ValueError(
                "membrane_reactor_5d expects a physical input array with "
                f"shape (n, {self.meta.dim_input}); got {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "membrane_reactor_5d received non-finite physical inputs; "
                "refusing to extrapolate the plug-flow membrane-reactor ODE."
            )

        tol = 1e-10
        for j, var in enumerate(self.meta.input_variables):
            lower = float(var.lower)
            upper = float(var.upper)
            scale = max(abs(lower), abs(upper), 1.0)
            bad = (X[:, j] < lower - tol * scale) | (X[:, j] > upper + tol * scale)
            if np.any(bad):
                raise ValueError(
                    "membrane_reactor_5d physical input "
                    f"{var.name!r} is outside the declared benchmark domain "
                    f"[{lower}, {upper}]."
                )

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        self._validate_physical_inputs(X_phys)
        X_phys = np.asarray(X_phys, dtype=float)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                Y[i] = self._reactor(
                    Da=float(X_phys[i, 0]),
                    Da_m=float(X_phys[i, 1]),
                    K_eq=float(X_phys[i, 2]),
                    Pe=float(X_phys[i, 3]),
                    C_B0=float(X_phys[i, 4]),
                )
            except Exception as exc:
                raise RuntimeError(
                    "membrane_reactor_5d refusing silent NaN after reactor "
                    f"ODE evaluation failure for Da={X_phys[i, 0]!r}, "
                    f"Da_m={X_phys[i, 1]!r}, K_eq={X_phys[i, 2]!r}, "
                    f"Pe={X_phys[i, 3]!r}, C_B0_frac={X_phys[i, 4]!r}"
                ) from exc
        if (
            Y.shape != (n, self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y[:, 0] < 0.0)
            or np.any(Y[:, 0] > 1.0)
            or np.any(Y[:, 1] < 0.0)
            or np.any(Y[:, 2] < 0.0)
            or np.any(Y[:, 2] > 1.0)
        ):
            raise RuntimeError(
                "membrane_reactor_5d produced a malformed, non-finite, or "
                "non-physical membrane-reactor response; refusing silent "
                "archive rows"
            )
        return Y

    @staticmethod
    def _reactor(Da, Da_m, K_eq, Pe, C_B0):
        if (
            not np.all(np.isfinite([Da, Da_m, K_eq, Pe, C_B0]))
            or Da <= 0.0
            or Da_m < 0.0
            or K_eq <= 0.0
            or Pe <= 0.0
            or C_B0 < 0.0
        ):
            raise ValueError(
                "invalid membrane_reactor_5d input outside the declared physical contract"
            )
        k_r_ratio = 1.0 / K_eq

        def rhs_with_permeation(z, y, membrane_removal):
            C_A, C_B, Q_B = y
            r_fwd = Da * C_A
            r_rev = Da * k_r_ratio * C_B
            J_B = Da_m * C_B
            dCA = -r_fwd + r_rev
            dCB = r_fwd - r_rev - (J_B if membrane_removal else 0.0)
            return [dCA, dCB, J_B]

        sol = solve_ivp(
            lambda z, y: rhs_with_permeation(z, y, membrane_removal=True),
            [0, 1],
            [1.0, C_B0, 0.0],
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            dense_output=False,
            t_eval=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        if not sol.success:
            raise RuntimeError(f"membrane reactor ODE solve failed: {sol.message}")
        if not np.all(np.isfinite(sol.y)):
            raise RuntimeError("non-finite membrane reactor ODE trajectory")

        C_A_exit = float(sol.y[0, -1])
        C_B_exit = float(sol.y[1, -1])
        X_A = 1.0 - C_A_exit
        ideal_flux = solve_ivp(
            lambda z, y: rhs_with_permeation(z, y, membrane_removal=False),
            [0, 1],
            [1.0, C_B0, 0.0],
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            t_eval=[1.0],
        )
        if not ideal_flux.success:
            raise RuntimeError(
                f"no-membrane baseline ODE solve failed: {ideal_flux.message}"
            )
        if not np.all(np.isfinite(ideal_flux.y)):
            raise RuntimeError("non-finite no-membrane baseline ODE trajectory")
        if C_A_exit < 0.0 or C_B_exit < 0.0 or X_A < 0.0 or X_A > 1.0:
            raise RuntimeError("non-physical membrane reactor concentration response")
        Q_actual = float(sol.y[2, -1])
        Q_ideal = float(ideal_flux.y[2, -1])
        if Q_actual < 0.0 or Q_ideal <= 0.0:
            raise RuntimeError("non-physical membrane reactor permeation integral")
        eta_mem = float(Q_actual / Q_ideal)
        if not np.isfinite(eta_mem) or eta_mem < 0.0 or eta_mem > 1.0:
            raise RuntimeError("non-physical membrane reactor utilisation ratio")

        return np.array([X_A, C_B_exit, eta_mem])
