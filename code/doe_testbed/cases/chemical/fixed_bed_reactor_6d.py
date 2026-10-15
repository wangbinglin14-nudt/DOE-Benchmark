"""
Case C01: Adiabatic fixed-bed catalytic hydrogenation reactor
==============================================================

Engineering background
----------------------
Fixed-bed catalytic reactors are core equipment in the chemical industry,
widely used for ammonia synthesis, methanol synthesis, hydrotreating
(desulphurisation, denitrogenation), Fischer–Tropsch synthesis, etc.

This case simulates an **adiabatic** fixed-bed catalytic hydrogenation
reactor (1-D steady state):
  · One-dimensional tubular reactor with axial flow
  · Single irreversible reaction: A + H₂ → B (exothermic, first-order)
  · Adiabatic operation: no side-wall heat exchange; all released heat
    raises the fluid temperature

Typical industrial application: jet-fuel hydrodesulphurisation (converting
thiophene to H₂S + hydrocarbons).  Operating conditions: T_in ≈ 330–380 °C,
P ≈ 3–7 MPa, WHSV ≈ 1–5 h⁻¹.

DOE research significance
-------------------------
This reactor model contains several classic DOE research challenges:

1. **Nonlinear coupling** (VS-B):
   Feed temperature T_in and activation energy Ea are strongly coupled —
   at low T_in (< 600 K) the reaction rate is extremely temperature-
   sensitive (Arrhenius exponential), while at high T_in it approaches
   equilibrium conversion and T_in's influence saturates.
   This coupling can mislead main-effect analysis (second-order models
   may be insufficient).

2. **Adiabatic temperature-rise constraint** (VS-G):
   In this irreversible adiabatic model the axial temperature is monotone
   increasing, so the peak bed temperature occurs at the outlet
   (T_max ≡ T_outlet).  One sequential-method objective is to locate the
   constrained operating boundary where conversion is high but the outlet
   temperature remains below the catalyst limit.

3. **Operating vs catalyst parameter screening** (VS-F):
   Among 6 parameters, GHSV and T_in have the largest effect on
   conversion; bed length L affects residence time but barely affects
   final conversion — analogous to the real-engineering "important-
   parameter identification" problem.

Governing equations
-------------------
1-D steady-state adiabatic fixed-bed ODE system (catalyst pellet-scale
effects neglected):

  Species balance (dimensionless concentration θ = C_A / C_A0):
    dθ/dz = −Da · θ · exp(γ(1 − 1/T̃))           [1]

  Energy balance (dimensionless temperature T̃ = T / T_in):
    dT̃/dz = β · Da · θ · exp(γ(1 − 1/T̃))        [2]

  Initial conditions: θ(0) = 1, T̃(0) = 1

Dimensionless groups:
  Da = k₀·exp(−Ea/R·T_in) · τ      (Damköhler number: reaction rate / convection rate)
  β = (−ΔH_r) · C_A0 / (ρ·C_p·T_in) (adiabatic temperature-rise number)
  γ = Ea / (R·T_in)                 (Arrhenius dimensionless activation energy)

Fixed physical parameters:
  R  = 8.314 J/(mol·K)
  ρ  = 800  kg/m³ (catalyst bed)
  C_p = 2000 J/(kg·K)

Input variables
---------------
T_in      Feed temperature       [550, 700]  K          Primary control; exponentially affects rate
C_A0      Feed concentration     [0.5, 5.0]  mol/m³     Linearly affects adiabatic temperature rise
Ea        Activation energy      [50, 150]   kJ/mol     Determines temperature sensitivity (strong interaction with T_in)
tau       Residence time         [1.0, 20.0] s          Equivalent to L/u; controls conversion
dHr       Reaction enthalpy      [−200, −50] kJ/mol     Determines adiabatic temperature rise (negative = exothermic)
k0_log    Log₁₀ pre-exponential  [6.0, 12.0] -          Arrhenius pre-exponential factor stressor

Output responses
----------------
X_A       Exit conversion        -          Primary output; 0 = no reaction, 1 = complete
T_max     Bed peak temperature   K          Outlet peak temperature in this adiabatic irreversible reactor
T_outlet  Exit temperature       K          Adiabatic rise = T_outlet − T_in

DOE validation value
--------------------
VS-B (effect estimation):
    T_in and Ea are strongly nonlinearly coupled (γ = Ea/R/T_in is the
    product of two variables).  CCD or Box–Behnken should capture the
    second-order interaction; resolution-III designs (ignoring interactions)
    fail severely on this problem.

VS-C (surrogate accuracy):
    The conversion curve has a clear "S-shape" near Da ∼ 1 (low-to-high
    conversion transition); GP or RBF accuracy is challenged in this
    transition region.

VS-G (sequential optimisation / modelling):
    Objective: locate the feasible-domain boundary where X_A > 0.95 and
    T_max < 750 K.  Sequential methods should be more efficient than
    uniform designs in finding this constrained feasible region.

References
----------
1. Levenspiel, O. (1999). Chemical Reaction Engineering, 3rd ed. Wiley.
   ISBN 978-0-471-25424-9.
2. Froment, G. F. & Bischoff, K. B. (1990). Chemical Reactor Analysis
   and Design, 2nd ed. Wiley. ISBN 978-0-471-51040-0.
3. Walas, S. M. (1991). Reaction Kinetics for Chemical Engineers.
   Butterworth-Heinemann.
4. University of Michigan Elements of Chemical Reaction Engineering,
   Adiabatic energy-balance learning module. Public teaching anchor for
   adiabatic PFR energy-balance trends, not case-level calibration data.
   https://websites.umich.edu/~elements/asyLearn/bits/ebforadiabatic/index.htm
5. COMSOL Packed Bed Reactor documentation. Public packed-bed workflow
   boundary reference for omitted multigeometry species transport, pellet
   diffusion, and pressure-drop modelling, not validation of this compact ODE.
   https://doc.comsol.com/6.0/doc/com.comsol.help.models.chem.packed_bed_reactor/packed_bed_reactor.html
"""

import numpy as np
from scipy.integrate import solve_ivp

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class FixedBedReactor6D(BaseCase):
    """Adiabatic fixed-bed catalytic hydrogenation reactor (1-D steady-state ODE).

    6-D input → 3-D output (exit conversion, bed peak temperature, exit
    temperature).  Solver: scipy.integrate.solve_ivp (RK45, adaptive step).
    """

    meta = CaseMeta(
        case_id="fixed_bed_reactor_6d",
        name="Adiabatic fixed-bed catalytic reactor",
        version="1.0.0",
        domain="chemical_process",
        tags=[
            "reactor",
            "ode",
            "fixed_bed",
            "adiabatic",
            "arrhenius",
            "adiabatic_temperature_rise",
            "nonlinear",
            "interaction",
        ],
        dim_input=6,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="T_in",
                symbol="T_{in}",
                description="Reactor feed temperature",
                unit="K",
                lower=550.0,
                upper=700.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="C_A0",
                symbol="C_{A0}",
                description="Feed concentration of reactant A",
                unit="mol/m³",
                lower=0.5,
                upper=5.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="Ea",
                symbol="E_a",
                description="Activation energy (determines temperature sensitivity)",
                unit="kJ/mol",
                lower=50.0,
                upper=150.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="tau",
                symbol=r"\tau",
                description="Reactor space residence time",
                unit="s",
                lower=1.0,
                upper=20.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="dHr",
                symbol=r"\Delta H_r",
                description="Reaction enthalpy (negative = exothermic)",
                unit="kJ/mol",
                lower=-200.0,
                upper=-50.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="k0_log",
                symbol="\\log_{10} k_0",
                description="Log₁₀ of Arrhenius pre-exponential factor",
                unit="-",
                lower=6.0,
                upper=12.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="X_A",
                symbol="X_A",
                description="Exit conversion of reactant A",
                unit="-",
                is_primary=True,
                expected_range=(0.0, 1.0),
                failure_threshold=0.95,
                failure_direction="below",  # conversion below 95 %
            ),
            OutputVariable(
                name="T_max",
                symbol="T_{max}",
                description="Bed axial peak temperature (= outlet temperature for this irreversible adiabatic model)",
                unit="K",
                is_primary=True,
                expected_range=(500.0, 900.0),
                failure_threshold=800.0,  # exceeds catalyst thermal limit
                failure_direction="above",
            ),
            OutputVariable(
                name="T_outlet",
                symbol="T_{out}",
                description="Reactor exit temperature (= T_in + adiabatic rise)",
                unit="K",
                is_primary=False,
                expected_range=(500.0, 900.0),
            ),
        ],
        governing_equation=(
            "1-D adiabatic fixed-bed ODE system (residence time as independent var):\n"
            "  dθ/dτ = −Da · θ · exp(γ(1 − 1/T̃))\n"
            "  dT̃/dτ = β · Da · θ · exp(γ(1 − 1/T̃))\n"
            "where θ = C_A/C_A0, T̃ = T/T_in,\n"
            "  Da = k₀·exp(−Ea/R/T_in) [Damköhler number]\n"
            "  β = (−ΔH_r)·C_A0/(ρ·C_p·T_in) [adiabatic temp-rise number]\n"
            "  γ = Ea·1000/(R·T_in) [Arrhenius dimensionless activation energy]\n"
            "  For the irreversible exothermic adiabatic model, dT̃/dτ ≥ 0 and "
            "T_max = T_outlet."
        ),
        pde_type="ode_system",
        spatial_dim=1,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Levenspiel, O. (1999). Chemical Reaction Engineering, 3rd ed. Wiley. "
            "ISBN 978-0-471-25424-9.",
            "Froment, G. F. & Bischoff, K. B. (1990). Chemical Reactor Analysis "
            "and Design, 2nd ed. Wiley. ISBN 978-0-471-51040-0.",
            "University of Michigan Elements of Chemical Reaction Engineering, "
            "Adiabatic energy-balance learning module. Public teaching anchor "
            "for adiabatic PFR energy-balance trends, not calibrated "
            "fixed-bed-reactor validation data. "
            "https://websites.umich.edu/~elements/asyLearn/bits/ebforadiabatic/index.htm",
            "COMSOL Packed Bed Reactor documentation. Public boundary "
            "reference for omitted packed-bed multigeometry species transport, "
            "pellet diffusion, Ergun/pressure-drop, and reactor-design "
            "workflow scope; not calibration data for this compact "
            "adiabatic ODE surrogate. "
            "https://doc.comsol.com/6.0/doc/com.comsol.help.models.chem.packed_bed_reactor/packed_bed_reactor.html",
        ],
        solver_library="scipy.integrate.solve_ivp",
        solver_details=(
            "RK45 adaptive step, rtol=1e-6, atol=1e-8, 100 axial nodes "
            "recorded; private physical-input evaluation rejects malformed "
            "arrays, non-finite values, and values outside the declared "
            "benchmark domain before ODE assembly; intentionally omits "
            "packed-bed multigeometry species transport, pellet diffusion/"
            "effectiveness factors, Ergun or other pressure-drop equations, "
            "radial heat-transfer, catalyst deactivation, and multiphase "
            "plant-reactor operation"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 8.0},
        cost_level="L1",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_unimodal", "strong_interaction", "heterogeneous"],
        effective_dim=6,
        important_variables=["Ea", "k0_log", "T_in", "tau"],
        interaction_pairs=[("T_in", "Ea"), ("T_in", "tau"), ("C_A0", "dHr")],
        has_failure_boundary=True,
        failure_condition="X_A < 0.95 (conversion below process requirement) or T_max > 800 K (catalyst overheat)",
        sobol_S1={
            "T_in": 0.006,
            "C_A0": 0.000,
            "Ea": 0.475,
            "tau": 0.009,
            "dHr": 0.000,
            "k0_log": 0.204,
        },
        sobol_ST={
            "T_in": 0.119,
            "C_A0": 0.000,
            "Ea": 0.760,
            "tau": 0.028,
            "dHr": 0.000,
            "k0_log": 0.481,
        },
        sobol_computation="salib_saltelli_n1024_seed42_X_A_output",
        applicable_scenarios=["VS-B", "VS-C", "VS-D", "VS-G"],
        scenario_notes={
            "VS-B": "T_in and Ea are strongly coupled via the Arrhenius term (γ = Ea/RT_in); "
            "CCD is the minimum requirement; resolution-III designs completely miss this interaction.",
            "VS-C": "Near Da ∼ 1 (low→high conversion transition), the X_A vs tau response "
            "surface has a clear S-shape nonlinearity — a surrogate-accuracy challenge zone.",
            "VS-D": "6-D problem, all variables matter; how does surrogate accuracy fare "
            "at small sample sizes (N = 15–25)?  A typical small-sample efficiency test.",
            "VS-G": "Locate the feasible-domain boundary where X_A > 0.95 and outlet T_max < 800 K; "
            "a classic application for sequential reliability methods (ALM, EFF, etc.).",
        },
        recommended_n_train={"default": 80},
        n_saturation=400,
        noise_factor_candidates=["T_in", "C_A0"],
        control_factor_candidates=["tau", "Ea", "k0_log"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    # Fixed physical constants
    _R = 8.314e-3  # kJ/(mol·K)
    _rho = 800.0  # kg/m³
    _Cp = 2.0  # kJ/(kg·K)
    _N_POINTS = 100  # number of axial ODE nodes

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> None:
        """Reject malformed or out-of-domain private physical inputs."""
        X = np.asarray(X_phys, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.meta.dim_input:
            raise ValueError(
                "fixed_bed_reactor_6d expects a physical input array with "
                f"shape (n, {self.meta.dim_input}); got {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "fixed_bed_reactor_6d received non-finite physical inputs; "
                "refusing to extrapolate the adiabatic fixed-bed ODE surrogate."
            )

        tol = 1e-10
        for j, var in enumerate(self.meta.input_variables):
            lower = float(var.lower)
            upper = float(var.upper)
            scale = max(abs(lower), abs(upper), 1.0)
            bad = (X[:, j] < lower - tol * scale) | (X[:, j] > upper + tol * scale)
            if np.any(bad):
                raise ValueError(
                    "fixed_bed_reactor_6d physical input "
                    f"{var.name!r} is outside the declared benchmark domain "
                    f"[{lower}, {upper}]."
                )

    def _evaluate_core(
        self,
        X_phys: np.ndarray,
        fidelity: str = "default",
        **kwargs,
    ) -> np.ndarray:
        """Solve the reactor ODE system for each sample point.

        Parameters
        ----------
        X_phys : ndarray, shape (n, 6)
            Columns: [T_in(K), C_A0(mol/m³), Ea(kJ/mol),
                      tau(s), dHr(kJ/mol), k0_log(-)]
        """
        self._validate_physical_inputs(X_phys)
        X_phys = np.asarray(X_phys, dtype=float)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                Y[i] = self._solve_reactor(
                    T_in=X_phys[i, 0],
                    C_A0=X_phys[i, 1],
                    Ea=X_phys[i, 2],
                    tau=X_phys[i, 3],
                    dHr=X_phys[i, 4],
                    k0=10.0 ** X_phys[i, 5],
                )
            except Exception as exc:
                raise RuntimeError(
                    "fixed_bed_reactor_6d refusing silent NaN after fixed-bed "
                    f"reactor ODE failure for T_in={X_phys[i, 0]!r}, "
                    f"C_A0={X_phys[i, 1]!r}, Ea={X_phys[i, 2]!r}, "
                    f"tau={X_phys[i, 3]!r}, dHr={X_phys[i, 4]!r}, "
                    f"k0_log={X_phys[i, 5]!r}"
                ) from exc
        if (
            Y.shape != (n, self.meta.dim_output)
            or not np.all(np.isfinite(Y))
            or np.any(Y[:, 0] < -1e-9)
            or np.any(Y[:, 0] > 1.0 + 1e-9)
            or np.any(Y[:, 1] <= 0.0)
            or np.any(Y[:, 2] <= 0.0)
            or np.any(Y[:, 1] < X_phys[:, 0] - 1e-6)
            or np.any(np.abs(Y[:, 1] - Y[:, 2]) > 1e-6)
        ):
            raise RuntimeError(
                "fixed_bed_reactor_6d produced a malformed, non-finite, or "
                "non-physical adiabatic-reactor response; refusing silent "
                "archive rows"
            )
        return Y

    def _solve_reactor(
        self,
        T_in: float,
        C_A0: float,
        Ea: float,
        tau: float,
        dHr: float,
        k0: float,
    ) -> np.ndarray:
        """Solve the 1-D adiabatic reactor ODE for a single parameter set.

        State vector: [θ, T̃], where θ = C_A/C_A0, T̃ = T/T_in.
        Independent variable: t (residence-time fraction, 0 → 1, mapping to 0 → tau).

        Steps:
        1. Compute dimensionless groups Da, β, γ.
        2. Integrate with scipy.integrate.solve_ivp (RK45).
        3. Extract exit conversion, peak temperature, and exit temperature.
        """
        R = self._R
        rho = self._rho
        Cp = self._Cp
        if (
            not np.all(np.isfinite([T_in, C_A0, Ea, tau, dHr, k0]))
            or T_in <= 0.0
            or C_A0 < 0.0
            or Ea <= 0.0
            or tau < 0.0
            or dHr >= 0.0
            or k0 <= 0.0
        ):
            raise ValueError(
                "invalid fixed_bed_reactor_6d input outside the declared "
                "exothermic adiabatic-reactor contract"
            )

        # ── Dimensionless group computation ─────────────────────────────────
        # Arrhenius rate constant at feed temperature
        k_in = k0 * np.exp(-Ea / (R * T_in))

        # Damköhler number
        Da = k_in * tau

        # Adiabatic temperature-rise number β = (-dHr) * C_A0 / (rho * Cp * T_in)
        beta = (-dHr) * C_A0 / (rho * Cp * T_in)

        # Dimensionless activation energy γ = Ea / (R * T_in)
        gamma = Ea / (R * T_in)

        # ── ODE right-hand side ───────────────────────────────────────────────────
        def rhs(t, y):
            theta, T_tilde = y
            theta = max(theta, 0.0)  # prevent numerical negatives
            T_tilde_safe = float(np.clip(T_tilde, 0.2, 5.0))
            # Arrhenius term (corrected: rate at current temperature)
            exponent = -gamma * (1.0 / T_tilde_safe - 1.0)
            arr = float(np.exp(np.clip(exponent, -80.0, 80.0)))
            arr = min(arr, 1e6)  # keep the ODE non-stiff at extreme trials
            r = Da * theta * arr
            dtheta_dt = -r
            dT_dt = beta * r
            return [dtheta_dt, dT_dt]

        # ── Solve ODE ───────────────────────────────────────────────────────────
        y0 = [1.0, 1.0]  # θ(0)=1, T̃(0)=1
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0, 1, self._N_POINTS)

        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solve failed: {sol.message}")
        if not np.all(np.isfinite(sol.y)):
            raise RuntimeError("non-finite fixed-bed reactor ODE trajectory")

        theta_raw = sol.y[0]
        T_tilde_profile = sol.y[1]
        tol = 1e-6
        if (
            np.any(theta_raw < -tol)
            or np.any(theta_raw > 1.0 + tol)
            or np.any(T_tilde_profile < 1.0 - tol)
            or np.any(np.diff(theta_raw) > tol)
            or np.any(np.diff(T_tilde_profile) < -tol)
        ):
            raise RuntimeError("non-physical fixed-bed reactor ODE trajectory")
        theta_profile = np.clip(theta_raw, 0.0, 1.0)

        # ── Extract outputs ─────────────────────────────────────────────────────────
        X_A = float(1.0 - theta_profile[-1])
        T_max_K = float(np.max(T_tilde_profile) * T_in)
        T_outlet_K = float(T_tilde_profile[-1] * T_in)

        return np.array([X_A, T_max_K, T_outlet_K])
