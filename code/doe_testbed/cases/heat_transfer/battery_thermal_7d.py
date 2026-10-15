"""
Case T03: Lithium-ion battery cell transient thermal management
================================================================

Engineering background
----------------------
Thermal management of lithium-ion batteries in electric vehicles, energy-
storage systems, and consumer electronics is critical for safety and cycle
life.  Heat generation during battery operation originates from:
  · Ohmic heat (Joule heat): Q_ohm = I²·R_internal
  · Electrochemical heat (reversible heat): Q_ecs = I·T·dU/dT (phase-change
    latent heat)

This case uses a lumped-parameter (thermal-mass) model + ODE system to
simulate the temperature–time history of a single battery cell during
constant-current discharge, studying the effects of cooling conditions
and internal resistance on peak temperature.

Physics model (lumped thermal-capacity ODE)
-------------------------------------------
Battery thermal model:
  m·c_p·dT/dt = Q_gen − Q_cool
  Q_gen = I²·R(T) − I·T·dU/dT
          (irreversible Joule heat + Bernardi reversible entropic heat,
           using the discharge sign convention)
  Q_cool = h·A·(T − T_cool)  (convective cooling)

Internal-resistance temperature dependence (Arrhenius type):
  R(T) = R_ref · exp(E_{a,R}/R_gas · (1/T − 1/T_ref))

Discharge process (constant current, duration t_end = 3600/C_rate s)

Input variables
---------------
C_rate      Discharge rate      [0.5, 5.0]  -       Discharge speed (higher → faster / hotter)
R_ref       Reference resistance [1, 50]    mΩ      Battery quality indicator
h_cool      Cooling HTC          [10, 200]  W/m²K   Cooling conditions
T_cool      Coolant temperature  [10, 35]   °C      Thermal-management system parameter
Ea_R        Resistance activation energy [2000, 8000] J/mol  Temperature-dependence strength
cp          Specific heat capacity [800, 1200] J/(kg·K) Battery material
A_cool      Cooling area          [0.01, 0.1] m²    Cell geometry parameter

Output responses
----------------
T_max       Peak temperature     °C   Primary output (safety indicator; > 60 °C = warning)
T_end       End-of-discharge temp °C   Thermal-management design target
delta_T     Maximum temperature rise °C Temperature-rise magnitude

DOE validation value
--------------------
VS-B (effect estimation):
    C_rate has a quadratic effect on heat generation via I², with a
    multiplicative interaction with R_ref.

VS-G (sequential / reliability):
    Find the safe operating domain where T_max < 60 °C (combinations
    of C_rate, h_cool, T_cool).

References
----------
1. Bernardi, D. et al. (1985). A general energy balance for battery systems.
   J. Electrochem. Soc., 132(1):5-12.
2. Pals, C. R. & Newman, J. (1995). Thermal modeling of the lithium/polymer
   battery. J. Electrochem. Soc., 142(10):3274-3281.
3. OSTI record for Bernardi et al., used as public energy-balance provenance.
   https://www.osti.gov/biblio/5913742
4. NREL Energy Storage Thermal Management, used as boundary evidence for
   cell/module/pack/system thermal characterization and pack-level thermal
   management workflows not implemented here.
   https://www.nrel.gov/transportation/energy-storage-thermal-mgmt
5. NREL Multi-Scale Battery Physics Modeling, used as boundary evidence for
   reduced-order electrochemical submodels and high-fidelity 3-D thermal,
   mechanical, electrochemical, and pack-level models not implemented here.
   https://www.nrel.gov/transportation/multi-scale-battery-physics-modeling.html
6. NASA NESC lithium-ion thermal-runaway calorimetry material, used as
   boundary evidence for abuse/thermal-runaway propagation and calorimetry
   workflows outside this normal-operation lumped ODE.
   https://www.nasa.gov/centers-and-facilities/nesc/calorimetry-of-lithium-ion-cells-during-thermal-runaway/
"""

import numpy as np
from scipy.integrate import solve_ivp

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class BatteryThermal7D(BaseCase):
    """Lithium-ion battery transient thermal management (7D → 3D, ODE system)."""

    _M = 0.1  # kg, battery mass (fixed)
    _T_REF = 298.15  # K, reference temperature
    _R_GAS = 8.314  # J/(mol·K)
    _CAPACITY = 3.0  # Ah, rated capacity
    _DUDT = -3.0e-4  # V/K, representative Li-ion entropy coefficient

    meta = CaseMeta(
        case_id="battery_thermal_7d",
        name="Lithium-ion battery cell transient thermal management",
        version="1.0.0",
        domain="heat_transfer",
        tags=["battery", "thermal", "ode", "lithium_ion", "7d", "transient", "thermal_management"],
        dim_input=7,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="C_rate",
                symbol="C",
                description="Discharge C-rate (1C = full discharge in 3600 s)",
                unit="-",
                lower=0.5,
                upper=5.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="R_ref",
                symbol="R_{ref}",
                description="Reference internal resistance at 25 °C",
                unit="mΩ",
                lower=1.0,
                upper=50.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="h_cool",
                symbol="h_{cool}",
                description="Cell-surface convective heat-transfer coefficient",
                unit="W/(m²·K)",
                lower=10.0,
                upper=200.0,
                log_scale=True,
                distribution="loguniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="T_cool",
                symbol="T_{cool}",
                description="Coolant temperature",
                unit="°C",
                lower=10.0,
                upper=35.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="Ea_R",
                symbol="E_{a,R}",
                description="Internal-resistance Arrhenius activation energy",
                unit="J/mol",
                lower=2000.0,
                upper=8000.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="cp",
                symbol="c_p",
                description="Battery specific heat capacity",
                unit="J/(kg·K)",
                lower=800.0,
                upper=1200.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="A_cool",
                symbol="A_{cool}",
                description="Cell cooling area",
                unit="m²",
                lower=0.01,
                upper=0.10,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="T_max",
                symbol="T_{max}",
                description="Peak temperature during discharge",
                unit="°C",
                is_primary=True,
                expected_range=(10.0, 105.0),
                failure_threshold=60.0,
                failure_direction="above",
            ),
            OutputVariable(
                name="T_end",
                symbol="T_{end}",
                description="Temperature at end of discharge",
                unit="°C",
                is_primary=False,
                expected_range=(10.0, 105.0),
            ),
            OutputVariable(
                name="delta_T",
                symbol=r"\Delta T",
                description="Maximum temperature rise (T_max − T_init)",
                unit="°C",
                is_primary=False,
                expected_range=(0.0, 72.0),
            ),
        ],
        governing_equation=(
            "Battery lumped-parameter thermal model (ODE):\n"
            "  m·c_p·dT/dt = I²·R(T) − I·T·dU/dT − h·A·(T − T_cool)\n"
            "  R(T) = R_ref·exp(E_a/R_gas·(1/T − 1/T_ref))  (Arrhenius)\n"
            "  I = C_rate × rated capacity; dU/dT = −3e−4 V/K "
            "(Bernardi reversible heat, discharge convention)"
        ),
        pde_type="ode_system",
        spatial_dim=0,
        is_steady=False,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Bernardi, D. et al. (1985). J. Electrochem. Soc., 132(1):5-12.",
            "OSTI record: Bernardi et al., A general energy balance for battery systems. https://www.osti.gov/biblio/5913742",
            "Pals, C. R. & Newman, J. (1995). J. Electrochem. Soc., 142(10):3274-3281.",
            "NREL Energy Storage Thermal Management. https://www.nrel.gov/transportation/energy-storage-thermal-mgmt",
            "NREL Multi-Scale Battery Physics Modeling. https://www.nrel.gov/transportation/multi-scale-battery-physics-modeling.html",
            "NASA NESC Calorimetry of Lithium-ion Cells During Thermal Runaway. https://www.nasa.gov/centers-and-facilities/nesc/calorimetry-of-lithium-ion-cells-during-thermal-runaway/",
        ],
        solver_library="scipy.integrate.solve_ivp",
        solver_details=(
            "Private _validate_physical_inputs rejects malformed, non-finite, "
            "or out-of-domain C_rate/R_ref/h_cool/T_cool/Ea_R/cp/A_cool "
            "physical arrays before lumped thermal ODE assembly; RK45, "
            "rtol=1e-6, atol=1e-9, max_step=t_end/50, 100 time points. "
            "This is a single-node normal-operation thermal surrogate; it is "
            "not calibrated cell calorimetry, electrochemical-thermal P2D, "
            "3-D CFD, cell/module/pack/system thermal-characterization "
            "workflow, pack/module propagation, ageing, abuse, or thermal-runaway "
            "validation."
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 5.0},
        cost_level="L1",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_unimodal", "strong_interaction"],
        effective_dim=5,
        important_variables=["T_cool", "C_rate", "R_ref", "h_cool", "A_cool"],
        interaction_pairs=[("C_rate", "R_ref"), ("h_cool", "A_cool")],
        has_failure_boundary=True,
        failure_condition="T_max > 60 °C (thermal-runaway risk zone)",
        sobol_S1={
            "C_rate": 0.008,
            "R_ref": 0.013,
            "h_cool": 0.007,
            "T_cool": 0.939,
            "Ea_R": 0.000,
            "cp": 0.000,
            "A_cool": 0.011,
        },
        sobol_ST={
            "C_rate": 0.031,
            "R_ref": 0.039,
            "h_cool": 0.021,
            "T_cool": 0.939,
            "Ea_R": 0.000,
            "cp": 0.000,
            "A_cool": 0.020,
        },
        sobol_computation="salib_saltelli_n1024_seed42_T_max_output",
        applicable_scenarios=["VS-B", "VS-C", "VS-G"],
        scenario_notes={
            "VS-B": "C_rate²·R_ref is a multiplicative interaction for heat generation; typical nonlinear effect.",
            "VS-G": "Find the safe operating region where T_max < 60 °C for the normal-operation lumped ODE, not thermal-runaway onset or propagation.",
        },
        recommended_n_train={"default": 80},
        n_saturation=300,
        noise_factor_candidates=["R_ref", "Ea_R", "cp"],
        control_factor_candidates=["C_rate", "h_cool", "T_cool", "A_cool"],
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "battery_thermal_7d physical inputs must be a 2-D array with "
                f"{self.meta.dim_input} columns"
            )
        if not np.all(np.isfinite(X_phys)):
            raise ValueError("battery_thermal_7d physical inputs must be finite")

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
                "battery_thermal_7d physical inputs outside the registered "
                "benchmark domain for "
                "C_rate/R_ref/h_cool/T_cool/Ea_R/cp/A_cool; "
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
                Y[i] = self._simulate(
                    C_rate=float(X_phys[i, 0]),
                    R_ref=float(X_phys[i, 1]) * 1e-3,  # mΩ → Ω
                    h_cool=float(X_phys[i, 2]),
                    T_cool=float(X_phys[i, 3]) + 273.15,  # °C → K
                    Ea_R=float(X_phys[i, 4]),
                    cp=float(X_phys[i, 5]),
                    A_cool=float(X_phys[i, 6]),
                )
                if (
                    Y[i].shape != (self.meta.dim_output,)
                    or not np.all(np.isfinite(Y[i]))
                    or Y[i, 0] < float(X_phys[i, 3])
                    or Y[i, 1] < float(X_phys[i, 3])
                    or Y[i, 2] < 0.0
                    or not np.isclose(Y[i, 2], Y[i, 0] - float(X_phys[i, 3]), rtol=1e-7, atol=1e-7)
                ):
                    raise ValueError(
                        "malformed, non-finite, or non-physical battery thermal response "
                        f"T_max/T_end/delta_T={np.array2string(Y[i], precision=6, separator=', ')}"
                    )
            except Exception as exc:
                raise RuntimeError(
                    "battery_thermal_7d refusing silent NaN after thermal "
                    f"ODE evaluation failure for C_rate={X_phys[i, 0]!r}, "
                    f"R_ref_mOhm={X_phys[i, 1]!r}, h_cool={X_phys[i, 2]!r}, "
                    f"T_cool_C={X_phys[i, 3]!r}, Ea_R={X_phys[i, 4]!r}, "
                    f"cp={X_phys[i, 5]!r}, A_cool={X_phys[i, 6]!r}"
                ) from exc
        return Y

    def _simulate(self, C_rate, R_ref, h_cool, T_cool, Ea_R, cp, A_cool):
        if (
            not np.all(np.isfinite([C_rate, R_ref, h_cool, T_cool, Ea_R, cp, A_cool]))
            or C_rate <= 0.0
            or R_ref <= 0.0
            or h_cool <= 0.0
            or T_cool <= 0.0
            or cp <= 0.0
            or A_cool <= 0.0
        ):
            raise ValueError(
                "invalid battery_thermal_7d input outside the declared "
                "lumped thermal ODE contract"
            )
        m = self._M
        T_ref = self._T_REF
        R_gas = self._R_GAS
        cap = self._CAPACITY  # Ah
        I = C_rate * cap  # A (constant-current discharge)
        t_end = 3600.0 / C_rate  # discharge duration (seconds)
        T0 = T_cool  # initial temperature = coolant temperature

        def rhs(t, T_arr):
            T = float(T_arr[0])
            T_safe = float(np.clip(T, 200.0, 600.0))
            # Internal resistance (Arrhenius temperature dependence)
            R_T = R_ref * np.exp(Ea_R / R_gas * (1.0 / T_safe - 1.0 / T_ref))
            R_T = np.clip(R_T, 1e-6, 10.0)  # prevent numerical overflow
            Q_joule = I**2 * R_T
            # Bernardi reversible heat.  For the representative negative
            # dU/dT used here, this adds heat during discharge.
            Q_rev = -I * T_safe * self._DUDT
            Q_gen = Q_joule + Q_rev
            Q_cool = h_cool * A_cool * (T - T_cool)
            dTdt = (Q_gen - Q_cool) / (m * cp)
            return [dTdt]

        t_eval = np.linspace(0, t_end, 100)
        sol = solve_ivp(
            rhs,
            [0, t_end],
            [T0],
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
            max_step=t_end / 50,
        )
        if not sol.success:
            raise RuntimeError(f"battery thermal ODE solve failed: {sol.message}")

        T_hist = sol.y[0]
        if T_hist.shape != (100,) or not np.all(np.isfinite(T_hist)):
            raise RuntimeError("non-finite battery thermal ODE trajectory")
        T_max_K = float(np.max(T_hist))
        T_end_K = float(T_hist[-1])
        delta_T = T_max_K - T0
        if T_max_K < T0 or T_end_K < T0 or delta_T < 0.0:
            raise RuntimeError("non-physical battery thermal cooling response")

        return np.array(
            [
                T_max_K - 273.15,
                T_end_K - 273.15,
                delta_T,
            ]
        )
