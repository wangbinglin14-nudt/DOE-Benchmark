"""
Case M04: Wind-turbine power curve (Betz surrogate + Rayleigh AEP)
==================================================================

Engineering background
----------------------
The power curve (power vs. wind speed) of a wind turbine is the central
input for wind-resource assessment and wind-farm design.  The turbine
optimises power output at different wind speeds via pitch and speed control.

This case computes the power output, power coefficient, and annual energy
production (AEP, based on a Rayleigh wind-speed distribution) of a three-
bladed horizontal-axis wind turbine with specified design parameters.

Physics model (low-order wind-turbine surrogate)
------------------------------------------------
The model is intentionally not a full BEM induction-factor iteration. It uses
a Betz-limit power expression with simple finite-blade/solidity corrections
and integrates the resulting power curve over a Rayleigh annual wind-speed
distribution:
  · Betz limit: Cp_max = 16/27 ≈ 0.593 (Betz, 1919)
  · Power: P = 0.5·ρ·π·R²·V³·Cp
  · Annual energy: AEP = ∫ P(V) f_Rayleigh(V) dV · 8760 h,
    evaluated analytically over cut-in to rated speed plus the Rayleigh
    probability mass from rated to cut-out, with zero power outside the
    operating interval and no probability renormalisation

This compact benchmark surrogate does not model blade-element airfoil tables,
Prandtl tip-loss iteration, induction factors, drivetrain or generator
efficiency, air-density correction with altitude/temperature, wind shear,
turbulence intensity, wind-direction distributions, wind-rose/farm-layout wake
losses, terrain/roughness/obstacle effects, hub-height extrapolation,
site-specific Weibull fitting, hub-height interpolation between measured wind
heights, ideal-gas air-density adjustment from time-series temperature and
pressure, NREL WIND Toolkit WRF-based 2 km/5-minute wind-resource time series,
site-specific power-curve adjustment, electrical availability,
curtailment, degradation, Region-2.5 transition behaviour, detailed pitch or
torque controllers, AeroDyn steady BEM wake or induction solves, DBEMT
dynamic-inflow options, AeroDyn unsteady-aerodynamics or dynamic-stall models,
InflowWind time-series coupling, AeroDyn driver/input/output workflows,
structural loads, fatigue, aeroelastic FAST/OpenFAST simulation,
IEC 61400-12-1 power-performance measurement/certification workflows, or
calibration to a measured manufacturer power curve.
Current IEC 61400-12-1:2022 material is used only as boundary evidence: it
covers single-turbine power-performance measurement, while wind-measurement
method, site calibration, and obstacle/terrain assessment details are handled
by separate IEC standards that this compact surrogate does not implement.

Blade control strategy:
  V < V_rated: variable-speed operation, maintaining optimal tip-speed ratio λ_opt
  V > V_rated: pitch-to-feather power limiting; Cp drops, P = P_rated
  V > V_cut-out: turbine is shut down for safety (P = 0)

Input variables
---------------
R        Rotor radius            [30, 80]   m     Turbine size
lambda_  Design tip-speed ratio  [6, 10]    -     Optimal operating point
c_r      Root chord length       [2, 5]     m     Blade geometry
theta_0  Root twist angle        [8, 20]    deg   Recorded geometry variable; neutral in this surrogate
V_rated  Rated wind speed        [10, 14]   m/s   Control strategy
V_bar    Annual mean wind speed  [5, 12]    m/s   Wind resource (Rayleigh dist.)

Output responses
----------------
Cp_max   Maximum power coefficient  -     Aerodynamic efficiency
P_rated  Rated power                MW    Turbine rating
AEP      Annual energy production (Rayleigh) GWh  Economic value indicator

DOE validation value
--------------------
VS-B (effect estimation):
    lambda_ and R interact through the finite-blade and solidity correction.

VS-H (robust Taguchi):
    V_bar (annual mean wind speed) is the noise factor (climate variability);
    R and lambda_ are control factors.

References
----------
1. Hansen, M. O. L. (2015). Aerodynamics of Wind Turbines, 3rd ed. Routledge.
2. Burton, T. et al. (2011). Wind Energy Handbook, 2nd ed. Wiley.
3. NREL. (2014). Reference Manual for the System Advisor Model's Wind Power
   Performance Model. NREL/TP-6A20-60570.
   https://www.nrel.gov/docs/fy14osti/60570.pdf
4. NREL SAM wind-power help pages, used as current boundary evidence for
   time-series/SAM-CSV wind resource files, Weibull resource options,
   hub-height/wind-shear, air-density, wake/farm, hourly time-step,
   loss-factor, and system-availability/operating-loss workflows not
   implemented here.  The SAM Losses page further separates internal,
   external, and future wake losses; turbine, balance-of-plant, and grid
   availability; electrical/parasitic losses; generic and site-specific power
   curve adjustments; high-wind hysteresis; environmental degradation or
   shutdowns; and curtailment/operational-strategy losses.
   https://samrepo.nrelcloud.org/help/wind_power.html
   https://samrepo.nrelcloud.org/help/wind_resource.html
   https://samrepo.nrelcloud.org/help/weather_format_csv_wind.html
   https://samrepo.nrelcloud.org/help/wind_losses.html
   https://samrepo.nlr.gov/help/wind_power.html
   https://samrepo.nlr.gov/help/wind_resource.html
   https://samrepo.nlr.gov/help/weather_format_csv_wind.html
   https://samrepo.nlr.gov/help/wind_losses.html
5. NREL. (2009). Definition of a 5-MW Reference Wind Turbine for Offshore
   System Development. NREL/TP-500-38060.
   https://www.nrel.gov/docs/fy09osti/38060.pdf
6. NREL. OpenFAST documentation, aero-hydro-servo-elastic turbine simulation
   context. https://openfast.readthedocs.io/
7. NREL OpenFAST overview page, used as official boundary evidence for
   FAST-based aero-hydro-servo-elastic wind-turbine and wind-plant models.
   https://www.nrel.gov/wind/nwtc/openfast
8. OpenFAST AeroDyn documentation, used as official boundary evidence for
   steady BEM wake, dynamic inflow, unsteady-aerodynamics/dynamic-stall,
   InflowWind, and AeroDyn input/output workflows not implemented here.
   https://openfast.readthedocs.io/en/dev/source/user/aerodyn/
9. NREL WindWatts for Distributed Wind, used as wind-resource assessment
   boundary context for terrain, obstacle, observational-data, and wind-rose
   workflows not included here.
   https://www.nrel.gov/wind/windwatts
10. IEC 61400-12-1:2022, Wind energy generation systems - Part 12-1:
    Power performance measurements of electricity producing wind turbines,
    used as current boundary evidence for measured power-curve, derived
    energy-production, and uncertainty-assessment workflows not implemented
    here.  The 2022 edition cancels and replaces IEC 61400-12-1:2017 and
    IEC 61400-12-2:2013 and moves wind-speed measurement, site calibration,
    and obstacle/terrain assessment material to separate standards outside
    this benchmark.
    https://webstore.iec.ch/en/publication/68499
11. NREL WIND Toolkit, official high-resolution wind-resource time-series
    boundary context for WRF-based 2 km / 5-minute wind-speed, wind-direction,
    temperature, pressure, density, and power fields not implemented here.
    https://www.nrel.gov/grid/wind-toolkit
"""

import numpy as np
from scipy.special import gammainc, gamma

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class WindTurbinePower6D(BaseCase):
    """Wind-turbine power curve (6D → 3D, Betz parameterised model + Rayleigh integration)."""

    _RHO = 1.225  # kg/m³, air density
    _B = 3  # number of blades
    _V_CUT_IN = 3.0  # m/s, cut-in wind speed
    _V_CUT_OUT = 25.0  # m/s, cut-out wind speed

    meta = CaseMeta(
        case_id="wind_turbine_power_6d",
        name="Wind-turbine power curve (Betz parameterised + Rayleigh AEP)",
        version="1.0.0",
        domain="multidisciplinary",
        tags=[
            "wind_turbine",
            "power_curve",
            "betz_surrogate",
            "renewable",
            "6d",
            "rayleigh",
            "annual_energy",
        ],
        dim_input=6,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="R",
                symbol="R",
                description="Rotor radius (blade length)",
                unit="m",
                lower=30.0,
                upper=80.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="lam_opt",
                symbol=r"\lambda_{opt}",
                description="Optimal tip-speed ratio (design point)",
                unit="-",
                lower=6.0,
                upper=10.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="c_r",
                symbol="c_r",
                description="Blade root chord length",
                unit="m",
                lower=2.0,
                upper=5.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="theta_0",
                symbol=r"\theta_0",
                description="Blade root twist/pitch angle recorded for geometry context; neutral in this low-order surrogate",
                unit="deg",
                lower=8.0,
                upper=20.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="V_rated",
                symbol="V_{rated}",
                description="Rated wind speed",
                unit="m/s",
                lower=10.0,
                upper=14.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="V_bar",
                symbol=r"\bar{V}",
                description="Annual mean wind speed (Rayleigh distribution parameter)",
                unit="m/s",
                lower=5.0,
                upper=12.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="Cp_max",
                symbol="C_{p,max}",
                description="Maximum power coefficient (Betz limit = 16/27 ≈ 0.593)",
                unit="-",
                is_primary=True,
                expected_range=(0.2, 0.57),
            ),
            OutputVariable(
                name="P_rated",
                symbol="P_{rated}",
                description="Rated power (maximum power at V = V_rated)",
                unit="MW",
                is_primary=True,
                expected_range=(0.5, 20.0),
            ),
            OutputVariable(
                name="AEP",
                symbol="AEP",
                description="Annual energy production (based on Rayleigh wind-speed distribution)",
                unit="GWh",
                is_primary=False,
                expected_range=(0.5, 86.0),
            ),
        ],
        governing_equation=(
            "Betz parameterised power model (not full BEM induction-factor iteration):\n"
            "  Cp_max = (16/27)·(1−1.5/(B·λ))·(1−0.3σ)\n"
            "  P = 0.5·ρ·π·R²·V³·Cp_max  (V ≤ V_rated)\n"
            "  P = P_rated for V_rated < V ≤ V_cut-out, and P = 0 above cut-out\n"
            "  AEP = 8760·[k∫_{V_ci}^{V_rated} V³ f_Rayleigh(V)dV "
            "+ P_rated·Pr(V_rated < V ≤ V_cut-out)]"
        ),
        pde_type="algebraic",
        spatial_dim=0,
        is_steady=True,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Hansen, M. O. L. (2015). Aerodynamics of Wind Turbines, 3rd ed. Routledge.",
            "Burton, T. et al. (2011). Wind Energy Handbook, 2nd ed. Wiley.",
            "NREL. (2014). Reference Manual for the System Advisor Model's Wind Power Performance Model. NREL/TP-6A20-60570.",
            "NREL. (2014). Reference Manual for the System Advisor Model's Wind Power Performance Model. https://www.nrel.gov/docs/fy14osti/60570.pdf",
            "NREL SAM wind-power help, current model-boundary context for "
            "time-series/SAM-CSV/SRW wind resources, Weibull resource options, "
            "single-year hourly simulation, hub-height/wind-shear, air-density, "
            "wake/farm, wind-farm loss-factor, system-availability/"
            "operating-loss workflows, hub-height interpolation between "
            "measured wind speeds, and ideal-gas air-density adjustment from "
            "time-series temperature and pressure. "
            "https://samrepo.nrelcloud.org/help/wind_power.html",
            "NREL/NLR SAM wind-power help, current help-domain mirror for the "
            "same wind-resource, hourly simulation, hub-height, density, wake, "
            "loss, and grid-delivery boundary context. "
            "https://samrepo.nlr.gov/help/wind_power.html",
            "NREL SAM wind losses help, current loss-boundary context for "
            "internal/external/future wake losses, turbine/balance-of-plant/"
            "grid availability, electrical/parasitic losses, generic and "
            "site-specific power-curve adjustments, high-wind hysteresis, "
            "environmental degradation/shutdown losses, and curtailment/"
            "operational-strategy losses. "
            "https://samrepo.nrelcloud.org/help/wind_losses.html",
            "NREL/NLR SAM wind losses help, current help-domain mirror for "
            "wake, availability, electrical/parasitic, turbine-performance, "
            "environmental, and curtailment/operational-strategy loss "
            "boundary context. https://samrepo.nlr.gov/help/wind_losses.html",
            "NREL SAM wind-resource help. https://samrepo.nrelcloud.org/help/wind_resource.html",
            "NREL/NLR SAM wind-resource help current mirror. https://samrepo.nlr.gov/help/wind_resource.html",
            "NREL SAM CSV Format for Wind help. "
            "https://samrepo.nrelcloud.org/help/weather_format_csv_wind.html",
            "NREL/NLR SAM CSV Format for Wind help current mirror. "
            "https://samrepo.nlr.gov/help/weather_format_csv_wind.html",
            "NREL. (2009). Definition of a 5-MW Reference Wind Turbine for Offshore System Development. NREL/TP-500-38060. https://www.nrel.gov/docs/fy09osti/38060.pdf",
            "NREL. (2019). Distributed Wind Resource Assessment Framework. NREL/TP-5000-72523. https://docs.nrel.gov/docs/fy19osti/72523.pdf",
            "NREL OpenFAST documentation, aero-hydro-servo-elastic wind-turbine simulation context. "
            "https://openfast.readthedocs.io/",
            "NREL OpenFAST overview page, official FAST-based aero-hydro-servo-elastic "
            "wind-turbine and wind-plant modelling context, with regression and "
            "unit testing in an open-source software-engineering workflow outside "
            "this compact benchmark. "
            "https://www.nrel.gov/wind/nwtc/openfast",
            "OpenFAST AeroDyn documentation, official boundary context for "
            "steady BEM wake/induction modelling, DBEMT dynamic-inflow options, "
            "AeroDyn unsteady-aerodynamics and dynamic-stall models, InflowWind "
            "coupling, and AeroDyn input/output/driver workflows not implemented "
            "by this Betz/Rayleigh surrogate. "
            "https://openfast.readthedocs.io/en/dev/source/user/aerodyn/",
            "NREL WindWatts for Distributed Wind, wind-resource assessment context "
            "for terrain, obstacle, observational-data, and wind-rose workflows. "
            "https://www.nrel.gov/wind/windwatts",
            "IEC 61400-12-1:2022, Wind energy generation systems - Part 12-1: "
            "Power performance measurements of electricity producing wind turbines, "
            "current measured-power-curve, derived energy-production, and "
            "combined uncertainty-assessment boundary context; this third edition "
            "replaces IEC 61400-12-1:2017 and IEC 61400-12-2:2013 and moves "
            "wind-speed measurement, site calibration, and obstacle/terrain "
            "assessment material to separate standards outside this benchmark. "
            "https://webstore.iec.ch/en/publication/68499",
            "NREL WIND Toolkit, official high-resolution WRF-based wind-resource "
            "time-series boundary context for 2 km / 5-minute wind-speed, "
            "wind-direction, temperature, pressure, density, and power fields "
            "outside this single-mean-speed Rayleigh surrogate. "
            "https://www.nrel.gov/grid/wind-toolkit",
        ],
        solver_library="numpy/scipy.special (Betz parameterised model + analytical Rayleigh AEP)",
        solver_details=(
            "Betz limit with simplified finite-blade/tip-loss-inspired factor "
            "and solidity correction (no Prandtl BEM tip-loss calculation or "
            "BEM induction-factor iteration); Rayleigh AEP by closed-form "
            "integration of the cubic sub-rated power segment plus the rated-power "
            "Rayleigh probability mass, with explicit checks that the sub-rated "
            "third-moment contribution is finite/non-negative and that the rated "
            "Rayleigh probability mass stays within [0, 1]; "
            "zero turbine power is assumed below cut-in and above cut-out, and "
            "the Rayleigh probability mass outside that interval is not "
            "renormalised; no airfoil-table aerodynamics, drivetrain/generator "
            "efficiency, air-density correction, wind shear, turbulence, wind "
            "direction/wind-rose model, terrain/roughness/obstacle adjustment, "
            "hub-height extrapolation, site-specific Weibull fitting, wake/farm "
            "losses, SAM wind-resource files, SAM CSV/SRW time-series resource "
            "simulation, SAM single-year hourly production simulation, wind-farm "
            "loss-factor and system-availability/operating-loss modelling, "
            "hub-height interpolation between measured wind heights, ideal-gas "
            "air-density adjustment from time-series temperature/pressure, "
            "NREL WIND Toolkit WRF-based 2 km/5-minute wind-resource time "
            "series, multi-height wind speed/direction, temperature, pressure, "
            "density, and power fields, "
            "site-specific power-curve adjustment, "
            "internal/external/future wake accounting, turbine/balance-of-plant "
            "availability-loss accounting, curtailment/operational-strategy "
            "losses, grid availability, electrical/parasitic losses, high-wind "
            "hysteresis, environmental degradation/shutdown losses, "
            "observational-data assimilation, WindWatts-style resource "
            "assessment, IEC 61400-12-1:2022 power-performance measurement, "
            "derived-energy-production, combined-uncertainty, or certification "
            "workflow, IEC wind-measurement-method standards, IEC "
            "site-calibration standards, IEC obstacle/terrain-assessment "
            "standards, measurement-sector filtering, site calibration, met-mast/lidar "
            "measurement uncertainty, air-density normalisation, terrain/obstacle "
            "assessment, measured power-curve binning, system-loss/availability modelling, curtailment, degradation, Region-2.5 "
            "transition, detailed pitch/torque controller, structural loads, "
            "fatigue, OpenFAST/AeroDyn steady BEM wake or induction solves, "
            "DBEMT dynamic-inflow options, AeroDyn unsteady-aerodynamics/"
            "dynamic-stall models, InflowWind time-series coupling, AeroDyn "
            "driver/input/output workflows, FAST/OpenFAST aeroelastic simulation, "
            "or measured-power-curve "
            "calibration; named _validate_physical_inputs private physical-input "
            "evaluation rejects malformed arrays and R/lam_opt/c_r/theta_0/"
            "V_rated/V_bar values outside the declared benchmark domain before "
            "Betz/Rayleigh response assembly; non-finite, non-physical, or "
            "out-of-contract Cp_max/P_rated/AEP responses are rejected before "
            "dispatch can archive a row"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 5.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_monotone", "strong_interaction"],
        # Case-level multi-output union: Cp_max depends on R/lam_opt/c_r,
        # P_rated additionally depends on V_rated, and AEP additionally
        # depends on V_bar. The Sobol table below is explicitly for Cp_max only.
        effective_dim=5,
        important_variables=["R", "lam_opt", "c_r", "V_rated", "V_bar"],
        interaction_pairs=[
            ("R", "lam_opt"),
            ("R", "c_r"),
            ("R", "V_rated"),
            ("R", "V_bar"),
            ("lam_opt", "c_r"),
            ("lam_opt", "V_rated"),
            ("lam_opt", "V_bar"),
            ("c_r", "V_rated"),
            ("c_r", "V_bar"),
            ("V_rated", "V_bar"),
        ],
        has_failure_boundary=False,
        sobol_S1={
            "R": 0.074,
            "lam_opt": 0.813,
            "c_r": 0.048,
            "theta_0": 0.000,
            "V_rated": 0.000,
            "V_bar": 0.000,
        },
        sobol_ST={
            "R": 0.120,
            "lam_opt": 0.861,
            "c_r": 0.087,
            "theta_0": 0.000,
            "V_rated": 0.000,
            "V_bar": 0.000,
        },
        sobol_computation=(
            "salib_saltelli_n2048_seed42_Cp_max_output; "
            "important_variables/effective_dim are case-level multi-output union"
        ),
        applicable_scenarios=["VS-C", "VS-F", "VS-H"],
        scenario_notes={
            "VS-C": "Cp increases monotonically toward the Betz-limited asymptote in lam_opt; AEP is strongly monotone w.r.t. V_bar.",
            "VS-F": "NREL SAM/5-MW/DWRA/WindWatts/WIND-Toolkit/OpenFAST/AeroDyn and IEC 61400-12-1:2022 references provide public power-curve, Weibull/Rayleigh AEP, wind-resource-assessment, measured power-performance, BEM/dynamic-inflow/dynamic-stall boundary, high-resolution wind-resource time-series, and aero-hydro-servo-elastic context; this compact algebraic surrogate is not calibrated to the NREL 5-MW turbine, manufacturer power-curve data, SAM wind time-series/resource-file or single-year hourly simulations, NREL WIND Toolkit WRF-based 2 km/5-minute wind-resource fields, SAM hub-height/shear/density/wake/loss workflows, SAM wind-farm loss-factor or system-availability/operating-loss workflows, SAM Losses wake/availability/electrical/turbine-performance/environmental/curtailment workflows, AeroDyn steady BEM wake/induction, DBEMT dynamic-inflow, unsteady-aerodynamics/dynamic-stall, InflowWind, or driver/input/output workflows, IEC measured-power-curve, wind-measurement-method, site-calibration, obstacle/terrain-assessment, derived-energy-production, combined-uncertainty, or power-performance test/certification workflows, site-specific wind-resource measurements, WindWatts resource estimates, or a full BEM/FAST/OpenFAST model.",
            "VS-H": "V_bar is the inter-annual noise factor; design a robust turbine insensitive to wind-speed variability.",
        },
        recommended_n_train={"default": 60},
        n_saturation=300,
        noise_factor_candidates=["V_bar"],
        control_factor_candidates=["R", "lam_opt", "c_r", "theta_0", "V_rated"],
        reference_solution_available=True,
        reference_grid_size=3000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "wind_turbine_power_6d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        R = X_phys[:, 0]
        lam_opt = X_phys[:, 1]
        c_r = X_phys[:, 2]
        theta_0 = X_phys[:, 3]
        V_rated = X_phys[:, 4]
        V_bar = X_phys[:, 5]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(R < 30.0 - tol)
            or np.any(R > 80.0 + tol)
            or np.any(lam_opt < 6.0 - tol)
            or np.any(lam_opt > 10.0 + tol)
            or np.any(c_r < 2.0 - tol)
            or np.any(c_r > 5.0 + tol)
            or np.any(theta_0 < 8.0 - tol)
            or np.any(theta_0 > 20.0 + tol)
            or np.any(V_rated < 10.0 - tol)
            or np.any(V_rated > 14.0 + tol)
            or np.any(V_bar < 5.0 - tol)
            or np.any(V_bar > 12.0 + tol)
        ):
            raise ValueError(
                "wind_turbine_power_6d physical inputs must stay inside the "
                "declared R/lam_opt/c_r/theta_0/V_rated/V_bar benchmark domain "
                "before Betz/Rayleigh response assembly"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                response = self._turbine(
                    R=float(X_phys[i, 0]),
                    lam_opt=float(X_phys[i, 1]),
                    c_r=float(X_phys[i, 2]),
                    theta_0=float(X_phys[i, 3]),
                    V_rated=float(X_phys[i, 4]),
                    V_bar=float(X_phys[i, 5]),
                )
                full_year_rated_gwh = response[1] * 8760.0 / 1000.0
                if (
                    response.shape != (3,)
                    or not np.all(np.isfinite(response))
                    or not (0.0 < response[0] <= 16.0 / 27.0)
                    or response[1] <= 0.0
                    or response[2] <= 0.0
                    or response[2] > full_year_rated_gwh * (1.0 + 1e-12)
                ):
                    raise ValueError(
                        "non-finite or non-physical wind-turbine response "
                        f"Cp/P_rated/AEP={np.array2string(response, precision=6, separator=', ')}"
                    )
                output_ranges = np.asarray(
                    [output.expected_range for output in self.meta.output_variables],
                    dtype=float,
                )
                below = response < (output_ranges[:, 0] - 1e-10)
                above = response > (output_ranges[:, 1] + 1e-10)
                if np.any(below) or np.any(above):
                    raise ValueError(
                        "out-of-contract wind-turbine Cp_max/P_rated/AEP "
                        "response outside the declared output ranges"
                    )
                Y[i] = response
            except Exception as exc:
                raise RuntimeError(
                    "wind_turbine_power_6d refusing silent NaN after "
                    "wind-turbine power-curve evaluation failure for "
                    f"R={X_phys[i, 0]!r}, lam_opt={X_phys[i, 1]!r}, "
                    f"c_r={X_phys[i, 2]!r}, theta_0={X_phys[i, 3]!r}, "
                    f"V_rated={X_phys[i, 4]!r}, V_bar={X_phys[i, 5]!r}"
                ) from exc
        return Y

    def _turbine(self, R, lam_opt, c_r, theta_0, V_rated, V_bar):
        """Betz-parameterized power curve + closed-form Rayleigh AEP integration."""
        if (
            not np.all(np.isfinite([R, lam_opt, c_r, theta_0, V_rated, V_bar]))
            or R <= 0.0
            or lam_opt <= 0.0
            or c_r <= 0.0
            or V_rated <= self._V_CUT_IN
            or V_rated >= self._V_CUT_OUT
            or V_bar <= 0.0
        ):
            raise ValueError(
                "invalid wind_turbine_power_6d input outside the declared "
                "Betz/Rayleigh surrogate contract"
            )
        rho = self._RHO
        B = self._B

        # Blade solidity (simplified uniform chord)
        c_avg = c_r * 0.6  # mean chord (tapered-blade approximation)
        sigma = B * c_avg / (np.pi * R)

        # Maximum power coefficient (low-order correction for blade count and solidity).
        # This is a finite-blade/tip-loss-inspired algebraic factor, not a
        # Prandtl BEM tip-loss calculation.
        tip_loss = 1.0 - 1.5 / (B * lam_opt)
        Cp_max = (16.0 / 27.0) * tip_loss * (1.0 - 0.3 * sigma)
        Cp_max = float(Cp_max)
        if not np.isfinite(Cp_max) or not (0.0 < Cp_max <= 16.0 / 27.0):
            raise ValueError("non-physical wind-turbine power coefficient")

        # Rated power (V = V_rated, Cp = Cp_max)
        P_rated_W = 0.5 * rho * np.pi * R**2 * V_rated**3 * Cp_max
        P_rated = float(P_rated_W / 1e6)  # MW
        if not np.isfinite(P_rated_W) or P_rated_W <= 0.0:
            raise ValueError("non-physical wind-turbine rated power")

        # Rayleigh wind-speed distribution PDF (Weibull k = 2). The AEP
        # integral is evaluated in closed form over the piecewise power curve,
        # avoiding grid sensitivity at the rated-speed kink.
        c_rayleigh = V_bar * 2.0 / np.sqrt(np.pi)  # scale parameter
        k_power = 0.5 * rho * np.pi * R**2 * Cp_max
        u_cut_in = (self._V_CUT_IN / c_rayleigh) ** 2
        u_rated = (V_rated / c_rayleigh) ** 2
        u_cut_out = (self._V_CUT_OUT / c_rayleigh) ** 2
        subrated_expectation = c_rayleigh**3 * gamma(2.5) * (
            gammainc(2.5, u_rated) - gammainc(2.5, u_cut_in)
        )
        rated_probability = np.exp(-u_rated) - np.exp(-u_cut_out)
        if not np.isfinite(subrated_expectation) or subrated_expectation < 0.0:
            raise ValueError("non-physical Rayleigh sub-rated wind-speed moment")
        if not np.isfinite(rated_probability) or rated_probability < 0.0 or rated_probability > 1.0:
            raise ValueError("non-physical Rayleigh rated-power probability mass")

        # Annual energy production (GWh)
        AEP_Wh = (k_power * subrated_expectation + P_rated_W * rated_probability) * 8760.0
        AEP_GWh = float(AEP_Wh / 1e9)
        if (
            not np.isfinite(AEP_GWh)
            or AEP_GWh <= 0.0
            or AEP_GWh > P_rated * 8760.0 / 1000.0 * (1.0 + 1e-12)
        ):
            raise ValueError("non-physical wind-turbine annual energy production")

        return np.array([Cp_max, P_rated, AEP_GWh])
