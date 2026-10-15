# Cross-Validation: `wind_turbine_power_6d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `wind_turbine_power_6d` |
| Implemented solver | Betz-parameterized algebraic power model plus closed-form Rayleigh AEP integration |
| Evidence tier | P2 plausibility-only |
| Validation scope | Local Betz-limit, rated-power, Rayleigh-AEP, and cut-in/rated/cut-out power-curve controls. Public wind-turbine tools and standards are boundary evidence only. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/wind_turbine_power_6d.py` and
`tests/test_wind_turbine_power_physics.py`. It is intentionally explicit about
what the case is not, because wind-power benchmarks are easy to overclaim as
SAM, OpenFAST/AeroDyn, IEC, manufacturer-curve, or site-resource validation.

## 2. Implemented Formula Class

The case uses a compact algebraic turbine surrogate:

```text
Cp_max  = (16/27) (1 - 1.5/(B lambda_opt)) (1 - 0.3 sigma)
P       = 0.5 rho pi R^2 V^3 Cp_max, for cut-in <= V <= V_rated
P       = P_rated, for V_rated < V <= V_cut_out
P       = 0, below cut-in and above cut-out
AEP     = 8760 [k int_{V_ci}^{V_rated} V^3 f_Rayleigh(V)dV
          + P_rated Pr(V_rated < V <= V_cut_out)]
```

The Rayleigh AEP is evaluated in closed form over the sub-rated cubic segment
plus the rated-power probability mass. Probability outside the operating
interval is not renormalized. `theta_0` is retained as a recorded geometry
coordinate but is neutral in this low-order response surface.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| NREL SAM Wind Power Performance Model manual: https://www.nrel.gov/docs/fy14osti/60570.pdf | Public context for wind power-curve/performance modelling and SAM-style wind model scope. | Calibration of this algebraic Betz/Rayleigh surrogate or proof of SAM-equivalent outputs. |
| NREL SAM wind power help: https://samrepo.nrelcloud.org/help/wind_power.html | Boundary evidence for weather-file / Weibull resource options, 8760 time-step outputs, farm losses, wake models, and power-curve workflows omitted here. | Evidence that this case performs single-year hourly SAM simulation. |
| NREL SAM wind resource help: https://samrepo.nrelcloud.org/help/wind_resource.html and CSV wind format help: https://samrepo.nrelcloud.org/help/weather_format_csv_wind.html | Boundary evidence for wind-resource files, measurement-height fields, direction/speed inputs, and resource workflows not represented by a single Rayleigh mean wind-speed coordinate. | Site-resource validation or observational-data assimilation. |
| NREL SAM wind losses help: https://samrepo.nrelcloud.org/help/wind_losses.html | Boundary evidence for wake losses, availability, electrical/parasitic losses, curtailment, degradation, high-wind hysteresis, and site-specific power-curve adjustments. | Any loss-accounting calibration for this no-loss compact case. |
| Current SAM help mirror: https://samrepo.nrelcloud.org/help/wind_power.html, https://samrepo.nrelcloud.org/help/wind_resource.html, https://samrepo.nrelcloud.org/help/weather_format_csv_wind.html, and https://samrepo.nrelcloud.org/help/wind_losses.html | Current official help-domain mirrors for the same boundary/source checks after SAM help URL migration. | Evidence that the benchmark case is SAM-equivalent or site calibrated. |
| NREL 5-MW reference turbine report: https://www.nrel.gov/docs/fy09osti/38060.pdf | Reference-turbine context for what a full turbine definition contains. | Calibration to the NREL 5-MW turbine, manufacturer power curve, controller, structural, or aeroelastic behavior. |
| NREL OpenFAST overview and docs: https://www.nrel.gov/wind/nwtc/openfast and https://openfast.readthedocs.io/ | Boundary evidence for aero-hydro-servo-elastic wind-turbine/plant simulation workflows. | OpenFAST-equivalent simulation or validation. |
| OpenFAST AeroDyn docs: https://openfast.readthedocs.io/en/dev/source/user/aerodyn/ | Boundary evidence for steady BEM wake/induction, dynamic inflow, unsteady aerodynamics, dynamic stall, InflowWind coupling, and AeroDyn driver/input/output workflows omitted here. | AeroDyn-equivalent aerodynamics or BEM calibration. |
| NREL WindWatts: https://www.nrel.gov/wind/windwatts | Boundary evidence for terrain, obstacles, observational data, and wind-rose/resource-assessment workflows. | Site-resource validated power production. |
| NREL WIND Toolkit: https://www.nrel.gov/grid/wind-toolkit | Boundary evidence for WRF-based 2 km / 5-minute wind-resource time series, wind speed/direction, temperature, pressure, density, and power fields. | Validation that this single-mean-speed Rayleigh surrogate reproduces high-resolution wind-resource time-series outputs. |
| IEC 61400-12-1:2022: https://webstore.iec.ch/en/publication/68499 | Boundary evidence for measured power-performance, derived energy production, and uncertainty-assessment workflows. | IEC certification, measured-power-curve calibration, wind-measurement-method, site-calibration, obstacle/terrain-assessment, or uncertainty-budget validation. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not SAM-equivalent, not
OpenFAST-equivalent, not AeroDyn-equivalent, not WindWatts-equivalent, and not
WIND-Toolkit-equivalent, and not IEC power-performance validated. It is not calibrated to the NREL 5-MW
reference turbine, a manufacturer power curve, a measured power-performance
campaign, wind-farm production, or site-resource observations.

It omits blade-element airfoil tables, Prandtl BEM tip-loss iteration,
induction factors, drivetrain/generator efficiency, air-density correction,
wind shear, turbulence, wind direction, wind roses, terrain/roughness/obstacle
effects, hub-height extrapolation/interpolation, site-specific Weibull fitting,
WIND Toolkit WRF-based 2 km / 5-minute resource fields,
wake/farm losses, availability and curtailment losses, high-wind hysteresis,
degradation, Region-2.5 transition, detailed pitch/torque controllers,
structural loads, fatigue, aeroelastic simulation, AeroDyn driver/input/output
contracts, IEC measurement-sector filtering, site calibration, met-mast/lidar
measurement uncertainty, air-density normalization, measured power-curve
binning, derived-energy-production reporting, and combined uncertainty
assessment.

These source anchors support local Betz/Rayleigh plausibility and honest
production-tool boundaries. They do not support a strong external
case-validation claim unless a future pass adds a registered validation-note
folder with case-specific SAM/OpenFAST/BEM, manufacturer, wind-resource, or IEC
measurement comparisons.
