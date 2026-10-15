# Cross-Validation: `solar_pv_yield_6d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `solar_pv_yield_6d` |
| Implemented solver | Fixed-parameter PVWatts-style annual algebraic surrogate with Liu-Jordan-inspired transposition |
| Evidence tier | P2 plausibility-only |
| Validation scope | Annual-yield formula identities, PV performance trends, metric definitions, tool-boundary, and omitted-workflow controls. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/solar_pv_yield_6d.py` and
`tests/test_solar_pv_yield_physics.py`. It clarifies that the case is a
fixed-parameter annual PV-yield stressor, not a bankable site-assessment or
measured-plant validation workflow.

## 2. Implemented Formula Class

The case uses a compact annual per-kWp response:

```text
G_eff    = GHI_annual * f_tilt(tilt) * f_azimuth(azimuth_dev)
T_cell   = 25 + (20 - 60*(eta_STC - 0.20)) * G_eff/1400
T_corr   = 1 + (beta_T/100) * (T_cell - 25)
E_annual = G_eff * T_corr * PR
CF       = E_annual / 8760
```

The stored `CF` is a dimensionless fraction, not the percent-form PVWatts V8
API `capacity_factor` output. `eta_STC` affects the per-kWp yield only through
the simplified temperature coupling.

The targeted tests also pin the annual azimuth-deviation identity: because the
registered surrogate uses `cos^2(azimuth_dev)`, eastward and westward
deviations of equal magnitude produce the same annual `E_annual`, `CF`, and
`T_cell_avg`. This is a formula identity of the compact annual surrogate, not
a claim about hourly or site-specific east/west PV asymmetry.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| NREL PVWatts V5 manual: https://www.nrel.gov/docs/fy14osti/62641.pdf | PVWatts-style annual-energy and performance-ratio context. | Validation of the registered annual algebraic response. |
| NREL PVWatts Calculator: https://pvwatts.nrel.gov/ | Boundary evidence for the public calculator workflow. | Evidence that this case calls or validates the calculator. |
| NREL PVWatts V8 API: https://developer.nrel.gov/docs/solar/pvwatts/v8/ | Boundary evidence for API inputs/outputs, NSRDB/TMY plus PSM V3/V4 weather datasets, `timeframe=monthly\|hourly`, `weather_data_source`, hourly AC/POA/DC/ambient-temperature/cell-temperature traces, and percent-form capacity factor. | PVWatts V8/API/SSC validation. |
| PVWatts Version 8 calculator page: https://pvwatts.nrel.gov/version_8.php | Boundary evidence for Version 8.5.2, 2025-09-25 release date, SSC version 275, SSC `pvwattsv8`, NSRDB 2020 TMY inputs, bifacial/albedo/monthly-loss features, and PVWatts uncertainty cautions. | Evidence that this fixed annual surrogate reproduces PVWatts V8 outputs or site-specific expected energy. |
| SAM PVWatts help: https://samrepo.nrelcloud.org/help/pvwatts.html and https://samrepo.nrelcloud.org/help/pvwatts.html | Boundary evidence for SAM PVWatts workflow, weather-file choices, and current/legacy help-page traceability. | SAM PVWatts validation of this case. |
| SAM PVWatts system design: https://samrepo.nrelcloud.org/help/pvwatts_system_design.html and https://samrepo.nrelcloud.org/help/pvwatts_system_design.html | Boundary evidence for DC/AC sizing, inverter losses, and system-design options. | Validation of inverter clipping or system-sizing workflows. |
| SAM detailed photovoltaic model: https://samrepo.nrelcloud.org/help/detailed_photovoltaic_model.html and https://samrepo.nrelcloud.org/help/detailed_photovoltaic_model.html | Boundary evidence for detailed module, inverter, bifacial, snow, shading, and time-series workflows. | SAM Detailed PV validation. |
| pvlib irradiance transposition docs: https://pvlib-python.readthedocs.io/en/stable/reference/irradiance/transposition.html | Boundary evidence for Perez, Hay-Davies, Reindl, and full plane-of-array transposition methods. | pvlib/Perez/Hay-Davies/Reindl validation. |
| pvlib PVWatts and ModelChain docs: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html, https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_losses.html, and https://pvlib-python.readthedocs.io/en/stable/user_guide/modeling_topics/modelchain.html | Boundary evidence that pvlib can assemble PVWatts-style DC, inverter, loss, solar-position, POA, AOI, spectral, temperature, DC, AC, and loss workflows. | pvlib validation, PVWatts-function equivalence, or evidence that this fixed annual surrogate calls pvlib. |
| PVPMC performance ratio: https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-ratio/ | Metric-definition anchor for PR. | Measured-plant monitoring or contractual PR validation. |
| PVPMC annual yield: https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/annual-yield/ | Metric-definition anchor for kWh/kWp annual yield. | Expected-energy or uncertainty analysis. |
| PVPMC performance index: https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-index/ | Boundary evidence for performance-index workflows. | Contractual performance-index validation. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not PVWatts V8/API/SSC validated,
not PVWatts Calculator validated, not SAM PVWatts validated, not SAM Detailed
PV validated, not pvlib/Perez/Hay-Davies/Reindl validated, not hourly-weather
validated, not NSRDB/TMY weather-file based, not inverter-clipping validated,
not bifacial validated, not battery/grid validated, not single-diode
validated, not bankable site-assessment validated, not project-finance PV
model output, not measured-plant monitoring, not expected-energy analysis, not
uncertainty analysis, not pvlib PVWatts/ModelChain validated, and not
contractual performance-index reporting.

It omits hourly weather, solar-position time series, module and inverter model
selection, DC/AC sizing, inverter clipping, IAM, spectral losses, soiling,
snow, shading/layout losses, albedo, bifaciality, degradation, grid
curtailment, battery/grid workflows, PVWatts monthly/hourly outputs,
PSM V3/V4 weather-dataset selection, `timeframe=monthly|hourly` selection,
`weather_data_source`, hourly AC/POA/DC/ambient-temperature/cell-temperature
traces, SAM
detailed PV workflows, pvlib POA transposition, pvlib PVWatts DC/inverter/loss
functions, pvlib ModelChain, measured production,
expected-energy baselines, uncertainty budgets, and performance-index
contracts.

These anchors support annual-yield plausibility and honest toolchain
boundaries. They do not support a strong external-validation claim unless a
future registered validation-note folder adds PVWatts/SAM/pvlib or measured
plant reference comparisons for the frozen response surface.
