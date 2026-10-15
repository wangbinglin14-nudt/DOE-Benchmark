"""
Case M03: Photovoltaic system annual energy yield
==================================================

Engineering background
----------------------
The annual energy yield of a photovoltaic (PV) system is the central
metric for investment decisions.  It is jointly determined by:
  · Solar radiation resource (depends on geographic location and climate)
  · PV module parameters (efficiency, temperature coefficient)
  · System installation parameters (tilt angle, orientation)
  · System losses (dust, cabling, inverter)

This case uses a simplified PVWatts-style algebraic surrogate
(parameterised, independent of pvlib, SAM, or PVWatts SSC modules) to compute
specific annual yield per kWp installed capacity. It is intended for DOE trend
testing, not for site-specific bankable PV assessment. Current official NREL
documentation also makes clear that PVWatts V8 API, the PVWatts Calculator,
and SAM PVWatts share the same underlying `pvwattsv8` model and weather-data
path; this benchmark does not emulate that shared hourly toolchain. The
current official PVWatts Version 8 calculator page records Version 8.5.2
(released 2025-09-25), SSC version 275, SSC `pvwattsv8`, NSRDB 2020 TMY
inputs, bifacial/albedo/monthly-loss features, and explicit cautions about
site-specific uncertainty; those are boundary sources, not validation data for
this compact response. The current official PVWatts API endpoint also records
that `developer.nrel.gov` was retired on 2026-05-29 in favor of
`developer.nlr.gov`, so both legacy and current URLs are retained as traceable
source anchors. Its current V8 input/output contract also includes PSM V3
GOES/Himawari `tmy-2020`, PSM V4 Meteosat Prime Meridian `tmy-2022`, PSM V4
Polar `tmy-2023`, monthly or hourly `timeframe` selection,
`weather_data_source`, and hourly AC/POA/DC/ambient-temperature/
cell-temperature traces, none of which are emitted by this annual algebraic
benchmark case.

Physics model
-------------
The implementation does not run hourly weather, NSRDB/TMY weather-file lookup,
PVWatts V8 API/SSC, PVWatts Calculator, SAM PVWatts, SAM detailed PV,
solar-position, Perez/Hay-Davies/Reindl transposition, incident-angle modifier,
spectral, snow, soiling, monthly loss, albedo, bifacial, single-diode,
DC/AC sizing, inverter, inverter clipping, degradation, battery, PVPMC/Sandia
performance-index or uncertainty workflow, or grid-curtailment models. The current SAM detailed-photovoltaic workflow also
includes weather-file choice, module and inverter model selection, explicit
system-design choices, shading/layout losses, snow-depth-dependent shading,
and loss-page accounting that are intentionally outside this benchmark. The
current PVWatts V8 calculator/API path also includes pvwattsv8 SSC execution,
NSRDB-backed weather-data selection, module/array/GCR/DC-AC/inverter-efficiency
inputs, V8 module/inverter/thermal-effect model updates, NSRDB 2020 TMY
weather updates for covered locations, PSM V4 Meteosat/Polar TMY dataset
selection, monthly/hourly timeframes, weather-data source reporting, and
hourly AC/POA/DC/ambient-temperature/cell-temperature outputs that are not
replicated here. Instead it uses an annual irradiance
transposition factor, a performance ratio, and a temperature-coefficient
correction. Current pvlib documentation is used only to sharpen omitted
workflow boundaries: pvlib exposes PVWatts DC, inverter, and loss functions
and ModelChain can assemble solar-position, POA irradiance, AOI, spectral,
temperature, DC, AC, and loss models; this benchmark does not call those
functions or reproduce that chained hourly workflow. PVPMC annual-yield and
performance-ratio pages are used only to
anchor metric definitions (kWh/kWp annual yield and PR-style loss factor
semantics), not to validate this fixed-parameter annual surrogate against
measured plant data or a contractual expected-energy model.

Specific annual yield (per kWp installed capacity, industry-standard metric,
IEC 61724):
    E_annual [kWh/kWp] = G_eff [kWh/m²] · T_corr · PR / G_STC

where:
  · G_eff = GHI · f_tilt(β) · f_azimuth(γ): tilted-plane annual irradiation
  · f_tilt(β): Liu-Jordan-inspired component-weighted annual tilt approximation
  · f_azimuth = cos²(γ_dev): cosine correction for azimuth deviation
  · T_corr = 1 + (β_T/100) · (T_cell − 25 °C): temperature correction
  · T_cell ≈ 25 + [20 − 60·(η_STC − 0.20)]·G_eff/1400: empirical annual-average operating temperature
  · G_STC = 1 kW/m²: STC reference irradiance
  · PR: system performance ratio (inverter, cabling, dust, availability, etc.)

Note: η_STC cancels out in the per-kWp normalisation (Pmax_STC = η·A·G_STC);
its direct effect on specific yield appears only through temperature coupling
(η↑ → T_cell↓), hence it is a second-order effect.

Input variables
---------------
GHI_annual  Annual horizontal irradiation  [800, 2200]  kWh/m²  Location indicator
tilt        Module tilt angle               [0, 60]      deg     Installation design
azimuth_dev Azimuth deviation (from south)  [−30, 30]    deg     Installation deviation
eta_STC     STC module efficiency           [0.14, 0.25] -       Module grade
beta_T      Power temperature coefficient   [−0.5, −0.2] %/°C    Module thermal property
PR          Performance ratio (system loss) [0.70, 0.90] -       System quality

Output responses
----------------
E_annual    Annual yield (per kWp installed)  kWh/kWp  Primary output
CF          Capacity factor (annual mean)     -        Resource utilisation
T_cell_avg  Annual average operating temperature  °C   Module thermal indicator

The benchmark stores CF as a dimensionless fraction
``CF = E_annual / 8760``.  This is deliberately not the percent-form
``capacity_factor`` field returned by the current PVWatts V8 API.

DOE validation value
--------------------
VS-C (surrogate accuracy):
    E_annual has a clear optimum w.r.t. tilt (optimal tilt ≈ latitude),
    is nearly linear w.r.t. GHI, and exhibits multiplicative GHI × PR and
    weak eta_STC × temperature-coefficient interactions.

VS-H (robust Taguchi):
    Inter-annual GHI fluctuation and module temperature-coefficient
    uncertainty are noise factors; tilt, azimuth, module grade, and PR are
    design/control factors for robust-yield trend testing.

References
----------
1. Duffie, J. A. & Beckman, W. A. (2013). Solar Engineering of Thermal
   Processes, 4th ed. Wiley.
2. IEC 61724 (2021). Photovoltaic System Performance. IEC.
3. Dobos, A. P. (2014). PVWatts Version 5 Manual. NREL/TP-6A20-62641.
   https://www.nrel.gov/docs/fy14osti/62641.pdf
4. NREL PVWatts Calculator.
   https://pvwatts.nrel.gov/
5. NREL PVWatts V8 API documentation.
   https://developer.nrel.gov/docs/solar/pvwatts/v8/
   https://developer.nlr.gov/docs/solar/pvwatts/v8/
   https://pvwatts.nrel.gov/version_8.php
6. NREL System Advisor Model documentation. PVWatts and detailed
   photovoltaic model help pages.
   https://samrepo.nrelcloud.org/help/pvwatts.html
   https://samrepo.nlr.gov/help/pvwatts.html
   https://samrepo.nrelcloud.org/help/pvwatts_system_design.html
   https://samrepo.nlr.gov/help/pvwatts_system_design.html
   https://samrepo.nrelcloud.org/help/detailed_photovoltaic_model.html
   https://samrepo.nlr.gov/help/detailed_photovoltaic_model.html
7. pvlib-python transposition model documentation.
   https://pvlib-python.readthedocs.io/en/stable/reference/irradiance/transposition.html
8. pvlib-python PVWatts and ModelChain documentation.
   https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html
   https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html
   https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_losses.html
   https://pvlib-python.readthedocs.io/en/stable/user_guide/modeling_topics/modelchain.html
9. Sandia PV Performance Modeling Collaborative. Performance Ratio.
   https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-ratio/
10. Sandia PV Performance Modeling Collaborative. Annual Yield.
   https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/annual-yield/
11. Sandia PV Performance Modeling Collaborative. Performance Index.
   https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-index/
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case


@register_case
class SolarPVYield6D(BaseCase):
    """Photovoltaic system annual energy yield (6D → 3D, parameterised physics model)."""

    meta = CaseMeta(
        case_id="solar_pv_yield_6d",
        name="PV system annual energy yield (solar resource + module parameters)",
        version="1.0.0",
        domain="multidisciplinary",
        tags=[
            "solar",
            "pv",
            "photovoltaic",
            "energy_yield",
            "6d",
            "renewable",
            "tilt_optimization",
        ],
        dim_input=6,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="GHI_annual",
                symbol="G_{HI}",
                description="Annual global horizontal irradiation",
                unit="kWh/m²",
                lower=800.0,
                upper=2200.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="tilt",
                symbol=r"\beta",
                description="Module tilt angle (relative to horizontal)",
                unit="deg",
                lower=0.0,
                upper=60.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="azimuth_dev",
                symbol=r"\gamma_{dev}",
                description="Azimuth deviation (0 = due south, positive = westward)",
                unit="deg",
                lower=-30.0,
                upper=30.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="eta_STC",
                symbol=r"\eta_{STC}",
                description="Module efficiency under standard test conditions (STC)",
                unit="-",
                lower=0.14,
                upper=0.25,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="beta_T",
                symbol=r"\beta_T",
                description="Power temperature coefficient (negative)",
                unit="%/°C",
                lower=-0.5,
                upper=-0.2,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="PR",
                symbol="PR",
                description="System performance ratio (incl. cabling losses, inverter efficiency, etc.)",
                unit="-",
                lower=0.70,
                upper=0.90,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="E_annual",
                symbol="E_{annual}",
                description="Specific annual yield (per kWp installed)",
                unit="kWh/kWp",
                is_primary=True,
                expected_range=(300.0, 2300.0),
            ),
            OutputVariable(
                name="CF",
                symbol="CF",
                description="Capacity factor (annual mean output / peak power)",
                unit="-",
                is_primary=False,
                expected_range=(0.035, 0.28),
            ),
            OutputVariable(
                name="T_cell_avg",
                symbol="T_{cell}",
                description="Annual average operating temperature",
                unit="°C",
                is_primary=False,
                expected_range=(20.0, 70.0),
            ),
        ],
        governing_equation=(
            "PV specific annual yield model (IEC 61724):\n"
            "  G_eff = GHI · f_tilt(β) · f_azimuth(γ_dev)\n"
            "  T_cell_avg ≈ 25 + [20 − 60·(η_STC − 0.20)]·G_eff/1400\n"
            "  T_corr = 1 + (β_T/100)·(T_cell − 25)  [β_T in %/°C]\n"
            "  E_annual [kWh/kWp] = G_eff · T_corr · PR / G_STC,  G_STC = 1 kW/m²"
        ),
        pde_type="algebraic",
        spatial_dim=0,
        is_steady=False,
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Duffie, J. A. & Beckman, W. A. (2013). Solar Engineering of Thermal Processes, 4th ed. Wiley.",
            "IEC 61724 (2021). Photovoltaic System Performance. IEC.",
            "Dobos, A. P. (2014). PVWatts Version 5 Manual. NREL/TP-6A20-62641.",
            "Dobos, A. P. (2014). PVWatts Version 5 Manual. https://www.nrel.gov/docs/fy14osti/62641.pdf",
            "NREL SAM PVWatts help. https://samrepo.nrelcloud.org/help/pvwatts.html",
            "NREL SAM PVWatts help, current NLR domain. https://samrepo.nlr.gov/help/pvwatts.html",
            "NREL SAM PVWatts system-design help. "
            "https://samrepo.nrelcloud.org/help/pvwatts_system_design.html",
            "NREL SAM PVWatts system-design help, current NLR domain. "
            "https://samrepo.nlr.gov/help/pvwatts_system_design.html",
            "NREL SAM Detailed Photovoltaic Model help. https://samrepo.nrelcloud.org/help/detailed_photovoltaic_model.html",
            "NREL SAM Detailed Photovoltaic Model help, current NLR domain. "
            "https://samrepo.nlr.gov/help/detailed_photovoltaic_model.html",
            "NREL PVWatts Calculator. https://pvwatts.nrel.gov/",
            "NLR PVWatts Calculator current domain. https://pvwatts.nlr.gov/",
            "NREL PVWatts V8 API documentation. "
            "https://developer.nrel.gov/docs/solar/pvwatts/v8/",
            "NLR PVWatts V8 API documentation current domain. "
            "https://developer.nlr.gov/docs/solar/pvwatts/v8/",
            "NREL/NLR PVWatts Version 8 calculator technical page. "
            "https://pvwatts.nrel.gov/version_8.php",
            "pvlib-python transposition models. https://pvlib-python.readthedocs.io/en/stable/reference/irradiance/transposition.html",
            "pvlib-python PVWatts DC model. "
            "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html",
            "pvlib-python PVWatts inverter model. "
            "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html",
            "pvlib-python PVWatts loss model. "
            "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_losses.html",
            "pvlib-python ModelChain documentation. "
            "https://pvlib-python.readthedocs.io/en/stable/user_guide/modeling_topics/modelchain.html",
            "Sandia PV Performance Modeling Collaborative, Performance Ratio. "
            "https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-ratio/",
            "Sandia PV Performance Modeling Collaborative, Annual Yield. "
            "https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/annual-yield/",
            "Sandia PV Performance Modeling Collaborative, Performance Index. "
            "https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-index/",
        ],
        solver_library="numpy (parameterised physics model)",
        solver_details=(
            "Liu-Jordan-inspired annual irradiance transposition approximation + "
            "empirical temperature correction + performance-ratio loss model; "
            "named _validate_physical_inputs private physical-input evaluation "
            "rejects malformed arrays and GHI_annual/tilt/azimuth_dev/eta_STC/"
            "beta_T/PR values outside the declared annual-yield surrogate "
            "contract before PVWatts-style response assembly, including tilt "
            "outside [0, 60] deg or azimuth deviation outside [-30, 30] deg; "
            "no hourly weather, NSRDB/TMY weather-file lookup, PVWatts V8 "
            "API/SSC, PVWatts Calculator, SAM PVWatts, SAM detailed PV, "
            "the shared PVWatts V8/pvwattsv8 hourly toolchain, "
            "PVWatts module_type/array_type/GCR/DC-AC-ratio/inverter-efficiency "
            "inputs, PVWatts V8 module/inverter/thermal-effect algorithm "
            "updates, NSRDB 2020 TMY weather updates, PSM V4 Meteosat/Polar "
            "TMY dataset selection, solar-position, "
            "PVWatts V8 timeframe=monthly/hourly selection, weather_data_source, "
            "pvlib/Perez/Hay-Davies/Reindl, "
            "IAM/spectral/snow/soiling/monthly-loss, albedo, bifacial, SDM, "
            "DC/AC sizing, inverter model, inverter clipping, shading/layout, "
            "weather-file/module/inverter/system-design workflow, weather-data "
            "selection, hourly AC/POA/DC/ambient-temperature/module-temperature "
            "outputs, snow-depth "
            "shading, inverter undersizing warnings, PVWatts Version 8.5.2 "
            "calculator release-date tracking, SSC version 275 tracking, "
            "PVWatts uncertainty cautions, degradation, "
            "battery, grid-curtailment, monthly/hourly output simulation, SAM, "
            "pvlib.pvsystem.pvwatts_dc, pvlib.inverter.pvwatts, "
            "pvlib.pvsystem.pvwatts_losses, pvlib ModelChain solar-position/"
            "airmass/AOI/spectral/temperature/DC/AC/loss workflow, "
            "PVPMC/Sandia performance-index or uncertainty workflow, "
            "or PVWatts SSC simulation; PVPMC annual-yield and "
            "performance-ratio references anchor metric definitions only, "
            "not measured-plant monitoring, expected-energy, or contractual "
            "performance-index validation; CF is archived as a dimensionless "
            "fraction E_annual/8760 rather than the percent-form "
            "capacity_factor field reported by PVWatts V8 API examples; "
            "non-finite, non-physical, or out-of-contract E_annual/CF/"
            "T_cell_avg responses are rejected before dispatch can archive a row"
        ),
        fidelity_levels=["default"],
        typical_eval_time_ms={"default": 0.5},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=False,
        response_topology=["smooth_unimodal", "strong_interaction"],
        # Effective dimension and important_variables follow the project ST > 0.05
        # convention for E_annual. eta_STC and beta_T still affect the formula through
        # temperature coupling; eta_STC is weak enough to stay out of the important
        # variable list but remains in interaction_pairs where it couples to T_corr.
        effective_dim=4,
        important_variables=["GHI_annual", "tilt", "azimuth_dev", "PR"],
        interaction_pairs=[
            ("GHI_annual", "tilt"),
            ("GHI_annual", "azimuth_dev"),
            ("GHI_annual", "eta_STC"),
            ("GHI_annual", "beta_T"),
            ("GHI_annual", "PR"),
            ("tilt", "azimuth_dev"),
            ("tilt", "eta_STC"),
            ("tilt", "beta_T"),
            ("tilt", "PR"),
            ("azimuth_dev", "eta_STC"),
            ("azimuth_dev", "beta_T"),
            ("azimuth_dev", "PR"),
            ("eta_STC", "beta_T"),
            ("eta_STC", "PR"),
            ("beta_T", "PR"),
        ],
        has_failure_boundary=False,
        sobol_S1={
            "GHI_annual": 0.775,
            "tilt": 0.069,
            "azimuth_dev": 0.072,
            "eta_STC": 0.001,
            "beta_T": 0.005,
            "PR": 0.063,
        },
        sobol_ST={
            "GHI_annual": 0.787,
            "tilt": 0.074,
            "azimuth_dev": 0.077,
            "eta_STC": 0.001,
            "beta_T": 0.007,
            "PR": 0.070,
        },
        sobol_computation="salib_saltelli_n2048_seed42_E_annual_output",
        applicable_scenarios=["VS-C", "VS-F", "VS-H"],
        scenario_notes={
            "VS-C": "E_annual has an optimum w.r.t. tilt, is nearly linear w.r.t. GHI; medium-difficulty surrogate test.",
            "VS-F": (
                "Public PVWatts/SAM/pvlib references support the PV yield context "
                "and model-boundary wording; this compact annual algebraic "
                "surrogate does not call the current PVWatts V8 API/SSC model, "
                "PVWatts Calculator, PVWatts V5, SAM PVWatts, SAM detailed "
                "PV, NSRDB/TMY weather-file lookup, PVWatts module/array/GCR/"
                "DC-AC-ratio/inverter-efficiency inputs, albedo, bifaciality, "
                "PVWatts V8 detailed module/inverter/thermal-effect updates, "
                "PVWatts V8 timeframe/weather_data_source/hourly AC/POA/DC/"
                "temperature output traces, "
                "PVWatts Version 8.5.2 calculator release-date tracking, "
                "SSC version 275 tracking, PSM V4 Meteosat/Polar TMY "
                "dataset selection, PVWatts uncertainty cautions, "
                "monthly soiling/loss inputs, monthly/hourly PVWatts outputs, "
                "pvlib/Perez/Hay-Davies/Reindl, pvlib PVWatts DC/inverter/"
                "loss functions, pvlib ModelChain, NOCT/Fuentes thermal, "
                "single-diode, DC/AC sizing, inverter clipping, IAM, spectral, "
                "soiling, snow, shading, degradation, bifacial, battery, "
                "grid-curtailment, or bankable site-assessment model."
            ),
            "VS-H": (
                "GHI_annual and beta_T are noise factors; tilt, azimuth_dev, "
                "eta_STC, and PR are control/design factors for robust-yield "
                "trend testing."
            ),
        },
        recommended_n_train={"default": 50},
        n_saturation=200,
        noise_factor_candidates=["GHI_annual", "beta_T"],
        control_factor_candidates=["tilt", "azimuth_dev", "eta_STC", "PR"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "solar_pv_yield_6d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        GHI = X_phys[:, 0]
        tilt = X_phys[:, 1]
        az_dev = X_phys[:, 2]
        eta = X_phys[:, 3]
        beta_T = X_phys[:, 4]
        PR = X_phys[:, 5]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(GHI < 800.0 - tol)
            or np.any(GHI > 2200.0 + tol)
            or np.any(tilt < 0.0 - tol)
            or np.any(tilt > 60.0 + tol)
            or np.any(az_dev < -30.0 - tol)
            or np.any(az_dev > 30.0 + tol)
            or np.any(eta < 0.14 - tol)
            or np.any(eta > 0.25 + tol)
            or np.any(beta_T < -0.5 - tol)
            or np.any(beta_T > -0.2 + tol)
            or np.any(PR < 0.70 - tol)
            or np.any(PR > 0.90 + tol)
        ):
            raise ValueError(
                "solar_pv_yield_6d physical inputs must stay inside the declared "
                "GHI_annual/tilt/azimuth_dev/eta_STC/beta_T/PR benchmark domain "
                "before PVWatts-style annual-yield response assembly"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                response = self._pv_yield(
                    GHI=float(X_phys[i, 0]),
                    tilt=float(X_phys[i, 1]),
                    az_dev=float(X_phys[i, 2]),
                    eta=float(X_phys[i, 3]),
                    beta_T=float(X_phys[i, 4]),
                    PR=float(X_phys[i, 5]),
                )
                if (
                    response.shape != (3,)
                    or not np.all(np.isfinite(response))
                    or response[0] <= 0.0
                    or not (0.0 < response[1] < 1.0)
                    or not (0.0 < response[2] < 100.0)
                ):
                    raise ValueError(
                        "non-finite or non-physical PV response "
                        f"E_annual/CF/T_cell_avg={np.array2string(response, precision=6, separator=', ')}"
                    )
                output_ranges = np.asarray(
                    [output.expected_range for output in self.meta.output_variables],
                    dtype=float,
                )
                below = response < (output_ranges[:, 0] - 1e-10)
                above = response > (output_ranges[:, 1] + 1e-10)
                if np.any(below) or np.any(above):
                    raise ValueError(
                        "out-of-contract PV E_annual/CF/T_cell_avg response "
                        "outside the declared output ranges"
                    )
                Y[i] = response
            except Exception as exc:
                raise RuntimeError(
                    "solar_pv_yield_6d refusing silent NaN after PV yield "
                    f"evaluation failure for GHI_annual={X_phys[i, 0]!r}, "
                    f"tilt={X_phys[i, 1]!r}, azimuth_dev={X_phys[i, 2]!r}, "
                    f"eta_STC={X_phys[i, 3]!r}, beta_T={X_phys[i, 4]!r}, "
                    f"PR={X_phys[i, 5]!r}"
                ) from exc
        return Y

    @staticmethod
    def _pv_yield(GHI, tilt, az_dev, eta, beta_T, PR):
        """Parameterised PV specific annual yield model.

        Tilt correction: Liu-Jordan-inspired component-weighted annual approximation
            G_eff = GHI · f_tilt(β) · f_azimuth(γ_dev)

        Specific annual yield (per kWp nominal):
            E_annual [kWh/kWp] = G_eff · T_corr · PR / G_STC

        η_STC cancels in the per-kWp normalisation:
            Pmax_STC = A · η_STC · G_STC  →  E_annual/Pmax is independent of η.
            The influence of η_STC appears only via temperature coupling
            (higher efficiency → less waste heat → lower T_cell).
        """
        if (
            not np.all(np.isfinite([GHI, tilt, az_dev, eta, beta_T, PR]))
            or GHI <= 0.0
            or not (0.0 <= tilt <= 60.0)
            or not (-30.0 <= az_dev <= 30.0)
            or eta <= 0.0
            or not (-5.0 < beta_T < 0.0)
            or not (0.0 < PR <= 1.0)
        ):
            raise ValueError(
                "invalid solar_pv_yield_6d input outside the declared "
                "PVWatts-style annual-yield surrogate contract"
            )
        tilt_rad = np.deg2rad(tilt)
        az_rad = np.deg2rad(az_dev)

        # ── Tilt correction (Liu–Jordan isotropic sky) ───────────────────────
        # Assume reference latitude 35°, ground reflectance ρ = 0.2, beam/diffuse/ground weights 0.8/0.15/0.05
        Rb_ratio = 1.2
        f_beam = np.cos(tilt_rad - np.deg2rad(35.0)) ** 2 * Rb_ratio
        f_diff = (1.0 + np.cos(tilt_rad)) / 2.0
        f_ground = 0.2 * (1.0 - np.cos(tilt_rad)) / 2.0
        f_tilt = 0.8 * f_beam + 0.15 * f_diff + 0.05 * f_ground

        # ── Azimuth deviation: cosine-squared approximation ───────────────────────────
        f_azimuth = np.cos(az_rad) ** 2

        # ── Tilted-plane annual irradiation (kWh/m²) ─────────────────────────────
        G_eff = GHI * f_tilt * f_azimuth
        if not np.isfinite(G_eff) or G_eff <= 0.0:
            raise ValueError("non-physical PV effective annual irradiation")

        # ── Annual average module operating temperature (empirical; higher-efficiency modules run cooler) ──
        # Simplified NOCT: T_cell ≈ 25 + (20 − 60·(η − 0.2))·G_eff/1400
        # Higher η → less waste heat → annual avg temperature drops ~(−6 °C) relative to η = 0.20 baseline
        T_rise_ref = 20.0 - 60.0 * (eta - 0.20)
        T_cell_avg = 25.0 + T_rise_ref * G_eff / 1400.0
        if not np.isfinite(T_cell_avg) or not (0.0 < T_cell_avg < 100.0):
            raise ValueError("non-physical PV annual-average cell temperature")

        # ── Temperature correction (β_T in %/°C) ────────────────────────────────
        T_corr = 1.0 + beta_T / 100.0 * (T_cell_avg - 25.0)
        if not np.isfinite(T_corr) or T_corr <= 0.0:
            raise ValueError("non-physical PV temperature correction")

        # ── Specific annual yield (kWh/kWp) ───────────────────────────────────────
        G_STC = 1.0  # kW/m²
        E_annual = G_eff * T_corr * PR / G_STC
        if not np.isfinite(E_annual) or E_annual <= 0.0:
            raise ValueError("non-physical PV specific annual yield")

        # ── Capacity factor (annual mean output / peak power, dimensionless) ─────────
        CF = E_annual / 8760.0
        if not np.isfinite(CF) or not (0.0 < CF < 1.0):
            raise ValueError("non-physical PV capacity factor")

        return np.array([E_annual, CF, T_cell_avg])
