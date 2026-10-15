"""Physics regression tests for ``wind_turbine_power_6d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammainc, gamma

import doe_testbed.cases.multidisciplinary.wind_turbine_power_6d as wind_case_module
from doe_testbed.cases.multidisciplinary.wind_turbine_power_6d import (
    WindTurbinePower6D,
)


def _domain_corners(case: WindTurbinePower6D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_declared_output_ranges_cover_domain_corners():
    case = WindTurbinePower6D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_validation_material_declares_non_bem_surrogate_scope():
    case = WindTurbinePower6D()
    validation_text = Path(
        "tests/_cross_validation/wind_turbine_power_6d.py"
    ).read_text(encoding="utf-8")
    source_text = Path(
        "doe_testbed/cases/multidisciplinary/wind_turbine_power_6d.py"
    ).read_text(encoding="utf-8")

    normalized_validation = " ".join(validation_text.split())
    assert "low-order Betz surrogate" in validation_text
    assert "not a full BEM induction-factor" in normalized_validation
    assert "Prandtl tip-loss validation" in normalized_validation
    assert "bemt" not in case.meta.tags
    assert "betz_surrogate" in case.meta.tags
    assert "not a full BEM induction-factor iteration" in source_text
    assert "V > V_cut-out" in source_text
    assert "P = 0 above cut-out" in case.meta.governing_equation
    assert "theta_0" in source_text
    assert "neutral in this low-order surrogate" in source_text
    assert "no Prandtl BEM tip-loss calculation" in case.meta.solver_details
    assert "BEM induction-factor iteration" in case.meta.solver_details
    assert "closed-form" in case.meta.solver_details
    assert hasattr(case, "_validate_physical_inputs")
    assert "named _validate_physical_inputs private physical-input evaluation rejects" in case.meta.solver_details
    assert "before Betz/Rayleigh response assembly" in case.meta.solver_details
    assert "out-of-contract Cp_max/P_rated/AEP responses are rejected" in case.meta.solver_details
    assert "Target accuracy" not in validation_text
    assert "external validation" not in validation_text


def test_metadata_declares_omitted_wind_turbine_physics_scope():
    case = WindTurbinePower6D()
    validation_text = Path(
        "tests/_cross_validation/wind_turbine_power_6d.py"
    ).read_text(encoding="utf-8")
    source_text = Path(
        "doe_testbed/cases/multidisciplinary/wind_turbine_power_6d.py"
    ).read_text(encoding="utf-8")
    combined = "\n".join(
        [
            case.meta.solver_details,
            case.meta.scenario_notes["VS-F"],
            validation_text,
            source_text,
        ]
    ).lower()

    for phrase in (
        "airfoil-table",
        "drivetrain/generator efficiency",
        "air-density correction",
        "wind shear",
        "turbulence",
        "wind direction",
        "wind-rose",
        "terrain/roughness",
        "obstacle",
        "hub-height extrapolation",
        "hub-height interpolation between measured wind heights",
        "nrel wind toolkit wrf-based 2 km/5-minute wind-resource time",
        "multi-height wind speed/direction",
        "site-specific weibull",
        "wake/farm",
        "sam wind-resource files",
        "sam csv/srw time-series resource simulation",
        "internal/external/future wake",
        "turbine/balance-of-plant availability",
        "grid availability",
        "electrical/parasitic losses",
        "high-wind hysteresis",
        "environmental degradation",
        "curtailment/operational-strategy losses",
        "observational-data assimilation",
        "windwatts-style resource assessment",
        "iec 61400-12-1:2022 power-performance",
        "iec wind-measurement-method standards",
        "iec site-calibration standards",
        "iec obstacle/terrain-assessment standards",
        "derived-energy-production",
        "combined-uncertainty",
        "measurement-sector filtering",
        "site calibration",
        "met-mast/lidar measurement uncertainty",
        "air-density normalisation",
        "power-curve binning",
        "test/certification",
        "system-loss/availability modelling",
        "curtailment",
        "degradation",
        "site-specific power-curve adjustment",
        "region-2.5",
        "pitch/torque controller",
        "openfast/aerodyn steady bem wake",
        "dbemt dynamic-inflow",
        "aerodyn unsteady-aerodynamics/dynamic-stall",
        "inflowwind time-series",
        "aerodyn driver/input/output",
        "structural loads",
        "fatigue",
        "fast/openfast",
        "measured-power-curve calibration",
        "manufacturer power-curve",
    ):
        assert phrase in combined


def test_public_nrel_sources_and_scope_are_declared():
    case = WindTurbinePower6D()
    refs = "\n".join(case.meta.physics_references)
    scenario = case.meta.scenario_notes["VS-F"]

    assert "https://www.nrel.gov/docs/fy14osti/60570.pdf" in refs
    assert "https://samrepo.nrelcloud.org/help/wind_power.html" in refs
    assert "https://samrepo.nrelcloud.org/help/wind_losses.html" in refs
    assert "https://samrepo.nrelcloud.org/help/wind_resource.html" in refs
    assert "https://samrepo.nrelcloud.org/help/weather_format_csv_wind.html" in refs
    assert "https://samrepo.nrelcloud.org/help/wind_power.html" in refs
    assert "https://samrepo.nrelcloud.org/help/wind_losses.html" in refs
    assert "https://samrepo.nrelcloud.org/help/wind_resource.html" in refs
    assert "https://samrepo.nrelcloud.org/help/weather_format_csv_wind.html" in refs
    assert "https://www.nrel.gov/docs/fy09osti/38060.pdf" in refs
    assert "not calibrated to the NREL 5-MW turbine" in scenario
    assert "manufacturer power-curve data" in scenario
    assert "SAM wind time-series/resource-file or single-year hourly simulations" in scenario
    assert "single-year hourly simulations" in scenario
    assert "SAM hub-height/shear/density/wake/loss workflows" in scenario
    assert "hub-height interpolation between measured wind speeds" in refs
    assert "ideal-gas air-density adjustment from time-series temperature and pressure" in refs
    assert "SAM wind-farm loss-factor or system-availability/operating-loss workflows" in scenario
    assert "site-specific wind-resource measurements" in scenario
    assert "full BEM/FAST/OpenFAST model" in scenario
    assert "Distributed Wind Resource Assessment Framework" in refs
    assert "https://docs.nrel.gov/docs/fy19osti/72523.pdf" in refs
    assert "OpenFAST documentation" in refs
    assert "https://openfast.readthedocs.io/" in refs
    assert "https://www.nrel.gov/wind/nwtc/openfast" in refs
    assert "regression and unit testing" in refs
    assert "OpenFAST AeroDyn documentation" in refs
    assert "https://openfast.readthedocs.io/en/dev/source/user/aerodyn/" in refs
    assert "steady BEM wake/induction" in refs
    assert "DBEMT dynamic-inflow" in refs
    assert "dynamic-stall models" in refs
    assert "InflowWind" in refs
    assert "AeroDyn input/output/driver workflows" in refs
    assert "https://www.nrel.gov/wind/windwatts" in refs
    assert "https://www.nrel.gov/grid/wind-toolkit" in refs
    assert "NREL WIND Toolkit" in refs
    assert "2 km / 5-minute" in refs
    assert "wind-direction" in refs
    assert "temperature, pressure, density, and power fields" in refs
    assert "WindWatts" in refs
    assert "WindWatts resource estimates" in scenario
    assert "WIND-Toolkit" in scenario
    assert "NREL WIND Toolkit WRF-based 2 km/5-minute wind-resource fields" in scenario
    assert "IEC 61400-12-1:2022" in refs
    assert "https://webstore.iec.ch/en/publication/68499" in refs
    assert "moves wind-speed measurement" in refs
    assert "obstacle/terrain assessment material to separate standards" in refs
    assert "IEC measured-power-curve" in scenario
    assert "derived-energy-production" in scenario
    assert "wind-measurement-method" in scenario
    assert "site-calibration" in scenario
    assert "obstacle/terrain-assessment" in scenario
    assert case._B == 3
    assert case._V_CUT_IN == 3.0
    assert case._V_CUT_OUT == 25.0


def test_solver_details_include_current_sam_losses_boundary_scope():
    case = WindTurbinePower6D()
    details = case.meta.solver_details.lower()

    assert "internal/external/future wake accounting" in details
    assert "turbine/balance-of-plant availability-loss accounting" in details
    assert "grid availability" in details
    assert "electrical/parasitic losses" in details
    assert "high-wind hysteresis" in details
    assert "environmental degradation/shutdown losses" in details
    assert "curtailment/operational-strategy losses" in details


def test_openfast_is_boundary_evidence_not_calibration_claim():
    validation_text = Path(
        "tests/_cross_validation/wind_turbine_power_6d.py"
    ).read_text(encoding="utf-8")
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit_text.split())
    normalized_validation = " ".join(validation_text.split())
    row = next(
        line
        for line in audit_text.splitlines()
        if "`wind_turbine_power_6d`" in line and "OpenFAST" in line
    )

    assert "OpenFAST is used here only as boundary evidence" in validation_text
    assert "AeroDyn documentation is used here only as boundary evidence" in validation_text
    assert "steady BEM wake/induction" in validation_text
    assert "DBEMT dynamic inflow" in validation_text
    assert "dynamic stall" in validation_text
    assert "InflowWind time-series coupling" in validation_text
    assert "AeroDyn driver/input/output contracts" in validation_text
    assert "WindWatts/DWRA material is used only as boundary evidence" in validation_text
    assert "WIND Toolkit material is used only as boundary evidence" in validation_text
    assert "WRF-based 2 km / 5-minute" in validation_text
    assert "wind-speed, wind-direction, temperature, pressure, density, and power fields" in validation_text
    assert "not calibration or validation evidence for this single-mean-speed" in normalized_validation
    assert "SAM wind help is used here only as boundary evidence" in validation_text
    assert "IEC 61400-12-1:2022 material is used only as boundary evidence" in validation_text
    assert "single-year hourly production" in validation_text
    assert "wind-farm loss-factor" in validation_text
    assert "hub-height interpolation" in validation_text
    assert "ideal-gas density adjustment" in validation_text
    assert "site-specific power-curve-adjustment" in normalized_validation
    assert "regression/unit-tested" in validation_text
    assert "not calibration or validation evidence for this Betz/Rayleigh surrogate" in normalized_validation
    assert "Evidence tier remains P2" in row
    assert "SAM wind-resource files" in row
    assert "SAM CSV/SRW time-series" in row
    assert "OpenFAST" in row
    assert "WindWatts" in row
    assert "IEC 61400-12-1" in row
    assert "measurement-sector filtering" in row
    assert "site calibration" in row
    assert "power-curve binning" in row
    assert "derived energy-production reporting" in validation_text
    assert "combined uncertainty assessment" in validation_text
    assert "2022 edition replaces the older 2017" in validation_text
    assert "moves wind-speed measurement" in validation_text
    assert "site calibration, and obstacle/terrain assessment material to separate" in validation_text
    assert "not calibration evidence" in row
    assert "site-resource workflows" in row
    assert "OpenFAST/AeroDyn aerodynamic workflow boundary note" in normalized_audit
    assert "AeroDyn driver/input/output workflows" in normalized_audit


def test_current_official_source_recheck_keeps_wind_case_p2_boundary():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `wind_turbine_power_6d` 2026-06-08 official-source re-check" in audit_text
    assert "full SAM wind runs can use one-year hourly/subhourly wind-resource files" in audit_text
    assert "hub-height/wind-shear adjustment" in audit_text
    assert "air-density adjustment from temperature/pressure" in audit_text
    assert "wake-effect models" in audit_text
    assert "internal/external/future wake effects" in audit_text
    assert "turbine/balance-of-plant availability losses" in audit_text
    assert "coupled nonlinear aero-hydro-servo-elastic time-domain simulation" in audit_text
    assert "terrain inputs" in audit_text
    assert "wind-rose visualization" in audit_text
    assert "site-resource validated" in audit_text
    assert "Evidence tier remains P2 plausibility-only" in audit_text
    assert "No numerical response resweep is needed solely for this official-source re-check" in audit_text
    assert "2026-06-08 official-source boundary tightening" in audit_text
    assert "single-year hourly wind-farm simulation" in audit_text
    assert "wind-farm loss factors" in audit_text
    assert "system availability/other operating losses" in audit_text
    assert "site-specific power curve adjustments" in audit_text
    assert "extensive regression and unit testing" in audit_text
    assert "OpenFAST/AeroDyn aerodynamic workflow boundary note" in normalized
    assert "steady BEM wake/induction modelling" in normalized
    assert "DBEMT dynamic-inflow options" in normalized
    assert "unsteady-aerodynamics and dynamic-stall models" in normalized
    assert "InflowWind coupling" in normalized
    assert "AeroDyn driver/input/output workflows" in normalized
    assert "Do not describe this case as AeroDyn-equivalent" in normalized
    assert "not SAM, OpenFAST, WindWatts, IEC power-performance" in normalized
    assert "manufacturer-curve, wind-farm, or site-resource validated" in normalized


def test_current_wind_turbine_iec2022_and_sam_losses_boundary_is_pinned():
    case = WindTurbinePower6D()
    refs = "\n".join(case.meta.physics_references)
    validation_text = Path(
        "tests/_cross_validation/wind_turbine_power_6d.py"
    ).read_text(encoding="utf-8")
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    combined = "\n".join(
        [refs, case.meta.solver_details, case.meta.scenario_notes["VS-F"], validation_text]
    )
    normalized_audit = " ".join(audit_text.split())

    assert "https://samrepo.nrelcloud.org/help/wind_losses.html" in combined
    for phrase in (
        "internal/external/future wake",
        "turbine/balance-of-plant/grid availability",
        "electrical/parasitic losses",
        "generic and site-specific power-curve adjustments",
        "high-wind hysteresis",
        "environmental degradation or shutdown losses",
        "curtailment/operational",
    ):
        assert phrase in combined

    assert "IEC 61400-12-1:2022" in combined
    assert "https://webstore.iec.ch/en/publication/68499" in combined
    assert "derived energy-production" in combined
    assert "combined uncertainty" in combined
    assert "replaces IEC 61400-12-1:2017" in combined
    assert (
        "Additional `wind_turbine_power_6d` 2026-06-08 IEC-2022/SAM-losses "
        "boundary update"
    ) in normalized_audit
    assert "IEC 61400-12-1:2022" in normalized_audit
    assert "wind-measurement methods, site calibration, and obstacle/terrain assessment" in normalized_audit
    assert "electrical/parasitic losses" in normalized_audit
    assert "high-wind hysteresis" in normalized_audit
    assert (
        "No numerical response resweep is needed solely for this IEC/SAM "
        "boundary update"
    ) in normalized_audit
    assert "Additional `wind_turbine_power_6d` 2026-06-08 IEC-scope note" in normalized_audit
    assert "wind-measurement-method workflow" in normalized_audit
    assert "site-calibration workflow" in normalized_audit
    assert "obstacle/terrain-assessment workflow" in normalized_audit
    assert "uncertainty-budget report" in normalized_audit
    assert "Current IEC 61400-12-1:2017 material" not in audit_text
    assert "Current IEC 61400-12-1:2022 material describes single-turbine" in audit_text


def test_current_sam_wind_help_mirror_boundary_is_pinned():
    case = WindTurbinePower6D()
    refs = "\n".join(case.meta.physics_references)
    validation_text = Path(
        "tests/_cross_validation/wind_turbine_power_6d.py"
    ).read_text(encoding="utf-8")
    note_text = Path("tests/_cross_validation/wind_turbine_power_6d.md").read_text(
        encoding="utf-8"
    )
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    combined = "\n".join([refs, validation_text, note_text, audit_text])
    normalized = " ".join(combined.split())

    for source_anchor in (
        "https://samrepo.nrelcloud.org/help/wind_power.html",
        "https://samrepo.nrelcloud.org/help/wind_resource.html",
        "https://samrepo.nrelcloud.org/help/weather_format_csv_wind.html",
        "https://samrepo.nrelcloud.org/help/wind_losses.html",
    ):
        assert source_anchor in combined
    assert "current SAM help pages" in normalized
    assert "legacy `samrepo.nrelcloud.org` URLs" in normalized
    assert "SRW/SAM-CSV resource files" in normalized
    assert "hourly wind-farm simulation" in normalized
    assert "hub-height interpolation and wind-shear adjustment" in normalized
    assert "ideal-gas air-density adjustment" in normalized
    assert "wake models" in normalized
    assert "availability, electrical and parasitic losses" in normalized
    assert "curtailment/operational strategies" in normalized
    assert "boundary/context anchors only" in normalized
    assert "do not validate the Betz/Rayleigh response surface" in normalized
    assert "do not make the case SAM-equivalent" in normalized
    assert "P2 plausibility-only evidence tier" in normalized
    assert (
        "No numerical response resweep is needed solely for this source-domain "
        "traceability note"
    ) in normalized


def test_current_wind_turbine_page_spot_check_boundary_is_pinned():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `wind_turbine_power_6d` 2026-06-09 current-page spot check:"
        in audit_text
    )
    for phrase in (
        "current SAM wind power/resource/SRW/loss pages",
        "OpenFAST/AeroDyn pages",
        "NREL WIND Toolkit material",
        "IEC 61400-12-1:2022 material",
        "SRW/SAM CSV wind-resource files",
        "hub-height interpolation",
        "air-density calculation",
        "wake/loss/availability/ curtailment workflows",
        "driver/input/output workflows",
        "steady BEM wake/induction options",
        "dynamic inflow",
        "unsteady aerodynamics",
        "dynamic stall",
        "InflowWind coupling",
        "high-resolution WRF-based wind-resource",
        "measured single-turbine power-performance procedures",
        "separate wind-measurement, site-calibration, and obstacle/terrain-assessment standards",
        "single-mean-speed Betz/Rayleigh algebraic surrogate",
        "Do not describe this case as SAM validated",
        "OpenFAST validated",
        "AeroDyn validated",
        "WIND-Toolkit validated",
        "IEC power-performance validated",
        "manufacturer-curve validated",
        "wind-farm validated",
        "site-resource validated",
        "No numerical response resweep is needed solely for this current-page spot check",
    ):
        assert phrase in normalized


def test_current_wind_toolkit_boundary_is_pinned():
    case = WindTurbinePower6D()
    refs = "\n".join(case.meta.physics_references)
    validation_text = Path(
        "tests/_cross_validation/wind_turbine_power_6d.py"
    ).read_text(encoding="utf-8")
    note_text = Path("tests/_cross_validation/wind_turbine_power_6d.md").read_text(
        encoding="utf-8"
    )
    normalized_note = " ".join(note_text.split())
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    combined = "\n".join(
        [refs, case.meta.solver_details, case.meta.scenario_notes["VS-F"], validation_text, note_text]
    )
    normalized_audit = " ".join(audit_text.split())

    assert "https://www.nrel.gov/grid/wind-toolkit" in combined
    assert "NREL WIND Toolkit" in combined
    assert "WRF-based" in combined
    assert "2 km / 5-minute" in combined or "2 km/5-minute" in combined
    assert "wind-speed" in combined
    assert "wind-direction" in combined
    assert "temperature, pressure, density, and power" in combined
    assert "single-mean-speed Rayleigh surrogate" in combined
    assert "not WIND-Toolkit-equivalent" in normalized_note
    assert "Additional `wind_turbine_power_6d` 2026-06-09 WIND-Toolkit boundary re-check" in audit_text
    assert "time-correlated meteorological fields" in normalized_audit
    assert "derived power time series" in normalized_audit
    assert "no WRF, WIND Toolkit, gridded resource, wind-rose" in normalized_audit
    assert "Evidence tier remains P2 plausibility-only" in normalized_audit
    assert "Do not describe this case as WIND-Toolkit-equivalent" in normalized_audit
    assert "No numerical response resweep is needed solely for this official-source boundary note" in normalized_audit


def test_reviewer_markdown_note_keeps_wind_turbine_p2_scope_boundary():
    note = Path("tests/_cross_validation/wind_turbine_power_6d.md")
    text = note.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "P2 plausibility-only" in normalized
    assert "No response-surface change" in normalized
    assert "Betz-parameterized algebraic power model" in normalized
    assert "closed-form Rayleigh AEP integration" in normalized
    assert "not SAM-equivalent" in normalized
    assert "not OpenFAST-equivalent" in normalized
    assert "not AeroDyn-equivalent" in normalized
    assert "not WindWatts-equivalent" in normalized
    assert "not WIND-Toolkit-equivalent" in normalized
    assert "not IEC power-performance validated" in normalized
    assert "not calibrated to the NREL 5-MW" in normalized
    assert "Cp_max = (16/27)" in normalized
    assert "P = 0, below cut-in and above cut-out" in normalized
    assert "Probability outside the operating interval is not renormalized" in normalized
    assert "`theta_0` is retained" in normalized
    for source_anchor in [
        "https://www.nrel.gov/docs/fy14osti/60570.pdf",
        "https://samrepo.nrelcloud.org/help/wind_power.html",
        "https://samrepo.nrelcloud.org/help/wind_resource.html",
        "https://samrepo.nrelcloud.org/help/weather_format_csv_wind.html",
        "https://samrepo.nrelcloud.org/help/wind_losses.html",
        "https://samrepo.nrelcloud.org/help/wind_power.html",
        "https://samrepo.nrelcloud.org/help/wind_resource.html",
        "https://samrepo.nrelcloud.org/help/weather_format_csv_wind.html",
        "https://samrepo.nrelcloud.org/help/wind_losses.html",
        "https://www.nrel.gov/docs/fy09osti/38060.pdf",
        "https://www.nrel.gov/wind/nwtc/openfast",
        "https://openfast.readthedocs.io/",
        "https://openfast.readthedocs.io/en/dev/source/user/aerodyn/",
        "https://www.nrel.gov/wind/windwatts",
        "https://www.nrel.gov/grid/wind-toolkit",
        "https://webstore.iec.ch/en/publication/68499",
    ]:
        assert source_anchor in text


def test_readme_uses_betz_surrogate_label_not_bem():
    readme = Path("README.md").read_text(encoding="utf-8")
    readme_zh = Path("README_zh.md").read_text(encoding="utf-8")

    assert "Betz surrogate + Rayleigh integration" in readme
    assert "Simplified BEM + Rayleigh integration" not in readme
    assert "Betz 代理 + Rayleigh 积分" in readme_zh
    assert "简化 BEM + Rayleigh 积分" not in readme_zh


def test_metadata_distinguishes_cp_sobol_from_multi_output_importance():
    case = WindTurbinePower6D()

    assert "Cp_max_output" in case.meta.sobol_computation
    assert "multi-output union" in case.meta.sobol_computation
    assert set(case.meta.important_variables) == {
        "R",
        "lam_opt",
        "c_r",
        "V_rated",
        "V_bar",
    }
    assert case.meta.sobol_ST["V_rated"] == 0.0
    assert case.meta.sobol_ST["V_bar"] == 0.0
    assert "theta_0" not in case.meta.important_variables
    assert any("System Advisor Model" in ref for ref in case.meta.physics_references)


def test_rotor_radius_scales_rated_power_and_aep():
    case = WindTurbinePower6D()
    X_phys = np.array(
        [
            [30.0, 8.0, 3.0, 12.0, 12.0, 8.0],
            [50.0, 8.0, 3.0, 12.0, 12.0, 8.0],
            [80.0, 8.0, 3.0, 12.0, 12.0, 8.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.all(np.diff(Y[:, 1]) > 0.0)
    assert np.all(np.diff(Y[:, 2]) > 0.0)


def test_rated_power_matches_betz_parameterized_power_formula():
    case = WindTurbinePower6D()
    X_phys = np.array(
        [
            [50.0, 8.0, 3.0, 12.0, 12.0, 8.0],
            [80.0, 10.0, 2.0, 8.0, 14.0, 12.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    Cp_max = Y[:, 0]
    P_rated = Y[:, 1]
    expected_p_rated = (
        0.5
        * case._RHO
        * np.pi
        * X_phys[:, 0] ** 2
        * X_phys[:, 4] ** 3
        * Cp_max
        / 1e6
    )

    assert np.allclose(P_rated, expected_p_rated, rtol=1e-12, atol=1e-12)


def test_cp_max_matches_declared_betz_correction_formula():
    case = WindTurbinePower6D()
    X_phys = np.array(
        [
            [50.0, 8.0, 3.0, 12.0, 12.0, 8.0],
            [80.0, 10.0, 2.0, 8.0, 14.0, 12.0],
            [30.0, 6.0, 5.0, 20.0, 10.0, 5.0],
        ],
        dtype=float,
    )
    cp = case.evaluate(case.normalize(X_phys))[:, 0]
    c_avg = 0.6 * X_phys[:, 2]
    sigma = case._B * c_avg / (np.pi * X_phys[:, 0])
    tip_loss = 1.0 - 1.5 / (case._B * X_phys[:, 1])
    expected_cp = (16.0 / 27.0) * tip_loss * (1.0 - 0.3 * sigma)

    assert np.allclose(cp, expected_cp, rtol=1e-12, atol=1e-12)
    assert np.all(cp <= 16.0 / 27.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"R": 0.0},
        {"lam_opt": 0.0},
        {"c_r": -1.0},
        {"theta_0": np.nan},
        {"V_rated": 2.0},
        {"V_bar": 0.0},
    ],
)
def test_turbine_rejects_inputs_outside_betz_rayleigh_contract(kwargs):
    case = WindTurbinePower6D()
    params = {
        "R": 50.0,
        "lam_opt": 8.0,
        "c_r": 3.0,
        "theta_0": 12.0,
        "V_rated": 12.0,
        "V_bar": 8.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match="outside the declared"):
        case._turbine(**params)


def test_tip_speed_ratio_improves_cp_until_betz_limited_plateau():
    case = WindTurbinePower6D()
    X_phys = np.array(
        [
            [50.0, 6.0, 3.0, 12.0, 12.0, 8.0],
            [50.0, 8.0, 3.0, 12.0, 12.0, 8.0],
            [50.0, 10.0, 3.0, 12.0, 12.0, 8.0],
        ],
        dtype=float,
    )
    cp = case.evaluate(case.normalize(X_phys))[:, 0]

    assert np.all(np.diff(cp) >= 0.0)
    assert np.nanmax(cp) <= 16.0 / 27.0


def test_topology_metadata_matches_monotone_tip_speed_ratio_surrogate():
    case = WindTurbinePower6D()
    notes = " ".join(case.meta.scenario_notes.values()).lower()

    assert "smooth_monotone" in case.meta.response_topology
    assert "smooth_unimodal" not in case.meta.response_topology
    assert "monotonically" in notes
    assert "optimum w.r.t. lam_opt" not in notes


def test_interaction_metadata_covers_power_curve_and_aep_product_structure():
    case = WindTurbinePower6D()

    expected_pairs = {
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
    }

    assert case.meta.effective_dim == 5
    assert "theta_0" not in case.meta.important_variables
    assert set(case.meta.interaction_pairs) == expected_pairs


def test_rayleigh_distribution_parameter_matches_declared_mean_wind_speed():
    V_bar = 8.0
    V_arr = np.linspace(0.0, 60.0, 4000)
    c_rayleigh = V_bar * 2.0 / np.sqrt(np.pi)
    f_rayleigh = (2 * V_arr / c_rayleigh**2) * np.exp(-((V_arr / c_rayleigh) ** 2))

    assert np.isclose(np.trapezoid(f_rayleigh, V_arr), 1.0, rtol=1e-4)
    assert np.isclose(np.trapezoid(V_arr * f_rayleigh, V_arr), V_bar, rtol=1e-4)


def test_aep_integration_matches_cut_in_cut_out_zero_power_mask():
    case = WindTurbinePower6D()
    X_phys = np.array([[50.0, 8.0, 3.0, 12.0, 12.0, 8.0]], dtype=float)
    Cp_max, P_rated_MW, AEP_GWh = case.evaluate(case.normalize(X_phys))[0]
    R = X_phys[0, 0]
    V_rated = X_phys[0, 4]
    V_bar = X_phys[0, 5]
    P_rated_W = P_rated_MW * 1e6

    V = np.linspace(0.0, 60.0, 40000)
    P = np.zeros_like(V)
    active = (V >= case._V_CUT_IN) & (V <= case._V_CUT_OUT)
    subrated = active & (V <= V_rated)
    rated = active & (V > V_rated)
    P[subrated] = np.minimum(
        0.5 * case._RHO * np.pi * R**2 * V[subrated] ** 3 * Cp_max,
        P_rated_W,
    )
    P[rated] = P_rated_W
    c_rayleigh = V_bar * 2.0 / np.sqrt(np.pi)
    f_rayleigh = (2 * V / c_rayleigh**2) * np.exp(-((V / c_rayleigh) ** 2))
    expected_gwh = np.trapezoid(P * f_rayleigh, V) * 8760.0 / 1e9

    assert "not renormalised" in case.meta.solver_details
    assert "closed-form" in case.meta.solver_details
    assert np.isclose(AEP_GWh, expected_gwh, rtol=5e-6)


def test_aep_matches_closed_form_rayleigh_piecewise_power_curve():
    case = WindTurbinePower6D()
    X_phys = np.array([[30.0, 6.0, 5.0, 8.0, 10.0, 5.0]], dtype=float)
    Cp_max, P_rated_MW, AEP_GWh = case.evaluate(case.normalize(X_phys))[0]
    R = X_phys[0, 0]
    V_rated = X_phys[0, 4]
    V_bar = X_phys[0, 5]

    c_rayleigh = V_bar * 2.0 / np.sqrt(np.pi)
    u_cut_in = (case._V_CUT_IN / c_rayleigh) ** 2
    u_rated = (V_rated / c_rayleigh) ** 2
    u_cut_out = (case._V_CUT_OUT / c_rayleigh) ** 2
    k_power = 0.5 * case._RHO * np.pi * R**2 * Cp_max
    subrated_expectation = c_rayleigh**3 * gamma(2.5) * (
        gammainc(2.5, u_rated) - gammainc(2.5, u_cut_in)
    )
    rated_probability = np.exp(-u_rated) - np.exp(-u_cut_out)
    expected_gwh = (
        k_power * subrated_expectation + P_rated_MW * 1e6 * rated_probability
    ) * 8760.0 / 1e9

    assert np.isclose(AEP_GWh, expected_gwh, rtol=1e-12, atol=1e-12)


def test_mean_wind_speed_changes_aep_not_design_coefficients():
    case = WindTurbinePower6D()
    X_phys = np.array(
        [
            [50.0, 8.0, 3.0, 12.0, 12.0, 5.0],
            [50.0, 8.0, 3.0, 12.0, 12.0, 8.0],
            [50.0, 8.0, 3.0, 12.0, 12.0, 12.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.allclose(Y[:, 0], Y[0, 0], rtol=1e-12, atol=1e-12)
    assert np.allclose(Y[:, 1], Y[0, 1], rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(Y[:, 2]) > 0.0)


def test_aep_does_not_exceed_full_year_rated_energy():
    case = WindTurbinePower6D()
    X_phys = np.array(
        [
            [30.0, 6.0, 5.0, 20.0, 10.0, 5.0],
            [80.0, 10.0, 2.0, 8.0, 14.0, 12.0],
            [50.0, 8.0, 3.0, 12.0, 12.0, 8.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    P_rated_MW = Y[:, 1]
    AEP_GWh = Y[:, 2]
    full_year_rated_GWh = P_rated_MW * 8760.0 / 1000.0

    assert np.all(AEP_GWh > 0.0)
    assert np.all(AEP_GWh <= full_year_rated_GWh)


def test_rayleigh_closed_form_contract_rejects_nonfinite_subrated_moment(monkeypatch):
    case = WindTurbinePower6D()

    def bad_gammainc(*_args, **_kwargs):
        return np.nan

    monkeypatch.setattr(wind_case_module, "gammainc", bad_gammainc)
    with pytest.raises(ValueError, match="sub-rated wind-speed moment"):
        case._turbine(50.0, 8.0, 3.0, 12.0, 12.0, 8.0)


def test_rayleigh_closed_form_contract_rejects_invalid_rated_probability(monkeypatch):
    case = WindTurbinePower6D()
    calls = {"n": 0}

    def bad_exp(_values):
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 1.0

    monkeypatch.setattr(wind_case_module.np, "exp", bad_exp)
    with pytest.raises(ValueError, match="rated-power probability mass"):
        case._turbine(50.0, 8.0, 3.0, 12.0, 12.0, 8.0)


def test_declared_power_curve_shuts_down_above_cutout():
    case = WindTurbinePower6D()
    source_text = Path(
        "doe_testbed/cases/multidisciplinary/wind_turbine_power_6d.py"
    ).read_text(encoding="utf-8")

    assert case._V_CUT_OUT == 25.0
    assert "V > V_cut-out" in source_text
    assert "P = 0 above cut-out" in case.meta.governing_equation


def test_theta0_is_declared_neutral_in_low_order_surrogate():
    case = WindTurbinePower6D()
    theta_var = next(var for var in case.meta.input_variables if var.name == "theta_0")
    X_phys = np.array(
        [
            [50.0, 8.0, 3.0, 8.0, 12.0, 8.0],
            [50.0, 8.0, 3.0, 14.0, 12.0, 8.0],
            [50.0, 8.0, 3.0, 20.0, 12.0, 8.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert "neutral" in theta_var.description.lower()
    assert "theta_0" not in case.meta.important_variables
    assert np.allclose(Y, Y[0], rtol=1e-12, atol=1e-12)


def test_rated_wind_speed_increases_rated_power_and_aep():
    case = WindTurbinePower6D()
    X_phys = np.array(
        [
            [50.0, 8.0, 3.0, 12.0, 10.0, 8.0],
            [50.0, 8.0, 3.0, 12.0, 12.0, 8.0],
            [50.0, 8.0, 3.0, 12.0, 14.0, 8.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.all(np.diff(Y[:, 1]) > 0.0)
    assert np.all(np.diff(Y[:, 2]) > 0.0)


def test_wind_turbine_surrogate_failure_is_not_silent_nan(monkeypatch):
    case = WindTurbinePower6D()

    def fail_turbine(*_args, **_kwargs):
        raise ValueError("simulated wind-turbine power-curve failure")

    monkeypatch.setattr(case, "_turbine", fail_turbine)
    X_phys = np.array([[50.0, 8.0, 3.0, 12.0, 12.0, 8.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_turbine(monkeypatch):
    case = WindTurbinePower6D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("wind-turbine surrogate should not be called for invalid input")

    monkeypatch.setattr(case, "_turbine", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._validate_physical_inputs(np.array([50.0, 8.0, 3.0, 12.0, 12.0, 8.0]))

    invalid_rows = [
        [29.9, 8.0, 3.0, 12.0, 12.0, 8.0],
        [80.1, 8.0, 3.0, 12.0, 12.0, 8.0],
        [50.0, 5.9, 3.0, 12.0, 12.0, 8.0],
        [50.0, 10.1, 3.0, 12.0, 12.0, 8.0],
        [50.0, 8.0, 1.9, 12.0, 12.0, 8.0],
        [50.0, 8.0, 5.1, 12.0, 12.0, 8.0],
        [50.0, 8.0, 3.0, 7.9, 12.0, 8.0],
        [50.0, 8.0, 3.0, 20.1, 12.0, 8.0],
        [50.0, 8.0, 3.0, 12.0, 9.9, 8.0],
        [50.0, 8.0, 3.0, 12.0, 14.1, 8.0],
        [50.0, 8.0, 3.0, 12.0, 12.0, 4.9],
        [50.0, 8.0, 3.0, 12.0, 12.0, 12.1],
        [50.0, 8.0, 3.0, np.nan, 12.0, 8.0],
    ]
    for row in invalid_rows:
        with pytest.raises(
            ValueError,
            match="R/lam_opt/c_r/theta_0/V_rated/V_bar benchmark domain",
        ):
            case._evaluate_core(np.array([row], dtype=float))


def test_nonfinite_or_nonphysical_wind_turbine_response_is_not_silent_nan(monkeypatch):
    case = WindTurbinePower6D()

    def bad_turbine(*_args, **_kwargs):
        return np.array([0.7, 2.0, 100.0])

    monkeypatch.setattr(case, "_turbine", bad_turbine)
    X_phys = np.array([[50.0, 8.0, 3.0, 12.0, 12.0, 8.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_out_of_contract_wind_turbine_response_is_not_archived(monkeypatch):
    case = WindTurbinePower6D()
    X_phys = np.array([[50.0, 8.0, 3.0, 12.0, 12.0, 8.0]], dtype=float)

    invalid_responses = [
        np.array([0.19, 2.0, 10.0]),
        np.array([0.56, 0.49, 10.0]),
        np.array([0.56, 2.0, 86.1]),
    ]

    for response in invalid_responses:
        monkeypatch.setattr(
            case,
            "_turbine",
            lambda *_args, _response=response, **_kwargs: _response,
        )
        with pytest.raises(RuntimeError, match="refusing silent NaN"):
            case._evaluate_core(X_phys)
