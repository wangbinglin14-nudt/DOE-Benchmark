"""Physics regression tests for ``solar_pv_yield_6d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.multidisciplinary.solar_pv_yield_6d import SolarPVYield6D


def _domain_corners(case: SolarPVYield6D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_solar_pv_surrogate_failure_is_not_silent_nan(monkeypatch):
    case = SolarPVYield6D()

    def fail_pv_yield(*_args, **_kwargs):
        raise ValueError("simulated PV yield failure")

    monkeypatch.setattr(case, "_pv_yield", fail_pv_yield)
    X_phys = np.array([[1500.0, 30.0, 0.0, 0.2, -0.35, 0.8]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_nonfinite_or_nonphysical_pv_response_is_not_silent_nan(monkeypatch):
    case = SolarPVYield6D()

    def bad_pv_yield(*_args, **_kwargs):
        return np.array([1200.0, 1.2, np.nan])

    monkeypatch.setattr(case, "_pv_yield", bad_pv_yield)
    X_phys = np.array([[1500.0, 30.0, 0.0, 0.2, -0.35, 0.8]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_out_of_contract_pv_response_is_not_archived(monkeypatch):
    case = SolarPVYield6D()
    X_phys = np.array([[1500.0, 30.0, 0.0, 0.2, -0.35, 0.8]], dtype=float)

    invalid_responses = [
        np.array([299.0, 0.10, 40.0]),
        np.array([2301.0, 0.10, 40.0]),
        np.array([1200.0, 0.034, 40.0]),
        np.array([1200.0, 0.281, 40.0]),
        np.array([1200.0, 0.10, 19.9]),
        np.array([1200.0, 0.10, 70.1]),
    ]

    for response in invalid_responses:
        monkeypatch.setattr(
            case,
            "_pv_yield",
            lambda *_args, _response=response, **_kwargs: _response,
        )
        with pytest.raises(RuntimeError, match="refusing silent NaN"):
            case._evaluate_core(X_phys)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"GHI": 0.0},
        {"eta": 0.0},
        {"beta_T": 0.1},
        {"beta_T": -6.0},
        {"PR": 0.0},
        {"PR": 1.2},
        {"tilt": np.nan},
        {"tilt": -1.0},
        {"tilt": 61.0},
        {"az_dev": -31.0},
        {"az_dev": 31.0},
    ],
)
def test_pv_yield_rejects_inputs_outside_annual_yield_contract(kwargs):
    params = {
        "GHI": 1400.0,
        "tilt": 35.0,
        "az_dev": 0.0,
        "eta": 0.20,
        "beta_T": -0.35,
        "PR": 0.80,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match="outside the declared"):
        SolarPVYield6D._pv_yield(**params)


def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_pv_yield(monkeypatch):
    case = SolarPVYield6D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("PV yield surrogate should not be called for invalid input")

    monkeypatch.setattr(case, "_pv_yield", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._validate_physical_inputs(np.array([1400.0, 35.0, 0.0, 0.20, -0.35, 0.80]))

    invalid_rows = [
        [799.0, 35.0, 0.0, 0.20, -0.35, 0.80],
        [2201.0, 35.0, 0.0, 0.20, -0.35, 0.80],
        [1400.0, -0.1, 0.0, 0.20, -0.35, 0.80],
        [1400.0, 60.1, 0.0, 0.20, -0.35, 0.80],
        [1400.0, 35.0, -30.1, 0.20, -0.35, 0.80],
        [1400.0, 35.0, 30.1, 0.20, -0.35, 0.80],
        [1400.0, 35.0, 0.0, 0.13, -0.35, 0.80],
        [1400.0, 35.0, 0.0, 0.26, -0.35, 0.80],
        [1400.0, 35.0, 0.0, 0.20, -0.51, 0.80],
        [1400.0, 35.0, 0.0, 0.20, -0.19, 0.80],
        [1400.0, 35.0, 0.0, 0.20, -0.35, 0.69],
        [1400.0, 35.0, 0.0, 0.20, -0.35, 0.91],
        [1400.0, np.nan, 0.0, 0.20, -0.35, 0.80],
    ]
    for row in invalid_rows:
        with pytest.raises(
            ValueError,
            match="GHI_annual/tilt/azimuth_dev/eta_STC/beta_T/PR benchmark domain",
        ):
            case._evaluate_core(np.array([row], dtype=float))


def test_declared_output_ranges_cover_domain_corners():
    case = SolarPVYield6D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_metadata_matches_low_order_pvwatts_style_scope():
    case = SolarPVYield6D()
    equation = case.meta.governing_equation
    details = case.meta.solver_details.lower()
    references = " ".join(case.meta.physics_references)
    source_text = Path(
        "doe_testbed/cases/multidisciplinary/solar_pv_yield_6d.py"
    ).read_text(encoding="utf-8")
    normalized_source = " ".join(source_text.split())

    assert "smooth_unimodal" in case.meta.response_topology
    assert "strong_interaction" in case.meta.response_topology
    assert "(β_T/100)" in equation
    assert "η_STC" in equation
    assert "pvwatts" in references.lower()
    assert "liu-jordan-inspired" in details
    assert "annual irradiance transposition approximation" in details
    assert hasattr(case, "_validate_physical_inputs")
    assert "named _validate_physical_inputs private physical-input evaluation rejects" in case.meta.solver_details
    assert "before PVWatts-style response assembly" in case.meta.solver_details
    assert "tilt outside [0, 60]" in case.meta.solver_details
    assert "azimuth deviation outside [-30, 30]" in case.meta.solver_details
    assert "no hourly weather" in details
    assert "nsrdb/tmy weather-file lookup" in details
    assert "pvwatts v8 api/ssc" in details
    assert "pvwattsv8 hourly toolchain" in details
    assert "pvwatts module_type/array_type/gcr/dc-ac-ratio/inverter-efficiency" in details
    assert "pvwatts v8 module/inverter/thermal-effect algorithm updates" in details
    assert "nsrdb 2020 tmy weather updates" in details
    assert "psm v4 meteosat/polar tmy dataset selection" in details
    assert "pvwatts v8 timeframe=monthly/hourly selection" in details
    assert "weather_data_source" in details
    assert "solar-position" in details
    assert "pvlib/perez/hay-davies/reindl" in details
    assert "pvlib.pvsystem.pvwatts_dc" in details
    assert "pvlib.inverter.pvwatts" in details
    assert "pvlib.pvsystem.pvwatts_losses" in details
    assert "pvlib modelchain solar-position/airmass/aoi/spectral/temperature/dc/ac/loss workflow" in details
    assert "iam/spectral/snow/soiling/monthly-loss" in details
    assert "albedo" in details
    assert "bifacial" in details
    assert "sdm" in details
    assert "inverter" in details
    assert "monthly/hourly output simulation" in details
    assert "pvwatts ssc simulation" in details
    assert "weather-data selection" in details
    assert "hourly ac/poa/dc/ambient-temperature/module-temperature outputs" in details
    assert "snow-depth shading" in details
    assert "inverter undersizing warnings" in details
    assert "pvwatts version 8.5.2 calculator release-date tracking" in details
    assert "ssc version 275 tracking" in details
    assert "pvwatts uncertainty cautions" in details
    assert "pvpmc/sandia performance-index or uncertainty workflow" in details
    assert "performance-ratio references anchor metric definitions only" in details
    assert "not measured-plant monitoring" in details
    assert "expected-energy" in details
    assert "contractual performance-index validation" in details
    assert "cf is archived as a dimensionless fraction e_annual/8760" in details
    assert "rather than the percent-form capacity_factor field" in details
    assert "out-of-contract e_annual/cf/t_cell_avg responses are rejected" in details
    assert "share the same underlying `pvwattsv8` model and weather-data path" in normalized_source
    assert "weather-file choice, module and inverter model selection" in normalized_source
    assert "shading/layout losses, snow-depth-dependent shading" in normalized_source

    for url in [
        "https://www.nrel.gov/docs/fy14osti/62641.pdf",
        "https://pvwatts.nrel.gov/",
        "https://developer.nrel.gov/docs/solar/pvwatts/v8/",
        "https://developer.nrel.gov/docs/solar/pvwatts/v8/",
        "https://pvwatts.nrel.gov/version_8.php",
        "https://samrepo.nrelcloud.org/help/pvwatts.html",
        "https://samrepo.nrelcloud.org/help/pvwatts.html",
        "https://samrepo.nrelcloud.org/help/pvwatts_system_design.html",
        "https://samrepo.nrelcloud.org/help/pvwatts_system_design.html",
        "https://samrepo.nrelcloud.org/help/detailed_photovoltaic_model.html",
        "https://samrepo.nrelcloud.org/help/detailed_photovoltaic_model.html",
        "https://pvlib-python.readthedocs.io/en/stable/reference/irradiance/transposition.html",
        "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html",
        "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html",
        "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_losses.html",
        "https://pvlib-python.readthedocs.io/en/stable/user_guide/modeling_topics/modelchain.html",
        "https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-ratio/",
        "https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/annual-yield/",
        "https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-index/",
    ]:
        assert url in references

    vs_f_note = case.meta.scenario_notes["VS-F"]
    assert "model-boundary wording" in vs_f_note
    assert "does not call the current PVWatts V8 API/SSC model" in vs_f_note
    assert "PVWatts V5" in vs_f_note
    assert "PVWatts Calculator" in vs_f_note
    assert "SAM PVWatts" in vs_f_note
    assert "SAM detailed PV" in vs_f_note
    assert "NSRDB/TMY weather-file lookup" in vs_f_note
    assert "PVWatts module/array/GCR/" in vs_f_note
    assert "DC-AC-ratio/inverter-efficiency inputs" in vs_f_note
    assert "PVWatts V8 detailed module/inverter/thermal-effect updates" in vs_f_note
    assert "PVWatts V8 timeframe/weather_data_source/hourly AC/POA/DC/" in vs_f_note
    assert "temperature output traces" in vs_f_note
    assert "PVWatts Version 8.5.2 calculator release-date tracking" in vs_f_note
    assert "SSC version 275 tracking" in vs_f_note
    assert "PSM V4 Meteosat/Polar TMY dataset selection" in vs_f_note
    assert "albedo" in vs_f_note
    assert "bifaciality" in vs_f_note
    assert "monthly soiling/loss inputs" in vs_f_note
    assert "monthly/hourly PVWatts outputs" in vs_f_note
    assert "pvlib/Perez/Hay-Davies/Reindl" in vs_f_note
    assert "pvlib PVWatts DC/inverter/loss functions" in vs_f_note
    assert "pvlib ModelChain" in vs_f_note
    assert "NOCT/Fuentes thermal" in vs_f_note
    assert "single-diode" in vs_f_note
    assert "DC/AC sizing" in vs_f_note
    assert "inverter clipping" in vs_f_note
    assert "IAM" in vs_f_note
    assert "spectral" in vs_f_note
    assert "soiling" in vs_f_note
    assert "snow" in vs_f_note
    assert "inverter" in vs_f_note
    assert "degradation" in vs_f_note
    assert "bifacial" in vs_f_note
    assert "grid-curtailment" in vs_f_note
    assert "PVWatts Calculator" in vs_f_note
    assert "bankable site-assessment model" in vs_f_note


def test_interaction_metadata_covers_annual_yield_product_structure():
    case = SolarPVYield6D()

    expected_pairs = {
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
    }

    assert set(case.meta.interaction_pairs) == expected_pairs


def test_important_variable_metadata_matches_st_threshold_scope():
    case = SolarPVYield6D()

    assert case.meta.effective_dim == 4
    assert set(case.meta.important_variables) == {
        "GHI_annual",
        "tilt",
        "azimuth_dev",
        "PR",
    }
    assert all(case.meta.sobol_ST[name] > 0.05 for name in case.meta.important_variables)
    assert case.meta.sobol_ST["beta_T"] < 0.05
    assert case.meta.sobol_ST["eta_STC"] < 0.05
    assert "beta_T" in case.meta.noise_factor_candidates
    assert "eta_STC" not in case.meta.important_variables
    assert ("eta_STC", "beta_T") in case.meta.interaction_pairs
    assert ("GHI_annual", "beta_T") in case.meta.interaction_pairs


def test_robust_design_metadata_lists_all_noise_and_control_factors():
    case = SolarPVYield6D()
    note = case.meta.scenario_notes["VS-H"]

    assert set(case.meta.noise_factor_candidates) == {"GHI_annual", "beta_T"}
    assert set(case.meta.control_factor_candidates) == {
        "tilt",
        "azimuth_dev",
        "eta_STC",
        "PR",
    }
    assert "GHI_annual and beta_T are noise factors" in note
    assert "tilt, azimuth_dev, eta_STC, and PR are control/design factors" in note


def test_validation_material_declares_non_bankable_surrogate_scope():
    validation_text = Path("tests/_cross_validation/solar_pv_yield_6d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path(
        "doe_testbed/cases/multidisciplinary/solar_pv_yield_6d.py"
    ).read_text(encoding="utf-8")

    normalized_validation = " ".join(validation_text.split())
    normalized_source = " ".join(source_text.split())
    assert "PVWatts-style algebraic surrogate" in validation_text
    assert "Liu-Jordan-inspired annual transposition approximation" in normalized_validation
    assert "do not call or validate the current PVWatts V8 API/SSC model" in normalized_validation
    assert "PVWatts V5" in validation_text
    assert "SAM PVWatts" in validation_text
    assert "NSRDB/TMY weather-file lookup" in validation_text
    assert "PVWatts\nmodule/array/GCR/DC-AC-ratio/inverter-efficiency inputs" in validation_text
    assert "albedo" in validation_text
    assert "bifaciality" in validation_text
    assert "PVWatts V8 detailed module/inverter/thermal-effect updates" in validation_text
    assert "pvlib PVWatts DC/inverter/loss functions" in validation_text
    assert "pvlib ModelChain" in validation_text
    assert "pvlib exposes PVWatts DC, inverter, and loss functions" in normalized_validation
    assert "solar-position, airmass, AOI, spectral, temperature, DC, AC, and loss" in normalized_validation
    assert "does not call pvlib or reproduce that chained workflow" in normalized_validation
    assert "PVWatts-function equivalence" in normalized_validation
    assert "official PVWatts Version 8 calculator page" in normalized_validation
    assert "Version 8.5.2" in validation_text
    assert "release date 2025-09-25" in validation_text
    assert "SSC version 275" in validation_text
    assert "`pvwattsv8`" in validation_text
    assert "PVWatts uncertainty cautions" in validation_text
    assert "annual azimuth-deviation factor is symmetric by construction" in validation_text
    assert "eastward and westward deviations of equal magnitude" in normalized_validation
    assert "NSRDB 2020 TMY\nweather updates" in validation_text
    assert "PSM V4 Meteosat/Polar TMY dataset selection" in validation_text
    assert "timeframe=monthly|hourly" in validation_text
    assert "weather_data_source" in validation_text
    assert "hourly AC/POA/DC/" in validation_text
    assert "ambient-temperature/cell-temperature traces" in validation_text
    assert "monthly soiling/loss inputs" in normalized_validation
    assert "monthly/hourly PVWatts outputs" in validation_text
    assert "pvlib/Perez/Hay-Davies/Reindl" in validation_text
    assert "NOCT/Fuentes thermal-model" in validation_text
    assert "DC/AC sizing" in validation_text
    assert "inverter clipping" in validation_text
    assert "PVWatts Calculator" in validation_text
    assert "grid-curtailment" in validation_text
    assert "PVPMC/Sandia\nperformance-index or uncertainty workflows" in validation_text
    assert "measured-plant monitoring" in validation_text
    assert "expected-energy contracts" in validation_text
    assert "metric definition anchors" in normalized_validation
    assert "seasonal temperature-normalized PR analysis" in normalized_validation
    assert "contractual performance index workflows" in normalized_validation
    assert "dimensionless" in validation_text
    assert "percent-form PVWatts V8 API" in validation_text
    assert "capacity_factor output" in validation_text
    assert "bankable site-assessment validation" in normalized_validation
    assert "not for site-specific bankable PV assessment" in normalized_source
    assert "Target accuracy" not in validation_text
    assert "external validation" not in validation_text


def test_reviewer_markdown_note_keeps_solar_pv_p2_scope_boundary():
    note_text = Path("tests/_cross_validation/solar_pv_yield_6d.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(note_text.split())

    required_phrases = [
        "P2 plausibility-only",
        "No response-surface change",
        "Fixed-parameter PVWatts-style annual algebraic surrogate",
        "Liu-Jordan-inspired transposition",
        "not PVWatts V8/API/SSC validated",
        "not PVWatts Calculator validated",
        "not SAM PVWatts validated",
        "not SAM Detailed PV validated",
        "not pvlib/Perez/Hay-Davies/Reindl validated",
        "not hourly-weather validated",
        "not NSRDB/TMY weather-file based",
        "PSM V3/V4 weather-dataset selection",
        "not inverter-clipping validated",
        "not bankable site-assessment validated",
        "not measured-plant monitoring",
        "not contractual performance-index reporting",
        "not pvlib PVWatts/ModelChain validated",
        "annual azimuth-deviation identity",
        "eastward and westward deviations of equal magnitude",
        "https://www.nrel.gov/docs/fy14osti/62641.pdf",
        "https://developer.nrel.gov/docs/solar/pvwatts/v8/",
        "https://developer.nrel.gov/docs/solar/pvwatts/v8/",
        "https://pvwatts.nrel.gov/version_8.php",
        "https://samrepo.nrelcloud.org/help/pvwatts.html",
        "https://samrepo.nrelcloud.org/help/pvwatts.html",
        "https://pvlib-python.readthedocs.io/en/stable/reference/irradiance/transposition.html",
        "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html",
        "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html",
        "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_losses.html",
        "https://pvlib-python.readthedocs.io/en/stable/user_guide/modeling_topics/modelchain.html",
        "https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/performance-ratio/",
    ]
    for phrase in required_phrases:
        assert phrase in normalized

    for forbidden_phrase in [
        "Evidence tier | P1",
        "Evidence tier | P0",
        "Evidence tier | externally validated",
    ]:
        assert forbidden_phrase not in normalized


def test_web_validation_audit_preserves_solar_p2_boundary():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `solar_pv_yield_6d` |")
        and "PVWatts" in line
        and "bankable" in line
    )

    assert "Evidence tier remains P2" in row
    assert "PVWatts Calculator" in row
    assert "PVWatts V8 API" in row
    assert "DC/AC sizing" in row
    assert "inverter clipping" in row
    assert "not validation data" in row
    assert "Keep P2 plausibility-only" in row


def test_web_audit_records_solar_pv_official_source_recheck():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `solar_pv_yield_6d` 2026-06-08 official-source re-check" in audit_text
    assert "public NREL and pvlib sources were re-opened as PV-yield context" in normalized
    assert "full-toolchain boundary evidence" in normalized
    assert "not as validation data for the registered response surface" in normalized
    assert "NREL PVWatts V8 API documentation supports the omitted-scope boundary" in normalized
    assert "updated module, inverter, and module-thermal-effect algorithms" in normalized
    assert "NSRDB 2020 TMY weather updates" in normalized
    assert "PSM V4 Meteosat/Polar TMY dataset selection" in normalized
    assert "annual AC energy" in normalized
    assert "monthly AC energy" in normalized
    assert "hourly AC output" in normalized
    assert "plane-of-array irradiance" in normalized
    assert "DC output" in normalized
    assert "cell-temperature traces" in normalized
    for pvwatts_boundary in (
        "NSRDB/TMY weather datasets",
        "`module_type`",
        "`array_type`",
        "`gcr`",
        "`dc_ac_ratio`",
        "`inv_eff`",
        "`timeframe`",
        "`weather_data_source`",
        "albedo",
        "bifaciality",
        "monthly soiling",
        "monthly/hourly response workflows",
    ):
        assert pvwatts_boundary in normalized
    assert "`ac_annual`" in normalized
    assert "percent-form `capacity_factor`" in normalized
    assert "dimensionless `CF = E_annual/8760`" in normalized
    assert "NREL SAM PVWatts and Detailed PV help support omitted workflows" in normalized
    for sam_boundary in (
        "weather-file choice",
        "detailed module and inverter models",
        "inverter clipping",
        "system sizing",
        "bifacial/rear-side irradiance",
        "battery/grid workflows",
        "detailed time-series PV simulation",
    ):
        assert sam_boundary in normalized
    assert "snow/soiling/inverter-loss workflows" in normalized
    assert "shading/layout losses" in normalized
    assert "snow-depth-dependent shading" in normalized
    assert "hourly AC/POA/DC/ambient-temperature/cell-temperature traces" in normalized
    assert "inverter is undersized" in normalized
    assert "pvlib transposition documentation supports the boundary" in normalized
    assert "`irradiance.get_total_irradiance`" in normalized
    assert "Perez" in normalized
    assert "Hay-Davies" in normalized
    assert "Reindl" in normalized
    assert "fixed-parameter, annual, per-kWp, PVWatts-style algebraic surrogate" in normalized
    assert "Liu-Jordan-inspired component-weight tilt factor" in normalized
    assert "performance-ratio multiplier" in normalized
    assert "PVPMC/Sandia annual-yield and performance-ratio pages support" in normalized
    assert "performance-index page sharpens the omitted workflow boundary" in normalized
    assert "measured production" in normalized
    assert "expected-energy baselines" in normalized
    assert "uncertainty-aware plant monitoring" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    for forbidden_claim in (
        "PVWatts V8/API/SSC validated",
        "PVWatts Calculator validated",
        "SAM PVWatts validated",
        "SAM Detailed PV validated",
        "pvlib/Perez/Hay-Davies/Reindl validated",
        "hourly-weather validated",
        "NSRDB/TMY weather-file based",
        "inverter-clipping",
        "bifacial",
        "battery/grid",
        "single-diode",
        "bankable site-assessment",
        "project-finance PV model output",
        "measured plant monitoring",
        "expected-energy analysis",
        "uncertainty analysis",
        "contractual performance-index reporting",
    ):
        assert forbidden_claim in normalized
    assert (
        "No numerical response resweep is needed solely for this official-source re-check"
        in normalized
    )


def test_web_audit_records_solar_pv_version8_boundary_recheck():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `solar_pv_yield_6d` 2026-06-09 PVWatts-Version-8 page re-check" in audit_text
    assert "official PVWatts Version 8 calculator page" in normalized
    assert "Version 8.5.2" in normalized
    assert "2025-09-25 release date" in normalized
    assert "SSC version 275" in normalized
    assert "`pvwattsv8` SSC model family" in normalized
    assert "NSRDB 2020 TMY weather data" in normalized
    assert "PSM V3 GOES/Himawari `tmy-2020`" in normalized
    assert "PSM V4 Meteosat Prime Meridian `tmy-2022`" in normalized
    assert "PSM V4 Polar `tmy-2023`" in normalized
    assert "bifacial/rear-side irradiance" in normalized
    assert "albedo and monthly irradiance-loss controls" in normalized
    assert "updated module/inverter/thermal-effect algorithms" in normalized
    assert "explicit uncertainty cautions" in normalized
    assert "`solar_pv_yield_6d` remains a fixed-parameter annual per-kWp algebraic surrogate" in normalized
    assert "does not run PVWatts V8/SSC" in normalized
    assert "the public PVWatts Calculator" in normalized
    assert "hourly weather" in normalized
    assert "PSM V3/V4 weather-dataset selection" in normalized
    assert "contractual expected-energy workflow" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert "No numerical response resweep is needed solely for this official-source boundary note" in normalized


def test_web_audit_records_solar_pv_annual_azimuth_symmetry_control():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `solar_pv_yield_6d` 2026-06-09 annual azimuth-symmetry control note" in audit_text
    assert "registered annual `cos^2(azimuth_dev)` factor" in normalized
    assert "Eastward and westward deviations of equal magnitude" in normalized
    assert "identical annual `E_annual`, `CF`, and `T_cell_avg` responses" in normalized
    assert "not PVWatts/SAM/pvlib validation" in normalized
    assert "not a claim about hourly or site-specific east/west asymmetry" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert "No numerical response resweep is needed solely for this symmetry-control note" in normalized


def test_web_audit_records_solar_pv_pvlib_modelchain_boundary_note():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `solar_pv_yield_6d` 2026-06-09 pvlib-PVWatts/ModelChain boundary note" in normalized
    assert "PVWatts DC, inverter, and loss functions" in normalized
    assert "ModelChain workflow" in normalized
    assert "solar-position, airmass, AOI, spectral, temperature, DC, AC, and loss models" in normalized
    assert "`pvlib.pvsystem.pvwatts_dc`" in normalized
    assert "`pvlib.inverter.pvwatts`" in normalized
    assert "`pvlib.pvsystem.pvwatts_losses`" in normalized
    assert "does not reproduce a pvlib hourly POA/DC/AC/loss workflow" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert "PVWatts-function validated" in normalized
    assert "pvlib ModelChain validated" in normalized
    assert "PVWatts-equivalent" in normalized
    assert "No numerical response resweep is needed solely for this source-boundary sync" in normalized
    assert (
        "Additional `solar_pv_yield_6d` 2026-06-09 current-page spot check:"
        in audit_text
    )
    assert "PVWatts Version 8/API pages" in normalized
    assert "SAM PVWatts and detailed-photovoltaic help pages" in normalized
    assert "pvlib PVWatts/ModelChain/transposition documentation" in normalized
    assert "PVPMC/Sandia annual-yield/performance-ratio/performance-index pages" in normalized
    assert "weather-data selection" in normalized
    assert "hourly/monthly simulation" in normalized
    assert "module and array configuration" in normalized
    assert "inverter and DC/AC sizing" in normalized
    assert "bifacial/albedo/monthly-loss features" in normalized
    assert "detailed transposition" in normalized
    assert "solar-position/AOI/spectral/temperature models" in normalized
    assert "measured-production metrics" in normalized
    assert "expected-energy baselines" in normalized
    assert "uncertainty or performance-index workflows" in normalized
    assert "symmetric annual azimuth factor" in normalized
    assert "Do not describe this case as PVWatts V8/API/SSC validated" in normalized
    assert "SAM PVWatts validated" in normalized
    assert "SAM Detailed PV validated" in normalized
    assert "pvlib validated" in normalized
    assert "PVPMC measured-plant validated" in normalized
    assert "bankable-site-assessment validated" in normalized
    assert "expected-energy validated" in normalized
    assert "performance-index validated" in normalized
    assert "No numerical response resweep is needed solely for this current-page spot check" in normalized


def test_irradiance_and_performance_ratio_increase_specific_yield():
    case = SolarPVYield6D()
    irradiance_sweep = np.array(
        [
            [800.0, 35.0, 0.0, 0.20, -0.35, 0.80],
            [1400.0, 35.0, 0.0, 0.20, -0.35, 0.80],
            [2200.0, 35.0, 0.0, 0.20, -0.35, 0.80],
        ],
        dtype=float,
    )
    pr_sweep = np.array(
        [
            [1400.0, 35.0, 0.0, 0.20, -0.35, 0.70],
            [1400.0, 35.0, 0.0, 0.20, -0.35, 0.80],
            [1400.0, 35.0, 0.0, 0.20, -0.35, 0.90],
        ],
        dtype=float,
    )

    irradiance_y = case.evaluate(case.normalize(irradiance_sweep))
    pr_y = case.evaluate(case.normalize(pr_sweep))

    assert np.all(np.diff(irradiance_y[:, 0]) > 0.0)
    assert np.all(np.diff(irradiance_y[:, 1]) > 0.0)
    assert np.all(np.diff(pr_y[:, 0]) > 0.0)
    assert np.all(np.diff(pr_y[:, 1]) > 0.0)


def test_performance_ratio_is_linear_loss_multiplier():
    case = SolarPVYield6D()
    X_phys = np.array(
        [
            [1400.0, 35.0, 0.0, 0.20, -0.35, 0.70],
            [1400.0, 35.0, 0.0, 0.20, -0.35, 0.90],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.isclose(Y[1, 0] / Y[0, 0], 0.90 / 0.70, rtol=1e-12, atol=1e-12)
    assert np.isclose(Y[1, 1] / Y[0, 1], 0.90 / 0.70, rtol=1e-12, atol=1e-12)


def test_specific_yield_matches_declared_algebraic_surrogate_formula():
    case = SolarPVYield6D()
    X_phys = np.array(
        [
            [1400.0, 35.0, 0.0, 0.20, -0.35, 0.80],
            [1800.0, 20.0, 20.0, 0.22, -0.40, 0.85],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    GHI, tilt, az_dev, eta, beta_T, PR = X_phys.T
    tilt_rad = np.deg2rad(tilt)
    az_rad = np.deg2rad(az_dev)
    f_beam = np.cos(tilt_rad - np.deg2rad(35.0)) ** 2 * 1.2
    f_diff = (1.0 + np.cos(tilt_rad)) / 2.0
    f_ground = 0.2 * (1.0 - np.cos(tilt_rad)) / 2.0
    f_tilt = 0.8 * f_beam + 0.15 * f_diff + 0.05 * f_ground
    f_azimuth = np.cos(az_rad) ** 2
    g_eff = GHI * f_tilt * f_azimuth
    t_cell = 25.0 + (20.0 - 60.0 * (eta - 0.20)) * g_eff / 1400.0
    t_corr = 1.0 + beta_T / 100.0 * (t_cell - 25.0)
    e_annual = g_eff * t_corr * PR
    cf = e_annual / 8760.0
    expected = np.stack([e_annual, cf, t_cell], axis=1)

    assert np.allclose(Y, expected, rtol=1e-12, atol=1e-12)


def test_more_negative_temperature_coefficient_reduces_hot_module_yield():
    case = SolarPVYield6D()
    X_phys = np.array(
        [
            [1400.0, 35.0, 0.0, 0.20, -0.20, 0.80],
            [1400.0, 35.0, 0.0, 0.20, -0.50, 0.80],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert Y[0, 2] > 25.0
    assert Y[1, 2] == Y[0, 2]
    assert Y[1, 0] < Y[0, 0]
    assert Y[1, 1] < Y[0, 1]


def test_tilt_has_midlatitude_yield_optimum():
    case = SolarPVYield6D()
    X_phys = np.array(
        [
            [1400.0, 0.0, 0.0, 0.20, -0.35, 0.80],
            [1400.0, 35.0, 0.0, 0.20, -0.35, 0.80],
            [1400.0, 60.0, 0.0, 0.20, -0.35, 0.80],
        ],
        dtype=float,
    )
    y = case.evaluate(case.normalize(X_phys))[:, 0]

    assert y[1] > y[0]
    assert y[1] > y[2]


def test_azimuth_deviation_reduces_yield_and_cell_temperature():
    case = SolarPVYield6D()
    X_phys = np.array(
        [
            [1400.0, 35.0, 0.0, 0.20, -0.35, 0.80],
            [1400.0, 35.0, 30.0, 0.20, -0.35, 0.80],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert Y[1, 0] < Y[0, 0]
    assert Y[1, 1] < Y[0, 1]
    assert Y[1, 2] < Y[0, 2]


def test_annual_azimuth_deviation_factor_is_symmetric_by_construction():
    case = SolarPVYield6D()
    positive = np.array(
        [
            [1400.0, 35.0, 5.0, 0.20, -0.35, 0.80],
            [1400.0, 35.0, 15.0, 0.20, -0.35, 0.80],
            [1400.0, 35.0, 30.0, 0.20, -0.35, 0.80],
        ],
        dtype=float,
    )
    negative = positive.copy()
    negative[:, 2] *= -1.0

    y_positive = case.evaluate(case.normalize(positive))
    y_negative = case.evaluate(case.normalize(negative))

    assert np.allclose(y_positive, y_negative, rtol=1e-12, atol=1e-12)


def test_higher_module_efficiency_reduces_operating_temperature():
    case = SolarPVYield6D()
    X_phys = np.array(
        [
            [1400.0, 35.0, 0.0, 0.14, -0.35, 0.80],
            [1400.0, 35.0, 0.0, 0.25, -0.35, 0.80],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert Y[1, 2] < Y[0, 2]
    assert Y[1, 0] > Y[0, 0]


def test_per_kwp_yield_efficiency_effect_is_only_temperature_coupled():
    case = SolarPVYield6D()
    X_phys = np.array(
        [
            [800.0, 35.0, 0.0, 0.14, -0.35, 0.80],
            [800.0, 35.0, 0.0, 0.25, -0.35, 0.80],
            [2200.0, 35.0, 0.0, 0.14, -0.35, 0.80],
            [2200.0, 35.0, 0.0, 0.25, -0.35, 0.80],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    low_resource_gain = (Y[1, 0] - Y[0, 0]) / Y[0, 0]
    high_resource_gain = (Y[3, 0] - Y[2, 0]) / Y[2, 0]

    assert Y[1, 2] < Y[0, 2]
    assert Y[3, 2] < Y[2, 2]
    assert 0.0 < low_resource_gain < 0.02
    assert 0.0 < high_resource_gain < 0.05
    assert Y[1, 0] / Y[0, 0] < 0.25 / 0.14
    assert Y[3, 0] / Y[2, 0] < 0.25 / 0.14


def test_capacity_factor_matches_specific_yield_definition():
    case = SolarPVYield6D()
    X_phys = np.array([[1400.0, 35.0, 0.0, 0.20, -0.35, 0.80]], dtype=float)
    Y = case.evaluate(case.normalize(X_phys))

    assert np.isclose(Y[0, 1], Y[0, 0] / 8760.0, rtol=1e-12, atol=1e-12)
    assert 0.0 < Y[0, 1] < 1.0
