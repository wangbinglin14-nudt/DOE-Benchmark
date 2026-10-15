"""Cross-validation: solar_pv_yield_6d — PVWatts-style annual yield surrogate.

Independent references  (see references.bib)
----------------------------------------------
\\cite{DuffieBeckman2013}  §2 (Liu–Jordan model), §23 (PV system performance).

Key physics:
  Specific yield E_annual ∝ GHI · PR; η_STC cancels in the kWh/kWp
  normalisation and only enters through the temperature correction.
  η_STC is therefore below the important-variable threshold for E_annual,
  but it is still part of weak temperature-correction interaction structure.
  Higher GHI → more energy.
  Higher tilt (up to latitude) → better capture.
  Higher PR → less system losses → more energy.
  The annual azimuth-deviation factor is symmetric by construction:
  because the registered surrogate uses cos^2(gamma_dev), eastward and
  westward deviations of equal magnitude give identical annual per-kWp
  responses. This is a surrogate identity, not a claim about hourly or
  site-specific east/west asymmetry.
  For a per-kWp specific yield, the benchmark stores dimensionless
  CF = E_annual / 8760 rather than the percent-form PVWatts V8 API
  capacity_factor output.

Scope: low-order PVWatts-style algebraic surrogate. These checks validate
formula identities and expected PV performance trends for a fixed-parameter,
Liu-Jordan-inspired annual transposition approximation. They do not call or
validate the current PVWatts V8 API/SSC model, PVWatts Calculator, PVWatts V5,
SAM PVWatts, SAM detailed PV, the shared PVWatts V8/pvwattsv8 hourly
toolchain, NSRDB/TMY weather-file lookup, PVWatts
module/array/GCR/DC-AC-ratio/inverter-efficiency inputs, albedo, bifaciality,
PVWatts V8 detailed module/inverter/thermal-effect updates, NSRDB 2020 TMY
weather updates, PSM V4 Meteosat/Polar TMY dataset selection, monthly
soiling/loss inputs, monthly/hourly PVWatts outputs,
pvlib/Perez/Hay-Davies/Reindl, pvlib PVWatts DC/inverter/loss functions,
pvlib ModelChain, NOCT/Fuentes thermal-model, single-diode,
DC/AC sizing, inverter clipping, inverter, weather-file/module/inverter/system
design workflow, shading/layout losses, IAM, spectral, soiling, snow,
degradation, bifacial, battery, grid-curtailment, PVPMC/Sandia
performance-index or uncertainty workflows, measured-plant monitoring,
expected-energy contracts, or bankable site-assessment validation.
Current official NREL help also shows that the omitted workflow boundary is
broader than the basic PVWatts formula list: weather-file choice, module and
inverter model selection, shading/layout losses, snow-depth-dependent shading,
hourly AC/POA/module-temperature outputs, and inverter undersizing warnings
all belong to the real PVWatts/SAM toolchain rather than this annual algebraic
surrogate.
Current pvlib documentation likewise sharpens the omitted toolchain boundary:
pvlib exposes PVWatts DC, inverter, and loss functions, and ModelChain can
assemble solar-position, airmass, AOI, spectral, temperature, DC, AC, and loss
models. This benchmark does not call pvlib or reproduce that chained workflow.
The official PVWatts/SAM web pages are tracked at their canonical NREL URLs
(for example developer.nrel.gov and samrepo.nrelcloud.org) so the validation
record stays auditable even if legacy URLs redirect. This is a traceability
note, not a response-surface claim. The same API boundary exposes
PSM V3 GOES/Himawari `tmy-2020`, PSM V4 Meteosat Prime Meridian `tmy-2022`,
PSM V4 Polar `tmy-2023`, `timeframe=monthly|hourly`, `weather_data_source`,
and hourly AC/POA/DC/ambient-temperature/cell-temperature traces that this
benchmark does not emit.
The official PVWatts Version 8 calculator page is used as a current boundary
source for Version 8.5.2, release date 2025-09-25, SSC version 275,
`pvwattsv8`, NSRDB 2020 TMY inputs, bifacial/albedo/monthly-loss features, and
PVWatts uncertainty cautions; it is not used as response-surface validation.
pvlib PVWatts and ModelChain documentation is used only as omitted-workflow
boundary evidence, not as pvlib validation or PVWatts-function equivalence.
PVPMC/Sandia annual-yield and performance-ratio pages are used only as metric
definition anchors for kWh/kWp annual yield and PR-style loss factors. They are
not treated as evidence that this fixed-parameter annual surrogate reproduces
measured plant monitoring, seasonal temperature-normalized PR analysis,
expected-energy baselines, uncertainty analysis, or contractual performance
index workflows.
The named _validate_physical_inputs private physical-input path rejects
malformed arrays and GHI_annual/tilt/azimuth_dev/eta_STC/beta_T/PR values
outside the registered benchmark domain before PVWatts-style annual-yield
response assembly, so direct private calls cannot extrapolate beyond the
declared annual PV-yield surrogate bounds.
Finite but out-of-contract E_annual/CF/T_cell_avg responses are also rejected
before row archival; this is a server-run data-integrity contract and does not
change valid annual PV-yield responses inside the declared output envelopes.
"""

import numpy as np
import pytest

CASE_ID = "solar_pv_yield_6d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify low-order PV yield identities and performance trends."""

    def test_higher_ghi_more_energy(self, case):
        """More solar resource → more annual energy."""
        X_low = np.array([[900.0, 30.0, 0.0, 0.20, -0.4, 0.80]])
        X_high = np.array([[2000.0, 30.0, 0.0, 0.20, -0.4, 0.80]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert Y_high[0, 0] > Y_low[0, 0]

    def test_higher_pr_more_energy(self, case):
        """Higher performance ratio → less losses → more output."""
        X_low = np.array([[1500.0, 25.0, 0.0, 0.18, -0.35, 0.72]])
        X_high = np.array([[1500.0, 25.0, 0.0, 0.18, -0.35, 0.88]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert Y_high[0, 0] > Y_low[0, 0]

    def test_stc_efficiency_has_only_temperature_coupled_effect(self, case):
        """Specific yield is per kWp, so η_STC does not scale output directly."""
        X_low_eta = np.array([[1500.0, 30.0, 0.0, 0.14, -0.4, 0.80]])
        X_high_eta = np.array([[1500.0, 30.0, 0.0, 0.25, -0.4, 0.80]])
        Y_low = case.evaluate(case.normalize(X_low_eta))
        Y_high = case.evaluate(case.normalize(X_high_eta))

        assert Y_high[0, 2] < Y_low[0, 2]
        assert 0.0 < (Y_high[0, 0] - Y_low[0, 0]) / Y_low[0, 0] < 0.05

    def test_cf_bounded(self, case):
        """Capacity factor must be in (0, 1)."""
        X_phys = np.array([[1500.0, 30.0, 0.0, 0.20, -0.4, 0.80]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert 0 < Y[0, 1] < 1.0, f"CF={Y[0,1]}, expected in (0, 1)"
