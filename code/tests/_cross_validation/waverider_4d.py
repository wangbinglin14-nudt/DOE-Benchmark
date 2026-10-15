r"""Cross-validation: waverider_4d - modified-Newtonian waverider surrogate.

Independent references  (see references.bib)
----------------------------------------------
\\cite{AndersonHypersonic2006}  §3.5 (modified Newtonian theory), §4 (waveriders).
\\cite{Anderson2010}            §14 (hypersonic aerothermodynamics).
NASA TP-3559 and NASA CR-182076 provide public waverider and modified-Newtonian
method context, including the caveat that integrated hypersonic configurations
depend on base drag, viscous effects, and component build-up.
Public NTRS metadata for NASA TP-3559 lists a design Mach number of 4.0
and experimental / limited CFD data over Mach 1.6-4.63.  That is below
this case's registered Mach 5-12 box, so TP-3559 is context and
order-of-magnitude evidence rather than domain-matched validation.
NACA Report 1135 and NASA Glenn normal-shock material provide public
compressible-flow / Rayleigh-Pitot context for the fixed-gamma Cpmax formula.
NASA Glenn real-gas material is treated as boundary evidence for omitted
thermochemistry, variable-specific-heat effects, vibrational excitation,
dissociation, ionization, and chemical/thermal nonequilibrium, not as
validation of this fixed-gamma calorically-perfect surrogate.
NASA Glenn Mach-number material independently anchors the Ma > 5 hypersonic
classification and shock/high-temperature gas-effect boundary for this case's
registered Mach 5-12 box, without validating the fixed-gamma response surface.
NASA Glenn's hypersonic-missions page is additional official boundary evidence
that Mach 5-7 vehicle design must evaluate real-gas effects, so the benchmark's
fixed-gamma Mach 5-12 surrogate should not be read as mission-grade
hypersonic design analysis.
NASA Glenn's re-entry-aircraft page is additional boundary evidence that
hypersonic flight can involve strong shock waves, plasma, thermal protection,
and deliberately high-drag trajectories; those are not outputs of this compact
waverider surrogate.
NASA X-43A/Hyper-X material and NASA's X-43A Mach-7 CFD image provide official
boundary context for airframe-integrated scramjet flight, additional thermal
protection, engine-operating CFD, and post-engine hypersonic aerodynamic data.
NASA Glenn's Hypersonic Tunnel Facility page provides official boundary context
for Mach 5-7 true-enthalpy ground testing. These sources strengthen the
omitted-workflow boundary; they do not validate this fixed-gamma algebraic
surrogate.
NASA Glenn's 1x1 Supersonic Wind Tunnel page is an additional official
boundary source for benchmark-quality CFD-validation experiments in
supersonic and hypersonic fluid mechanics, with continuous Mach 1.3-6.0
operation.  That facility context is what external validation would have to
look like; it is not validation data for the registered response surface.
NASA Ames arc-jet / Interaction Heating Facility material and NASA FIAT
thermal-response software material are boundary evidence for aerothermal
qualification, thermal-protection testing, surface heat flux, material thermal
response, and ablation analysis. Those workflows are outside this aerodynamic
L/D surrogate and are not validation data for it.
NASA Hypersonic Technology Project material is current NASA program-scope
boundary evidence for reusable airbreathing hypersonic-flight research,
system-level design, analysis, and validation, propulsion, vehicle technologies,
and high-temperature durable materials.  It is not case-level validation data
for this four-input algebraic L/D surrogate.

Public source anchors:
  NASA TP-3559, waverider-derived hypersonic cruise configurations:
    https://ntrs.nasa.gov/api/citations/19960045290/downloads/19960045290.pdf
  NASA CR-182076, Aerodynamic Preliminary Analysis System II:
    https://ntrs.nasa.gov/api/citations/19910013767/downloads/19910013767.pdf
  NACA Report 1135, compressible-flow tables:
    https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf
  NASA Glenn Normal Shock Wave Equations:
    https://www.grc.nasa.gov/WWW/BGH/normal.html
  NASA Glenn Real Gas Effects:
    https://www.grc.nasa.gov/www/BGH/realgas.html
  NASA Glenn Mach Number:
    https://www.grc.nasa.gov/www/BGH/mach.html
  NASA Glenn Hypersonic Missions:
    https://www.grc.nasa.gov/WWW/BGH/hmission.html
  NASA Glenn Re-Entry Aircraft:
    https://www.grc.nasa.gov/www/BGH/hihyper.html
  NASA X-43A Hyper-X:
    https://www.nasa.gov/reference/x-43a/
  NASA CFD Image of X-43A at Mach 7 Test Condition:
    https://www.nasa.gov/image-article/cfd-image-of-x-43a-mach-7-test-condition/
  NASA Glenn Hypersonic Tunnel Facility:
    https://www.nasa.gov/centers-and-facilities/glenn/hypersonic-tunnel-facility/
  NASA Glenn 1x1 Supersonic Wind Tunnel:
    https://www.nasa.gov/centers-and-facilities/glenn/1x1-supersonic-wind-tunnel/
  NASA Ames Interaction Heating Facility / Arc Jet Complex:
    https://www.nasa.gov/ames/interaction-heating-facility/
  NASA FIAT thermal response / ablation software:
    https://software.nasa.gov/software/ARC-15779-1A
  NASA Hypersonic Technology Project:
    https://www.nasa.gov/directorates/armd/aavp/ht/
  NASA Hypersonic Research Topics:
    https://www.nasa.gov/directorates/armd/aavp/ht/hypersonic-research-topics/

Key physics:
  At hypersonic speeds, the Rayleigh-Pitot modified-Newtonian Cp_max
  used by this case assumes fixed γ=1.4 and approaches about 1.84.
  The classical
  Newtonian coefficient 2 is a separate idealised upper bound.
  L/D_max for waveriders: typically 4–8 at Ma 5–10 (Anderson §4).
  In this model, L/D increases with Ma because an empirical
  van-Driest-inspired skin-friction factor drops as (1 + 0.02·Ma²)^{-0.3}
  while Cp_max saturates.  This is not a full van Driest II boundary-layer
  calculation.

Validation scope: qualitative hypersonic trend, Rayleigh-Pitot Cpmax, and
representative L/D magnitude controls for a modified-Newtonian surrogate.
This file must not be read as quantitative external validation of a
CFD-calibrated or wind-tunnel-calibrated waverider design model.
It is not a shock-attachment verification, leading-edge-bluntness, real-gas,
vibrational-excitation, dissociation, ionization, chemical/thermal-nonequilibrium,
plasma, rarefied-flow, trim/control, stability, propulsion-integration, TPS, or
integrated-vehicle performance validation. It is also not a Hyper-X/X-43A
airframe-integrated scramjet, thermal-protection, engine-operating CFD, or
Mach 5-7 true-enthalpy facility-test validation.
It is also not a NASA Glenn 1x1-style benchmark-quality CFD-validation
experiment, wind-tunnel database, not a NASA Ames arc-jet TPS test, and not a
FIAT-style thermal-response/ablation analysis.
Scenario notes are benchmark-use rationales, not claims that a specific
sample count is sufficient before the server results are available.

The public NASA/NACA anchors are order-of-magnitude, method-context,
compressible-flow formula, and omitted-physics boundary evidence only. They
are not external validation data for this fixed-gamma modified-Newtonian
surrogate, empirical friction term,
full Mach-domain response, shock-attached design accuracy,
CFD/wind-tunnel/arc-jet calibration, mission-grade real-gas hypersonic design,
TPS material response, ablation analysis, or integrated hypersonic vehicle
performance.
They also do not validate NASA Hypersonic Technology Project-style
system-level design/analysis/validation, propulsion, vehicle-technology, or
high-temperature durable-material workflows.
The NASA Glenn Mach-number source is likewise classification and scope
evidence only; it does not validate the Rayleigh-Pitot Cpmax implementation,
empirical friction term, or any high-temperature gas calculation.
The named _validate_physical_inputs private physical-input path rejects malformed arrays and
Ma/theta_c/alpha/phi values outside the declared benchmark domain before
modified-Newtonian formula evaluation, so direct private calls cannot
extrapolate beyond the registered hypersonic surrogate bounds.  The public
evaluate path also rejects non-finite, non-positive, or out-of-contract
L_D/CL/eta_v responses before a row can enter a benchmark archive.
"""

import numpy as np
import pytest

CASE_ID = "waverider_4d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    r"""Verify against \cite{AndersonHypersonic2006} waverider theory."""

    def test_ld_range(self, case):
        """L/D should be in a physically reasonable range (2–12) at Ma=6."""
        X_phys = np.array([[6.0, 10.0, 2.0, 60.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert 1.0 < Y[0, 0] < 15.0, f"L/D={Y[0,0]}, expected 3–8"

    def test_higher_mach_higher_ld(self, case):
        """In the modified-Newtonian model with Van Driest friction, higher Ma
        → lower Cf → higher L/D (Cp_max saturates while friction decays)."""
        X_low = np.array([[5.0, 10.0, 2.0, 60.0]])
        X_high = np.array([[11.0, 10.0, 2.0, 60.0]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert (
            Y_high[0, 0] > Y_low[0, 0]
        ), f"Higher Ma → higher L/D: {Y_high[0,0]:.4f} vs {Y_low[0,0]:.4f}"

    def test_positive_cl(self, case):
        """At positive α, CL should be positive."""
        X_phys = np.array([[8.0, 12.0, 3.0, 70.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 1] > 0, f"CL={Y[0,1]}, expected > 0"
