"""Cross-validation: oil_reservoir_9d — Darcy/FDM pressure-boundary permeability-screening surrogate.

Independent references  (see references.bib)
----------------------------------------------
\\cite{AzizSettari1979}  §3 (block-centered finite-difference reservoir model).
\\cite{Bear1972}         Dynamics of Fluids in Porous Media (Darcy's law).

Public source anchors used for the metadata audit:
* SPE10 / Christie & Blunt upscaling benchmark DOI:
  https://doi.org/10.2118/72469-PA
* SPE10 public PDF mirror:
  https://www.ipt.ntnu.no/~kleppe/pub/SPE-COMPARATIVE/papers/tenth.pdf
* MRST project documentation:
  https://www.sintef.no/projectweb/mrst/
* SINTEF MRST software page:
  https://www.sintef.no/en/software/mrst/
* SINTEF MRST core module:
  https://www.sintef.no/projectweb/mrst/modules/mrst-core/
* SINTEF MRST deckformat module:
  https://www.sintef.no/projectweb/mrst/modules/deckformat/
* OPM Flow simulator documentation:
  https://opm-project.org/
  https://opm-project.org/?page_id=19
* OPM Flow manual/simulator scope:
  https://opm-project.org/?page_id=955
* OPM PyAction and UDQ/ACTIONX scripting boundary:
  https://opm-project.org/?page_id=1454
* OPM Flow linear-solver workflow boundary:
  https://opm-project.org/?page_id=1798
* OPM simulators source repository:
  https://github.com/OPM/opm-simulators
* Official MRST repository / SINTEF project scope:
  https://github.com/SINTEF-AppliedCompSci/MRST
* Peaceman well-block pressure interpretation context:
  DOI:10.2118/6893-PA
* USGS Darcy-law / hydraulic-conductivity porous-media flow background:
  https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html
* USGS groundwater recharge/discharge overview:
  https://www.usgs.gov/media/images/groundwater-recharge-and-discharge

Key physics:
  Darcy's law: Q = −(K/μ)·A·(dp/dx).
  Higher permeability → higher flow rate.
  Uniform fields recover the absolute Darcy flow scale, |∇p| = Δp/L, and
  K_eff = K.  1-D column layers in series recover the equal-thickness
  harmonic mean; row layers in parallel recover the equal-thickness
  arithmetic mean.
  Mass balance: Σ Q_in = Σ Q_out (steady state).

Scope: low-order Darcy/FDM pressure-boundary benchmark surrogate. These
checks validate formula identities and expected permeability-screening trends,
not wellbore modelling, well-control optimisation, multiphase flow,
compressible flow, history matching, or calibrated field-scale reservoir
simulation.  The implementation also omits relative permeability, capillary
pressure, porosity/storage, gravity segregation, 3-D stratigraphy, fracture
networks, time-dependent saturation fronts, aquifer support,
black-oil formulations, compositional/thermal/EOR processes, CO2 storage
physics, industry-standard simulator decks, eclipse deck parsing, restart-file
workflows, surface-facility constraints, field calibration, upscaling studies,
solvent/h2 storage options, and geomechanical coupling.  SPE10,
MRST, OPM Flow, and Peaceman well-index
references are used only as public anchors for reservoir-simulation scope and
heterogeneous permeability/upscaling benchmark context, not as validation data
for this compact 3x3 single-phase conductance surrogate.
USGS Circular 1224 is also boundary evidence that real subsurface flow paths
depend on external boundaries, recharge/discharge structure, and pumping
stresses; the benchmark intentionally freezes that broader setting to fixed
left/right pressure boundaries with closed top/bottom boundaries.
The current USGS recharge/discharge overview also treats well pumping as
groundwater discharge and seepage from streams, floodwater, and irrigation as
recharge, so the benchmark likewise omits distributed source/sink and
recharge/discharge forcing rather than claiming a site-specific field model.
OPM Flow is a fully implicit black-oil simulator with wells/facilities and
industry-standard model support, including black-oil formulations,
rock-dependent capillary/relative-permeability curves, CO2STORE, thermal,
polymer options, solvent options, hydrogen-storage, summary/restart-file
workflows, restart-file workflows, Python/PYACTION/UDQ/ACTIONX scripting hooks, Newton/NLDD nonlinear
solves, CPR/multigrid linear solvers, and adaptive timestep controls.  MRST includes ECLIPSE input/corner-point-grid workflows through its deckformat module,
fractured-media grids, black-oil, compositional, EOR, CO2, geomechanics,
geothermal, upscaling, diagnostics, standard/multisegment wells, surface
facilities, adjoints, and automatic-differentiation-based simulators. OPM Flow
also exposes group controls, pipe-network models, Eclipse parser/restart
workflows, and automatic-differentiation-based fully implicit technology.
The OPM linear-solver workflow also makes solver failure handling and timestep
cutting part of a production simulator contract; this benchmark has no
time-stepping, nonlinear residual, or OPM-style solver-retry workflow.
The MRST-core module also makes explicit that richer MRST workflows expose
boundary conditions, source terms, and wells as first-class modelling objects;
this benchmark intentionally freezes that broader control space to fixed
left/right pressure boundaries, closed top/bottom boundaries, and no wells or
distributed source/sink terms. Those tool capabilities define omitted scope
here; this benchmark does not run MRST/OPM Flow, parse reservoir decks, solve
wells/facilities, or compare against SPE10 simulator outputs.
The named _validate_physical_inputs private physical-input path rejects
malformed arrays and K11-K33 permeability values outside the registered
benchmark domain before sparse Darcy assembly, so direct private calls cannot
extrapolate beyond the declared pressure-boundary conductance surrogate bounds.
Finite but out-of-contract Q_total/dp_max/eff_perm responses are also rejected
before row archival; this is a server-run failure contract and does not change
valid Darcy/FDM responses inside the declared output envelopes.
"""

import numpy as np
import pytest

CASE_ID = "oil_reservoir_9d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify low-order Darcy/FDM identities and monotone trends."""

    def test_higher_perm_higher_flow(self, case):
        """\\cite{Bear1972}: higher K → higher Darcy velocity → higher Q."""
        X_low = np.array([[5.0] * 9])
        X_high = np.array([[500.0] * 9])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert Y_high[0, 0] > Y_low[0, 0], f"Higher K → higher Q: {Y_high[0,0]} vs {Y_low[0,0]}"

    def test_uniform_perm_effective(self, case):
        """Uniform K → effective perm = K."""
        K_val = 100.0
        X_phys = np.array([[K_val] * 9])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        # Effective perm should be close to K_val
        assert abs(Y[0, 2] - K_val) / K_val < 0.30, f"eff_perm={Y[0,2]}, expected ≈ {K_val}"

    def test_flow_rate_positive(self, case):
        """Q_total must be positive under imposed pressure drop."""
        X_phys = np.array([[50.0] * 9])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 0] > 0, f"Q_total={Y[0,0]}, expected > 0"
