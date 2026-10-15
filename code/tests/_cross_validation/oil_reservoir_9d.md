# Cross-Validation: `oil_reservoir_9d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `oil_reservoir_9d` |
| Implemented solver | Compact 3x3 single-phase incompressible Darcy pressure-boundary conductance surrogate |
| Evidence tier | P2 plausibility-only |
| Validation scope | Uniform-field Darcy scale, series/parallel layer controls, permeability monotonicity, mass-balance diagnostics, simulator-boundary, and omitted reservoir-workflow controls. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/oil_reservoir_9d.py` and
`tests/test_oil_reservoir_physics.py`. It clarifies that the case is a
permeability-screening stressor, not a field-scale reservoir simulator,
history-matching workflow, or oil-production forecast.

## 2. Implemented Formula Class

The case solves one steady sparse linear pressure system for a 3x3 block
permeability field:

```text
q = -(K/mu) grad p
left boundary  = fixed injection-side pressure
right boundary = fixed production-side pressure
top/bottom     = closed
Q_total        = mean internal vertical Darcy-face conductance
eff_perm       = Darcy-equivalent permeability from Q_total
```

All nine permeability blocks are active in the response surface. The case has
no wells, no distributed source/sink terms, and no saturation time stepping.
Because the pressure-boundary Darcy equation is homogeneous in K for fixed
Dirichlet pressures, multiplying all nine block permeabilities by one common
factor leaves the pressure field and `dp_max` invariant while scaling
`Q_total` and `eff_perm` by that same factor. The physics regression suite
pins this global scaling control for a heterogeneous field, complementing the
uniform-field scale and layered series/parallel controls.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| SPE10 / Christie-Blunt benchmark DOI: https://doi.org/10.2118/72469-PA | Heterogeneous-reservoir and upscaling benchmark context. | SPE10-validated response surface or simulator-comparison output. |
| SPE10 public PDF mirror: https://www.ipt.ntnu.no/~kleppe/pub/SPE-COMPARATIVE/papers/tenth.pdf | Public traceability for SPE10 context. | Validation of this 3x3 surrogate. |
| MRST project page: https://www.sintef.no/projectweb/mrst/ | Boundary evidence for reservoir-simulation workflows. | MRST-equivalent simulation. |
| SINTEF MRST software page: https://www.sintef.no/en/software/mrst/ | Boundary evidence for MRST capabilities. | Execution of MRST by this case. |
| MRST core module: https://www.sintef.no/projectweb/mrst/modules/mrst-core/ | Boundary evidence for boundary-condition, source-term, and well workflow objects. | Validation of the fixed-boundary no-well surrogate. |
| MRST deckformat module: https://www.sintef.no/projectweb/mrst/modules/deckformat/ | Boundary evidence for ECLIPSE deck/corner-point-grid workflows. | Deck parsing or restart workflow by this case. |
| OPM project and Flow docs: https://opm-project.org/ and https://opm-project.org/?page_id=19 | Boundary evidence for OPM Flow simulator scope. | OPM-equivalent reservoir simulation. |
| OPM Flow manual/scope: https://opm-project.org/?page_id=955 | Boundary evidence for black-oil, wells/facilities, industry-standard models, EOR, solvent, H2/CO2, and restart workflows. | Black-oil, multiphase, compositional, thermal, or EOR validation. |
| OPM PyAction/UDQ/ACTIONX docs: https://opm-project.org/?page_id=1454 | Boundary evidence for simulator scripting and user-defined action workflows. | Any OPM scripting, deck action, or control-logic execution by this case. |
| OPM linear-solver workflow: https://opm-project.org/?page_id=1798 | Boundary evidence for nonlinear solver, CPR/multigrid, and adaptive-timestep workflows. | OPM-style solver-retry or timestep-cut validation. |
| OPM simulators source: https://github.com/OPM/opm-simulators | Public traceability for simulator implementation scope. | Source-code equivalence with this benchmark. |
| MRST GitHub repository: https://github.com/SINTEF-AppliedCompSci/MRST | Public traceability for MRST scope. | MRST validation of the registered response. |
| Peaceman well-block pressure DOI: https://doi.org/10.2118/6893-PA | Well-index context. | Wellbore modelling by this case. |
| USGS Circular 1224: https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html | Darcy-law and boundary/recharge/discharge context. | Site-specific subsurface flow validation. |
| USGS recharge/discharge image: https://www.usgs.gov/media/images/groundwater-recharge-and-discharge | Boundary evidence for well pumping and recharge/discharge forcing. | Distributed source/sink validation. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not SPE10-validated, not
MRST/OPM-equivalent, not black-oil validated, not multiphase validated, not
compositional/thermal/EOR validated, not CO2/H2-storage validated, not
well/facility validated, not deck/restart validated, not upscaling validated,
not history-matched, not field-calibrated, and not oil-production-forecast
output.

It omits Peaceman well indices, well controls, relative permeability, capillary
pressure, compressibility, porosity/storage, gravity segregation, 3-D
stratigraphy, fractures, saturation fronts, aquifer support, black-oil
formulations, compositional/thermal/EOR processes, CO2/H2 storage, industry
simulator decks, ECLIPSE deck parsing, restart files, surface facilities,
standard/multisegment wells, polymer/surfactant/solvent options, geothermal
energy equations, OPM Python/PYACTION/UDQ/ACTIONX workflows, Newton/NLDD
nonlinear solvers, CPR/multigrid linear solvers, adaptive timestep controls,
history matching, field calibration, upscaling studies, geomechanical coupling,
recharge/discharge forcing, and distributed source/sink terms.

These anchors support Darcy plausibility and honest simulator/workflow
boundaries. They do not support a strong external-validation claim unless a
future registered validation-note folder adds domain-matched simulator,
SPE10-style, field-calibrated, or upscaling reference comparisons.
