# Cross-Validation: `dam_seepage_6d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `dam_seepage_6d` |
| Implemented solver | Two-dimensional steady Darcy/FDM uplift-pressure surrogate with grid-aligned curtain and single pressure-relief drain node |
| Evidence tier | P2 plausibility-only |
| Validation scope | Homogeneous no-drain linear-head control, uplift monotonicity, drain activation boundary, source-boundary, and omitted dam-safety workflow controls. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/dam_seepage_6d.py` and
`tests/test_dam_seepage_physics.py`. It clarifies that the case is an
uplift-pressure stress test, not a code-compliant dam-foundation seepage
design calculation.

## 2. Implemented Formula Class

The case solves a compact fixed-grid Darcy head problem:

```text
div(K(x,z) grad H) = 0
u(x)      = gamma_w * H(x, z=0)
U_total   = integral u(x) dx
U_center  = u(L_base/2)
eta_uplift = U_total / U_no_curtain
```

The drain is a benchmark pressure-relief surrogate. It activates only when the
prescribed relief head is below the no-drain natural head at that location, so
it cannot act as a recharge source in strong-curtain cases.

For a homogeneous no-drain slice with `K_base == K_curtain`, the head equation
reduces to a constant-coefficient Laplace problem.  The targeted tests now pin
that the dam-base head profile is linear between `H1` and `H2`, independent of
foundation depth, and independent of the absolute permeability scale.  This
checks the Darcy/FDM implementation's limiting behaviour without claiming
site-specific permeability calibration.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| USACE EM 1110-2-2200 Gravity Dam Design: https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2200.pdf | Gravity-dam design and uplift context. | Validation of this fixed-grid Darcy/FDM response surface. |
| USACE EM 1110-2-1901 Seepage Analysis and Control: https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-1901.pdf | Seepage-analysis and control-workflow boundary evidence. | Calibration of the single-node drain surrogate. |
| ASDSO seepage analysis/control summary: https://damsafety.org/content/seepage-analysis-and-control-dams | Monitoring, inspection, filter, drain, piping, and erosion-control boundary context. | External validation of benchmark uplift reduction. |
| USACE EM 1110-2-1914 relief wells: https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM%201110-2-1914_FINAL%2020250307.pdf | Relief-well design/construction/maintenance boundary evidence. | Validation of the benchmark's single drain node. |
| USACE Digital Library relief-well record: https://usace.contentdm.oclc.org/digital/collection/p16021coll9/id/2595/ | Public traceability anchor for relief-well guidance. | Site-specific relief-system validation. |
| USACE ERDC/GSL TR-25-5 relief-well current practice: https://www.erdc.usace.army.mil/Media/Publication-Notices/Article/4139695/usace-relief-wells-for-dams-and-levees-history-and-current-practice/ | Current-practice boundary evidence for relief-well maintenance/repair, pressure-relief, and seepage-remediation workflows. | Calibration or validation of the benchmark's single pressure-relief node. |
| USBR Design Standards No. 13, Chapter 8: https://www.usbr.gov/tsc/techreferences/designstandards-datacollectionguides/finalds-pdfs/DS13-8.pdf | Seepage and uncertainty boundary evidence. | Dam-specific calibration or mesh-converged FE validation. |
| FERC Engineering Guidelines, Chapter 3: https://www.ferc.gov/sites/default/files/2020-04/chap3.pdf | Straight-line uplift baseline and instrumentation boundary evidence. | Verified drainage-credit claim for this case. |
| FERC RIDM: https://www.ferc.gov/dam-safety-and-inspections/risk-informed-decision-making-ridm | Risk-informed dam-safety workflow boundary. | Risk likelihood, consequence, or system-response modelling. |
| FERC potential failure modes: https://www.ferc.gov/dam-safety-and-inspections/potential-failure-modes-pfms | Potential-failure-mode boundary context. | PFM/PFMA classification by this case. |
| FEMA P-1032 seepage/internal-erosion monitoring: https://www.fema.gov/sites/default/files/2020-08/fema_p1032_eval_monitoring_seepage_internal_erosion.pdf | Boundary evidence for seepage observation, flow/pressure/water-quality monitoring, PFM identification, life-cycle monitoring, emergency response, and remediation workflows. | Internal-erosion evaluation, monitoring-program validation, or remediation design by this benchmark. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not a calibrated seepage-discharge
model, not instrumented uplift validation, not a flow-net validation, not
relief-well design validation, not a filter-design workflow, not
piping/internal-erosion validation, not a dam-safety RIDM/PFM workflow, and not
a structural sliding/overturning stability model.

It omits seepage velocities, exit-gradient and piping indices, unsaturated
flow, transient reservoir operation, anisotropy, fracture/joint flow, 3-D
abutment flow, full pore-pressure-field reporting, piezometer interpretation,
permeability-uncertainty sensitivity studies, mesh convergence, drainage
gallery hydraulics, collector-pipe/header layout, relief-well spacing/sizing,
development/pump-testing, maintenance and rehabilitation planning, risk
likelihood, system response, failure consequences, surveillance, inspection,
instrumentation programs, and risk-reduction decisions.
It also omits FEMA P-1032-style seepage/internal-erosion evaluation,
flow-rate, pressure, and water-quality monitoring, PFM identification,
life-cycle monitoring, emergency response, and remediation decisions.
The 2025 USACE ERDC relief-well current-practice report reinforces that
relief-well maintenance/repair and seepage-remediation practice are omitted
workflow boundaries, not validation of the compact response surface.

These anchors support transparent seepage/uplift plausibility and honest
dam-safety workflow boundaries. They do not support a strong
external-validation claim unless a future registered validation-note folder
adds domain-matched flow-net, FE, instrumented dam, or calibrated seepage
reference comparisons.
