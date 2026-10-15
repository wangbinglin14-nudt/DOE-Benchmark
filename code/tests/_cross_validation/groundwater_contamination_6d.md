# Cross-Validation: `groundwater_contamination_6d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `groundwater_contamination_6d` |
| Implemented solver | One-dimensional Ogata-Banks-style advection-dispersion surrogate with linear retardation and separable first-order decay |
| Evidence tier | P2 plausibility-only |
| Validation scope | Analytical ADE identity checks, unit conventions, stable reflection-term evaluation, plume-front trends, simulator-boundary, and omitted reactive-transport workflow controls. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/groundwater_contamination_6d.py` and
`tests/test_groundwater_contamination_physics.py`. It clarifies that the case
is a one-dimensional transport stressor, not a field-plume, groundwater
vulnerability, or remediation-planning model.

## 2. Implemented Formula Class

The case uses a direct Ogata-Banks-style response with benchmark-specific
decay handling:

```text
D = alpha_L * v_pore + D_m * tau
R = 1 + rho_b*K_d/(1000*n_por)
C(x,t)/C0 = 0.5 * [erfc(A1) + exp(v*x/D)*erfc(A2)] * exp(-lambda*t/R)
A1 = (R*x - v*t)/(2*sqrt(D*R*t))
A2 = (R*x + v*t)/(2*sqrt(D*R*t))
```

The reflection term is evaluated with a stable `erfcx` expression to avoid
legal-domain overflow. The model is not the exact decaying constant-source ADE
solution; the decay is a separable attenuation used for benchmark stress
testing.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| USGS PP-411-A Ogata-Banks public PDF: https://pubs.usgs.gov/pp/0411a/report.pdf | One-dimensional ADE analytical formula anchor. | Field-plume calibration or full reactive-transport validation. |
| USGS Circular 1224 groundwater-flow overview: https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html | Flow-system boundary, average velocity, and pumping-stress context. | Site-specific groundwater vulnerability assessment. |
| USGS PHREEQC advection-reaction-dispersion equation: https://www.usgs.gov/software/phreeqc-version-3 | Formula context for treating `D_L` as a longitudinal hydrodynamic dispersion coefficient with dispersivity; supports the local regression that increasing `alpha_L` broadens the threshold front and advances first arrival at a monitoring point. | PHREEQC-equivalent reactive-transport chemistry, numerical model validation, or calibration of the benchmark plume. |
| MT3DMS model-use/calibration/validation: https://www.usgs.gov/publications/mt3dms-model-use-calibration-and-validation | Boundary evidence for calibrated 3-D multispecies transport workflows. | Claim that this case is MT3DMS validated. |
| MT3D-USGS documentation: https://pubs.usgs.gov/tm/06/a53/tm06a53.pdf | Boundary evidence for MODFLOW-linked solute-transport features. | MT3D-USGS-equivalent simulation. |
| MT3D-USGS software page: https://www.usgs.gov/software/mt3d-usgs-groundwater-solute-transport-simulator-modflow | Boundary evidence for package-level simulator capabilities. | Validation of this one-dimensional surrogate. |
| MODFLOW 6 GWT documentation: https://pubs.usgs.gov/publication/tm6A61 | Boundary evidence for GWF-coupled 3-D transport. | MODFLOW 6 GWT-equivalent execution. |
| MODFLOW 6 software overview: https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model | Boundary evidence for the broader modular groundwater platform. | Platform-level GWF/GWT/GWE/PRT coupling by this case. |
| MODFLOW 6 input/output docs: https://modflow6.readthedocs.io/en/latest/mf6io.html | Boundary evidence for stress packages, mover routing, particle tracking, and heat/energy transport. | MODFLOW stress-package validation. |
| USGS principal aquifers quality overview: https://www.usgs.gov/programs/environmental-health-program/science/groundwater-quality-principal-aquifers-united-states | Boundary evidence that irrigation, pumping, artificial recharge, and drainage can alter flow and quality. | Site-specific vulnerability or remediation-planning validation. |
| EPA Superfund groundwater contamination/remedy pages: https://www.epa.gov/superfund/how-superfund-addresses-groundwater-contamination, https://www.epa.gov/superfund/groundwater-technologies, and https://www.epa.gov/superfund/selecting-groundwater-remedy | Boundary evidence for plume containment, pump-and-treat, in situ treatment, monitored natural attenuation, institutional controls, alternative water supply, remedy selection, cleanup goals, and remedy-performance workflows. | Superfund remedy selection, cleanup-performance validation, or MNA validation by this benchmark. |
| EPA capture-zone, groundwater-modeling, and remedy-technology pages: https://cfpub.epa.gov/si/si_public_record_report.cfm?Lab=NRMRL&dirEntryId=187788, https://www.epa.gov/land-research/ground-water-modeling-research, and https://www.epa.gov/remedytech/remediation-technologies-cleaning-contaminated-sites | Boundary evidence for pump-and-treat capture-zone evaluation, converging lines of evidence, site characterization, monitoring, treatment, source/plume remediation modeling, monitored natural attenuation, permeable reactive barriers, in situ chemical oxidation/reduction, in situ thermal treatment, phytotechnologies, soil vapor extraction/air sparging, solidification/stabilization, vapor intrusion mitigation, and vertical engineered barriers. | Hydraulic-containment, extraction-well capture-zone, contaminated-site technology-selection, or site-support modeling validation. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not field-plume calibrated, not
MT3DMS/MT3D-USGS/MODFLOW-6-equivalent, not MODFLOW-coupled 3-D reactive
transport, not multispecies reaction transport, not site-specific groundwater
vulnerability assessment, not mesh/time-step converged simulator output, and
not a validated contaminant-remediation model.

It omits 2-D/3-D groundwater flow, heterogeneous aquifers, transient flow,
source depletion, finite-duration releases, site-specific boundary conditions,
wells or pumping stresses, MODFLOW-coupled hydraulic gradients, recharge and
evapotranspiration, variable water table, transverse/vertical dispersion,
nonlinear or nonequilibrium sorption, dual-domain mass transfer, immobile
storage, matrix diffusion, multispecies reactions, density-dependent flow,
geochemical speciation, source-strength uncertainty, numerical mesh/time-step
convergence, field-plume calibration, advanced stress-package transport,
source/sink mixing, unsaturated-zone transport, stream/lake exchange,
parent-daughter reactions, pump-and-treat recirculation, mover routing,
particle tracking, and heat/energy transport.
It also omits EPA Superfund remedy-selection and technology workflows,
including exposure-pathway protection, plume containment, pump-and-treat, in
situ treatment, vertical engineered barriers, monitored natural attenuation,
institutional controls, alternative water supply, cleanup goals, capture-zone
evaluation, converging lines of evidence, and site-specific characterization,
monitoring, treatment, source/plume remediation modelling, and cleanup-
performance validation.
It also omits EPA remedy-technology workflows for monitored natural
attenuation, permeable reactive barriers, in situ chemical
oxidation/reduction, in situ thermal treatment, phytotechnologies, soil vapor
extraction/air sparging, solidification/stabilization, vapor intrusion
mitigation, and vertical engineered barriers.

These anchors support analytical ADE plausibility and honest simulator-boundary
claims. They do not support a strong external-validation claim unless a future
registered validation-note folder adds domain-matched field-plume or
full-simulator reference comparisons for the frozen response surface.

## 5. Additional Local Physics Checks

The targeted test suite now includes a longitudinal-dispersivity control:
holding pore velocity, sorption, half-life, bulk density, and porosity fixed,
raising `alpha_L` increases the 1% concentration front distance and decreases
the first arrival time at 10 m.  This is an ADE plausibility check for the
implemented `D = alpha_L*v_pore + D_m*tau` term, not external validation of a
PHREEQC, MT3DMS, MT3D-USGS, MODFLOW 6 GWT, or field-plume calculation.
