# Cross-Validation: `waverider_4d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `waverider_4d` |
| Implemented solver | Fixed-gamma modified-Newtonian impact surrogate with Rayleigh-Pitot `Cpmax` and empirical friction |
| Evidence tier | P2 plausibility-only |
| Validation scope | Local hypersonic trend, Rayleigh-Pitot coefficient, representative L/D, and omitted-physics boundary controls. NASA/NACA sources are formula, order-of-magnitude, and scope evidence only. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/waverider_4d.py` and
`tests/test_waverider_physics.py`. It is intentionally explicit about the
boundary between a compact DOE surrogate and real waverider design evidence.

## 2. Implemented Formula Class

The case uses a fixed `gamma = 1.4` calorically perfect gas model:

```text
Cp_lower = Cpmax(Ma) sin^2(delta)
delta    = theta_c + alpha
CL       = Cp_lower cos(delta) sin(phi)
CD_wave  = Cp_lower sin(delta) sin(phi)
Cf       = 0.0015 (1 + 0.02 Ma^2)^(-0.3)
CD_fric  = Cf 2/sin(phi)
L/D      = CL / (CD_wave + CD_fric)
eta_v    = 0.22 tan(theta_c)^(2/3) sin(phi)^(-1/3)
```

`Cpmax(Ma)` is the calorically perfect Rayleigh-Pitot modified-Newtonian
coefficient. For `gamma = 1.4`, it approaches about `1.839` as Mach number
goes to infinity; the classical Newtonian value `2` is a separate idealized
upper bound.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| NASA TP-3559 waverider-derived hypersonic cruise configurations: https://ntrs.nasa.gov/api/citations/19960045290/downloads/19960045290.pdf | Order-of-magnitude L/D and integrated-configuration caveats. Public metadata describe design Mach 4.0 and data over Mach 1.6-4.63. | Domain-matched validation for this Mach 5-12 benchmark, full Mach-domain validation, or proof of shock-attached design accuracy. |
| NASA CR-182076 APAS II: https://ntrs.nasa.gov/api/citations/19910013767/downloads/19910013767.pdf | Modified-Newtonian / impact-method preliminary-analysis context. | Calibrated waverider performance or CFD/wind-tunnel validation. |
| NACA Report 1135 compressible-flow tables: https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf | Compressible-flow and normal-shock/Rayleigh-Pitot formula context. | Validation of the empirical friction term or integrated vehicle response. |
| NASA Glenn normal shock equations: https://www.grc.nasa.gov/WWW/BGH/normal.html | Calorically perfect/imperfect shock-equation boundary context. | Full high-temperature gas modelling. |
| NASA Glenn real-gas effects: https://www.grc.nasa.gov/www/BGH/realgas.html | Boundary evidence that variable specific heats, vibrational excitation, dissociation, ionization, and chemical/thermal nonequilibrium matter in high-Mach flows. | Validation of the fixed-gamma surrogate. |
| NASA Glenn Mach number: https://www.grc.nasa.gov/www/BGH/mach.html | Boundary evidence that Ma > 5 is hypersonic and that shock/high-temperature gas effects become central at these speeds. | Validation of the fixed-gamma Rayleigh-Pitot implementation, empirical friction term, or high-temperature gas calculation. |
| NASA Glenn hypersonic missions: https://www.grc.nasa.gov/WWW/BGH/hmission.html | Boundary evidence that Mach 5-7 vehicle design must evaluate real-gas effects. | Mission-grade hypersonic design analysis by this case. |
| NASA Glenn re-entry aircraft: https://www.grc.nasa.gov/www/BGH/hihyper.html | Boundary evidence for strong shock waves, plasma, thermal protection, and high-drag re-entry context. | Waverider performance validation. |
| NASA X-43A Hyper-X: https://www.nasa.gov/reference/x-43a/ | Boundary evidence for airframe-integrated scramjet flight and hypersonic aerodynamic data workflows. | Hyper-X/X-43A-equivalent vehicle modelling. |
| NASA X-43A Mach-7 CFD image: https://www.nasa.gov/image-article/cfd-image-of-x-43a-mach-7-test-condition/ | Boundary evidence for engine-operating CFD workflow. | CFD validation of the registered response surface. |
| NASA Glenn Hypersonic Tunnel Facility: https://www.nasa.gov/centers-and-facilities/glenn/hypersonic-tunnel-facility/ | Boundary evidence for Mach 5-7 true-enthalpy ground testing. | Wind-tunnel validation of this algebraic surrogate. |
| NASA Glenn 1x1 Supersonic Wind Tunnel: https://www.nasa.gov/centers-and-facilities/glenn/1x1-supersonic-wind-tunnel/ | Boundary evidence for benchmark-quality CFD-validation experiments in supersonic and hypersonic fluid mechanics, with continuous Mach 1.3-6.0 operation. | CFD-code validation, wind-tunnel database validation, or Mach-domain validation of this algebraic surrogate. |
| NASA Ames Interaction Heating Facility / Arc Jet Complex: https://www.nasa.gov/ames/interaction-heating-facility/ | Boundary evidence for convective/radiative heating environments and thermal-protection testing. | Aerothermal validation, TPS material qualification, or arc-jet calibration of this L/D surrogate. |
| NASA FIAT thermal response / ablation software: https://software.nasa.gov/software/ARC-15779-1A | Boundary evidence that TPS work requires material thermal-response and ablation analysis. | Any thermal-response, material recession, or ablation model in this case. |
| NASA Hypersonic Technology Project: https://www.nasa.gov/directorates/armd/aavp/ht/ and research topics: https://www.nasa.gov/directorates/armd/aavp/ht/hypersonic-research-topics/ | Boundary evidence that current NASA hypersonics work spans reusable airbreathing flight research, system-level design, analysis, validation, propulsion, vehicle technologies, and high-temperature durable materials. | System-level hypersonic vehicle validation, propulsion validation, high-temperature-materials validation, or validation of this four-input algebraic L/D surrogate. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not CFD-validated, not
wind-tunnel-calibrated, not Hyper-X/X-43A-equivalent, not full Mach 5-12
validated, not shock-attachment verified, not a real-gas/aerothermal solver,
and not a mission-grade integrated-vehicle model.

It omits osculating-cone or streamline-tracing design, shock fitting,
shock-attachment verification, leading-edge bluntness, boundary-layer heating,
base drag, propulsion integration, trim/control moments, static/dynamic
stability, rarefied-flow effects, variable specific heats, vibrational
excitation, dissociation, ionization, chemical/thermal nonequilibrium, plasma
effects, thermal protection, airframe-integrated scramjet propulsion,
high-enthalpy facility simulation, CFD, and wind-tunnel calibration.
It also omits benchmark-quality CFD-validation experiments and any
NASA Glenn 1x1-style wind-tunnel database comparison. It omits NASA Ames
arc-jet TPS testing and FIAT-style thermal-response/ablation analysis.
It also omits NASA Hypersonic Technology Project-style system-level
design/analysis/validation, propulsion, vehicle-technology, and
high-temperature durable-material workflows.

These source anchors support local modified-Newtonian plausibility and honest
hypersonic omitted-physics boundaries. They do not support a strong external
case-validation claim unless a future pass adds a registered validation-note
folder with domain-matched CFD, wind-tunnel, high-enthalpy, or real-gas
reference comparisons.
