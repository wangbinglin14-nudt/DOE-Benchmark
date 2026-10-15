# Cross-Validation: `supersonic_intake_5d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `supersonic_intake_5d` |
| Implemented solver | Inviscid two-ramp external-compression shock surrogate |
| Evidence tier | P2 plausibility-only |
| Validation scope | Local oblique-shock, normal-shock, total-pressure-ratio, and attached/detached-shock boundary controls. NASA/NACA/SUPIN sources are formula and production-tool boundary evidence only. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/supersonic_intake_5d.py` and
`tests/test_supersonic_intake_physics.py`. It clarifies that the benchmark
case is a compact shock-system surrogate, not a complete inlet design,
SUPIN-equivalent analysis, CFD model, or wind-tunnel validation package.

## 2. Implemented Formula Class

The case applies weak attached oblique-shock relations across two compression
ramps, followed by a terminal normal shock when the post-ramp flow remains
supersonic:

```text
tan(delta) = 2 cot(beta) (Ma^2 sin^2(beta) - 1)
             / (Ma^2 (gamma + cos(2 beta)) + 2)
eta_pr     = product_i (p0_downstream / p0_upstream)_i
p_ratio    = product_i (p_downstream / p_upstream)_i
```

The attached-shock cascade is pinned as a stagewise shock-product loss control:
each attached shock has `p0_downstream / p0_upstream <= 1`, so cumulative
total-pressure recovery is non-increasing as shocks are added, while static
pressure compression is accumulated as the product of stage pressure ratios.

The `gamma` input is a constant-gamma gas-property stressor over `[1.30, 1.40]`.
It is not a calorically imperfect, vibrational-energy, or thermochemical
real-gas calculation. Detached-shock states use an explicit low-recovery
surrogate penalty and create a discontinuous topology stressor.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| NASA Glenn oblique shock relations: https://www.grc.nasa.gov/www/k-12/airplane/oblique.html | Theta-beta-M relation and weak/strong attached-shock formula context. | Complete inlet validation, viscous effects, or detached-shock CFD. |
| NASA Glenn normal shock relations: https://www.grc.nasa.gov/WWW/BGH/normal.html | Normal-shock downstream Mach, total-pressure-loss, static-pressure-ratio, and calorically perfect/imperfect boundary context. | Calorically imperfect shock iteration or real-gas chemistry in this case. |
| NACA Report 1135 compressible-flow tables: https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf | Standard compressible-flow and shock-table reference context. | Wind-tunnel or CFD validation of the two-ramp response surface. |
| NASA Glenn inlet overview: https://www.grc.nasa.gov/WWW/k-12/airplane/inlet.html | Supersonic-inlet deceleration and pressure-recovery framing. | Full inlet aerodynamics, capture/spillage modelling, or engine matching. |
| NASA Glenn inlet performance: https://www.grc.nasa.gov/www/k-12/BGP/inleth.html | Boundary evidence that production inlet performance includes high pressure recovery, low spillage drag, and low distortion. | Validation that the compact `eta_pr` row captures spillage drag or compressor-face distortion. |
| NASA Glenn inlet performance model: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/inlet-performance-model/ | Boundary evidence that inlet-performance tests or demonstrations may sweep engine airflow while tracking pressure recovery, spillage drag, and distortion. | Wind-tunnel airflow-sweep validation, pressure-rake distortion measurement, or inlet-engine matching by this compact shock cascade. |
| SUPIN 2012, NASA/TM-2012-217660: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120013600.pdf | Production inlet-design-tool boundary context. | Proof that this benchmark is SUPIN-equivalent. |
| SUPIN 2024, NASA/TM-20240008586: https://ntrs.nasa.gov/citations/20240008586 | Boundary evidence that SUPIN performs geometric modelling and aerodynamic design/analysis for multiple inlet types and records flowpath components, flow rates, pressure recovery, drag, surfaces, and CFD grids. | Validation data for this compact two-ramp algebraic surrogate. |
| NASA Glenn inlets/nozzles design software: https://www.nasa.gov/glenn/research/inlets-and-nozzles/design-analysis-software/ | Boundary evidence for SUPIN geometry/performance outputs and structured volume grids for higher-fidelity CFD. | CFD-equivalent simulation by this case. |
| NASA Glenn support of flight demonstration projects: https://www.nasa.gov/glenn/research/inlets-and-nozzles/support-of-flight-demonstration-projects/ | Boundary evidence for X-59 inlet/nozzle CFD, wind-tunnel, real-time display, and propulsion-system integration workflows. | X-59 flight-demonstration validation. |
| NASA/TM-20220010443 X-59 inlet distortion test: https://ntrs.nasa.gov/citations/20220010443 | Boundary evidence for engine-face distortion and inlet/engine operability testing. | Distortion or operability prediction by this surrogate. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not CFD-validated, not
wind-tunnel validated, not SUPIN-equivalent, not an X-59 inlet model, not a
complete inlet design model, and not a real-gas or thermochemical inlet
solver.

It omits viscous shock/boundary-layer interaction, bleed scheduling,
cowl-lip geometry, capture area, spillage and spillage drag, shock-train
dynamics, inlet buzz, unstart dynamics, compressor-face distortion metrics,
diffuser area scheduling, flow-rate accounting, inlet drag build-up, surface
or CFD-grid output, inlet-engine matching, variable geometry, strong-shock
branch selection, back-pressure-driven shock-position solves, 3-D
sidewall/corner flow, X-59 inlet/nozzle CFD, wind-tunnel inlet-distortion
testing, engine-face distortion or operability assessment, propulsion-system
integration, real-time test displays, digital engine/aircraft architecture,
wind-tunnel engine-airflow sweeps, pressure-rake recovery/distortion
measurements,
calorically imperfect shock iteration, vibrational-energy modelling, real-gas
chemistry, wind-tunnel calibration, and SUPIN-style inlet design.

These source anchors support local shock-equation plausibility and honest
production-tool boundaries. They do not support a strong external
case-validation claim unless a future pass adds a registered validation-note
folder with intake-specific SUPIN, CFD, wind-tunnel, or inlet-test reference
comparisons.
