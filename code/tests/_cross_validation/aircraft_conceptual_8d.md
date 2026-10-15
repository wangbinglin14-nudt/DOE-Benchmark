# Cross-Validation: `aircraft_conceptual_8d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `aircraft_conceptual_8d` |
| Implemented solver | Low-order Breguet range plus parabolic drag-polar and Raymer-style empty-weight surrogate |
| Evidence tier | P2 plausibility-only |
| Validation scope | Formula identities, expected conceptual-design trends, source-boundary, and omitted-workflow controls. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/aircraft_conceptual_8d.py` and
`tests/test_aircraft_conceptual_physics.py`. It clarifies that the benchmark is
a compact conceptual-design stressor, not a calibrated aircraft sizing,
mission, or MDAO workflow.

## 2. Implemented Formula Class

The case applies closed-form conceptual-design equations:

```text
R       = (V_cruise / (g*SFC)) * (L/D) * ln(W_i/W_f)
L/D_max = 0.5 * sqrt(pi*AR*e/CD0)
CD0     = Cf * Swet/Sref * (1 + 1.5*t/c + 100*(t/c)^4)
We_frac = 0.55 * Ma^0.05 * AR^-0.03 * ((t/c)/0.14)^0.08
```

`SW` and `rho_alt` are retained as DOE stress-test inputs but are intentionally
neutral in the current constant-cruise algebraic response.
The targeted tests also pin Breguet separability: with geometry and L/D fixed,
range scales linearly with `Ma_cruise`, inversely with mass `SFC`, and with the
fuel-log term `ln(1/(1 - 0.95*W_frac))`.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| NASA Glenn range, constant velocity: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-constant-velocity/ | Breguet/range-equation context. | Full mission fuel-burn validation. |
| NASA Glenn range summary: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-summary/ | Range-equation assumptions and caveats. | Climb, descent, reserve, alternate, wind, or trajectory modelling. |
| MIT Unified Breguet range equation: https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node98.html | Independent formula context. | Calibration of the empty-weight power law. |
| NASA Glenn induced drag coefficient: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/ | Induced-drag and aspect-ratio trend context. | Full aircraft drag build-up validation. |
| NASA Glenn Mach/drag pages: https://www.grc.nasa.gov/www/BGH/mach.html and https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/what-is-drag/ | Boundary evidence for omitted compressibility and drag mechanisms. | Transonic drag-rise, buffet, or supercritical-airfoil validation. |
| NASA Software Catalog Aviary: https://software.nasa.gov/software/LEW-20538-1 | Boundary evidence for production aircraft synthesis tools. | Evidence that this case runs Aviary/FLOPS/GASP. |
| NASA Software Catalog FLOPS v9: https://software.nasa.gov/software/LAR-18934-1 | Boundary evidence for a full aircraft synthesis system with weights, aerodynamics, propulsion data, mission performance, takeoff/landing, and program-control modules. | FLOPS-equivalent analysis or calibration of this constant-cruise algebraic surrogate. |
| NASA NTRS FLOPS weights-estimation method: https://ntrs.nasa.gov/citations/20170005851 | Boundary evidence for NASA synthesis-tool weight-estimation scope. | Validation of the simplified Raymer-style empty-weight power law. |
| NASA NTRS GASP theoretical-development report: https://ntrs.nasa.gov/citations/19810010562 | Boundary evidence for an integrated preliminary-design synthesis program spanning technical disciplines, parametric studies, performance, economics, and configuration trade studies. | GASP-equivalent analysis or validation of this compact Breguet/Raymer-style surrogate. |
| NASA Aviary overview: https://www.nasa.gov/aeronautics/aviary-software-overview/ | Boundary evidence for OpenMDAO-based aircraft design workflows. | Validation of this compact Breguet/Raymer-style surrogate. |
| OpenMDAO Aviary docs: https://openmdao.github.io/Aviary/ | Boundary evidence for mission/vehicle coupling workflows. | OpenMDAO/Aviary-equivalent execution. |
| NASA NTRS Aviary MDAO paper: https://ntrs.nasa.gov/citations/20230015940 | Boundary evidence for analytic-gradient MDAO and tightly coupled aircraft/subsystem design. | Analytic-gradient optimization validation of this case. |
| Aviary external-subsystem docs: https://openmdao.github.io/Aviary/user_guide_unreviewed/subsystems/using_external_subsystems.html | Boundary evidence for external-subsystem integration. | Subsystem-builder workflow validation. |
| Aviary theory guide and FAQ: https://openmdao.github.io/Aviary/theory_guide/intro.html and https://openmdao.github.io/Aviary/misc_resources/FAQ.html | Boundary evidence for empirical aircraft modelling and workflow scope. | Upgrade of the evidence tier. |
| Aviary features/functionality page: https://openmdao.github.io/Aviary/user_guide_unreviewed/features_and_functionalities.html | Boundary evidence for geometry, mass, aerodynamics, propulsion, mission subsystem coverage, and mission-phase setup in Aviary models. | Evidence that this benchmark runs Aviary's coupled subsystem/mission workflow. |
| Aviary comparison to FLOPS: https://openmdao.github.io/Aviary/theory_guide/comparison_to_flops.html | Boundary evidence for FLOPS/GASP-derived geometry, mass, aerodynamic, propulsion, mission, takeoff/landing, reserve/off-design, and OpenMDAO optimization machinery. | Evidence that the benchmark is FLOPS/GASP/Aviary-equivalent or analytic-derivative MDAO validated. |
| NASA OpenVSP/VSPAERO basics: https://www.nasa.gov/reference/openvsp-vspaero-basics/ | Boundary evidence for geometry, DegenGeom, potential-flow, and pressure/wake review workflows. | OpenVSP/VSPAERO aerodynamic validation. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not externally validated aircraft
sizing, not a full Raymer/Roskam implementation, not FLOPS/GASP/Aviary/OpenMDAO
validated, not OpenVSP/VSPAERO validated, not mission fuel-burn validated, not
transonic-performance validated, and not a certification/cost/noise/emissions
model.

It omits payload/MTOW closure, wing-loading and thrust-loading constraint
analysis, takeoff/landing field length, engine lapse, climb/descent/loiter or
alternate segments, reserve fuel, wind/weather corrections, cruise
CL/dynamic-pressure trim closure, compressibility drag rise, wave drag,
drag-divergence Mach number, buffet, supercritical-airfoil design,
stability/control, certification, cost, noise, emissions, trajectory
optimization, mission-phase coupling, external-subsystem workflows, and
medium/high-fidelity MDAO.
It also omits FLOPS-style weights, aerodynamics, propulsion-data,
mission-performance, takeoff/landing, and program-control module workflows.
It omits GASP-style integrated technical-discipline synthesis, parametric
studies, performance/economics/configuration trade studies, and the
FLOPS/GASP-derived geometry, mass, empirical/tabular aerodynamics, external
aerodynamic model, engine-deck, reserve/off-design mission, takeoff/landing,
equations-of-motion, and OpenMDAO optimization machinery described by Aviary's
comparison-to-FLOPS documentation.
It does not instantiate Aviary's geometry, mass, aerodynamics, propulsion,
and mission subsystem set, nor Aviary's user-selected phase-options mission
setup.

These anchors support formula plausibility and honest workflow boundaries.
They do not support a strong external-validation claim unless a future
registered validation-note folder adds domain-matched aircraft sizing,
mission-analysis, or high-fidelity MDAO reference comparisons.
