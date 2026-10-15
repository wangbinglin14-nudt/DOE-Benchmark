# Cross-Validation: `wing_vlm_6d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `wing_vlm_6d` |
| Historical label | VLM, retained only for stable case ID/API compatibility |
| Implemented solver | Prandtl lifting-line theory (PLLT), Glauert Fourier form |
| Evidence tier | P2 plausibility-only |
| Validation scope | Formula and trend controls for a compact finite-wing surrogate; public sources are boundary evidence, not case-level validation data. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/wing_vlm_6d.py` and
`tests/test_wing_vlm_physics.py`. It exists because the stable case ID contains
`vlm`, while the current implementation is deliberately a lower-order PLLT
surrogate rather than AVL, a vortex-lattice solver, a panel method, or CFD.

## 2. Implemented Formula Class

The case uses a Fourier lifting-line collocation solve over a finite wing.
For the span transform `y = (b/2) cos(theta)`, circulation is expanded as:

```text
Gamma(theta) = 2 b V_inf sum_n A_n sin(n theta)
CL           = pi AR A_1
CDi          = pi AR sum_n n A_n^2
e            = CL^2 / (pi AR CDi), when CDi > 0
```

The Prandtl-Glauert correction enters through the section lift slope
`a0 = 2 pi / beta`, where `beta = sqrt(1 - Ma^2)`. A separate conceptual
normal-flow sweep factor applies:

```text
CL_swept  = CL cos^2(Lambda)
CDi_swept = CDi cos^4(Lambda)
```

The sweep factor is a low-order trend term. It is not a swept-wing VLM,
AVL, panel-method, buffet, shock, or CFD model.

The targeted tests also pin the denominator-free induced-drag corner cases.
A twisted near-zero-lift state keeps finite `CDi` through
`CDi = pi AR sum_n n A_n^2`; the separate trivial zero-loading state
(`alpha = 0` and `twist = 0`) is archived as `CL = 0`, `CDi = 0`, `e = 1`.
That final value is a finite-output convention for the benchmark, not a
physical Oswald-efficiency measurement at zero lift.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| NASA Glenn winglet/induced-drag material: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/winglets/ | The induced-drag identity `CDi = CL^2 / (pi AR e)` and the qualitative role of aspect ratio and planform efficiency. | Validation of this PLLT implementation, profile drag, parasite drag, stall, or swept-wing behavior. |
| NASA Glenn drag factors: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/factors-that-affect-drag/ | Boundary evidence that profile/skin-friction, form drag, wave drag, viscosity, and compressibility are separate drag mechanisms omitted by this induced-drag-only case. | Evidence that `CDi` here is a complete aircraft drag coefficient. |
| NASA History SP-4219, Chapter 3: https://www.nasa.gov/history/SP-4219/Chapter3.html | Prandtl-Glauert historical/compressibility-rule context below critical speed. | Transonic-burble, buffet, shock, drag-divergence, or swept-wing validation. |
| MIT AVL overview and documentation: https://web.mit.edu/drela/Public/web/avl/ | Boundary evidence for what AVL/VLM contains: extended vortex-lattice lifting surfaces, horseshoe vortices, slender-body options, control deflections, trim cases, and Trefftz-plane induced-drag analysis. | Proof that this compact PLLT case is AVL-equivalent, VLM-equivalent, or a rigid-aircraft workflow. |
| MIT AVL user documentation: https://web.mit.edu/drela/Public/web/avl/avl_doc.txt and https://web.mit.edu/drela/Public/web/avl/AVL_User_Primer.pdf | Tool-workflow boundary: geometry files, run cases, trim variables, control deflections, and output inspection. | Validation of the registered six-input response surface. |
| NASA SP-405 / NTRS vortex-lattice preliminary-design material: specific paper https://ntrs.nasa.gov/citations/19760021082 and PDF https://ntrs.nasa.gov/api/citations/19760021082/downloads/19760021082.pdf; collected SP-405 volume https://ntrs.nasa.gov/citations/19760021075 | Boundary evidence that vortex-lattice preliminary design covers planform/control-surface workflows beyond this PLLT surrogate. | Case-level validation that the current implementation solves a vortex-lattice design problem. |
| NASA OpenVSP VSPAERO basics: https://www.nasa.gov/reference/openvsp-vspaero-basics/ | Boundary evidence for potential-flow aerodynamic analysis using OpenVSP geometries, DegenGeom thin-surface representation, pressure/wake solution review workflows, and the documented VSPAERO v3.45 / solver v7 transition. | OpenVSP/VSPAERO-equivalent aerodynamic analysis, calibration data, stable solver-reference data, or validation of this PLLT response surface. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not AVL-equivalent, not
VLM-equivalent, not a swept-wing solver, not a panel method, and not CFD.
It reports induced-drag-only finite-wing coefficients. It omits profile drag,
parasite drag, viscous stall, camber/zero-lift-angle effects, dihedral,
winglets, fuselage interference, ground effect, true swept lifting-surface
physics, control-surface or body modelling, multi-lifting-surface design,
Trefftz-plane post-processing, OpenVSP/VSPAERO geometry or DegenGeom workflows,
pressure/wake-solution review, shock/buffet physics, and wind-tunnel
calibration.
OpenVSP/VSPAERO version-transition documentation is treated as workflow
boundary evidence only; it is not calibration or reference-output evidence for
the registered response surface.

These source anchors support local finite-wing identities and honest
production-tool boundaries. They do not support a strong external
case-validation claim unless a future pass adds a registered validation-note
folder with case-specific AVL/VLM, wind-tunnel, or high-fidelity reference
comparisons.
