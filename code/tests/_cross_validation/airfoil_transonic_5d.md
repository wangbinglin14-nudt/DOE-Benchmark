# Cross-Validation: `airfoil_transonic_5d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `airfoil_transonic_5d` |
| Implemented solver | NeuralFoil baseline polar plus explicit Prandtl-Glauert and empirical wave-drag corrections |
| Evidence tier | P2 plausibility-only |
| Validation scope | Local transonic drag-rise trend, subcritical compressibility scaling, source-boundary, and omitted-physics controls. NASA/NACA/XFOIL/AeroSandbox/NeuralFoil sources are context and boundary evidence only. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/airfoil_transonic_5d.py` and
`tests/test_airfoil_transonic_physics.py`. It clarifies that the registered
benchmark response is a compact DOE surrogate for drag-divergence stress
testing, not a calibrated transonic airfoil solver.

## 2. Implemented Formula Class

The formal response surface is pinned to the low-level
`neuralfoil.get_aero_from_airfoil` baseline polar. Mach effects are applied by
the benchmark code itself:

```text
beta_pg = sqrt(max(1 - Ma^2, 0.01))
CL      = CL_NeuralFoil(alpha, Re, NACA4) / beta_pg
CD      = CD_NeuralFoil(alpha, Re, NACA4) / beta_pg + CD_wave
Mcr     = 0.75 - 0.06*(t/c)/0.12
CD_wave = 20*(Ma - Mcr)^4 for Ma > Mcr
```

The Prandtl-Glauert term is a subcritical trend correction. The fourth-power
wave-drag trigger creates a sharp drag-rise elbow for DOE method stress
testing; it is not a derivative `M_dd` solver and is not a shock-resolving
transonic analysis.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| NASA TM-81927, Harris NACA 0012 transonic tunnel data: https://ntrs.nasa.gov/search.jsp?R=19810014503 | Public NACA 0012 transonic drag-divergence and tunnel-data context. | Calibration or validation of this NeuralFoil-baseline surrogate, camber/thickness sweep, or fourth-power wave-drag trigger. |
| NASA TM-81927 citation page: https://ntrs.nasa.gov/citations/19810014503 | Stable public metadata anchor for the same Harris tunnel report. | Proof that the benchmark response surface matches the report data. |
| NASA TM-100527, high-Reynolds-number NACA 0012 transonic tests: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19880009187.pdf | Additional transonic NACA 0012 test context and high-Reynolds-number boundary evidence. | Validation of current NeuralFoil/Prandtl-Glauert/wave-drag formulas across the registered input family. |
| NASA History SP-4219 transonic drag-divergence context: https://www.nasa.gov/history/SP-4219/Chapter3.html | Historical and qualitative drag-divergence phenomenon context. | Quantitative airfoil-performance validation. |
| NASA Glenn similarity parameters: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/similarity-parameters-2/ | Boundary evidence that Mach number controls compressibility similarity and that near/transonic flows can develop shock waves affecting drag. | Calibration of the benchmark's critical-Mach trigger, fourth-power wave-drag term, or Mach-resolved airfoil response. |
| NASA Langley Turbulence Modeling Resource NACA 0012 validation: https://turbmodels.larc.nasa.gov/naca0012_val.html | Boundary evidence for what high-fidelity RANS validation datasets look like. | Claim that this case is externally validated transonic CFD/RANS. |
| NASA NTRS transonic NACA-0012 turbulence/buffet simulations: https://ntrs.nasa.gov/citations/20220017612 and https://ntrs.nasa.gov/citations/20220018292 | Boundary evidence for pre/post-buffet NACA-0012 validation workflows using URANS, DDES, WMLES, FUN3D, shock-location comparisons, and wind-tunnel data. | Buffet prediction, shock-location validation, URANS/DDES/WMLES/FUN3D equivalence, or validation of this algebraic wave-drag response surface. |
| XFOIL online documentation: https://web.mit.edu/drela/Public/web/xfoil/ | Boundary evidence for panel/integral-boundary-layer operating limits and separated-flow caveats. | XFOIL-calibrated transonic prediction or validation of the wave-drag trigger. |
| AeroSandbox Airfoil wrapper documentation: https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/index.html | Boundary evidence that current wrappers expose `mach`; useful to distinguish the wrapper from the frozen low-level API. | Evidence that the benchmark uses the AeroSandbox-wrapper Mach model. |
| NeuralFoil arXiv preprint: https://arxiv.org/abs/2503.16323 | Baseline NeuralFoil airfoil-polar method context. | Mach-resolved NeuralFoil validation of this transonic surrogate. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not externally validated
transonic CFD/RANS, not Mach-resolved NeuralFoil, not an AeroSandbox-wrapper
Mach model, not a derivative `M_dd` solver, not Korn-relation validation, not
buffet prediction, not shock/boundary-layer separation validation, and not
wind-tunnel calibrated.

It also omits Euler/RANS shock resolution, grid convergence, viscous
shock/boundary-layer interaction, transition modelling, buffet onset,
separation dynamics, Reynolds-dependent shock motion, shock-location tracking,
URANS/DDES/WMLES/FUN3D simulations, airfoil-coordinate geometry variation
beyond the compact NACA4 naming rule, aeroelastic effects, 3-D wing effects,
control-surface effects, and experimental tunnel-wall or blockage corrections.

These source anchors support honest transonic plausibility and omitted-physics
boundaries. They do not support an external-validation claim unless a future
registered validation-note folder adds domain-matched CFD, wind-tunnel, or
other calibrated reference comparisons for the frozen response surface.
