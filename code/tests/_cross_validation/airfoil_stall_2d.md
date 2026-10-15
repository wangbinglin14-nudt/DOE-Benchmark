# Cross-Validation: `airfoil_stall_2d`

## 1. Evidence Tier

| Item | Detail |
|---|---|
| Case | `airfoil_stall_2d` |
| Implemented solver | Low-level NeuralFoil baseline polar with explicit low-subsonic Prandtl-Glauert scaling |
| Evidence tier | P2 plausibility-only |
| Validation scope | Low-alpha lift-slope, symmetric-section trend, Reynolds-number trend, high-drag stall-region stressor, backend-boundary, and omitted-physics controls. |
| Response-surface status | No response-surface change; this note documents the current implementation and source boundary only. |

This note is a reviewer-facing companion to
`tests/_cross_validation/airfoil_stall_2d.py` and
`tests/test_airfoil_stall_physics.py`. It clarifies that the case is a compact
airfoil trend surrogate for DOE stress testing, not a validated stall
prediction workflow.

## 2. Implemented Formula Class

The registered response surface is pinned to
`neuralfoil.get_aero_from_airfoil` for a NACA00xx baseline polar. The benchmark
then applies explicit mild-compressibility scaling:

```text
beta_pg = sqrt(1 - Ma^2)
CL      = CL_NeuralFoil(alpha, Re, NACA00xx) / beta_pg
CD      = CD_NeuralFoil(alpha, Re, NACA00xx) / beta_pg
L/D     = CL / CD
```

The Mach input is a low-subsonic/mild-compressibility stressor. It is not a
Mach-resolved airfoil validation claim.

## 3. Source Boundary Table

| Public source | Used for | Not used for |
|---|---|---|
| NASA TM-4074 low-speed NACA 0012 context | Single-section low-speed Mach/Reynolds-number context. | Validation of the full NACA00xx thickness sweep or explicit `CD/beta` scaling. |
| NASA Turbulence Modeling Resource NACA 0012 validation: https://turbmodels.larc.nasa.gov/naca0012_val.html | Boundary evidence for high-fidelity RANS validation-dataset practice. | Claim that this case is RANS/LES validated. |
| Sheldahl/Klimas SAND80-2114 airfoil tables | Broad wind-tunnel trend and stall-region context. | Calibration of this NeuralFoil response surface. |
| NASA Glenn inclination/stall pages: https://www.grc.nasa.gov/WWW/K-12/VirtualAero/BottleRocket/airplane/incline.html and https://www.grc.nasa.gov/WWW/k-12/VirtualAero/BottleRocket/airplane/inclind.html | General boundary evidence for near-linear small-angle lift, separated-flow stall, high-angle drag rise, and experimental stall-point sensitivity. | Calibration of the NeuralFoil response surface or proof of stall-angle accuracy over the full NACA00xx/Mach/Reynolds domain. |
| XFOIL online documentation: https://web.mit.edu/drela/Public/web/xfoil/ | Boundary evidence for panel/integral-boundary-layer solver limits. | XFOIL-calibrated prediction or separated-flow validation. |
| AeroSandbox Airfoil wrapper documentation: https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/index.html | Boundary evidence that current wrappers expose `mach`. | Evidence that the benchmark uses the AeroSandbox-wrapper Mach model. |
| NeuralFoil arXiv preprint: https://arxiv.org/abs/2503.16323 | Baseline NeuralFoil airfoil-polar method context. | External validation of post-stall, transition, or Mach-resolved airfoil behavior. |

## 4. Non-Overclaiming Boundary

The case remains P2 plausibility-only. It is not externally validated
NACA0012 wind-tunnel prediction, not Mach-resolved airfoil validation, not
XFOIL-calibrated, not RANS/LES validated, not transition/turbulence validated,
not dynamic-stall validated, not rotating-blade validated, and not a 3-D
finite-wing validation case.

It omits cambered sections, flap/control-surface deflection, roughness,
trip/transition-control sweeps, turbulence-model selection, RANS/LES grid
convergence, dynamic-stall hysteresis, unsteady pitching/plunging motion,
3-D finite-wing effects, rotating-blade corrections, icing/contamination,
aeroelastic coupling, and acoustic effects.

These source anchors support transparent airfoil plausibility and
omitted-physics boundaries. They do not support a strong external-validation
claim unless a future registered validation-note folder adds domain-matched
experimental or high-fidelity reference comparisons for the frozen response
surface.
