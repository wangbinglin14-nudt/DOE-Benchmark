# Evidence companion: `rotor_bemt_6d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [rotor_bemt_6d.py](rotor_bemt_6d.py); SHA-256 `2ec3c801ddc7fe336f953cf59f9dde8535302f729fe97a375fc0cde7fa847e58`.
- Model: Finite-station nondimensional hover BEMT with imposed induced-power floor.
- Registry: `docs/case_validation_status.csv` records `P2`, `plausibility only`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `R` | m | [0.5, 5.0] | linear |
| `Omega` | RPM | [100.0, 600.0] | linear |
| `c_bar` | - | [0.05, 0.2] | linear |
| `theta_0` | deg | [5.0, 20.0] | linear |
| `theta_tw` | deg/R | [-10.0, 0.0] | linear |
| `B` | - | [2.0, 6.0] | linear |

Outputs, in column order: `CT`, `CP`, `FM`.

## Equations, constants and implementation choices

15 uniform r/R stations from .05 to .99; sigma=B*c_bar/pi; theta=theta_0+theta_tw*r/R; B=round(B) clamped 2..6. CL_section=clip(2pi*(theta-atan(lambda/r)),-1.5,1.5). F=clip((2/pi)acos(exp[-B(1-r)/(2*r*sin(phi))]),.01,1). Solve lambda=sqrt(max(sigma*CL_section*r,0)/(8F)) with bisection on [1e-6,.5], tol1e-10, max80 iterations. CT and sectional power use trapezoidal integration. CD0=.012; CP=max(CP_induced_sectional,1.10*max(CT,0)^1.5/sqrt(2),0)+CP_profile, then floor1e-6. FM=CT^1.5/(sqrt(2)*CP) for CT>0, else0. Positive-thrust FM<=1/1.10. R and RPM cancel from the reported coefficients in this fixed-polar model.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy; no AeroDyn, OpenFAST, NDARC or rotor-test database is called. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [OpenFAST AeroDyn introduction](https://openfast.readthedocs.io/en/main/source/user/aerodyn/introduction.html) | opened 2026-09-17, v5.0.0 documentation; production-module scope boundary, not numerical reference data. |
| [Leishman Principles of Helicopter Aerodynamics](https://assets.cambridge.org/97805218/58601/frontmatter/9780521858601_frontmatter.pdf) | publisher frontmatter and edition verified in earlier audit; precise BEMT and tip-loss pages/sections not independently verified here. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_stationwise_inflow_residual_is_solved_before_integration](../../../tests/test_rotor_bemt_physics.py#L210) | For B4,c_bar.12,collective12,twist-5, station residual maximum is below1e-8; internal solver-consistency control. |
| [test_prandtl_tip_loss_matches_standard_hover_formula](../../../tests/test_rotor_bemt_physics.py#L257) | Analytical F identity. |
| [test_hover_coefficients_are_invariant_to_rotational_speed_scaling](../../../tests/test_rotor_bemt_physics.py#L113) | Pins inactive speed coordinate for these nondimensional coefficients. |
| [test_nonideal_induced_power_factor_bounds_positive_thrust_fm](../../../tests/test_rotor_bemt_physics.py#L192) | Pins imposed power-floor implication, not measured rotor efficiency. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/rotor_bemt_6d.md) and [cross-validation script](../../../tests/_cross_validation/rotor_bemt_6d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

No hub loss, swirl/tangential induction, dynamic inflow, forward flight, Reynolds/Mach-dependent polars, structural/acoustic coupling or rotor-specific calibration. The 15-point grid and power floor are benchmark choices; residual convergence is not grid convergence.

## Legacy comments and remaining checks

A code comment calls the radial distribution cosine; executable np.linspace is uniform. rho, physical r/dr and Omega do not introduce Reynolds/tip-Mach dependence into output coefficients. Do not imply RPM or radius sensitivity for CT/CP/FM.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
