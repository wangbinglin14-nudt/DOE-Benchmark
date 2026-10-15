# Evidence companion: `airfoil_transonic_5d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [airfoil_transonic_5d.py](airfoil_transonic_5d.py); SHA-256 `cf9078fe670750182b4f7264163ef9278c419efc967030c068886a0153f6f2c4`.
- Model: NeuralFoil polar plus PG scaling and algebraic fourth-power drag-rise trigger.
- Registry: `docs/case_validation_status.csv` records `P2`, `plausibility only`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `Ma` | - | [0.3, 0.88] | linear |
| `alpha` | deg | [-2.0, 6.0] | linear |
| `tc` | - | [0.08, 0.18] | linear |
| `fc` | - | [0.0, 0.06] | linear |
| `Re` | - | [1000000.0, 10000000.0] | log |

Outputs, in column order: `CD`, `CL`, `CL_CD`.

## Equations, constants and implementation choices

m=round(100*fc), t=round(100*tc); NACA name uses camber-position digit 4 for nonzero rounded camber and 0 otherwise. beta=sqrt(max(1-Ma^2,.01)); CL=CL_NF/beta; Mcr=.75-.06*tc/.12; CD=CD_NF/beta+20*max(Ma-Mcr,0)^4; CL_CD=CL/CD. Geometry changes in discrete integer-percentage steps; tc also enters continuous Mcr.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy, NeuralFoil and AeroSandbox. The low-level NeuralFoil call has no supplied Mach argument; AeroSandbox is used to construct geometry, not its Mach-aware airfoil wrapper. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [NeuralFoil author preprint](https://arxiv.org/abs/2503.16323) | opened 2026-09-17; baseline method, not these compressibility additions. |
| [Harris NASA TM-81927](https://ntrs.nasa.gov/citations/19810014503) | legacy official report anchor; publication identifier only inherited here, not a fresh full-text/page verification. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_neuralfoil_backend_contract_maps_naca_geometry_and_applies_external_corrections](../../../tests/test_airfoil_transonic_physics.py#L387) | Mock-backend interface and formula control, not aerodynamic calibration. |
| [test_drag_divergence_slope_steepens_above_critical_mach](../../../tests/test_airfoil_transonic_physics.py#L469) | At alpha2,tc.12,fc.02,Re5e6 the .80–.86 CD rise exceeds five times .68–.74 rise; tests the chosen trigger. |
| [test_subcritical_prandtl_glauert_scaling_preserves_lift_drag_ratio](../../../tests/test_airfoil_transonic_physics.py#L487) | Subcritical formula identity. |
| [test_neuralfoil_point_failure_is_not_silent_nan](../../../tests/test_airfoil_transonic_physics.py#L290) | Backend failure is explicit. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/airfoil_transonic_5d.md) and [cross-validation script](../../../tests/_cross_validation/airfoil_transonic_5d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

A subcritical PG formula is deliberately extended as a benchmark stressor. No shock location, buffet, viscous shock interaction, CFD/RANS, Korn calibration or Mach-resolved neural inference is computed. The constant 20 and Mcr law are implementation choices, not verified fitted tunnel coefficients.

## Legacy comments and remaining checks

Exact textbook section/page claims in legacy cross-validation notes remain unverified. NASA tunnel data provide context only; no residual-to-data comparison is established by the test suite.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
