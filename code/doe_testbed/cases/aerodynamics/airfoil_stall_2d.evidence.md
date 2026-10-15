# Evidence companion: `airfoil_stall_2d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [airfoil_stall_2d.py](airfoil_stall_2d.py); SHA-256 `88035ceaba53cea3b8934d355db880f75e4b8c7f4bd488bbf920ccf3a30db2e5`.
- Model: NeuralFoil baseline polar with explicit scalar Prandtl–Glauert correction.
- Registry: `docs/case_validation_status.csv` records `P2`, `plausibility only`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `alpha` | deg | [-5.0, 20.0] | linear |
| `Re` | - | [50000.0, 5000000.0] | log |
| `Ma` | - | [0.0, 0.4] | linear |
| `t_c` | - | [0.06, 0.18] | linear |

Outputs, in column order: `CL`, `CD`, `LoD`.

## Equations, constants and implementation choices

NACA section = naca00{round(100*t_c):02d}; beta = sqrt(max(1-Ma^2,0.01)); CL=CL_NF/beta; CD=CD_NF/beta; LoD=CL/CD. Both coefficients receive identical Mach scaling, so LoD is Mach-invariant at fixed remaining inputs. Thickness is quantized to NACA integer percentage digits; the family is not continuously parameterized geometry.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy, NeuralFoil and AeroSandbox are required. The formal path calls neuralfoil.get_aero_from_airfoil(model_size="large"); missing imports or failed points raise. The analytical helper is diagnostic-only and is never a production fallback. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [NeuralFoil author preprint](https://arxiv.org/abs/2503.16323) | opened 2026-09-17; title/method context only, not validation of this wrapper or PG-scaled drag. |
| [NASA NACA0012/TMR anchor](https://turbmodels.larc.nasa.gov/naca0012_val.html) | opened 2026-09-17 but redirects to general NASA TMR page; no current dataset-specific content verified. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_neuralfoil_backend_maps_thickness_to_naca00xx_and_squeezes_arrays](../../../tests/test_airfoil_stall_physics.py#L129) | Mock backend pins geometry naming, array conversion and external PG factors; does not test trained-model accuracy. |
| [test_low_alpha_lift_slope_matches_thin_airfoil_reference](../../../tests/test_airfoil_stall_physics.py#L184) | At alpha 0/5 deg, Re=3e6, Ma=0, t/c=.12, lift slope is within 4% of 2pi. This is one analytical plausibility control. |
| [test_low_speed_mach_scaling_preserves_lift_to_drag_ratio](../../../tests/test_airfoil_stall_physics.py#L200) | Pins implemented cancellation of equal CL/CD scaling. |
| [test_neuralfoil_point_failure_does_not_fallback_to_analytical](../../../tests/test_airfoil_stall_physics.py#L50) | Pins fail-fast backend contract. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/airfoil_stall_2d.md) and [cross-validation script](../../../tests/_cross_validation/airfoil_stall_2d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

No RANS, XFOIL execution, experimental calibration, dynamic stall, transition sweep, or 3-D wing physics. Alpha/Re/thickness sensitivity and a CD>0.05 benchmark threshold do not certify a physical stall boundary.

## Legacy comments and remaining checks

The stable suffix 2d denotes section geometry, but actual input dimension is four. The private helper docstring still says "preferred; sub-ms, engineering accuracy": neither preference/fallback nor a universal timing/accuracy claim describes the formal contract. Comments about post-stall lift reversal are not guarantees throughout the domain. Stored Sobol indices are local computed metadata, not externally measured sensitivities.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
