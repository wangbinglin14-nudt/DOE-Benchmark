# Evidence companion: `supersonic_intake_5d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [supersonic_intake_5d.py](supersonic_intake_5d.py); SHA-256 `9b2b1315cfa88aee53dd3cb27cb003e9cf14574e50d8df895611ceadc824312b`.
- Model: Two-ramp inviscid shock cascade and terminal normal shock.
- Registry: `docs/case_validation_status.csv` records `P2`, `plausibility only`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `Ma_in` | - | [2.0, 5.0] | linear |
| `delta1` | deg | [5.0, 15.0] | linear |
| `delta2` | deg | [5.0, 20.0] | linear |
| `gamma` | - | [1.3, 1.4] | linear |
| `alpha_off` | deg | [-3.0, 3.0] | linear |

Outputs, in column order: `eta_pr`, `Ma_exit`, `p_ratio`.

## Equations, constants and implementation choices

First deflection=delta1+alpha_off. Weak theta-beta-M root found by 120-point beta scan and brentq xtol1e-10. With Mn=M*sin(beta), apply normal-shock relations then M_down=Mn_down/sin(beta-delta). p0 recovery and static pressure ratios multiply across stages. Constant gamma is input1.30..1.40. First-stage detachment returns [eta_pr,Ma_exit,p_ratio]=[.3,.5,2]; first downstream M<=1 returns [.5,.8,p2/p1]; second-stage detachment drops that oblique shock and retains the terminal normal shock. These are explicit surrogate branches, not resolved detached-flow physics.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy plus scipy.optimize.brentq. No SUPIN or CFD backend. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [NASA normal-shock equations](https://www.grc.nasa.gov/WWW/BGH/normal.html) | opened 2026-09-17; calorically perfect equations and irreversible loss verified; no fixed page number. |
| [NASA oblique-shock equations](https://www.grc.nasa.gov/www/k-12/airplane/oblique.html) | opened 2026-09-17; attached weak/strong branch and inviscid restrictions verified. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_normal_shock_matches_standard_gamma14_reference_values](../../../tests/test_supersonic_intake_physics.py#L29) | M=3,gamma1.4 yields M2=.47519,p2/p1=10.33333,p02/p01=.32834 within5e-5 relative; analytical reference point. |
| [test_intake_outputs_equal_stagewise_shock_product_for_attached_case](../../../tests/test_supersonic_intake_physics.py#L503) | Pins cascade assembly, not independent inlet data. |
| [test_oblique_shock_detaches_when_wedge_angle_exceeds_limit](../../../tests/test_supersonic_intake_physics.py#L432) | Pins branch detection. |
| [test_private_solver_rejects_malformed_or_out_of_domain_inputs_before_cascade](../../../tests/test_supersonic_intake_physics.py#L689) | Pins input contract. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/supersonic_intake_5d.md) and [cross-validation script](../../../tests/_cross_validation/supersonic_intake_5d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

No boundary-layer interaction, bleed, capture/spillage drag, compressor-face distortion, unstart dynamics, engine matching, shock-train/back-pressure solve, 3-D flow or real-gas chemistry.

## Legacy comments and remaining checks

Comment "Total-pressure ratio (isentropic relation)" should not be read as isentropic passage across the shock: total pressure decreases and entropy increases. Exceptions in brentq are converted to None, which can trigger the same surrogate branch as physical detachment; no separate failure diagnostic is reported there.

The legacy cross-validation script also has a test named `test_zero_alpha_symmetry` that evaluates identical arrays: it checks repeat consistency only, not symmetry or spillage. Its Ma2 test docstring says recovery >0.9 while the assertion is >0.85; the executable assertion is the tested threshold.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
