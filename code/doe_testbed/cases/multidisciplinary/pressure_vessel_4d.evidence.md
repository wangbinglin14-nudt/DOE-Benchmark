# Evidence companion: `pressure_vessel_4d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [pressure_vessel_4d.py](pressure_vessel_4d.py); SHA-256 `f1c490fd5f6f06b4c6ece4420d2335d781839293879dcc59253c0222cb339049`.
- Model: Thin-wall closed-end membrane strength analytical control.
- Registry: `docs/case_validation_status.csv` records `P2`, `analytical control; not external ASME validation`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `P` | MPa | [1.0, 20.0] | linear |
| `R` | m | [0.5, 2.0] | linear |
| `t` | m | [0.005, 0.05] | linear |
| `sigma_y` | MPa | [200.0, 600.0] | linear |

Outputs, in column order: `FS`, `W_per_L`, `sigma_VM`.

## Equations, constants and implementation choices

sigma_hoop=P*R/t; sigma_axial=P*R/(2t); sigma_VM=(P*R/t)*sqrt(3)/2; FS=sigma_y/sigma_VM; mass_per_length=7850*pi*((R+t)^2-R^2). P and sigma useMPa consistently; geometry usesm and densitykg/m^3. Output W_per_L is mass per length, not force per length. FS<1.5 is the benchmark failure threshold, not a code-qualified allowable-stress design rule.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy only; no FE, ASME rule engine or vessel-test dataset. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [NASA/TM-2015-218216](https://ntrs.nasa.gov/citations/20160000456) | legacy official thin-vessel formulation anchor; full report equations/pages not read this pass. |
| [ASME SectionVIII division1](https://www.asme.org/codes-standards/find-codes-standards/bpvc-viii-1-bpvc-section-viii-rules-construction-pressure-vessels-division-1) | official scope anchor inherited from cross-validation note; code clauses not accessed or verified. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_outputs_match_thin_wall_analytical_formulae](../../../tests/test_pressure_vessel_physics.py#L237) | Two physical states reproduce sigmaVM,FS and exact-annulus mass to1e-12. This is same-formula regression, not independent validation. |
| [test_von_mises_response_matches_biaxial_membrane_stress_identity](../../../tests/test_pressure_vessel_physics.py#L257) | Cross-checks biaxial identity within thin-wall assumptions. |
| [test_failure_boundary_classifies_points_on_both_sides](../../../tests/test_pressure_vessel_physics.py#L307) | Pins benchmark classification. |
| [test_private_physical_input_contract_rejects_nonfinite_values](../../../tests/test_pressure_vessel_physics.py#L361) | Pins invalid-input handling. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/pressure_vessel_4d.md) and [cross-validation script](../../../tests/_cross_validation/pressure_vessel_4d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

Thin-wall membrane approximation ignores radial stress; at domain corner t/R=.1 approximation quality needs assessment. Exact annular mass does not make stress thick-wall exact. No end-cap mass, support/nozzle stress, weld efficiency, corrosion allowance, fatigue/fracture, external-pressure buckling, proof test, fabrication or ASME certification.

## Legacy comments and remaining checks

Prior cross-validation Markdown contains rounded computed tables and machine-precision wording: use executable formula tests for numerical evidence, not rounded printed values as full-precision references. Shigley §3-14 and Boresi §4.5 are inherited locators not full-text verified in this pass.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
