# Evidence companion: `waverider_4d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [waverider_4d.py](waverider_4d.py); SHA-256 `8705dd8af153bf8e2629e30de2ff5fb6d560df65f3d4c1b09f522b27ce39c3b1`.
- Model: Fixed-gamma modified-Newtonian algebraic hypersonic surrogate.
- Registry: `docs/case_validation_status.csv` records `P2`, `plausibility only`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `Ma` | - | [5.0, 12.0] | linear |
| `theta_c` | deg | [5.0, 20.0] | linear |
| `alpha` | deg | [0.0, 5.0] | linear |
| `phi` | deg | [30.0, 90.0] | linear |

Outputs, in column order: `L_D`, `CL`, `eta_v`.

## Equations, constants and implementation choices

gamma=1.4; p02/p1=[(gamma+1)^2*M^2/(4gamma*M^2-2(gamma-1))]^[gamma/(gamma-1)]*(1-gamma+2gamma*M^2)/(gamma+1); Cpmax=2*(p02/p1-1)/(gamma*M^2). delta=theta_c+alpha (radians); Cp_lower=Cpmax*sin(delta)^2, Cp_upper=0; CL=Cp_lower*cos(delta)*sin(phi); CD_wave=Cp_lower*sin(delta)*sin(phi); Cf=.0015*(1+.02*M^2)^(-.3); CD_fric=2*Cf/sin(phi); L_D=CL/(CD_wave+CD_fric); eta_v=.22*tan(theta_c)^(2/3)*sin(phi)^(-1/3). Constants .0015,.02,-.3,.22 are surrogate assumptions, not established fitted parameters.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy only; no streamline/shock-fitting, CFD or real-gas backend. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [NASA normal-shock equations](https://www.grc.nasa.gov/WWW/BGH/normal.html) | opened 2026-09-17; calorically perfect/imperfect distinction and shock identities verified. |
| [NASA TP-3559](https://ntrs.nasa.gov/citations/19960045290) | legacy official context; exact report pages and data extraction not checked in this pass. |
| [NASA CR-182076 APAS II](https://ntrs.nasa.gov/citations/19910013767) | legacy preliminary-method context, not calibration of constants. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_modified_newtonian_cpmax_matches_gamma14_rayleigh_pitot_limit](../../../tests/test_waverider_physics.py#L28) | M1000 limit matches1.8393710511306678 and analytic asymptote within1e-6; analytic control. |
| [test_aerodynamic_outputs_match_modified_newtonian_plus_friction_formula](../../../tests/test_waverider_physics.py#L160) | Formula regression, not independent validation. |
| [test_representative_hypersonic_ld_is_in_waverider_order_of_magnitude](../../../tests/test_waverider_physics.py#L133) | Broad plausibility envelope only. |
| [test_out_of_contract_modified_newtonian_response_is_not_archived](../../../tests/test_waverider_physics.py#L774) | Pins response rejection. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/waverider_4d.md) and [cross-validation script](../../../tests/_cross_validation/waverider_4d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

At Mach5–12 fixed gamma omits high-temperature gas physics. No verified shock attachment, cone-streamline design, base drag, heating, TPS, propulsion integration, wind-tunnel comparison or full-domain CFD validation.

## Legacy comments and remaining checks

Cpmax asymptote is about1.83937 for gamma1.4, not2. The .22 volume-efficiency coefficient is a chosen scale. Legacy NASA TP-3559 source context covers design Mach4 and reported Mach1.6–4.63, outside most/all of the registered Mach5–12 range; it cannot validate that range.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
