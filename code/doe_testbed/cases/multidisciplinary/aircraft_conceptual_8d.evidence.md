# Evidence companion: `aircraft_conceptual_8d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [aircraft_conceptual_8d.py](aircraft_conceptual_8d.py); SHA-256 `e4c5f735d5c92aa6b21ffe6a991dd0414c771fe4bbd31d4d29a5cf3c397cbacb`.
- Model: Breguet constant-cruise range and Raymer-style algebraic trend model.
- Registry: `docs/case_validation_status.csv` records `P2`, `plausibility only`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `AR` | - | [7.0, 14.0] | linear |
| `SW` | m² | [100.0, 400.0] | linear |
| `lam` | - | [0.2, 0.7] | linear |
| `tc` | - | [0.1, 0.18] | linear |
| `SFC` | kg/(N·h) | [0.05, 0.08] | linear |
| `W_frac` | - | [0.2, 0.45] | linear |
| `rho_alt` | kg/m³ | [0.3, 0.8] | linear |
| `Ma_cruise` | - | [0.6, 0.9] | linear |

Outputs, in column order: `range_km`, `L_D_max`, `We_frac`.

## Equations, constants and implementation choices

g=9.80665; a=295.1m/s; V=Ma*a; Cf=.0025; Swet/Sref=5.5; CDmin=Cf*5.5*(1+1.5tc+100tc^4); e=1/[1+.03AR(1-lam)^2+.01AR*tc/.12]; L_Dmax=.5*sqrt(pi*AR*e/CDmin); We_frac=.55*Ma^.05*AR^(-.03)*(tc/.14)^.08; SFC_SI=SFC/3600; range_km=V/(g*SFC_SI)*L_Dmax*ln[1/(1-.95W_frac)]/1000. SW and rho_alt are unused in these output formulas. Empty-weight output is not coupled back into fuel/MTOW closure.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy only; no Aviary/FLOPS/GASP/OpenMDAO or aircraft database. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [MIT Unified section13.3 Breguet range](https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node98.html) | opened 2026-09-17; section title and Breguet context, not fitted weight law. |
| [Raymer6th edition2018 library record](https://raes.soutron.net/Portal/Default/en-GB/RecordView/Index/150787) | verified in 2026-09-16 bibliography audit; ISBN9781624104909; quoted chapter locators not full-text verified. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_breguet_separable_range_scaling_for_mach_sfc_and_fuel_log_term](../../../tests/test_aircraft_conceptual_physics.py#L665) | Pins Mach-linear, inverse-SFC and logarithmic fuel scaling at1e-12; algebraic identity. |
| [test_lift_to_drag_matches_parabolic_drag_polar_formula](../../../tests/test_aircraft_conceptual_physics.py#L566) | Pins analytic drag-polar optimum. |
| [test_unused_sizing_and_density_inputs_do_not_change_simplified_model](../../../tests/test_aircraft_conceptual_physics.py#L704) | Pins inactive SW/rho_alt inputs. |
| [test_aircraft_surrogate_out_of_contract_response_is_not_archived](../../../tests/test_aircraft_conceptual_physics.py#L809) | Pins archive envelope. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/aircraft_conceptual_8d.md) and [cross-validation script](../../../tests/_cross_validation/aircraft_conceptual_8d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

No full sizing, payload/MTOW closure, wing loading/trim, altitude-dependent engine lapse, climb/descent/route/wind, transonic drag rise, mission reserve calculation, certification or calibrated aircraft performance. The .95 multiplier is a convention, not a mission reserve analysis.

## Legacy comments and remaining checks

"Raymer-style" means inspired trend model, not a verbatim Raymer table/equation. The empty-weight power law and drag-efficiency constants have no demonstrated case-specific fit. The a=295.1 constant does not vary with rho_alt.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
