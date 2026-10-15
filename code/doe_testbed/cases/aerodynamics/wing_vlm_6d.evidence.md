# Evidence companion: `wing_vlm_6d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [wing_vlm_6d.py](wing_vlm_6d.py); SHA-256 `59851eb49d46baacbf2c1eea7c6d3699af7710cd99f0fbf9504fd9cff8108fba`.
- Model: 40-term Prandtl lifting-line Fourier collocation, not VLM.
- Registry: `docs/case_validation_status.csv` records `P2`, `plausibility only`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `AR` | - | [5.0, 14.0] | linear |
| `TR` | - | [0.2, 1.0] | linear |
| `sweep` | deg | [0.0, 35.0] | linear |
| `alpha` | deg | [0.0, 8.0] | linear |
| `Ma` | - | [0.1, 0.7] | linear |
| `twist` | deg | [-5.0, 5.0] | linear |

Outputs, in column order: `CL`, `CDi`, `e`.

## Equations, constants and implementation choices

Sref=1; b=sqrt(AR); croot=2/[b(1+TR)]; taper is linear in abs(cos(theta)); theta uses40 full-span interior stations. a0=2pi/sqrt(1-Ma^2); mu=c*a0/(4b). Solve sum_n A_n*sin(n theta)*(1+mu*n/sin(theta))=mu*alpha_geom, alpha_geom=alpha+twist*abs(cos(theta)). CL=pi*AR*A1; CDi=pi*AR*sum(n*A_n^2); e=CL^2/(pi*AR*CDi), clipped0..1, and convention e=1 when CDi=0. Multiply CL by cos(sweep)^2 and CDi by cos(sweep)^4. CDi is induced drag only.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy linear solve; no AVL, VSPAERO, panel or vortex-lattice backend. Singular solve is an error, not a least-squares replacement. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [NASA induced-drag coefficient](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/) | opened 2026-09-17; induced-drag identity only, no PLLT implementation validation. |
| [MIT AVL](https://web.mit.edu/drela/Public/web/avl/) | legacy official software boundary anchor; this case does not call AVL. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_induced_drag_matches_lifting_line_oswald_identity](../../../tests/test_wing_vlm_physics.py#L110) | Three finite-wing states match CDi=CL^2/(pi*AR*e) to1e-12; assembly identity. |
| [test_twisted_zero_lift_state_keeps_finite_induced_drag](../../../tests/test_wing_vlm_physics.py#L130) | Pins denominator-free Fourier drag at nontrivial zero total lift. |
| [test_sweep_scaling_is_documented_cos_squared_normal_flow_surrogate](../../../tests/test_wing_vlm_physics.py#L49) | Pins chosen conceptual sweep scaling, not swept-wing accuracy. |
| [test_lifting_line_solver_does_not_fall_back_to_least_squares](../../../tests/test_wing_vlm_physics.py#L197) | Pins linear-solver failure behavior. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/wing_vlm_6d.md) and [cross-validation script](../../../tests/_cross_validation/wing_vlm_6d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

No viscous/profile drag, stall, real swept lifting surface, body/control geometry, shock/buffet or tunnel calibration. Zero-loading e=1 is a finite-output convention, not measured efficiency.

## Legacy comments and remaining checks

Private method docstring Gamma=V*c*sum(A_n*sin(n theta)) conflicts with the matrix/output normalization (standard compatible form Gamma=2*b*V*sum). Another comment claims CL=CL_incomp/beta exactly: a0 enters both sides of the matrix, so finite-wing CL need not obey that exact scalar scaling. Cite code matrix/output equations as authoritative; textbook Eq8.27/Eq5.45 locators not checked against editions.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
