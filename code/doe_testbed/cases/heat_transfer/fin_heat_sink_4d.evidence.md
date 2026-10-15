# fin_heat_sink_4d: implementation and evidence

Implementation: [fin_heat_sink_4d.py](fin_heat_sink_4d.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model

A rectangular straight fin has width W=0.05 m, height H, thickness t, conductivity k and convection coefficient h. With A_c=Wt, P=2(W+t), m=sqrt(hP/(kA_c)) and base/ambient temperatures 80/25 C, eta=tanh(mH)/(mH), Q=sqrt(hPkA_c)*tanh(mH)*55 W and R_th=55/Q K/W. At |mH|<1e-4 the efficiency uses 1-(mH)^2/3+2(mH)^4/15.

These are uniform-property, one-dimensional **insulated-tip** fin relations; convection is from the lateral perimeter. Fin-array interactions, tip convection, radiation, contact resistance, base spreading and fan performance are absent. The 'heat sink' name does not imply an assembly-level thermal model.

## Source evidence

[MIT Unified Engineering, section 18.2: Heat Transfer From a Fin](https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node128.html) provides the governing ODE in equation 18.8, the zero tip-gradient boundary immediately following equation 18.9, the insulated-tip solution in equation 18.11, and the matching heat-rate expression in equation 18.13. These equations were visible in the official HTML, providing direct formula-level support from one teaching source. The specific Incropera equation 3.70 locator in project references was not checked against the cited edition's full text. This is analytical support, not empirical validation.

## What tests establish

The targeted formula test reconstructs Q, eta and R_th to 1e-12 tolerance at a selected case; other tests check the small-argument limit, physical bounds, corners and failure behavior. Cross-validation checks short-fin efficiency and heat-transfer trends. These establish closed-form implementation consistency, not validation of a specific physical fin or convection coefficient.

The benchmark is appropriate for an isolated uniform fin when transverse gradients and property changes can be neglected. The h range is an assigned parameter range, not a documented experimental correlation. A measured fin dataset and an applicability check for one-dimensional conduction are remaining gaps if engineering accuracy is claimed.

中文简述：解析控制案例采用绝热端直翅片；公式一致性已写入测试，但没有真实散热器或对流系数实验验证，不能扩展为整机散热性能。


## Audited local artifacts

- Implementation SHA-256: `ff6ddb7545511635e91ca335e485235288a13ac6fa163e54346b81227eb13f6a`.
- Targeted tests: `tests/test_fin_heat_sink_physics.py` (9 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/fin_heat_sink_4d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P1`, disposition `accepted metadata correction`, formula status `fixed_targeted_tested`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_insulated_tip_efficiency_uses_smooth_small_argument_limit` (line 14).
- `test_heat_rate_matches_lateral_area_insulated_tip_formula` (line 24).
- `test_efficiency_stays_in_physical_range_without_clipping` (line 47).
- `test_failure_boundary_targets_low_fin_efficiency` (line 57).
- `test_metadata_declares_nonlinear_monotone_interaction_response_scope` (line 78).
- `test_declared_output_ranges_cover_domain_corners` (line 86).
- `test_private_physical_input_validator_rejects_before_formula_evaluation` (line 96).
- `test_nonfinite_or_nonphysical_fin_response_is_not_silent_archive_row` (line 124).
- `test_fin_audit_keeps_20260608_boundary_honest` (line 134).
