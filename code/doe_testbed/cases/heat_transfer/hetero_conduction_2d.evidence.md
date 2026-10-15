# hetero_conduction_2d: implementation and evidence

Implementation: [hetero_conduction_2d.py](hetero_conduction_2d.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model and analytical reduction

Despite the '2d' case-id suffix, there are five inputs: k_A, k_B, vertical-interface location x_split, bottom hot temperature and top cold temperature. The unit-square problem solves div(k grad T)=0, with piecewise conductivity k(x), prescribed temperatures at bottom/top, and insulated left/right edges. Finite differences use harmonic conductivity across horizontal material faces and a sparse linear solve. Default grid size is 25; the alternate grid is 60.

For these exact boundary conditions the continuum solution is T(x,y)=T_hot+(T_cold-T_hot)y: there is no horizontal temperature gradient or nontrivial thermal bridge. Thus centerline temperature is (T_hot+T_cold)/2, maximum face-component heat flux is max(k_A,k_B)*(T_hot-T_cold), provided both materials are represented, and hot-surface temperature is the imposed boundary value. The reported q_max is the maximum absolute separate face component, not a reconstructed vector-magnitude maximum. Interface location does not change this maximum; it would change total integrated heat flow, which is not the reported observable.

T_mid is sampled at row N//2. It corresponds to y=1/2 for odd N=25, but y=30/59 for even N=60. Therefore changing the fidelity introduces a sampling-location shift rather than evaluating an identical midpoint observable. These reductions and the sampling observation follow from code and the boundary-value equations.

## Sources and support

The Fourier-conduction/resistance framework is documented in [MIT Unified Engineering, section 16.4: Thermal Resistance Circuits](https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node118.html). This is general theory, not a heterogeneous-interface numerical validation dataset. Patankar's finite-volume reference is methodological context; no cited full-text discretization page was independently inspected. Neither reference validates the chosen spatial grid, interface placement, or the description of a complex two-dimensional response.

## Tests and limitations

Targeted tests check harmonic means, uniform Fourier flow, the analytical reduction for the declared boundary conditions, split-location invariance and finite large-contrast outputs. Private solver conductivity flooring is separately tested; public input validation rejects illegal conductivity. These are useful solver/algebra controls, not independent measured-field validation. Cross-validation plausibility cannot establish richer geometry than the implemented problem.

The model omits contact resistance, temperature-dependent coefficients, internal heat sources, convection and transient behavior. In this geometry it is effectively a parallel-conduction control. The midpoint fidelity mismatch and claims of nontrivial thermal bridging remain issues to resolve before stronger scientific interpretation.

中文简述：虽然使用二维网格，但本边界条件可解析退化为线性温度场；界面位置不影响已报告主要输出。60点网格的“中点”不是几何中点，需披露保真度采样差异。


## Audited local artifacts

- Implementation SHA-256: `07a709b222bf347ac306702c6f785e27732b4d72e3d46f2555036bb81d13f2a9`.
- Targeted tests: `tests/test_hetero_conduction_physics.py` (11 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/hetero_conduction_2d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P1`, disposition `accepted after resweep`, formula status `fixed_targeted_tested`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_heterogeneous_conduction_solver_failure_is_not_silent_nan` (line 8).
- `test_private_physical_input_validator_rejects_before_fdm` (line 21).
- `test_malformed_heterogeneous_conduction_response_is_not_archived` (line 55).
- `test_face_flux_uses_harmonic_mean_at_material_jump` (line 68).
- `test_flux_for_uniform_vertical_conduction_matches_fourier_law` (line 83).
- `test_declared_boundary_conditions_reduce_to_vertical_conduction_control` (line 97).
- `test_x_split_is_inert_for_current_outputs_and_boundary_conditions` (line 112).
- `test_metadata_declares_nonlinear_response_and_inert_x_split_scope` (line 133).
- `test_solve_single_keeps_qmax_finite_for_large_conductivity_contrast` (line 152).
- `test_negative_conductivity_perturbation_is_clipped_to_positive_floor` (line 167).
- `test_hetero_conduction_audit_keeps_20260608_boundary_honest` (line 180).
