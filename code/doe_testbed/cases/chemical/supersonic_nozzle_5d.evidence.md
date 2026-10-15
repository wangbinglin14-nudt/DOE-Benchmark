# supersonic_nozzle_5d: implementation and evidence

Implementation: [supersonic_nozzle_5d.py](supersonic_nozzle_5d.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model

The code solves F(M)=epsilon with F(M)=M^-1[2/(gamma+1)*(1+(gamma-1)M^2/2)]^((gamma+1)/(2(gamma-1))) on subsonic or supersonic branches. Normal-shock Mach and stagnation-pressure ratios are the ideal perfect-gas relations. T_exit=T0/[1+(gamma-1)M_exit^2/2]; eta is the shock stagnation-pressure ratio (or one when no shock).

Branch logic matters. At p_back/p0>=0.9999 it uses the subsonic area–Mach root while retaining the throat-as-critical-area assumption. At sufficiently low back pressure it returns the supersonic design exit. Between design exit pressure and the post-exit-shock pressure, it returns the same post-normal-shock exit Mach across the interval. At larger back pressure it places an internal shock using the separate prescribed shock_pos parameter: A_shock/A_throat=1+(epsilon-1)*shock_pos, then propagates a subsonic downstream area relation.

**The solver does not determine shock position from back pressure or enforce a continuous back-pressure-consistent nozzle solution.** The high-back-pressure branch is not a full unchoked-flow solution. A zero-position shock can reach the sonic endpoint, for which the selected numerical bracket may fail; explicit failure handling does not remove that modelling boundary.

## Sources and exact support

[NASA Glenn isentropic equations](https://www.grc.nasa.gov/www/k-12/airplane/isentrop.html), equations 6, 7 and 9, support the ideal algebra. [NASA Glenn normal-shock relations](https://www.grc.nasa.gov/www/k-12/airplane/normal.html), the Mach and total-pressure equations, support the shock jump. [NASA's record for NACA Report 1135](https://ntrs.nasa.gov/citations/19930091059) confirms the report identity: **NACA-TR-1135**, not TN-1135. These official sources do not validate the benchmark's branch interpolation or independently chosen shock position; no full report table-by-table check was performed.

## Tests and applicability

Targeted tests verify the implemented branch equations, boundary choices, root failures and domain contracts. Cross-validation checks design Mach examples and subsonic/supersonic plausibility. This is algebraic/internal verification, not comparison against a back-pressure-resolved flow field, CFD or experiment. It is suitable as a piecewise ideal-gas response benchmark; it omits viscous loss, separation, boundary layers, shock trains and real-gas effects. A physically closed back-pressure sweep remains a separate validation gap.

中文简述：等熵与正激波公式可核实，但分支规则未完整闭合背压条件；shock_pos为独立指定量，不能当作真实喷管背压扫描求解器。


## Audited local artifacts

- Implementation SHA-256: `335a9a5f692fd2e5cd87567059d48d646c8a4113327fe7ac1e86b21785d7a701`.
- Targeted tests: `tests/test_supersonic_nozzle_physics.py` (9 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/supersonic_nozzle_5d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P1`, disposition `accepted after resweep`, formula status `fixed_targeted_tested`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_nozzle_surrogate_failure_is_not_silent_nan` (line 19).
- `test_area_mach_solver_failure_is_not_silent_fallback` (line 32).
- `test_nonfinite_or_nonphysical_nozzle_response_is_not_silent_archive_row` (line 42).
- `test_private_physical_input_contract_rejects_malformed_and_nonfinite` (line 60).
- `test_private_physical_input_contract_rejects_declared_domain_violations` (line 82).
- `test_invalid_private_inputs_do_not_reach_nozzle_solver` (line 92).
- `test_in_nozzle_shock_exit_mach_uses_downstream_subsonic_branch` (line 118).
- `test_flow_regime_boundaries_use_exact_critical_pressure_ratios` (line 161).
- `test_declared_output_ranges_cover_domain_corners` (line 201).
