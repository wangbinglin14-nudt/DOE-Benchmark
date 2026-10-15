# membrane_reactor_5d: implementation and evidence

Implementation: [membrane_reactor_5d.py](membrane_reactor_5d.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model

With s in [0,1], A(0)=1, B(0)=B0 and Q(0)=0, the code solves A'=-Da A+(Da/K_eq)B, B'=Da A-(Da/K_eq)B-Da_m B, Q'=Da_m B. A second baseline integration removes the membrane sink from B' but still accumulates hypothetical Q'. Outputs are 1-A(1), B(1), and eta=Q_membrane/Q_baseline. RK45 tolerances are 1e-8 and 1e-10.

**Pe is validated but unused.** This is a spatial plug-flow, constant-coefficient reaction/removal model with four effective parameters, not a five-variable axial-dispersion model. The ODE is linear in state; its parameter-to-output map is nonlinear. Eta is an integrated-removal ratio, not a measured membrane separation efficiency or a local outlet-flux ratio.

## Sources and exact support

The [University of Michigan membrane-reactor introduction](https://websites.umich.edu/~elements/web_mod/membrane/intro.htm) supports selective removal as a reactor concept. [Fogler's fifth-edition publisher contents](https://www.informit.com/store/elements-of-chemical-reaction-engineering-9780133887518) places membrane reactors in **section 6.4, page 217**, not the section 6.7 cited in older notes. Neither page establishes the precise dimensionless equations, chosen rate constants, or the baseline eta definition. Full textbook equations were not inspected. These two sources supply institutional/publisher context, not an independent quantitative reference case.

## What tests establish

The constant-coefficient matrix-exponential test compares the ODE solution to exp(M) at Da=2, Da_m=1, K_eq=2 and B0=0.03, at rtol=1e-8 and atol=1e-10. It is a meaningful independent numerical solution of the same mathematical model, not independent physical validation. Other tests establish that removal raises conversion and lowers B, Pe is inert, and eta uses integrated removal. Cross-validation adds bounds and expected trend checks; no measured membrane permeance, axial-dispersion solver or experiment is compared.

## Scope and unresolved provenance

Useful as an analytic/numerical reactor control. It omits variable flow, pressure drop, transport resistance, nonisothermal kinetics, membrane material selectivity and axial dispersion. The dimensionless parameter domain and eta convention are benchmark choices. Source matching remains necessary before assigning these equations to a particular textbook example or real membrane system.

中文简述：常系数ODE有矩阵指数对照，但属于模型内验证；Pe完全不参与计算，实际为四维响应，不能宣称验证了轴向弥散。


## Audited local artifacts

- Implementation SHA-256: `c783701685596c2bbe1ca2d722e10cf79c8daf68ba47e3cb894cb7cad58a5026`.
- Targeted tests: `tests/test_membrane_reactor_physics.py` (15 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/membrane_reactor_5d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P1`, disposition `accepted after resweep`, formula status `fixed_targeted_tested`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_membrane_reactor_solver_failure_is_not_silent_nan` (line 9).
- `test_nonfinite_or_nonphysical_membrane_response_is_not_silent_archive_row` (line 22).
- `test_private_physical_input_contract_rejects_malformed_and_nonfinite` (line 40).
- `test_private_physical_input_contract_rejects_declared_domain_violations` (line 62).
- `test_invalid_private_inputs_do_not_reach_ode_solver` (line 72).
- `test_ode_solve_failures_are_not_silent_pseudo_results` (line 85).
- `test_no_membrane_baseline_solve_failure_is_not_silent_pseudo_eta` (line 99).
- `test_membrane_permeation_increases_conversion_and_lowers_product_exit` (line 129).
- `test_eta_mem_is_integrated_flux_ratio_not_static_equilibrium_shortfall` (line 137).
- `test_reactor_matches_constant_coefficient_matrix_exponential_control` (line 146).
- `test_membrane_utilisation_drops_when_permeation_depletes_product_along_reactor` (line 192).
- `test_pe_remains_declared_inert_variable_in_plug_flow_limit` (line 200).
- `test_metadata_declares_pe_inert_and_response_nonlinear_scope` (line 207).
- `test_membrane_reactor_audit_keeps_20260608_boundary_honest` (line 240).
- `test_membrane_reactor_audit_records_comsol_workflow_boundary` (line 257).
