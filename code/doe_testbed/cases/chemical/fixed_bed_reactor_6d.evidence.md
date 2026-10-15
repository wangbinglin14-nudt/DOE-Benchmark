# fixed_bed_reactor_6d: implementation and evidence

Implementation: [fixed_bed_reactor_6d.py](fixed_bed_reactor_6d.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model

The independent coordinate is normalized residence time s in [0,1]. With theta=C_A/C_A0 and u=T/T_in, the code integrates theta'=-Da theta exp[gamma(1-1/u)] and u'=beta Da theta exp[gamma(1-1/u)]. Here Da=10^k0_log exp[-Ea/(R T_in)] tau, beta=-dHr C_A0/(rho Cp T_in), gamma=Ea/(R T_in), R=0.008314 kJ mol^-1 K^-1, rho=800 kg m^-3 and Cp=2 kJ kg^-1 K^-1. Outputs are conversion, maximum sampled temperature and outlet temperature. RK45 uses rtol=1e-6 and atol=1e-9; bounded temperatures/exponents/rates are numerical safeguards.

This is an irreversible, first-order, adiabatic plug-flow response. It does not solve reaction equilibrium, axial dispersion, catalyst diffusion, pressure drop or heat exchange. References to equilibrium conversion or GHSV/bed length in descriptive text are not the implemented variables. The metadata expression for Da should include tau, as the implementation does.

## Sources and exact support

[Fogler publisher contents, fifth edition](https://www.informit.com/store/elements-of-chemical-reaction-engineering-9780133887518) locates **Adiabatic Operation at section 11.4, page 508**, not section 12.2. The [University of Michigan adiabatic energy-balance teaching module](https://websites.umich.edu/~elements/asyLearn/bits/ebforadiabatic/index.htm) supports the adiabatic mole/energy-balance framework. These are independent publisher/author-institution sources for structure and locator; neither authenticates the chosen rho, Cp, kinetic range or a specific chemical species. The full book equation was not inspected.

## What the tests establish

The targeted peak-equals-outlet test and energy-balance test establish internal agreement with T_out-T_in=-dHr C_A0 X/(rho Cp), with the latter using 2% relative tolerance. Extreme-Arrhenius and failure tests establish guarded numerical/input behavior. Cross-validation checks bounded conversion and expected activation-energy/residence-time trends. They do not compare measured kinetics or a separate packed-bed solver.

## Remaining limitation

Within declared bounds the largest adiabatic rise is 200*5/(800*2)=0.625 K. Thus T_max cannot exceed about 700.625 K in the unclipped balance; the 800 K thermal-failure narrative is unreachable in this domain. This is a code-derived bound, not a measurement. Material identity and coefficient provenance remain unspecified.

中文简述：实现为不可逆绝热一级反应代理；现有测试验证守恒和趋势。域内最大温升仅0.625 K，800 K失效阈值无法触及；书籍节号应改为11.4。


## Audited local artifacts

- Implementation SHA-256: `514f5c5c0cc569a98381ffc468e50be8d69c578c7f5a8515d75dc3a9b6e57f4f`.
- Targeted tests: `tests/test_fixed_bed_reactor_physics.py` (12 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/fixed_bed_reactor_6d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P1`, disposition `accepted after resweep`, formula status `fixed_targeted_tested`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_fixed_bed_reactor_solver_failure_is_not_silent_nan` (line 15).
- `test_source_input_documentation_matches_k0_log_variable` (line 28).
- `test_private_physical_input_contract_rejects_malformed_arrays` (line 50).
- `test_private_physical_input_contract_rejects_declared_domain_violations` (line 84).
- `test_private_physical_input_contract_rejects_nonfinite_values` (line 102).
- `test_nonfinite_or_nonphysical_fixed_bed_response_is_not_silent_archive_row` (line 117).
- `test_ode_nonfinite_or_nonphysical_trajectory_is_not_silent` (line 131).
- `test_adiabatic_irreversible_peak_is_outlet_temperature` (line 163).
- `test_temperature_rise_matches_adiabatic_energy_balance` (line 177).
- `test_arrhenius_guard_keeps_extreme_trials_finite` (line 191).
- `test_fixed_bed_audit_keeps_20260608_boundary_honest` (line 203).
- `test_fixed_bed_audit_records_packed_bed_multiphysics_boundary` (line 219).
