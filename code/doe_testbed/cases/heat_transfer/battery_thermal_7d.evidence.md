# battery_thermal_7d: implementation and evidence

Implementation: [battery_thermal_7d.py](battery_thermal_7d.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model

A single lumped temperature obeys m c_p dT/dt=I^2 R(T)-I T dU/dT-h A(T-T_cool), with m=0.1 kg, nominal capacity 3 Ah, I=3*C_rate A and discharge duration 3600/C_rate s. Initial temperature equals coolant temperature. R(T)=R_ref exp[(Ea/8.314)(1/T-1/298.15)], with milliohms converted to ohms and a constant dU/dT=-3e-4 V/K. Temperatures used in generation are clipped to 200–600 K and resistance to 1e-6–10 ohm as numerical guards. RK45 tolerances are 1e-6/1e-9 with 100 requested samples and a maximum step of duration/50. Outputs are maximum and end temperatures in Celsius and temperature rise.

## Source evidence

[Bernardi, Pawlikowski and Newman, original LBL manuscript](https://escholarship.org/content/qt9fx5f0h8/qt9fx5f0h8.pdf) supports the identity of the general battery energy-balance paper. Its internal title page, PDF page 4 in the audited copy, spells **Ellen Pawlikowski**, not Ed. Journal DOI: [10.1149/1.2113792](https://doi.org/10.1149/1.2113792). Prior Crossref and original-title-page verification are multiple metadata sources; the complete derivation/sign convention was not independently rederived from the full paper. The general citation does not establish this specific cell's 0.1 kg mass, 3 Ah capacity, constant entropy slope or temperature-resistance law.

## Tests and limitations

Targeted tests compare the reversible-heat contribution with a zero-entropy-term variant, enforce output consistency, finite extreme behavior and failures. Cross-validation checks current/cooling/resistance trends. These test model behavior rather than measured cell temperature or identified entropy curves. Ledger wording 'accepted after resweep' is a recorded workflow disposition, not external validation.

The model assumes one uniform cell temperature, constant state-independent properties apart from R(T), fixed current, and fixed coolant boundary. It omits state-of-charge-dependent entropy/resistance, aging, pack gradients, side reactions and thermal runaway. The lumped-temperature assumption has no documented Biot-number check. Cell-specific parameter measurements and their uncertainties remain missing.

中文简述：为单节点能量平衡，测试验证项和趋势而非真实电芯。Bernardi理论不能证明本例质量、容量及固定熵系数；尚缺材料参数和集总假设证据。


## Audited local artifacts

- Implementation SHA-256: `29d437699939fa7e44f834f5fefb1d2e88e4638f8c3ad23209210a7380f99255`.
- Targeted tests: `tests/test_battery_thermal_physics.py` (9 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/battery_thermal_7d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P1`, disposition `accepted after resweep`, formula status `fixed_targeted_tested`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_battery_thermal_solver_failure_is_not_silent_nan` (line 21).
- `test_malformed_battery_response_is_not_archived` (line 34).
- `test_delta_t_must_match_peak_rise` (line 47).
- `test_private_physical_input_validator_rejects_before_ode` (line 60).
- `test_simulate_rejects_inputs_outside_lumped_ode_contract` (line 109).
- `test_reversible_entropic_heat_increases_low_crate_temperature` (line 126).
- `test_arrhenius_temperature_guard_keeps_extreme_case_finite` (line 142).
- `test_declared_output_ranges_cover_domain_corners` (line 155).
- `test_official_source_boundary_keeps_battery_case_lumped_ode_scope` (line 165).
