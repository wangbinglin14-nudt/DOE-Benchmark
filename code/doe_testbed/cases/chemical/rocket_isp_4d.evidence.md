# rocket_isp_4d: implementation and evidence

Implementation: [rocket_isp_4d.py](rocket_isp_4d.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model

Inputs are oxidizer/fuel ratio MR, chamber pressure Pc, nozzle area ratio epsilon and inlet temperature T0. The thermochemistry is a hand-built surrogate: phi=clip(8/MR,0.2,3), phi_opt=8/5.8; T_c uses a 3500 K scale, exp[-0.6(phi-phi_opt)^2], an inlet-temperature factor 1+0.05(T0-90)/210, and pressure factor 1+0.01 log10(Pc/10), with stated internal clips and a final 2000–3700 K clip. Excess hydrogen sets an effective molecular mass between 8 and 18 kg/kmol and gamma=clip(1.15+0.05 excess,1.1,1.35).

With R=8314/M_eff, the ideal characteristic velocity is sqrt(gamma R T_c)/[gamma(2/(gamma+1))^((gamma+1)/(2gamma-2))], subsequently clipped to 1000–2600 m/s. A supersonic area–Mach root determines exit pressure. Vacuum thrust coefficient includes ideal momentum and epsilon*p_e/p_c; I_sp=c_star*C_F/9.80665. Ambient-pressure thrust loss is absent.

**There is no NASA polynomial evaluation or CEA equilibrium call.** Statements about interpolation of a Sutton table are not backed by an embedded table, fitted dataset or documented fit residuals in this implementation.

## Sources and exact support

[NASA Glenn isentropic-flow equations](https://www.grc.nasa.gov/www/k-12/airplane/isentrop.html), equations 6, 7 and 9 on the page, support perfect-gas pressure/temperature ratios and the area–Mach relation. [NASA CEA history](https://www.nasa.gov/glenn/research/chemical-equilibrium-with-applications/cea-history/) describes the actual equilibrium program; it is context for an omitted capability, not a backend used here. Sutton/CEA bibliographic references do not authenticate the surrogate's 3500, 0.6, 0.05, 0.01 or composition factors; no claimed original table was read.

## Tests and remaining gaps

Targeted tests reconstruct ideal nozzle C_F and I_sp using the implementation's own returned c_star, test an expected MR peak, temperature trends, bounds and failures. This checks algebra and surrogate behavior; it does not independently validate c_star or chamber equilibrium. Broad cross-validation I_sp bands do not establish prediction accuracy against a real engine.

Scope is a vacuum, calorically perfect ideal-nozzle benchmark with empirical thermochemical response. It omits finite-rate chemistry, dissociation-consistent gamma, frozen/equilibrium expansion choice, losses, cooling and feed-system constraints. A documented CEA grid and fitted-error assessment would be required to call its thermochemistry CEA-based.

中文简述：喷管代数有依据，但燃烧室温度/组分是手工代理；代码没有NASA多项式或CEA调用，不能据书籍引用宣称真实热化学验证。


## Audited local artifacts

- Implementation SHA-256: `0b510476170bee1781bceb25efc8b59a4e78a8656dcb7cbb7def3f9568c42b27`.
- Targeted tests: `tests/test_rocket_isp_physics.py` (15 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/rocket_isp_4d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P1`, disposition `accepted after resweep`, formula status `fixed_targeted_tested`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_rocket_isp_surrogate_failure_is_not_silent_nan` (line 8).
- `test_nonfinite_or_nonphysical_rocket_response_is_not_silent_archive_row` (line 21).
- `test_private_physical_input_contract_rejects_malformed_and_nonfinite` (line 39).
- `test_private_physical_input_contract_rejects_declared_domain_violations` (line 59).
- `test_invalid_private_inputs_do_not_reach_surrogate_solver` (line 69).
- `test_nozzle_area_mach_solve_failure_is_not_silent_fallback` (line 82).
- `test_isp_peak_is_fuel_rich_not_stoichiometric` (line 124).
- `test_chamber_temperature_peak_is_fuel_rich` (line 139).
- `test_initial_temperature_adds_chamber_enthalpy` (line 146).
- `test_high_expansion_solution_remains_finite_and_physical` (line 154).
- `test_declared_domain_corners_remain_inside_expected_output_ranges` (line 163).
- `test_out_of_expected_response_envelope_is_not_silent_archive_row` (line 184).
- `test_isp_matches_independent_vacuum_nozzle_equation` (line 196).
- `test_metadata_and_validation_material_declare_not_cea_scope` (line 210).
- `test_metadata_pins_official_nasa_boundary_sources` (line 240).
