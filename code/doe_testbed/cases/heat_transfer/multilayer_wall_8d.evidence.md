# multilayer_wall_8d: implementation and evidence

Implementation: [multilayer_wall_nd.py](multilayer_wall_nd.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model

This instance has 8 independent layer conductivities k_i in [0.05,50] W/(m K). Every layer has fixed thickness delta_i=0.05 m; total thickness is **0.40 m**. Boundary temperatures are fixed at 80 C and 20 C. R_i=delta_i/k_i and R_total=sum R_i have units m^2 K/W. The reported heat flux is q=60/R_total W/m^2; midpoint interface temperature is 80-q*sum of the first 4 resistances. This is an exact steady one-dimensional series-resistance calculation, not a numerical PDE solve.

The 4-, 8- and 16-layer variants have total thicknesses 0.2, 0.4 and 0.8 m respectively. Dimension changes therefore also change the physical thickness; they are not refinements of one fixed-thickness wall. 'Midpoint' here is the central layer interface and geometric midpoint because layer thicknesses are equal.

## Source evidence

[MIT Unified Engineering, section 16.4: Thermal Resistance Circuits](https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node118.html), its conduction-resistance and series-circuit discussion, supports the resistance framework. The benchmark's equations also follow immediately by matching steady heat flux in each layer. This is theory-based analytical control evidence. No material manufacturer data, measured multilayer assembly or independently validated building model was used. The specific book equation/page locators elsewhere in project notes have not been verified from the complete cited edition.

## What tests establish

The shared targeted tests parameterize all three dimensions and compare exact heat flux, resistance and midpoint temperature at 1e-12 tolerance. Conductivity-scaling tests check that q scales and R inversely scales; layer-order tests check unchanged total resistance/flux but changed midpoint temperature. Input/output/failure tests establish contracts. Cross-validation supplies broad consistency checks; any equal-layer sensitivity test that merely reads stored sensitivity metadata is **not** a freshly calculated Sobol validation.

## Scope and remaining gaps

The model assumes perfect layer contact, fixed boundary surface temperatures, temperature-independent isotropic k, constant area and no generation. It omits convection films, radiation, moisture, transient storage and thermal bridges. No experimental validation is needed to establish the algebraic identity, but none should be claimed for a material stack. The conductivity domain is a benchmark range with no layer-specific material provenance. Comparisons across dimensions should state the changed total thickness.

中文简述：8层串联导热解析控制，总厚度0.40 m；测试验证代数，不代表材料墙体实验验证。不同维数同时改变总厚度，应明确说明。


## Audited local artifacts

- Implementation SHA-256: `baba9984246e5eb1d9b3d93ed30520894934890ede8359d8390fbc9b35ae1d15`.
- Targeted tests: `tests/test_multilayer_wall_physics.py` (10 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/multilayer_wall_8d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P2`, disposition `accepted analytic conduction control`, formula status `targeted_tested_no_code_change`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_series_resistance_flux_and_mid_temperature_are_exact` (line 33).
- `test_uniform_conductivity_resistance_scales_with_layer_count` (line 54).
- `test_layer_order_changes_mid_temperature_not_total_resistance_or_flux` (line 67).
- `test_metadata_marks_input_output_response_as_nonlinear` (line 85).
- `test_validation_material_declares_exact_series_resistance_scope` (line 100).
- `test_metadata_declares_public_sources_and_non_building_model_scope` (line 123).
- `test_web_audit_keeps_multilayer_wall_analytic_tier_and_scope_honest` (line 154).
- `test_declared_output_ranges_cover_domain_corners` (line 201).
- `test_nonfinite_or_nonphysical_wall_response_is_not_silent_archive_row` (line 215).
- `test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_series` (line 234).
