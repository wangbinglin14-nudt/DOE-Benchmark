# casting_solidification_5d: implementation and evidence

Implementation: [casting_solidification_5d.py](casting_solidification_5d.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model

The implementation advances a one-dimensional half-slab enthalpy field explicitly. Constants are T_m=615 C, rho_s=2680, rho_l=2560 kg/m^3, c_s=963, c_l=1080 J/(kg K), k_s=155, k_l=90 W/(m K), and latent heat 397000 J/kg. Mold/contact exchange uses h_eff=[1/h_contact+0.001/k_mold]^-1, so mold conductivity acts through an assumed 1 mm resistance length.

The piecewise H-to-T mapping includes an isothermal latent-heat plateau. Solid fraction is clipped from (H_liquid-H)/(rho_s L_f); conductivity is interpolated between solid/liquid values and face conductivities are arithmetic means. The time step is 0.45 dx^2/alpha_max. One end exchanges heat with the mold; the center plane is insulated. The simulation horizon is min[80*(L_half/2)^2/alpha_max,10000] s.

The reported solidification time is the first checked time with every cell below the enthalpy threshold corresponding to approximately 98% local solidification. **If this is not reached, the function returns t_max without a censoring indicator.** Reported front speed is distance of a solid-prefix/mushy-cell front divided by that time; when no front is resolved, the fallback uses L_half/time. It is not an instantaneous interface speed or a measured completion time. The center-peak output includes the initial pouring temperature in source-free cooling.

## Sources and support

Flemings (1974) and Poirier & Geiger are genuine casting/transport references (see the preceding bibliography audit); no original table establishing this alloy's coefficient tuple or exact equation locator was inspected. Flemings ISBN correction is 978-0-07-021283-1. [COMSOL's apparent-heat-capacity theory](https://doc.comsol.com/5.3/doc/com.comsol.help.heat/heat_ug_theory.07.15.html) explains latent heat and density-conservation requirements, but uses a different numerical formulation. [COMSOL's metal solidification example](https://doc.comsol.com/6.4/doc/com.comsol.help.models.heat.cooling_solidification_metal/cooling_solidification_metal.html) is method context; its material table is not the source of this code's alloy constants. These references must not be presented as an independent reproduction of this implementation.

## Tests, scope and gaps

Targeted tests check front interpolation on constructed arrays, average-speed behavior, mold-conductivity trends and the positive fallback. Cross-validation checks pouring-temperature/contact/thickness trends and broad temperatures. They do not establish mesh/time convergence, a Stefan analytical solution, energy conservation with the two densities, or measured casting agreement. No documented material identity and coefficient table is supplied. Suitable as a simplified enthalpy-response benchmark; excludes shrinkage, flow, nucleation, phase segregation, temperature-dependent properties and coupled mold evolution. Censored times and fallback speeds require explicit disclosure before physical interpretation.

中文简述：焓法代理缺网格收敛、Stefan解析解及具体合金参数证据；未凝固时仍返回截止时间，未解析前沿仍返回备用速度，不能解释为实测完成凝固。


## Audited local artifacts

- Implementation SHA-256: `5f444d32fa126707c511362d72afcaa18341b580ccf9074529c326cb5e28be2b`.
- Targeted tests: `tests/test_casting_solidification_physics.py` (10 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/casting_solidification_5d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P1`, disposition `accepted after resweep`, formula status `fixed_targeted_tested`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_casting_solver_failure_is_not_silent_nan` (line 10).
- `test_private_physical_input_validator_rejects_before_fdm` (line 23).
- `test_malformed_casting_response_is_not_archived` (line 57).
- `test_front_position_interpolates_across_mushy_cell` (line 70).
- `test_front_position_handles_initial_mushy_boundary_cell` (line 81).
- `test_front_position_reaches_full_length_when_all_cells_are_solid` (line 92).
- `test_mean_front_speed_matches_interpolated_solidification_depth` (line 103).
- `test_higher_mold_conductivity_shortens_solidification_time` (line 122).
- `test_front_speed_falls_back_to_nonzero_average_when_front_is_unresolved` (line 140).
- `test_casting_audit_keeps_20260608_boundary_honest` (line 160).
