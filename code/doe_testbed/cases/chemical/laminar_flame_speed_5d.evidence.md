# laminar_flame_speed_5d: implementation and evidence

Implementation: [laminar_flame_speed_5d.py](laminar_flame_speed_5d.py).

## Evidence boundary

Audit date: 2026-09-17. This note compares the current Python implementation, targeted tests, cross-validation script and the status ledger. It is an implementation/source audit, not an experimental validation certificate. The ten targeted physics-test files covering these twelve cases were run together on 2026-09-17 with pytest; the process exited 0 with all displayed tests passing. Cross-validation scripts were inspected but not rerun in this audit. Book metadata and contents do not establish that a numerical coefficient appears on a cited page. No inaccessible full-text equation or table was treated as read.

## Implemented model and backend identity

**Default fidelity uses a parameterized Gülder-shaped empirical surrogate, not Cantera.** Low fidelity computes `S_L = 100 W phi^eta exp[-xi(phi-sigma)^2] * (T/298)^alpha * (P/1 atm)^beta * (1-2.5 d)`, in cm/s. Methane coefficients are W=0.422, eta=0.15, xi=5.18, sigma=1.075, alpha=2.105, beta=-0.504. Hydrogen coefficients are 2.30, 0, 2, 1, 1.75 and -0.2 respectively. Every coefficient is linearly blended by methane mole fraction. The hydrogen values are tuned proxy values; CO2 fraction d is 0–0.2.

The low-fidelity adiabatic temperature is a separate prescribed curve using reference temperatures 2220 K (methane) and 2380 K (hydrogen), asymmetric quadratic equivalence-ratio factors 0.4/0.6 and dilution factor (1-0.35d). Expansion is T_ad/T_initial times a small prescribed molecular-weight correction. These outputs are not chemical-equilibrium calculations.

High fidelity calls Cantera Solution('gri30.yaml'), sets CH4/H2 fuel with O2:N2=1:3.76, adds CO2 by mole fraction, and solves FreeFlame with width 0.05/sqrt(P_atm). Refinement starts at ratio=3, slope=0.08, curve=0.12, with a coarser retry. **Any exception at the high-fidelity dispatch causes unmarked fallback to the low-fidelity formula.** Therefore a requested high-fidelity result alone does not prove a Cantera solve occurred.

## Sources and exact support

[Gülder, SAE 841000, DOI 10.4271/841000](https://saemobilus.sae.org/papers/correlations-laminar-combustion-data-alternative-si-engine-fuels-841000) is genuine. Its accessible abstract names alcohols, propane and isooctane-related fuels; it does not establish the methane coefficient tuple above. No original coefficient table was inspected. The [OpenFOAM Gülder source interface](https://cpp.openfoam.org/dev/src_2thermophysicalModels_2laminarFlameSpeed_2Gulder_2Gulder_8H_source.html) establishes a related parameterized implementation, not these CH4/H2 blending constants. [Cantera's official flame-speed example](https://cantera.org/stable/examples/python/onedim/flame_speed.html), its FreeFlame setup/solve blocks, documents the numerical workflow. This is method documentation, not validation of this benchmark's output or fallback.

## Tests and remaining gaps

Targeted tests impose broad methane (25–50 cm/s) and hydrogen (200–300 cm/s) reference bands, approximate temperature/expansion ranges, dilution behavior and finite corners. A monkeypatched fallback test verifies that high-fidelity failure returns the proxy. Cross-validation checks trend signs and broad methane plausibility. These are not measured-error estimates, a grid-convergence study, a GRI-Mech validation for all mixtures, or evidence for linear hydrogen/methane coefficient mixing.

Applicability is a bounded smooth flame-response benchmark. Missing primary provenance: methane tuple, hydrogen tuning, CO2 factor 2.5, temperature/expansion proxy factors, and blend law. Fidelity logging is needed before any archive can be called Cantera-generated.

中文简述：默认是Gülder形状代理，高保真异常会静默回退；不能把所有结果称作Cantera计算。甲烷系数、氢混合与稀释系数尚缺原始表格证据。


## Audited local artifacts

- Implementation SHA-256: `80e2d0f6cc26b09b1072aa19956aaf77e9d0be31b8e0fd67de9b047b3c6a773b`.
- Targeted tests: `tests/test_laminar_flame_speed_physics.py` (14 test functions; parameterized executions can be more numerous).
- Cross-validation: `tests/_cross_validation/laminar_flame_speed_5d.py` (exists: True).
- Ledger: `docs/case_validation_status.csv`; recorded risk `P1`, disposition `accepted after resweep`, formula status `fixed_targeted_tested`. These are ledger labels, not a new approval.

### Actual targeted test inventory

- `test_low_fidelity_flame_speed_failure_is_not_silent_nan` (line 23).
- `test_invalid_low_fidelity_inputs_raise_before_archive_rows` (line 45).
- `test_private_physical_input_contract_rejects_malformed_and_nonfinite` (line 60).
- `test_private_physical_input_contract_rejects_declared_domain_violations` (line 82).
- `test_invalid_private_inputs_do_not_reach_flame_solvers` (line 93).
- `test_gulder_formula_rejects_inputs_outside_contract` (line 116).
- `test_high_fidelity_cantera_failure_falls_back_to_low_fidelity` (line 121).
- `test_low_fidelity_pure_hydrogen_stp_speed_matches_reference_band` (line 136).
- `test_low_fidelity_methane_stp_reference_preserved` (line 145).
- `test_low_fidelity_adiabatic_flame_temperature_uses_absolute_reference` (line 154).
- `test_dilution_preserves_fuel_to_oxidizer_ratio` (line 171).
- `test_low_fidelity_temperature_and_expansion_shapes_are_finite` (line 185).
- `test_declared_output_ranges_cover_low_fidelity_domain_corners` (line 200).
- `test_metadata_and_validation_material_declare_default_low_fidelity_scope` (line 210).
