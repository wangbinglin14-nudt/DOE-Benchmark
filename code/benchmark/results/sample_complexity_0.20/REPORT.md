# Sample-Complexity Recovery — Discovery 2

**Metric**: `nrmse`  
**Threshold**: 0.200  
**Rule budget**: 10·d (Loeppky et al. 2009)

## Headline

At budget = 10·d, 3 of 7 main topology classes have ≥50 % of methods reaching the target 20% NRMSE.

## Per-topology summary

| topology           | analysis_type                   | is_primary   |   methods_total |   methods_pass_at_10d |   frac_pass_at_10d |   median_n_star |   max_n_star |   n_methods_never_pass |   n_cases | case_ids                                                                                                                                                                                                                                                                                                                                                                                                                  | exploratory   |   n_failed_cells_per_topology |
|:-------------------|:--------------------------------|:-------------|----------------:|----------------------:|-------------------:|----------------:|-------------:|-----------------------:|----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|------------------------------:|
| smooth_unimodal    | penalized_failed_in_denominator | True         |              20 |                    11 |              0.550 |           5.000 |       50.000 |                      5 |        14 | airfoil_stall_2d;airfoil_transonic_5d;battery_thermal_7d;casting_solidification_5d;fixed_bed_reactor_6d;laminar_flame_speed_5d;rocket_isp_4d;rotor_bemt_6d;solar_pv_yield_6d;supersonic_intake_5d;thin_plate_buckling_5d;tuned_mass_damper_3d;waverider_4d;wing_vlm_6d                                                                                                                                                    | False         |                           521 |
| discontinuous      | penalized_failed_in_denominator | True         |              20 |                    11 |              0.550 |          10.000 |       50.000 |                      5 |         5 | dam_seepage_6d;groundwater_contamination_6d;rotor_bemt_6d;supersonic_intake_5d;supersonic_nozzle_5d                                                                                                                                                                                                                                                                                                                       | False         |                           270 |
| strong_interaction | penalized_failed_in_denominator | True         |              20 |                    12 |              0.600 |          10.000 |       20.000 |                      6 |        21 | aircraft_conceptual_8d;battery_thermal_7d;cantilever_beam_5d;casting_solidification_5d;dam_seepage_6d;fin_heat_sink_4d;fixed_bed_reactor_6d;groundwater_contamination_6d;laminar_flame_speed_5d;membrane_reactor_5d;oil_reservoir_9d;plate_with_hole_5d;pressure_vessel_4d;rotor_bemt_6d;soil_consolidation_5d;solar_pv_yield_6d;supersonic_intake_5d;tuned_mass_damper_3d;waverider_4d;wind_turbine_power_6d;wing_vlm_6d | False         |                          1164 |
| threshold_jump     | penalized_failed_in_denominator | True         |              20 |                     6 |              0.300 |          15.000 |       50.000 |                      8 |         8 | airfoil_stall_2d;airfoil_transonic_5d;dam_seepage_6d;groundwater_contamination_6d;hetero_conduction_2d;supersonic_intake_5d;supersonic_nozzle_5d;thin_plate_buckling_5d                                                                                                                                                                                                                                                   | False         |                           180 |
| heterogeneous      | penalized_failed_in_denominator | True         |              20 |                     0 |              0.000 |          20.000 |       50.000 |                      8 |         8 | airfoil_stall_2d;airfoil_transonic_5d;casting_solidification_5d;fixed_bed_reactor_6d;groundwater_contamination_6d;hetero_conduction_2d;soil_consolidation_5d;tuned_mass_damper_3d                                                                                                                                                                                                                                         | False         |                           180 |
| smooth_monotone    | penalized_failed_in_denominator | True         |              20 |                     0 |              0.000 |          50.000 |       50.000 |                      7 |        19 | aircraft_conceptual_8d;cantilever_beam_5d;dam_seepage_6d;fin_heat_sink_4d;groundwater_contamination_6d;hetero_conduction_2d;membrane_reactor_5d;multilayer_wall_16d;multilayer_wall_4d;multilayer_wall_8d;oil_reservoir_9d;plate_with_hole_5d;pressure_vessel_4d;soil_consolidation_5d;ten_bar_truss_10d;variable_beam_10d;variable_beam_20d;variable_beam_5d;wind_turbine_power_6d                                       | False         |                          2474 |
| sparse_effects     | penalized_failed_in_denominator | True         |              20 |                     0 |              0.000 |          50.000 |       50.000 |                     10 |         5 | groundwater_contamination_6d;ten_bar_truss_10d;variable_beam_10d;variable_beam_20d;variable_beam_5d                                                                                                                                                                                                                                                                                                                       | False         |                          1275 |

## Exploratory topologies

None.

## n*/d table (rows=method, columns=topology)

Lower is better.  ``NaN`` means *no observed budget* satisfied the threshold for that (method, topology) pair — typically indicating a fundamental sample-efficiency failure.

| method         | discontinuous   | heterogeneous   | smooth_monotone   | smooth_unimodal   | sparse_effects   | strong_interaction   | threshold_jump   |
|:---------------|:----------------|:----------------|:------------------|:------------------|:-----------------|:---------------------|:-----------------|
| akmcs          | 10.0            | 50.0            | 20.0              | 10.0              | 50.0             | 10.0                 | 20.0             |
| botorch_qei    | 10.0            | 50.0            | 20.0              | 10.0              | ∞                | 10.0                 | 20.0             |
| botorch_saasbo | 10.0            | 20.0            | 50.0              | 5.0               | 50.0             | 10.0                 | 10.0             |
| ego_ei         | 10.0            | 20.0            | 20.0              | 10.0              | 50.0             | 10.0                 | 20.0             |
| halton+gp      | 5.0             | 20.0            | 50.0              | 5.0               | 50.0             | 5.0                  | 10.0             |
| lhs+gp         | 10.0            | 20.0            | 50.0              | 5.0               | 50.0             | 10.0                 | 10.0             |
| lhs_opt+gb     | ∞               | ∞               | ∞                 | 50.0              | ∞                | ∞                    | ∞                |
| lhs_opt+gp     | 5.0             | 20.0            | 50.0              | 5.0               | 50.0             | 5.0                  | 10.0             |
| lhs_opt+gp_rbf | 10.0            | 20.0            | 50.0              | 5.0               | 50.0             | 10.0                 | 20.0             |
| lhs_opt+knn    | ∞               | ∞               | ∞                 | ∞                 | ∞                | ∞                    | ∞                |
| lhs_opt+mlp    | ∞               | ∞               | ∞                 | ∞                 | ∞                | ∞                    | ∞                |
| lhs_opt+poly3  | 20.0            | ∞               | ∞                 | 50.0              | ∞                | 20.0                 | ∞                |
| lhs_opt+poly5  | 20.0            | 50.0            | 50.0              | 50.0              | ∞                | 20.0                 | 50.0             |
| lhs_opt+rbf    | 20.0            | ∞               | 50.0              | 20.0              | ∞                | 10.0                 | ∞                |
| lhs_opt+rf     | ∞               | ∞               | ∞                 | ∞                 | ∞                | ∞                    | ∞                |
| lhs_opt+svr    | 50.0            | ∞               | ∞                 | ∞                 | ∞                | ∞                    | ∞                |
| max_variance   | 5.0             | 20.0            | 20.0              | 5.0               | 20.0             | 5.0                  | 10.0             |
| random+gp      | 10.0            | 20.0            | 50.0              | 5.0               | 50.0             | 10.0                 | 20.0             |
| random+linear  | ∞               | ∞               | ∞                 | ∞                 | ∞                | ∞                    | ∞                |
| sobol+gp       | 10.0            | 20.0            | 50.0              | 5.0               | 50.0             | 10.0                 | 10.0             |

## Interpretation

- The 10·d rule of thumb (Loeppky 2009) is calibrated on analytical benchmarks.  Topologies with `frac_pass_at_10d` below 0.5 and median `n_star > 10` falsify the rule on physics-grounded engineering cases.
- `n_methods_never_pass` quantifies how many methods are *structurally* unable to reach the accuracy target on this topology at any tested budget.
- `n_failed_cells_per_topology` records failed or timed-out cells retained in the denominator audit.