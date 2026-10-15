# Supplementary Table S2 · Per-case Sobol' characterisation

_Auto-generated from `benchmark/data/sobol_corrected_full_current_source.json` —_
_SALib Saltelli sampling, N=1024, seed=20260831; see Methods §M1._

| Case | Domain | d | Topology | Top-effect var | $S_T$(top) | $\Sigma S_T$ | Interaction mass |
|---|---|---|---|---|---|---|---|
| `airfoil_stall_2d` | aerodynamics | 4 | smooth_unimodal, threshold_jump, heterogeneous | `alpha` | 0.962 | 1.112 | 0.198 |
| `waverider_4d` | aerodynamics | 4 | smooth_unimodal, strong_interaction | `theta_c` | 0.816 | 1.133 | 0.266 |
| `airfoil_transonic_5d` | aerodynamics | 5 | threshold_jump, heterogeneous, smooth_unimodal | `Ma` | 0.960 | 1.079 | 0.151 |
| `supersonic_intake_5d` | aerodynamics | 5 | smooth_unimodal, threshold_jump, strong_interaction, discontinuous | `Ma_in` | 0.957 | 1.012 | 0.016 |
| `rotor_bemt_6d` | aerodynamics | 6 | smooth_unimodal, strong_interaction, discontinuous | `theta_0` | 0.711 | 1.093 | 0.168 |
| `wing_vlm_6d` | aerodynamics | 6 | smooth_unimodal, strong_interaction | `alpha` | 0.736 | 1.025 | 0.041 |
| `rocket_isp_4d` | chemical_process | 4 | smooth_unimodal | `MR` | 0.755 | 1.006 | 0.008 |
| `laminar_flame_speed_5d` | chemical_process | 5 | smooth_unimodal, strong_interaction | `fuel_frac` | 0.488 | 1.101 | 0.206 |
| `membrane_reactor_5d` | chemical_process | 5 | smooth_monotone, strong_interaction | `Da` | 0.879 | 1.076 | 0.150 |
| `supersonic_nozzle_5d` | chemical_process | 5 | threshold_jump, discontinuous | `p_back_ratio` | 0.754 | 1.717 | 1.249 |
| `fixed_bed_reactor_6d` | chemical_process | 6 | smooth_unimodal, strong_interaction, heterogeneous | `Ea` | 0.802 | 1.409 | 0.596 |
| `soil_consolidation_5d` | geomechanics | 5 | smooth_monotone, strong_interaction, heterogeneous | `m_v` | 0.758 | 1.755 | 1.089 |
| `dam_seepage_6d` | geomechanics | 6 | smooth_monotone, strong_interaction, threshold_jump, discontinuous | `H1` | 0.564 | 1.115 | 0.250 |
| `groundwater_contamination_6d` | geomechanics | 6 | smooth_monotone, strong_interaction, sparse_effects, heterogeneous, threshold_jump, discontinuous | `t_half` | 0.680 | 1.108 | 0.197 |
| `oil_reservoir_9d` | geomechanics | 9 | smooth_monotone, strong_interaction | `K22` | 0.301 | 1.730 | 1.270 |
| `fin_heat_sink_4d` | heat_transfer | 4 | smooth_monotone, strong_interaction | `h_coef` | 0.815 | 1.108 | 0.190 |
| `multilayer_wall_4d` | heat_transfer | 4 | smooth_monotone | `k4` | 0.250 | 0.999 | 0.000 |
| `casting_solidification_5d` | heat_transfer | 5 | smooth_unimodal, strong_interaction, heterogeneous | `L_half` | 0.716 | 1.118 | 0.229 |
| `hetero_conduction_2d` | heat_transfer | 5 | smooth_monotone, threshold_jump, heterogeneous | `T_hot` | 0.800 | 1.000 | 0.000 |
| `battery_thermal_7d` | heat_transfer | 7 | smooth_unimodal, strong_interaction | `T_cool` | 0.899 | 1.061 | 0.125 |
| `multilayer_wall_8d` | heat_transfer | 8 | smooth_monotone | `k3` | 0.127 | 1.011 | 0.024 |
| `multilayer_wall_16d` | heat_transfer | 16 | smooth_monotone | `k14` | 0.064 | 1.011 | 0.029 |
| `pressure_vessel_4d` | multidisciplinary | 4 | smooth_monotone, strong_interaction | `P` | 0.779 | 1.368 | 0.709 |
| `solar_pv_yield_6d` | multidisciplinary | 6 | smooth_unimodal, strong_interaction | `GHI_annual` | 0.788 | 1.017 | 0.031 |
| `wind_turbine_power_6d` | multidisciplinary | 6 | smooth_monotone, strong_interaction | `lam_opt` | 0.828 | 1.007 | 0.012 |
| `aircraft_conceptual_8d` | multidisciplinary | 8 | smooth_monotone, strong_interaction | `W_frac` | 0.645 | 1.025 | 0.059 |
| `tuned_mass_damper_3d` | structures | 3 | smooth_unimodal, strong_interaction, heterogeneous | `mu` | 0.423 | 1.191 | 0.337 |
| `cantilever_beam_5d` | structures | 5 | smooth_monotone, strong_interaction | `h` | 0.740 | 1.583 | 0.961 |
| `plate_with_hole_5d` | structures | 5 | smooth_monotone, strong_interaction | `sigma_x` | 0.466 | 1.230 | 0.473 |
| `thin_plate_buckling_5d` | structures | 5 | threshold_jump, smooth_unimodal | `h_over_b` | 0.834 | 1.159 | 0.329 |
| `variable_beam_5d` | structures | 5 | smooth_monotone, sparse_effects | `h1` | 0.566 | 1.237 | 0.467 |
| `ten_bar_truss_10d` | structures | 10 | smooth_monotone, sparse_effects | `A2` | 0.318 | 1.012 | 0.006 |
| `variable_beam_10d` | structures | 10 | smooth_monotone, sparse_effects | `h1` | 0.376 | 1.279 | 0.530 |
| `variable_beam_20d` | structures | 20 | smooth_monotone, sparse_effects | `h2` | 0.253 | 1.345 | 0.570 |
