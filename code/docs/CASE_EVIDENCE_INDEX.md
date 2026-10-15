# Case evidence index

Updated 2026-09-17. Each of the 34 registered cases has an evidence note
beside its implementation. Notes distinguish equations and adaptations,
parameter choices, inspected source passages, actual tests and unresolved
limitations. They do not certify experimental validity. All computational
source bytes still match the frozen run; no historical fingerprint was repinned.

## Verification scope

Thirty targeted physics-test modules covering all 34 cases passed in three
groups during this documentation audit. The structures/geomechanics group
contains 163 tests across ten modules; the other two groups each ran ten
modules. Shared variable-beam and multilayer-wall tests must not be counted
again for each dimensional variant. Cross-validation scripts were inspected
for assertion scope but were not rerun as a complete suite in this pass.

A passing regression, trend or analytic-limit test verifies its stated
computational assertion, not measured real-world error. The existing risk
and evidence-tier labels remain unchanged. The source release includes the
tests and all source links needed to assess these claims; internal audit
paths mentioned in some notes are supplementary history, not prerequisites.

## Per-case code and evidence

| Domain | Case | Input dimension | Implementation | Evidence note |
|---|---|---:|---|---|
| aerodynamics | `airfoil_stall_2d` | 4 | [code](../doe_testbed/cases/aerodynamics/airfoil_stall_2d.py) | [evidence](../doe_testbed/cases/aerodynamics/airfoil_stall_2d.evidence.md) |
| aerodynamics | `airfoil_transonic_5d` | 5 | [code](../doe_testbed/cases/aerodynamics/airfoil_transonic_5d.py) | [evidence](../doe_testbed/cases/aerodynamics/airfoil_transonic_5d.evidence.md) |
| aerodynamics | `rotor_bemt_6d` | 6 | [code](../doe_testbed/cases/aerodynamics/rotor_bemt_6d.py) | [evidence](../doe_testbed/cases/aerodynamics/rotor_bemt_6d.evidence.md) |
| aerodynamics | `supersonic_intake_5d` | 5 | [code](../doe_testbed/cases/aerodynamics/supersonic_intake_5d.py) | [evidence](../doe_testbed/cases/aerodynamics/supersonic_intake_5d.evidence.md) |
| aerodynamics | `waverider_4d` | 4 | [code](../doe_testbed/cases/aerodynamics/waverider_4d.py) | [evidence](../doe_testbed/cases/aerodynamics/waverider_4d.evidence.md) |
| aerodynamics | `wing_vlm_6d` | 6 | [code](../doe_testbed/cases/aerodynamics/wing_vlm_6d.py) | [evidence](../doe_testbed/cases/aerodynamics/wing_vlm_6d.evidence.md) |
| chemical_process | `fixed_bed_reactor_6d` | 6 | [code](../doe_testbed/cases/chemical/fixed_bed_reactor_6d.py) | [evidence](../doe_testbed/cases/chemical/fixed_bed_reactor_6d.evidence.md) |
| chemical_process | `laminar_flame_speed_5d` | 5 | [code](../doe_testbed/cases/chemical/laminar_flame_speed_5d.py) | [evidence](../doe_testbed/cases/chemical/laminar_flame_speed_5d.evidence.md) |
| chemical_process | `membrane_reactor_5d` | 5 | [code](../doe_testbed/cases/chemical/membrane_reactor_5d.py) | [evidence](../doe_testbed/cases/chemical/membrane_reactor_5d.evidence.md) |
| chemical_process | `rocket_isp_4d` | 4 | [code](../doe_testbed/cases/chemical/rocket_isp_4d.py) | [evidence](../doe_testbed/cases/chemical/rocket_isp_4d.evidence.md) |
| chemical_process | `supersonic_nozzle_5d` | 5 | [code](../doe_testbed/cases/chemical/supersonic_nozzle_5d.py) | [evidence](../doe_testbed/cases/chemical/supersonic_nozzle_5d.evidence.md) |
| geomechanics | `dam_seepage_6d` | 6 | [code](../doe_testbed/cases/geomechanics/dam_seepage_6d.py) | [evidence](../doe_testbed/cases/geomechanics/dam_seepage_6d.evidence.md) |
| geomechanics | `groundwater_contamination_6d` | 6 | [code](../doe_testbed/cases/geomechanics/groundwater_contamination_6d.py) | [evidence](../doe_testbed/cases/geomechanics/groundwater_contamination_6d.evidence.md) |
| geomechanics | `oil_reservoir_9d` | 9 | [code](../doe_testbed/cases/geomechanics/oil_reservoir_9d.py) | [evidence](../doe_testbed/cases/geomechanics/oil_reservoir_9d.evidence.md) |
| geomechanics | `soil_consolidation_5d` | 5 | [code](../doe_testbed/cases/geomechanics/soil_consolidation_5d.py) | [evidence](../doe_testbed/cases/geomechanics/soil_consolidation_5d.evidence.md) |
| heat_transfer | `battery_thermal_7d` | 7 | [code](../doe_testbed/cases/heat_transfer/battery_thermal_7d.py) | [evidence](../doe_testbed/cases/heat_transfer/battery_thermal_7d.evidence.md) |
| heat_transfer | `casting_solidification_5d` | 5 | [code](../doe_testbed/cases/heat_transfer/casting_solidification_5d.py) | [evidence](../doe_testbed/cases/heat_transfer/casting_solidification_5d.evidence.md) |
| heat_transfer | `fin_heat_sink_4d` | 4 | [code](../doe_testbed/cases/heat_transfer/fin_heat_sink_4d.py) | [evidence](../doe_testbed/cases/heat_transfer/fin_heat_sink_4d.evidence.md) |
| heat_transfer | `hetero_conduction_2d` | 5 | [code](../doe_testbed/cases/heat_transfer/hetero_conduction_2d.py) | [evidence](../doe_testbed/cases/heat_transfer/hetero_conduction_2d.evidence.md) |
| heat_transfer | `multilayer_wall_16d` | 16 | [code](../doe_testbed/cases/heat_transfer/multilayer_wall_nd.py) | [evidence](../doe_testbed/cases/heat_transfer/multilayer_wall_16d.evidence.md) |
| heat_transfer | `multilayer_wall_4d` | 4 | [code](../doe_testbed/cases/heat_transfer/multilayer_wall_nd.py) | [evidence](../doe_testbed/cases/heat_transfer/multilayer_wall_4d.evidence.md) |
| heat_transfer | `multilayer_wall_8d` | 8 | [code](../doe_testbed/cases/heat_transfer/multilayer_wall_nd.py) | [evidence](../doe_testbed/cases/heat_transfer/multilayer_wall_8d.evidence.md) |
| multidisciplinary | `aircraft_conceptual_8d` | 8 | [code](../doe_testbed/cases/multidisciplinary/aircraft_conceptual_8d.py) | [evidence](../doe_testbed/cases/multidisciplinary/aircraft_conceptual_8d.evidence.md) |
| multidisciplinary | `pressure_vessel_4d` | 4 | [code](../doe_testbed/cases/multidisciplinary/pressure_vessel_4d.py) | [evidence](../doe_testbed/cases/multidisciplinary/pressure_vessel_4d.evidence.md) |
| multidisciplinary | `solar_pv_yield_6d` | 6 | [code](../doe_testbed/cases/multidisciplinary/solar_pv_yield_6d.py) | [evidence](../doe_testbed/cases/multidisciplinary/solar_pv_yield_6d.evidence.md) |
| multidisciplinary | `wind_turbine_power_6d` | 6 | [code](../doe_testbed/cases/multidisciplinary/wind_turbine_power_6d.py) | [evidence](../doe_testbed/cases/multidisciplinary/wind_turbine_power_6d.evidence.md) |
| structures | `cantilever_beam_5d` | 5 | [code](../doe_testbed/cases/structures/cantilever_beam_5d.py) | [evidence](../doe_testbed/cases/structures/cantilever_beam_5d.evidence.md) |
| structures | `plate_with_hole_5d` | 5 | [code](../doe_testbed/cases/structures/plate_with_hole_5d.py) | [evidence](../doe_testbed/cases/structures/plate_with_hole_5d.evidence.md) |
| structures | `ten_bar_truss_10d` | 10 | [code](../doe_testbed/cases/structures/ten_bar_truss_10d.py) | [evidence](../doe_testbed/cases/structures/ten_bar_truss_10d.evidence.md) |
| structures | `thin_plate_buckling_5d` | 5 | [code](../doe_testbed/cases/structures/thin_plate_buckling_5d.py) | [evidence](../doe_testbed/cases/structures/thin_plate_buckling_5d.evidence.md) |
| structures | `tuned_mass_damper_3d` | 3 | [code](../doe_testbed/cases/structures/tuned_mass_damper_3d.py) | [evidence](../doe_testbed/cases/structures/tuned_mass_damper_3d.evidence.md) |
| structures | `variable_beam_10d` | 10 | [code](../doe_testbed/cases/structures/variable_beam_nd.py) | [evidence](../doe_testbed/cases/structures/variable_beam_10d.evidence.md) |
| structures | `variable_beam_20d` | 20 | [code](../doe_testbed/cases/structures/variable_beam_nd.py) | [evidence](../doe_testbed/cases/structures/variable_beam_20d.evidence.md) |
| structures | `variable_beam_5d` | 5 | [code](../doe_testbed/cases/structures/variable_beam_nd.py) | [evidence](../doe_testbed/cases/structures/variable_beam_5d.evidence.md) |

The machine-readable companion is [case_evidence_manifest.json](case_evidence_manifest.json).
For extension and data/code assembly, see [RESEARCH_REUSE.md](RESEARCH_REUSE.md).

中文说明：这里的原始来源证据指能够追溯到原文公式、表格、段落或数据记录的依据。
仅核实论文/书籍存在，不代表其中每个参数已验证。各案例未取得的原文依据和模型
限制均在对应说明中明确保留；本次没有用注释修改去替换产生历史结果的计算代码。
