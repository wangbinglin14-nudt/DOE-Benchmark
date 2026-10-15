# Supplementary Table S3 · Case characterisation

_Auto-generated from `CaseMeta` and `docs/case_validation_status.csv` —_
_see `doe_testbed/cases/` for full implementations and_
_`docs/CASE_VALIDATION_SUMMARY.md` for evidence-tier definitions._

> Note: `d` is the number of input dimensions. A case-ID suffix such as `_2d` may denote the physical spatial dimension and is therefore not necessarily equal to `d`.

| Case | Domain | d | m | PDE | Governing eq. | Cost | Failure? | Evidence tier |
|---|---|---|---|---|---|---|---|---|
| `airfoil_stall_2d` | aerodynamics | 4 | 3 | integral | NeuralFoil baseline airfoil polar + explicit low-subsonic Pr | L0 | ✓ | `targeted_plausibility_only` |
| `waverider_4d` | aerodynamics | 4 | 3 | algebraic | Modified Newtonian impact theory: | L0 | — | `targeted_plausibility_only` |
| `airfoil_transonic_5d` | aerodynamics | 5 | 3 | algebraic | NeuralFoil baseline polar + explicit engineering transonic c | L0 | ✓ | `targeted_plausibility_only` |
| `supersonic_intake_5d` | aerodynamics | 5 | 3 | algebraic | Two-ramp inviscid external-compression intake surrogate: | L0 | ✓ | `targeted_plausibility_only` |
| `rotor_bemt_6d` | aerodynamics | 6 | 3 | ode_system | BEMT (coupled blade-element / momentum): | L0 | ✓ | `targeted_plausibility_only` |
| `wing_vlm_6d` | aerodynamics | 6 | 3 | integral | Prandtl Lifting-Line Theory (PLLT): | L0 | — | `targeted_plausibility_only` |
| `rocket_isp_4d` | chemical_process | 4 | 3 | algebraic | LOX/LH2 thermochemical surrogate + ideal nozzle equations: | L0 | — | `external_note_targeted_regression` |
| `laminar_flame_speed_5d` | chemical_process | 5 | 3 | ode_system | 1-D premixed planar flame (BVP): | L3 | ✓ | `external_note_targeted_regression` |
| `membrane_reactor_5d` | chemical_process | 5 | 3 | ode_system | 1-D tubular membrane reactor, plug-flow (Pe → ∞) limit: | L0 | — | `external_note_targeted_regression` |
| `supersonic_nozzle_5d` | chemical_process | 5 | 3 | algebraic | 1-D isentropic flow + normal-shock relations: | L0 | ✓ | `external_note_targeted_regression` |
| `fixed_bed_reactor_6d` | chemical_process | 6 | 3 | ode_system | 1-D adiabatic fixed-bed ODE system (residence time as indepe | L1 | ✓ | `external_note_targeted_regression` |
| `soil_consolidation_5d` | geomechanics | 5 | 3 | parabolic | Terzaghi 1-D consolidation PDE: | L0 | ✓ | `targeted_analytical_control` |
| `dam_seepage_6d` | geomechanics | 6 | 3 | elliptic | Darcy seepage (Laplace equation, heterogeneous): | L1 | — | `targeted_plausibility_only` |
| `groundwater_contamination_6d` | geomechanics | 6 | 3 | parabolic | 1-D ADE (advection–dispersion–reaction): | L0 | ✓ | `targeted_plausibility_only` |
| `oil_reservoir_9d` | geomechanics | 9 | 3 | elliptic | 2-D steady-state incompressible Darcy seepage: | L1 | — | `targeted_plausibility_only` |
| `fin_heat_sink_4d` | heat_transfer | 4 | 3 | ode_system | 1-D fin equation: d²θ/dx² − m²θ = 0, θ = T − T_∞ | L0 | ✓ | `external_note_targeted_regression` |
| `multilayer_wall_4d` | heat_transfer | 4 | 3 | algebraic | N-layer planar-wall steady-state conduction: | L0 | — | `external_note_analytic_control` |
| `casting_solidification_5d` | heat_transfer | 5 | 3 | parabolic | 1-D Stefan problem (enthalpy-method FDM): | L1 | — | `external_note_targeted_regression` |
| `hetero_conduction_2d` | heat_transfer | 5 | 3 | elliptic | ∇·(k(x,y)·∇T) = 0  [2-D steady state] | L1 | — | `external_note_targeted_regression` |
| `battery_thermal_7d` | heat_transfer | 7 | 3 | ode_system | Battery lumped-parameter thermal model (ODE): | L1 | ✓ | `external_note_targeted_regression` |
| `multilayer_wall_8d` | heat_transfer | 8 | 3 | algebraic | N-layer planar-wall steady-state conduction: | L0 | — | `external_note_analytic_control` |
| `multilayer_wall_16d` | heat_transfer | 16 | 3 | algebraic | N-layer planar-wall steady-state conduction: | L0 | — | `external_note_analytic_control` |
| `pressure_vessel_4d` | multidisciplinary | 4 | 3 | analytical | Thin-walled cylindrical pressure-vessel membrane-stress surr | L0 | ✓ | `targeted_analytical_control` |
| `solar_pv_yield_6d` | multidisciplinary | 6 | 3 | algebraic | PV specific annual yield model (IEC 61724): | L0 | — | `targeted_plausibility_only` |
| `wind_turbine_power_6d` | multidisciplinary | 6 | 3 | algebraic | Betz parameterised power model (not full BEM induction-facto | L0 | — | `targeted_plausibility_only` |
| `aircraft_conceptual_8d` | multidisciplinary | 8 | 3 | algebraic | Raymer-style conceptual-design surrogate: | L0 | — | `targeted_plausibility_only` |
| `tuned_mass_damper_3d` | structures | 3 | 3 | algebraic | 2-DOF forced vibration (harmonic excitation): | L0 | ✓ | `external_note_targeted_regression` |
| `cantilever_beam_5d` | structures | 5 | 3 | analytical | Euler–Bernoulli bending + Timoshenko shear-deflection correc | L0 | ✓ | `external_note_targeted_regression` |
| `plate_with_hole_5d` | structures | 5 | 3 | algebraic | Inglis solution (elliptical hole, infinite plate): | L0 | ✓ | `external_note_targeted_regression` |
| `thin_plate_buckling_5d` | structures | 5 | 3 | algebraic | Kirchhoff–Love plate buckling: D·∇⁴w = Nx·∂²w/∂x² | L0 | — | `external_note_targeted_regression` |
| `variable_beam_5d` | structures | 5 | 3 | algebraic | Euler–Bernoulli beam FEM: | L0 | — | `external_note_targeted_regression` |
| `ten_bar_truss_10d` | structures | 10 | 3 | algebraic | Truss FEM: K·u = F | L0 | ✓ | `external_note_targeted_regression` |
| `variable_beam_10d` | structures | 10 | 3 | algebraic | Euler–Bernoulli beam FEM: | L0 | — | `external_note_targeted_regression` |
| `variable_beam_20d` | structures | 20 | 3 | algebraic | Euler–Bernoulli beam FEM: | L0 | — | `external_note_targeted_regression` |
