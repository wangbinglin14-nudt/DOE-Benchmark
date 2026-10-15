"""Generate SI Table S15 and a linked overview from reviewed case descriptions.

Run from any directory. No simulations or result files are changed.
"""
from pathlib import Path
import inspect
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import doe_testbed as db

# Concise descriptions checked against current registry equations and outputs.
DESCRIPTIONS = {
    'aircraft_conceptual_8d': 'Conceptual aircraft correlations and the Breguet range equation; range, lift-to-drag ratio and empty-weight fraction.',
    'airfoil_stall_2d': 'Low-speed airfoil polar with stall and a subsonic correction; lift, drag and lift-to-drag ratio. The historical identifier denotes a two-dimensional airfoil, not two inputs.',
    'airfoil_transonic_5d': 'Airfoil polar with engineering transonic corrections; drag, lift and lift-to-drag ratio. This is a drag-divergence surrogate, not direct transonic CFD.',
    'battery_thermal_7d': 'Lumped transient battery heat balance; peak temperature, final temperature and maximum temperature rise.',
    'cantilever_beam_5d': 'Rectangular cantilever bending with shear-deflection correction; maximum bending stress, tip deflection and maximum shear stress.',
    'casting_solidification_5d': 'One-dimensional enthalpy-method solidification model; solidification time, mean front speed and centre peak temperature.',
    'dam_seepage_6d': 'Two-dimensional Darcy seepage with a simplified curtain and drain; total uplift, midpoint uplift pressure and uplift reduction factor.',
    'fin_heat_sink_4d': 'One-dimensional rectangular-fin heat equation; heat dissipation, fin efficiency and thermal resistance.',
    'fixed_bed_reactor_6d': 'Adiabatic reaction and heat-balance equations; exit conversion, peak temperature and exit temperature. The two temperatures coincide in this model.',
    'groundwater_contamination_6d': 'Advection--dispersion surrogate with adsorption and separable decay; relative concentration maximum, front position and arrival time.',
    'hetero_conduction_2d': 'Two-dimensional steady heterogeneous conduction; mid-line mean temperature, peak heat flux and a boundary-check temperature. Five inputs are declared.',
    'laminar_flame_speed_5d': 'Low-order premixed-flame correlations with an optional detailed backend; burning velocity, adiabatic flame temperature and expansion ratio. Backend scope is recorded in the evidence note.',
    'membrane_reactor_5d': 'Plug-flow reaction and membrane-removal equations; conversion, product concentration and permeation efficiency. The declared Peclet input is inactive in this version.',
    'oil_reservoir_9d': 'Two-dimensional Darcy flow through nine permeability blocks; boundary conductance, maximum pressure gradient and effective permeability.',
    'plate_with_hole_5d': 'Elliptical-hole stress concentration with a finite-width correction; peak edge stress, concentration factor and peak shear stress.',
    'pressure_vessel_4d': 'Thin-wall cylindrical membrane-stress model; strength safety factor, mass per unit length and equivalent stress.',
    'rocket_isp_4d': 'Low-order thermochemical correlations and ideal-nozzle equations; vacuum specific impulse, characteristic velocity and chamber temperature.',
    'rotor_bemt_6d': 'Blade-element/momentum hover model; thrust coefficient, power coefficient and figure of merit.',
    'soil_consolidation_5d': 'Terzaghi one-dimensional consolidation series; settlement, average consolidation degree and mid-depth excess pore pressure.',
    'solar_pv_yield_6d': 'Solar-resource and module-performance correlations; specific annual yield, capacity factor and mean operating temperature.',
    'supersonic_intake_5d': 'Two-ramp shock relations with a terminal normal shock; total-pressure recovery, downstream Mach number and static-pressure ratio.',
    'supersonic_nozzle_5d': 'One-dimensional isentropic and normal-shock nozzle relations; exit Mach number, exit temperature and total-pressure recovery.',
    'ten_bar_truss_10d': 'Planar truss stiffness solution with ten member areas; maximum horizontal displacement, maximum vertical displacement and total mass.',
    'thin_plate_buckling_5d': 'Thin-plate buckling relations with boundary-condition corrections; critical line load, critical stress and buckling coefficient.',
    'tuned_mass_damper_3d': 'Two-degree-of-freedom forced-vibration model; peak amplification, half-power bandwidth and peak damper-force ratio.',
    'waverider_4d': 'Modified-Newtonian hypersonic model; lift-to-drag ratio, lift coefficient and volumetric efficiency.',
    'wind_turbine_power_6d': 'Parameterised wind-power curve and Rayleigh wind distribution; maximum power coefficient, rated power and annual energy production.',
    'wing_vlm_6d': 'Prandtl lifting-line model; lift coefficient, induced drag coefficient and Oswald efficiency. The VLM identifier is historical.',
}
for n in [5, 10, 20]:
    DESCRIPTIONS[f'variable_beam_{n}d'] = f'Finite-element vibration model with {n} variable-height beam segments; first natural frequency, second natural frequency and total mass.'
for n in [4, 8, 16]:
    DESCRIPTIONS[f'multilayer_wall_{n}d'] = f'Series conduction through {n} layers; total thermal resistance, steady heat flux and mid-layer interface temperature.'

DOMAINS = {'structures':'Structures', 'heat_transfer':'Heat transfer', 'aerodynamics':'Aerodynamics', 'chemical_process':'Chemical engineering', 'geomechanics':'Geomechanics', 'multidisciplinary':'Multidisciplinary design'}

def tex(text):
    return text.replace('&', r'\&').replace('_', r'\_\allowbreak{}').replace('%', r'\%')

def main():
    metas = sorted(db.list_cases(), key=lambda m: (list(DOMAINS).index(m.domain), m.case_id))
    assert len(metas) == 34 and {m.case_id for m in metas} == set(DESCRIPTIONS)
    out = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.12}',
           r'\begin{longtable}{@{}p{0.29\linewidth}r p{0.63\linewidth}@{}}',
           r'\caption{\textbf{Supplementary Table S15.} Physical overview of all 34 simulation cases. Registry identifiers map to code and evidence notes; $d$ counts declared inputs. Scalable beam and wall variants are listed separately.}\label{tab:case_overview}\\',
           r'\toprule Case identifier & $d$ & Physical model and returned quantities \\ \midrule', r'\endfirsthead',
           r'\multicolumn{3}{l}{\textbf{Supplementary Table S15 (continued)}}\\',
           r'\toprule Case identifier & $d$ & Physical model and returned quantities \\ \midrule',r'\endhead',
           r'\midrule\multicolumn{3}{r}{Continued on next page}\\', r'\endfoot', r'\bottomrule', r'\endlastfoot']
    md = ['# Case overview', '', 'All 34 registered cases are simplified simulation models. Dimensions count declared inputs; adjacent evidence notes document implementation limitations and validation scope. Generated by `scripts/generate_case_overview.py`.', '', '| Domain / case | d | Model and outputs | Implementation | Evidence |', '|---|---:|---|---|---|']
    last = None
    for m in metas:
        if m.domain != last:
            out.append(r'\multicolumn{3}{@{}l}{\textbf{' + DOMAINS[m.domain] + r'}}\\*')
            last = m.domain
        out.append(tex(m.case_id) + f' & {m.dim_input} & ' + tex(DESCRIPTIONS[m.case_id]) + r' \\')
        source = Path(inspect.getfile(type(db.get_case(m.case_id)))).resolve()
        note = source.parent / (m.case_id + '.evidence.md')
        assert source.is_file() and note.is_file(), m.case_id
        md.append(f'| {DOMAINS[m.domain]} / `{m.case_id}` | {m.dim_input} | {DESCRIPTIONS[m.case_id]} | [Code](../{source.relative_to(ROOT).as_posix()}) | [Evidence](../{note.relative_to(ROOT).as_posix()}) |')
    out += [r'\end{longtable}',r'\endgroup','']
    (ROOT.parent/'manuscripts/tables/si_case_overview.tex').write_text('\n'.join(out),encoding='utf-8')
    (ROOT/'docs/CASE_OVERVIEW.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('Generated S15 and linked overview: 34 cases, 68 existing source/evidence links.')

if __name__ == '__main__':
    main()
