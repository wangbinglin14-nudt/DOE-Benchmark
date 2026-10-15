"""Physics regression tests for ``aircraft_conceptual_8d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.multidisciplinary.aircraft_conceptual_8d import (
    AircraftConceptual8D,
)


def _domain_corners(case: AircraftConceptual8D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_declared_output_ranges_cover_domain_corners():
    case = AircraftConceptual8D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_validation_material_declares_low_order_sizing_surrogate_scope():
    case = AircraftConceptual8D()
    validation_text = Path(
        "tests/_cross_validation/aircraft_conceptual_8d.py"
    ).read_text(encoding="utf-8")
    normalized_validation = " ".join(validation_text.split())
    source_text = Path(
        "doe_testbed/cases/multidisciplinary/aircraft_conceptual_8d.py"
    ).read_text(encoding="utf-8")
    normalized_source = " ".join(source_text.split())
    normalized_source_lower = normalized_source.lower()

    assert "low-order conceptual-design surrogate" in validation_text
    assert "not a calibrated full Raymer sizing" in validation_text
    assert "aircraft-design validation case" in validation_text
    assert "not a full aircraft sizing workflow" in normalized_source
    assert "not a full aircraft sizing workflow or mission/trajectory fuel-burn analysis" in normalized_source
    for omitted in [
        "payload",
        "take-off weight",
        "wing loading",
        "thrust loading",
        "constraint diagrams",
        "take-off/landing field length",
        "engine lapse",
        "wind/weather",
        "cruise lift",
        "dynamic pressure",
        "compressibility drag rise",
        "wave drag",
        "drag-divergence mach",
        "buffet",
        "supercritical-airfoil",
        "stability/control",
        "certification",
        "cost",
        "noise",
        "emissions",
        "flops/gasp/aviary/openmdao",
        "phase_info",
        "trajectory",
        "external-subsystem",
        "subsystem-builder",
        "openvsp",
        "vspaero",
        "degengeom",
        "potential-flow",
        "empirical geometry",
        "conventional tube-and-wing",
        "novel configurations",
        "medium- and high-fidelity",
        "npss",
        "pycycle",
        "openaerostruct",
        "anopp",
        "electrical modeling",
    ]:
        assert omitted in normalized_source_lower
    assert "not a calibrated full aircraft sizing code" in case.meta.solver_details
    assert "constraint-analysis workflow" in case.meta.solver_details
    assert "payload/MTOW closure" in case.meta.solver_details
    assert "mission/trajectory fuel-burn model" in case.meta.solver_details
    assert "out-of-contract range_km/L_D_max/We_frac responses are rejected" in case.meta.solver_details
    assert hasattr(case, "_validate_physical_inputs")
    assert "named _validate_physical_inputs private physical-input evaluation rejects" in case.meta.solver_details
    assert "before computing the Breguet/Raymer-style response" in case.meta.solver_details
    assert "Raymer Table 6.1" not in source_text
    assert "verbatim" in source_text
    assert "Raymer/Roskam sizing table" in source_text
    references = "\n".join(case.meta.physics_references)
    for public_anchor in [
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-constant-velocity/",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-summary/",
        "https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node98.html",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/",
        "https://www.grc.nasa.gov/www/BGH/mach.html",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/what-is-drag/",
        "https://software.nasa.gov/software/LEW-20538-1",
        "https://software.nasa.gov/software/LAR-18934-1",
        "https://ntrs.nasa.gov/citations/20170005851",
        "https://ntrs.nasa.gov/citations/19810010562",
        "https://www.nasa.gov/aeronautics/aviary-software-overview/",
        "https://openmdao.github.io/Aviary/",
        "https://ntrs.nasa.gov/citations/20230015940",
        "https://openmdao.github.io/Aviary/user_guide/using_external_subsystems.html",
        "https://openmdao.github.io/Aviary/theory_guide/intro.html",
        "https://openmdao.github.io/Aviary/user_guide/features_and_functionalities.html",
        "https://openmdao.github.io/Aviary/misc_resources/FAQ.html",
        "https://openmdao.github.io/Aviary/misc_resources/comparison_to_flops.html",
        "https://www.nasa.gov/reference/openvsp-vspaero-basics/",
    ]:
        assert public_anchor in references
    assert "constant-cruise" in normalized_source
    assert "climb, descent, maneuver, wind/weather, alternate, or reserve mission segments" in normalized_source
    assert "Wing reference area and cruise-altitude density are retained as DOE stress-test inputs but are not used" in normalized_source
    assert "Mach number enters only through the assumed cruise speed" in normalized_source
    assert "R is separable in cruise speed, inverse SFC, and the fuel-log" in validation_text
    assert "R ∝ Ma_cruise · SFC^{-1} · ln(1/(1 - 0.95 W_frac))" in validation_text
    assert "no cruise CL/dynamic-pressure trim closure" in case.meta.solver_details
    assert "no wind/weather mission correction" in case.meta.solver_details
    assert "no compressibility drag rise" in case.meta.solver_details
    assert "drag-divergence Mach number" in case.meta.solver_details
    assert "altitude/Mach-dependent engine lapse" in case.meta.solver_details
    assert "FLOPS/GASP/Aviary/OpenMDAO aircraft synthesis" in case.meta.solver_details
    assert "mission analysis, or optimisation" in case.meta.solver_details
    assert "FLOPS weights/aerodynamics/propulsion-data/mission/takeoff-landing/program-control module workflow" in case.meta.solver_details
    assert "GASP integrated preliminary-design/parametric-study/performance/economics/configuration-trade workflow" in case.meta.solver_details
    assert "no Aviary phase_info" in case.meta.solver_details
    assert "trajectory, mission-phase, external-subsystem" in case.meta.solver_details
    assert "subsystem-builder workflow" in case.meta.solver_details
    assert "analytic-gradient MDAO" in case.meta.solver_details
    assert "tightly coupled aircraft/subsystem design" in case.meta.solver_details
    assert "modular subsystem integration across trajectories" in case.meta.solver_details
    assert "external-tool coupling" in case.meta.solver_details
    assert "empirical Aviary geometry/mass/aerodynamic/propulsion/mission core-subsystem workflow" in case.meta.solver_details
    assert "Aviary five-subsystem model and mission-phase setup workflow" in case.meta.solver_details
    assert "conventional tube-and-wing empirical-equation workflow" in case.meta.solver_details
    assert "novel-configuration external-subsystem workflow" in case.meta.solver_details
    assert "NPSS" in case.meta.solver_details
    assert "pyCycle" in case.meta.solver_details
    assert "OpenAeroStruct" in case.meta.solver_details
    assert "VSPAero" in case.meta.solver_details
    assert "ANOPP" in case.meta.solver_details
    assert "electrical modeling" in case.meta.solver_details
    assert "Aviary comparison-to-FLOPS workflow" in case.meta.solver_details
    assert "empirical/tabular aerodynamics" in case.meta.solver_details
    assert "engine decks" in case.meta.solver_details
    assert "reserve/off-design missions" in case.meta.solver_details
    assert "OpenMDAO optimization" in case.meta.solver_details
    assert "OpenVSP/VSPAERO-style parametric geometry" in case.meta.solver_details
    assert "DegenGeom" in case.meta.solver_details
    assert "potential-flow" in case.meta.solver_details
    assert "pressure/wake-solution review" in case.meta.solver_details
    assert "custom command-line input setup" in case.meta.solver_details
    assert "smooth_monotone" in case.meta.response_topology
    assert "smooth_unimodal" not in case.meta.response_topology
    assert "strong_interaction" in case.meta.response_topology
    assert set(case.meta.important_variables) == {
        "AR",
        "W_frac",
        "SFC",
        "Ma_cruise",
        "tc",
        "lam",
    }
    assert case.meta.effective_dim == len(case.meta.important_variables)
    assert "V/(g·SFC)" in source_text
    assert "V_cruise/(g·SFC)" in case.meta.governing_equation
    assert "R = (V/SFC)" not in source_text
    assert "NASA/MIT source anchors support the Breguet/range/Mach/drag/induced-drag" in validation_text
    assert "formula context only" in validation_text
    assert "not external validation of the Raymer-style empty-weight power law" in normalized_validation
    assert "They are not external validation data for the Raymer-style empty-weight power" in source_text
    assert "FLOPS/GASP/Aviary/OpenMDAO and OpenVSP/VSPAERO references are boundary evidence" in normalized_source
    assert "DegenGeom" in normalized_source
    assert "potential-flow" in normalized_source
    assert "pressure/wake-solution review" in normalized_source
    assert "custom command-line/input-setup workflows" in normalized_source
    assert "Aviary phase_info, trajectory, and external subsystem-builder workflows" in normalized_source
    assert "multidisciplinary design/analysis/optimization with analytic gradients" in normalized_source
    assert "tightly coupled aircraft/subsystem design" in normalized_source
    assert "NASA NTRS, Aviary MDAO/analytic-gradient paper" in references
    assert "Flight Optimization System (FLOPS) Software v.9" in references
    assert "weights, aerodynamics, propulsion data" in references
    assert "mission performance, takeoff and landing" in references
    assert "The Flight Optimization System Weights Estimation Method" in references
    assert "not a source for calibrating the benchmark's simplified Raymer-style empty-weight power law" in references
    assert "GASP - General Aviation Synthesis Program" in references
    assert "preliminary-design technical disciplines" in references
    assert "configuration trade studies" in references
    assert "not validation data for this compact Breguet/Raymer-style algebraic surrogate" in references
    assert "analytic gradients" in references
    assert "tightly coupled aircraft/subsystem design" in references
    assert "external-subsystems documentation" in references
    assert "modular OpenMDAO subsystems" in references
    assert "across aircraft trajectories" in references
    assert "empirical geometry, mass" in references
    assert "aerodynamic, propulsion, and mission models" in references
    assert "features and functionalities" in references.lower()
    assert "geometry, mass, aerodynamics, propulsion, and mission subsystems" in references
    assert "user-selected phase options" in references
    assert "coupled subsystem/mission-phase machinery" in references
    assert "conventional tube-and-wing aircraft" in references
    assert "external subsystems for novel configurations" in references
    assert "NPSS" in references
    assert "pyCycle" in references
    assert "OpenAeroStruct" in references
    assert "VSPAero" in references
    assert "ANOPP" in references
    assert "electrical modeling" in references
    assert "medium- and high-fidelity analysis/optimization" in references
    assert "OpenMDAO Aviary comparison to FLOPS" in references
    assert "FLOPS/GASP-derived workflows" in references
    assert "empirical and tabular aerodynamics" in references
    assert "external aerodynamic models" in references
    assert "propulsion engine decks" in references
    assert "mission equations of motion" in references
    assert "reserve/off-design missions" in references
    assert "optimization/analytic-derivative machinery" in references
    assert "creates virtual aircraft models from user-provided inputs" in references
    assert "link with other codes/tools" in references
    assert "OpenMDAO project site" in references
    assert "built on top of OpenMDAO" in references
    assert "mission/vehicle coupling workflows" in references
    assert "pressure/wake solution review" in references
    assert "custom-input/command-line workflows" in references
    for public_anchor in [
        "Range - Constant Velocity",
        "Range Summary",
        "Breguet range equation",
        "Induced Drag Coefficient",
        "Aviary",
        "OpenVSP",
    ]:
        assert public_anchor in validation_text
        assert public_anchor in source_text
    assert "Aviary functionality overview" in validation_text
    assert "Aviary FAQ" in validation_text
    assert "empirical geometry, mass, aerodynamic, propulsion, and mission" in normalized_validation
    assert "conventional tube-and-wing empirical-equation" in normalized_validation
    assert "novel-configuration external-subsystem" in normalized_validation
    assert "medium- and high-fidelity analysis/optimization" in normalized_validation
    assert "minimum geometry, mass, aerodynamics, propulsion, and mission subsystem coverage" in normalized_validation
    assert "user-selected mission-phase setup" in normalized_validation
    assert "NPSS, pyCycle, OpenAeroStruct, VSPAero, ANOPP, or electrical modeling" in normalized_validation
    assert "GASP theoretical-development report" in normalized_validation
    assert "Aviary comparison-to-FLOPS page" in normalized_validation
    assert "integrated technical disciplines" in normalized_validation
    assert "richer mission/performance modules" in normalized_validation
    assert "Target accuracy" not in validation_text
    assert "externally validated" not in validation_text


def test_reviewer_markdown_note_keeps_aircraft_conceptual_p2_scope_boundary():
    note_text = Path("tests/_cross_validation/aircraft_conceptual_8d.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(note_text.split())

    required_phrases = [
        "P2 plausibility-only",
        "No response-surface change",
        "Low-order Breguet range",
        "Raymer-style empty-weight surrogate",
        "not externally validated aircraft sizing",
        "not a full Raymer/Roskam implementation",
        "not FLOPS/GASP/Aviary/OpenMDAO validated",
        "not OpenVSP/VSPAERO validated",
        "not mission fuel-burn validated",
        "not transonic-performance validated",
        "FLOPS-style weights, aerodynamics, propulsion-data",
        "program-control module workflows",
        "GASP-style integrated technical-discipline synthesis",
        "FLOPS/GASP-derived geometry, mass",
        "empirical/tabular aerodynamics",
        "engine-deck",
        "reserve/off-design mission",
        "OpenMDAO optimization machinery",
        "Aviary's geometry, mass, aerodynamics, propulsion",
        "user-selected phase-options mission setup",
        "payload/MTOW closure",
        "wing-loading and thrust-loading constraint analysis",
        "drag-divergence Mach number",
        "Breguet separability",
        "range scales linearly with `Ma_cruise`",
        "inversely with mass `SFC`",
        "fuel-log term `ln(1/(1 - 0.95*W_frac))`",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-constant-velocity/",
        "https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node98.html",
        "https://openmdao.github.io/Aviary/",
        "https://ntrs.nasa.gov/citations/20230015940",
        "https://openmdao.github.io/Aviary/user_guide_unreviewed/features_and_functionalities.html",
        "https://ntrs.nasa.gov/citations/19810010562",
        "https://openmdao.github.io/Aviary/theory_guide/comparison_to_flops.html",
        "https://www.nasa.gov/reference/openvsp-vspaero-basics/",
    ]
    for phrase in required_phrases:
        assert phrase in normalized

    for forbidden_phrase in [
        "Evidence tier | P1",
        "Evidence tier | P0",
        "validated aircraft sizing, mission, or MDAO workflow",
    ]:
        assert forbidden_phrase not in normalized


def test_readme_and_web_audit_keep_aircraft_scope_honest():
    readme = Path("README.md").read_text(encoding="utf-8")
    readme_zh = Path("README_zh.md").read_text(encoding="utf-8")
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit_text.split())
    row = next(
        line for line in audit_text.splitlines()
        if line.startswith("| `aircraft_conceptual_8d` |")
        and "constant-cruise conceptual-design stress test" in line
    )

    assert "Breguet + Raymer-style weight surrogate" in readme
    assert "Breguet + MTOW equations" not in readme
    assert "Breguet + Raymer 风格重量代理" in readme_zh
    assert "Breguet + MTOW 方程" not in readme_zh
    assert "Evidence tier remains P2" in row
    assert "constant-cruise conceptual-design stress test" in row
    assert "not an external numerical validation" in row
    assert "full Raymer aircraft-sizing workflow" in row
    assert "payload/MTOW closure" in row
    assert "transonic performance" in row
    assert "full mission fuel-burn" in row
    assert "OpenMDAO" in audit_text
    assert "https://openmdao.github.io/Aviary/" in audit_text
    assert "DegenGeom" in audit_text
    assert "potential-flow" in audit_text
    assert "FLOPS/GASP/Aviary/OpenMDAO/OpenVSP/VSPAERO validation case" in normalized_audit
    assert "README solver label was corrected" in row
    assert "Full sweep is required" in row
    assert "Oswald" in row
    assert "2026-06-08 Aircraft Conceptual Official-Source Re-Check" in audit_text
    assert "official NASA pages" in audit_text
    assert "constant-velocity range page" in audit_text
    assert "real aircraft range is more complex" in normalized_audit
    assert "descends from FLOPS and GASP" in audit_text
    assert "potential-flow aerodynamics tool using OpenVSP geometries" in normalized_audit
    assert "DegenGeom thin-surface representation" in normalized_audit
    assert "user-provided inputs" in normalized_audit
    assert "link with other codes/tools" in normalized_audit
    assert "OpenMDAO project site" in normalized_audit
    assert "built on top of OpenMDAO" in normalized_audit
    assert "mission/vehicle coupling and optimization workflows" in normalized_audit
    assert "phase_info / trajectory / mission-phase workflows" in normalized_audit
    assert "external-subsystem/subsystem-builder integration" in normalized_audit
    assert "Aviary MDAO/source-boundary note" in audit_text
    assert "aircraft design/analysis/optimization code built on OpenMDAO" in normalized_audit
    assert "analytic gradients" in normalized_audit
    assert "tightly coupled aircraft/subsystem design" in normalized_audit
    assert "modular OpenMDAO subsystems across trajectories" in normalized_audit
    assert "not validation data for the benchmark's constant-cruise" in normalized_audit
    assert "OpenMDAO MDAO validated" in normalized_audit
    assert "analytic-gradient optimization validated" in normalized_audit
    assert "external-subsystem workflow validated" in normalized_audit
    assert "pressure/wake-solution review" in normalized_audit
    assert "custom-input/command-line workflows" in normalized_audit
    assert "They do not upgrade the evidence tier" in normalized_audit
    assert "No numerical response resweep is needed solely for this official-source re-check" in normalized_audit
    assert (
        "Additional `aircraft_conceptual_8d` 2026-06-08 official-source re-check:"
        in audit_text
    )
    assert "Evidence tier remains P2 plausibility-only" in normalized_audit
    assert "not validation data for this compact Breguet/parabolic-drag-polar" in normalized_audit
    assert "Do not describe this case as externally validated aircraft sizing" in normalized_audit
    assert "payload/MTOW closure" in normalized_audit
    assert "legal-domain finite responses are unchanged" in normalized_audit
    assert "2026-06-08 official-source boundary tightening" in audit_text
    assert "2026-06-08 OpenMDAO-Aviary boundary note" in audit_text
    assert "2026-06-08 Aviary functionality/FAQ boundary note" in audit_text
    assert "2026-06-09 NASA-FLOPS boundary note" in audit_text
    assert "FLOPS v9" in normalized_audit
    assert "propulsion data scaling/interpolation" in normalized_audit
    assert "takeoff and landing" in normalized_audit
    assert "FLOPS-equivalent" in normalized_audit
    assert "FLOPS-weight-method calibrated" in normalized_audit
    assert "aircraft-synthesis validated" in normalized_audit
    assert "2026-06-09 NASA-GASP/Aviary-comparison" in audit_text
    assert "boundary note: the NASA NTRS GASP theoretical-development report" in normalized_audit
    assert "https://ntrs.nasa.gov/citations/19810010562" in audit_text
    assert "https://openmdao.github.io/Aviary/theory_guide/comparison_to_flops.html" in audit_text
    assert "integrated preliminary-design technical disciplines" in normalized_audit
    assert "configuration trade studies" in normalized_audit
    assert "empirical and tabular aerodynamics" in normalized_audit
    assert "external aerodynamic models" in normalized_audit
    assert "propulsion engine decks" in normalized_audit
    assert "mission equations of motion" in normalized_audit
    assert "reserve/off-design missions" in normalized_audit
    assert "OpenMDAO optimization/analytic-derivative machinery" in normalized_audit
    assert "GASP-equivalent" in normalized_audit
    assert "empirical/tabular-aerodynamics validated" in normalized_audit
    assert "engine-deck validated" in normalized_audit
    assert "equations-of-motion validated" in normalized_audit
    assert "empirical geometry, mass, aerodynamic, propulsion, and mission" in normalized_audit
    assert "conventional tube-and-wing empirical equations" in normalized_audit
    assert "external subsystems for novel configurations" in normalized_audit
    assert "NPSS, pyCycle, OpenAeroStruct, VSPAero, ANOPP, and electrical modeling" in normalized_audit
    assert "medium/high-fidelity analysis" in normalized_audit
    assert "No numerical response resweep is needed solely for this Aviary functionality/FAQ note" in normalized_audit
    assert "2026-06-09 Aviary features/functionality boundary note" in audit_text
    assert "five-subsystem minimum" in normalized_audit
    assert "mission phase options" in normalized_audit
    assert "coupled Aviary subsystem/mission workflow" in normalized_audit
    assert "No numerical response resweep is needed solely for this Aviary features/functionality note" in normalized_audit
    assert "2026-06-09 Breguet separability-control note" in audit_text
    assert "range scales linearly with `Ma_cruise`" in normalized_audit
    assert "inversely with mass `SFC`" in normalized_audit
    assert "fuel-log term `ln(1/(1 - 0.95*W_frac))`" in normalized_audit
    assert "No numerical response resweep is needed solely for this Breguet separability-control note" in normalized_audit
    assert "2026-06-09 current-page spot check" in audit_text
    assert "NASA Glenn range, NASA Glenn induced-drag, OpenMDAO Aviary" in normalized_audit
    assert "NASA OpenVSP VSPAERO, NASA FLOPS, NASA GASP" in normalized_audit
    assert "current public boundary anchors" in normalized_audit
    assert "compact constant-cruise Breguet/parabolic-drag-polar/Raymer-style trend surrogate" in normalized_audit
    assert "richer mission, synthesis, optimization, geometry" in normalized_audit
    assert "propulsion-deck" in normalized_audit
    assert "subsystem-coupled workflows" in normalized_audit
    assert "This current-page spot check does not add validation data" in normalized_audit
    assert "externally validated aircraft sizing" in normalized_audit
    assert "FLOPS/GASP/Aviary/OpenMDAO-equivalent" in normalized_audit
    assert "OpenVSP/VSPAERO-equivalent" in normalized_audit
    assert "mission-fuel-burn" in normalized_audit
    assert "No numerical response resweep is needed solely for this current-page spot check" in normalized_audit


def test_interaction_metadata_covers_breguet_product_structure():
    case = AircraftConceptual8D()

    expected_pairs = {
        ("AR", "lam"),
        ("AR", "tc"),
        ("AR", "SFC"),
        ("AR", "W_frac"),
        ("AR", "Ma_cruise"),
        ("lam", "SFC"),
        ("lam", "W_frac"),
        ("lam", "Ma_cruise"),
        ("tc", "SFC"),
        ("tc", "W_frac"),
        ("tc", "Ma_cruise"),
        ("SFC", "W_frac"),
        ("SFC", "Ma_cruise"),
        ("W_frac", "Ma_cruise"),
    }

    assert set(case.meta.interaction_pairs) == expected_pairs


def test_vs_b_note_distinguishes_l_d_driver_from_range_sobol_metadata():
    case = AircraftConceptual8D()
    note = case.meta.scenario_notes["VS-B"]

    assert "AR is the strongest main effect on L/D" in note
    assert case.meta.sobol_ST["W_frac"] > 0.6
    assert case.meta.sobol_ST["SFC"] > case.meta.sobol_ST["AR"]
    assert case.meta.sobol_ST["Ma_cruise"] > case.meta.sobol_ST["AR"]
    assert "stored range_km Sobol metadata" in note
    assert "W_frac dominates" in note
    assert "geometry enters range through L/D" in note


def test_aspect_ratio_improves_lift_to_drag_and_range():
    case = AircraftConceptual8D()
    X_phys = np.array(
        [
            [7.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78],
            [10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78],
            [14.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.all(np.diff(Y[:, 0]) > 0.0)
    assert np.all(np.diff(Y[:, 1]) > 0.0)


def test_taper_ratio_improves_oswald_efficiency_lift_to_drag_and_range():
    case = AircraftConceptual8D()
    X_phys = np.array(
        [
            [10.0, 200.0, 0.2, 0.12, 0.06, 0.30, 0.5, 0.78],
            [10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78],
            [10.0, 200.0, 0.7, 0.12, 0.06, 0.30, 0.5, 0.78],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.all(np.diff(Y[:, 0]) > 0.0)
    assert np.all(np.diff(Y[:, 1]) > 0.0)


def test_breguet_range_responds_to_sfc_fuel_fraction_and_mach():
    case = AircraftConceptual8D()
    sfc_sweep = np.array(
        [
            [10.0, 200.0, 0.4, 0.12, 0.05, 0.30, 0.5, 0.78],
            [10.0, 200.0, 0.4, 0.12, 0.08, 0.30, 0.5, 0.78],
        ],
        dtype=float,
    )
    fuel_sweep = np.array(
        [
            [10.0, 200.0, 0.4, 0.12, 0.06, 0.20, 0.5, 0.78],
            [10.0, 200.0, 0.4, 0.12, 0.06, 0.45, 0.5, 0.78],
        ],
        dtype=float,
    )
    mach_sweep = np.array(
        [
            [10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.60],
            [10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.90],
        ],
        dtype=float,
    )

    sfc_y = case.evaluate(case.normalize(sfc_sweep))
    fuel_y = case.evaluate(case.normalize(fuel_sweep))
    mach_y = case.evaluate(case.normalize(mach_sweep))

    assert sfc_y[1, 0] < sfc_y[0, 0]
    assert fuel_y[1, 0] > fuel_y[0, 0]
    assert mach_y[1, 0] > mach_y[0, 0]


def test_range_matches_breguet_identity_with_declared_units():
    case = AircraftConceptual8D()
    row = np.array([[10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78]], dtype=float)
    range_km, L_D_max, _ = case.evaluate(case.normalize(row))[0]

    g = 9.80665
    a_cruise = 295.1
    V_cruise = row[0, 7] * a_cruise
    SFC_SI = row[0, 4] / 3600.0
    fuel_fraction_for_cruise = 0.95 * row[0, 5]
    expected_range_km = (
        V_cruise
        / (g * SFC_SI)
        * L_D_max
        * np.log(1.0 / (1.0 - fuel_fraction_for_cruise))
        / 1000.0
    )

    assert np.isclose(range_km, expected_range_km, rtol=1e-12, atol=1e-12)


def test_lift_to_drag_matches_parabolic_drag_polar_formula():
    case = AircraftConceptual8D()
    row = np.array([[10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78]], dtype=float)
    _, L_D_max, _ = case.evaluate(case.normalize(row))[0]
    AR = row[0, 0]
    lam = row[0, 2]
    tc = row[0, 3]
    swet_sref = 5.5
    cf = 0.0025
    cd_min = cf * swet_sref * (1.0 + 1.5 * tc + 100.0 * tc**4)
    oswald_e = 1.0 / (1.0 + 0.03 * AR * (1.0 - lam) ** 2 + 0.01 * AR * tc / 0.12)
    expected_l_d = 0.5 * np.sqrt(np.pi * AR * oswald_e / cd_min)

    assert np.isclose(L_D_max, expected_l_d, rtol=1e-12, atol=1e-12)
    assert 0.0 < oswald_e < 1.0


def test_empty_weight_fraction_matches_declared_power_law():
    case = AircraftConceptual8D()
    X_phys = np.array(
        [
            [7.0, 100.0, 0.2, 0.10, 0.05, 0.20, 0.3, 0.60],
            [10.0, 200.0, 0.4, 0.14, 0.06, 0.30, 0.5, 0.78],
            [14.0, 400.0, 0.7, 0.18, 0.08, 0.45, 0.8, 0.90],
        ],
        dtype=float,
    )
    we_frac = case.evaluate(case.normalize(X_phys))[:, 2]
    expected = (
        0.55
        * (X_phys[:, 7] ** 0.05)
        * (X_phys[:, 0] ** (-0.03))
        * ((X_phys[:, 3] / 0.14) ** 0.08)
    )

    assert np.allclose(we_frac, expected, rtol=1e-12, atol=1e-12)
    assert np.all((0.0 < we_frac) & (we_frac < 1.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"AR": 0.0},
        {"AR": 6.9},
        {"AR": 14.1},
        {"SW": 0.0},
        {"SW": 99.0},
        {"SW": 401.0},
        {"lam": 0.0},
        {"lam": 0.19},
        {"lam": 0.71},
        {"tc": -0.1},
        {"tc": 0.09},
        {"tc": 0.19},
        {"SFC": 0.0},
        {"SFC": 0.049},
        {"SFC": 0.081},
        {"W_frac": 0.19},
        {"W_frac": 0.46},
        {"W_frac": 1.0},
        {"rho_alt": np.nan},
        {"rho_alt": 0.29},
        {"rho_alt": 0.81},
        {"Ma_cruise": 0.0},
        {"Ma_cruise": 0.59},
        {"Ma_cruise": 0.91},
    ],
)
def test_aircraft_design_rejects_inputs_outside_breguet_contract(kwargs):
    params = {
        "AR": 10.0,
        "SW": 200.0,
        "lam": 0.4,
        "tc": 0.12,
        "SFC": 0.06,
        "W_frac": 0.30,
        "rho_alt": 0.5,
        "Ma_cruise": 0.78,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match="outside the declared"):
        AircraftConceptual8D._design(**params)


def test_range_ratio_follows_breguet_sfc_inverse_scaling():
    case = AircraftConceptual8D()
    X_phys = np.array(
        [
            [10.0, 200.0, 0.4, 0.12, 0.05, 0.30, 0.5, 0.78],
            [10.0, 200.0, 0.4, 0.12, 0.08, 0.30, 0.5, 0.78],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.isclose(Y[0, 0] / Y[1, 0], 0.08 / 0.05, rtol=1e-12, atol=1e-12)


def test_breguet_separable_range_scaling_for_mach_sfc_and_fuel_log_term():
    case = AircraftConceptual8D()
    baseline = np.array([[10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.60]], dtype=float)
    high_mach = baseline.copy()
    high_mach[0, 7] = 0.90
    high_sfc = baseline.copy()
    high_sfc[0, 4] = 0.08
    high_fuel = baseline.copy()
    high_fuel[0, 5] = 0.45

    y0 = case.evaluate(case.normalize(baseline))[0, 0]
    y_mach = case.evaluate(case.normalize(high_mach))[0, 0]
    y_sfc = case.evaluate(case.normalize(high_sfc))[0, 0]
    y_fuel = case.evaluate(case.normalize(high_fuel))[0, 0]

    expected_fuel_ratio = np.log(1.0 / (1.0 - 0.95 * 0.45)) / np.log(
        1.0 / (1.0 - 0.95 * 0.30)
    )

    assert np.isclose(y_mach / y0, 0.90 / 0.60, rtol=1e-12, atol=1e-12)
    assert np.isclose(y_sfc / y0, 0.06 / 0.08, rtol=1e-12, atol=1e-12)
    assert np.isclose(y_fuel / y0, expected_fuel_ratio, rtol=1e-12, atol=1e-12)


def test_thicker_wing_trades_aerodynamic_for_structural_weight():
    case = AircraftConceptual8D()
    X_phys = np.array(
        [
            [10.0, 200.0, 0.4, 0.10, 0.06, 0.30, 0.5, 0.78],
            [10.0, 200.0, 0.4, 0.18, 0.06, 0.30, 0.5, 0.78],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert Y[1, 1] < Y[0, 1]
    assert Y[1, 2] > Y[0, 2]


def test_unused_sizing_and_density_inputs_do_not_change_simplified_model():
    case = AircraftConceptual8D()
    X_phys = np.array(
        [
            [10.0, 100.0, 0.4, 0.12, 0.06, 0.30, 0.3, 0.78],
            [10.0, 400.0, 0.4, 0.12, 0.06, 0.30, 0.8, 0.78],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.allclose(Y[1], Y[0], rtol=1e-12, atol=1e-12)


def test_aircraft_surrogate_failure_is_not_silent_nan(monkeypatch):
    case = AircraftConceptual8D()

    def fail_design(*_args, **_kwargs):
        raise ValueError("simulated conceptual-design calculation failure")

    monkeypatch.setattr(case, "_design", fail_design)
    X_phys = np.array([[10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_design(monkeypatch):
    case = AircraftConceptual8D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("aircraft design surrogate should not be called for invalid input")

    monkeypatch.setattr(case, "_design", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._validate_physical_inputs(
            np.array([10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78], dtype=float)
        )

    invalid_rows = [
        [6.9, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78],
        [14.1, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78],
        [10.0, 99.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78],
        [10.0, 401.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78],
        [10.0, 200.0, 0.19, 0.12, 0.06, 0.30, 0.5, 0.78],
        [10.0, 200.0, 0.71, 0.12, 0.06, 0.30, 0.5, 0.78],
        [10.0, 200.0, 0.4, 0.09, 0.06, 0.30, 0.5, 0.78],
        [10.0, 200.0, 0.4, 0.19, 0.06, 0.30, 0.5, 0.78],
        [10.0, 200.0, 0.4, 0.12, 0.049, 0.30, 0.5, 0.78],
        [10.0, 200.0, 0.4, 0.12, 0.081, 0.30, 0.5, 0.78],
        [10.0, 200.0, 0.4, 0.12, 0.06, 0.19, 0.5, 0.78],
        [10.0, 200.0, 0.4, 0.12, 0.06, 0.46, 0.5, 0.78],
        [10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.29, 0.78],
        [10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.81, 0.78],
        [10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.59],
        [10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.91],
        [10.0, 200.0, 0.4, np.nan, 0.06, 0.30, 0.5, 0.78],
    ]
    for row in invalid_rows:
        with pytest.raises(
            ValueError,
            match="AR/SW/lam/tc/SFC/W_frac/rho_alt/Ma_cruise benchmark domain",
        ):
            case._evaluate_core(np.array([row], dtype=float))


def test_aircraft_surrogate_nonfinite_or_malformed_response_is_not_silent(monkeypatch):
    case = AircraftConceptual8D()
    X_phys = np.array([[10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78]], dtype=float)

    def nonfinite_design(*_args, **_kwargs):
        return np.array([np.nan, 18.0, 0.5])

    monkeypatch.setattr(case, "_design", nonfinite_design)
    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)

    def malformed_design(*_args, **_kwargs):
        return np.array([12000.0, 18.0])

    monkeypatch.setattr(case, "_design", malformed_design)
    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_aircraft_surrogate_finite_nonphysical_response_is_not_silent(monkeypatch):
    case = AircraftConceptual8D()
    X_phys = np.array([[10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78]], dtype=float)

    def negative_range_design(*_args, **_kwargs):
        return np.array([-1.0, 18.0, 0.5])

    monkeypatch.setattr(case, "_design", negative_range_design)
    with pytest.raises(RuntimeError, match="non-physical output"):
        case._evaluate_core(X_phys)

    def invalid_weight_fraction_design(*_args, **_kwargs):
        return np.array([12000.0, 18.0, 1.2])

    monkeypatch.setattr(case, "_design", invalid_weight_fraction_design)
    with pytest.raises(RuntimeError, match="non-physical output"):
        case._evaluate_core(X_phys)


def test_aircraft_surrogate_out_of_contract_response_is_not_archived(monkeypatch):
    case = AircraftConceptual8D()
    X_phys = np.array([[10.0, 200.0, 0.4, 0.12, 0.06, 0.30, 0.5, 0.78]], dtype=float)

    invalid_responses = [
        np.array([1999.0, 18.0, 0.5]),
        np.array([27001.0, 18.0, 0.5]),
        np.array([12000.0, 9.9, 0.5]),
        np.array([12000.0, 25.1, 0.5]),
        np.array([12000.0, 18.0, 0.39]),
        np.array([12000.0, 18.0, 0.66]),
    ]

    for response in invalid_responses:
        monkeypatch.setattr(case, "_design", lambda *_args, **_kwargs: response)
        with pytest.raises(RuntimeError, match="declared output ranges"):
            case._evaluate_core(X_phys)
