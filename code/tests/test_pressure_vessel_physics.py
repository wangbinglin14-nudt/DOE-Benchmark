"""Physics regression tests for ``pressure_vessel_4d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.multidisciplinary.pressure_vessel_4d import PressureVessel4D


def _domain_corners(case: PressureVessel4D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_declared_domain_respects_thin_wall_assumption():
    case = PressureVessel4D()
    radii = case.meta.input_variables[1]
    thicknesses = case.meta.input_variables[2]
    source_text = Path(
        "doe_testbed/cases/multidisciplinary/pressure_vessel_4d.py"
    ).read_text(encoding="utf-8")

    assert thicknesses.upper / radii.lower <= 0.1
    assert "t/R <= 0.1" in source_text
    assert "t/R < 0.1" not in source_text
    assert "t/R << 1" not in source_text


def test_metadata_declares_strength_only_surrogate_scope():
    case = PressureVessel4D()
    sigma_y = next(var for var in case.meta.input_variables if var.name == "sigma_y")
    source_text = Path(
        "doe_testbed/cases/multidisciplinary/pressure_vessel_4d.py"
    ).read_text(encoding="utf-8")

    assert "surrogate" in case.meta.governing_equation.lower()
    assert "not a full asme" in case.meta.governing_equation.lower()
    solver_details = case.meta.solver_details.lower()
    assert "buckling" in solver_details
    for omitted in (
        "thick-wall",
        "lame",
        "radial-stress",
        "end-cap/support",
        "nozzle/opening",
        "weld efficiency",
        "corrosion allowance",
        "fatigue",
        "fracture",
        "hydrostatic-test",
        "fabrication/inspection",
        "material-class rule set",
        "pressure-relief-device certification",
        "nondestructive examination",
        "inspection-acceptance appendices",
        "full asme code check",
    ):
        assert omitted in solver_details
    references = "\n".join(case.meta.physics_references)
    for public_anchor in [
        "https://engineeringlibrary.org/reference/simple-thin-pressure-vessels-air-force-stress-manual",
        "https://ntrs.nasa.gov/api/citations/20160000456/downloads/20160000456.pdf",
        "https://www.asme.org/codes-standards/find-codes-standards/bpvc-viii-1-bpvc-section-viii-rules-construction-pressure-vessels-division-1",
        "https://www.asme.org/certification-accreditation/Boiler-and-Pressure-Vessel-Certification",
    ]:
        assert public_anchor in references
    assert "design-code context only" in references.lower()
    assert "certification/quality-control-system context only" in references.lower()
    assert "code-stamp" in references.lower()
    assert "external-pressure coverage" in references.lower()
    assert "fabrication-method rules" in references.lower()
    assert "assembly, erection, examination" in references.lower()
    assert "nondestructive-examination / inspection acceptance appendices" in references.lower()
    assert "uv/ud pressure-relief-device scopes" in references.lower()
    assert "2021" not in references
    assert "sigma_theta = p r / t" in references.lower()
    assert "sigma_l = p r / (2t)" in references.lower()
    assert "radial stress neglected" in references.lower()
    assert case.meta.is_linear is False
    assert "smooth_monotone" in case.meta.response_topology
    assert "strong_interaction" in case.meta.response_topology
    assert ("P", "sigma_y") in case.meta.interaction_pairs
    assert ("R", "sigma_y") in case.meta.interaction_pairs
    assert ("t", "sigma_y") in case.meta.interaction_pairs
    assert "reference" in sigma_y.description.lower()
    assert "fs < 1.5" in case.meta.failure_condition.lower()
    assert "fs > 1.5" not in case.meta.failure_condition.lower()
    normalized_source = " ".join(source_text.split())
    assert "m/L = ρ·π·((R+t)²−R²)" in source_text
    assert "benchmark-domain limit rather than a high-fidelity pressure-vessel design guarantee" in normalized_source
    assert "construction, fabrication, inspection, testing, certification" in normalized_source
    assert "weld-joint efficiency, corrosion allowance, openings, material-class rules, fabrication methods, assembly, erection, examination, and pressure-relief-device scopes" in normalized_source
    assert "pressure-relief-device certification" in normalized_source
    assert "external-pressure vessel rules" in normalized_source
    assert "material-class rules, fabrication methods, assembly, erection, examination, and pressure-relief-device scopes" in normalized_source
    assert "S1(P) = S1(R) = S1(t) = S1(σ_y)" not in source_text
    assert "Saltelli computation for the declared linear input distribution" in normalized_source


def test_scenario_notes_do_not_overstate_pre_result_sample_sufficiency():
    case = PressureVessel4D()
    vs_d = case.meta.scenario_notes["VS-D"].lower()
    vs_g = case.meta.scenario_notes["VS-G"].lower()

    assert "analytical baseline" in vs_d
    for forbidden in ("suffices", "accurate surrogate", "minimum samples", "fewest"):
        assert forbidden not in vs_d
        assert forbidden not in vs_g
    assert "relative to the preregistered mcs reference" in vs_g


def test_robust_design_metadata_includes_material_control_variable():
    case = PressureVessel4D()
    vs_h = case.meta.scenario_notes["VS-H"]

    assert case.meta.noise_factor_candidates == ["P"]
    assert set(case.meta.control_factor_candidates) == {"R", "t", "sigma_y"}
    assert "sigma_y" in vs_h
    assert "R, t, and sigma_y are control/design factors" in vs_h


def test_validation_material_declares_non_asme_strength_surrogate_scope():
    validation_text = Path("tests/_cross_validation/pressure_vessel_4d.py").read_text(
        encoding="utf-8"
    )
    validation_text_lower = validation_text.lower()
    normalized_validation = " ".join(validation_text.split())
    source_text = Path(
        "doe_testbed/cases/multidisciplinary/pressure_vessel_4d.py"
    ).read_text(encoding="utf-8")

    assert "Thin-walled cylindrical pressure vessel" in validation_text
    assert "implemented strength-only" in validation_text
    assert "not a full ASME" in validation_text
    assert "ASME certification overview is cited only" in validation_text
    assert "quality-control-system" in validation_text
    assert "Certification Mark" in validation_text
    assert "U/UM/U2/U3" in validation_text
    assert "UV/UD" in validation_text
    assert "code-stamp" in validation_text
    assert "Air Force/NASA anchors support the same thin-wall formulae" in validation_text
    assert "hoop and axial membrane stresses separately" in normalized_validation
    assert "biaxial von-Mises identity" in normalized_validation
    assert "do not provide external case-level validation" in validation_text
    assert "Private physical-input contract" in validation_text
    assert "domain violations are rejected before" in validation_text
    assert "does not change legal-domain finite responses" in validation_text
    for phrase in (
        "buckling",
        "thick-wall",
        "radial stress",
        "end-cap",
        "support",
        "nozzle/opening",
        "fatigue",
        "fracture",
        "certification",
        "material allowables",
        "external pressure",
        "nondestructive examination",
    ):
        assert phrase in validation_text_lower
    assert "not a claim of ASME BPVC Section VIII compliance" in source_text
    assert "Validation scope" in validation_text
    assert "Target accuracy" not in validation_text
    assert "externally validated" not in source_text.lower()


def test_web_validation_audit_preserves_p2_boundary():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit_text.split())
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `pressure_vessel_4d` | Evidence tier remains P2 analytical-control. Thin-wall pressure-vessel")
    )

    assert "Evidence tier remains P2 analytical-control" in row
    assert "Air Force" in row
    assert "NASA/TM-2015-218216" in row
    assert "ASME BPVC Section VIII" in row
    assert "U`/`UM`/`U2`/`U3" in row
    assert "not a full ASME design/certification/code-stamp check" in row
    assert "no-fixed-edition-ASME-overclaim" in row
    assert "biaxial von-Mises stress" in row
    assert "implemented closed-form equations are directly checked" in row
    assert "Keep P2 analytical-control" in row
    assert "externally validated ASME pressure-vessel" in row
    assert (
        "Additional `pressure_vessel_4d` 2026-06-08 official-source re-check:"
        in audit_text
    )
    assert "Evidence tier remains P2 analytical-control" in normalized_audit
    assert "Do not describe this case as externally validated ASME pressure-vessel design" in normalized_audit
    assert "fixed-edition BPVC compliance" in normalized_audit
    assert "code-stamp authorization" in normalized_audit
    assert "quality-control systems and Certification Mark scopes" in normalized_audit
    assert "legal-domain finite analytical responses are unchanged" in normalized_audit
    assert (
        "Additional `pressure_vessel_4d` 2026-06-08 ASME scope tightening:"
        in audit_text
    )
    assert "internal or external pressures exceeding 15 psig" in normalized_audit
    assert "fabrication-method requirements" in normalized_audit
    assert "NDE/inspection-acceptance appendices" in normalized_audit
    assert "`UV`/`UD` pressure-relief-device programs" in normalized_audit
    assert (
        "Additional `pressure_vessel_4d` 2026-06-09 current-page spot check:"
        in audit_text
    )
    assert "NASA/TM-2015-218216 thin-pressure-vessel pipe formulation" in normalized_audit
    assert "public Air Force simple-thin-pressure-vessel stress-manual material" in normalized_audit
    assert "hoop and longitudinal membrane stresses" in normalized_audit
    assert "neglected radial stress in the thin-wall setting" in normalized_audit
    assert "conventional `t/R <= 0.1` benchmark-domain limit" in normalized_audit
    assert "certification-ready analysis" in normalized_audit
    assert "calibrated vessel failure prediction" in normalized_audit
    assert "No numerical response resweep is needed solely for this current-page spot check" in normalized_audit


def test_declared_output_ranges_cover_domain_corners():
    case = PressureVessel4D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_outputs_match_thin_wall_analytical_formulae():
    case = PressureVessel4D()
    X_phys = np.array(
        [
            [10.0, 1.0, 0.02, 400.0],
            [15.0, 1.5, 0.03, 500.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    P, R, t, sigma_y = X_phys.T
    sigma_vm = P * R / t * np.sqrt(3.0) / 2.0
    fs = sigma_y / sigma_vm
    w_per_l = 7850.0 * np.pi * ((R + t) ** 2 - R**2)
    expected = np.stack([fs, w_per_l, sigma_vm], axis=1)

    assert np.allclose(Y, expected, rtol=1e-12, atol=1e-12)


def test_von_mises_response_matches_biaxial_membrane_stress_identity():
    case = PressureVessel4D()
    X_phys = np.array([[12.0, 1.25, 0.025, 450.0]], dtype=float)

    fs, _, sigma_vm = case.evaluate(case.normalize(X_phys))[0]
    P, R, t, sigma_y = X_phys[0]
    sigma_hoop = P * R / t
    sigma_axial = P * R / (2.0 * t)
    expected_vm = np.sqrt(
        sigma_hoop**2 - sigma_hoop * sigma_axial + sigma_axial**2
    )

    assert np.isclose(sigma_axial, 0.5 * sigma_hoop, rtol=1e-12, atol=1e-12)
    assert np.isclose(sigma_vm, expected_vm, rtol=1e-12, atol=1e-12)
    assert np.isclose(fs, sigma_y / expected_vm, rtol=1e-12, atol=1e-12)


def test_safety_factor_is_monotone_in_load_geometry_and_material():
    case = PressureVessel4D()
    base = np.array([[10.0, 1.0, 0.02, 400.0]], dtype=float)
    higher_pressure = np.array([[15.0, 1.0, 0.02, 400.0]], dtype=float)
    higher_radius = np.array([[10.0, 1.5, 0.02, 400.0]], dtype=float)
    thicker_wall = np.array([[10.0, 1.0, 0.03, 400.0]], dtype=float)
    stronger_material = np.array([[10.0, 1.0, 0.02, 500.0]], dtype=float)

    fs_base = case.evaluate(case.normalize(base))[0, 0]
    fs_pressure = case.evaluate(case.normalize(higher_pressure))[0, 0]
    fs_radius = case.evaluate(case.normalize(higher_radius))[0, 0]
    fs_thicker = case.evaluate(case.normalize(thicker_wall))[0, 0]
    fs_stronger = case.evaluate(case.normalize(stronger_material))[0, 0]

    assert fs_pressure < fs_base
    assert fs_radius < fs_base
    assert fs_thicker > fs_base
    assert fs_stronger > fs_base


def test_failure_boundary_matches_declared_strength_threshold():
    case = PressureVessel4D()
    P = 10.0
    R = 1.0
    sigma_y = 400.0
    t_at_threshold = 1.5 * P * R * np.sqrt(3.0) / (2.0 * sigma_y)
    X_phys = np.array([[P, R, t_at_threshold, sigma_y]], dtype=float)

    fs = case.evaluate(case.normalize(X_phys))[0, 0]

    assert np.isclose(fs, 1.5, rtol=1e-12, atol=1e-12)


def test_failure_boundary_classifies_points_on_both_sides():
    case = PressureVessel4D()
    P = 10.0
    R = 1.0
    sigma_y = 400.0
    t_at_threshold = 1.5 * P * R * np.sqrt(3.0) / (2.0 * sigma_y)
    X_phys = np.array(
        [
            [P, R, 0.9 * t_at_threshold, sigma_y],
            [P, R, 1.1 * t_at_threshold, sigma_y],
        ],
        dtype=float,
    )

    fs = case.evaluate(case.normalize(X_phys))[:, 0]

    assert fs[0] < case.meta.output_variables[0].failure_threshold
    assert fs[1] > case.meta.output_variables[0].failure_threshold


def test_private_physical_input_contract_rejects_malformed_arrays():
    case = PressureVessel4D()

    with pytest.raises(ValueError, match=r"shape \(n, 4\)"):
        case._evaluate_core(np.array([10.0, 1.0, 0.02, 400.0]))

    with pytest.raises(ValueError, match=r"shape \(n, 4\)"):
        case._evaluate_core(np.array([[10.0, 1.0, 0.02]], dtype=float))


@pytest.mark.parametrize(
    ("column", "bad_value", "name"),
    [
        (0, 1.0 - 1e-6, "P"),
        (0, 20.0 + 1e-6, "P"),
        (1, 0.5 - 1e-6, "R"),
        (1, 2.0 + 1e-6, "R"),
        (2, 0.005 - 1e-8, "t"),
        (2, 0.05 + 1e-8, "t"),
        (3, 200.0 - 1e-6, "sigma_y"),
        (3, 600.0 + 1e-6, "sigma_y"),
    ],
)
def test_private_physical_input_contract_rejects_declared_domain_violations(
    column: int, bad_value: float, name: str
):
    case = PressureVessel4D()
    X_phys = np.array([[10.0, 1.0, 0.02, 400.0]], dtype=float)
    X_phys[0, column] = bad_value

    with pytest.raises(ValueError, match=f"{name!r} is outside"):
        case._evaluate_core(X_phys)


def test_private_physical_input_contract_rejects_nonfinite_values():
    case = PressureVessel4D()
    X_phys = np.array([[10.0, 1.0, np.nan, 400.0]], dtype=float)

    with pytest.raises(ValueError, match="non-finite physical inputs"):
        case._evaluate_core(X_phys)


def test_nonfinite_analytical_response_is_not_silent_nan():
    case = PressureVessel4D()
    X_phys = np.array([[10.0, 1.0, 0.0, 400.0]], dtype=float)

    with pytest.raises(ValueError, match="'t' is outside"):
        case._evaluate_core(X_phys)


def test_finite_nonphysical_analytical_response_is_not_silent():
    case = PressureVessel4D()
    X_phys = np.array([[10.0, 1.0, -0.02, 400.0]], dtype=float)

    with pytest.raises(ValueError, match="'t' is outside"):
        case._evaluate_core(X_phys)
