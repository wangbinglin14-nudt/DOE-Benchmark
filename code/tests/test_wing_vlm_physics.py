"""Physics regression tests for ``wing_vlm_6d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.aerodynamics.wing_vlm_6d import WingVLM6D


def _evaluate(case: WingVLM6D, rows: list[list[float]]) -> np.ndarray:
    X = np.array(rows, dtype=float)
    return case.evaluate(case.normalize(X))


def test_lifting_line_lift_increases_with_angle_of_attack():
    case = WingVLM6D()

    low_alpha, high_alpha = _evaluate(
        case,
        [
            [8.0, 1.0, 0.0, 2.0, 0.1, 0.0],
            [8.0, 1.0, 0.0, 6.0, 0.1, 0.0],
        ],
    )

    assert high_alpha[0] > low_alpha[0]
    assert high_alpha[1] > low_alpha[1]


def test_sweep_reduces_lift_for_fixed_wing_condition():
    case = WingVLM6D()

    unswept, swept = _evaluate(
        case,
        [
            [8.0, 1.0, 0.0, 6.0, 0.1, 0.0],
            [8.0, 1.0, 35.0, 6.0, 0.1, 0.0],
        ],
    )

    assert swept[0] < unswept[0]
    assert swept[1] < unswept[1]


def test_sweep_scaling_is_documented_cos_squared_normal_flow_surrogate():
    case = WingVLM6D()
    sweep_deg = 35.0

    unswept, swept = _evaluate(
        case,
        [
            [8.0, 0.7, 0.0, 6.0, 0.3, -2.0],
            [8.0, 0.7, sweep_deg, 6.0, 0.3, -2.0],
        ],
    )

    scale = np.cos(np.deg2rad(sweep_deg)) ** 2
    assert np.isclose(swept[0] / unswept[0], scale, rtol=1e-12, atol=1e-12)
    assert np.isclose(swept[1] / unswept[1], scale**2, rtol=1e-12, atol=1e-12)
    assert np.isclose(swept[2], unswept[2], rtol=1e-12, atol=1e-12)


def test_prandtl_glauert_correction_increases_lift_with_mach():
    case = WingVLM6D()

    low_mach, high_mach = _evaluate(
        case,
        [
            [8.0, 1.0, 0.0, 6.0, 0.1, 0.0],
            [8.0, 1.0, 0.0, 6.0, 0.7, 0.0],
        ],
    )

    assert high_mach[0] > low_mach[0]


def test_finite_wing_lift_slope_matches_prandtl_order_of_magnitude():
    case = WingVLM6D()
    AR = 10.0
    alpha_deg = 4.0
    mach = 0.1
    y = _evaluate(case, [[AR, 1.0, 0.0, alpha_deg, mach, 0.0]])[0]

    beta = np.sqrt(1.0 - mach**2)
    a0 = 2.0 * np.pi / beta
    expected_cl = a0 * np.deg2rad(alpha_deg) / (1.0 + a0 / (np.pi * AR))

    assert np.isclose(y[0], expected_cl, rtol=0.05)


def test_negative_tip_twist_washout_reduces_lift_for_fixed_root_angle():
    case = WingVLM6D()

    washout, untwisted, washin = _evaluate(
        case,
        [
            [8.0, 1.0, 0.0, 4.0, 0.1, -5.0],
            [8.0, 1.0, 0.0, 4.0, 0.1, 0.0],
            [8.0, 1.0, 0.0, 4.0, 0.1, 5.0],
        ],
    )

    assert washout[0] < untwisted[0] < washin[0]


def test_induced_drag_matches_lifting_line_oswald_identity():
    case = WingVLM6D()
    X_phys = np.array(
        [
            [8.0, 1.0, 0.0, 6.0, 0.1, 0.0],
            [8.0, 0.4, 20.0, 4.0, 0.4, -2.0],
            [14.0, 0.7, 35.0, 8.0, 0.7, 5.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    CL = Y[:, 0]
    CDi = Y[:, 1]
    e = Y[:, 2]
    AR = X_phys[:, 0]

    assert np.all((e > 0.0) & (e <= 1.0))
    assert np.allclose(CDi, CL**2 / (np.pi * AR * e), rtol=1e-12, atol=1e-12)


def test_twisted_zero_lift_state_keeps_finite_induced_drag():
    case = WingVLM6D()

    # For this legal rectangular-wing slice, alpha cancels the first Fourier
    # coefficient while the twist-induced higher modes remain nonzero.
    y = _evaluate(case, [[8.0, 1.0, 0.0, 2.2952777811501086, 0.1, -5.0]])[0]

    assert abs(y[0]) < 1e-12
    assert y[1] > 1e-4
    assert 0.0 <= y[2] < 1e-20


def test_trivial_zero_loading_state_uses_finite_efficiency_convention():
    case = WingVLM6D()
    X_phys = np.array(
        [
            [8.0, 1.0, 0.0, 0.0, 0.1, 0.0],
            [12.0, 0.4, 20.0, 0.0, 0.5, 0.0],
            [14.0, 0.2, 35.0, 0.0, 0.7, 0.0],
        ],
        dtype=float,
    )

    Y = case.evaluate(case.normalize(X_phys))

    assert np.allclose(Y[:, 0], 0.0, rtol=0.0, atol=1e-14)
    assert np.allclose(Y[:, 1], 0.0, rtol=0.0, atol=1e-14)
    assert np.allclose(Y[:, 2], 1.0, rtol=0.0, atol=1e-14)


def test_prandtl_glauert_beta_floor_is_not_needed_in_legal_domain():
    case = WingVLM6D()
    rng = np.random.default_rng(20260606)
    lower = np.array([var.lower for var in case.meta.input_variables], dtype=float)
    upper = np.array([var.upper for var in case.meta.input_variables], dtype=float)
    X_phys = lower + (upper - lower) * rng.random((256, case.meta.dim_input))
    Y = case.evaluate(case.normalize(X_phys))

    beta_arg = 1.0 - X_phys[:, 4] ** 2
    assert beta_arg.min() > 0.01
    assert np.all(np.isfinite(Y))

    with pytest.raises(ValueError, match="0 <= Ma < 1"):
        case._lifting_line(AR=8.0, TR=1.0, sweep_deg=0.0, alpha_deg=4.0, Ma=1.0, twist_deg=0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"AR": 0.0}, "AR > 0"),
        ({"TR": 0.0}, "TR > 0"),
        ({"Ma": -0.1}, "0 <= Ma < 1"),
        ({"Ma": 1.0}, "0 <= Ma < 1"),
        ({"alpha_deg": np.nan}, "finite"),
        ({"N_terms": 1}, "integer N_terms >= 2"),
        ({"N_terms": 2.5}, "integer N_terms >= 2"),
        ({"N_terms": np.nan}, "integer N_terms >= 2"),
    ],
)
def test_lifting_line_rejects_formula_domain_violations(kwargs, message):
    args = dict(AR=8.0, TR=1.0, sweep_deg=0.0, alpha_deg=4.0, Ma=0.1, twist_deg=0.0)
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        WingVLM6D._lifting_line(**args)


def test_lifting_line_solver_does_not_fall_back_to_least_squares(monkeypatch):
    case = WingVLM6D()
    called_lstsq = False

    def fail_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("simulated singular matrix")

    def track_lstsq(*_args, **_kwargs):
        nonlocal called_lstsq
        called_lstsq = True
        raise AssertionError("lstsq fallback should not be used")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    monkeypatch.setattr(np.linalg, "lstsq", track_lstsq)

    with pytest.raises(ValueError, match="collocation matrix is singular"):
        case._lifting_line(AR=8.0, TR=1.0, sweep_deg=0.0, alpha_deg=4.0, Ma=0.1, twist_deg=0.0)
    assert called_lstsq is False


def test_metadata_marks_input_output_response_as_nonlinear():
    case = WingVLM6D()

    assert case.meta.is_linear is False
    assert "strong_interaction" in case.meta.response_topology


def test_multioutput_topology_is_not_globally_monotone_due_to_induced_drag():
    case = WingVLM6D()

    assert "smooth_monotone" not in case.meta.response_topology
    assert "smooth_unimodal" in case.meta.response_topology

    X_phys = np.array(
        [
            [8.0, 1.0, 0.0, 0.0, 0.1, -5.0],
            [8.0, 1.0, 0.0, 2.5, 0.1, -5.0],
            [8.0, 1.0, 0.0, 5.0, 0.1, -5.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert Y[0, 0] < Y[1, 0] < Y[2, 0]
    assert Y[1, 1] < Y[0, 1]
    assert Y[1, 1] < Y[2, 1]


def test_metadata_declares_pllt_references_and_interaction_scope():
    case = WingVLM6D()
    references = "\n".join(case.meta.physics_references)

    assert "pllt" in case.meta.tags
    assert "Anderson" in references
    assert "NASA Glenn, Induced Drag Coefficient" in references
    assert "NASA Glenn, Winglets" in references
    assert "NASA Glenn, Factors that Affect Drag" in references
    assert "Prandtl-Glauert" in references
    assert "https://web.mit.edu/drela/Public/web/avl/avl_doc.txt" in references
    assert (
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/"
        in references
    )
    assert "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/winglets/" in references
    assert (
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/factors-that-affect-drag/"
        in references
    )
    assert "https://www.nasa.gov/history/SP-4219/Chapter3.html" in references
    assert set(case.meta.interaction_pairs) == {
        ("AR", "TR"),
        ("AR", "sweep"),
        ("AR", "alpha"),
        ("AR", "Ma"),
        ("AR", "twist"),
        ("TR", "alpha"),
        ("TR", "Ma"),
        ("TR", "twist"),
        ("sweep", "alpha"),
        ("sweep", "Ma"),
        ("sweep", "twist"),
        ("alpha", "Ma"),
        ("alpha", "twist"),
        ("Ma", "twist"),
    }


def test_vs_b_note_matches_current_cl_sobol_metadata():
    case = WingVLM6D()
    note = case.meta.scenario_notes["VS-B"]

    assert case.meta.sobol_ST["alpha"] > 0.7
    assert case.meta.sobol_ST["twist"] > 0.2
    assert case.meta.sobol_ST["TR"] == 0.0
    assert "alpha dominates" in note
    assert "tip twist is second" in note
    assert "near-zero CL effect" in note
    assert "AR and alpha have dominant main effects" not in note


def test_declared_output_ranges_cover_lifting_line_domain_corners():
    case = WingVLM6D()
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    X_phys = np.array(list(itertools.product(*bounds)), dtype=float)
    Y = case.evaluate(case.normalize(X_phys))

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_validation_material_declares_lifting_line_surrogate_scope():
    case = WingVLM6D()
    validation_text = Path("tests/_cross_validation/wing_vlm_6d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path("doe_testbed/cases/aerodynamics/wing_vlm_6d.py").read_text(
        encoding="utf-8"
    )

    assert "Prandtl lifting-line/Fourier surrogate" in validation_text
    assert "externally calibrated" in validation_text
    assert "full VLM" in validation_text
    assert "not a calibrated VLM/CFD replacement" in source_text
    assert case.meta.name == "Wing PLLT aerodynamic characteristics (historical VLM label)"
    assert "PLLT" in case.meta.solver_details
    assert "AVL/VLM horseshoe-vortex lattice" in case.meta.solver_details
    assert "Trefftz-plane" in case.meta.solver_details
    assert "low-level PLLT helper rejects" in case.meta.solver_details
    assert "AR <= 0" in case.meta.solver_details
    assert "TR <= 0" in case.meta.solver_details
    assert "non-subsonic or negative Mach" in case.meta.solver_details
    assert "invalid Fourier-term counts" in case.meta.solver_details
    assert "rejects malformed arrays" in case.meta.solver_details
    assert "before PLLT/Fourier assembly" in case.meta.solver_details
    assert "out-of-contract" in case.meta.solver_details
    assert hasattr(case, "_validate_physical_inputs")
    details = case.meta.solver_details.lower()
    for omitted_scope in [
        "profile/parasite drag",
        "viscous stall",
        "nasa glenn drag-factor or winglet workflow calibration",
        "camber/zero-lift-angle",
        "dihedral",
        "winglet",
        "fuselage-interference",
        "ground-effect",
        "true swept lifting-surface",
        "vlm",
        "horseshoe-vortex lattice",
        "control-surface/body model",
        "trefftz-plane",
        "flap/control-surface design",
        "multi-lifting-surface vlm design",
        "vspaero version-transition workflow",
        "panel-method",
        "cfd",
    ]:
        assert omitted_scope in details
    normalized_source = " ".join(source_text.split())
    normalized_validation = " ".join(validation_text.split())
    assert "induced-drag-only finite-wing coefficients" in normalized_source
    assert "denominator-free" in normalized_source
    assert "PLLT CDi formula stays finite" in normalized_source
    assert "twisted zero-lift states keep their finite induced drag" in normalized_source
    assert "trivial zero-loading state CL = CDi = 0" in normalized_source
    assert "finite-output convention for the benchmark" in normalized_source
    assert "not AVL/VLM" in normalized_source
    assert "does not construct a lifting-surface lattice" in normalized_source
    assert "not be used as evidence for swept-wing vortex-lattice" in normalized_source
    for source_anchor in [
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/winglets/",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/factors-that-affect-drag/",
        "https://www.nasa.gov/history/SP-4219/Chapter3.html",
        "https://web.mit.edu/drela/Public/web/avl/avl_doc.txt",
        "https://web.mit.edu/drela/Public/web/avl/AVL_User_Primer.pdf",
        "https://ntrs.nasa.gov/citations/19760021082",
        "https://ntrs.nasa.gov/api/citations/19760021082/downloads/19760021082.pdf",
        "https://ntrs.nasa.gov/citations/19760021075",
        "https://www.nasa.gov/reference/openvsp-vspaero-basics/",
    ]:
        assert source_anchor in source_text
        assert source_anchor in validation_text
    assert "public NASA/MIT anchors support only the induced-drag identity" in normalized_source
    assert "They are not external validation data for this PLLT/Fourier surrogate" in normalized_source
    assert "NASA Glenn winglet and drag-factor pages are used only to clarify" in normalized_source
    assert "induced drag is one drag component" in normalized_validation
    assert "do not make ``CDi`` a complete aircraft drag coefficient" in normalized_validation
    assert "trivial zero-loading state" in normalized_validation
    assert "not a claim that Oswald efficiency is physically measurable at zero lift" in normalized_validation
    assert "NASA NTRS vortex-lattice preliminary-design material" in normalized_validation
    assert "rigid-aircraft vortex-lattice/AVL lifting-surface model" in normalized_validation
    assert "official MIT AVL User Primer" in normalized_validation
    assert "planform, twist, camber, flap/control-surface" in normalized_validation
    assert "broader conceptual-aircraft design context" in normalized_validation
    assert "multi-lifting-surface design" in normalized_validation
    assert "NASA OpenVSP VSPAERO basics" in normalized_validation
    assert "potential-flow aerodynamic analysis" in normalized_validation
    assert "OpenVSP geometries" in normalized_validation
    assert "DegenGeom-style thin-surface representations" in normalized_validation
    assert "does not run OpenVSP/VSPAERO" in normalized_validation
    assert "VSPAERO v3.45 / solver v7 transition" in normalized_validation
    assert "not as a stable numerical reference dataset" in normalized_validation
    assert "version-transition notes likewise cannot be used as calibration" in normalized_validation
    assert "review pressure/wake solutions" in normalized_validation
    assert "The public NASA/MIT anchors are formula and boundary evidence only" in normalized_validation
    assert "not external validation data for the PLLT/Fourier surrogate" in normalized_validation
    assert "OpenVSP/VSPAERO aerodynamic analysis" in normalized_validation
    assert "rejects malformed arrays" in normalized_validation
    assert "before PLLT/Fourier assembly" in normalized_validation
    assert "cannot extrapolate beyond the registered case bounds" in normalized_validation


def test_reviewer_markdown_note_keeps_wing_vlm_p2_scope_boundary():
    note = Path("tests/_cross_validation/wing_vlm_6d.md")
    text = note.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "P2 plausibility-only" in normalized
    assert "No response-surface change" in normalized
    assert "Prandtl lifting-line theory (PLLT)" in normalized
    assert "not AVL-equivalent" in normalized
    assert "not VLM-equivalent" in normalized
    assert "not a swept-wing solver" in normalized
    assert "not a panel method" in normalized
    assert "not CFD" in normalized
    assert "Gamma(theta) = 2 b V_inf sum_n A_n sin(n theta)" in normalized
    assert "CDi = pi AR sum_n n A_n^2" in normalized
    assert "trivial zero-loading state" in normalized
    assert "not a physical Oswald-efficiency measurement at zero lift" in normalized
    assert "CL_swept = CL cos^2(Lambda)" in normalized
    assert "CDi_swept = CDi cos^4(Lambda)" in normalized
    assert "induced-drag-only finite-wing coefficients" in normalized
    for source_anchor in [
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/winglets/",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/factors-that-affect-drag/",
        "https://www.nasa.gov/history/SP-4219/Chapter3.html",
        "https://web.mit.edu/drela/Public/web/avl/",
        "https://web.mit.edu/drela/Public/web/avl/avl_doc.txt",
        "https://web.mit.edu/drela/Public/web/avl/AVL_User_Primer.pdf",
        "https://ntrs.nasa.gov/citations/19760021082",
        "https://ntrs.nasa.gov/api/citations/19760021082/downloads/19760021082.pdf",
        "https://ntrs.nasa.gov/citations/19760021075",
        "https://www.nasa.gov/reference/openvsp-vspaero-basics/",
    ]:
        assert source_anchor in text
    assert "NASA OpenVSP VSPAERO basics" in normalized
    assert "potential-flow aerodynamic analysis using OpenVSP geometries" in normalized
    assert "DegenGeom thin-surface representation" in normalized
    assert "pressure/wake solution review workflows" in normalized
    assert "VSPAERO v3.45 / solver v7 transition" in normalized
    assert "stable solver-reference data" in normalized
    assert "workflow boundary evidence only" in normalized
    assert "OpenVSP/VSPAERO-equivalent aerodynamic analysis" in normalized
    assert "OpenVSP/VSPAERO geometry or DegenGeom workflows" in normalized


def test_readme_and_si_use_lifting_line_label_not_full_vlm():
    readme = Path("README.md").read_text(encoding="utf-8")
    readme_zh = Path("README_zh.md").read_text(encoding="utf-8")
    si_cases = Path("paper/si_cases.md").read_text(encoding="utf-8")
    si_cases_tex = Path("paper/si_cases.tex").read_text(encoding="utf-8")

    assert "Wing lifting-line aerodynamics" in readme
    assert "Lifting-line theory + P\u2013G correction" in readme
    assert "机翼升力线气动力学" in readme_zh
    assert "升力线理论 + P\u2013G 修正" in readme_zh
    assert "Prandtl Lifting-Line Theory (PLLT)" in si_cases
    assert "Prandtl Lifting-Line Theory (PLLT)" in si_cases_tex
    assert "Wing VLM aerodynamics" not in readme
    assert "VLM + P\u2013G correction" not in readme
    assert "VLM + P\u2013G 修正" not in readme_zh


def test_web_audit_keeps_wing_vlm_evidence_tier_and_source_scope_honest():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `wing_vlm_6d` |")
        and "NASA Glenn supports the induced-drag identity" in line
    )

    assert "Evidence tier remains P2" in row
    assert "NASA Glenn supports the induced-drag identity" in row
    assert "NASA History SP-4219 supports Prandtl-Glauert" in row
    assert "MIT AVL documentation is boundary evidence" in row
    assert "AVL User Primer" in row
    assert "NASA NTRS vortex-lattice preliminary-design material" in row
    assert "NASA OpenVSP VSPAERO basics" in row
    assert "potential-flow aerodynamics tool" in row
    assert "OpenVSP geometries" in row
    assert "DegenGeom thin-surface" in row
    assert "OpenVSP/VSPAERO" in row
    assert "planform/twist/camber/flap/control-surface" in row
    assert "extended vortex-lattice lifting surfaces" in row
    assert "horseshoe vortices" in row
    assert "Trefftz-plane induced-drag analysis" in row
    assert "not external validation of a swept-wing VLM" in row
    assert "not AVL/VLM" in row
    assert "not AVL/VLM" in row
    assert "OpenVSP/VSPAERO aerodynamic analysis" in row
    assert "or CFD validation" in row


def test_web_audit_records_wing_vlm_official_source_recheck():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `wing_vlm_6d` 2026-06-08 official-source re-check" in audit_text
    assert "public NASA/MIT sources were re-opened as formula and boundary evidence" in normalized
    assert "not as case-level validation data" in normalized
    assert "NASA Glenn induced-drag material supports the `CDi = CL^2/(pi*AR*e)` identity" in normalized
    assert "NASA History SP-4219 remains Prandtl-Glauert background" in normalized
    assert "not a transonic-burble, buffet, or swept-wing validation source" in normalized
    assert "MIT AVL documentation, the official MIT AVL User Primer, and NASA NTRS vortex-lattice preliminary-design material" in normalized
    assert "AVL User Primer" in normalized
    assert "lifting-surface vortex lattices" in normalized
    assert "horseshoe vortices" in normalized
    assert "Trefftz-plane induced-drag analysis" in normalized
    assert "outside the registered PLLT/Fourier surrogate" in normalized
    assert "Additional `wing_vlm_6d` 2026-06-08 official-source boundary tightening" in audit_text
    assert "rigid-aircraft extended vortex-lattice workflow" in normalized
    assert "run cases, trimming, control deflections, and output inspection" in normalized
    assert "broader rigid-aircraft conceptual-design workflow" in normalized
    assert "conceptual-aircraft planform/control-surface design" in normalized
    assert (
        "Additional `wing_vlm_6d` 2026-06-09 NASA OpenVSP/VSPAERO boundary confirmation"
        in normalized
    )
    assert "potential-flow aerodynamics tool" in normalized
    assert "OpenVSP geometries" in normalized
    assert "DegenGeom thin-surface" in normalized
    assert "does not construct OpenVSP geometry" in normalized
    assert "does not run VSPAERO" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    for forbidden_claim in (
        "externally validated VLM/AVL",
        "rigid-aircraft AVL/VLM",
        "swept-wing VLM",
        "panel-method",
        "all-drag",
        "viscous-stall",
        "flap/control-surface",
        "multi-lifting-surface",
        "Trefftz-plane",
        "CFD output",
    ):
        assert forbidden_claim in normalized
    assert (
        "No numerical response resweep is needed solely for this official-source re-check"
        in normalized
    )
    assert "previously recorded response-changing PLLT fixes still require" in normalized
    assert "Additional `wing_vlm_6d` 2026-06-09 NASA Glenn drag-component boundary note" in normalized
    assert "`CDi` in this case is induced drag only" in normalized
    assert "profile/skin-friction, form, wave, viscosity" in normalized
    assert "must not be described as an all-drag wing model" in normalized
    assert "No numerical response resweep is needed solely for this official-source boundary note" in normalized
    assert "Additional `wing_vlm_6d` 2026-06-09 zero-loading convention note" in normalized
    assert "trivial zero-loading state (`alpha=0` and `twist=0`) as `CL=0, CDi=0, e=1`" in normalized
    assert "finite benchmark-output convention" in normalized
    assert "not a physical Oswald-efficiency measurement at zero lift" in normalized
    assert "No numerical response resweep is needed solely for this convention note" in normalized
    assert "Additional `wing_vlm_6d` 2026-06-09 OpenVSP/VSPAERO version-boundary note" in normalized
    assert "VSPAERO v3.45 / solver v7 transition" in normalized
    assert "versioned production tool/workflow" in normalized
    assert "not a stable reference-output dataset" in normalized
    assert "does not construct OpenVSP geometry" in normalized
    assert "select a VSPAERO solver version" in normalized
    assert "calibrate the PLLT/Fourier response surface against VSPAERO output" in normalized
    assert "No numerical response resweep is needed solely for this official-source boundary note" in normalized


def test_web_audit_records_wing_vlm_current_page_spot_check():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `wing_vlm_6d` 2026-06-09 current-page spot check:"
        in audit_text
    )
    for phrase in (
        "current NASA Glenn induced-drag, winglet, and drag-factor pages",
        "NASA History SP-4219",
        "MIT AVL documentation/User Primer",
        "NASA SP-405/NTRS vortex-lattice material",
        "NASA OpenVSP VSPAERO basics",
        "finite-wing induced-drag identity",
        "role of aspect ratio and span efficiency",
        "induced drag is only one drag component",
        "profile/skin-friction, form, wave, viscosity, compressibility",
        "Prandtl-Glauert as subcritical linearized-compressibility background",
        "not a shock, buffet, drag-divergence, or swept-wing correction",
        "true vortex-lattice / AVL workflows",
        "lifting-surface vortex lattices",
        "horseshoe vortices",
        "slender-body options",
        "control deflections",
        "trim/run cases",
        "Trefftz-plane induced-drag analysis",
        "planform/twist/camber/flap/control-surface studies",
        "versioned potential-flow tool",
        "OpenVSP geometries and DegenGeom thin-surface representations",
        "VSPAERO v3.45 / solver v7 transition",
        "reference-value setup",
        "pressure/wake solution viewing",
        "results-manager plots",
        "thin-surface VLM and thick-surface panel outputs",
        "unsteady-result tabs",
        "customized input setup",
        "command-line execution",
        "CSV/result-file workflows",
        "historical-label case",
        "Prandtl lifting-line / Glauert-Fourier surrogate",
        "conceptual cos^2 sweep scaling",
        "induced- drag-only finite-wing outputs",
        "Do not describe this case as AVL validated",
        "VLM validated",
        "OpenVSP/VSPAERO validated",
        "panel-method validated",
        "pressure/wake-solution validated",
        "swept-wing validated",
        "all-drag validated",
        "profile/parasite-drag validated",
        "winglet-workflow validated",
        "control-surface/flap validated",
        "trim/run-case validated",
        "multi-lifting-surface validated",
        "Trefftz-plane validated",
        "wind-tunnel validated",
        "CFD validated",
        "Evidence tier remains P2 plausibility-only",
        "No numerical response resweep is needed solely for this current-page spot check",
    ):
        assert phrase in normalized


def test_applicable_scenarios_have_non_overclaiming_notes():
    case = WingVLM6D()

    assert set(case.meta.applicable_scenarios) <= set(case.meta.scenario_notes)
    vs_d = case.meta.scenario_notes["VS-D"]
    assert "small-sample convergence curves" in vs_d
    assert "without pre-claiming" in vs_d
    assert "sufficient N" in vs_d
    vs_c = case.meta.scenario_notes["VS-C"]
    assert "finite-wing trend/identity checks" in vs_c
    assert "not profile-drag, stall, or swept-wing validation" in vs_c


def test_lifting_line_solver_failure_is_not_silent_nan(monkeypatch):
    case = WingVLM6D()

    def fail_solve(*_args, **_kwargs):
        raise ValueError("simulated PLLT linear-solve failure")

    monkeypatch.setattr(case, "_lifting_line", fail_solve)
    X_phys = np.array([[8.0, 0.7, 15.0, 4.0, 0.3, -2.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_nonfinite_or_nonphysical_lifting_line_response_is_not_silent_nan(monkeypatch):
    case = WingVLM6D()

    def bad_solve(*_args, **_kwargs):
        return np.array([np.nan, -0.01, 1.2])

    monkeypatch.setattr(case, "_lifting_line", bad_solve)
    X_phys = np.array([[8.0, 0.7, 15.0, 4.0, 0.3, -2.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_out_of_contract_lifting_line_response_is_not_archived(monkeypatch):
    case = WingVLM6D()
    X_phys = np.array([[8.0, 0.7, 15.0, 4.0, 0.3, -2.0]], dtype=float)

    invalid_responses = [
        np.array([-0.31, 0.01, 0.8]),
        np.array([1.51, 0.01, 0.8]),
        np.array([0.5, 0.21, 0.8]),
        np.array([0.5, 0.01, -0.1]),
        np.array([0.5, 0.01, 1.1]),
    ]

    for response in invalid_responses:
        monkeypatch.setattr(case, "_lifting_line", lambda *_args, **_kwargs: response)
        with pytest.raises(RuntimeError, match="declared output ranges"):
            case._evaluate_core(X_phys)


def test_private_solver_rejects_malformed_or_out_of_domain_inputs_before_pllt(monkeypatch):
    case = WingVLM6D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("PLLT helper should not be called for invalid physical input")

    monkeypatch.setattr(case, "_lifting_line", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._evaluate_core(np.array([8.0, 0.7, 15.0, 4.0, 0.3, -2.0], dtype=float))

    invalid_rows = [
        [4.99, 0.7, 15.0, 4.0, 0.3, -2.0],
        [14.01, 0.7, 15.0, 4.0, 0.3, -2.0],
        [8.0, 0.19, 15.0, 4.0, 0.3, -2.0],
        [8.0, 1.01, 15.0, 4.0, 0.3, -2.0],
        [8.0, 0.7, -0.1, 4.0, 0.3, -2.0],
        [8.0, 0.7, 35.1, 4.0, 0.3, -2.0],
        [8.0, 0.7, 15.0, -0.1, 0.3, -2.0],
        [8.0, 0.7, 15.0, 8.1, 0.3, -2.0],
        [8.0, 0.7, 15.0, 4.0, 0.09, -2.0],
        [8.0, 0.7, 15.0, 4.0, 0.71, -2.0],
        [8.0, 0.7, 15.0, 4.0, 0.3, -5.1],
        [8.0, 0.7, 15.0, 4.0, 0.3, 5.1],
        [8.0, np.nan, 15.0, 4.0, 0.3, -2.0],
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError, match="AR/TR/sweep/alpha/Ma/twist benchmark domain"):
            case._evaluate_core(np.array([row], dtype=float))
