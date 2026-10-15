"""Physics regression tests for ``supersonic_intake_5d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.aerodynamics.supersonic_intake_5d import SupersonicIntake5D


def _evaluate(case: SupersonicIntake5D, rows: list[list[float]]) -> np.ndarray:
    X = np.array(rows, dtype=float)
    return case.evaluate(case.normalize(X))


def test_oblique_shock_reduces_mach_and_loses_total_pressure():
    result = SupersonicIntake5D._oblique_shock(Ma1=3.0, delta_deg=8.0, gamma=1.4)

    assert result is not None
    ma2, p02_p01, p2_p1 = result
    assert 1.0 < ma2 < 3.0
    assert 0.0 < p02_p01 < 1.0
    assert p2_p1 > 1.0


def test_normal_shock_matches_standard_gamma14_reference_values():
    ma2, p02_p01, p2_p1 = SupersonicIntake5D._normal_shock(Ma1=3.0, gamma=1.4)

    assert np.isclose(ma2, 0.47519, rtol=5e-5)
    assert np.isclose(p2_p1, 10.33333, rtol=5e-5)
    assert np.isclose(p02_p01, 0.32834, rtol=5e-5)


def test_metadata_declares_low_order_intake_surrogate_scope():
    case = SupersonicIntake5D()
    equation = case.meta.governing_equation
    equation_lower = equation.lower()
    details = case.meta.solver_details.lower()
    references = " ".join(case.meta.physics_references)

    assert "ma²(γ+cos2β)+2" in equation_lower
    assert "(γ+1)ma²" in equation_lower
    assert "constant-gamma shock equations" in equation_lower
    assert "gas-property stressor" in equation_lower
    assert "surrogate penalty" in equation_lower
    assert "no viscous" in details
    assert "unstart" in details
    assert "cfd" in details
    for omitted in (
        "shock/boundary-layer interaction",
        "bleed schedule",
        "cowl-lip geometry",
        "capture-area",
        "spillage",
        "spillage-drag build-up",
        "shock-train dynamics",
        "inlet buzz",
        "unstart dynamics",
        "compressor-face distortion metrics",
        "wind-tunnel engine-airflow sweep",
        "pressure-rake recovery/distortion measurement",
        "diffuser area schedule",
        "inlet flow-rate accounting",
        "inlet drag build-up",
        "surface-grid or cfd-grid output",
        "inlet-engine matching",
        "variable geometry",
        "strong-shock branch selection",
        "back-pressure-driven shock-position solve",
        "3-d sidewall/corner flow",
        "x-59 inlet/nozzle cfd",
        "wind-tunnel inlet-distortion testing",
        "engine-face distortion or operability assessment",
        "propulsion-system integration",
        "real-time test displays",
        "digital engine/aircraft architecture",
        "calorically imperfect shock iteration",
        "vibrational-energy model",
        "real-gas chemistry",
        "wind-tunnel calibration",
        "supin-style inlet design calculation",
    ):
        assert omitted in details
    assert "NASA Glenn, Oblique Shock Relations" in references
    assert "NASA Glenn, Normal Shock Relations" in references
    assert "NASA Glenn, Inlets" in references
    assert "pressure-recovery index" in references
    assert "spillage/capture context" in references
    assert "https://www.grc.nasa.gov/WWW/k-12/airplane/inlet.html" in references
    assert "NASA Glenn, Inlet Performance" in references
    assert "spillage drag / distortion context" in references
    assert "https://www.grc.nasa.gov/www/k-12/airplane/inleth.html" in references
    assert "https://www.grc.nasa.gov/www/k-12/BGP/inleth.html" in references
    assert "NASA Glenn, Inlet Performance Model" in references
    assert "wind-tunnel engine-airflow sweep" in references
    assert "distortion-rake measurement context" in references
    assert (
        "https://ntrs.nasa.gov/citations/20220010443"
        in references
    )
    assert "NACA Report 1135" in references
    assert "calorically perfect versus imperfect normal-shock context" in references
    assert "NASA/TM-2012-217660" in references
    assert "NASA/TM-20240008586" in references
    assert "Support of Flight Demonstration Projects" in references
    assert "support-of-flight-demonstration-projects" in references
    assert "Design and Analysis Software - Inlets and Nozzles" in references
    assert "design-analysis-software" in references
    assert "NASA/TM-20220010443" in references
    assert "20220010443" in references
    assert "engine-face distortion" in references
    assert "constant-gamma shock relations" in case.meta.input_variables[3].description
    assert "low-level shock helpers reject formula-domain violations" in details
    assert "ma <= 1" in details
    assert "non-positive compression angle" in details
    assert "gamma <= 1" in details
    assert "rejects malformed arrays" in details
    assert "before oblique-shock cascade evaluation" in details
    assert hasattr(case, "_validate_physical_inputs")
    assert "threshold_jump" in case.meta.response_topology
    assert "discontinuous" in case.meta.response_topology
    assert case.meta.effective_dim == 5
    assert set(case.meta.important_variables) == {
        "Ma_in",
        "delta1",
        "delta2",
        "gamma",
        "alpha_off",
    }
    assert set(case.meta.interaction_pairs) == {
        ("Ma_in", "delta1"),
        ("Ma_in", "delta2"),
        ("Ma_in", "gamma"),
        ("Ma_in", "alpha_off"),
        ("delta1", "delta2"),
        ("delta1", "gamma"),
        ("delta1", "alpha_off"),
        ("delta2", "gamma"),
        ("delta2", "alpha_off"),
        ("gamma", "alpha_off"),
    }


def test_applicable_scenarios_have_sobol_and_boundary_notes():
    case = SupersonicIntake5D()

    assert set(case.meta.applicable_scenarios) <= set(case.meta.scenario_notes)
    vs_f = case.meta.scenario_notes["VS-F"]
    assert "stored eta_pr Sobol metadata" in vs_f
    assert "Ma_in dominates" in vs_f
    assert "full five-input importance list is retained" in vs_f
    assert case.meta.sobol_ST["Ma_in"] > 0.9
    assert case.meta.sobol_ST["delta1"] < 0.05
    assert case.meta.sobol_ST["delta2"] < 0.05
    assert case.meta.sobol_ST["gamma"] < 0.05
    assert case.meta.sobol_ST["alpha_off"] < 0.05
    assert "eta_pr > 0.95" in case.meta.scenario_notes["VS-G"]


def test_validation_material_declares_uncalibrated_shock_surrogate_scope():
    validation_text = Path("tests/_cross_validation/supersonic_intake_5d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path(
        "doe_testbed/cases/aerodynamics/supersonic_intake_5d.py"
    ).read_text(encoding="utf-8")

    assert "inviscid" in validation_text
    assert "two-ramp" in validation_text
    assert "5% external validation" in validation_text
    assert "2024 SUPIN technical memorandum" in validation_text
    assert "not validation data for this compact two-ramp benchmark" in validation_text
    normalized_validation_text = " ".join(validation_text.split())
    assert "pressure recovery is an inlet performance index" in normalized_validation_text
    assert "spillage/capture losses are real inlet effects" in normalized_validation_text
    assert "high pressure recovery" in normalized_validation_text
    assert "low spillage drag" in normalized_validation_text
    assert "low distortion" in normalized_validation_text
    assert "engine airflow is varied" in normalized_validation_text
    assert "pressure recovery, spillage drag, and distortion are measured" in normalized_validation_text
    assert "rake-style measurement workflows are outside" in normalized_validation_text
    assert "shock-train" in validation_text
    normalized_validation = " ".join(validation_text.lower().split())
    for phrase in (
        "shock/boundary-layer interaction",
        "bleed",
        "cowl-lip",
        "capture-area",
        "spillage",
        "spillage drag",
        "inlet buzz",
        "compressor-face distortion",
        "inlet-engine matching",
        "variable geometry",
        "strong-shock branch selection",
        "back-pressure-driven shock-position solves",
        "3-d sidewall",
        "x-59 inlet/nozzle cfd",
        "wind-tunnel inlet-distortion",
        "wind-tunnel engine-airflow sweeps",
        "pressure-rake recovery/distortion",
        "engine-face distortion",
        "propulsion-system integration",
        "real-time test displays",
        "digital engine/aircraft architecture",
        "calorically imperfect shocks",
        "vibrational-energy",
        "real-gas chemistry",
        "wind-tunnel",
        "supin-style",
    ):
        assert phrase in normalized_validation
    assert "constant-gamma gas-property stressor" in normalized_validation
    assert "not a high-temperature thermochemical model" in normalized_validation
    assert "weak attached-shock branch" in normalized_validation
    assert "explicitly discontinuous" in normalized_validation
    assert "rather than globally smooth" in normalized_validation
    assert "formula-domain guard" in normalized_validation
    assert "upstream ma <= 1" in normalized_validation
    assert "non-positive compression" in normalized_validation
    assert "gamma <= 1" in normalized_validation
    assert "rejects malformed arrays" in normalized_validation
    assert "before oblique-shock cascade evaluation" in normalized_validation
    assert "cannot extrapolate beyond the declared two-ramp intake bounds" in normalized_validation
    assert "does not include viscous boundary layers" in source_text
    assert "calorically perfect, constant-γ shock equations" in source_text


def test_reviewer_markdown_note_keeps_supersonic_intake_p2_scope_boundary():
    note = Path("tests/_cross_validation/supersonic_intake_5d.md")
    text = note.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "P2 plausibility-only" in normalized
    assert "No response-surface change" in normalized
    assert "Inviscid two-ramp external-compression shock surrogate" in normalized
    assert "constant-gamma gas-property stressor" in normalized
    assert "not CFD-validated" in normalized
    assert "not wind-tunnel validated" in normalized
    assert "not SUPIN-equivalent" in normalized
    assert "not an X-59 inlet model" in normalized
    assert "not a complete inlet design model" in normalized
    assert "not a real-gas or thermochemical inlet solver" in normalized
    assert "viscous shock/boundary-layer interaction" in normalized
    assert "compressor-face distortion metrics" in normalized
    assert "wind-tunnel engine-airflow sweeps" in normalized
    assert "pressure-rake recovery/distortion measurements" in normalized
    assert "back-pressure-driven shock-position solves" in normalized
    assert "SUPIN-style inlet design" in normalized
    assert "stagewise shock-product loss control" in normalized
    assert "each attached shock has `p0_downstream / p0_upstream <= 1`" in normalized
    for source_anchor in [
        "https://www.grc.nasa.gov/www/k-12/airplane/oblique.html",
        "https://www.grc.nasa.gov/WWW/BGH/normal.html",
        "https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf",
        "https://www.grc.nasa.gov/WWW/k-12/airplane/inlet.html",
        "https://www.grc.nasa.gov/www/k-12/BGP/inleth.html",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/inlet-performance-model/",
        "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120013600.pdf",
        "https://ntrs.nasa.gov/citations/20240008586",
        "https://www.nasa.gov/glenn/research/inlets-and-nozzles/design-analysis-software/",
        "https://www.nasa.gov/glenn/research/inlets-and-nozzles/support-of-flight-demonstration-projects/",
        "https://ntrs.nasa.gov/citations/20220010443",
    ]:
        assert source_anchor in text


def test_web_audit_records_supersonic_intake_official_source_recheck():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `supersonic_intake_5d` 2026-06-08 official-source re-check"
        in audit_text
    )
    assert "NASA Glenn oblique- and normal-shock pages" in normalized
    assert "constant-gamma shock-equation anchors" in normalized
    assert "NASA Glenn inlet material" in normalized
    assert "pressure recovery and spillage/capture framing" in normalized
    assert "NASA/TM-2012-217660 and NASA/TM-20240008586 SUPIN material" in normalized
    assert "production inlet-tool boundary evidence" in normalized
    assert "not validation data for this compact two-ramp surrogate" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    for forbidden_claim in (
        "CFD-validated",
        "wind-tunnel validated",
        "SUPIN-equivalent",
        "complete inlet design model",
        "unstart/shock-train/bleed/cowl/spillage/distortion",
        "engine-matching/variable-geometry",
        "strong-shock/back-pressure-branch",
        "3-D sidewall/corner-flow",
        "real-gas model",
    ):
        assert forbidden_claim in normalized
    assert (
        "No numerical response resweep is needed solely for this official-source re-check"
        in normalized
    )


def test_web_audit_records_supersonic_intake_inlet_performance_boundary_note():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `supersonic_intake_5d` 2026-06-08 inlet-performance boundary re-check"
        in audit_text
    )
    assert "NASA Glenn inlet performance page" in normalized
    assert "high pressure recovery, low spillage drag, and low distortion" in normalized
    assert "compressor-face pressure-recovery / spillage-drag / distortion" in normalized
    assert "strong-shock branch selection" in normalized
    assert "back-pressure-driven shock-position solves" in normalized
    assert "SUPIN 2024 continues to document flow rates, total pressure recovery, inlet drag" in normalized
    assert "These production metrics are boundary evidence only" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized


def test_web_audit_records_supersonic_intake_x59_boundary_note():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `supersonic_intake_5d` 2026-06-08 NASA-Glenn-X59 boundary note"
        in audit_text
    )
    assert "Support of Flight Demonstration Projects" in normalized
    assert "Design and Analysis Software - Inlets and Nozzles" in normalized
    assert "High Speed Inlet Distortion Test for the X-59" in normalized
    assert "CFD" in normalized
    assert "wind-tunnel" in normalized
    assert "engine-face distortion" in normalized
    assert "propulsion-system integration" in normalized
    assert "digital engine/aircraft architecture" in normalized
    assert "three-dimensional structured volume grids" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert (
        "No numerical response resweep is needed solely for this NASA Glenn "
        "X-59/SUPIN-software boundary note"
    ) in normalized
    assert (
        "Additional `supersonic_intake_5d` 2026-06-09 stagewise shock-product "
        "loss-control note"
    ) in normalized
    assert "cumulative total-pressure recovery is a non-increasing product" in normalized
    assert "implementation-control regression" in normalized
    assert (
        "Additional `supersonic_intake_5d` 2026-06-09 NASA Glenn inlet-performance-model boundary note"
        in normalized
    )
    assert "engine airflow is varied" in normalized
    assert "pressure recovery, spillage drag, and distortion" in normalized
    assert "pressure-rake recovery/distortion measurements" in normalized
    assert "does not run a wind-tunnel airflow sweep" in normalized
    assert "does not compute rake distortion metrics" in normalized


def test_web_audit_records_supersonic_intake_current_page_spot_check():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `supersonic_intake_5d` 2026-06-09 current-page spot check:"
        in audit_text
    )
    for phrase in (
        "current NASA Glenn oblique-shock, normal-shock, inlet, and inlet-performance model pages",
        "NASA/TM-20240008586 SUPIN material",
        "NASA Glenn inlets/nozzles project pages",
        "NASA/TM-20220010443 X-59 inlet-distortion material",
        "constant-gamma oblique-shock theta-beta-M relation",
        "weak/strong attached-shock branches",
        "normal-shock total-pressure loss",
        "static-pressure rise",
        "calorically perfect versus calorically imperfect gas boundary",
        "pressure recovery",
        "spillage/capture effects",
        "low spillage drag",
        "low distortion",
        "engine-airflow sweeps",
        "pressure-rake or distortion-style measurements",
        "full inlet design/analysis workflows",
        "flowpath geometry",
        "flow rates",
        "total-pressure recovery",
        "inlet drag",
        "surface outputs",
        "structured CFD-volume-grid handoff",
        "CFD",
        "wind-tunnel testing",
        "inlet-distortion measurement",
        "engine-face distortion",
        "propulsion-system integration",
        "real-time test displays",
        "digital engine/aircraft architecture",
        "two-ramp inviscid, constant-gamma, weak-attached-shock cascade",
        "terminal normal-shock branch",
        "detached-shock surrogate penalty",
        "Do not describe this case as CFD validated",
        "wind-tunnel validated",
        "SUPIN validated",
        "SUPIN-equivalent",
        "X-59 validated",
        "inlet-distortion validated",
        "engine-operability validated",
        "spillage-drag validated",
        "pressure-rake validated",
        "inlet-engine matched",
        "variable-geometry validated",
        "unstart/shock-train validated",
        "calorically imperfect validated",
        "real-gas validated",
        "complete inlet-design validated",
        "Evidence tier remains P2 plausibility-only",
        "No numerical response resweep is needed solely for this current-page spot check",
    ):
        assert phrase in normalized


def test_oblique_shock_detaches_when_wedge_angle_exceeds_limit():
    assert SupersonicIntake5D._oblique_shock(Ma1=2.0, delta_deg=30.0, gamma=1.4) is None


@pytest.mark.parametrize(
    ("ma1", "delta_deg", "gamma"),
    [
        (1.0, 8.0, 1.4),
        (3.0, 0.0, 1.4),
        (3.0, 8.0, 1.0),
        (np.nan, 8.0, 1.4),
    ],
)
def test_oblique_shock_rejects_formula_domain_violations(ma1, delta_deg, gamma):
    with pytest.raises(ValueError, match="Ma1 > 1"):
        SupersonicIntake5D._oblique_shock(Ma1=ma1, delta_deg=delta_deg, gamma=gamma)


@pytest.mark.parametrize(
    ("ma1", "gamma"),
    [
        (1.0, 1.4),
        (3.0, 1.0),
        (np.inf, 1.4),
    ],
)
def test_normal_shock_rejects_formula_domain_violations(ma1, gamma):
    with pytest.raises(ValueError, match="Ma1 > 1"):
        SupersonicIntake5D._normal_shock(Ma1=ma1, gamma=gamma)


def test_intake_perf_rejects_formula_domain_violations_before_archive_contract():
    case = SupersonicIntake5D()

    for row in (
        [1.0, 10.0, 10.0, 1.4, 0.0],
        [3.0, 2.0, 10.0, 1.4, -3.0],
        [3.0, 10.0, 0.0, 1.4, 0.0],
        [3.0, 10.0, 10.0, 1.0, 0.0],
        [3.0, np.nan, 10.0, 1.4, 0.0],
    ):
        with pytest.raises(ValueError):
            case._intake_perf(*row)


def test_total_pressure_recovery_degrades_at_higher_mach():
    case = SupersonicIntake5D()

    low_mach, high_mach = _evaluate(
        case,
        [
            [2.0, 5.0, 5.0, 1.4, 0.0],
            [5.0, 5.0, 5.0, 1.4, 0.0],
        ],
    )

    assert high_mach[0] < low_mach[0]
    assert high_mach[2] > low_mach[2]


def test_external_compression_has_better_recovery_than_direct_normal_shock():
    case = SupersonicIntake5D()
    X_phys = np.array([[3.0, 10.0, 10.0, 1.4, 0.0]], dtype=float)
    intake = case.evaluate(case.normalize(X_phys))[0]
    direct_normal = SupersonicIntake5D._normal_shock(Ma1=3.0, gamma=1.4)

    assert intake[0] > direct_normal[1]
    assert intake[2] > direct_normal[2]
    assert intake[1] > direct_normal[0]


def test_intake_outputs_equal_stagewise_shock_product_for_attached_case():
    case = SupersonicIntake5D()
    X_phys = np.array([[3.0, 8.0, 8.0, 1.4, 0.0]], dtype=float)
    y = case.evaluate(case.normalize(X_phys))[0]

    ma2, p02_p01_1, p2_p1_1 = SupersonicIntake5D._oblique_shock(
        Ma1=3.0, delta_deg=8.0, gamma=1.4
    )
    ma3, p02_p01_2, p2_p1_2 = SupersonicIntake5D._oblique_shock(
        Ma1=ma2, delta_deg=8.0, gamma=1.4
    )
    ma_exit, p02_p01_3, p2_p1_3 = SupersonicIntake5D._normal_shock(
        Ma1=ma3, gamma=1.4
    )

    assert np.allclose(
        y,
        [
            p02_p01_1 * p02_p01_2 * p02_p01_3,
            ma_exit,
            p2_p1_1 * p2_p1_2 * p2_p1_3,
        ],
        rtol=1e-12,
        atol=1e-12,
    )


def test_attached_shock_cascade_recovery_is_stagewise_loss_product():
    case = SupersonicIntake5D()
    y = case.evaluate(case.normalize(np.array([[3.0, 8.0, 8.0, 1.4, 0.0]])))[0]

    ma2, pr1, sr1 = SupersonicIntake5D._oblique_shock(3.0, 8.0, 1.4)
    ma3, pr2, sr2 = SupersonicIntake5D._oblique_shock(ma2, 8.0, 1.4)
    ma_exit, pr3, sr3 = SupersonicIntake5D._normal_shock(ma3, 1.4)
    recovery_after_stages = np.cumprod([pr1, pr2, pr3])
    compression_after_stages = np.cumprod([sr1, sr2, sr3])

    assert np.all((0.0 < np.array([pr1, pr2, pr3])) & (np.array([pr1, pr2, pr3]) <= 1.0))
    assert np.all(np.diff(recovery_after_stages) < 0.0)
    assert np.all(np.diff(compression_after_stages) > 0.0)
    assert np.isclose(y[0], recovery_after_stages[-1], rtol=1e-12, atol=1e-12)
    assert np.isclose(y[1], ma_exit, rtol=1e-12, atol=1e-12)
    assert np.isclose(y[2], compression_after_stages[-1], rtol=1e-12, atol=1e-12)


def test_stronger_external_compression_raises_static_pressure_ratio():
    case = SupersonicIntake5D()
    Y = _evaluate(
        case,
        [
            [3.0, 5.0, 5.0, 1.4, 0.0],
            [3.0, 10.0, 10.0, 1.4, 0.0],
            [3.0, 15.0, 20.0, 1.4, 0.0],
        ],
    )

    assert np.all(np.diff(Y[:, 2]) > 0.0)
    assert np.all(np.diff(Y[:, 1]) > 0.0)


def test_positive_off_design_angle_increases_first_wedge_compression():
    case = SupersonicIntake5D()
    Y = _evaluate(
        case,
        [
            [3.0, 10.0, 10.0, 1.4, -3.0],
            [3.0, 10.0, 10.0, 1.4, 0.0],
            [3.0, 10.0, 10.0, 1.4, 3.0],
        ],
    )

    assert np.all(np.diff(Y[:, 2]) > 0.0)
    assert np.all(np.diff(Y[:, 1]) > 0.0)


def test_specific_heat_ratio_changes_shock_loss_and_compression():
    case = SupersonicIntake5D()
    low_gamma, high_gamma = _evaluate(
        case,
        [
            [3.0, 10.0, 10.0, 1.3, 0.0],
            [3.0, 10.0, 10.0, 1.4, 0.0],
        ],
    )

    assert high_gamma[0] > low_gamma[0]
    assert high_gamma[1] > low_gamma[1]
    assert high_gamma[2] < low_gamma[2]


def test_failure_boundary_targets_low_recovery_or_detached_shock_region():
    case = SupersonicIntake5D()
    eta_output = case.meta.output_variables[0]
    assert eta_output.name == "eta_pr"
    assert eta_output.failure_threshold == 0.80
    assert eta_output.failure_direction == "below"
    assert "eta_pr < 0.80" in case.meta.failure_condition
    assert "shock detachment" in case.meta.failure_condition

    eta = _evaluate(
        case,
        [
            [2.0, 5.0, 5.0, 1.4, 0.0],
            [5.0, 15.0, 20.0, 1.4, 3.0],
        ],
    )[:, 0]

    assert eta[0] > eta_output.failure_threshold
    assert eta[1] < eta_output.failure_threshold
    assert SupersonicIntake5D._oblique_shock(Ma1=2.0, delta_deg=30.0, gamma=1.4) is None


def test_declared_output_ranges_cover_intake_domain_corners():
    case = SupersonicIntake5D()
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    X_phys = np.array(list(itertools.product(*bounds)), dtype=float)
    Y = case.evaluate(case.normalize(X_phys))

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_declared_mach_exit_range_covers_near_sonic_valid_case():
    case = SupersonicIntake5D()
    X_phys = np.array([[2.21722513, 12.65738976, 16.23297245, 1.36077619, 2.30278409]])
    ma_exit = case.evaluate(case.normalize(X_phys))[0, 1]
    lower, upper = case.meta.output_variables[1].expected_range

    assert ma_exit > 0.95
    assert lower <= ma_exit <= upper


def test_intake_solver_failure_is_not_silent_nan(monkeypatch):
    case = SupersonicIntake5D()

    def fail_solve(*_args, **_kwargs):
        raise ValueError("simulated shock-system solve failure")

    monkeypatch.setattr(case, "_intake_perf", fail_solve)
    X_phys = np.array([[3.0, 10.0, 10.0, 1.4, 0.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_nonfinite_or_nonphysical_intake_response_is_not_silent_nan(monkeypatch):
    case = SupersonicIntake5D()

    def bad_response(*_args, **_kwargs):
        return np.array([np.nan, 1.2, 0.5])

    monkeypatch.setattr(case, "_intake_perf", bad_response)
    X_phys = np.array([[3.0, 10.0, 10.0, 1.4, 0.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_out_of_contract_shock_product_is_not_silently_clipped(monkeypatch):
    case = SupersonicIntake5D()
    oblique_calls = {"count": 0}

    def fake_oblique(_ma, _delta, _gamma):
        oblique_calls["count"] += 1
        if oblique_calls["count"] == 1:
            return 2.0, 0.99, 10.0
        return 1.5, 0.99, 10.0

    def fake_normal(_ma, _gamma):
        return 0.5, 0.99, 3.0

    monkeypatch.setattr(case, "_oblique_shock", fake_oblique)
    monkeypatch.setattr(case, "_normal_shock", fake_normal)

    X_phys = np.array([[3.0, 10.0, 10.0, 1.4, 0.0]], dtype=float)
    raw = case._intake_perf(*X_phys[0])

    assert raw[2] > 200.0
    oblique_calls["count"] = 0
    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_private_solver_rejects_malformed_or_out_of_domain_inputs_before_cascade(monkeypatch):
    case = SupersonicIntake5D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("intake shock cascade should not be called for invalid input")

    monkeypatch.setattr(case, "_intake_perf", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._evaluate_core(np.array([3.0, 10.0, 10.0, 1.4, 0.0], dtype=float))

    invalid_rows = [
        [1.99, 10.0, 10.0, 1.4, 0.0],
        [5.01, 10.0, 10.0, 1.4, 0.0],
        [3.0, 4.99, 10.0, 1.4, 0.0],
        [3.0, 15.01, 10.0, 1.4, 0.0],
        [3.0, 10.0, 4.99, 1.4, 0.0],
        [3.0, 10.0, 20.01, 1.4, 0.0],
        [3.0, 10.0, 10.0, 1.299, 0.0],
        [3.0, 10.0, 10.0, 1.401, 0.0],
        [3.0, 10.0, 10.0, 1.4, -3.1],
        [3.0, 10.0, 10.0, 1.4, 3.1],
        [3.0, 10.0, np.nan, 1.4, 0.0],
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError, match="Ma_in/delta1/delta2/gamma/alpha_off benchmark domain"):
            case._evaluate_core(np.array([row], dtype=float))
