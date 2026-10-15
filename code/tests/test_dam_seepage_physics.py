"""Physics regression tests for ``dam_seepage_6d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.geomechanics.dam_seepage_6d import DamSeepage6D


def _domain_corners(case: DamSeepage6D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_declared_output_ranges_cover_domain_corners():
    case = DamSeepage6D()
    Y = case.evaluate(case.normalize(_domain_corners(case)))

    assert np.all(np.isfinite(Y))
    for idx, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        assert np.nanmin(Y[:, idx]) >= lower
        assert np.nanmax(Y[:, idx]) <= upper


def test_metadata_declares_fixed_head_drain_surrogate_scope():
    case = DamSeepage6D()
    equation = case.meta.governing_equation
    details = case.meta.solver_details.lower()
    drain = next(var for var in case.meta.input_variables if var.name == "drain_eff")
    references = "\n".join(case.meta.physics_references)

    assert "∇·(K(x,z)·∇H) = 0" in equation
    assert "H_d = H2 + (1-drain_eff)·(H1-H2)/2" in equation
    assert "no-drain natural head" in equation
    assert "grid-aligned grout-curtain mask" in details
    assert "declared curtain position/width" in details
    assert "pressure-relief surrogate" in details
    assert "drain node" in details
    assert "preventing recharge-like uplift increases" in details
    assert "seepage-discharge" in details
    assert "exit-gradient" in details
    assert "piping/internal-erosion" in details
    assert "unsaturated" in details
    assert "transient" in details
    assert "anisotropic" in details
    assert "fracture/joint" in details
    assert "3-d abutment-flow" in details
    assert "flow-net construction" in details
    assert "piezometer/instrumentation interpretation" in details
    assert "permeability-uncertainty sensitivity study" in details
    assert "mesh-convergence evidence" in details
    assert "seepage-velocity field" in details
    assert "full pore-pressure-field reporting" in details
    assert "drainage-gallery" in details
    assert "collector-pipe/header layout" in details
    assert "relief-well" in details
    assert "relief-well design/construction/maintenance" in details
    assert "spacing/sizing" in details
    assert "relief-well maintenance/repair current-practice review" in details
    assert "filter-design" in details
    assert "rehabilitation-planning" in details
    assert "fema p-1032 seepage/internal-erosion evaluation" in details
    assert "emergency-response" in details
    assert "flow-rate/pressure/water-quality measurement" in details
    assert "risk-informed dam-safety likelihood/loading/system-response/consequence assessment" in details
    assert "potential-failure-mode analysis" in details
    assert "structural-stability" in details
    assert hasattr(case, "_validate_physical_inputs")
    assert "named _validate_physical_inputs" in details
    assert "rejects malformed arrays" in details
    assert "before sparse fdm assembly" in details
    assert "benchmark pressure-relief factor" in drain.description.lower()
    assert "not a design-code drain-efficiency recommendation" in drain.description
    assert "not a calibrated drain-hole-network parameter" in drain.description
    assert any("USACE EM 1110-2-2200" in ref for ref in case.meta.physics_references)
    assert any("USACE EM 1110-2-1901" in ref for ref in case.meta.physics_references)
    assert any("USACE EM 1110-2-1914" in ref for ref in case.meta.physics_references)
    assert any("ERDC/GSL TR-25-5" in ref for ref in case.meta.physics_references)
    assert any("USBR Design Standards No. 13" in ref for ref in case.meta.physics_references)
    assert any("FERC Engineering Guidelines" in ref for ref in case.meta.physics_references)
    assert any("FERC Risk-Informed Decision Making" in ref for ref in case.meta.physics_references)
    assert any("Potential Failure Modes" in ref for ref in case.meta.physics_references)
    assert any("Association of State Dam Safety Officials" in ref for ref in case.meta.physics_references)
    assert any("FEMA P-1032" in ref for ref in case.meta.physics_references)
    assert "https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2200.pdf" in references
    assert "https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-1901.pdf" in references
    assert "https://damsafety.org/content/seepage-analysis-and-control-dams" in references
    assert "EM%201110-2-1914_FINAL%2020250307.pdf" in references
    assert "https://usace.contentdm.oclc.org/digital/collection/p16021coll9/id/2595/" in references
    assert "usace-relief-wells-for-dams-and-levees-history-and-current-practice" in references
    assert "https://www.usbr.gov/tsc/techreferences/designstandards-datacollectionguides/finalds-pdfs/DS13-8.pdf" in references
    assert "https://www.ferc.gov/sites/default/files/2020-04/chap3.pdf" in references
    assert "https://www.ferc.gov/dam-safety-and-inspections/risk-informed-decision-making-ridm" in references
    assert "https://www.ferc.gov/dam-safety-and-inspections/potential-failure-modes-pfms" in references
    assert "dam-safety-performance-monitoring-program-dspmp-and-potential-failure" in references
    assert "fema_p1032_eval_monitoring_seepage_internal_erosion.pdf" in references
    assert "flow-rate/pressure/water-quality monitoring" in references
    assert "life-cycle monitoring" in references
    assert case.meta.is_linear is False
    assert "strong_interaction" in case.meta.response_topology
    assert "threshold_jump" in case.meta.response_topology
    assert "discontinuous" in case.meta.response_topology
    assert "no-drain natural head" in case.meta.scenario_notes["VS-C"]
    assert set(case.meta.interaction_pairs) == {
        ("H1", "K_curtain"),
        ("H2", "drain_eff"),
        ("K_base", "K_curtain"),
        ("K_curtain", "D_base"),
        ("drain_eff", "H1"),
    }


def test_validation_material_declares_uncalibrated_drain_surrogate_scope():
    validation_text = Path("tests/_cross_validation/dam_seepage_6d.py").read_text(
        encoding="utf-8"
    )
    normalized_validation_text = " ".join(validation_text.split())
    source_text = Path("doe_testbed/cases/geomechanics/dam_seepage_6d.py").read_text(
        encoding="utf-8"
    )
    normalized_source_text = " ".join(source_text.split()).lower()

    assert "Darcy seepage" in validation_text
    assert "Validation scope" in validation_text
    assert "USACE EM 1110-2-2200" in validation_text
    assert "USACE EM 1110-2-1914" in validation_text
    assert "USACE Digital Library historical record" in validation_text
    assert "https://usace.contentdm.oclc.org/digital/collection/p16021coll9/id/2595/" in validation_text
    assert "USACE ERDC/GSL TR-25-5" in validation_text
    assert "usace-relief-wells-for-dams-and-levees-history-and-current-practice" in validation_text
    assert "USBR Design Standards No. 13" in validation_text
    assert "FERC Engineering Guidelines" in validation_text
    assert "FEMA P-1032" in validation_text
    assert "fema_p1032_eval_monitoring_seepage_internal_erosion.pdf" in validation_text
    assert "plausibility and limiting-case controls" in validation_text
    assert "Darcy/FDM uplift surrogate" in validation_text
    assert "not a design-code drain-efficiency recommendation" in normalized_validation_text
    assert "maintenance/rehabilitation" in normalized_validation_text
    assert "pressure-relief surrogate" in source_text
    assert "not a detailed drain-hole network" in " ".join(source_text.split())
    assert "not a code-compliant dam-foundation design calculation" in source_text
    assert "not calibrated external validation" in normalized_validation_text
    assert "rejects malformed arrays" in normalized_validation_text
    assert "before sparse FDM assembly" in validation_text
    assert "cannot extrapolate beyond the declared uplift-surrogate bounds" in normalized_validation_text
    assert "collector-pipe/header layouts" in normalized_validation_text
    assert "relief-well spacing/sizing" in normalized_validation_text
    assert "rehabilitation planning" in normalized_validation_text
    assert "site-specific drains, galleries, collector pipes, relief wells" in normalized_validation_text
    assert "development/pump-testing" in normalized_validation_text
    assert "operation, and maintenance workflows" in normalized_validation_text
    assert "maintenance/repair practice" in normalized_validation_text
    assert "seepage-remediation use" in normalized_validation_text
    assert "visual observation, geophysical investigation, seepage collection" in normalized_validation_text
    assert "flow-rate, pressure, and water-quality monitoring" in normalized_validation_text
    assert "life-cycle monitoring, emergency response, and remediation decisions" in normalized_validation_text
    assert "not a calibrated seepage-discharge" in source_text
    assert "uplift force and midpoint uplift-pressure intensity" in source_text
    assert "uplift force / moment" not in source_text
    assert "exit-gradient" in source_text
    assert "piping" in source_text
    assert "dam-stability design model" in source_text
    for phrase in (
        "seepage discharge",
        "exit gradients",
        "piping/internal erosion",
        "unsaturated flow",
        "transient reservoir operation",
        "anisotropy",
        "fractures/joints",
        "three-dimensional abutment flow",
        "seepage velocities",
        "full pore-pressure-field reporting",
        "flow-net construction",
        "piezometer or instrumentation interpretation",
        "permeability-uncertainty sensitivity studies",
        "mesh-convergence evidence",
        "drainage-gallery hydraulics",
        "collector-pipe / header layouts",
        "relief-well systems",
        "relief-well spacing/sizing",
        "relief-well design/construction/maintenance",
        "relief-well maintenance/repair current-practice review",
        "development/pump-testing",
        "filter design",
        "rehabilitation planning",
        "dam-safety monitoring/inspection program",
        "maintenance/rehabilitation",
        "seepage-control remediation selection",
        "fema p-1032 seepage/internal-erosion evaluation",
        "flow-rate/",
        "pressure/water-quality measurement",
        "emergency-response",
        "loading-likelihood",
        "system-response",
        "failure-consequence",
        "pfm/pfma",
        "surveillance",
        "risk-management workflows",
        "structural sliding/overturning stability checks",
    ):
        assert phrase in normalized_source_text


def test_reviewer_markdown_note_keeps_dam_seepage_p2_scope_boundary():
    note_text = Path("tests/_cross_validation/dam_seepage_6d.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(note_text.split())

    required_phrases = [
        "P2 plausibility-only",
        "No response-surface change",
        "Two-dimensional steady Darcy/FDM uplift-pressure surrogate",
        "single pressure-relief drain node",
        "not a calibrated seepage-discharge model",
        "not instrumented uplift validation",
        "not a flow-net validation",
        "not relief-well design validation",
        "not a dam-safety RIDM/PFM workflow",
        "https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2200.pdf",
        "https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-1901.pdf",
        "https://damsafety.org/content/seepage-analysis-and-control-dams",
        "usace-relief-wells-for-dams-and-levees-history-and-current-practice",
        "https://www.ferc.gov/sites/default/files/2020-04/chap3.pdf",
        "https://www.ferc.gov/dam-safety-and-inspections/risk-informed-decision-making-ridm",
        "fema_p1032_eval_monitoring_seepage_internal_erosion.pdf",
        "flow-rate, pressure, and water-quality monitoring",
        "life-cycle monitoring",
    ]
    for phrase in required_phrases:
        assert phrase in normalized

    for forbidden_phrase in [
        "Evidence tier | P1",
        "Evidence tier | P0",
        "Evidence tier | externally validated",
    ]:
        assert forbidden_phrase not in normalized


def test_ferc_uplift_baseline_and_instrumentation_boundary_are_declared():
    validation_text = Path("tests/_cross_validation/dam_seepage_6d.py").read_text(
        encoding="utf-8"
    )
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    row = next(
        line
        for line in audit_text.splitlines()
        if "`dam_seepage_6d`" in line and "FERC" in line
    )

    assert "straight-line headwater-to-tailwater distribution" in validation_text
    assert "Any real-world claim of uplift reduction would require instrumentation" in validation_text
    assert "Evidence tier remains P2" in row
    assert "instrumentation" in row
    assert "straight-line" in row


def test_web_audit_records_dam_seepage_official_source_recheck():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `dam_seepage_6d` 2026-06-08 official-source re-check" in audit_text
    assert "public USACE/FERC/USBR sources were re-opened as seepage/uplift scope evidence" in normalized
    assert "not as validation data for the registered response surface" in normalized
    assert "USACE EM 1110-2-1901 supports seepage-analysis/control context" in normalized
    assert "foundation and abutment seepage" in normalized
    assert "excessive uplift pressure" in normalized
    assert "piping, erosion" in normalized
    assert "USBR Design Standards No. 13 Chapter 8 supports the uncertainty boundary" in normalized
    assert "permeability and foundation conditions" in normalized
    assert "rather than dam-specific calibration" in normalized
    assert "FERC Chapter 3 remains the key boundary source" in normalized
    assert "verified by project instrumentation" in normalized
    assert "straight-line headwater-to-tailwater uplift should not be reduced" in normalized
    assert "two-dimensional Darcy/FDM uplift-pressure stress test" in normalized
    assert "single pressure-relief drain node" in normalized
    assert "grid-aligned curtain mask" in normalized
    assert "computes total uplift force and midpoint uplift-pressure intensity" in normalized
    for omitted in (
        "not a moment response",
        "sliding/overturning stability margin",
        "seepage discharge",
        "exit gradient",
        "piping/internal-erosion index",
        "instrumentation interpretation",
        "relief-well design",
        "filter design",
        "code-compliant dam-foundation design calculation",
    ):
        assert omitted in normalized
    assert "collector-pipe/gallery layout design" in normalized
    assert "relief-well spacing/sizing" in normalized
    assert "development/pump-testing" in normalized
    assert "rehabilitation-planning" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert "Full resweep is still required" in normalized
    assert (
        "Additional `dam_seepage_6d` 2026-06-08 public-scope boundary tightening:"
        in normalized
    )
    assert "maintenance/rehabilitation" in normalized
    assert "instrumented, rehabilitation-planning, or seepage-remediation design calculation" in normalized
    assert (
        "no additional numerical response resweep is needed solely for this official-source re-check"
        in normalized.lower()
    )
    assert "FERC risk/PFM boundary note" in audit_text
    assert "loading likelihood" in normalized
    assert "system response" in normalized
    assert "failure consequences" in normalized
    assert "potential failure modes" in normalized
    assert "performance monitoring" in normalized
    assert "surveillance" in normalized
    assert "risk-reduction measures" in normalized
    assert "seepage velocity fields" in normalized
    assert "full pore-pressure fields" in normalized
    assert "PFM/PFMA classifications" in normalized
    assert "risk-management decisions" in normalized
    assert "No numerical response resweep is needed solely for this risk/PFM boundary note" in normalized
    assert (
        "Additional `dam_seepage_6d` 2026-06-08 USACE relief-well publication-record "
        "confirmation"
        in normalized
    )
    assert "USACE Digital Library record" in normalized
    assert "public traceability anchors" in normalized
    assert "real relief-well systems involve siting, spacing, hydraulic design" in normalized
    assert "They do not validate the benchmark's single pressure-relief node" in normalized
    assert (
        "No numerical response resweep is needed solely for this publication-record "
        "confirmation"
        in normalized
    )
    assert "Additional `dam_seepage_6d` 2026-06-09 USACE-ERDC relief-well current-practice boundary note" in normalized
    assert "real relief-well maintenance/repair" in normalized
    assert "site-specific seepage-remediation design" in normalized
    assert "No numerical response resweep is needed solely for this ERDC source-boundary note" in normalized
    assert "Additional `dam_seepage_6d` 2026-06-09 FEMA P-1032 seepage/internal-erosion boundary note" in normalized
    assert "visual observation" in normalized
    assert "geophysical investigation" in normalized
    assert "seepage collection" in normalized
    assert "water-quality monitoring" in normalized
    assert "life-cycle monitoring" in normalized
    assert "emergency response" in normalized
    assert "internal-erosion or remediation workflow" in normalized
    assert "No numerical response resweep is needed solely for this FEMA P-1032 boundary note" in normalized
    assert (
        "Additional `dam_seepage_6d` 2026-06-09 homogeneous-Darcy limiting-control note"
        in normalized
    )
    assert "constant-coefficient Laplace head profile is linear" in normalized
    assert "independent of foundation depth" in normalized
    assert "independent of the absolute permeability scale" in normalized
    assert "first-principles Darcy/FDM limiting check" in normalized
    assert "not calibration evidence for a dam-specific permeability field" in normalized
    assert "mesh-converged seepage model" in normalized
    assert "code-compliant foundation design calculation" in normalized
    assert (
        "No numerical response resweep is needed solely for this test/documentation tightening"
        in normalized
    )
    assert (
        "Additional `dam_seepage_6d` 2026-06-09 current-page spot check"
        in normalized
    )
    assert "official/public pages for USACE EM 1110-2-1901" in normalized
    assert "USACE EM 1110-2-1914" in normalized
    assert "USBR Design Standards No. 13 Chapter 8" in normalized
    assert "FERC Engineering Guidelines and RIDM/PFM pages" in normalized
    assert "FEMA P-1032" in normalized
    assert "USACE ERDC/GSL TR-25-5" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert "Do not describe this case as externally validated" in normalized
    assert "instrument-verified" in normalized
    assert "relief-well-system validated" in normalized
    assert "FEMA P-1032 monitoring validated" in normalized
    assert "FERC RIDM/PFM validated" in normalized
    assert "code-compliant dam-foundation design" in normalized
    assert "No numerical response resweep is needed solely for this current-page spot check" in normalized


def test_uplift_reduction_factor_is_clipped_to_physical_upper_bound():
    case = DamSeepage6D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    eta = Y[:, 2]
    assert np.nanmin(eta) >= case.meta.output_variables[2].expected_range[0]
    assert np.nanmax(eta) <= case.meta.output_variables[2].expected_range[1]


def test_upstream_and_downstream_head_raise_uplift_loads():
    case = DamSeepage6D()
    upstream_sweep = np.array(
        [
            [30.0, 5.0, 1e-5, 1e-8, 30.0, 0.3],
            [80.0, 5.0, 1e-5, 1e-8, 30.0, 0.3],
        ],
        dtype=float,
    )
    downstream_sweep = np.array(
        [
            [60.0, 0.0, 1e-5, 1e-8, 30.0, 0.3],
            [60.0, 10.0, 1e-5, 1e-8, 30.0, 0.3],
        ],
        dtype=float,
    )

    upstream_y = case.evaluate(case.normalize(upstream_sweep))
    downstream_y = case.evaluate(case.normalize(downstream_sweep))

    assert upstream_y[1, 0] > upstream_y[0, 0]
    assert upstream_y[1, 1] > upstream_y[0, 1]
    assert downstream_y[1, 0] > downstream_y[0, 0]
    assert downstream_y[1, 1] > downstream_y[0, 1]


def test_homogeneous_no_drain_case_recovers_linear_uplift_control():
    case = DamSeepage6D()
    H1 = 60.0
    H2 = 10.0
    K = 1e-7
    D_base = 30.0
    U_total, U_center, eta = case._seepage(
        H1=H1,
        H2=H2,
        K_base=K,
        K_curtain=K,
        D_base=D_base,
        drain_eff=0.0,
    )

    L_base = case._L_BASE
    gamma_w = 9.81
    expected_total = gamma_w * (H1 + H2) / 2.0 * L_base / 1000.0
    expected_center = gamma_w * (H1 + H2) / 2.0 / 1000.0

    assert np.isclose(U_total, expected_total, rtol=1e-10, atol=1e-10)
    assert np.isclose(U_center, expected_center, rtol=1e-10, atol=1e-10)
    assert np.isclose(eta, 1.0, rtol=1e-10, atol=1e-10)


def test_homogeneous_no_drain_case_has_linear_base_head_profile():
    case = DamSeepage6D()
    H1 = 60.0
    H2 = 10.0
    K = 1e-7

    x_nodes, H_base = case._base_head_profile(
        H1=H1,
        H2=H2,
        K_base=K,
        K_curtain=K,
        D_base=30.0,
        drain_eff=0.0,
    )

    expected = H1 + (H2 - H1) * x_nodes / case._L_BASE
    assert np.allclose(H_base, expected, rtol=1e-10, atol=1e-10)


def test_homogeneous_no_drain_uplift_is_independent_of_absolute_permeability_scale():
    case = DamSeepage6D()
    low_k = case._seepage(60.0, 10.0, 1e-7, 1e-7, 30.0, 0.0)
    high_k = case._seepage(60.0, 10.0, 1e-4, 1e-4, 30.0, 0.0)

    assert np.allclose(high_k, low_k, rtol=1e-10, atol=1e-10)


def test_declared_curtain_width_selects_the_current_single_grid_column():
    case = DamSeepage6D()
    dx = case._L_BASE / (case._N_X - 1)
    x_nodes = np.linspace(0.0, case._L_BASE, case._N_X)
    curtain_half_width = 0.5 * max(case._CURTAIN_W, dx)
    curtain_mask = np.abs(x_nodes - case._CURTAIN_X) <= curtain_half_width
    curtain_mask[0] = False
    curtain_mask[-1] = False

    assert curtain_mask.sum() == 1
    assert np.isclose(x_nodes[curtain_mask][0], 4.7368421052631575)


def test_center_uplift_reports_exact_midpoint_not_nearest_grid_node():
    case = DamSeepage6D()
    H1 = 70.0
    H2 = 4.0
    K = 1e-7
    _, U_center, _ = case._seepage(
        H1=H1,
        H2=H2,
        K_base=K,
        K_curtain=K,
        D_base=30.0,
        drain_eff=0.0,
    )

    gamma_w = 9.81
    exact_midpoint = gamma_w * (H1 + H2) / 2.0 / 1000.0
    nearest_node_x = (case._N_X // 2) * case._L_BASE / (case._N_X - 1)
    nearest_node_value = gamma_w * (H1 + (H2 - H1) * nearest_node_x / case._L_BASE) / 1000.0

    assert np.isclose(U_center, exact_midpoint, rtol=1e-10, atol=1e-10)
    assert not np.isclose(U_center, nearest_node_value, rtol=1e-4, atol=1e-8)


def test_homogeneous_no_drain_control_is_independent_of_foundation_depth():
    case = DamSeepage6D()
    shallow = case._seepage(60.0, 10.0, 1e-7, 1e-7, 10.0, 0.0)
    deep = case._seepage(60.0, 10.0, 1e-7, 1e-7, 50.0, 0.0)

    assert np.allclose(shallow, deep, rtol=1e-10, atol=1e-10)


def test_positive_drain_efficiency_activates_relief_when_it_lowers_natural_head():
    case = DamSeepage6D()
    no_drain = case._seepage(60.0, 5.0, 1e-7, 1e-7, 30.0, 0.0)
    active_drain = case._seepage(60.0, 5.0, 1e-7, 1e-7, 30.0, 0.2)

    assert np.all(np.isfinite(no_drain))
    assert np.all(np.isfinite(active_drain))
    assert active_drain[0] < no_drain[0]
    assert active_drain[1] < no_drain[1]


def test_drain_surrogate_never_recharges_strong_curtain_cases():
    case = DamSeepage6D()

    # In this strong-curtain slice the natural no-drain head at the drain
    # location is already below the small-positive relief-head prescription.
    # The drain must remain inactive rather than acting as a recharge source.
    no_drain = case._seepage(
        H1=76.97877211075031,
        H2=0.825299559346111,
        K_base=1.1270562616206799e-05,
        K_curtain=3.5927066258752296e-08,
        D_base=12.145110372205163,
        drain_eff=0.0,
    )
    weak_positive_drain = case._seepage(
        H1=76.97877211075031,
        H2=0.825299559346111,
        K_base=1.1270562616206799e-05,
        K_curtain=3.5927066258752296e-08,
        D_base=12.145110372205163,
        drain_eff=0.05,
    )

    assert np.allclose(weak_positive_drain, no_drain, rtol=1e-12, atol=1e-12)


def test_more_permeable_curtain_increases_uplift_factor():
    case = DamSeepage6D()
    X_phys = np.array(
        [
            [60.0, 5.0, 1e-5, 1e-9, 30.0, 0.3],
            [60.0, 5.0, 1e-5, 1e-8, 30.0, 0.3],
            [60.0, 5.0, 1e-5, 1e-7, 30.0, 0.3],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.all(np.diff(Y[:, 0]) > 0.0)
    assert np.all(np.diff(Y[:, 2]) > 0.0)


def test_higher_positive_drain_efficiency_reduces_uplift():
    case = DamSeepage6D()
    X_phys = np.array(
        [
            [60.0, 5.0, 1e-5, 1e-7, 30.0, 0.2],
            [60.0, 5.0, 1e-5, 1e-7, 30.0, 0.8],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert Y[1, 0] < Y[0, 0]
    assert Y[1, 1] < Y[0, 1]
    assert Y[1, 2] < Y[0, 2]


def test_drain_efficiency_sweep_is_nonincreasing_after_recharge_guard():
    case = DamSeepage6D()
    fixed_slices = [
        [60.0, 5.0, 1e-5, 1e-7, 30.0],
        [80.0, 1.0, 1e-5, 3e-8, 15.0],
        [40.0, 8.0, 1e-6, 1e-8, 45.0],
    ]

    for fixed in fixed_slices:
        X_phys = np.array([fixed + [eff] for eff in np.linspace(0.0, 0.8, 9)])
        U_total = case.evaluate(case.normalize(X_phys))[:, 0]
        assert np.all(np.diff(U_total) <= 1e-10)


def test_sparse_solver_failure_is_not_silent_nan(monkeypatch):
    case = DamSeepage6D()

    def fail_solve(**_kwargs):
        raise ValueError("simulated singular sparse system")

    monkeypatch.setattr(case, "_seepage", fail_solve)
    X_phys = np.array([[60.0, 5.0, 1e-5, 1e-8, 30.0, 0.3]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN or nonphysical"):
        case._evaluate_core(X_phys)


def test_nonfinite_or_nonphysical_uplift_response_is_not_silent_nan(monkeypatch):
    case = DamSeepage6D()
    X_phys = np.array([[60.0, 5.0, 1e-5, 1e-8, 30.0, 0.3]], dtype=float)

    def invalid_nonfinite(*_args, **_kwargs):
        return np.array([np.nan, 0.4, 0.8])

    monkeypatch.setattr(case, "_seepage", invalid_nonfinite)
    with pytest.raises(RuntimeError, match="refusing silent NaN or nonphysical"):
        case._evaluate_core(X_phys)

    def invalid_nonphysical(*_args, **_kwargs):
        return np.array([-1.0, 0.4, 0.8])

    monkeypatch.setattr(case, "_seepage", invalid_nonphysical)
    with pytest.raises(RuntimeError, match="refusing silent NaN or nonphysical"):
        case._evaluate_core(X_phys)


def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_fdm(monkeypatch):
    case = DamSeepage6D()
    assert hasattr(case, "_validate_physical_inputs")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("sparse seepage solve should not be called for invalid input")

    monkeypatch.setattr(case, "_seepage", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._evaluate_core(np.array([60.0, 5.0, 1e-5, 1e-8, 30.0, 0.3], dtype=float))

    invalid_rows = [
        [19.9, 5.0, 1e-5, 1e-8, 30.0, 0.3],
        [100.1, 5.0, 1e-5, 1e-8, 30.0, 0.3],
        [60.0, -0.1, 1e-5, 1e-8, 30.0, 0.3],
        [60.0, 10.1, 1e-5, 1e-8, 30.0, 0.3],
        [60.0, 5.0, 0.99e-7, 1e-8, 30.0, 0.3],
        [60.0, 5.0, 1.01e-4, 1e-8, 30.0, 0.3],
        [60.0, 5.0, 1e-5, 0.99e-9, 30.0, 0.3],
        [60.0, 5.0, 1e-5, 1.01e-7, 30.0, 0.3],
        [60.0, 5.0, 1e-5, 1e-8, 9.9, 0.3],
        [60.0, 5.0, 1e-5, 1e-8, 50.1, 0.3],
        [60.0, 5.0, 1e-5, 1e-8, 30.0, -0.01],
        [60.0, 5.0, 1e-5, 1e-8, 30.0, 0.81],
        [60.0, 5.0, np.nan, 1e-8, 30.0, 0.3],
    ]
    for row in invalid_rows:
        with pytest.raises(
            ValueError,
            match="H1/H2/K_base/K_curtain/D_base/drain_eff benchmark domain",
        ):
            case._evaluate_core(np.array([row], dtype=float))
