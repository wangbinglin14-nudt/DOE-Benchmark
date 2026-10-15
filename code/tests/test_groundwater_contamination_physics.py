"""Physics regression tests for ``groundwater_contamination_6d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc, erfcx

from doe_testbed.cases.geomechanics.groundwater_contamination_6d import (
    GroundwaterContamination6D,
)


def _evaluate(case: GroundwaterContamination6D, rows: list[list[float]]) -> np.ndarray:
    X = np.array(rows, dtype=float)
    return case.evaluate(case.normalize(X))


def _domain_corners(case: GroundwaterContamination6D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_metadata_declares_retardation_units_and_surrogate_scope():
    case = GroundwaterContamination6D()
    equation = case.meta.governing_equation
    details = case.meta.solver_details.lower()
    kd = next(var for var in case.meta.input_variables if var.name == "K_d")
    references = "\n".join(case.meta.physics_references)

    assert kd.unit == "L/kg"
    assert "R = 1 + ρ_b·K_d/(1000·n)" in equation
    assert "separable decay attenuation" in equation
    assert "low-order" in details
    assert "not the exact decaying constant-source ade solution" in details
    assert "not a calibrated" in details
    assert "no numerical error" not in details
    assert case.meta.effective_dim == 6
    assert set(case.meta.important_variables) == {
        "v_pore",
        "alpha_L",
        "K_d",
        "rho_b",
        "t_half",
        "n_por",
    }
    assert set(case.meta.interaction_pairs) == {
        ("v_pore", "alpha_L"),
        ("v_pore", "K_d"),
        ("v_pore", "t_half"),
        ("v_pore", "n_por"),
        ("alpha_L", "K_d"),
        ("alpha_L", "t_half"),
        ("K_d", "rho_b"),
        ("K_d", "n_por"),
        ("K_d", "t_half"),
        ("rho_b", "n_por"),
        ("rho_b", "t_half"),
        ("t_half", "n_por"),
    }
    assert case.meta.is_linear is False
    assert "strong_interaction" in case.meta.response_topology
    assert "threshold_jump" in case.meta.response_topology
    assert "discontinuous" in case.meta.response_topology
    assert hasattr(case, "_validate_physical_inputs")
    assert "named _validate_physical_inputs" in details
    assert "rejects malformed arrays" in details
    assert "before ogata-banks surrogate evaluation" in details
    assert "out-of-contract c_max_frac/x_front/t_arrive_10m" in details
    assert "https://pubs.usgs.gov/pp/0411a/report.pdf" in references
    assert "https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html" in references
    assert "MT3D-USGS" in references
    assert "https://pubs.usgs.gov/tm/06/a53/tm6a53.pdf" in references
    assert (
        "https://www.usgs.gov/software/mt3d-usgs-groundwater-solute-transport-simulator-modflow"
        in references
    )
    assert (
        "https://www.usgs.gov/publications/mt3dms-model-use-calibration-and-validation"
        in references
    )
    assert "https://pubs.usgs.gov/publication/tm6A61" in references
    assert (
        "https://wwwbrr.cr.usgs.gov/projects/GWC_coupled/phreeqc.v1/html/phqc_69.htm"
        in references
    )
    assert "PHREEQC-equivalent reactive-transport chemistry" in references
    assert "phreeqc-equivalent reactive-transport chemistry" in details
    assert (
        "https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model"
        in references
    )
    assert "https://modflow6.readthedocs.io/en/latest/_mf6io/index.html" in references
    assert "GWF/GWT/GWE/PRT model types" in references
    assert "particle tracking" in references
    assert "heat/energy transport" in references
    assert "Modular coupled groundwater-flow/transport platform context only" in references
    assert (
        "https://www.usgs.gov/programs/environmental-health-program/science/groundwater-quality-principal-aquifers-united-states"
        in references
    )
    for epa_url in (
        "https://www.epa.gov/superfund/how-superfund-addresses-groundwater-contamination",
        "https://www.epa.gov/superfund/groundwater-technologies",
        "https://www.epa.gov/superfund/selecting-groundwater-remedy",
        "https://cfpub.epa.gov/si/si_public_record_report.cfm?Lab=NRMRL&dirEntryId=187788",
        "https://www.epa.gov/land-research/ground-water-modeling-research",
        "https://www.epa.gov/remedytech/remediation-technologies-cleaning-contaminated-sites",
    ):
        assert epa_url in references
    assert "plume containment" in references
    assert "monitored-natural-attenuation evaluation" in references
    assert "capture-zone" in references
    assert "Ground Water Modeling Research" in references
    assert "exposure-pathway protection" in details
    assert "pump-and-treat" in details
    assert "monitored natural attenuation" in details
    assert "capture-zone analysis" in details
    assert "converging lines of evidence" in details
    assert "permeable reactive barriers" in details
    assert "in situ chemical oxidation/reduction" in details
    assert "soil vapor extraction/air sparging" in details
    assert "vapor intrusion mitigation" in details


def test_validation_material_declares_low_order_transport_surrogate_scope():
    validation_text = Path(
        "tests/_cross_validation/groundwater_contamination_6d.py"
    ).read_text(encoding="utf-8")
    normalized_validation = " ".join(validation_text.split())
    source_text = Path(
        "doe_testbed/cases/geomechanics/groundwater_contamination_6d.py"
    ).read_text(encoding="utf-8")
    normalized_validation = " ".join(validation_text.split())

    assert "Ogata-Banks identity" in validation_text
    assert "low-order analytical transport surrogate" in validation_text
    assert "separable decay attenuation" in validation_text
    assert "not the exact decaying constant-source ade solution" in normalized_validation.lower()
    assert "x_front > 10 m after 1 year" in validation_text
    assert "not placed on C_max_frac" in validation_text
    assert "not a calibrated site-specific reactive-transport model" in normalized_validation.lower()
    assert "USGS PP-411-A public PDF" in validation_text
    assert "USGS Circular 1224" in validation_text
    assert "system boundaries, flow-pattern uncertainty, and pumping stresses" in normalized_validation
    assert "irrigation, pumping, artificial recharge, and drainage" in normalized_validation
    assert "one-dimensional constant-average-velocity surrogate with fixed source conditions" in normalized_validation
    assert "EPA Superfund groundwater-remedy and technology pages" in validation_text
    assert "exposure-pathway protection" in normalized_validation
    assert "plume containment" in normalized_validation
    assert "monitored natural attenuation" in normalized_validation
    assert "capture-zone evaluation" in normalized_validation
    assert "Superfund remedy-selection, hydraulic-containment, capture-zone, MNA" in validation_text
    assert "not a calibrated" in source_text
    assert "site-specific reactive-transport model" in source_text
    assert "rejects malformed arrays" in normalized_validation
    assert "before Ogata-Banks surrogate evaluation" in validation_text
    assert "cannot extrapolate beyond the declared 1-D ADE surrogate bounds" in normalized_validation
    assert "Target accuracy" not in validation_text
    assert "external validation" not in validation_text


def test_reviewer_markdown_note_keeps_groundwater_p2_scope_boundary():
    note_text = Path("tests/_cross_validation/groundwater_contamination_6d.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(note_text.split())

    required_phrases = [
        "P2 plausibility-only",
        "No response-surface change",
        "One-dimensional Ogata-Banks-style",
        "separable first-order decay",
        "not field-plume calibrated",
        "not MT3DMS/MT3D-USGS/MODFLOW-6-equivalent",
        "not MODFLOW-coupled 3-D reactive transport",
        "not multispecies reaction transport",
        "not site-specific groundwater vulnerability assessment",
        "not mesh/time-step converged simulator output",
        "not a validated contaminant-remediation model",
        "https://pubs.usgs.gov/pp/0411a/report.pdf",
        "https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html",
        "https://www.usgs.gov/software/mt3d-usgs-groundwater-solute-transport-simulator-modflow",
        "https://pubs.usgs.gov/publication/tm6A61",
        "https://modflow6.readthedocs.io/en/latest/mf6io.html",
        "USGS PHREEQC advection-reaction-dispersion equation",
        "longitudinal hydrodynamic dispersion coefficient",
        "https://www.epa.gov/superfund/how-superfund-addresses-groundwater-contamination",
        "https://www.epa.gov/superfund/groundwater-technologies",
        "https://www.epa.gov/superfund/selecting-groundwater-remedy",
        "https://www.epa.gov/land-research/ground-water-modeling-research",
        "https://www.epa.gov/remedytech/remediation-technologies-cleaning-contaminated-sites",
        "Superfund remedy-selection",
        "capture-zone evaluation",
        "permeable reactive barriers",
        "vapor intrusion mitigation",
    ]
    for phrase in required_phrases:
        assert phrase in normalized

    for forbidden_phrase in [
        "Evidence tier | P1",
        "Evidence tier | P0",
        "Evidence tier | externally validated",
    ]:
        assert forbidden_phrase not in normalized


def test_metadata_declares_omitted_reactive_transport_scope():
    case = GroundwaterContamination6D()
    validation_text = Path(
        "tests/_cross_validation/groundwater_contamination_6d.py"
    ).read_text(encoding="utf-8")
    source_text = Path(
        "doe_testbed/cases/geomechanics/groundwater_contamination_6d.py"
    ).read_text(encoding="utf-8")
    combined = "\n".join(
        [
            case.meta.solver_details,
            case.meta.scenario_notes["VS-C"],
            validation_text,
            source_text,
        ]
    ).lower()

    assert (
        "https://www.usgs.gov/publications/mt3dms-model-use-calibration-and-validation"
        in validation_text
    )
    assert (
        "https://www.usgs.gov/software/mt3d-usgs-groundwater-solute-transport-simulator-modflow"
        in validation_text
    )
    assert "https://pubs.usgs.gov/publication/tm6A61" in validation_text
    assert (
        "https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model"
        in validation_text
    )
    assert "three-dimensional, GWF-coupled" in validation_text
    assert "control-volume finite-difference transport model" in validation_text
    for phrase in (
        "2-d/3-d flow",
        "heterogeneous aquifer",
        "transient groundwater flow",
        "source depletion",
        "finite-duration release",
        "site-specific boundary condition",
        "well/pumping",
        "modflow-coupled flow",
        "spatially variable hydraulic",
        "recharge/evapotranspiration",
        "variable water table",
        "transverse or vertical dispersion",
        "nonlinear or nonequilibrium sorption",
        "dual-domain mass transfer",
        "immobile-domain storage",
        "matrix diffusion",
        "multispecies reaction",
        "density-dependent flow",
        "biodegradation kinetics beyond",
        "geochemical speciation",
        "source-strength uncertainty",
        "numerical mesh/time-step convergence",
        "field-plume calibration",
        "advanced stress-package transport",
        "source/sink mixing",
        "direct solute-mass loading",
        "unsaturated-zone transport",
        "stream/lake solute exchange",
        "parent-daughter reaction",
        "pump-and-treat recirculation",
        "transient mass-storage",
        "mobile/immobile-domain kd",
        "mt3dms-style",
        "mt3dms/mt3d-usgs/modflow 6 gwt package-level transport simulation",
        "gwf/gwt/gwe/prt model types",
        "hydrologic stress packages",
        "particle tracking",
        "mover routing",
        "groundwater energy transport",
        "heat/energy transport workflows",
        "flow-pattern uncertainty",
        "irrigation, pumping, artificial recharge, and drainage",
        "exposure-pathway protection",
        "plume containment",
        "pump-and-treat",
        "in situ treatment",
        "vertical engineered barriers",
        "monitored natural attenuation",
        "permeable reactive barriers",
        "in situ chemical oxidation/reduction",
        "in situ thermal treatment",
        "phytotechnologies",
        "soil vapor extraction/air sparging",
        "solidification/stabilization",
        "vapor intrusion mitigation",
        "institutional controls",
        "alternative water supply",
        "remedy selection",
        "capture-zone analysis",
        "converging lines of evidence",
        "groundwater characterization/monitoring/treatment workflows",
        "one-dimensional constant-average-velocity surrogate",
    ):
        assert phrase in combined


def test_usgs_transport_models_are_boundary_context_not_validation_claims():
    validation_text = Path(
        "tests/_cross_validation/groundwater_contamination_6d.py"
    ).read_text(encoding="utf-8")
    source_text = Path(
        "doe_testbed/cases/geomechanics/groundwater_contamination_6d.py"
    ).read_text(encoding="utf-8")
    normalized_validation = " ".join(validation_text.split())
    normalized_source_text = " ".join(source_text.split())
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    row = next(
        line
        for line in audit_text.splitlines()
        if "`groundwater_contamination_6d`" in line and "MT3DMS" in line
    )

    assert "three-dimensional multi-species transport model" in validation_text
    assert "interfaces directly with MODFLOW" in validation_text
    assert "MODFLOW 6 flow support" in validation_text
    assert "structured grids only" in validation_text
    assert "broader modular GWF/GWT platform" in validation_text
    assert "simultaneous GWF/GWT operation" in validation_text
    assert "unstructured-grid transport support" in validation_text
    assert "GWF, GWT, GWE, and PRT model types" in normalized_validation
    assert "hydrologic stress packages such as wells, recharge, rivers" in normalized_validation
    assert "wells" in normalized_validation.lower()
    assert "particle tracking" in normalized_validation
    assert "groundwater heat/energy transport" in normalized_validation
    assert "modular groundwater-simulation platform" in source_text
    assert "does not reproduce package-level flow/transport" in normalized_source_text
    assert "pump-and-treat recirculation" in validation_text
    assert "separate mobile/immobile-domain Kd specification" in validation_text
    assert "multi-aquifer wells" in validation_text
    assert "water mover-connected packages" in validation_text.lower()
    assert "omitted-scope context, not validation data" in validation_text
    assert "Evidence tier remains P2" in row
    assert "not validation data" in row


def test_web_audit_records_groundwater_official_source_recheck():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `groundwater_contamination_6d` 2026-06-08 official-source re-check"
        in audit_text
    )
    assert "official USGS sources were re-opened as formula and simulator-boundary evidence" in normalized
    assert "not as field-plume or full-simulator validation data" in normalized
    assert "USGS PP-411-A supports the one-dimensional longitudinal-dispersion" in normalized
    assert "Ogata-Banks analytical formula anchor" in normalized
    assert "homogeneous porous medium with constant average velocity" in normalized
    assert "semi-infinite boundary assumptions" in normalized
    assert "linear retardation and a separable first-order decay attenuation" in normalized
    assert "not the exact decaying constant-source ADE solution" in normalized
    assert "USGS Circular 1224 supports the groundwater-flow" in normalized
    assert "average-linear-velocity context" in normalized
    assert "Official MT3D-USGS and MODFLOW 6 GWT material" in normalized
    assert "production solute-transport tools" in normalized
    for capability in (
        "MODFLOW-linked flow",
        "three-dimensional control-volume transport",
        "hydrodynamic dispersion",
        "immobile-domain transfer",
        "source/sink mixing",
        "direct solute-mass loading",
        "advanced package transport",
        "multi-aquifer wells",
        "water mover-connected packages",
        "unsaturated-zone transport",
        "stream/lake exchange",
        "reaction packages",
        "pump-and-treat recirculation",
        "transient mass-storage accounting",
    ):
        assert capability in normalized
    assert "Those workflows are not run by the benchmark" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    for forbidden_claim in (
        "field-plume calibrated",
        "MT3DMS/MT3D-USGS/MODFLOW-6-equivalent",
        "MODFLOW-coupled 3-D reactive transport",
        "multispecies reaction transport",
        "site-specific groundwater vulnerability assessment",
        "mesh/time-step converged simulator output",
        "validated contaminant-remediation model",
    ):
        assert forbidden_claim in normalized
    assert "Full resweep is still required" in normalized
    assert (
        "no additional numerical response resweep is needed solely for this official-source re-check"
        in normalized.lower()
    )
    assert (
        "Additional `groundwater_contamination_6d` 2026-06-08 USGS boundary-setting tightening"
        in normalized
    )
    assert "system boundaries, flow-pattern uncertainty, and pumping as controls on plume movement" in normalized
    assert "site-specific groundwater-vulnerability or field-plume transport model" in normalized
    assert "irrigation, pumping, artificial recharge, and drainage can alter groundwater flow and groundwater quality" in normalized


def test_web_audit_records_groundwater_advanced_package_boundary_note():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `groundwater_contamination_6d` 2026-06-08 advanced-package boundary re-check"
        in audit_text
    )
    assert (
        "official USGS MT3D-USGS software page still states MODFLOW 6 flow support for structured grids only"
        in normalized
    )
    assert (
        "MODFLOW 6 GWT documentation explicitly includes multi-aquifer wells and water mover-connected packages"
        in normalized
    )
    assert "Those advanced-package workflows remain omitted-scope boundary evidence only" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized


def test_web_audit_records_groundwater_modflow_platform_boundary_note():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `groundwater_contamination_6d` 2026-06-08 MODFLOW-platform boundary note"
        in audit_text
    )
    assert "broader modular groundwater-simulation platform with coupled GWF/GWT packages" in normalized
    assert "simultaneous GWF/GWT operation" in normalized
    assert "unstructured-grid transport support" in normalized
    assert "does not reproduce package-level flow/transport coupling" in normalized
    assert "mover routing" in normalized
    assert "advanced boundary-package workflows" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized


def test_web_audit_records_groundwater_modflow_io_platform_boundary_note():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `groundwater_contamination_6d` 2026-06-08 MODFLOW I/O platform boundary note"
        in normalized
    )
    assert "`GWF`, `GWT`, `GWE`, and `PRT` model types" in normalized
    assert "hydrologic stress packages such as wells, recharge, rivers, drains, lakes, multi-aquifer wells" in normalized
    assert "evapotranspiration, and storage" in normalized
    assert "mover routing" in normalized
    assert "particle tracking" in normalized
    assert "groundwater heat/energy transport" in normalized
    assert "does not run MODFLOW 6 stress packages" in normalized
    assert "platform-level GWF/GWT/GWE/PRT coupling" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert "No numerical response resweep is needed solely for this MODFLOW I/O platform boundary note" in normalized


def test_web_audit_records_groundwater_principal_aquifer_boundary_note():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `groundwater_contamination_6d` 2026-06-08 principal-aquifer boundary note"
        in audit_text
    )
    assert "official USGS principal-aquifer quality guidance states that irrigation, pumping, artificial recharge, and drainage can alter groundwater flow and groundwater quality" in normalized
    assert "fixed source conditions and constant average velocity" in normalized
    assert "not a site-specific groundwater-vulnerability or remediation-planning model" in normalized


def test_web_audit_records_groundwater_usgs_transport_capability_boundary_confirmation():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `groundwater_contamination_6d` 2026-06-08 USGS transport-model "
        "capability-boundary confirmation"
        in normalized
    )
    for phrase in (
        "official USGS MT3D-USGS software page",
        "MODFLOW 6 GWT documentation",
        "MODFLOW 6 input/output documentation",
        "principal-aquifer groundwater-quality page",
        "MODFLOW-linked flow",
        "structured-grid MODFLOW 6 flow support for MT3D-USGS",
        "unsaturated-zone transport",
        "stream/lake exchange",
        "reaction packages",
        "pump-and-treat recirculation",
        "transient mass-storage accounting",
        "three-dimensional control-volume GWT advection/dispersion/sorption",
        "mobile/immobile-domain transfer",
        "source/sink mixing",
        "advanced-package transport",
        "multi-aquifer wells",
        "Water Mover-connected packages",
        "irrigation, pumping, artificial recharge, and drainage",
        "omitted-scope boundary evidence only",
        "does not run MT3D-USGS",
        "MODFLOW 6 stress packages",
        "field calibration",
        "remediation workflow simulation",
        "site-vulnerability assessment",
        "Evidence tier remains P2 plausibility-only",
        "No numerical response resweep is needed solely for this capability-boundary confirmation",
    ):
        assert phrase in normalized


def test_web_audit_records_groundwater_epa_remedy_boundary_note():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `groundwater_contamination_6d` 2026-06-09 EPA groundwater-remedy boundary note"
        in audit_text
    )
    for phrase in (
        "EPA Superfund groundwater contamination",
        "Groundwater Technologies",
        "Selecting a Groundwater Remedy",
        "capture-zone evaluation",
        "Ground Water Modeling Research",
        "exposure-pathway protection",
        "plume containment",
        "pump-and-treat",
        "in situ treatment",
        "vertical engineered barriers",
        "monitored natural attenuation",
        "institutional controls",
        "alternative water supply",
        "remedy selection",
        "cleanup goals",
        "converging lines of evidence",
        "site-specific characterization",
        "monitoring/treatment",
        "source/plume remediation modeling",
        "not validation data for this one-dimensional Ogata-Banks-style surrogate",
        "Evidence tier remains P2 plausibility-only",
        "No numerical response resweep is needed solely for this EPA remedy-boundary note",
    ):
        assert phrase in normalized


def test_web_audit_records_groundwater_phreeqc_dispersion_boundary_note():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `groundwater_contamination_6d` 2026-06-09 USGS PHREEQC dispersion boundary note"
        in normalized
    )
    for phrase in (
        "official USGS PHREEQC advection-reaction-dispersion equation page",
        "longitudinal hydrodynamic-dispersion coefficient",
        "dispersivity term",
        "increasing `alpha_L`",
        "`D = alpha_L*v_pore + D_m*tau`",
        "broadens the 1% concentration threshold front",
        "advances first arrival at a 10 m monitoring point",
        "not validation data for the benchmark plume",
        "does not make this case PHREEQC-equivalent",
        "Evidence tier remains P2 plausibility-only",
        "No numerical response resweep is needed solely for this test/documentation tightening",
    ):
        assert phrase in normalized


def test_web_audit_records_groundwater_current_page_spot_check():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `groundwater_contamination_6d` 2026-06-09 current-page spot check:"
        in audit_text
    )
    for phrase in (
        "current USGS PP-411-A, MODFLOW 6 GWT, MT3D-USGS, and EPA Superfund groundwater-remedy pages",
        "one-dimensional longitudinal dispersion / Ogata-Banks formula lineage",
        "three-dimensional GWF-coupled transport",
        "control-volume finite difference discretization",
        "advection",
        "hydrodynamic dispersion",
        "sorption",
        "mobile/immobile-domain transfer",
        "decay/production",
        "source/sink mixing",
        "direct solute-mass loading",
        "streams, lakes, multi-aquifer wells",
        "unsaturated-zone transport",
        "Water Mover-connected packages",
        "implicit time integration",
        "iterative numerical solution",
        "MODFLOW-linked solute-transport workflows",
        "structured-grid MODFLOW 6 flow support",
        "pump-and-treat recirculation",
        "transient mass-storage accounting",
        "separate mobile/immobile-domain `K_d` specification",
        "exposure-pathway protection",
        "plume containment",
        "pump-and-treat",
        "in situ treatment",
        "vertical engineered barriers",
        "monitored natural attenuation",
        "institutional controls",
        "alternative water supply",
        "permeable reactive barriers",
        "soil vapor extraction/air sparging",
        "solidification/stabilization",
        "vapor intrusion mitigation",
        "site characterization/monitoring/remediation guidance",
        "one-dimensional fixed-source Ogata-Banks-style algebraic/threshold surrogate",
        "linear retardation and separable first-order decay attenuation",
        "Do not describe this case as field-plume calibrated",
        "MODFLOW 6 GWT validated",
        "MT3D-USGS validated",
        "MT3DMS validated",
        "PHREEQC validated",
        "Superfund-remedy validated",
        "pump-and-treat/capture-zone validated",
        "MNA validated",
        "cleanup-performance validated",
        "site-vulnerability validated",
        "mesh/time-step converged simulator output",
        "Evidence tier remains P2 plausibility-only",
        "No numerical response resweep is needed solely for this current-page spot check",
    ):
        assert phrase in normalized


def test_front_position_increases_with_pore_velocity():
    case = GroundwaterContamination6D()

    slow, fast = _evaluate(
        case,
        [
            [0.001, 0.1, 0.0, 1600.0, 1000.0, 0.35],
            [1.0, 0.1, 0.0, 1600.0, 1000.0, 0.35],
        ],
    )

    assert fast[1] > slow[1]


def test_failure_boundary_targets_ten_metre_front_not_source_maximum():
    case = GroundwaterContamination6D()
    cmax_output, xfront_output, tarrive_output = case.meta.output_variables

    assert cmax_output.failure_threshold is None
    assert xfront_output.failure_threshold == 10.0
    assert xfront_output.failure_direction == "above"
    assert tarrive_output.failure_threshold is None
    assert "x_front > 10 m" in case.meta.failure_condition

    safe, failed = _evaluate(
        case,
        [
            [0.001, 0.1, 0.0, 1600.0, 1000.0, 0.35],
            [1.0, 0.1, 0.0, 1600.0, 1000.0, 0.35],
        ],
    )

    assert safe[1] < xfront_output.failure_threshold
    assert failed[1] > xfront_output.failure_threshold


def test_adsorption_retards_arrival_at_ten_metres():
    case = GroundwaterContamination6D()

    mobile, adsorbed = _evaluate(
        case,
        [
            [0.1, 0.1, 0.0, 1600.0, 1000.0, 0.35],
            [0.1, 0.1, 10.0, 1600.0, 1000.0, 0.35],
        ],
    )

    assert adsorbed[1] < mobile[1]
    assert adsorbed[2] > mobile[2]


def test_longitudinal_dispersivity_broadens_threshold_front_and_earlier_arrival():
    case = GroundwaterContamination6D()

    low_dispersion, high_dispersion = _evaluate(
        case,
        [
            [0.1, 0.01, 0.0, 1600.0, 1000.0, 0.35],
            [0.1, 5.0, 0.0, 1600.0, 1000.0, 0.35],
        ],
    )

    assert high_dispersion[1] > low_dispersion[1]
    assert high_dispersion[2] < low_dispersion[2]
    assert np.isclose(high_dispersion[0], low_dispersion[0])


def test_retardation_factor_controls_front_distance_and_arrival_time():
    case = GroundwaterContamination6D()
    v_pore = 0.1
    alpha_L = 0.01
    K_d = 1.0
    rho_b = 1600.0
    n_por = 0.4
    mobile, adsorbed = _evaluate(
        case,
        [
            [v_pore, alpha_L, 0.0, rho_b, 1000.0, n_por],
            [v_pore, alpha_L, K_d, rho_b, 1000.0, n_por],
        ],
    )

    retardation = 1.0 + rho_b * K_d / (1000.0 * n_por)

    assert np.isclose(adsorbed[2] / mobile[2], retardation, rtol=0.08)
    assert np.isclose(mobile[1] / adsorbed[1], retardation, rtol=0.08)


def test_relative_concentration_matches_declared_ogata_banks_surrogate_formula():
    v = 0.1
    alpha_L = 0.5
    K_d = 1.0
    rho_b = 1600.0
    n_por = 0.4
    t_half = 500.0
    x = 10.0
    t = 180.0
    D_m = 1e-9 * 86400
    tau = 0.7
    D = alpha_L * v + D_m * tau
    R = 1.0 + rho_b * K_d / (1000.0 * n_por)
    lam = np.log(2.0) / t_half

    A1 = (R * x - v * t) / (2.0 * np.sqrt(D * R * t))
    A2 = (R * x + v * t) / (2.0 * np.sqrt(D * R * t))
    expected = 0.5 * (erfc(A1) + np.exp(v * x / D) * erfc(A2))
    expected *= np.exp(-lam * t / R)

    observed = GroundwaterContamination6D._relative_concentration(
        x=x,
        t=t,
        v=v,
        D=D,
        R=R,
        lam=lam,
    )

    assert np.isclose(observed, expected, rtol=1e-12, atol=1e-12)
    assert R == 5.0


def test_relative_concentration_uses_stable_reflection_term_without_exponent_cap():
    v = 0.001
    alpha_L = 0.01
    D_m = 1e-9 * 86400
    tau = 0.7
    D = alpha_L * v + D_m * tau
    R = 1.0
    lam = np.log(2.0) / 1000.0
    x = 500.0
    t = 365.0

    A1 = (R * x - v * t) / (2.0 * np.sqrt(D * R * t))
    A2 = (R * x + v * t) / (2.0 * np.sqrt(D * R * t))
    vx_D = v * x / D
    assert vx_D > 700.0

    expected_reflection = erfcx(A2) * np.exp(vx_D - A2 * A2)
    expected = 0.5 * (erfc(A1) + expected_reflection)
    expected *= np.exp(-lam * t / R)

    observed = GroundwaterContamination6D._relative_concentration(
        x=x,
        t=t,
        v=v,
        D=D,
        R=R,
        lam=lam,
    )

    assert np.isfinite(observed)
    assert np.isclose(observed, expected, rtol=1e-12, atol=1e-300)


def test_stable_reflection_term_changes_legal_threshold_edge_from_legacy_cap():
    case = GroundwaterContamination6D()

    Y = _evaluate(
        case,
        [[0.7445759117529528, 0.013722959274081546, 3.135493773661259, 1851.8882420139657, 947.7631123269712, 0.3284738467774386]],
    )[0]

    assert np.isclose(Y[1], 16.03206412825651)
    assert not np.isclose(Y[1], 15.030060120240481)


def test_unphysical_relative_concentration_is_not_silently_clipped(monkeypatch):
    def bogus_erfc(_value):
        return 3.0

    monkeypatch.setattr(
        "doe_testbed.cases.geomechanics.groundwater_contamination_6d.erfc",
        bogus_erfc,
    )

    with pytest.raises(FloatingPointError, match="physical"):
        GroundwaterContamination6D._relative_concentration(
            x=0.0,
            t=365.0,
            v=0.1,
            D=0.1,
            R=1.0,
            lam=0.0,
        )


def test_arrival_time_detects_transient_threshold_crossing_before_decay():
    case = GroundwaterContamination6D()

    mobile = _evaluate(
        case,
        [[0.1, 0.1, 0.0, 1600.0, 1000.0, 0.35]],
    )[0]

    assert mobile[2] < 365.0


def test_short_half_life_suppresses_contaminant_front():
    case = GroundwaterContamination6D()

    persistent, decayed = _evaluate(
        case,
        [
            [1.0, 0.1, 0.0, 1600.0, 1000.0, 0.35],
            [1.0, 0.1, 0.0, 1600.0, 10.0, 0.35],
        ],
    )

    assert decayed[0] < persistent[0]
    assert decayed[1] < persistent[1]


def test_declared_output_ranges_cover_domain_corners():
    case = GroundwaterContamination6D()
    Y = case.evaluate(case.normalize(_domain_corners(case)))

    assert np.all(np.isfinite(Y))
    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_scenario_notes_do_not_claim_source_maximum_is_velocity_driven():
    case = GroundwaterContamination6D()
    note = case.meta.scenario_notes["VS-A"]

    assert "x_front and t_arrive" in note
    assert "C_max_frac is a source-including maximum" in note
    assert "decay/retardation" in note
    assert case.meta.sobol_ST["v_pore"] == 0.000


def test_all_applicable_scenarios_have_honest_notes():
    case = GroundwaterContamination6D()

    assert set(case.meta.scenario_notes) == set(case.meta.applicable_scenarios)

    vs_c = case.meta.scenario_notes["VS-C"].lower()
    assert "ogata-banks-style" in vs_c
    assert "surrogate-accuracy" in vs_c
    assert "separable decay attenuation" in vs_c
    assert "not a calibrated" in vs_c
    assert "not" in vs_c and "exact decaying constant-source ade validation" in vs_c


def test_transport_calculation_failure_is_not_silent_nan(monkeypatch):
    case = GroundwaterContamination6D()

    def fail_transport(**_kwargs):
        raise ValueError("simulated transport calculation failure")

    monkeypatch.setattr(case, "_ogata_banks", fail_transport)
    X_phys = np.array([[0.1, 0.1, 0.0, 1600.0, 1000.0, 0.35]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_threshold_derived_front_metrics_are_declared_as_discontinuous():
    case = GroundwaterContamination6D()

    assert "threshold_jump" in case.meta.response_topology
    assert "discontinuous" in case.meta.response_topology

    X_phys = np.array(
        [
            [0.100, 0.1, 0.0, 1600.0, 1000.0, 0.35],
            [0.101, 0.1, 0.0, 1600.0, 1000.0, 0.35],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert Y[1, 1] >= Y[0, 1]
    assert np.isclose((Y[1, 1] - Y[0, 1]) % (500.0 / 499.0), 0.0, atol=1e-10)


def test_nonfinite_transport_response_is_not_silent_nan(monkeypatch):
    case = GroundwaterContamination6D()

    def invalid_transport(**_kwargs):
        return np.array([np.nan, 10.0, 100.0])

    monkeypatch.setattr(case, "_ogata_banks", invalid_transport)
    X_phys = np.array([[0.1, 0.1, 0.0, 1600.0, 1000.0, 0.35]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_out_of_contract_transport_response_is_not_archived(monkeypatch):
    case = GroundwaterContamination6D()
    X_phys = np.array([[0.1, 0.1, 0.0, 1600.0, 1000.0, 0.35]], dtype=float)

    invalid_responses = [
        np.array([-0.01, 10.0, 100.0]),
        np.array([1.01, 10.0, 100.0]),
        np.array([0.5, -1.0, 100.0]),
        np.array([0.5, 501.0, 100.0]),
        np.array([0.5, 10.0, 0.9]),
        np.array([0.5, 10.0, 10001.0]),
    ]

    for response in invalid_responses:
        monkeypatch.setattr(case, "_ogata_banks", lambda **_kwargs: response)
        with pytest.raises(RuntimeError, match="out-of-contract"):
            case._evaluate_core(X_phys)


def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_transport(monkeypatch):
    case = GroundwaterContamination6D()
    assert hasattr(case, "_validate_physical_inputs")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Ogata-Banks surrogate should not be called for invalid input")

    monkeypatch.setattr(case, "_ogata_banks", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._evaluate_core(np.array([0.1, 0.1, 0.0, 1600.0, 1000.0, 0.35], dtype=float))

    invalid_rows = [
        [0.00099, 0.1, 0.0, 1600.0, 1000.0, 0.35],
        [1.01, 0.1, 0.0, 1600.0, 1000.0, 0.35],
        [0.1, 0.0099, 0.0, 1600.0, 1000.0, 0.35],
        [0.1, 5.01, 0.0, 1600.0, 1000.0, 0.35],
        [0.1, 0.1, -0.01, 1600.0, 1000.0, 0.35],
        [0.1, 0.1, 10.01, 1600.0, 1000.0, 0.35],
        [0.1, 0.1, 0.0, 1399.0, 1000.0, 0.35],
        [0.1, 0.1, 0.0, 1901.0, 1000.0, 0.35],
        [0.1, 0.1, 0.0, 1600.0, 9.9, 0.35],
        [0.1, 0.1, 0.0, 1600.0, 1001.0, 0.35],
        [0.1, 0.1, 0.0, 1600.0, 1000.0, 0.19],
        [0.1, 0.1, 0.0, 1600.0, 1000.0, 0.51],
        [0.1, 0.1, 0.0, np.nan, 1000.0, 0.35],
    ]
    for row in invalid_rows:
        with pytest.raises(
            ValueError,
            match="v_pore/alpha_L/K_d/rho_b/t_half/n_por benchmark domain",
        ):
            case._evaluate_core(np.array([row], dtype=float))
