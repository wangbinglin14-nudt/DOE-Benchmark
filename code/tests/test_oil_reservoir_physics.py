"""Physics regression tests for ``oil_reservoir_9d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.geomechanics.oil_reservoir_9d import OilReservoir9D


def _domain_corners(case: OilReservoir9D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_metadata_declares_all_blocks_active_screening_surrogate_scope():
    case = OilReservoir9D()
    text = "\n".join(
        [
            case.__doc__ or "",
            case.meta.name,
            " ".join(case.meta.tags),
            case.meta.governing_equation,
            " ".join(case.meta.scenario_notes.values()),
        ]
    ).lower()

    assert case.meta.effective_dim == 9
    assert case.meta.important_variables == [
        "K11",
        "K12",
        "K13",
        "K21",
        "K22",
        "K23",
        "K31",
        "K32",
        "K33",
    ]
    assert all(case.meta.sobol_ST[name] > 0.13 for name in case.meta.important_variables)
    assert set(case.meta.interaction_pairs) == {
        ("K11", "K12"),
        ("K12", "K13"),
        ("K21", "K22"),
        ("K22", "K23"),
        ("K31", "K32"),
        ("K32", "K33"),
        ("K11", "K21"),
        ("K21", "K31"),
        ("K12", "K22"),
        ("K22", "K32"),
        ("K13", "K23"),
        ("K23", "K33"),
        ("K11", "K22"),
        ("K13", "K22"),
        ("K22", "K31"),
        ("K22", "K33"),
    }
    assert "all 9" in text
    assert "screening" in text
    assert "no pre-result ranking-success claim" in text
    assert "diagonal" not in text
    assert "sparse-effect" not in text
    assert "sparse_effect" not in text
    assert "sparse problem" not in text


def test_metadata_declares_single_phase_conductance_surrogate_scope():
    case = OilReservoir9D()
    output = next(var for var in case.meta.output_variables if var.name == "Q_total")
    references = "\n".join(case.meta.physics_references)
    text = "\n".join(
        [
            output.description,
            case.__doc__ or "",
            references,
        ]
    )

    assert "Single-phase Darcy boundary conductance" in output.description
    assert "unit-depth domain" in output.description
    assert "Petroleum Reservoir Simulation" in text
    assert "https://doi.org/10.2118/72469-PA" in references
    assert "https://www.ipt.ntnu.no/~kleppe/pub/SPE-COMPARATIVE/papers/tenth.pdf" in references
    assert "https://www.sintef.no/projectweb/mrst/" in references
    assert "https://www.sintef.no/en/software/mrst/" in references
    assert "https://github.com/SINTEF-AppliedCompSci/MRST" in references
    assert "https://www.sintef.no/projectweb/mrst/modules/mrst-core/" in references
    assert "https://www.sintef.no/projectweb/mrst/modules/deckformat/" in references
    assert "OPM Flow" in references
    assert "https://opm-project.org/" in references
    assert "https://opm-project.org/?page_id=19" in references
    assert "https://opm-project.org/?page_id=955" in references
    assert "https://opm-project.org/?page_id=1454" in references
    assert "https://opm-project.org/?page_id=1798" in references
    assert "https://github.com/OPM/opm-simulators" in references
    assert "Peaceman" in references
    assert "10.2118/6893-PA" in references
    assert "https://doi.org/10.2118/6893-PA" in references
    assert "https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html" in references
    assert "https://www.usgs.gov/media/images/groundwater-recharge-and-discharge" in references
    assert "Darcy's Law and hydraulic-conductivity" in references
    assert case._MU == 1.0e-3


def test_source_declares_pressure_boundary_surrogate_not_well_model():
    source_text = Path("doe_testbed/cases/geomechanics/oil_reservoir_9d.py").read_text(
        encoding="utf-8"
    )
    case = OilReservoir9D()

    assert "left and right boundaries impose" in source_text
    assert "pressure field, total boundary conductance" in source_text
    assert "Injection well on the left boundary" not in source_text
    assert "block flow rates are solved" not in source_text
    assert "Compact 9-D all-active" in case.meta.scenario_notes["VS-E"]


def test_metadata_declares_omitted_reservoir_physics_scope():
    case = OilReservoir9D()
    details = case.meta.solver_details.lower()
    source_text = Path("doe_testbed/cases/geomechanics/oil_reservoir_9d.py").read_text(
        encoding="utf-8"
    ).lower()
    combined = details + "\n" + " ".join(source_text.split())

    for phrase in (
        "no wells",
        "peaceman well indices",
        "well controls",
        "multiphase relative permeability",
        "capillary pressure",
        "compressibility",
        "porosity/storage",
        "gravity segregation",
        "3-d stratigraphy",
        "fracture networks",
        "time-dependent saturation front",
        "aquifer support",
        "black-oil formulation",
        "compositional/thermal/eor process",
        "industry-standard simulator deck",
        "eclipse-deck parsing",
        "surface-facility constraint",
        "standard or multisegment wells",
        "local grid refinement",
        "restart-file workflows",
        "polymer/surfactant option",
        "solvent model",
        "h2 storage",
        "geothermal/energy equation",
        "eclipse-format summary/restart output",
        "opm python/pyaction/udq/actionx workflows",
        "newton/nldd nonlinear solvers",
        "cpr/multigrid linear solvers",
        "adaptive timestep controls",
        "linear-solver failure",
        "timestep-cut retry",
        "co2 storage physics",
        "history matching",
        "field calibration",
        "upscaling study",
        "geomechanical coupling",
        "scope/context anchors only",
        "not executed or used as validation data",
        "external boundaries, recharge/discharge structure, and pumping-well stresses",
        "well pumping as discharge",
        "floodwater/stream/irrigation seepage as recharge",
        "distributed source/sink forcing",
        "left/right-pressure-boundary surrogate",
        "mrst-core boundary-condition/source-term/well workflow objects",
        "no wells or distributed source/sink terms",
        "non-positive permeability",
        "pressure fields outside the dirichlet-boundary envelope",
        "non-positive conductance",
        "out-of-contract q_total/dp_max/eff_perm outputs",
        "raise explicit failures",
        "named _validate_physical_inputs",
        "before sparse darcy assembly",
    ):
        assert phrase in combined
    assert hasattr(case, "_validate_physical_inputs")


def test_metadata_marks_permeability_input_output_response_nonlinear():
    case = OilReservoir9D()

    assert case.meta.is_linear is False
    assert "strong_interaction" in case.meta.response_topology


def test_default_fidelity_grid_documentation_matches_solver_default():
    case = OilReservoir9D()
    source_text = Path("doe_testbed/cases/geomechanics/oil_reservoir_9d.py").read_text(
        encoding="utf-8"
    )

    assert case._N_PER_BLOCK == 6
    assert "18×18 grid at default fidelity" in case.meta.solver_details
    assert "18×18 grid at default fidelity" in source_text
    assert "36×36" not in source_text


def test_validation_material_declares_uncalibrated_reservoir_surrogate_scope():
    validation_text = Path("tests/_cross_validation/oil_reservoir_9d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path("doe_testbed/cases/geomechanics/oil_reservoir_9d.py").read_text(
        encoding="utf-8"
    )

    assert "Darcy/FDM pressure-boundary benchmark surrogate" in validation_text
    assert "SPE10" in validation_text
    assert "https://doi.org/10.2118/72469-PA" in validation_text
    assert "https://github.com/SINTEF-AppliedCompSci/MRST" in validation_text
    assert "https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html" in validation_text
    assert "https://www.usgs.gov/media/images/groundwater-recharge-and-discharge" in validation_text
    assert "USGS Darcy-law / hydraulic-conductivity" in validation_text
    assert "https://www.sintef.no/en/software/mrst/" in validation_text
    assert "https://www.sintef.no/projectweb/mrst/modules/mrst-core/" in validation_text
    assert "https://www.sintef.no/projectweb/mrst/modules/deckformat/" in validation_text
    assert "https://opm-project.org/" in validation_text
    assert "https://opm-project.org/?page_id=1454" in validation_text
    assert "https://opm-project.org/?page_id=1798" in validation_text
    assert "https://github.com/OPM/opm-simulators" in validation_text
    assert "not wellbore modelling" in validation_text
    normalized_validation = " ".join(validation_text.split())
    assert "not wellbore modelling, well-control optimisation, multiphase flow, compressible flow, history matching, or calibrated field-scale reservoir simulation" in normalized_validation
    assert "external boundaries, recharge/discharge structure, and pumping stresses" in normalized_validation
    assert "fixed left/right pressure boundaries with closed top/bottom boundaries" in normalized_validation
    assert "well pumping as groundwater discharge" in normalized_validation
    assert "streams, floodwater, and irrigation as recharge" in normalized_validation
    assert "distributed source/sink and recharge/discharge forcing" in normalized_validation
    assert "black-oil" in normalized_validation
    assert "simulator deck" in normalized_validation
    assert "eclipse deck parsing" in normalized_validation
    assert "restart-file" in normalized_validation
    assert "summary/restart-file workflows" in normalized_validation
    assert "Python/PYACTION/UDQ/ACTIONX scripting hooks" in validation_text
    assert "Newton/NLDD nonlinear" in validation_text
    assert "CPR/multigrid linear solvers" in validation_text
    assert "adaptive timestep controls" in validation_text
    assert "solver failure handling and timestep cutting" in normalized_validation
    assert "no time-stepping, nonlinear residual, or OPM-style solver-retry workflow" in normalized_validation
    assert "solvent/h2 storage" in normalized_validation
    assert "surface-facility" in normalized_validation
    assert "standard/multisegment wells" in normalized_validation
    assert "eclipse input" in normalized_validation.lower()
    assert "upscaling studies" in normalized_validation
    assert "boundary conditions, source terms, and wells as first-class modelling objects" in normalized_validation
    assert "no wells or distributed source/sink terms" in normalized_validation
    assert "does not run MRST/OPM Flow" in validation_text
    assert "parse reservoir decks" in validation_text
    assert "compare against SPE10 simulator outputs" in validation_text
    assert "not a calibrated field reservoir\nsimulation" in source_text
    assert "not a single-channel reservoir model" in source_text
    assert "production well)" not in source_text
    assert "Target accuracy" not in validation_text
    assert "external validation" not in validation_text


def test_reviewer_markdown_note_keeps_oil_reservoir_p2_scope_boundary():
    note_text = Path("tests/_cross_validation/oil_reservoir_9d.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(note_text.split())

    required_phrases = [
        "P2 plausibility-only",
        "No response-surface change",
        "Compact 3x3 single-phase incompressible Darcy",
        "All nine permeability blocks are active",
        "not SPE10-validated",
        "not MRST/OPM-equivalent",
        "not black-oil validated",
        "not multiphase validated",
        "not compositional/thermal/EOR validated",
        "not CO2/H2-storage validated",
        "not well/facility validated",
        "not deck/restart validated",
        "not history-matched",
        "not field-calibrated",
        "not oil-production-forecast output",
        "https://doi.org/10.2118/72469-PA",
        "https://www.sintef.no/projectweb/mrst/modules/mrst-core/",
        "https://opm-project.org/?page_id=955",
        "https://opm-project.org/?page_id=1454",
        "https://github.com/OPM/opm-simulators",
        "https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html",
    ]
    for phrase in required_phrases:
        assert phrase in normalized

    for forbidden_phrase in [
        "Evidence tier | P1",
        "Evidence tier | P0",
        "Evidence tier | externally validated",
    ]:
        assert forbidden_phrase not in normalized


def test_web_audit_keeps_oil_reservoir_evidence_tier_and_source_scope_honest():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit_text.split())
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `oil_reservoir_9d` |") and "USGS Circular 1224" in line
    )

    assert "Evidence tier remains P2" in row
    assert "USGS Circular 1224" in row
    assert "Darcy-law / hydraulic-conductivity" in row
    assert "SPE10 is not validation data" in row
    assert "OPM Flow/MRST are not being run" in row
    assert "scope/context evidence only" in row
    assert "deckformat" in row
    assert "restart files" in row
    assert "H2 storage" in row
    assert "solvent" in row
    assert "not a calibrated oil-production forecast" in row
    assert "do not describe as externally validated Darcy/reservoir simulation" in row
    assert "OPM/MRST-equivalent simulation" in row
    assert "internal vertical Darcy faces" in row
    assert "silent `K_eff` clipping" in row
    assert "strong-heterogeneity regression" in row
    assert "Full resweep remains required" in row
    assert "Additional `oil_reservoir_9d` 2026-06-08 USGS boundary-setting tightening:" in normalized_audit
    assert "external boundaries, recharge/ discharge structure, and pumping-well stresses" in normalized_audit
    assert "Additional `oil_reservoir_9d` 2026-06-08 MRST-core boundary note:" in normalized_audit
    assert "boundary conditions, source terms, and wells as first-class workflow objects" in normalized_audit
    assert "Additional `oil_reservoir_9d` 2026-06-08 USGS recharge/discharge boundary note:" in normalized_audit
    assert "groundwater discharge includes well pumping" in normalized_audit
    assert "recharge can include seepage from floodwater, streams, and irrigation" in normalized_audit
    assert "Additional `oil_reservoir_9d` 2026-06-08 OPM solver-workflow boundary note:" in normalized_audit
    assert "Newton-style nonlinear solves" in normalized_audit
    assert "CPR/multigrid linear solvers" in normalized_audit
    assert "adaptive timestepping" in normalized_audit
    assert "summary/restart output" in normalized_audit
    assert "Python/PYACTION documentation" in normalized_audit
    assert "Python/PYACTION/UDQ/ACTIONX logic" in normalized_audit
    assert "Additional `oil_reservoir_9d` 2026-06-09 OPM PyAction/UDQ/ACTIONX boundary confirmation:" in normalized_audit
    assert "release-versioned online API documentation" in normalized_audit
    assert "does not execute Python actions" in normalized_audit
    assert "OPM-style solver controls" in normalized_audit
    assert "global-Darcy scaling-control note" in normalized_audit
    assert "common permeability multiplier leaves the pressure field" in normalized_audit


def test_full_reservoir_simulators_are_boundary_context_not_validation_claims():
    validation_text = Path("tests/_cross_validation/oil_reservoir_9d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path("doe_testbed/cases/geomechanics/oil_reservoir_9d.py").read_text(
        encoding="utf-8"
    )
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `oil_reservoir_9d` |") and "OPM Flow/MRST" in line
    )

    assert "fully implicit black-oil simulator" in validation_text
    assert "wells/facilities" in validation_text
    assert "industry-standard model support" in validation_text
    assert "rock-dependent capillary/relative-permeability curves" in validation_text
    assert "CO2STORE" in validation_text
    assert "polymer options" in validation_text
    assert "solvent" in validation_text
    assert "hydrogen-storage" in validation_text
    assert "restart-file workflows" in validation_text
    assert "boundary conditions, source terms, and wells as first-class modelling objects" in validation_text
    assert "ECLIPSE input/corner-point-grid workflows" in validation_text
    assert "deckformat module" in validation_text
    assert "standard/multisegment wells" in validation_text
    assert "surface" in validation_text and "facilities" in validation_text
    assert "MRST includes ECLIPSE input/corner-point-grid workflows" in validation_text
    assert "black-oil, compositional, EOR, CO2" in validation_text
    assert "does not execute MRST/OPM Flow" in source_text
    assert "are not executed or used as validation" in source_text
    assert "Evidence tier remains P2" in row
    assert "OPM Flow/MRST are not being run" in row
    assert "SPE10 is not validation data" in row


def test_web_audit_records_oil_reservoir_official_source_recheck():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert "Additional `oil_reservoir_9d` 2026-06-08 official-source re-check" in audit_text
    assert "public SPE10/MRST/OPM/Peaceman/USGS sources were re-opened" in normalized
    assert "not as validation data for the registered response surface" in normalized
    assert "USGS Circular 1224 supports the Darcy-law / hydraulic-conductivity" in normalized
    assert "SPE10 / Christie-Blunt supports the existence of heterogeneous-reservoir upscaling comparison problems" in normalized
    assert "does not use the SPE10 grid, fluids, wells, waterflood, production curves" in normalized
    assert "participant simulator comparisons" in normalized
    assert "MRST material supports omitted-scope capability boundaries" in normalized
    for mrst_capability in (
        "ECLIPSE deck input",
        "corner-point grids",
        "boundary conditions",
        "source terms",
        "wells",
        "grid/fluid/rock/well object construction",
        "restart workflows",
        "black-oil",
        "compositional",
        "standard/multisegment wells",
    ):
        assert mrst_capability in normalized
    assert "OPM Flow material supports omitted-scope boundaries" in normalized
    for opm_capability in (
        "fully implicit black-oil simulation",
        "wells/facilities",
        "industry-standard models",
        "polymer",
        "solvent",
        "H2/CO2 storage",
        "thermal simulation",
        "output/restart workflows",
        "Newton-style nonlinear solves",
        "CPR/multigrid linear solvers",
        "adaptive timestepping",
        "Python/PYACTION documentation",
        "Python actions",
        "UDQ/ACTIONX-style control logic",
    ):
        assert opm_capability in normalized
    assert "Peaceman is well-block pressure interpretation context only" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert "Additional `oil_reservoir_9d` 2026-06-08 USGS boundary-setting tightening:" in normalized
    assert "external boundaries, recharge/ discharge structure, and pumping-well stresses" in normalized
    assert "fixed left/right pressure boundaries with closed top/bottom boundaries and no wells or source/sink terms" in normalized
    assert "Additional `oil_reservoir_9d` 2026-06-08 MRST-core boundary note:" in normalized
    assert "boundary conditions, source terms, and wells as first-class workflow objects" in normalized
    assert "Additional `oil_reservoir_9d` 2026-06-08 USGS recharge/discharge boundary note:" in normalized
    assert "groundwater discharge includes well pumping" in normalized
    assert "recharge can include seepage from floodwater, streams, and irrigation" in normalized
    assert "distributed source/sink and recharge/ discharge forcings" in normalized
    assert "OPM solver-workflow boundary note" in normalized
    assert "Python/PYACTION/UDQ/ACTIONX logic" in normalized
    assert "release-versioned online API documentation" in normalized
    assert "nonlinear black-oil equations" in normalized
    assert "adaptive timesteps" in normalized
    assert "summary/restart files" in normalized
    assert "OPM linear-solver/timestep boundary confirmation" in normalized
    assert "linear-solver failure handling" in normalized
    assert "timestep-cut retry" in normalized
    assert "one steady sparse linear Darcy pressure system per evaluation" in normalized
    assert "no saturation time stepping" in normalized
    assert "global-Darcy scaling-control note" in normalized
    assert "all nine block permeabilities are multiplied by one common factor" in normalized
    assert "pressure field and `dp_max` remain invariant" in normalized
    assert "MRST/Peaceman source-traceability note" in normalized
    assert "https://github.com/SINTEF-AppliedCompSci/MRST" in normalized
    assert "https://doi.org/10.2118/6893-PA" in normalized
    assert "MRST source availability does not make the benchmark MRST-equivalent" in normalized
    assert "Peaceman well-block pressure interpretation does not add wells or well indices" in normalized
    assert "fixed left/right pressure-boundary surrogate" in normalized
    assert "No numerical response resweep is needed solely for this source-traceability note" in normalized
    for forbidden_claim in (
        "SPE10-validated",
        "MRST/OPM-equivalent",
        "black-oil",
        "multiphase",
        "compositional",
        "thermal",
        "EOR",
        "CO2/H2-storage",
        "well/facility",
        "deck/restart",
        "upscaling",
        "history-matched",
        "field-calibrated",
        "oil-production-forecast output",
    ):
        assert forbidden_claim in normalized
    assert "compact 3×3 single-phase incompressible Darcy pressure-boundary conductance surrogate" in normalized
    assert "all nine permeability blocks active" in normalized
    assert "Full resweep is still required" in normalized
    assert (
        "no additional numerical response resweep is needed solely for this official-source re-check"
        in normalized.lower()
    )


def test_web_audit_records_oil_reservoir_20260609_boundary_recheck_and_axis_comment():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    source_text = Path("doe_testbed/cases/geomechanics/oil_reservoir_9d.py").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(audit_text.split())

    assert "Additional `oil_reservoir_9d` 2026-06-09 official-source boundary re-check" in audit_text
    assert "USGS Darcy/recharge-discharge" in normalized
    assert "SPE10" in normalized
    assert "SINTEF MRST" in normalized
    assert "OPM Flow/manual/PyAction/linear-solver" in normalized
    assert "Peaceman source anchors" in normalized
    assert "do not provide case-specific validation data" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert "No additional numerical response resweep is needed solely for this source-boundary and comment correction" in normalized
    assert "existing full-resweep requirement" in normalized
    assert "K[iy, ix], aligned with p_field[iy, ix]" in source_text
    assert "K[i,j] (i = x-direction, j = y-direction)" not in source_text


def test_uniform_permeability_recovers_effective_permeability():
    case = OilReservoir9D()
    for permeability_mD in (1.0, 100.0, 1000.0):
        X = np.full((1, case.meta.dim_input), permeability_mD, dtype=float)
        Y = case.evaluate(case.normalize(X))
        assert np.isclose(Y[0, 2], permeability_mD, rtol=1e-10, atol=1e-10)


def test_uniform_permeability_scales_total_flow_linearly():
    case = OilReservoir9D()
    X_phys = np.array(
        [
            [1.0] * case.meta.dim_input,
            [10.0] * case.meta.dim_input,
            [100.0] * case.meta.dim_input,
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.allclose(Y[:, 0] / Y[0, 0], [1.0, 10.0, 100.0], rtol=1e-10, atol=1e-10)
    assert np.allclose(Y[:, 1], Y[0, 1], rtol=1e-10, atol=1e-10)


def test_common_heterogeneous_permeability_scaling_preserves_pressure_shape():
    case = OilReservoir9D()
    base = np.array([2.0, 5.0, 20.0, 3.0, 10.0, 50.0, 4.0, 8.0, 25.0])
    scaled = 10.0 * base

    y_base = case._solve_reservoir(base, n_per_block=6)
    y_scaled = case._solve_reservoir(scaled, n_per_block=6)

    assert np.isclose(y_scaled[0] / y_base[0], 10.0, rtol=1e-10, atol=1e-10)
    assert np.isclose(y_scaled[2] / y_base[2], 10.0, rtol=1e-10, atol=1e-10)
    assert np.isclose(y_scaled[1], y_base[1], rtol=1e-10, atol=1e-10)


def test_uniform_permeability_matches_absolute_darcy_scale():
    case = OilReservoir9D()
    permeability_mD = 100.0

    Q_total, dp_max, eff_perm = case._solve_reservoir(
        np.full(case.meta.dim_input, permeability_mD, dtype=float),
        n_per_block=6,
    )

    expected_q = permeability_mD * 9.869e-16 / case._MU * 1e6 * 86400
    expected_grad = case._P_INJ - case._P_PROD

    assert np.isclose(Q_total, expected_q, rtol=1e-10, atol=1e-10)
    assert np.isclose(dp_max, expected_grad, rtol=1e-10, atol=1e-10)
    assert np.isclose(eff_perm, permeability_mD, rtol=1e-10, atol=1e-10)


def test_column_layered_reservoir_matches_series_permeability_control():
    case = OilReservoir9D()
    # Three equal-width columns in series: K_eff = 3 / (1/K_left + 1/K_mid + 1/K_right).
    left_to_right = np.array(
        [10.0, 100.0, 1000.0, 10.0, 100.0, 1000.0, 10.0, 100.0, 1000.0],
        dtype=float,
    )
    right_to_left = np.array(
        [1000.0, 100.0, 10.0, 1000.0, 100.0, 10.0, 1000.0, 100.0, 10.0],
        dtype=float,
    )
    expected_eff_perm = 3.0 / (1.0 / 10.0 + 1.0 / 100.0 + 1.0 / 1000.0)

    default = case._solve_reservoir(left_to_right, n_per_block=6)
    high = case._solve_reservoir(left_to_right, n_per_block=24)
    high_reversed = case._solve_reservoir(right_to_left, n_per_block=24)

    assert np.isclose(default[2], expected_eff_perm, rtol=0.03)
    assert np.isclose(high[2], expected_eff_perm, rtol=0.01)
    assert np.isclose(high_reversed[2], expected_eff_perm, rtol=0.01)
    assert np.isclose(high[0], high_reversed[0], rtol=1e-10, atol=1e-10)


def test_row_layered_reservoir_matches_parallel_permeability_control():
    case = OilReservoir9D()
    # Three equal-height rows in parallel: K_eff is the equal-thickness arithmetic mean.
    top_to_bottom = np.array(
        [10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0],
        dtype=float,
    )
    bottom_to_top = top_to_bottom.reshape(3, 3)[::-1, :].reshape(-1)
    expected_eff_perm = (10.0 + 100.0 + 1000.0) / 3.0

    default = case._solve_reservoir(top_to_bottom, n_per_block=6)
    high = case._solve_reservoir(top_to_bottom, n_per_block=24)
    high_reversed = case._solve_reservoir(bottom_to_top, n_per_block=24)

    assert np.isclose(default[2], expected_eff_perm, rtol=0.03)
    assert np.isclose(high[2], expected_eff_perm, rtol=0.01)
    assert np.isclose(high_reversed[2], expected_eff_perm, rtol=0.01)
    assert np.isclose(high[0], high_reversed[0], rtol=1e-10, atol=1e-10)


def test_central_block_permeability_increases_flow_capacity():
    case = OilReservoir9D()
    base = np.full(case.meta.dim_input, 100.0, dtype=float)
    low_center = base.copy()
    high_center = base.copy()
    low_center[4] = 1.0
    high_center[4] = 1000.0
    X_phys = np.vstack([low_center, base, high_center])

    Y = case.evaluate(case.normalize(X_phys))

    assert np.all(np.diff(Y[:, 0]) > 0.0)
    assert np.all(np.diff(Y[:, 2]) > 0.0)


def test_each_block_permeability_monotonically_increases_conductance():
    case = OilReservoir9D()
    base = np.array(
        [20.0, 50.0, 200.0, 5.0, 100.0, 600.0, 10.0, 80.0, 300.0],
        dtype=float,
    )

    for index, block_name in enumerate(case.meta.important_variables):
        low = base.copy()
        high = base.copy()
        low[index] = max(1.0, base[index] / 5.0)
        high[index] = min(1000.0, base[index] * 5.0)

        low_response = case._solve_reservoir(low, n_per_block=6)
        high_response = case._solve_reservoir(high, n_per_block=6)

        assert high_response[0] > low_response[0], block_name
        assert high_response[2] > low_response[2], block_name


def test_production_side_high_permeability_column_increases_total_flow():
    case = OilReservoir9D()
    base = np.full(case.meta.dim_input, 100.0, dtype=float)
    high_production_column = base.copy()
    high_production_column[[2, 5, 8]] = 1000.0
    X_phys = np.vstack([base, high_production_column])

    Y = case.evaluate(case.normalize(X_phys))

    assert Y[1, 0] > Y[0, 0]
    assert Y[1, 2] > Y[0, 2]


def test_top_bottom_mirror_symmetry_preserves_flow_metrics():
    case = OilReservoir9D()
    blocks = np.array(
        [20.0, 50.0, 200.0, 5.0, 100.0, 600.0, 10.0, 80.0, 300.0],
        dtype=float,
    )
    mirrored = blocks.reshape(3, 3)[::-1, :].reshape(-1)

    Y = case.evaluate(case.normalize(np.vstack([blocks, mirrored])))

    assert np.allclose(Y[0], Y[1], rtol=1e-10, atol=1e-10)


def test_left_right_mirror_symmetry_preserves_conductance_metrics():
    case = OilReservoir9D()
    blocks = np.array(
        [20.0, 50.0, 200.0, 5.0, 100.0, 600.0, 10.0, 80.0, 300.0],
        dtype=float,
    )
    mirrored = blocks.reshape(3, 3)[:, ::-1].reshape(-1)

    # The continuous Darcy problem is left-right symmetric. The node-assigned
    # block grid has a small boundary-integration discretization asymmetry, so
    # pin the high-fidelity approximation rather than requiring bitwise equality.
    Y = np.vstack(
        [
            case._solve_reservoir(blocks, n_per_block=24),
            case._solve_reservoir(mirrored, n_per_block=24),
        ]
    )

    assert np.allclose(Y[0, [0, 2]], Y[1, [0, 2]], rtol=0.01)
    assert np.isclose(Y[0, 1], Y[1, 1], rtol=1e-10, atol=1e-10)


def test_total_flow_uses_internal_face_average_for_heterogeneous_fields():
    case = OilReservoir9D()
    blocks = np.array(
        [1.0, 1000.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0, 1000.0, 1000.0],
        dtype=float,
    )

    Q_total, _, eff_perm = case._solve_reservoir(blocks, n_per_block=6)

    # This strong-contrast field exposes the legacy production-boundary-only
    # bias; averaging all internal vertical Darcy faces gives the finite-volume
    # style conductance used by the current response surface.
    assert np.isclose(Q_total, 40.77972460004418, rtol=1e-10, atol=1e-10)
    assert np.isclose(eff_perm, 478.2526631320019, rtol=1e-10, atol=1e-10)


def test_internal_face_flux_diagnostics_pin_mass_balance_convergence():
    case = OilReservoir9D()
    blocks = np.array(
        [20.0, 50.0, 200.0, 5.0, 100.0, 600.0, 10.0, 80.0, 300.0],
        dtype=float,
    )

    default_response, default_diag = case._solve_reservoir(
        blocks,
        n_per_block=6,
        return_diagnostics=True,
    )
    high_response, high_diag = case._solve_reservoir(
        blocks,
        n_per_block=24,
        return_diagnostics=True,
    )

    assert default_diag["grid_nodes"] == 18
    assert high_diag["grid_nodes"] == 72
    assert default_diag["n_per_block"] == 6
    assert high_diag["n_per_block"] == 24
    assert default_diag["Q_faces_SI"].shape == (17,)
    assert high_diag["Q_faces_SI"].shape == (71,)
    assert np.isclose(default_diag["Q_total_SI"], np.mean(default_diag["Q_faces_SI"]))
    assert np.isclose(high_diag["Q_total_SI"], np.mean(high_diag["Q_faces_SI"]))
    assert default_diag["face_flux_relative_spread"] < 0.04
    assert high_diag["face_flux_relative_spread"] < 0.01
    assert high_diag["face_flux_relative_spread"] < default_diag["face_flux_relative_spread"]
    assert np.isclose(
        default_response[0],
        default_diag["Q_total_SI"] / ((case._P_INJ - case._P_PROD) * 1e6) * 1e6 * 86400,
    )
    assert np.isclose(
        high_response[0],
        high_diag["Q_total_SI"] / ((case._P_INJ - case._P_PROD) * 1e6) * 1e6 * 86400,
    )


def test_diagnostic_mode_does_not_change_formal_reservoir_response():
    case = OilReservoir9D()
    blocks = np.array(
        [20.0, 50.0, 200.0, 5.0, 100.0, 600.0, 10.0, 80.0, 300.0],
        dtype=float,
    )

    formal = case._solve_reservoir(blocks, n_per_block=6)
    diagnostic, diagnostics = case._solve_reservoir(
        blocks,
        n_per_block=6,
        return_diagnostics=True,
    )

    assert isinstance(diagnostics, dict)
    np.testing.assert_allclose(diagnostic, formal, rtol=0.0, atol=0.0)


def test_effective_permeability_contract_does_not_silently_clip(monkeypatch):
    case = OilReservoir9D()

    def fake_trapezoid(values, *args, **kwargs):
        axis = kwargs.get("axis", None)
        if axis == 0:
            return np.full(values.shape[1], 1.0e6)
        return np.trapz(values, *args, **{k: v for k, v in kwargs.items() if k != "axis"})

    monkeypatch.setattr(np, "trapezoid", fake_trapezoid)

    with pytest.raises(ValueError, match="effective permeability outside"):
        case._solve_reservoir(np.full(case.meta.dim_input, 100.0, dtype=float), n_per_block=6)


def test_invalid_private_solver_inputs_are_rejected_before_sparse_solve():
    case = OilReservoir9D()

    with pytest.raises(ValueError, match="expects exactly 9 permeability blocks"):
        case._solve_reservoir(np.full(8, 100.0, dtype=float), n_per_block=6)

    bad_perm = np.full(case.meta.dim_input, 100.0, dtype=float)
    bad_perm[4] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        case._solve_reservoir(bad_perm, n_per_block=6)

    bad_perm[4] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        case._solve_reservoir(bad_perm, n_per_block=6)

    with pytest.raises(ValueError, match="integer >= 2"):
        case._solve_reservoir(np.full(case.meta.dim_input, 100.0, dtype=float), n_per_block=1)

    with pytest.raises(ValueError, match="integer >= 2"):
        case._solve_reservoir(np.full(case.meta.dim_input, 100.0, dtype=float), n_per_block=6.5)


def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_sparse_solve(monkeypatch):
    case = OilReservoir9D()
    assert hasattr(case, "_validate_physical_inputs")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("sparse Darcy solve should not be called for invalid physical input")

    monkeypatch.setattr(case, "_solve_reservoir", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._evaluate_core(np.full(case.meta.dim_input, 100.0, dtype=float))

    invalid_rows = [
        np.full(case.meta.dim_input, 0.99, dtype=float),
        np.full(case.meta.dim_input, 1000.1, dtype=float),
        np.full(case.meta.dim_input, np.nan, dtype=float),
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError, match="K11-K33 permeability benchmark domain"):
            case._evaluate_core(np.array([row], dtype=float))


def test_sparse_pressure_solution_contract_rejects_bad_solver_output(monkeypatch):
    import scipy.sparse.linalg as spla

    case = OilReservoir9D()
    X = np.full(case.meta.dim_input, 100.0, dtype=float)

    def malformed_solution(*_args, **_kwargs):
        return np.array([np.nan])

    monkeypatch.setattr(spla, "spsolve", malformed_solution)
    with pytest.raises(ValueError, match="pressure solution"):
        case._solve_reservoir(X, n_per_block=6)


def test_sparse_pressure_solution_contract_rejects_pressure_envelope_violation(monkeypatch):
    import scipy.sparse.linalg as spla

    case = OilReservoir9D()
    X = np.full(case.meta.dim_input, 100.0, dtype=float)

    def outside_boundary_envelope(A_mat, _b_vec):
        n_total = A_mat.shape[0]
        pressure = np.linspace(case._P_PROD * 1e6, case._P_INJ * 1e6, n_total)
        pressure[n_total // 2] = case._P_INJ * 1e6 + 1.0e5
        return pressure

    monkeypatch.setattr(spla, "spsolve", outside_boundary_envelope)
    with pytest.raises(ValueError, match="pressure-boundary envelope"):
        case._solve_reservoir(X, n_per_block=6)


def test_declared_output_ranges_cover_domain_corners():
    case = OilReservoir9D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower - 1e-10
        assert np.nanmax(values) <= upper + 1e-10


def test_sparse_reservoir_solver_failure_is_not_silent_nan(monkeypatch):
    case = OilReservoir9D()

    def fail_solve(*_args, **_kwargs):
        raise ValueError("simulated reservoir linear-solve failure")

    monkeypatch.setattr(case, "_solve_reservoir", fail_solve)
    X_phys = np.full((1, case.meta.dim_input), 100.0, dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_nonfinite_reservoir_response_is_not_silent_nan(monkeypatch):
    case = OilReservoir9D()

    def invalid_solve(*_args, **_kwargs):
        return np.array([np.nan, 9.0, 100.0])

    monkeypatch.setattr(case, "_solve_reservoir", invalid_solve)
    X_phys = np.full((1, case.meta.dim_input), 100.0, dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_out_of_contract_reservoir_response_is_not_archived(monkeypatch):
    case = OilReservoir9D()
    X_phys = np.full((1, case.meta.dim_input), 100.0, dtype=float)

    invalid_responses = [
        np.array([500.1, 9.0, 100.0]),
        np.array([1.0, 100.1, 100.0]),
        np.array([1.0, 9.0, 1000.1]),
    ]

    for response in invalid_responses:
        monkeypatch.setattr(
            case,
            "_solve_reservoir",
            lambda *_args, _response=response, **_kwargs: _response,
        )
        with pytest.raises(RuntimeError, match="refusing silent NaN"):
            case._evaluate_core(X_phys)
