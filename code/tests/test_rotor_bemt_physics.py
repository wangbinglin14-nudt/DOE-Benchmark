"""Physics regression tests for ``rotor_bemt_6d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.aerodynamics.rotor_bemt_6d import RotorBEMT6D


def _domain_corners(case: RotorBEMT6D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_declared_output_ranges_cover_domain_corners():
    case = RotorBEMT6D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_collective_pitch_increases_thrust_and_power_coefficients():
    case = RotorBEMT6D()
    X_phys = np.array(
        [
            [3.0, 350.0, 0.12, 12.0, -5.0, 4.0],
            [3.0, 350.0, 0.12, 16.0, -5.0, 4.0],
            [3.0, 350.0, 0.12, 20.0, -5.0, 4.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.all(np.diff(Y[:, 0]) > 0.0)
    assert np.all(np.diff(Y[:, 1]) > 0.0)


def test_solidity_and_blade_count_increase_thrust_coefficient():
    case = RotorBEMT6D()
    chord_sweep = np.array(
        [
            [3.0, 350.0, 0.05, 12.0, -5.0, 4.0],
            [3.0, 350.0, 0.10, 12.0, -5.0, 4.0],
            [3.0, 350.0, 0.20, 12.0, -5.0, 4.0],
        ],
        dtype=float,
    )
    blade_sweep = np.array(
        [
            [3.0, 350.0, 0.12, 12.0, -5.0, 2.0],
            [3.0, 350.0, 0.12, 12.0, -5.0, 4.0],
            [3.0, 350.0, 0.12, 12.0, -5.0, 6.0],
        ],
        dtype=float,
    )

    chord_ct = case.evaluate(case.normalize(chord_sweep))[:, 0]
    blade_ct = case.evaluate(case.normalize(blade_sweep))[:, 0]

    assert np.all(np.diff(chord_ct) > 0.0)
    assert np.all(np.diff(blade_ct) > 0.0)


def test_continuous_blade_count_input_is_rounded_and_clamped_to_integer_blades():
    case = RotorBEMT6D()
    X_phys = np.array(
        [
            [3.0, 350.0, 0.12, 12.0, -5.0, 2.49],
            [3.0, 350.0, 0.12, 12.0, -5.0, 2.00],
            [3.0, 350.0, 0.12, 12.0, -5.0, 2.51],
            [3.0, 350.0, 0.12, 12.0, -5.0, 3.00],
            [3.0, 350.0, 0.12, 12.0, -5.0, 5.51],
            [3.0, 350.0, 0.12, 12.0, -5.0, 6.00],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.allclose(Y[0], Y[1], rtol=1e-12, atol=1e-12)
    assert np.allclose(Y[2], Y[3], rtol=1e-12, atol=1e-12)
    assert np.allclose(Y[4], Y[5], rtol=1e-12, atol=1e-12)
    assert Y[3, 0] > Y[1, 0]
    assert Y[5, 0] > Y[3, 0]


def test_blade_count_rounding_is_declared_as_discontinuous_topology():
    case = RotorBEMT6D()

    assert "discontinuous" in case.meta.response_topology

    X_phys = np.array(
        [
            [3.0, 350.0, 0.12, 12.0, -5.0, 2.49],
            [3.0, 350.0, 0.12, 12.0, -5.0, 2.51],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert not np.allclose(Y[0], Y[1], rtol=1e-12, atol=1e-12)
    assert Y[1, 0] > Y[0, 0]


def test_hover_coefficients_are_invariant_to_rotational_speed_scaling():
    case = RotorBEMT6D()
    X_phys = np.array(
        [
            [3.0, 100.0, 0.12, 12.0, -5.0, 4.0],
            [3.0, 300.0, 0.12, 12.0, -5.0, 4.0],
            [3.0, 600.0, 0.12, 12.0, -5.0, 4.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.allclose(Y, Y[0], rtol=1e-10, atol=1e-12)


def test_hover_coefficients_are_invariant_to_radius_scaling():
    case = RotorBEMT6D()
    X_phys = np.array(
        [
            [0.5, 350.0, 0.12, 12.0, -5.0, 4.0],
            [3.0, 350.0, 0.12, 12.0, -5.0, 4.0],
            [5.0, 350.0, 0.12, 12.0, -5.0, 4.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert np.allclose(Y, Y[0], rtol=1e-10, atol=1e-12)


def test_figure_of_merit_matches_hover_momentum_definition():
    case = RotorBEMT6D()
    X_phys = np.array(
        [
            [3.0, 350.0, 0.12, 12.0, -5.0, 4.0],
            [2.0, 250.0, 0.08, 16.0, -3.0, 3.0],
            [5.0, 500.0, 0.16, 18.0, -8.0, 6.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    CT = Y[:, 0]
    CP = Y[:, 1]
    FM = Y[:, 2]
    expected = np.where(CT > 0.0, CT**1.5 / (np.sqrt(2.0) * CP), 0.0)

    assert np.all((FM >= 0.0) & (FM <= 1.0))
    assert np.allclose(FM, expected, rtol=1e-12, atol=1e-12)
    assert np.all(np.where(CT > 0.0, CT**1.5 / np.sqrt(2.0), 0.0) <= CP)


def test_non_lifting_hover_failure_keeps_negative_thrust_and_zero_fm():
    case = RotorBEMT6D()
    X_phys = np.array([[3.0, 350.0, 0.20, 5.0, -10.0, 6.0]], dtype=float)
    CT, CP, FM = case.evaluate(case.normalize(X_phys))[0]

    assert CT < 0.0
    assert CP > 0.0
    assert FM == 0.0


def test_power_coefficient_enforces_nonideal_hover_power_floor():
    case = RotorBEMT6D()
    X_phys = np.array(
        [
            [3.0, 350.0, 0.12, 12.0, -5.0, 4.0],
            [5.0, 500.0, 0.16, 18.0, -8.0, 6.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    CT = Y[:, 0]
    CP = Y[:, 1]
    ideal_cp = np.where(CT > 0.0, CT**1.5 / np.sqrt(2.0), 0.0)

    assert np.all(CP > ideal_cp)
    assert np.all(Y[:, 2] < 1.0)


def test_nonideal_induced_power_factor_bounds_positive_thrust_fm():
    case = RotorBEMT6D()
    X_phys = np.array(
        [
            [3.0, 350.0, 0.12, 12.0, -5.0, 4.0],
            [2.0, 250.0, 0.08, 16.0, -3.0, 3.0],
            [5.0, 500.0, 0.16, 18.0, -8.0, 6.0],
            [3.0, 350.0, 0.20, 20.0, -2.0, 6.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    positive_thrust = Y[:, 0] > 0.0

    assert np.all(positive_thrust)
    assert np.all(Y[:, 2] <= 1.0 / case._INDUCED_POWER_FACTOR)


def test_stationwise_inflow_residual_is_solved_before_integration():
    case = RotorBEMT6D()
    r_bar = np.linspace(0.05, 0.99, case._N_RADIAL)
    B = 4
    sigma = B * 0.12 / np.pi
    theta = np.deg2rad(12.0) + np.deg2rad(-5.0) * r_bar

    lam_i = np.array(
        [
            case._solve_station_inflow(float(rb), float(sigma), float(th), B)
            for rb, th in zip(r_bar, theta)
        ],
        dtype=float,
    )
    target = np.array(
        [
            case._station_target_inflow(float(rb), float(sigma), float(th), float(lam), B)
            for rb, th, lam in zip(r_bar, theta, lam_i)
        ],
        dtype=float,
    )

    assert np.max(np.abs(lam_i - target)) < 1e-8


def test_stationwise_solve_removes_legacy_iteration_limit_artifact():
    case = RotorBEMT6D()
    X_phys = np.array(
        [
            [
                3.497006758936604,
                532.0075356465613,
                0.19184527034016935,
                8.114877549111771,
                -5.411783975723413,
                3.6547312097427085,
            ]
        ],
        dtype=float,
    )
    CT, CP, FM = case.evaluate(case.normalize(X_phys))[0]

    assert CT > 0.0
    assert CP > 0.0
    assert FM > 0.0


def test_prandtl_tip_loss_matches_standard_hover_formula():
    case = RotorBEMT6D()
    r_bar = np.array([0.35, 0.70, 0.95])
    lam_i = np.array([0.055, 0.045, 0.035])
    B = 4

    observed = case._prandtl_tip_loss(r_bar, lam_i, B)
    phi = np.arctan(np.abs(lam_i) / r_bar)
    f = B / 2.0 * (1.0 - r_bar) / (r_bar * np.sin(phi))
    expected = 2.0 / np.pi * np.arccos(np.exp(-f))

    assert np.allclose(observed, expected, rtol=1e-12, atol=1e-12)
    assert observed[-1] < observed[0]


def test_prandtl_tip_loss_has_expected_tip_and_blade_count_trends():
    case = RotorBEMT6D()
    lam_i = np.full(5, 0.045)
    r_bar = np.array([0.30, 0.50, 0.70, 0.90, 0.99])

    two_blade = case._prandtl_tip_loss(r_bar, lam_i, B=2)
    six_blade = case._prandtl_tip_loss(r_bar, lam_i, B=6)

    assert np.all((two_blade >= 0.01) & (two_blade <= 1.0))
    assert np.all((six_blade >= 0.01) & (six_blade <= 1.0))
    assert np.all(np.diff(two_blade) < 0.0)
    assert np.all(np.diff(six_blade) < 0.0)
    assert np.all(six_blade >= two_blade)
    assert two_blade[-1] < 0.5
    assert six_blade[-1] > two_blade[-1]


def test_failure_boundary_targets_low_hover_figure_of_merit():
    case = RotorBEMT6D()
    fm_output = case.meta.output_variables[2]
    assert fm_output.name == "FM"
    assert fm_output.failure_threshold == 0.60
    assert fm_output.failure_direction == "below"
    assert "FM < 0.60" in case.meta.failure_condition

    X_phys = np.array(
        [
            [3.0, 350.0, 0.12, 15.0, -5.0, 4.0],
            [3.0, 350.0, 0.12, 5.0, -10.0, 2.0],
        ],
        dtype=float,
    )
    fm = case.evaluate(case.normalize(X_phys))[:, 2]

    assert fm[0] > fm_output.failure_threshold
    assert fm[1] < fm_output.failure_threshold


def test_governing_equation_states_nondimensional_hover_balance():
    case = RotorBEMT6D()
    equation = case.meta.governing_equation

    assert "dCT = 4F" in equation
    assert "lambda_i" in equation
    assert "dT = 4πr" not in equation


def test_metadata_declares_simplified_not_calibrated_bemt_scope():
    case = RotorBEMT6D()
    details = case.meta.solver_details.lower()
    references = "\n".join(case.meta.physics_references)
    source_text = Path(
        "doe_testbed/cases/aerodynamics/rotor_bemt_6d.py"
    ).read_text(encoding="utf-8")
    validation_text = Path("tests/_cross_validation/rotor_bemt_6d.py").read_text(
        encoding="utf-8"
    )
    normalized_source = " ".join(source_text.split())
    normalized_validation = " ".join(validation_text.split())

    assert "15 radial stations" in case.meta.solver_details
    assert "stationwise bisection solve" in case.meta.solver_details
    assert "hover inflow residual" in case.meta.solver_details
    assert "induced-power floor" in case.meta.solver_details
    assert "rejects malformed arrays and values outside the declared" in case.meta.solver_details
    assert "before rounding the blade count or solving BEMT" in case.meta.solver_details
    assert hasattr(case, "_validate_physical_inputs")
    assert "Leishman" in references
    assert "Johnson" in references
    assert "AeroDyn Theory Manual" in references
    assert "NREL/TP-500-36881" in references
    assert "NASA NTRS" in references
    assert "https://www.nrel.gov/docs/fy05osti/36881.pdf" in references
    assert "https://ntrs.nasa.gov/citations/20170000454" in references
    assert "https://rotorcraft.arc.nasa.gov/ndarc/index.php/reports-and-papers" in references
    assert "https://ntrs.nasa.gov/citations/20250010468" in references
    assert "release 1.19" in references
    assert "NREL/TP-500-36881" in references
    assert "wind-turbine BEM/tip-loss implementation context" in references
    assert "not helicopter-hover or 15-station grid-convergence validation" in references
    assert "AeroDyn v15" in references
    assert "OpenFAST" in references
    assert "BEMT option" in references
    assert "induction-tolerance" in references
    assert "https://www.nrel.gov/docs/libraries/wind-docs/aerodyn-manual.pdf" in references
    assert "NASA/TM-20260000902" in references
    assert "https://ntrs.nasa.gov/citations/20260000902" in references
    assert "Hover Validation and Acoustic Baseline" in references
    assert "https://rotorcraft.arc.nasa.gov/HVAB/" in references
    assert "data-use recommendations" in references
    assert "https://rotorcraft.arc.nasa.gov/HVAB/generalInformation.html" in references
    assert "performance data" in references
    assert "mean values and uncertainty estimates" in references
    assert "https://rotorcraft.arc.nasa.gov/HVAB/performanceData.html" in references
    assert "https://ntrs.nasa.gov/citations/20205002153" in references
    assert "https://ntrs.nasa.gov/citations/20230004378" in references
    assert "blade-set" in references
    assert "deformed-geometry validation-workflow boundary" in references
    assert "not CFD validation or data ingestion by this case" in references
    assert "Test Validation of Acoustically and Aerodynamically Optimized" in references
    assert "https://ntrs.nasa.gov/citations/20220015637" in references
    assert "static-hover / axial-forward-flight" in references
    assert "NASA/TP-20260001349" in references
    assert "https://ntrs.nasa.gov/citations/20260001349" in references
    assert "conversion/airplane/helicopter-mode data context" in references
    assert "NASA/TP-20260001346" in references
    assert "https://ntrs.nasa.gov/citations/20260001346" in references
    assert "vertical-climb/near-hover axial-flow data context" in references
    assert "OpenFAST AeroDyn online documentation" in references
    assert "current v5.0.0" in references
    assert "Wake_Mod/BEM_Mod" in references
    assert "DBEMT_Mod" in references
    assert "Skew_Mod" in references
    assert "UA_Mod" in references
    assert "BEM algorithm options" in references
    assert "dynamic-wake/dynamic-inflow options" in references
    assert "OLAF free-vortex-wake options" in references
    assert "unsteady-airfoil options" in references
    assert "TwrPotent" in references
    assert "TwrShadow" in references
    assert "TwrAero" in references
    assert "time-domain wind-turbine aerodynamics" in references
    assert "coupled aero-elastic simulation" in references
    assert "https://openfast.readthedocs.io/en/main/source/user/aerodyn/index.html" in references
    assert "https://openfast.readthedocs.io/en/main/source/user/aerodyn/input.html" in references
    assert "https://openfast.readthedocs.io/en/v4.0.5/source/user/aerodyn/input.html" in references
    assert "https://openfast.readthedocs.io/en/dev/source/user/aerodyn/introduction.html" in references
    assert "https://www.nrel.gov/wind/nwtc/aerodyn" in references
    assert "https://www.nrel.gov/wind/nwtc/aerodyn" in references
    assert case.meta.reference_solution_available is True
    assert "calibrated" not in details
    assert "hovering-rotor cfd validation" in details
    assert "cfd calibration" not in details
    for omitted_scope in [
        "hub-loss",
        "tangential-induction",
        "dynamic-inflow",
        "free-wake",
        "unsteady-airfoil",
        "dynamic-stall",
        "skewed-wake-correction",
        "production bem/dbemt mode selection",
        "induction-tolerance control",
        "configurable prandtl hub/tip-loss switches",
        "axial/tangential induction drag-term options",
        "olaf free-vortex-wake options",
        "Reynolds/Mach-resolved airfoil-table".lower(),
        "tower-influence",
        "tower-shadow",
        "tower-aerodynamic",
        "nacelle-drag",
        "structural-dynamics",
        "forward-flight",
        "rotor-trim",
        "ground-effect",
        "tip-mach-compressibility",
        "blade-vortex-interaction",
        "acoustic",
        "grid-convergence",
        "hover-test-dataset calibration",
        "hvab mean/uncertainty performance-data ingestion",
        "hvab blade-set geometry matching",
        "hovering-rotor cfd validation",
        "deformed-geometry or transition-state comparison",
        "nasa uam proprotor test-validation",
        "static-hover/axial-forward-flight test comparison",
        "acoustic/aerodynamic proprotor validation",
        "rotor-specific calibration",
        "aerodyn/openfast-equivalent",
        "time-domain wind-turbine aero-elastic simulation contract",
        "openfast driver/input/output-file contract",
        "production airfoil-table workflow",
    ]:
        assert omitted_scope in details
    assert "two-blade corner" in normalized_source
    assert "Prandtl tip-loss accuracy is weaker" in normalized_source
    assert "low-order benchmark choice, not a grid-convergence claim" in normalized_source
    assert "time-domain wind-turbine aerodynamics module" in normalized_source
    assert "coupled aero-elastic simulation" in normalized_source
    assert "BEM/dynamic-wake/free-vortex/unsteady-airfoil option families" in normalized_source
    assert "validation-oriented run lists" in normalized_source
    assert "data-use recommendations for combining" in normalized_source
    assert "uncertainty-aware performance products" in normalized_source
    assert "primary rotor performance files with mean values and uncertainty estimates" in normalized_source
    assert "not ingested, fitted, or compared by this benchmark" in normalized_source
    assert "Wind-turbine BEM implementation context" in source_text
    assert "not helicopter-hover or grid-convergence validation" in source_text
    assert "deliberately low-order benchmark" in normalized_source
    assert "radial grid" in normalized_source
    assert "not validated as a grid-converged" in normalized_source
    assert "AeroDyn/OpenFAST-equivalent" in case.meta.solver_details
    assert "Station nonlinear solve" in source_text
    assert "stationwise bisection solve" in source_text
    assert "not accepted after a fixed number of unconverged fixed-point iterations" in normalized_validation
    for source_anchor in [
        "https://www.nrel.gov/docs/fy05osti/36881.pdf",
        "https://www.nrel.gov/docs/libraries/wind-docs/aerodyn-manual.pdf",
        "https://ntrs.nasa.gov/citations/20170000454",
        "https://rotorcraft.arc.nasa.gov/ndarc/index.php/reports-and-papers",
        "https://ntrs.nasa.gov/citations/20250010468",
        "https://ntrs.nasa.gov/citations/20260000902",
        "https://rotorcraft.arc.nasa.gov/HVAB/",
        "https://rotorcraft.arc.nasa.gov/HVAB/generalInformation.html",
        "https://rotorcraft.arc.nasa.gov/HVAB/performanceData.html",
        "https://ntrs.nasa.gov/citations/20205002153",
        "https://ntrs.nasa.gov/citations/20230004378",
        "https://ntrs.nasa.gov/citations/20220015637",
        "https://ntrs.nasa.gov/citations/20260001349",
        "https://ntrs.nasa.gov/citations/20260001346",
        "https://openfast.readthedocs.io/en/main/source/user/aerodyn/index.html",
        "https://openfast.readthedocs.io/en/main/source/user/aerodyn/input.html",
        "https://openfast.readthedocs.io/en/v4.0.5/source/user/aerodyn/input.html",
        "https://openfast.readthedocs.io/en/dev/source/user/aerodyn/introduction.html",
        "https://www.nrel.gov/wind/nwtc/aerodyn",
        "https://www.nrel.gov/wind/nwtc/aerodyn",
    ]:
        assert source_anchor in source_text
        assert source_anchor in validation_text
    assert "These public anchors are used as BEMT equation" in normalized_validation
    assert "public validation-dataset context only" in normalized_validation
    assert "wind-turbine BEM implementation and production-tool scope context" in normalized_validation
    assert "not helicopter-hover or 15-station grid-convergence validation" in normalized_validation
    assert "not evidence that this compact six-input surrogate is AeroDyn-equivalent" in normalized_validation
    assert "OpenFAST-equivalent" in normalized_validation
    assert "Wake_Mod/BEM_Mod" in normalized_validation
    assert "DBEMT_Mod" in normalized_validation
    assert "Skew_Mod" in normalized_validation
    assert "UA_Mod" in normalized_validation
    assert "tower-influence/shadow/aero options" in normalized_validation
    assert "OpenFAST driver/input/output-file contracts" in normalized_validation
    assert "time-domain wind-turbine aerodynamics module" in normalized_validation
    assert "coupled aero-elastic simulation environment" in normalized_validation
    assert "current OpenFAST v5.0.0 online manual" in normalized_validation
    assert "OLAF free-vortex-wake options" in normalized_validation
    assert "unsteady-airfoil options" in normalized_validation
    assert "validation-oriented run lists" in normalized_validation
    assert "data-use recommendations" in normalized_validation
    assert "uncertainty-aware performance products" in normalized_validation
    assert "production BEMT residual-equation controls" in normalized_validation
    assert "configurable Prandtl hub/tip-loss switches" in normalized_validation
    assert "axial/tangential induction drag-term options" in normalized_validation
    assert "induction-tolerance settings" in normalized_validation
    assert "airfoil-table workflow" in normalized_validation
    assert "calibrated against NASA HVAB" in normalized_validation
    assert "primary performance spreadsheets with mean values and uncertainty estimates" in normalized_validation
    assert "Current NASA NDARC release 1.19 documentation" in validation_text
    assert "not component-map equivalence" in normalized_validation
    assert "NASA HVAB blade-set and hovering-rotor CFD-validation reports" in normalized_validation
    assert "geometry/test-article definition" in normalized_validation
    assert "deformation/transition/wake/airloads/performance/acoustic products" in normalized_validation
    assert "blade-set geometry matching" in normalized_validation
    assert "hovering-rotor CFD validation" in normalized_validation
    assert "airload/wake/acoustic validation" in normalized_validation
    assert "mean/uncertainty performance products exist but are not part of this registered response surface" in normalized_validation
    assert "exact test-article geometry" in normalized_validation
    assert "transition state" in normalized_validation
    assert "blade deformation" in normalized_validation
    assert "NASA UAM proprotor test-validation material is additional boundary evidence" in normalized_validation
    assert "static-hover, axial-forward-flight, and acoustic/aerodynamic validation" in normalized_validation
    assert "does not ingest, fit, or compare those data" in normalized_validation
    assert "NASA TTR/699 numerical-data addenda" in normalized_validation
    assert "conversion/airplane/helicopter-mode data" in normalized_validation
    assert "vertical-climb axial-flow conditions" in normalized_validation
    assert "not part of this registered response surface" in normalized_validation
    assert "rejects malformed arrays and values outside the declared" in normalized_validation
    assert "invalid rows cannot be hidden by integer blade-count clamping" in normalized_validation


def test_reviewer_markdown_note_keeps_rotor_bemt_p2_scope_boundary():
    note = Path("tests/_cross_validation/rotor_bemt_6d.md")
    text = note.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "P2 plausibility-only" in normalized
    assert "No response-surface change" in normalized
    assert "not AeroDyn-equivalent" in normalized
    assert "not OpenFAST-equivalent" in normalized
    assert "not HVAB-calibrated" in normalized
    assert "NDARC release 1.19" in normalized
    assert "HVAB blade-set geometry matching" in normalized
    assert "hovering-rotor CFD validation" in normalized
    assert "transition-state comparison" in normalized
    assert "deformed-geometry comparison" in normalized
    assert "airload/wake/acoustic validation" in normalized
    assert "NASA TTR/699 full-scale proprotor numerical-data addendum" in normalized
    assert "NASA TTR/699 vertical-climb numerical-data addendum" in normalized
    assert "not TTR/699 full-scale proprotor wind-tunnel validation" in normalized
    assert "dCT_momentum = 4 F lambda_i^2 r_bar dr_bar" in normalized
    assert "dCT_blade = 0.5 sigma CL r_bar^2 dr_bar" in normalized
    assert "r_bar sin(phi)" in normalized
    assert "FM = CT^(3/2) / (sqrt(2) CP)" in normalized
    for source_anchor in [
        "https://www.nrel.gov/docs/fy05osti/36881.pdf",
        "https://www.nrel.gov/docs/libraries/wind-docs/aerodyn-manual.pdf",
        "https://ntrs.nasa.gov/citations/20170000454",
        "https://rotorcraft.arc.nasa.gov/ndarc/index.php/reports-and-papers",
        "https://ntrs.nasa.gov/citations/20250010468",
        "https://ntrs.nasa.gov/citations/20260000902",
        "https://rotorcraft.arc.nasa.gov/HVAB/",
        "https://rotorcraft.arc.nasa.gov/HVAB/generalInformation.html",
        "https://rotorcraft.arc.nasa.gov/HVAB/performanceData.html",
        "https://ntrs.nasa.gov/citations/20205002153",
        "https://ntrs.nasa.gov/citations/20230004378",
        "https://ntrs.nasa.gov/citations/20220015637",
        "https://ntrs.nasa.gov/citations/20260001349",
        "https://ntrs.nasa.gov/citations/20260001346",
        "https://openfast.readthedocs.io/en/main/source/user/aerodyn/index.html",
        "https://openfast.readthedocs.io/en/main/source/user/aerodyn/input.html",
        "https://openfast.readthedocs.io/en/v4.0.5/source/user/aerodyn/input.html",
        "https://openfast.readthedocs.io/en/dev/source/user/aerodyn/introduction.html",
        "https://www.nrel.gov/wind/nwtc/aerodyn",
        "https://www.nrel.gov/wind/nwtc/aerodyn",
    ]:
        assert source_anchor in text


def test_web_audit_keeps_rotor_bemt_public_sources_as_scope_not_calibration():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit_text.split())
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `rotor_bemt_6d` |")
        and "NREL AeroDyn documentation" in line
    )

    assert "Evidence tier remains P2" in row
    assert "NREL AeroDyn documentation" in row
    assert "tip-loss, hub-loss, skewed-wake, dynamic-wake" in row
    assert "tangential-induction, airfoil-table, and convergence controls" in row
    assert "AeroDyn v15/OpenFAST" in row
    assert "production airfoil-table workflow" in row
    assert "NASA NDARC is used as conceptual-design" in row
    assert "not as evidence that this six-input surrogate matches NDARC component maps" in row
    assert "NASA RVLT/HVAB data provide high-quality hover performance" in row
    assert "not been calibrated or compared against HVAB data" in row
    assert "AeroDyn's wind-turbine BEM implementation context" in row
    assert "does not validate this helicopter-hover surrogate or its 15-station radial grid" in row
    assert "not for exact finite-station inflow/tip-loss calibration" in row
    assert "two-blade-corner accuracy" in row
    assert "grid convergence" in row
    assert "AeroDyn/NDARC equivalence" in row
    assert "OpenFAST equivalence" in row
    assert "hover-test-dataset validation" in row
    assert "Do not describe as AeroDyn-equivalent" in row
    assert "Additional `rotor_bemt_6d` 2026-06-08 official-source re-check" in audit_text
    assert "boundary evidence, not as validation data" in normalized_audit
    assert "horizontal-axis wind-turbine aerodynamics module" in normalized_audit
    assert "BEM and generalized dynamic-wake options" in normalized_audit
    assert "BEM/DBEMT mode choices" in normalized_audit
    assert "driver/output-file workflows" in normalized_audit
    assert "coupled aero-elastic simulation workflows" in normalized_audit
    assert "rotorcraft system analysis tool for conceptual design" in normalized_audit
    assert "multidisciplinary design/analysis/optimization" in normalized_audit
    assert "four-bladed 11.08-ft model-scale hover test" in normalized_audit
    assert "intended to validate state-of-the-art analysis codes" in normalized_audit
    assert "validation-oriented run lists" in normalized_audit
    assert "recommended data combinations" in normalized_audit
    assert "uncertainty-aware performance products" in normalized_audit
    assert "data-use recommendations" in normalized_audit
    assert "mean values and uncertainty estimates" in normalized_audit
    assert "not ingested, fitted, or compared" in normalized_audit
    assert "not NASA HVAB-calibrated rotor performance" in normalized_audit
    assert "Evidence tier remains P2 plausibility-only" in normalized_audit
    assert "No numerical response resweep is needed solely for this official-source re-check" in normalized_audit
    assert "Additional `rotor_bemt_6d` 2026-06-08 production-model boundary tightening:" in normalized_audit
    assert "Additional `rotor_bemt_6d` 2026-06-08 AeroDyn-control boundary note:" in normalized_audit
    assert "induction-tolerance settings" in normalized_audit
    assert "configurable hub/tip-loss switches" in normalized_audit
    assert "axial/tangential induction drag-term options" in normalized_audit
    assert "Additional `rotor_bemt_6d` 2026-06-09 NREL AeroDyn overview boundary confirmation:" in normalized_audit
    assert "aerodynamic model for horizontal-axis wind turbines" in normalized_audit
    assert "wake/induction, blade airfoil aerodynamics, and tower influence/drag" in normalized_audit
    assert "not an AeroDyn/OpenFAST wind-turbine aero-elastic module" in normalized_audit
    assert "Additional `rotor_bemt_6d` 2026-06-09 OpenFAST-v5/HVAB boundary spot-check:" in normalized_audit
    assert "dynamic-wake/dynamic-inflow controls" in normalized_audit
    assert "OLAF free-vortex-wake options" in normalized_audit
    assert "OpenFAST v5.0.0-equivalent" in normalized_audit
    assert "Additional `rotor_bemt_6d` 2026-06-09 NASA UAM proprotor validation-workflow boundary note" in normalized_audit
    assert "NASA NTRS 20220015637" in normalized_audit
    assert "static-hover and axial-forward-flight validation context" in normalized_audit
    assert "does not ingest, fit, or compare NASA UAM proprotor data" in normalized_audit
    assert "validate acoustic predictions" in normalized_audit
    assert "validate axial-forward-flight performance" in normalized_audit
    assert "proprotor-specific calibration" in normalized_audit
    assert (
        "Additional `rotor_bemt_6d` 2026-06-09 NASA TTR/699 numerical-data boundary note"
        in normalized_audit
    )
    assert "NASA NTRS 20260001349" in normalized_audit
    assert "NASA NTRS 20260001346" in normalized_audit
    assert "conversion/airplane/helicopter-mode" in normalized_audit
    assert "vertical-climb axial-flow" in normalized_audit
    assert "does not ingest, fit, or compare TTR/699 data" in normalized_audit
    assert "Additional `rotor_bemt_6d` 2026-06-09 hover-FM induced-power ceiling note" in normalized_audit
    assert "FM <= 1/kappa" in normalized_audit
    assert "implementation-control regression" in normalized_audit
    assert (
        "Additional `rotor_bemt_6d` 2026-06-09 current-NASA validation-boundary sync"
        in normalized_audit
    )
    assert "NASA/CR-20250010468 release 1.19" in normalized_audit
    assert "do not make this six-input hover surrogate NDARC-equivalent" in normalized_audit
    assert "https://ntrs.nasa.gov/citations/20205002153" in normalized_audit
    assert "https://ntrs.nasa.gov/citations/20230004378" in normalized_audit
    assert "exact test-article geometry" in normalized_audit
    assert "transition state" in normalized_audit
    assert "blade deformation" in normalized_audit
    assert "airloads, wake, performance, acoustics" in normalized_audit
    assert "Those data are not ingested, fitted, or compared" in normalized_audit
    assert "NDARC release-validated" in normalized_audit
    assert "HVAB blade-set matched" in normalized_audit
    assert "hovering-rotor CFD-validated" in normalized_audit
    assert "transition/deformation validated" in normalized_audit
    assert "airload/wake/acoustic validated" in normalized_audit


def test_web_audit_records_rotor_bemt_current_page_spot_check():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `rotor_bemt_6d` 2026-06-09 current-page spot check:"
        in audit_text
    )
    for phrase in (
        "current OpenFAST/AeroDyn v5 input documentation",
        "OpenFAST OLAF documentation",
        "NREL AeroDyn overview page",
        "NASA Ames HVAB performance/data-use pages",
        "NASA Ames NDARC reports pages",
        "primary AeroDyn input files",
        "separate blade and airfoil files",
        "standalone driver inputs",
        "output-file specifications",
        "Wake_Mod/BEM_Mod and DBEMT_Mod choices",
        "BEM algorithm options",
        "dynamic-wake/dynamic-inflow options",
        "OLAF free-vortex-wake options",
        "unsteady-airfoil options",
        "hub/nacelle/tower sections",
        "tower influence and aerodynamics",
        "nodal outputs",
        "SI-unit contracts",
        "file-format constraints",
        "WakeMod = 3",
        "BEM tip-loss, skew, and dynamic-model options are read and discarded",
        "horizontal-axis wind-turbine aerodynamics module",
        "not a helicopter-hover validation dataset",
        "run lists",
        "recommendations for validation use",
        "primary rotor performance files",
        "mean values",
        "uncertainty estimates",
        "atmospheric conditions",
        "blade-motion measurements",
        "rotorcraft conceptual-design and technology-impact assessment workflows",
        "fixed 15-station, stationwise-inflow, nondimensional CT/CP/FM hover BEMT stress test",
        "Do not describe this case as OpenFAST v5 validated",
        "AeroDyn validated",
        "OLAF/free-vortex-wake capable",
        "DBEMT/dynamic-inflow validated",
        "unsteady-airfoil validated",
        "tower/nacelle workflow validated",
        "OpenFAST input/output compatible",
        "HVAB performance-data validated",
        "HVAB uncertainty-product validated",
        "NDARC release-validated",
        "rotorcraft mission/trim validated",
        "grid-converged",
        "two-blade-corner validated",
        "rotor-test calibrated",
        "Evidence tier remains P2 plausibility-only",
        "No numerical response resweep is needed solely for this current-page spot check",
    ):
        assert phrase in normalized


def test_metadata_marks_only_nondimensional_design_drivers_as_important():
    case = RotorBEMT6D()

    assert case.meta.effective_dim == 4
    assert "R" not in case.meta.important_variables
    assert "Omega" not in case.meta.important_variables
    assert case.meta.noise_factor_candidates == []
    assert set(case.meta.control_factor_candidates) == {
        "c_bar",
        "theta_0",
        "theta_tw",
        "B",
    }


def test_applicable_scenarios_have_non_overclaiming_notes():
    case = RotorBEMT6D()

    assert set(case.meta.applicable_scenarios) <= set(case.meta.scenario_notes)
    vs_d = case.meta.scenario_notes["VS-D"]
    assert "small-sample" in vs_d
    assert "without pre-claiming" in vs_d
    assert "required sample count" in vs_d
    assert "B=2 is retained as a stress-test corner" in vs_d
    assert "Prandtl tip-loss accuracy is weaker" in vs_d


def test_bemt_solver_failure_is_not_silent_nan(monkeypatch):
    case = RotorBEMT6D()

    def fail_solve(*_args, **_kwargs):
        raise ValueError("simulated BEMT solve failure")

    monkeypatch.setattr(case, "_bemt", fail_solve)
    X_phys = np.array([[3.0, 350.0, 0.12, 12.0, -5.0, 4.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_bemt_malformed_or_nonphysical_response_is_not_silent(monkeypatch):
    case = RotorBEMT6D()
    X_phys = np.array([[3.0, 350.0, 0.12, 12.0, -5.0, 4.0]], dtype=float)

    def malformed(*_args, **_kwargs):
        return np.array([0.01, 0.002])

    monkeypatch.setattr(case, "_bemt", malformed)
    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)

    def nonfinite(*_args, **_kwargs):
        return np.array([0.01, np.nan, 0.7])

    monkeypatch.setattr(case, "_bemt", nonfinite)
    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)

    def negative_power(*_args, **_kwargs):
        return np.array([0.01, -0.002, 0.7])

    monkeypatch.setattr(case, "_bemt", negative_power)
    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)

    def impossible_fm(*_args, **_kwargs):
        return np.array([0.01, 0.002, 1.2])

    monkeypatch.setattr(case, "_bemt", impossible_fm)
    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_out_of_contract_rotor_bemt_response_is_not_archived(monkeypatch):
    case = RotorBEMT6D()
    X_phys = np.array([[3.0, 350.0, 0.12, 12.0, -5.0, 4.0]], dtype=float)

    out_of_contract_responses = [
        np.array([-0.026, 0.002, 0.0]),  # below CT archive envelope
        np.array([0.101, 0.002, 0.7]),  # above CT archive envelope
        np.array([0.01, 0.026, 0.7]),  # above CP archive envelope
        np.array([0.01, 0.002, 0.91]),  # above FM archive envelope
    ]

    for response in out_of_contract_responses:
        monkeypatch.setattr(case, "_bemt", lambda *_args, _response=response: _response)
        with pytest.raises(RuntimeError, match="refusing silent NaN"):
            case._evaluate_core(X_phys)


def test_private_solver_rejects_malformed_or_out_of_domain_inputs_before_bemt(monkeypatch):
    case = RotorBEMT6D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("BEMT solver should not be called for invalid physical input")

    monkeypatch.setattr(case, "_bemt", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._evaluate_core(np.array([3.0, 350.0, 0.12, 12.0, -5.0, 4.0], dtype=float))

    invalid_rows = [
        [0.49, 350.0, 0.12, 12.0, -5.0, 4.0],
        [5.01, 350.0, 0.12, 12.0, -5.0, 4.0],
        [3.0, 99.0, 0.12, 12.0, -5.0, 4.0],
        [3.0, 601.0, 0.12, 12.0, -5.0, 4.0],
        [3.0, 350.0, 0.049, 12.0, -5.0, 4.0],
        [3.0, 350.0, 0.201, 12.0, -5.0, 4.0],
        [3.0, 350.0, 0.12, 4.9, -5.0, 4.0],
        [3.0, 350.0, 0.12, 20.1, -5.0, 4.0],
        [3.0, 350.0, 0.12, 12.0, -10.1, 4.0],
        [3.0, 350.0, 0.12, 12.0, 0.1, 4.0],
        [3.0, 350.0, 0.12, 12.0, -5.0, 1.99],
        [3.0, 350.0, 0.12, 12.0, -5.0, 6.01],
        [3.0, 350.0, np.nan, 12.0, -5.0, 4.0],
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError, match="R/Omega/c_bar/theta_0/theta_tw/B benchmark domain"):
            case._evaluate_core(np.array([row], dtype=float))
