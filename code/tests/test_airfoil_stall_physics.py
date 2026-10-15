"""Physics regression tests for ``airfoil_stall_2d``."""

from __future__ import annotations

import inspect
import itertools
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from doe_testbed.cases.aerodynamics.airfoil_stall_2d import AirfoilStall2D


def _domain_corners(case: AirfoilStall2D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_neuralfoil_backend_is_used_when_available():
    case = AirfoilStall2D()
    X = np.array([[20.0, 5e6, 0.4, 0.18]])

    neural = case._eval_neuralfoil(X)
    analytical = case._eval_analytical(X)

    assert neural[0, 0] > 1.5
    assert not np.allclose(neural, analytical)


def test_evaluate_core_requires_neuralfoil_backend(monkeypatch):
    case = AirfoilStall2D()
    X = np.array([[10.0, 1e6, 0.05, 0.12]], dtype=float)

    def fail_backend(_X):
        raise ImportError("simulated missing neuralfoil")

    def analytical_must_not_run(_X):
        raise AssertionError("analytical fallback must not be used by evaluate")

    monkeypatch.setattr(case, "_eval_neuralfoil", fail_backend)
    monkeypatch.setattr(case, "_eval_analytical", analytical_must_not_run)

    with pytest.raises(ImportError, match="simulated missing neuralfoil"):
        case._evaluate_core(X)


def test_neuralfoil_point_failure_does_not_fallback_to_analytical(monkeypatch):
    case = AirfoilStall2D()
    X = np.array([[10.0, 1e6, 0.05, 0.12]], dtype=float)

    class DummyAirfoil:
        def __init__(self, name):
            self.name = name

    def fail_aero(**_kwargs):
        raise ValueError("simulated neuralfoil point failure")

    def analytical_must_not_run(_X):
        raise AssertionError("analytical fallback must not be used by evaluate")

    monkeypatch.setitem(sys.modules, "aerosandbox", types.SimpleNamespace(Airfoil=DummyAirfoil))
    monkeypatch.setitem(
        sys.modules,
        "neuralfoil",
        types.SimpleNamespace(get_aero_from_airfoil=fail_aero),
    )
    monkeypatch.setattr(case, "_eval_analytical", analytical_must_not_run)

    with pytest.raises(RuntimeError, match="refusing analytical fallback"):
        case._eval_neuralfoil(X)


def test_neuralfoil_nonfinite_or_nonpositive_polar_is_rejected(monkeypatch):
    case = AirfoilStall2D()

    class DummyAirfoil:
        def __init__(self, name):
            self.name = name

    def nonfinite_aero(**_kwargs):
        return {"CL": np.array([np.nan]), "CD": np.array([0.0])}

    monkeypatch.setitem(sys.modules, "aerosandbox", types.SimpleNamespace(Airfoil=DummyAirfoil))
    monkeypatch.setitem(
        sys.modules,
        "neuralfoil",
        types.SimpleNamespace(get_aero_from_airfoil=nonfinite_aero),
    )

    X = np.array([[10.0, 1e6, 0.05, 0.12]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing analytical fallback"):
        case._eval_neuralfoil(X)


def test_private_neuralfoil_backend_rejects_malformed_or_out_of_domain_inputs_before_import(monkeypatch):
    case = AirfoilStall2D()
    assert hasattr(case, "_validate_physical_inputs")

    class BackendMustNotLoad:
        def __getattr__(self, name):
            raise AssertionError(f"backend import should not be used for invalid input: {name}")

    monkeypatch.setitem(sys.modules, "aerosandbox", BackendMustNotLoad())
    monkeypatch.setitem(sys.modules, "neuralfoil", BackendMustNotLoad())

    with pytest.raises(ValueError, match="shape"):
        case._eval_neuralfoil(np.array([10.0, 1e6, 0.05, 0.12], dtype=float))

    invalid_rows = [
        [-5.1, 1e6, 0.05, 0.12],
        [20.1, 1e6, 0.05, 0.12],
        [10.0, 4.9e4, 0.05, 0.12],
        [10.0, 5.1e6, 0.05, 0.12],
        [10.0, 1e6, -0.01, 0.12],
        [10.0, 1e6, 0.41, 0.12],
        [10.0, 1e6, 0.05, 0.059],
        [10.0, 1e6, 0.05, 0.181],
        [10.0, np.nan, 0.05, 0.12],
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError, match="alpha/Re/Ma/t_c benchmark domain"):
            case._eval_neuralfoil(np.array([row], dtype=float))


def test_neuralfoil_backend_maps_thickness_to_naca00xx_and_squeezes_arrays(monkeypatch):
    case = AirfoilStall2D()
    observed_airfoils: list[str] = []

    class DummyAirfoil:
        def __init__(self, name):
            self.name = name
            observed_airfoils.append(name)

    def fake_aero(**kwargs):
        assert kwargs["model_size"] == "large"
        assert isinstance(kwargs["alpha"], float)
        assert isinstance(kwargs["Re"], float)
        return {"CL": np.array([1.2]), "CD": np.array([0.03])}

    monkeypatch.setitem(sys.modules, "aerosandbox", types.SimpleNamespace(Airfoil=DummyAirfoil))
    monkeypatch.setitem(
        sys.modules,
        "neuralfoil",
        types.SimpleNamespace(get_aero_from_airfoil=fake_aero),
    )

    X = np.array(
        [
            [5.0, 1e6, 0.0, 0.06],
            [5.0, 1e6, 0.4, 0.18],
        ],
        dtype=float,
    )
    Y = case._eval_neuralfoil(X)

    beta = np.sqrt(1.0 - X[:, 2] ** 2)
    assert observed_airfoils == ["naca0006", "naca0018"]
    assert np.allclose(Y[:, 0], 1.2 / beta, rtol=1e-12, atol=1e-12)
    assert np.allclose(Y[:, 1], 0.03 / beta, rtol=1e-12, atol=1e-12)
    assert np.allclose(Y[:, 2], 40.0, rtol=1e-12, atol=1e-12)


def test_attached_flow_lift_increases_with_angle_of_attack():
    case = AirfoilStall2D()
    X_phys = np.array(
        [
            [0.0, 1e6, 0.05, 0.12],
            [5.0, 1e6, 0.05, 0.12],
            [10.0, 1e6, 0.05, 0.12],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert abs(Y[0, 0]) < 0.1
    assert np.all(np.diff(Y[:, 0]) > 0.0)
    assert Y[1, 2] > 20.0


def test_low_alpha_lift_slope_matches_thin_airfoil_reference():
    case = AirfoilStall2D()
    X_phys = np.array(
        [
            [0.0, 3e6, 0.0, 0.12],
            [5.0, 3e6, 0.0, 0.12],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    slope_per_rad = (Y[1, 0] - Y[0, 0]) / np.radians(5.0)

    assert abs(Y[0, 0]) < 1e-6
    assert np.isclose(slope_per_rad, 2.0 * np.pi, rtol=0.04)


def test_low_speed_mach_scaling_preserves_lift_to_drag_ratio():
    case = AirfoilStall2D()
    X_phys = np.array(
        [
            [5.0, 1e6, 0.0, 0.12],
            [5.0, 1e6, 0.4, 0.12],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    beta = np.sqrt(1.0 - X_phys[:, 2] ** 2)
    expected_ratio = beta[0] / beta[1]

    assert np.isclose(Y[1, 0] / Y[0, 0], expected_ratio, rtol=1e-12, atol=1e-12)
    assert np.isclose(Y[1, 1] / Y[0, 1], expected_ratio, rtol=1e-12, atol=1e-12)
    assert np.isclose(Y[1, 2], Y[0, 2], rtol=1e-12, atol=1e-12)


def test_post_stall_drag_surge_reduces_lift_to_drag():
    case = AirfoilStall2D()
    X_phys = np.array(
        [
            [10.0, 1e6, 0.05, 0.12],
            [18.0, 1e6, 0.05, 0.12],
        ],
        dtype=float,
    )
    pre_stall, post_stall = case.evaluate(case.normalize(X_phys))

    assert post_stall[1] > pre_stall[1]
    assert post_stall[2] < pre_stall[2]


def test_high_drag_stall_failure_threshold_targets_separated_region():
    case = AirfoilStall2D()
    threshold = case.meta.output_variables[1].failure_threshold
    assert threshold == 0.05
    assert case.meta.output_variables[1].failure_direction == "above"
    assert case.meta.output_variables[0].failure_threshold is None

    X_phys = np.array(
        [
            [0.0, 3e6, 0.05, 0.12],
            [10.0, 3e6, 0.05, 0.12],
            [20.0, 3e6, 0.05, 0.12],
        ],
        dtype=float,
    )
    zero_alpha, attached, high_drag = case.evaluate(case.normalize(X_phys))

    assert zero_alpha[1] < threshold
    assert attached[1] < threshold
    assert high_drag[1] > threshold
    assert high_drag[2] < attached[2]


def test_metadata_declares_neuralfoil_plus_low_order_pg_scope():
    case = AirfoilStall2D()
    equation = case.meta.governing_equation
    details = case.meta.solver_details

    assert hasattr(case, "_validate_physical_inputs")
    assert "NeuralFoil baseline" in equation
    assert "Prandtl-Glauert" in equation
    assert "1/sqrt(1-Ma^2)" in details
    assert "not Mach-resolved airfoil validation" in details
    assert "Low-level neuralfoil.get_aero_from_airfoil" in details
    assert "Airfoil.get_aero_from_neuralfoil wrapper exposes a mach argument" in details
    assert "changing that backend is a response-surface change" in details
    assert "Named _validate_physical_inputs" in details
    assert "rejects values outside the declared low-speed airfoil domain" in details
    assert "diagnostic-only" in details
    assert "single-section NACA 0012 context" in details
    assert "full NACA00xx thickness sweep" in details
    assert "explicit CD/beta scaling" in details
    assert "NASA Glenn inclination/stall pages" in details
    assert "experimental stall-point sensitivity" in details
    assert case.meta.typical_eval_time_ms["default"] >= 15.0
    assert "low-subsonic/mild-compressibility stressor" in case.meta.input_variables[2].description
    assert any("AeroSandbox Airfoil API documentation" in ref for ref in case.meta.physics_references)
    assert any("https://arxiv.org/abs/2503.16323" in ref for ref in case.meta.physics_references)
    assert any("NASA TM-4074" in ref for ref in case.meta.physics_references)
    assert any("NACA 0012 Airfoil Validation" in ref for ref in case.meta.physics_references)
    assert any("NASA Turbulence Modeling Resource" in ref for ref in case.meta.physics_references)
    assert any("NASA Glenn Beginner's Guide" in ref for ref in case.meta.physics_references)
    assert any("incline.html" in ref and "inclind.html" in ref for ref in case.meta.physics_references)
    assert any("Sheldahl" in ref and "SAND80-2114" in ref for ref in case.meta.physics_references)
    assert any("XFOIL online documentation" in ref for ref in case.meta.physics_references)
    assert any("https://web.mit.edu/drela/Public/web/xfoil/" in ref for ref in case.meta.physics_references)
    for omitted_scope in [
        "No cambered airfoils",
        "flaps/control surfaces",
        "roughness/trip/transition-control sweep",
        "turbulence-model selection",
        "RANS/LES grid-convergence validation",
        "dynamic-stall hysteresis",
        "unsteady pitching/plunging motion",
        "3-D finite-wing effect",
        "rotating-blade correction",
        "icing/contamination",
        "aeroelastic coupling",
        "acoustic model",
    ]:
        assert omitted_scope in details


def test_installed_neuralfoil_airfoil_api_has_no_mach_argument():
    """Pin the local backend boundary that motivates the explicit PG factor."""
    import neuralfoil as nf

    signature = inspect.signature(nf.get_aero_from_airfoil)

    assert "Ma" not in signature.parameters
    assert "mach" not in {name.lower() for name in signature.parameters}


def test_aerosandbox_wrapper_mach_argument_is_not_the_formal_backend():
    """Document the deliberate backend choice if AeroSandbox/NeuralFoil APIs evolve."""
    import aerosandbox as asb

    case = AirfoilStall2D()
    wrapper_signature = inspect.signature(asb.Airfoil.get_aero_from_neuralfoil)

    assert "mach" in wrapper_signature.parameters
    assert "Low-level neuralfoil.get_aero_from_airfoil" in case.meta.solver_details
    assert "formal response surface is pinned" in case.meta.solver_details


def test_scenario_notes_do_not_preclaim_sequential_sample_savings():
    case = AirfoilStall2D()
    source_text = Path("doe_testbed/cases/aerodynamics/airfoil_stall_2d.py").read_text(
        encoding="utf-8"
    )
    vs_g_note = case.meta.scenario_notes["VS-G"]

    assert "30–50" not in source_text
    assert "30-50" not in source_text
    assert "should locate the boundary" not in source_text
    assert "test whether sequential methods" in source_text
    assert "more efficiently than space-filling designs" in vs_g_note


def test_validation_material_declares_qualitative_surrogate_scope():
    validation_text = Path("tests/_cross_validation/airfoil_stall_2d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path("doe_testbed/cases/aerodynamics/airfoil_stall_2d.py").read_text(
        encoding="utf-8"
    )
    normalized_source = " ".join(source_text.split())

    assert "qualitative low-alpha lift-slope" in validation_text
    assert "quantitative external validation" in validation_text
    assert "5-10%" not in validation_text
    assert "Target accuracy" not in validation_text
    assert "case-specific XFOIL" in source_text
    assert "RANS, or wind-tunnel validation claim" in source_text
    assert "single-section NACA 0012 low-speed Mach/Reynolds context" in source_text
    assert "validation-dataset boundary for higher-fidelity CFD workflows" in normalized_source
    assert "low-level ``neuralfoil.get_aero_from_airfoil`` call" in validation_text
    assert "switching to that wrapper" in validation_text
    assert "formal backend" in validation_text
    assert "full NACA00xx thickness sweep" in validation_text
    assert "explicit ``CD/beta`` scaling" in validation_text
    assert "frozen low-level NeuralFoil backend" in validation_text
    assert "NASA Glenn inclination/stall pages" in validation_text
    assert "boundary anchor only" in validation_text
    for omitted_scope in [
        "cambered sections",
        "flaps/control-surface deflection",
        "surface roughness",
        "trip/transition-control sweeps",
        "turbulence-model selection",
        "RANS/LES grid convergence",
        "dynamic-stall hysteresis",
        "unsteady pitching/plunging",
        "3-D finite-wing",
        "rotating-blade correction",
        "icing/contamination",
        "aeroelastic coupling",
        "acoustic effects",
    ]:
        assert omitted_scope in validation_text


def test_reviewer_markdown_note_keeps_airfoil_stall_p2_scope_boundary():
    note_text = Path("tests/_cross_validation/airfoil_stall_2d.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(note_text.split())

    required_phrases = [
        "P2 plausibility-only",
        "No response-surface change",
        "Low-level NeuralFoil baseline polar",
        "explicit low-subsonic Prandtl-Glauert scaling",
        "not externally validated NACA0012 wind-tunnel prediction",
        "not Mach-resolved airfoil validation",
        "not XFOIL-calibrated",
        "not RANS/LES validated",
        "not transition/turbulence validated",
        "not dynamic-stall validated",
        "not rotating-blade validated",
        "not a 3-D finite-wing validation case",
        "https://turbmodels.larc.nasa.gov/naca0012_val.html",
        "https://www.grc.nasa.gov/WWW/K-12/VirtualAero/BottleRocket/airplane/incline.html",
        "https://www.grc.nasa.gov/WWW/k-12/VirtualAero/BottleRocket/airplane/inclind.html",
        "https://web.mit.edu/drela/Public/web/xfoil/",
        "https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/index.html",
        "https://arxiv.org/abs/2503.16323",
    ]
    for phrase in required_phrases:
        assert phrase in normalized

    for forbidden_phrase in [
        "Evidence tier | P1",
        "Evidence tier | P0",
        "Evidence tier | externally validated",
    ]:
        assert forbidden_phrase not in normalized


def test_web_audit_records_official_source_boundary_recheck():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    marker = "Additional `airfoil_stall_2d` 2026-06-08 official-source re-check"
    start = audit_text.index(marker)
    section = audit_text[start : audit_text.index("Additional `airfoil_transonic_5d`", start)]
    normalized = " ".join(section.split())

    assert "NASA TM-4074" in normalized
    assert "low-speed NACA 0012 Mach/Reynolds-number context" in normalized
    assert "not as validation data" in normalized
    assert "NASA TMR NACA 0012 pages as RANS validation dataset" in normalized
    assert "Sheldahl/Klimas SAND80-2114" in normalized
    assert "MIT/XFOIL documentation" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    for forbidden_claim in [
        "externally validated NACA0012 wind-tunnel data",
        "Mach-resolved airfoil validation",
        "XFOIL-calibrated prediction",
        "RANS/LES validation",
        "transition/turbulence validation",
        "dynamic-stall validation",
        "rotating-blade validation",
        "3-D finite-wing validation",
    ]:
        assert forbidden_claim in normalized
    assert (
        "Additional `airfoil_stall_2d` 2026-06-08 single-section boundary tightening:"
        in section
    )
    assert "full NACA00xx thickness sweep" in normalized
    assert "explicit `CD/beta` scaling" in normalized
    assert "frozen low-level NeuralFoil backend" in normalized
    assert "No numerical response resweep is needed solely for this official-source re-check" in normalized
    assert "Additional `airfoil_stall_2d` 2026-06-09 NASA Glenn stall-boundary note" in section
    assert "near-linear small-angle lift" in normalized
    assert "separated-flow stall" in normalized
    assert "high-angle drag rise" in normalized
    assert "experimental stall-point sensitivity" in normalized
    assert "not calibration evidence for the frozen NeuralFoil-plus-explicit-PG response surface" in normalized


def test_web_audit_records_airfoil_stall_current_page_spot_check():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    marker = "Additional `airfoil_stall_2d` 2026-06-09 current-page spot check"
    start = audit_text.index(marker)
    section = audit_text[start : audit_text.index("Additional `airfoil_transonic_5d`", start)]
    normalized = " ".join(section.split())

    assert "current NeuralFoil, AeroSandbox, NASA, and MIT pages were re-opened" in normalized
    assert "NeuralFoil arXiv page" in normalized
    assert "AeroSandbox airfoil API page" in normalized
    assert "Airfoil.get_aero_from_neuralfoil(..., mach=0.0, ...)" in normalized
    assert "lower-level `neuralfoil.get_aero_from_airfoil` backend" in normalized
    assert "NASA TM-4074 remains NACA 0012 low-speed Mach/Reynolds tunnel context" in normalized
    assert "NASA TMR remains higher-fidelity CFD/turbulence validation context" in normalized
    assert "NASA Glenn inclination pages remain general stall/lift/drag boundary context" in normalized
    assert "MIT XFOIL remains panel/integral-boundary layer tool context" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    assert "No numerical response resweep is needed solely for this current-page spot check" in normalized
    for forbidden_claim in [
        "externally validated NACA0012 wind-tunnel data",
        "Mach-resolved NeuralFoil/AeroSandbox validation",
        "XFOIL-calibrated prediction",
        "RANS/LES validated stall prediction",
        "transition/turbulence validated",
        "dynamic-stall validated",
        "rotating-blade validated",
        "3-D finite-wing validated",
    ]:
        assert forbidden_claim in normalized


def test_reynolds_number_improves_attached_flow_efficiency():
    case = AirfoilStall2D()
    X_phys = np.array(
        [
            [10.0, 5e4, 0.05, 0.12],
            [10.0, 5e6, 0.05, 0.12],
        ],
        dtype=float,
    )
    low_re, high_re = case.evaluate(case.normalize(X_phys))

    assert high_re[0] > low_re[0]
    assert high_re[1] < low_re[1]
    assert high_re[2] > low_re[2]


def test_declared_output_ranges_cover_domain_corners():
    case = AirfoilStall2D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper
