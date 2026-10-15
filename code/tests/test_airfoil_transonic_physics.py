"""Physics regression tests for ``airfoil_transonic_5d``."""

from __future__ import annotations

import inspect
import itertools
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from doe_testbed.cases.aerodynamics.airfoil_transonic_5d import AirfoilTransonic5D


def _domain_corners(case: AirfoilTransonic5D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_declared_output_ranges_cover_domain_corners():
    case = AirfoilTransonic5D()
    X = case.normalize(_domain_corners(case))
    Y = case.evaluate(X)

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_metadata_declares_engineering_transonic_surrogate_scope():
    case = AirfoilTransonic5D()
    equation = case.meta.governing_equation.lower()
    details = case.meta.solver_details.lower()

    assert case.meta.pde_type == "algebraic"
    assert "beta_pg" in equation
    assert "cd_wave" in equation
    assert "not direct rans/sa" in equation
    assert "prandtl-glauert" in details
    assert "empirical wave-drag" in details
    assert "not a direct mach-resolved" in details
    assert "low-level neuralfoil.get_aero_from_airfoil" in details
    assert "airfoil.get_aero_from_neuralfoil wrapper exposes a mach argument" in details
    assert "changing that backend is a response-surface change" in details
    assert "rejects values outside the declared transonic airfoil domain" in details
    assert "no derivative-based m_dd" in details
    assert "korn relation" in details
    assert "buffet onset" in details
    assert "shock/boundary-layer separation" in details
    assert "urans/ddes/wmles/fun3d" in details
    assert "shock-location/buffet behavior" in details
    assert "single-section context" in details
    assert "camber/thickness sweep" in details
    assert "fourth-power wave-drag trigger" in details
    assert "similarity-parameter material supports the mach/compressibility" in details
    assert "not calibration of the critical-mach trigger" in details
    assert hasattr(case, "_validate_physical_inputs")
    assert any("AeroSandbox Airfoil API documentation" in ref for ref in case.meta.physics_references)
    assert any("https://arxiv.org/abs/2503.16323" in ref for ref in case.meta.physics_references)
    assert any("NASA-TM-100527" in ref for ref in case.meta.physics_references)
    assert any("NASA-TM-81927" in ref for ref in case.meta.physics_references)
    assert any("https://ntrs.nasa.gov/citations/19810014503" in ref for ref in case.meta.physics_references)
    assert any("drag-divergence" in ref for ref in case.meta.physics_references)
    assert any("similarity parameters" in ref for ref in case.meta.physics_references)
    assert any("similarity-parameters-2" in ref for ref in case.meta.physics_references)
    assert any("turbmodels.larc.nasa.gov/naca0012_val.html" in ref for ref in case.meta.physics_references)
    assert any("20220017612" in ref for ref in case.meta.physics_references)
    assert any("20220018292" in ref for ref in case.meta.physics_references)
    assert any("XFOIL online documentation" in ref for ref in case.meta.physics_references)
    assert any("https://web.mit.edu/drela/Public/web/xfoil/" in ref for ref in case.meta.physics_references)


def test_sobol_metadata_distinguishes_cd_output_from_multioutput_importance():
    case = AirfoilTransonic5D()
    sobol_note = case.meta.sobol_computation
    vs_f_note = case.meta.scenario_notes["VS-F"]

    assert "CD_output" in sobol_note
    assert "multi-output" in sobol_note
    assert set(case.meta.important_variables) == {"Ma", "alpha", "tc", "fc", "Re"}
    assert case.meta.effective_dim == 5
    assert case.meta.sobol_ST["Ma"] > 0.9
    assert case.meta.sobol_ST["tc"] > 0.05
    assert case.meta.sobol_ST["alpha"] < 0.05
    assert case.meta.sobol_ST["fc"] < 0.05
    assert case.meta.sobol_ST["Re"] < 0.05
    assert "stored CD Sobol metadata" in vs_f_note
    assert "case-level important-variable list is a multi-output union" in vs_f_note


def test_scenario_notes_do_not_preclaim_adaptive_sampling_success():
    case = AirfoilTransonic5D()
    source_text = Path("doe_testbed/cases/aerodynamics/airfoil_transonic_5d.py").read_text(
        encoding="utf-8"
    )

    assert "should automatically focus" not in source_text
    assert "are evaluated on whether" in source_text
    assert "test whether" in case.meta.scenario_notes["VS-G"]


def test_validation_material_declares_qualitative_transonic_surrogate_scope():
    case = AirfoilTransonic5D()
    validation_text = Path("tests/_cross_validation/airfoil_transonic_5d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path("doe_testbed/cases/aerodynamics/airfoil_transonic_5d.py").read_text(
        encoding="utf-8"
    )

    assert "qualitative" in validation_text
    assert "range/trend controls" in validation_text
    assert "not a claim of 10 % Mach-resolved" in validation_text
    assert "transonic CFD accuracy" in validation_text
    assert "Mach-resolved NeuralFoil" in validation_text
    assert "Euler, RANS" in validation_text
    assert "wind-tunnel validation" in validation_text
    assert "low-level ``neuralfoil.get_aero_from_airfoil`` call" in validation_text
    assert "switching to the wrapper would be a formal backend change" in validation_text
    assert "not a derivative M_dd solver" in validation_text
    assert "not a Korn-relation" in validation_text
    assert "not buffet-onset" in validation_text
    assert "not shock/separation validation" in validation_text
    normalized_validation = " ".join(validation_text.split())
    normalized_source = " ".join(source_text.split())
    assert "specific NACA 0012 sections and test conditions" in normalized_source
    assert "camber/thickness sweep" in normalized_source
    assert "fourth-power wave-drag trigger" in normalized_source
    assert "camber/thickness sweep" in normalized_validation
    assert "fourth-power wave-drag trigger" in normalized_validation
    for source_anchor in [
        "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19880009187.pdf",
        "https://www.nasa.gov/history/SP-4219/Chapter3.html",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/similarity-parameters-2/",
        "https://turbmodels.larc.nasa.gov/naca0012_val.html",
        "https://ntrs.nasa.gov/citations/20220017612",
        "https://ntrs.nasa.gov/citations/20220018292",
    ]:
        assert source_anchor in source_text
        assert source_anchor in validation_text
    assert "public NASA anchors support transonic NACA 0012 test-data context" in normalized_source
    assert "They are not external validation data for this NeuralFoil-baseline" in normalized_source
    assert "The NASA anchors are transonic-test, drag-divergence-phenomenology" in normalized_validation
    assert "not external validation data for this NeuralFoil-baseline" in normalized_validation
    assert "NASA Glenn similarity-parameter material supports the Mach/compressibility" in normalized_validation
    assert "does not calibrate the critical-Mach trigger" in normalized_validation
    assert "pre/post-buffet operating conditions" in normalized_validation
    assert "URANS, DDES, WMLES, FUN3D" in normalized_validation
    assert "shock-location comparisons" in normalized_validation
    assert "do not validate its compact wave-drag trigger" in normalized_validation
    assert "not a high-fidelity transonic CFD/RANS replacement" in source_text
    assert "switching this frozen benchmark response surface" in normalized_source
    assert "does not solve that derivative definition" in source_text
    assert "not a direct Mach-resolved" in case.meta.solver_details
    assert "Target accuracy" not in validation_text
    assert "externally validated" not in validation_text


def test_reviewer_markdown_note_keeps_transonic_airfoil_p2_scope_boundary():
    note_text = Path("tests/_cross_validation/airfoil_transonic_5d.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(note_text.split())

    required_phrases = [
        "P2 plausibility-only",
        "No response-surface change",
        "NeuralFoil baseline polar",
        "explicit Prandtl-Glauert",
        "empirical wave-drag",
        "not externally validated transonic CFD/RANS",
        "not Mach-resolved NeuralFoil",
        "not an AeroSandbox-wrapper Mach model",
        "not a derivative `M_dd` solver",
        "not Korn",
        "not buffet",
        "not shock/boundary-layer separation",
        "not wind-tunnel calibrated",
        "https://ntrs.nasa.gov/search.jsp?R=19810014503",
        "https://ntrs.nasa.gov/citations/19810014503",
        "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19880009187.pdf",
        "https://www.nasa.gov/history/SP-4219/Chapter3.html",
        "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/similarity-parameters-2/",
        "https://turbmodels.larc.nasa.gov/naca0012_val.html",
        "https://ntrs.nasa.gov/citations/20220017612",
        "https://ntrs.nasa.gov/citations/20220018292",
        "https://web.mit.edu/drela/Public/web/xfoil/",
        "https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/index.html",
        "https://arxiv.org/abs/2503.16323",
        "URANS, DDES, WMLES, FUN3D",
        "shock-location validation",
    ]
    for phrase in required_phrases:
        assert phrase in normalized

    forbidden_phrases = [
        "Evidence tier | P1",
        "Evidence tier | P0",
        "Mach-resolved NeuralFoil validated",
        "wind-tunnel validated response surface",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in normalized


def test_backend_contract_distinguishes_low_level_api_from_aerosandbox_wrapper():
    import aerosandbox as asb
    import neuralfoil as nf

    case = AirfoilTransonic5D()
    low_level_signature = inspect.signature(nf.get_aero_from_airfoil)
    wrapper_signature = inspect.signature(asb.Airfoil.get_aero_from_neuralfoil)

    assert "mach" not in {name.lower() for name in low_level_signature.parameters}
    assert "mach" in wrapper_signature.parameters
    assert "formal response surface is pinned" in case.meta.solver_details


def test_web_audit_keeps_transonic_airfoil_scope_as_trend_not_validation():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `airfoil_transonic_5d` |")
        and "Harris NASA TM-81927 supports NACA 0012 transonic" in line
    )

    assert "Evidence tier remains P2" in row
    assert "NASA TM-100527 provides additional high-Reynolds-number transonic NACA 0012 test context" in row
    assert "NASA history material supports the drag-divergence phenomenon" in row
    assert "not 10 % Mach-resolved CFD accuracy" in row
    assert "not a high-fidelity transonic CFD replacement" in row
    assert "does not solve the derivative definition" in row
    assert "does not implement Korn's drag-divergence relation" in row
    assert "buffet onset" in row
    assert "shock/boundary-layer separation" in row
    assert "Keep P2 plausibility-only" in row
    assert "Do not describe as externally validated" in row


def test_web_audit_records_official_source_boundary_recheck():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    marker = "Additional `airfoil_transonic_5d` 2026-06-08 official-source re-check"
    start = audit_text.index(marker)
    section = audit_text[start : audit_text.index("Additional `waverider_4d`", start)]
    normalized = " ".join(section.split())

    assert "NASA TM-81927" in normalized
    assert "NASA TM-100527" in normalized
    assert "transonic NACA 0012 drag-divergence" in normalized
    assert "not validate the benchmark's NeuralFoil-baseline" in normalized
    assert "fourth-power wave-drag surrogate" in normalized
    assert "Evidence tier remains P2 plausibility-only" in normalized
    for forbidden_claim in [
        "externally validated transonic CFD/RANS",
        "NASA tunnel-data matched",
        "Mach-resolved NeuralFoil/AeroSandbox validated",
        "derivative-Mdd validated",
        "Korn/buffet validated",
        "shock/separation validated",
    ]:
        assert forbidden_claim in normalized
    assert (
        "Additional `airfoil_transonic_5d` 2026-06-08 single-section boundary tightening:"
        in normalized
    )
    assert "camber/thickness sweep" in normalized
    assert "fourth-power wave-drag trigger" in normalized
    assert "No numerical response resweep is needed solely for this official-source re-check" in normalized
    assert "Additional `airfoil_transonic_5d` 2026-06-09 NASA transonic-buffet workflow boundary note" in normalized
    assert "pre/post-buffet operating conditions" in normalized
    assert "URANS, DDES, WMLES" in normalized
    assert "NASA FUN3D" in normalized
    assert "shock-location comparisons" in normalized
    assert "does not run FUN3D" in normalized
    assert "predict buffet onset" in normalized
    assert "validate shock location" in normalized
    assert "wall/blockage corrections" in normalized
    assert "Additional `airfoil_transonic_5d` 2026-06-09 NASA Glenn similarity-parameter boundary note" in normalized
    assert "compressibility similarity parameter" in normalized
    assert "near/transonic shock-wave effects" in normalized
    assert "not calibration evidence for the frozen NeuralFoil-baseline" in normalized
    assert "does not validate the compact critical-Mach trigger" in normalized


def test_neuralfoil_point_failure_is_not_silent_nan(monkeypatch):
    case = AirfoilTransonic5D()
    X = np.array([[0.70, 2.0, 0.12, 0.02, 5e6]], dtype=float)

    class DummyAirfoil:
        def __init__(self, name):
            self.name = name

    def fail_aero(**_kwargs):
        raise ValueError("simulated transonic backend failure")

    monkeypatch.setitem(sys.modules, "aerosandbox", types.SimpleNamespace(Airfoil=DummyAirfoil))
    monkeypatch.setitem(
        sys.modules,
        "neuralfoil",
        types.SimpleNamespace(get_aero_from_airfoil=fail_aero),
    )

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X)


def test_neuralfoil_nonfinite_response_is_not_silent_nan(monkeypatch):
    case = AirfoilTransonic5D()
    X = np.array([[0.70, 2.0, 0.12, 0.02, 5e6]], dtype=float)

    class DummyAirfoil:
        def __init__(self, name):
            self.name = name

    def nonfinite_aero(**_kwargs):
        return {"CL": np.array([np.nan]), "CD": np.array([0.03])}

    monkeypatch.setitem(sys.modules, "aerosandbox", types.SimpleNamespace(Airfoil=DummyAirfoil))
    monkeypatch.setitem(
        sys.modules,
        "neuralfoil",
        types.SimpleNamespace(get_aero_from_airfoil=nonfinite_aero),
    )

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X)


def test_neuralfoil_nonpositive_drag_response_is_not_silently_clipped(monkeypatch):
    case = AirfoilTransonic5D()
    X = np.array([[0.70, 2.0, 0.12, 0.02, 5e6]], dtype=float)

    class DummyAirfoil:
        def __init__(self, name):
            self.name = name

    def negative_drag_aero(**_kwargs):
        return {"CL": np.array([0.8]), "CD": np.array([-0.01])}

    monkeypatch.setitem(sys.modules, "aerosandbox", types.SimpleNamespace(Airfoil=DummyAirfoil))
    monkeypatch.setitem(
        sys.modules,
        "neuralfoil",
        types.SimpleNamespace(get_aero_from_airfoil=negative_drag_aero),
    )

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X)


def test_private_backend_rejects_malformed_or_out_of_domain_inputs_before_import(monkeypatch):
    case = AirfoilTransonic5D()

    class BackendMustNotLoad:
        def __getattr__(self, name):
            raise AssertionError(f"backend import should not be used for invalid input: {name}")

    monkeypatch.setitem(sys.modules, "aerosandbox", BackendMustNotLoad())
    monkeypatch.setitem(sys.modules, "neuralfoil", BackendMustNotLoad())

    with pytest.raises(ValueError, match="shape"):
        case._evaluate_core(np.array([0.70, 2.0, 0.12, 0.02, 5e6], dtype=float))

    invalid_rows = [
        [0.29, 2.0, 0.12, 0.02, 5e6],
        [0.89, 2.0, 0.12, 0.02, 5e6],
        [0.70, -2.1, 0.12, 0.02, 5e6],
        [0.70, 6.1, 0.12, 0.02, 5e6],
        [0.70, 2.0, 0.079, 0.02, 5e6],
        [0.70, 2.0, 0.181, 0.02, 5e6],
        [0.70, 2.0, 0.12, -0.001, 5e6],
        [0.70, 2.0, 0.12, 0.061, 5e6],
        [0.70, 2.0, 0.12, 0.02, 0.99e6],
        [0.70, 2.0, 0.12, 0.02, 1.01e7],
        [0.70, 2.0, np.nan, 0.02, 5e6],
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError, match="Ma/alpha/tc/fc/Re benchmark domain"):
            case._evaluate_core(np.array([row], dtype=float))


def test_neuralfoil_backend_contract_maps_naca_geometry_and_applies_external_corrections(
    monkeypatch,
):
    case = AirfoilTransonic5D()
    observed_airfoils: list[str] = []

    class DummyAirfoil:
        def __init__(self, name):
            self.name = name
            observed_airfoils.append(name)

    def fake_aero(**kwargs):
        assert kwargs["model_size"] == "large"
        assert isinstance(kwargs["alpha"], float)
        assert isinstance(kwargs["Re"], float)
        assert "Mach" not in kwargs
        assert "Ma" not in kwargs
        return {"CL": np.array([1.2]), "CD": np.array([0.03])}

    monkeypatch.setitem(sys.modules, "aerosandbox", types.SimpleNamespace(Airfoil=DummyAirfoil))
    monkeypatch.setitem(
        sys.modules,
        "neuralfoil",
        types.SimpleNamespace(get_aero_from_airfoil=fake_aero),
    )

    X = np.array(
        [
            [0.30, 2.0, 0.08, 0.02, 5e6],
            [0.80, 2.0, 0.18, 0.06, 5e6],
        ],
        dtype=float,
    )
    Y = case._evaluate_core(X)

    beta = np.sqrt(1.0 - X[:, 0] ** 2)
    mcr = 0.75 - 0.06 * X[:, 2] / 0.12
    cd_wave = np.where(X[:, 0] > mcr, 20.0 * (X[:, 0] - mcr) ** 4, 0.0)
    expected_cd = 0.03 / beta + cd_wave
    expected_cl = 1.2 / beta

    assert observed_airfoils == ["naca2408", "naca6418"]
    assert np.allclose(Y[:, 0], expected_cd, rtol=1e-12, atol=1e-12)
    assert np.allclose(Y[:, 1], expected_cl, rtol=1e-12, atol=1e-12)
    assert np.allclose(Y[:, 2], expected_cl / expected_cd, rtol=1e-12, atol=1e-12)


def test_high_lift_transonic_response_is_not_silently_clipped(monkeypatch):
    case = AirfoilTransonic5D()

    class DummyAirfoil:
        def __init__(self, name):
            self.name = name

    def fake_aero(**_kwargs):
        return {"CL": np.array([1.8]), "CD": np.array([0.04])}

    monkeypatch.setitem(sys.modules, "aerosandbox", types.SimpleNamespace(Airfoil=DummyAirfoil))
    monkeypatch.setitem(
        sys.modules,
        "neuralfoil",
        types.SimpleNamespace(get_aero_from_airfoil=fake_aero),
    )

    X = np.array([[0.80, 6.0, 0.18, 0.06, 5e6]], dtype=float)
    Y = case._evaluate_core(X)
    beta = np.sqrt(1.0 - X[0, 0] ** 2)
    expected_cl = 1.8 / beta
    mcr = 0.75 - 0.06 * X[0, 2] / 0.12
    expected_cd = 0.04 / beta + 20.0 * (X[0, 0] - mcr) ** 4

    assert expected_cl > 2.5
    assert np.isclose(Y[0, 1], expected_cl, rtol=1e-12, atol=1e-12)
    assert np.isclose(Y[0, 0], expected_cd, rtol=1e-12, atol=1e-12)
    assert np.isclose(Y[0, 2], expected_cl / expected_cd, rtol=1e-12, atol=1e-12)


def test_zero_camber_uses_standard_symmetric_naca_name():
    assert AirfoilTransonic5D._naca4_name(fc=0.0, tc=0.12) == "naca0012"
    assert AirfoilTransonic5D._naca4_name(fc=0.02, tc=0.08) == "naca2408"


def test_drag_divergence_slope_steepens_above_critical_mach():
    case = AirfoilTransonic5D()
    X_phys = np.array(
        [
            [0.68, 2.0, 0.12, 0.02, 5e6],
            [0.74, 2.0, 0.12, 0.02, 5e6],
            [0.80, 2.0, 0.12, 0.02, 5e6],
            [0.86, 2.0, 0.12, 0.02, 5e6],
        ],
        dtype=float,
    )
    cd = case.evaluate(case.normalize(X_phys))[:, 0]

    subcritical_rise = cd[1] - cd[0]
    transonic_rise = cd[3] - cd[2]
    assert transonic_rise > 5.0 * subcritical_rise


def test_subcritical_prandtl_glauert_scaling_preserves_lift_drag_ratio():
    case = AirfoilTransonic5D()
    X_phys = np.array(
        [
            [0.30, 2.0, 0.08, 0.02, 5e6],
            [0.60, 2.0, 0.08, 0.02, 5e6],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    beta = np.sqrt(1.0 - X_phys[:, 0] ** 2)
    expected_ratio = beta[0] / beta[1]

    assert np.isclose(Y[1, 1] / Y[0, 1], expected_ratio, rtol=1e-12, atol=1e-12)
    assert np.isclose(Y[1, 0] / Y[0, 0], expected_ratio, rtol=1e-12, atol=1e-12)
    assert np.isclose(Y[1, 2], Y[0, 2], rtol=1e-12, atol=1e-12)


def test_thinner_airfoil_delays_drag_divergence_at_fixed_mach():
    case = AirfoilTransonic5D()
    X_phys = np.array(
        [
            [0.80, 2.0, 0.08, 0.02, 5e6],
            [0.80, 2.0, 0.18, 0.02, 5e6],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))

    assert Y[0, 0] < Y[1, 0]
    assert Y[0, 2] > Y[1, 2]


def test_failure_threshold_marks_high_mach_drag_divergence_not_subcritical_flow():
    case = AirfoilTransonic5D()
    cd_output = case.meta.output_variables[0]
    assert cd_output.name == "CD"
    assert cd_output.failure_threshold == 0.05
    assert cd_output.failure_direction == "above"

    X_phys = np.array(
        [
            [0.30, 2.0, 0.12, 0.02, 5e6],
            [0.68, 2.0, 0.12, 0.02, 5e6],
            [0.88, 2.0, 0.18, 0.02, 5e6],
        ],
        dtype=float,
    )
    cd = case.evaluate(case.normalize(X_phys))[:, 0]

    assert cd[0] < cd_output.failure_threshold
    assert cd[1] < cd_output.failure_threshold
    assert cd[2] > cd_output.failure_threshold
