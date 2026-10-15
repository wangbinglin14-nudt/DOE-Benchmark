"""Physics regression tests for ``soil_consolidation_5d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.geomechanics.soil_consolidation_5d import SoilConsolidation5D


def _terzaghi_series(
    *,
    k: float,
    m_v: float,
    H_dr: float,
    delta_s: float,
    t_obs: float,
    z: float | None = None,
    n_terms: int = 500,
) -> tuple[float, float | None]:
    """Return Terzaghi single-drainage U(t) and optional u(z,t) in kPa."""
    gamma_w = 9810.0
    c_v = k / (m_v * gamma_w)
    t_seconds = t_obs * 365.25 * 24.0 * 3600.0
    T_v = c_v * t_seconds / H_dr**2
    m = np.arange(n_terms)
    M = np.pi * (2 * m + 1) / 2.0
    U_series = 1.0 - np.sum(2.0 / M**2 * np.exp(-(M**2) * T_v))
    if z is None:
        return float(U_series), None
    u_series = np.sum(
        (2.0 * delta_s / M)
        * np.sin(M * z / H_dr)
        * np.exp(-(M**2) * T_v)
    )
    return float(U_series), float(u_series)


def _solve(row: list[float]) -> np.ndarray:
    return SoilConsolidation5D._solve_consolidation(
        k=row[0],
        m_v=row[1],
        H_dr=row[2],
        delta_s=row[3],
        t_obs=row[4],
        n_z=60,
        n_t=1200,
    )


def _domain_corners(case: SoilConsolidation5D) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    return np.array(list(itertools.product(*bounds)), dtype=float)


def test_consolidation_progress_increases_with_observation_time():
    early = _solve([1e-9, 1e-6, 5.0, 100.0, 1.0])
    late = _solve([1e-9, 1e-6, 5.0, 100.0, 50.0])

    assert late[0] > early[0]
    assert late[1] > early[1]
    assert late[2] < early[2]


def test_higher_hydraulic_conductivity_accelerates_pore_pressure_dissipation():
    slow = _solve([1e-10, 1e-6, 5.0, 100.0, 1.0])
    fast = _solve([1e-7, 1e-6, 5.0, 100.0, 1.0])

    assert fast[1] > slow[1]
    assert fast[2] < slow[2]


def test_settlement_scales_linearly_with_surcharge_for_fixed_consolidation_degree():
    low_load = _solve([1e-9, 1e-6, 5.0, 20.0, 10.0])
    high_load = _solve([1e-9, 1e-6, 5.0, 200.0, 10.0])

    assert np.isclose(high_load[1], low_load[1], rtol=1e-6)
    assert np.isclose(high_load[0] / low_load[0], 10.0, rtol=1e-6)
    assert np.isclose(high_load[2] / low_load[2], 10.0, rtol=1e-6)


def test_time_factor_similarity_controls_consolidation_degree_and_normalized_pore_pressure():
    """Holding Tv fixed must hold U(Tv) and u_mid / delta_s fixed."""
    baseline = _solve([1e-9, 1e-6, 5.0, 100.0, 10.0])
    scaled = _solve([4e-9, 1e-6, 10.0, 100.0, 10.0])

    assert np.isclose(scaled[1], baseline[1], rtol=1e-12, atol=1e-12)
    assert np.isclose(scaled[2] / 100.0, baseline[2] / 100.0, rtol=1e-12, atol=1e-12)
    assert np.isclose(scaled[0] / baseline[0], 2.0, rtol=1e-12, atol=1e-12)


def test_degree_of_consolidation_matches_terzaghi_series():
    k = 1e-9
    m_v = 1e-6
    H_dr = 5.0
    delta_s = 100.0
    t_obs = 10.0
    _, U_numeric, _ = SoilConsolidation5D._solve_consolidation(
        k=k,
        m_v=m_v,
        H_dr=H_dr,
        delta_s=delta_s,
        t_obs=t_obs,
        n_z=120,
        n_t=3000,
    )

    U_series, _ = _terzaghi_series(
        k=k,
        m_v=m_v,
        H_dr=H_dr,
        delta_s=delta_s,
        t_obs=t_obs,
    )

    assert np.isclose(U_numeric, U_series, rtol=0.02, atol=0.005)


def test_mid_depth_pore_pressure_matches_terzaghi_series():
    k = 1e-9
    m_v = 1e-6
    H_dr = 5.0
    delta_s = 100.0
    t_obs = 10.0
    _, _, u_mid_numeric = SoilConsolidation5D._solve_consolidation(
        k=k,
        m_v=m_v,
        H_dr=H_dr,
        delta_s=delta_s,
        t_obs=t_obs,
        n_z=120,
        n_t=3000,
    )

    _, u_mid_series = _terzaghi_series(
        k=k,
        m_v=m_v,
        H_dr=H_dr,
        delta_s=delta_s,
        t_obs=t_obs,
        z=H_dr / 2.0,
    )

    assert u_mid_series is not None
    assert np.isclose(u_mid_numeric, u_mid_series, rtol=0.02, atol=0.005)


def test_settlement_matches_degree_of_consolidation_identity():
    row = [1e-9, 2e-6, 7.0, 120.0, 12.0]
    S_t, U_t, _ = _solve(row)

    _, m_v, H_dr, delta_s, _ = row
    S_final = m_v * delta_s * 1e3 * H_dr * 1e3

    assert np.isclose(S_t, U_t * S_final, rtol=1e-10, atol=1e-10)
    assert S_t <= S_final


def test_metadata_declares_single_drainage_analytic_reference_scope():
    case = SoilConsolidation5D()
    equation = case.meta.governing_equation
    references = "\n".join(case.meta.physics_references)
    details = case.meta.solver_details

    assert "u_e(0,t) = 0" in equation
    assert "du_e/dz(H,t) = 0" in equation
    assert "U(t) = 1 - sum" in equation
    assert "single-drainage" in details
    assert "Direct Terzaghi single-drainage Fourier-series evaluation" in details
    assert "n_z/n_t fidelity arguments are accepted" in details
    assert "do not change the analytical response" in details
    assert case.meta.reference_solution_available is True
    assert case.meta.is_linear is False
    assert "analytical" in case.meta.tags
    assert "fourier_series" in case.meta.tags
    assert "fdm" not in case.meta.tags
    assert "strong_interaction" in case.meta.response_topology
    assert set(case.meta.interaction_pairs) == {
        ("k", "m_v"),
        ("k", "H_dr"),
        ("k", "t_obs"),
        ("m_v", "H_dr"),
        ("m_v", "t_obs"),
        ("m_v", "delta_s"),
        ("H_dr", "t_obs"),
        ("H_dr", "delta_s"),
        ("delta_s", "k"),
        ("delta_s", "t_obs"),
    }
    assert "Terzaghi" in references
    assert "https://docs.itascacg.com/itasca940/flac3d/zone/test3d/Fluid/1DConsolidation/1dconsolidation.html" in references
    assert "Biot-theory" in references
    assert "Darcy-flow" in references
    assert "drained-boundary" in references
    assert "impervious-base" in references
    assert "numerical-vs-analytical pore-pressure comparison boundary context" in references
    assert "https://mole-docs.readthedocs.io/en/main/examples/Parabolic/1D/Terzaghi1D.html" in references
    assert "https://opengti.io/consolidation" in references
    assert "https://www.opengeosys.org/docs/benchmarks/hydro-mechanics/" in references
    assert "https://www.opengeosys.org/stable/docs/benchmarks/hydro-mechanics/" in references
    assert "ConsolidationBenchmark" in references
    assert "not validation data for this benchmark surrogate" in references
    for phrase in (
        "single-vs-double-drainage design selection",
        "radial drains",
        "layered/anisotropic soils",
        "stress-dependent k/m_v",
        "finite-strain",
        "secondary compression",
        "construction staging",
        "2-D/3-D coupled Biot poromechanics",
        "staggered/monolithic pressure-displacement HM solves",
        "fixed-stress stabilization",
        "Backward-Euler time stepping",
        "displacement/stress/strain/velocity field outputs",
        "stress/strain field recovery",
        "unsaturated flow",
        "field-instrument interpretation",
        "laboratory/field oedometer parameter fitting",
    ):
        assert phrase in details
    assert hasattr(case, "_validate_physical_inputs")
    assert "named _validate_physical_inputs private physical-input evaluation rejects" in details
    assert "before Terzaghi Fourier-series evaluation" in details
    assert "out-of-contract S_t/U_t/u_mid responses are rejected" in details


def test_fidelity_arguments_do_not_change_analytical_response():
    case = SoilConsolidation5D()
    X_phys = np.array([[1e-9, 1e-6, 5.0, 100.0, 10.0]], dtype=float)

    default = case.evaluate(case.normalize(X_phys), fidelity="default")
    high = case.evaluate(case.normalize(X_phys), fidelity="high")
    direct_small = SoilConsolidation5D._solve_consolidation(
        k=1e-9,
        m_v=1e-6,
        H_dr=5.0,
        delta_s=100.0,
        t_obs=10.0,
        n_z=10,
        n_t=10,
    )

    assert np.allclose(default, high, rtol=1e-14, atol=1e-14)
    assert np.allclose(default[0], direct_small, rtol=1e-14, atol=1e-14)
    assert "analytical" in case.meta.solver_library.lower()


def test_large_time_factor_reaches_full_consolidation_without_cn_lag():
    S_t, U_t, u_mid = SoilConsolidation5D._solve_consolidation(
        k=1e-7,
        m_v=1e-7,
        H_dr=2.0,
        delta_s=20.0,
        t_obs=50.0,
        n_z=100,
        n_t=3000,
    )

    assert np.isclose(S_t, 4.0, rtol=1e-12, atol=1e-12)
    assert U_t == 1.0
    assert u_mid == 0.0


def test_validation_material_declares_one_dimensional_consolidation_scope():
    validation_text = Path("tests/_cross_validation/soil_consolidation_5d.py").read_text(
        encoding="utf-8"
    )
    normalized_validation_text = " ".join(validation_text.split())
    source_text = Path("doe_testbed/cases/geomechanics/soil_consolidation_5d.py").read_text(
        encoding="utf-8"
    )
    normalized_source_text = " ".join(source_text.split())

    assert "Terzaghi 1-D consolidation" in validation_text
    assert "Fourier-series solution" in validation_text
    assert "analytical control" in validation_text
    assert "Itasca FLAC3D" in validation_text
    assert "MOLE Terzaghi" in validation_text
    assert "OpenGTI" in validation_text
    assert "OpenGeoSys hydro-mechanics" in validation_text
    assert "Consolidation benchmark with the staggered scheme" in normalized_validation_text
    assert "single-vs-double-drainage design choice" in normalized_validation_text
    assert "pressure-displacement HM/Biot" in validation_text
    assert "HYDRO_MECHANICS workflow" in validation_text
    assert "fixed-stress stabilization" in validation_text
    assert "Backward-Euler time stepping" in validation_text
    assert "displacement/stress/strain/velocity outputs" in validation_text
    assert "not used as validation data" in normalized_validation_text
    assert "do not establish a calibrated site-specific" in normalized_validation_text
    assert "2-D/3-D coupled geotechnical analysis" in validation_text
    assert "Biot theory" in source_text
    assert "Darcy flow" in source_text
    assert "impervious base" in source_text
    assert "not calibration data" in source_text
    assert "rejects malformed arrays" in normalized_validation_text
    assert "before Terzaghi Fourier-series evaluation" in normalized_validation_text
    assert "cannot extrapolate beyond the declared one-dimensional consolidation bounds" in normalized_validation_text
    assert "time-factor similarity" in normalized_validation_text
    assert "U(T_v) and u_mid / Delta_sigma" in normalized_validation_text
    assert "Target accuracy" not in validation_text
    assert "single-drainage" in source_text
    assert "not a calibrated site-specific settlement model" in source_text
    assert "secondary compression/creep" in source_text
    assert "2-D/3-D coupled Biot poromechanics" in source_text
    assert "poroelastic pressure-displacement coupling" in source_text
    assert "stress/strain field recovery" in source_text
    assert "oedometer parameter fitting" in source_text
    assert 'the published "Consolidation benchmark with the staggered scheme"' in normalized_source_text
    assert "external validation" not in validation_text


def test_readme_labels_soil_consolidation_as_analytical_not_crank_nicolson():
    readme_en = Path("README.md").read_text(encoding="utf-8")
    readme_zh = Path("README_zh.md").read_text(encoding="utf-8")

    en_row = next(
        line for line in readme_en.splitlines() if line.startswith("| `soil_consolidation_5d` |")
    )
    zh_row = next(
        line for line in readme_zh.splitlines() if line.startswith("| `soil_consolidation_5d` |")
    )

    assert "Fourier analytical" in en_row
    assert "Fourier 解析" in zh_row
    assert "Crank" not in en_row
    assert "Crank" not in zh_row
    assert "FDM" not in en_row
    assert "FDM" not in zh_row


def test_web_validation_audit_preserves_soil_p2_analytic_control_boundary():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit_text.split())
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `soil_consolidation_5d` |")
        and "Terzaghi 1-D consolidation" in line
    )

    assert "Evidence tier remains P2 analytic-control" in row
    assert "Itasca/FLAC3D" in row
    assert "MOLE" in row
    assert "OpenGTI" in row
    assert "not validation data" in row
    assert "Keep P2 analytical-control" in row
    assert "not externally validated site-specific" in row
    assert (
        "Additional `soil_consolidation_5d` 2026-06-08 official-source re-check:"
        in audit_text
    )
    assert "Evidence tier remains P2 analytic-control" in normalized_audit
    assert "framework of Biot theory" in normalized_audit
    assert "Dirichlet drained boundary" in normalized_audit
    assert "Neumann no-flow boundary" in normalized_audit
    assert '"Consolidation benchmark with the staggered scheme"' in normalized_audit
    assert "Additional `soil_consolidation_5d` 2026-06-08 official-source boundary tightening" in audit_text
    assert "single-vs-double-drainage judgement" in normalized_audit
    assert "field/laboratory parameter estimation" in normalized_audit
    assert "staggered pressure/displacement HM simulation" in normalized_audit
    assert "not validation data for this single-drainage analytical response surface" in normalized_audit
    assert "OpenGeoSys HM project-boundary confirmation" in normalized_audit
    assert "`HYDRO_MECHANICS` workflow" in normalized_audit
    assert "fixed-stress stabilization" in normalized_audit
    assert "Backward-Euler time stepping" in normalized_audit
    assert "displacement, stress/strain, and velocity outputs" in normalized_audit
    assert "does not solve OpenGeoSys-style pressure-displacement systems" in normalized_audit
    assert "No numerical response resweep is needed solely for this HM project-boundary confirmation" in normalized_audit
    assert "Do not describe this case as externally validated site-specific settlement prediction" in normalized_audit
    assert "OpenGeoSys/FLAC3D-equivalent poromechanics validation" in normalized_audit
    assert "field/lab calibrated settlement prediction" in normalized_audit
    assert "legal-domain finite analytical responses are unchanged" in normalized_audit
    assert "Additional `soil_consolidation_5d` 2026-06-09 Terzaghi time-factor similarity-control note" in audit_text
    assert "holding `T_v = c_v t / H_dr^2` fixed" in normalized_audit
    assert "No numerical response resweep is needed solely for this similarity-control note" in normalized_audit
    assert "https://www.opengeosys.org/stable/docs/benchmarks/hydro-mechanics/" in audit_text
    assert "Itasca Biot/Darcy drained-boundary and impervious-base boundary context" in normalized_audit
    assert "Additional `soil_consolidation_5d` 2026-06-09 response-contract note" in audit_text
    assert "`0 <= S_t <= S_final`" in normalized_audit
    assert "`0 <= U_t <= 1`" in normalized_audit
    assert "`0 <= u_mid <= Delta_sigma`" in normalized_audit
    assert "normal single-drainage analytical Terzaghi responses in the declared input domain are unchanged" in normalized_audit


def test_soil_consolidation_public_itasca_anchor_is_consistent_and_not_stale():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    validation_text = Path("tests/_cross_validation/soil_consolidation_5d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path("doe_testbed/cases/geomechanics/soil_consolidation_5d.py").read_text(
        encoding="utf-8"
    )

    current_anchor = (
        "https://docs.itascacg.com/itasca940/flac3d/zone/test3d/Fluid/"
        "1DConsolidation/1dconsolidation.html"
    )
    stale_anchor = (
        "https://docs.itascacg.com/pfc600/flac3d/zone/test3d/Fluid/"
        "1DConsolidation/1dconsolidation.html"
    )

    assert current_anchor in audit_text
    assert current_anchor in validation_text
    assert current_anchor in source_text
    assert stale_anchor not in audit_text
    assert stale_anchor not in validation_text
    assert stale_anchor not in source_text


def test_declared_output_ranges_cover_domain_corners():
    case = SoilConsolidation5D()
    Y = case.evaluate(case.normalize(_domain_corners(case)))

    assert np.all(np.isfinite(Y))
    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_consolidation_solver_failure_is_not_silent_nan(monkeypatch):
    case = SoilConsolidation5D()

    def fail_solve(*_args, **_kwargs):
        raise ValueError("simulated consolidation linear-solve failure")

    monkeypatch.setattr(case, "_solve_consolidation", fail_solve)
    X_phys = np.array([[1e-9, 1e-6, 5.0, 100.0, 10.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_nonfinite_consolidation_response_is_not_silent_nan(monkeypatch):
    case = SoilConsolidation5D()

    def invalid_solve(*_args, **_kwargs):
        return np.array([np.nan, 0.5, 10.0])

    monkeypatch.setattr(case, "_solve_consolidation", invalid_solve)
    X_phys = np.array([[1e-9, 1e-6, 5.0, 100.0, 10.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_nonphysical_or_out_of_contract_consolidation_response_is_not_archived(monkeypatch):
    case = SoilConsolidation5D()
    X_phys = np.array([[1e-9, 1e-6, 5.0, 100.0, 10.0]], dtype=float)

    invalid_responses = [
        np.array([-1.0, 0.5, 10.0]),
        np.array([600.0, 0.5, 10.0]),
        np.array([100.0, -0.01, 10.0]),
        np.array([100.0, 1.01, 10.0]),
        np.array([100.0, 0.5, -0.1]),
        np.array([100.0, 0.5, 100.1]),
        np.array([20001.0, 0.5, 10.0]),
    ]

    for response in invalid_responses:
        monkeypatch.setattr(
            case,
            "_solve_consolidation",
            lambda *_args, _response=response, **_kwargs: _response,
        )
        with pytest.raises(RuntimeError, match="refusing silent NaN"):
            case._evaluate_core(X_phys)


def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_series(monkeypatch):
    case = SoilConsolidation5D()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Terzaghi solver should not be called for invalid input")

    monkeypatch.setattr(case, "_solve_consolidation", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._validate_physical_inputs(
            np.array([1e-9, 1e-6, 5.0, 100.0, 10.0], dtype=float)
        )

    invalid_rows = [
        [0.99e-10, 1e-6, 5.0, 100.0, 10.0],
        [1.01e-7, 1e-6, 5.0, 100.0, 10.0],
        [1e-9, 0.99e-7, 5.0, 100.0, 10.0],
        [1e-9, 5.01e-6, 5.0, 100.0, 10.0],
        [1e-9, 1e-6, 1.9, 100.0, 10.0],
        [1e-9, 1e-6, 20.1, 100.0, 10.0],
        [1e-9, 1e-6, 5.0, 19.9, 10.0],
        [1e-9, 1e-6, 5.0, 200.1, 10.0],
        [1e-9, 1e-6, 5.0, 100.0, 0.49],
        [1e-9, 1e-6, 5.0, 100.0, 50.1],
        [1e-9, np.nan, 5.0, 100.0, 10.0],
    ]
    for row in invalid_rows:
        with pytest.raises(
            ValueError,
            match="k/m_v/H_dr/delta_s/t_obs benchmark domain",
        ):
            case._validate_physical_inputs(np.array([row], dtype=float))
