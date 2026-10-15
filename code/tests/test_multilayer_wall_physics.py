"""Targeted physics checks for the multilayer wall conduction cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import doe_testbed as db


def _domain_corners(case):
    lb = np.array([v.lower for v in case.meta.input_variables], dtype=float)
    ub = np.array([v.upper for v in case.meta.input_variables], dtype=float)
    return np.array(np.meshgrid(*zip(lb, ub), indexing="ij")).reshape(
        case.meta.dim_input, -1
    ).T


@pytest.mark.parametrize(
    ("case_id", "k_values"),
    [
        ("multilayer_wall_4d", [0.1, 0.5, 2.0, 10.0]),
        ("multilayer_wall_8d", [0.08, 0.12, 0.3, 0.9, 1.5, 4.0, 12.0, 30.0]),
        (
            "multilayer_wall_16d",
            [0.06, 0.08, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2,
             2.0, 3.5, 5.0, 8.0, 12.0, 18.0, 25.0, 40.0],
        ),
    ],
)
def test_series_resistance_flux_and_mid_temperature_are_exact(case_id, k_values):
    """Composite plane-wall equations must hold exactly for all scalable N."""
    case = db.get_case(case_id)
    k = np.asarray(k_values, dtype=float)
    y = case.evaluate(case.normalize(k.reshape(1, -1)))[0]
    r_total, q_flux, t_mid = y

    delta = case._DELTA_PER_LAYER
    t_hot = case._T_HOT
    t_cold = case._T_COLD
    r_layers = delta / k
    expected_r = float(r_layers.sum())
    expected_q = float((t_hot - t_cold) / expected_r)
    expected_t_mid = float(t_hot - expected_q * r_layers[: len(k) // 2].sum())

    np.testing.assert_allclose(r_total, expected_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q_flux, expected_q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t_mid, expected_t_mid, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q_flux * r_total, t_hot - t_cold, rtol=1e-12)


def test_uniform_conductivity_resistance_scales_with_layer_count():
    """Fixed per-layer thickness means uniform-k resistance scales as N."""
    k = 2.5
    observed = {}
    for case_id in ("multilayer_wall_4d", "multilayer_wall_8d", "multilayer_wall_16d"):
        case = db.get_case(case_id)
        x = np.full((1, case.meta.dim_input), k)
        observed[case.meta.dim_input] = case.evaluate(case.normalize(x))[0, 0]

    np.testing.assert_allclose(observed[8], 2.0 * observed[4], rtol=1e-12)
    np.testing.assert_allclose(observed[16], 4.0 * observed[4], rtol=1e-12)


def test_layer_order_changes_mid_temperature_not_total_resistance_or_flux():
    """Series resistance is order-invariant, but interface temperature is not."""
    case = db.get_case("multilayer_wall_4d")
    x_a = np.array([[0.1, 10.0, 10.0, 10.0]], dtype=float)
    x_b = np.array([[10.0, 10.0, 10.0, 0.1]], dtype=float)

    y_a = case.evaluate(case.normalize(x_a))[0]
    y_b = case.evaluate(case.normalize(x_b))[0]

    np.testing.assert_allclose(y_a[0], y_b[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_a[1], y_b[1], rtol=1e-12, atol=1e-12)
    assert y_a[2] < y_b[2]


@pytest.mark.parametrize(
    "case_id",
    ["multilayer_wall_4d", "multilayer_wall_8d", "multilayer_wall_16d"],
)
def test_metadata_marks_input_output_response_as_nonlinear(case_id):
    case = db.get_case(case_id)

    assert case.meta.is_linear is False
    assert "smooth_monotone" in case.meta.response_topology


@pytest.mark.parametrize(
    ("case_id", "validation_path"),
    [
        ("multilayer_wall_4d", "tests/_cross_validation/multilayer_wall_4d.py"),
        ("multilayer_wall_8d", "tests/_cross_validation/multilayer_wall_8d.py"),
        ("multilayer_wall_16d", "tests/_cross_validation/multilayer_wall_16d.py"),
    ],
)
def test_validation_material_declares_exact_series_resistance_scope(
    case_id, validation_path
):
    case = db.get_case(case_id)
    validation_text = Path(validation_path).read_text(encoding="utf-8")
    normalized_validation_text = " ".join(validation_text.split())
    source_text = Path("doe_testbed/cases/heat_transfer/multilayer_wall_nd.py").read_text(
        encoding="utf-8"
    )

    assert "Series thermal resistance" in validation_text
    assert "Validation scope" in validation_text
    assert "exact series-resistance controls" in validation_text
    assert "fixed per-layer thickness" in normalized_validation_text
    assert "calibrated material-stack" in normalized_validation_text
    assert "building-envelope design model" in normalized_validation_text
    assert "NASA passive-thermal-control material is boundary context" in normalized_validation_text
    assert "hardware-integration, and test/analysis workflows" in normalized_validation_text
    assert "series thermal-resistance model" in source_text
    assert "no numerical error" in case.meta.solver_details
    assert "external validation" not in validation_text


def test_metadata_declares_public_sources_and_non_building_model_scope():
    source_text = Path("doe_testbed/cases/heat_transfer/multilayer_wall_nd.py").read_text(
        encoding="utf-8"
    )
    for case_id in ("multilayer_wall_4d", "multilayer_wall_8d", "multilayer_wall_16d"):
        case = db.get_case(case_id)
        refs = "\n".join(case.meta.physics_references)
        details = case.meta.solver_details

        assert "MIT Unified Engineering" in refs
        assert "https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node118.html" in refs
        assert "NPTEL Heat Transfer" in refs
        assert "lecture-4.pdf" in refs
        assert "NASA Passive Thermal Control Engineering Guidebook" in refs
        assert "https://ntrs.nasa.gov/citations/20230013900" in refs
        assert "not a calibrated material-stack" in details
        assert "thermal-bridge" in details
        assert "moisture" in details
        assert "contact-resistance" in details
        assert "radiation/convection" in details
        assert "spacecraft passive-thermal hardware/" in details
        assert "code-compliance design model" in details
        assert hasattr(case, "_validate_physical_inputs")
        assert "named _validate_physical_inputs private physical-input evaluation rejects" in details
        assert "before series-resistance assembly" in details
        assert "balanced-main-effect metadata convention" in case.meta.sobol_computation
        assert "not a fresh Saltelli estimate" in case.meta.sobol_computation
    assert "Thermal Resistance Circuits" in source_text
    assert "conduction through composite walls" in source_text


def test_web_audit_keeps_multilayer_wall_analytic_tier_and_scope_honest():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit_text.split())
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `multilayer_wall_4d`, `multilayer_wall_8d`, `multilayer_wall_16d` |")
    )

    assert "Evidence tier remains P2 analytic-control" in row
    assert "MIT thermal-resistance-circuit notes" in row
    assert "NPTEL composite-wall lecture material" in row
    assert "formula validation" in row
    assert "not external validation" in row
    assert "building-envelope energy model" in row
    assert "thermal bridge" in row
    assert "contact-resistance model" in row
    assert "moisture-coupled wall" in row
    assert "not a fresh Saltelli validation" in row
    assert "Normal finite responses are unchanged" in row
    assert "No numerical response resweep is needed" in row
    for case_id in ("multilayer_wall_4d", "multilayer_wall_8d", "multilayer_wall_16d"):
        assert f"Additional `{case_id}` 2026-06-08 official-source re-check:" in audit_text
    assert "MIT thermal-resistance-circuit material" in normalized_audit
    assert "NPTEL composite-wall conduction material" in normalized_audit
    assert "NASA thermal-control material is boundary evidence" in normalized_audit
    assert (
        "Additional `multilayer_wall_4d`, `multilayer_wall_8d`, and "
        "`multilayer_wall_16d` 2026-06-08 official-source boundary tightening"
        in audit_text
    )
    assert "thermal-path definition, hardware integration, and thermal test/analysis practice" in normalized_audit
    assert "analysis (including documentation and review)" in normalized_audit
    assert "testing, and flight operations" in normalized_audit
    assert "Evidence tier remains P2 analytic-control" in normalized_audit
    assert "Do not describe this case as externally validated building physics" in normalized_audit
    assert "spacecraft passive-thermal hardware design" in normalized_audit
    assert "integrated thermal test/analysis" in normalized_audit
    assert "flight-operations thermal support" in normalized_audit
    assert "calibrated material-stack simulation" in normalized_audit
    assert "legal-domain finite analytical responses are unchanged" in normalized_audit


@pytest.mark.parametrize(
    "case_id",
    ["multilayer_wall_4d", "multilayer_wall_8d", "multilayer_wall_16d"],
)
def test_declared_output_ranges_cover_domain_corners(case_id):
    case = db.get_case(case_id)
    y = case.evaluate(case.normalize(_domain_corners(case)))

    for j, output in enumerate(case.meta.output_variables):
        lo, hi = output.expected_range
        assert float(np.nanmin(y[:, j])) >= lo
        assert float(np.nanmax(y[:, j])) <= hi


@pytest.mark.parametrize(
    "case_id",
    ["multilayer_wall_4d", "multilayer_wall_8d", "multilayer_wall_16d"],
)
def test_nonfinite_or_nonphysical_wall_response_is_not_silent_archive_row(case_id):
    case = db.get_case(case_id)
    x = np.ones((1, case.meta.dim_input), dtype=float)

    x_zero = x.copy()
    x_zero[0, 0] = 0.0
    with pytest.raises(ValueError, match="layer-conductivity benchmark domain"):
        case._evaluate_core(x_zero)

    x_nan = x.copy()
    x_nan[0, -1] = np.nan
    with pytest.raises(ValueError, match="layer-conductivity benchmark domain"):
        case._evaluate_core(x_nan)


@pytest.mark.parametrize(
    "case_id",
    ["multilayer_wall_4d", "multilayer_wall_8d", "multilayer_wall_16d"],
)
def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_series(
    case_id,
):
    case = db.get_case(case_id)
    x = np.ones((1, case.meta.dim_input), dtype=float)

    with pytest.raises(ValueError, match="shape"):
        case._validate_physical_inputs(np.ones(case.meta.dim_input, dtype=float))

    for bad_value in (0.049, 50.1, np.nan):
        x_bad = x.copy()
        x_bad[0, 0] = bad_value
        with pytest.raises(ValueError, match="layer-conductivity benchmark domain"):
            case._evaluate_core(x_bad)
