import numpy as np
from pathlib import Path
import pytest

from doe_testbed.cases.structures.tuned_mass_damper_3d import TunedMassDamper3D


def test_frequency_grid_is_refined_around_tmd_resonance():
    r = TunedMassDamper3D._frequency_grid()

    assert r[0] == 0.01
    assert r[-1] == 4.0
    core = r[(r >= 0.5) & (r <= 1.5)]
    left = r[r < 0.5]
    right = r[r > 1.5]

    assert len(core) > len(left)
    assert len(core) > len(right)
    assert np.max(np.diff(core)) < np.max(np.diff(right))


def test_den_hartog_optimum_remains_resolved_with_refined_sweep():
    mu = 0.05
    f_opt = 1.0 / (1.0 + mu)
    zeta_opt = np.sqrt(3.0 * mu / (8.0 * (1.0 + mu)))
    expected = np.sqrt(1.0 + 2.0 / mu)

    h_max, bw, f_ratio = TunedMassDamper3D()._tmd_response(mu, f_opt, zeta_opt)

    assert abs(h_max - expected) / expected < 0.02
    assert bw > 0.0
    assert np.isfinite(f_ratio)


def test_bandwidth_returns_zero_when_half_power_region_is_not_closed(monkeypatch):
    model = TunedMassDamper3D()
    monkeypatch.setattr(model, "_frequency_grid", lambda: np.linspace(0.01, 0.2, 80))

    h_max, bw, f_ratio = model._tmd_response(mu=0.05, f=1.0, zeta_d=0.2)

    assert np.isfinite([h_max, bw, f_ratio]).all()
    assert bw == 0.0


def test_under_damped_response_remains_finite_with_shared_denominator():
    h_max, bw, f_ratio = TunedMassDamper3D()._tmd_response(mu=0.005, f=1.0, zeta_d=0.01)

    assert np.isfinite([h_max, bw, f_ratio]).all()
    assert h_max <= 100.0
    assert f_ratio <= 50.0


def test_metadata_declares_nonlinear_peak_response_scope():
    case = TunedMassDamper3D()

    assert case.meta.is_linear is False
    assert case.meta.response_topology == [
        "smooth_unimodal",
        "strong_interaction",
        "heterogeneous",
    ]
    assert "rational transfer-function peak" in Path(
        "doe_testbed/cases/structures/tuned_mass_damper_3d.py"
    ).read_text(encoding="utf-8")


def test_validation_material_declares_den_hartog_control_scope():
    validation_text = Path("tests/_cross_validation/tuned_mass_damper_3d.py").read_text(
        encoding="utf-8"
    )
    validation_note = Path("tests/_cross_validation/tuned_mass_damper_3d.md").read_text(
        encoding="utf-8"
    )

    assert "Validation scope" in validation_text
    assert "resonance-refined" in validation_text
    assert "calibrated accuracy" in validation_text
    assert "Target accuracy" not in validation_text
    assert "500-point" not in validation_note


def test_tmd_response_failure_is_not_silent_nan(monkeypatch):
    case = TunedMassDamper3D()

    def fail_response(*_args, **_kwargs):
        raise ValueError("simulated TMD frequency-response failure")

    monkeypatch.setattr(case, "_tmd_response", fail_response)
    X_phys = np.array([[0.05, 1.0, 0.10]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(X_phys)


def test_private_physical_input_validator_rejects_before_frequency_response(monkeypatch):
    case = TunedMassDamper3D()
    assert hasattr(case, "_validate_physical_inputs")
    assert "_validate_physical_inputs" in case.meta.solver_details

    with pytest.raises(ValueError, match="2-D array"):
        case._validate_physical_inputs(np.array([0.05, 1.0, 0.1]))

    def should_not_call_response(*_args, **_kwargs):
        raise AssertionError("_tmd_response should not be reached for invalid physical inputs")

    monkeypatch.setattr(case, "_tmd_response", should_not_call_response)
    base = np.array([[0.05, 1.0, 0.10]], dtype=float)
    invalid_values = [
        (0, 0.0049),
        (1, 0.699),
        (2, 0.009),
        (0, 0.1001),
        (1, 1.301),
        (2, 0.301),
        (1, np.nan),
    ]

    for col, value in invalid_values:
        x_bad = base.copy()
        x_bad[0, col] = value
        with pytest.raises(ValueError, match="tuned_mass_damper_3d physical inputs"):
            case._evaluate_core(x_bad)


def test_malformed_tmd_response_is_not_archived(monkeypatch):
    case = TunedMassDamper3D()

    def bad_response(*_args, **_kwargs):
        return np.array([np.nan, -1.0, -1.0])

    monkeypatch.setattr(case, "_tmd_response", bad_response)
    x_phys = np.array([[0.05, 1.0, 0.10]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent NaN"):
        case._evaluate_core(x_phys)
