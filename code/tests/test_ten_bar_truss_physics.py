import numpy as np
import pytest

from doe_testbed.cases.structures.ten_bar_truss_10d import TenBarTruss10D

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _domain_corners(case):
    lb = np.array([v.lower for v in case.meta.input_variables], dtype=float)
    ub = np.array([v.upper for v in case.meta.input_variables], dtype=float)
    return np.array(np.meshgrid(*zip(lb, ub), indexing="ij")).reshape(10, -1).T


def _assemble_stiffness(case, A_arr):
    nodes = case._NODES
    n_dof = len(nodes) * 2
    K = np.zeros((n_dof, n_dof))

    for ei, (ni, nj) in enumerate(case._ELEMENTS):
        dx = nodes[nj, 0] - nodes[ni, 0]
        dy = nodes[nj, 1] - nodes[ni, 1]
        length = np.sqrt(dx**2 + dy**2)
        cx = dx / length
        cy = dy / length
        k_local = case._E * A_arr[ei] / length * np.outer(
            [cx, cy, -cx, -cy],
            [cx, cy, -cx, -cy],
        )
        dofs = [ni * 2, ni * 2 + 1, nj * 2, nj * 2 + 1]
        for a_i, a in enumerate(dofs):
            for b_i, b in enumerate(dofs):
                K[a, b] += k_local[a_i, b_i]

    return K


def _load_vector(case):
    n_dof = len(case._NODES) * 2
    F = np.zeros(n_dof)
    for node_idx, (fx, fy) in case._LOADS.items():
        F[node_idx * 2] += fx
        F[node_idx * 2 + 1] += fy
    return F


def test_fixed_support_dofs_are_removed_and_reconstructed_as_zero():
    case = TenBarTruss10D()
    A_arr = np.full(10, 10e-4)
    K = _assemble_stiffness(case, A_arr)
    F = _load_vector(case)
    fixed_dofs = case._fixed_dofs(K.shape[0])

    u = case._solve_constrained_system(K, F, fixed_dofs)

    np.testing.assert_array_equal(fixed_dofs, np.array([6, 7, 12, 13]))
    assert np.all(u[fixed_dofs] == 0.0)
    assert np.isfinite(u).all()


def test_constrained_solve_matches_penalty_response_without_ill_conditioning():
    case = TenBarTruss10D()
    A_arr = np.full(10, 10e-4)
    K = _assemble_stiffness(case, A_arr)
    F = _load_vector(case)
    fixed_dofs = case._fixed_dofs(K.shape[0])

    u_constrained = case._solve_constrained_system(K, F, fixed_dofs)

    K_penalty = K.copy()
    for dof in fixed_dofs:
        K_penalty[dof, dof] += 1e30
    u_penalty = np.linalg.solve(K_penalty, F)

    free_dofs = np.setdiff1d(np.arange(K.shape[0]), fixed_dofs)
    K_free = K[np.ix_(free_dofs, free_dofs)]

    np.testing.assert_allclose(u_constrained, u_penalty, rtol=1e-12, atol=1e-20)
    assert np.linalg.cond(K_free) < 1e4
    cond_penalty = np.linalg.cond(K_penalty)
    assert not np.isfinite(cond_penalty) or cond_penalty > 1e20


def test_declared_output_ranges_cover_domain_corners():
    case = TenBarTruss10D()
    y = case.evaluate(case.normalize(_domain_corners(case)))

    for j, output in enumerate(case.meta.output_variables):
        lo, hi = output.expected_range
        assert float(np.nanmin(y[:, j])) >= lo
        assert float(np.nanmax(y[:, j])) <= hi


def test_metadata_marks_input_output_response_as_nonlinear():
    case = TenBarTruss10D()

    assert case.meta.is_linear is False
    assert "smooth_monotone" in case.meta.response_topology


def test_screening_documentation_matches_current_sobol_metadata():
    case = TenBarTruss10D()
    source_text = (
        ROOT / "doe_testbed" / "cases" / "structures" / "ten_bar_truss_10d.py"
    ).read_text(encoding="utf-8")

    assert case.meta.important_variables == ["A2", "A3", "A10", "A8"]
    assert "Bottom-chord bars (A2, A3) and main diagonals (A8, A10)" in source_text
    assert "Bottom-chord bars (A1, A2)" not in source_text
    assert "main diagonals (A7, A9)" not in source_text


def test_cross_validation_scope_is_not_overclaimed():
    text = (
        ROOT / "tests" / "_cross_validation" / "ten_bar_truss_10d.py"
    ).read_text(encoding="utf-8")

    assert "Validation scope" in text
    assert "ideal pin-jointed" in text
    assert "linear-elastic axial-bar truss model" in text
    assert "Target accuracy" not in text
    assert "exact for truss elements" not in text
    assert "arbitrary truss structures" in text


def test_truss_solver_failure_is_not_silent_nan(monkeypatch):
    case = TenBarTruss10D()

    def fail_solve(*_args, **_kwargs):
        raise ValueError("simulated truss stiffness solve failure")

    monkeypatch.setattr(case, "_truss_fem", fail_solve)
    X_phys = np.full((1, case.meta.dim_input), 10.0, dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent"):
        case._evaluate_core(X_phys)


def test_private_physical_input_validator_rejects_before_fem(monkeypatch):
    case = TenBarTruss10D()
    assert hasattr(case, "_validate_physical_inputs")
    assert "_validate_physical_inputs" in case.meta.solver_details

    with pytest.raises(ValueError, match="2-D array"):
        case._validate_physical_inputs(np.full(case.meta.dim_input, 10.0))

    def should_not_call_fem(*_args, **_kwargs):
        raise AssertionError("_truss_fem should not be reached for invalid physical inputs")

    monkeypatch.setattr(case, "_truss_fem", should_not_call_fem)
    base = np.full((1, case.meta.dim_input), 10.0, dtype=float)
    invalid_values = [
        (0, 0.644),
        (1, 25.801),
        (9, np.nan),
    ]

    for col, value in invalid_values:
        x_bad = base.copy()
        x_bad[0, col] = value
        with pytest.raises(ValueError, match="ten_bar_truss_10d physical inputs"):
            case._evaluate_core(x_bad)


def test_malformed_truss_response_is_not_archived(monkeypatch):
    case = TenBarTruss10D()

    def bad_fem(*_args, **_kwargs):
        return np.array([np.nan, -1.0, 0.0])

    monkeypatch.setattr(case, "_truss_fem", bad_fem)
    x_phys = np.full((1, case.meta.dim_input), 10.0, dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent"):
        case._evaluate_core(x_phys)
