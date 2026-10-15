"""
tests/test_cases.py
====================
Full automated test suite at the case layer.

Coverage
--------
- All case interface contracts (input/output shape, dtype, normalisation, no NaN)
- check_implementation() for every case
- Batch-evaluate performance (within the threshold implied by cost_level)
- Metadata completeness
"""

import warnings

import numpy as np
import pytest

from doe_testbed import list_cases

# Dynamically collect all registered cases
from doe_testbed.registry.registry import get_case

warnings.filterwarnings("ignore")

ALL_CASE_IDS = [m.case_id for m in list_cases()]
ALLOWED_RESPONSE_TOPOLOGY = {
    "smooth_monotone",
    "smooth_unimodal",
    "threshold_jump",
    "discontinuous",
    "strong_interaction",
    "sparse_effects",
    "constrained_domain",
    "multi_modal",
    "smooth_multimodal",
    "heterogeneous",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(params=ALL_CASE_IDS)
def case(request):
    """Parameterised fixture: run once per registered case."""
    return get_case(request.param)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: check_implementation passes all items
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckImplementation:
    def test_passes_all_checks(self, case):
        """All boolean items in check_implementation must pass."""
        result = case.check_implementation(verbose=False)
        bool_items = {k: v for k, v in result.items() if isinstance(v, bool)}
        failed = [k for k, v in bool_items.items() if not v]
        total = len(bool_items)
        assert len(failed) == 0, (
            f"[{case.meta.case_id}] check_implementation failed items: {failed} "
            f"({total - len(failed)}/{total})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: evaluate output shape and dtype
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateInterface:

    def test_single_sample_2d(self, case):
        """Single sample (1, d) → (1, m)."""
        X = np.random.rand(1, case.dim)
        Y = case.evaluate(X)
        assert Y.shape == (
            1,
            case.meta.dim_output,
        ), f"[{case.meta.case_id}] expected (1, {case.meta.dim_output}), got {Y.shape}"

    def test_single_sample_1d(self, case):
        """Single sample (d,) → (1, m)."""
        X = np.random.rand(case.dim)
        Y = case.evaluate(X)
        assert Y.shape == (1, case.meta.dim_output)

    def test_batch_shape(self, case):
        """Batch samples (n, d) → (n, m)."""
        n = 20
        X = np.random.rand(n, case.dim)
        Y = case.evaluate(X)
        assert Y.shape == (n, case.meta.dim_output)

    def test_output_dtype(self, case):
        """Output dtype must be float64."""
        X = np.random.rand(5, case.dim)
        Y = case.evaluate(X)
        assert Y.dtype == np.float64, f"[{case.meta.case_id}] dtype={Y.dtype}, expected float64"

    def test_no_nan_in_valid_range(self, case):
        """Batch evaluation within normalised [0,1]^d must not produce NaN."""
        np.random.seed(42)
        X = np.random.rand(50, case.dim)
        Y = case.evaluate(X)
        n_nan = int(np.sum(np.isnan(Y)))
        assert n_nan == 0, f"[{case.meta.case_id}] {n_nan}/50 samples produced NaN"

    def test_reproducible(self, case):
        """Two calls with identical input produce identical results."""
        X = np.random.rand(10, case.dim)
        Y1 = case.evaluate(X)
        Y2 = case.evaluate(X)
        assert np.array_equal(
            Y1, Y2
        ), f"[{case.meta.case_id}] two evaluations with identical input differ"

    def test_boundary_center_and_random_points_are_finite(self, case):
        rng = np.random.default_rng(20260523)
        X = np.vstack([
            np.zeros((1, case.dim)),
            np.ones((1, case.dim)),
            np.full((1, case.dim), 0.5),
            rng.random((16, case.dim)),
        ])
        Y = case.evaluate(X)
        assert Y.shape == (len(X), case.meta.dim_output)
        assert np.all(np.isfinite(Y)), (
            f"[{case.meta.case_id}] boundary/center/random audit produced "
            "non-finite output"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: normalisation and denormalisation
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalization:

    def test_round_trip(self, case):
        """normalize(denormalize(X_norm)) ≈ X_norm, error < 1e-10."""
        X_norm = np.random.rand(20, case.dim)
        X_phys = case.denormalize(X_norm)
        X_back = case.normalize(X_phys)
        err = np.max(np.abs(X_back - X_norm))
        assert (
            err < 1e-10
        ), f"[{case.meta.case_id}] normalisation round-trip error {err:.2e} > 1e-10"

    def test_denormalize_bounds(self, case):
        """denormalize(0) = lower, denormalize(1) = upper."""
        lb = np.array(case.meta.bounds_lower)
        ub = np.array(case.meta.bounds_upper)
        X0 = np.zeros((1, case.dim))
        X1 = np.ones((1, case.dim))
        assert np.allclose(case.denormalize(X0)[0], lb, rtol=1e-8)
        assert np.allclose(case.denormalize(X1)[0], ub, rtol=1e-8)

    def test_normalize_bounds(self, case):
        """normalize(lower) = 0, normalize(upper) = 1."""
        lb = np.array(case.meta.bounds_lower).reshape(1, -1)
        ub = np.array(case.meta.bounds_upper).reshape(1, -1)
        assert np.allclose(case.normalize(lb), 0.0, atol=1e-8)
        assert np.allclose(case.normalize(ub), 1.0, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: metadata completeness
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetadata:

    def test_required_fields(self, case):
        """Required metadata fields must not be empty."""
        m = case.meta
        assert m.case_id, "case_id must not be empty"
        assert m.name, "name must not be empty"
        assert m.domain, "domain must not be empty"
        assert m.dim_input > 0, "dim_input must be > 0"
        assert m.dim_output > 0, "dim_output must be > 0"

    def test_bounds_consistency(self, case):
        """bounds_lower < bounds_upper; dimensions match dim_input."""
        lb = np.array(case.meta.bounds_lower)
        ub = np.array(case.meta.bounds_upper)
        assert len(lb) == case.meta.dim_input
        assert len(ub) == case.meta.dim_input
        assert np.all(lb < ub), f"[{case.meta.case_id}] found variable with lower >= upper"

    def test_input_variables_count(self, case):
        """Number of input_variables equals dim_input."""
        if case.meta.input_variables:
            assert len(case.meta.input_variables) == case.meta.dim_input

    def test_output_variables_count(self, case):
        """Number of output_variables equals dim_output."""
        if case.meta.output_variables:
            assert len(case.meta.output_variables) == case.meta.dim_output

    def test_applicable_scenarios_nonempty(self, case):
        """At least one applicable scenario is specified."""
        assert len(case.meta.applicable_scenarios) >= 1

    def test_governing_equation_nonempty(self, case):
        """governing_equation must not be empty (a physical equation description is required)."""
        assert (
            case.meta.governing_equation
        ), f"[{case.meta.case_id}] governing_equation field is empty"

    def test_input_output_names_unique(self, case):
        in_names = [v.name for v in case.meta.input_variables]
        out_names = [v.name for v in case.meta.output_variables]
        assert len(in_names) == len(set(in_names)), (
            f"[{case.meta.case_id}] duplicate input variable names: {in_names}"
        )
        assert len(out_names) == len(set(out_names)), (
            f"[{case.meta.case_id}] duplicate output variable names: {out_names}"
        )

    def test_log_scale_inputs_have_positive_lower_bound(self, case):
        bad = [
            (v.name, v.lower)
            for v in case.meta.input_variables
            if getattr(v, "log_scale", False) and v.lower <= 0
        ]
        assert not bad, f"[{case.meta.case_id}] invalid log-scale bounds: {bad}"

    def test_response_topology_tags_are_known(self, case):
        unknown = sorted(set(case.meta.response_topology) - ALLOWED_RESPONSE_TOPOLOGY)
        assert not unknown, f"[{case.meta.case_id}] unknown response_topology tags: {unknown}"

    def test_failure_boundary_has_thresholded_output(self, case):
        if not case.meta.has_failure_boundary:
            return
        thresholded = [
            ov.name
            for ov in case.meta.output_variables
            if ov.failure_threshold is not None
        ]
        assert thresholded, (
            f"[{case.meta.case_id}] has_failure_boundary=True but no "
            "OutputVariable.failure_threshold is defined"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: performance constraint (per-sample time not exceeding 20× typical_eval_time_ms)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformance:
    import time as _time

    def test_timing_within_limit(self, case):
        """Total time for 100 samples must not exceed 20× the declared typical time.

        (Generous margin — CI machine performance varies considerably.)
        """
        import time

        n = 100
        X = np.random.rand(n, case.dim)
        t0 = time.perf_counter()
        case.evaluate(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ms_per = elapsed_ms / n

        typical = case.meta.typical_eval_time_ms.get("default", 1000.0)
        limit = max(typical * 20.0, 5.0)
        assert ms_per < limit, (
            f"[{case.meta.case_id}] actual {ms_per:.1f} ms/sample > "
            f"max(declared {typical} ms * 20, 5.0 ms) = {limit} ms"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: sample() helper method
# ═══════════════════════════════════════════════════════════════════════════════


class TestSampleMethod:

    def test_sample_shape(self, case):
        """sample(n) returns normalised [0,1]^d coordinates of shape (n, dim)."""
        n = 30
        X = case.sample(n, seed=0)
        assert X.shape == (n, case.dim)

    def test_sample_within_bounds(self, case):
        """Points returned by sample() lie within normalised [0,1]^d."""
        X = case.sample(50, seed=1)
        assert np.all(X >= -1e-8)
        assert np.all(X <= 1 + 1e-8)

    def test_sample_reproducible(self, case):
        """sample() with identical seed returns identical results."""
        X1 = case.sample(20, seed=42)
        X2 = case.sample(20, seed=42)
        assert np.array_equal(X1, X2)
