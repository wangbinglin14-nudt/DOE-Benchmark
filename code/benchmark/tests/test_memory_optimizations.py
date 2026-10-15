"""Tests for the memory-optimization suite (buffered append + memory-class
routing + gc hygiene).

These tests cover changes made to mitigate local Windows OOMs:
  1. :class:`BenchmarkResult` now buffers appends O(1) and materialises
     on flush (no more O(n^2) concat churn).
  2. :class:`BenchmarkMethod` has a ``memory_class`` attribute used by
     the runner to dispatch heavy BoTorch methods with fewer workers.
  3. :class:`BenchmarkRunner` accepts ``bo_n_jobs`` + ``mem_cap_gb``
     and routes each partition via ``_n_jobs_for_class``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.core import (
    BenchmarkMethod,
    BenchmarkResult,
    BenchmarkRunner,
    StaticDesignMethod,
)
from benchmark.samplers import LHSSampler
from benchmark.surrogates import LinearSurrogate


# ─── BenchmarkResult buffered append ────────────────────────────────────
class TestBenchmarkResultBuffered:
    def test_empty_result_has_empty_df(self):
        r = BenchmarkResult()
        assert isinstance(r.df, pd.DataFrame)
        assert len(r.df) == 0

    def test_append_is_buffered_until_flush(self):
        r = BenchmarkResult()
        r.append({"case_id": "a", "method": "m", "status": "ok"})
        r.append({"case_id": "b", "method": "m", "status": "ok"})
        # Before flush: rows sit in _pending, df is still empty.
        assert len(r._pending) == 2
        assert len(r.df) == 0
        r.flush()
        # After flush: rows are materialised, buffer cleared.
        assert len(r._pending) == 0
        assert len(r.df) == 2
        assert list(r.df["case_id"]) == ["a", "b"]

    def test_flush_is_idempotent(self):
        r = BenchmarkResult()
        r.append({"case_id": "a"})
        r.flush()
        r.flush()  # second call should be a no-op
        assert len(r.df) == 1

    def test_flush_preserves_schema_across_calls(self):
        """Columns added late must not silently drop earlier columns."""
        r = BenchmarkResult()
        r.append({"case_id": "a", "metric_v1": 1.0})
        r.flush()
        r.append({"case_id": "b", "metric_v2": 2.0})
        r.flush()
        assert set(r.df.columns) >= {"case_id", "metric_v1", "metric_v2"}
        # Row 0 has only metric_v1; row 1 has only metric_v2.
        assert np.isnan(r.df.loc[1, "metric_v1"])
        assert np.isnan(r.df.loc[0, "metric_v2"])

    def test_large_batch_is_fast(self):
        """10,000 appends must be O(n), not O(n^2).

        A quadratic implementation on this size would exceed several
        seconds on any contemporary machine; the buffered append
        completes in well under 1 s.
        """
        r = BenchmarkResult()
        t0 = time.perf_counter()
        for i in range(10_000):
            r.append({"case_id": f"c{i % 10}", "repeat": i, "status": "ok"})
        r.flush()
        elapsed = time.perf_counter() - t0
        assert len(r.df) == 10_000
        # Generous bound (< 1 s) — observed is ~20-50 ms.  A quadratic
        # implementation would take > 30 s here.
        assert elapsed < 1.0, f"buffered append too slow: {elapsed:.2f}s"

    def test_to_csv_auto_flushes(self, tmp_path: Path):
        r = BenchmarkResult()
        r.append({"case_id": "a", "status": "ok"})
        csv_path = tmp_path / "r.csv"
        r.to_csv(csv_path)   # must flush internally
        assert len(pd.read_csv(csv_path)) == 1

    def test_aggregate_flushes_buffer_first(self):
        """aggregate() must drain the _pending buffer before groupby()
        touches self.df.  We can observe this by checking _pending after
        the call — regardless of whether aggregate itself succeeds, the
        flush precondition must hold."""
        r = BenchmarkResult()
        r.append({"case_id": "a", "method": "m", "budget_mult": 5,
                  "repeat": 0, "nrmse": 0.1})
        assert len(r._pending) == 1
        try:
            _ = r.aggregate()  # may raise on pre-existing aggregate quirks
        except Exception:
            pass
        # Regardless of aggregate() outcome, flush() must have been called.
        assert len(r._pending) == 0
        assert len(r.df) == 1


# ─── Memory-class attribute on methods ──────────────────────────────────
class TestMemoryClassAttribute:
    def test_default_memory_class_is_light(self):
        m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
        assert getattr(m, "memory_class", "light") == "light"

    def test_static_methods_are_light(self):
        m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="x")
        assert m.memory_class == "light"

    def test_sequential_methods_are_medium(self):
        from benchmark.sequential import AKMCSMethod, EGOMethod, MaxVarianceMethod
        assert EGOMethod().memory_class == "medium"
        assert MaxVarianceMethod().memory_class == "medium"
        assert AKMCSMethod().memory_class == "medium"

    def test_botorch_qei_is_medium(self):
        pytest.importorskip("botorch")
        from benchmark.bo_methods import BoTorchQEIMethod
        assert BoTorchQEIMethod().memory_class == "medium"

    def test_botorch_saasbo_is_heavy(self):
        pytest.importorskip("botorch")
        from benchmark.bo_methods import BoTorchSAASMethod
        try:
            m = BoTorchSAASMethod()
        except ImportError:  # SAAS not available in this BoTorch build
            pytest.skip("SAASBO unavailable")
        assert m.memory_class == "heavy"


# ─── Runner memory-class dispatch routing ───────────────────────────────
class TestRunnerMemoryClassRouting:
    def test_bo_n_jobs_defaults_to_quarter(self):
        runner = BenchmarkRunner(
            case_ids=["cantilever_beam_5d"],
            methods=[StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")],
            budgets=[5], n_repeats=1, n_jobs=8,
        )
        assert runner.bo_n_jobs == 2  # max(1, 8 // 4)

    def test_bo_n_jobs_explicit_override(self):
        runner = BenchmarkRunner(
            case_ids=["cantilever_beam_5d"],
            methods=[StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")],
            budgets=[5], n_repeats=1, n_jobs=8, bo_n_jobs=1,
        )
        assert runner.bo_n_jobs == 1

    def test_bo_n_jobs_clamped_to_min_1(self):
        runner = BenchmarkRunner(
            case_ids=["cantilever_beam_5d"],
            methods=[StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")],
            budgets=[5], n_repeats=1, n_jobs=1,  # // 4 = 0
        )
        assert runner.bo_n_jobs == 1

    def test_n_jobs_for_class_routing(self):
        runner = BenchmarkRunner(
            case_ids=["cantilever_beam_5d"],
            methods=[StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")],
            budgets=[5], n_repeats=1, n_jobs=8,
        )
        assert runner._n_jobs_for_class("light") == 8
        assert runner._n_jobs_for_class("medium") == 4   # 8 // 2
        assert runner._n_jobs_for_class("heavy") == 2    # 8 // 4
        # Unknown classes must fall through to light (full speed).
        assert runner._n_jobs_for_class("unknown") == 8

    def test_mem_cap_gb_stored(self):
        runner = BenchmarkRunner(
            case_ids=["cantilever_beam_5d"],
            methods=[StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")],
            budgets=[5], n_repeats=1, n_jobs=1, mem_cap_gb=12.0,
        )
        assert runner.mem_cap_gb == 12.0

    def test_rss_gb_returns_float_or_none(self):
        """psutil is optional; the helper must not raise when absent."""
        val = BenchmarkRunner._rss_gb()
        assert val is None or (isinstance(val, float) and val > 0)


# ─── End-to-end: memory-class-aware partitioning doesn't break results ──
class TestEndToEndMemoryClassAware:
    def test_mixed_mem_class_partitioning(self, tmp_path: Path):
        """A sweep with light + medium + heavy classes must produce the
        same rows as a single-class sweep (up to row count)."""

        class MemoryClassToyMethod(BenchmarkMethod):
            """Identity method: emits zero predictions; used to verify dispatch."""

            def __init__(self, name: str, memory_class: str):
                self.name = name
                self.family = "test_toy"
                self.memory_class = memory_class
                self.parallel_backend = "serial"

            def run(self, case, n_train, X_test, seed):
                Y_pred = np.zeros((X_test.shape[0], case.meta.dim_output))
                return Y_pred, None, {"n_train_effective": int(n_train)}

        methods = [
            MemoryClassToyMethod("light_m", "light"),
            MemoryClassToyMethod("medium_m", "medium"),
            MemoryClassToyMethod("heavy_m", "heavy"),
        ]
        runner = BenchmarkRunner(
            case_ids=["cantilever_beam_5d"],
            methods=methods,
            budgets=[5],
            n_repeats=2,
            n_test=10,
            n_jobs=1,
            heartbeat_s=0,
            output_dir=tmp_path,
            verbose=False,
        )
        result = runner.run()
        # 1 case * 3 methods * 1 budget * 2 reps = 6 rows
        assert len(result.df) == 6
        assert set(result.df["method"]) == {"light_m", "medium_m", "heavy_m"}
        assert (result.df["status"] == "ok").all()


# ─── Stream semantics preserved across flush boundaries ─────────────────
class TestStreamedCSVAndMemoryBuffer:
    def test_stream_and_flush_agree(self, tmp_path: Path):
        """Rows written to the streamed CSV must match the final ``result.df``."""
        m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
        runner = BenchmarkRunner(
            case_ids=["cantilever_beam_5d"],
            methods=[m],
            budgets=[5],
            n_repeats=2,
            n_test=20,
            n_jobs=1,
            heartbeat_s=0,
            output_dir=tmp_path,
            verbose=False,
        )
        result = runner.run()
        csv_rows = pd.read_csv(tmp_path / "results.csv")
        assert len(result.df) == len(csv_rows) == 2
        # Every (case_id, method, budget_mult, repeat) should appear in both.
        left_keys = set(
            map(tuple, result.df[["case_id", "method", "budget_mult", "repeat"]].values.tolist())
        )
        right_keys = set(
            map(tuple, csv_rows[["case_id", "method", "budget_mult", "repeat"]].values.tolist())
        )
        assert left_keys == right_keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
