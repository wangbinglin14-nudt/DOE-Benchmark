"""Tests for the multi-backend dispatch + stall-recovery added in v0.5.

These complement ``test_smoke.py`` by exercising specifically:

* :attr:`benchmark.core.BenchmarkMethod.parallel_backend` routing
  (loky vs threading vs serial);
* :meth:`benchmark.core.BenchmarkRunner._effective_backend`
  degrades to ``'serial'`` when ``n_jobs <= 1``;
* per-cell soft timeouts surface as ``status='timeout'`` rows;
* mixing methods with different ``parallel_backend`` declarations
  in the same sweep keeps both partitions running and produces the
  expected number of rows;
* :func:`benchmark.core._call_with_soft_timeout` returns
  ``(None, reason)`` instead of raising on slow callables;
* :func:`benchmark.core._iter_with_watchdog` aborts a stalled
  generator and ``BenchmarkRunner.run`` recovers gracefully;
* :class:`benchmark.core._Heartbeat` emits a state line on its own
  thread without blocking the caller;
* ``max_wall_clock_s`` triggers an early stop and preserves the
  partial CSV.
"""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Optional, Tuple

import numpy as np
import pytest

import doe_testbed as db
from benchmark import (
    BenchmarkMethod,
    BenchmarkRunner,
    StaticDesignMethod,
)
from benchmark.samplers import LHSSampler
from benchmark.surrogates import LinearSurrogate


ROOT = Path(__file__).resolve().parents[2]


# ─── Test fixtures: deterministic synthetic methods ──────────────────────
class _ConstantMethod(BenchmarkMethod):
    """Returns Y_pred = const for every test point.  No surrogate fit."""

    family = "test"

    def __init__(self, name: str, *, sleep_s: float = 0.0,
                 backend: str = "loky") -> None:
        self.name = name
        self.sleep_s = float(sleep_s)
        self.parallel_backend = backend  # instance attr overrides class default

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        Y_pred = np.zeros((X_test.shape[0], case.meta.dim_output))
        return Y_pred, None, {"sleep_s": self.sleep_s}


class _FailingMethod(BenchmarkMethod):
    """Raises inside method.run so runner records a failed cell."""

    family = "test"
    name = "failing"
    parallel_backend = "serial"

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        raise RuntimeError("synthetic failure")


class _ProtocolClobberMethod(BenchmarkMethod):
    """Returns diagnostics that try to overwrite runner-owned protocol fields."""

    family = "test"
    name = "protocol_clobber"
    parallel_backend = "serial"

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        y = np.zeros((X_test.shape[0], case.meta.dim_output))
        return y, None, {
            "case_id": "malicious_case",
            "seed": -1,
            "protocol_digest": "0" * 64,
            "custom_diagnostic": "kept",
        }


def _resume_row(**overrides):
    from benchmark.core import case_source_fingerprint, method_source_fingerprint
    from benchmark.runtime_contracts import RESULT_SCHEMA_VERSION, protocol_digest_for_row

    case_id = str(overrides.pop("case_id", "cantilever_beam_5d"))
    case = db.get_case(case_id)
    budget_mult = overrides.get("budget_mult", 5)
    method = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    row = {
        "case_id": case_id,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        **case_source_fingerprint(case),
        "domain": case.meta.domain,
        "dim": case.meta.dim_input,
        "n_output": case.meta.dim_output,
        "method": "lhs+linear",
        "method_family": "static",
        **method_source_fingerprint(method),
        "budget_mult": 5,
        "n_train": int(float(budget_mult) * case.meta.dim_input),
        "n_test": 20,
        "repeat": 0,
        "seed": 1000,
        "seed_strategy": "case_repeat",
        "seed_start": 1000,
        "test_seed": 0,
        "status": "ok",
        "failure_category": "ok",
        "t_total_s": 0.0,
        "error": "",
        "nrmse": 0.0,
        "worker_host": "server-a",
        "worker_pid": 12345,
        "cell_started_utc": "2026-06-01T00:00:00+00:00",
        "cell_finished_utc": "2026-06-01T00:00:01+00:00",
        "rss_gb_start": 1.25,
        "rss_gb_end": 1.50,
        "rss_gb_peak": 1.75,
    }
    row.update(overrides)
    if "protocol_digest" not in overrides:
        row["protocol_digest"] = protocol_digest_for_row(row)
    return row


def _assert_worker_provenance(row) -> None:
    for field in [
        "worker_host",
        "worker_pid",
        "cell_started_utc",
        "cell_finished_utc",
        "rss_gb_start",
        "rss_gb_end",
        "rss_gb_peak",
    ]:
        assert field in row
    assert str(row["worker_host"]).strip()
    assert int(row["worker_pid"]) > 0
    assert "+00:00" in str(row["cell_started_utc"])
    assert "+00:00" in str(row["cell_finished_utc"])
    assert float(row["rss_gb_peak"]) >= float(row["rss_gb_start"])
    assert float(row["rss_gb_peak"]) >= float(row["rss_gb_end"])


# ─── _effective_backend ──────────────────────────────────────────────────
def test_effective_backend_serial_when_n_jobs_le_1():
    m_loky = _ConstantMethod("a", backend="loky")
    m_thread = _ConstantMethod("b", backend="threading")
    assert BenchmarkRunner._effective_backend(m_loky, n_jobs=1) == "serial"
    assert BenchmarkRunner._effective_backend(m_thread, n_jobs=1) == "serial"
    assert BenchmarkRunner._effective_backend(m_loky, n_jobs=0) == "serial"


def test_effective_backend_honours_declaration_when_parallel():
    m_loky = _ConstantMethod("a", backend="loky")
    m_thread = _ConstantMethod("b", backend="threading")
    m_serial = _ConstantMethod("c", backend="serial")
    m_unknown = _ConstantMethod("d", backend="ipyparallel")  # bogus → loky

    assert BenchmarkRunner._effective_backend(m_loky, n_jobs=4) == "loky"
    assert BenchmarkRunner._effective_backend(m_thread, n_jobs=4) == "threading"
    assert BenchmarkRunner._effective_backend(m_serial, n_jobs=4) == "serial"
    assert BenchmarkRunner._effective_backend(m_unknown, n_jobs=4) == "loky"


# ─── Per-cell timeout ────────────────────────────────────────────────────
def test_timeout_records_status_timeout(tmp_path):
    """A method that sleeps past the cap must yield status='timeout'."""
    slow = _ConstantMethod("slow", sleep_s=2.0, backend="loky")
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[slow],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        n_jobs=1,         # serial: timeout fires inside _run_one_cell
        timeout_s=0.2,    # well below the 2 s sleep
        output_dir=tmp_path,
        verbose=False,
    )
    result = runner.run()
    assert len(result.df) == 1
    assert result.df.iloc[0]["status"] == "timeout"
    assert result.df.iloc[0]["failure_category"] == "timeout"
    assert "timeout after" in str(result.df.iloc[0]["error"])
    assert float(result.df.iloc[0]["t_total_s"]) >= 0.0
    _assert_worker_provenance(result.df.iloc[0])


def test_failed_cell_records_elapsed_time(tmp_path):
    """Failed cells must retain timing for penalized cost analysis."""
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[_FailingMethod()],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        n_jobs=1,
        output_dir=tmp_path,
        verbose=False,
    )
    result = runner.run()
    row = result.df.iloc[0]
    assert row["status"] == "failed"
    assert row["failure_category"] == "exception"
    assert "synthetic failure" in str(row["error"])
    assert float(row["t_total_s"]) >= 0.0
    assert len(str(row["case_source_sha256"])) == 64
    _assert_worker_provenance(row)


def test_method_info_cannot_override_protocol_provenance(tmp_path):
    """Method diagnostics must not overwrite runner-owned protocol fields."""
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[_ProtocolClobberMethod()],
        budgets=[5],
        n_repeats=1,
        n_test=10,
        n_jobs=1,
        output_dir=tmp_path,
        verbose=False,
    )

    result = runner.run()
    row = result.df.iloc[0]

    assert row["case_id"] == "cantilever_beam_5d"
    assert int(row["seed"]) != -1
    assert row["protocol_digest"] != "0" * 64
    assert row["custom_diagnostic"] == "kept"


def test_timeout_falls_through_when_below_cap(tmp_path):
    """A fast method must still report status='ok' even with a cap."""
    fast = _ConstantMethod("fast", sleep_s=0.0, backend="loky")
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[fast],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        n_jobs=1,
        timeout_s=5.0,
        output_dir=tmp_path,
        verbose=False,
    )
    result = runner.run()
    assert len(result.df) == 1
    assert result.df.iloc[0]["status"] == "ok"
    assert result.df.iloc[0]["failure_category"] == "ok"
    assert int(result.df.iloc[0]["n_test"]) == 20
    assert int(result.df.iloc[0]["test_seed"]) == 0
    assert "case_source_path" in result.df.columns
    assert "case_source_sha256" in result.df.columns
    assert "error" in result.df.columns
    assert result.df.iloc[0]["error"] == ""
    _assert_worker_provenance(result.df.iloc[0])


def test_bo_timeout_overrides_for_threading_backend(tmp_path):
    """`bo_timeout_s` must be used for threading methods, `timeout_s` for the rest."""
    fast_loky = _ConstantMethod("fast_loky", sleep_s=0.0, backend="loky")
    slow_thread = _ConstantMethod("slow_thread", sleep_s=2.0, backend="threading")
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[fast_loky, slow_thread],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        n_jobs=1,
        timeout_s=10.0,    # generous for loky
        bo_timeout_s=0.2,  # tight for threading
        output_dir=tmp_path,
        verbose=False,
    )
    result = runner.run()
    by_method = {r["method"]: r["status"] for _, r in result.df.iterrows()}
    assert by_method["fast_loky"] == "ok"
    assert by_method["slow_thread"] == "timeout"


# ─── Mixed-backend dispatch ──────────────────────────────────────────────
def test_mixed_backends_produce_full_row_count(tmp_path):
    """Mixing loky + threading + serial methods must complete every cell."""
    m_static = StaticDesignMethod(
        LHSSampler(), LinearSurrogate(), name="lhs+linear",
    )
    # Force one of each backend declaration.  StaticDesignMethod inherits
    # the default 'loky' from BenchmarkMethod; we don't touch it.
    m_threadlike = _ConstantMethod("thread_like", backend="threading")
    m_serial_only = _ConstantMethod("serial_only", backend="serial")

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m_static, m_threadlike, m_serial_only],
        budgets=[5],
        n_repeats=2,
        n_test=20,
        n_jobs=2,    # actually exercise the parallel path
        output_dir=tmp_path,
        verbose=False,
    )
    result = runner.run()
    # 1 case * 3 methods * 1 budget * 2 reps = 6 rows
    assert len(result.df) == 6
    assert set(result.df["method"]) == {"lhs+linear", "thread_like", "serial_only"}
    assert (result.df["status"] == "ok").all()


def test_skip_existing_round_trip(tmp_path):
    """Running once + re-running with --skip-existing must skip everything."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    common = dict(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=2,
        n_test=20,
        output_dir=tmp_path,
        verbose=False,
    )
    runner = BenchmarkRunner(n_jobs=1, **common)
    res1 = runner.run()
    assert len(res1.df) == 2

    # Second run with skip-existing: nothing new.
    runner2 = BenchmarkRunner(n_jobs=1, skip_existing=True, **common)
    res2 = runner2.run()
    # In-memory result holds only newly computed rows; should be empty.
    assert len(res2.df) == 0


def test_skip_existing_requires_matching_run_config_snapshot(tmp_path):
    """Resume must fail closed when results.csv exists but run_config.json is missing."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    common = dict(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        verbose=False,
    )
    BenchmarkRunner(n_jobs=1, **common).run()
    (tmp_path / "run_config.json").unlink()

    runner = BenchmarkRunner(n_jobs=1, skip_existing=True, **common)
    with pytest.raises(ValueError, match="requires a sibling run_config.json startup snapshot"):
        runner.run()


def test_skip_existing_rejects_run_config_protocol_drift(tmp_path):
    """Resume must not trust a sibling startup snapshot from another Stage-2 grid."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    common = dict(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        verbose=False,
    )
    BenchmarkRunner(n_jobs=1, **common).run()

    stale_runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[10],
        n_repeats=2,
        n_test=30,
        output_dir=tmp_path,
        verbose=False,
    )
    stale_config = stale_runner._run_config_payload(total_cells=2)
    stale_config["skip_existing"] = True
    (tmp_path / "run_config.json").write_text(
        json.dumps(stale_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runner = BenchmarkRunner(n_jobs=1, skip_existing=True, **common)
    with pytest.raises(ValueError, match="run_config.json does not match the current declared Stage-2 grid/protocol"):
        runner.run()


def test_skip_existing_rejects_run_config_without_temp_storage_snapshot(tmp_path):
    """Resume must not continue from a startup snapshot missing scratch provenance."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    common = dict(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        verbose=False,
    )
    BenchmarkRunner(n_jobs=1, **common).run()

    config_path = tmp_path / "run_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.pop("temp_storage")
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runner = BenchmarkRunner(n_jobs=1, skip_existing=True, **common)
    with pytest.raises(ValueError, match="temp_storage startup snapshot"):
        runner.run()


def test_runner_writes_server_run_config_snapshot(tmp_path):
    """Every streamed sweep should leave a durable configuration manifest."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        timeout_s=10.0,
        bo_timeout_s=20.0,
        ref_timeout_s=30.0,
        heartbeat_s=0,
        verbose=False,
    )

    runner.run()

    config_path = tmp_path / "run_config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["artifact"] == "doe-benchmark-stage2-run-config"
    assert config["result_schema_version"] == 2
    assert config["total_cells"] == 1
    assert config["cases"][0]["case_id"] == "cantilever_beam_5d"
    assert len(config["cases"][0]["case_source_sha256"]) == 64
    assert config["methods"][0]["method"] == "lhs+linear"
    assert config["methods"][0]["method_family"] == m.family
    assert len(config["methods"][0]["method_source_sha256"]) == 64
    assert config["budgets"] == [5]
    assert config["n_repeats"] == 1
    assert config["n_test"] == 20
    assert config["test_seed"] == 0
    assert config["seed_strategy"] == "case_repeat"
    assert config["timeout_s"] == 10.0
    assert config["bo_timeout_s"] == 20.0
    assert config["ref_timeout_s"] == 30.0
    assert "OMP_NUM_THREADS" in config["thread_env"]
    environment = config["environment"]
    assert environment["schema_version"] == 1
    assert environment["python_version"]
    assert environment["python_executable"]
    assert environment["platform"]
    assert environment["node"]
    assert environment["cwd"]
    for package in [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "gpytorch",
        "botorch",
        "neuralfoil",
        "aerosandbox",
        "psutil",
    ]:
        assert package in environment["packages"]
        assert environment["packages"][package]["status"] in {"installed", "missing", "error"}
    temp_storage = config["temp_storage"]
    assert temp_storage["schema_version"] == 1
    assert isinstance(temp_storage["resolved_path"], str)
    assert isinstance(temp_storage["exists"], bool)
    assert isinstance(temp_storage["writable"], bool)
    assert "TMPDIR" in temp_storage["env"]
    assert not list(tmp_path.glob(".run_config.json.tmp.*.json"))


def test_runner_run_config_snapshot_updates_on_recompute(tmp_path):
    """A fresh recompute must not leave a stale run_config.json behind."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        heartbeat_s=0,
        verbose=False,
    ).run()
    first = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))

    BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[10],
        n_repeats=2,
        n_test=30,
        output_dir=tmp_path,
        heartbeat_s=0,
        verbose=False,
    ).run()
    second = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))

    assert first["budgets"] == [5]
    assert second["budgets"] == [10]
    assert second["n_repeats"] == 2
    assert second["n_test"] == 30
    assert second["total_cells"] == 2


def test_full_recompute_replaces_existing_results_csv(tmp_path):
    """Without skip-existing, a rerun must replace results.csv instead of appending."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([_resume_row(repeat=99, n_train=999)]).to_csv(
        tmp_path / "results.csv",
        index=False,
    )

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=False,
        verbose=False,
    )
    result = runner.run()
    written = pd.read_csv(tmp_path / "results.csv")

    assert len(result.df) == 1
    assert len(written) == 1
    assert int(written.iloc[0]["repeat"]) == 0
    assert int(written.iloc[0]["n_train"]) == 25


def test_stream_append_row_evolves_schema_without_losing_existing_rows(tmp_path):
    from benchmark.core import BenchmarkRunner
    pd = pytest.importorskip("pandas")

    csv_path = tmp_path / "results.csv"
    BenchmarkRunner._stream_append_row(csv_path, {"case_id": "c1", "metric": 1.0})
    BenchmarkRunner._stream_append_row(
        csv_path,
        {"case_id": "c2", "metric": 2.0, "new_metric": 3.0},
    )

    written = pd.read_csv(csv_path)

    assert written.columns.tolist() == ["case_id", "metric", "new_metric"]
    assert written["case_id"].tolist() == ["c1", "c2"]
    assert written["metric"].tolist() == [1.0, 2.0]
    assert pd.isna(written.loc[0, "new_metric"])
    assert written.loc[1, "new_metric"] == 3.0
    assert not list(tmp_path.glob(".results.csv.tmp.*.csv"))


def test_stream_append_row_fsyncs_each_write_path(tmp_path, monkeypatch):
    import benchmark.core as core

    calls: list[int] = []
    monkeypatch.setattr(core.os, "fsync", lambda fd: calls.append(int(fd)))

    csv_path = tmp_path / "results.csv"
    core.BenchmarkRunner._stream_append_row(csv_path, {"case_id": "c1", "metric": 1.0})
    core.BenchmarkRunner._stream_append_row(csv_path, {"case_id": "c2", "metric": 2.0})
    core.BenchmarkRunner._stream_append_row(
        csv_path,
        {"case_id": "c3", "metric": 3.0, "new_metric": 4.0},
    )

    assert len(calls) >= 3


def test_stream_append_row_refuses_to_overwrite_corrupt_existing_csv(tmp_path):
    from benchmark.core import BenchmarkRunner

    csv_path = tmp_path / "results.csv"
    csv_path.write_text('case_id,metric\n"unterminated,1\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="existing CSV is not parseable"):
        BenchmarkRunner._stream_append_row(
            csv_path,
            {"case_id": "c2", "metric": 2.0, "new_metric": 3.0},
        )

    assert csv_path.read_text(encoding="utf-8") == 'case_id,metric\n"unterminated,1\n'


def test_skip_existing_rejects_stale_results_schema(tmp_path):
    """Resume must reject old CSVs that predate the paper-audit row schema."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    stale = _resume_row()
    for col in (
        "case_source_path",
        "case_source_sha256",
        "method_source_path",
        "method_source_sha256",
        "n_test",
        "test_seed",
        "error",
        "failure_category",
    ):
        stale.pop(col)
    pd.DataFrame([stale]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="stale or incomplete schema"):
        runner.run()


def test_skip_existing_rejects_unparseable_results_csv(tmp_path):
    """Resume must fail closed instead of appending to a corrupt CSV."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    csv_path = tmp_path / "results.csv"
    bad_text = 'case_id,method\n"unterminated,lhs+linear\n'
    csv_path.write_text(bad_text, encoding="utf-8")

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="unreadable or corrupt results.csv"):
        runner.run()

    assert csv_path.read_text(encoding="utf-8") == bad_text


def test_skip_existing_rejects_header_only_results_csv(tmp_path):
    """Resume must reject a stale startup CSV that has no completed cells."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "results.csv"
    pd.DataFrame(columns=list(_resume_row().keys())).to_csv(csv_path, index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="header but no completed data rows"):
        runner.run()


def test_skip_existing_rejects_parseable_truncated_tail_row(tmp_path):
    """Resume must reject a crash-truncated tail row even if pandas parses it."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "results.csv"
    good = _resume_row()
    pd.DataFrame([good]).to_csv(csv_path, index=False)
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    # Simulate a dead machine after only the protocol columns were durable.
    # pandas can parse this as a row with NaNs in the runtime/provenance tail;
    # resume must still fail before treating the cell as completed.
    csv_path.write_text(
        header
        + "\n"
        + ",".join(
            [
                str(good["case_id"]),
                str(good["case_source_path"]),
                str(good["case_source_sha256"]),
                str(good["domain"]),
                str(good["dim"]),
                str(good["n_output"]),
                str(good["method"]),
                str(good["method_family"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(
        ValueError,
        match="CSV contract errors|missing protocol key values|missing status values|invalid t_total_s",
    ):
        runner.run()


def test_skip_existing_rejects_ragged_row_before_pandas_index_inference(tmp_path):
    """Resume must reject extra fields before pandas can reinterpret them."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "case_id,method,budget_mult,repeat,status\n"
        "cantilever_beam_5d,lhs+linear,5,0,ok,extra_field\n",
        encoding="utf-8",
    )

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="CSV contract errors"):
        runner.run()


def test_skip_existing_rejects_symlinked_results_csv(tmp_path, monkeypatch):
    """Resume must not read from or append to a linked external CSV target."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "results.csv"
    pd.DataFrame([_resume_row()]).to_csv(csv_path, index=False)
    real_is_symlink = Path.is_symlink

    def fake_is_symlink(path):
        if Path(path) == csv_path:
            return True
        return real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="symlinked results.csv"):
        runner.run()


def test_stream_append_rejects_symlinked_results_csv(tmp_path, monkeypatch):
    """Streaming writes must fail closed if the output CSV path is linked."""
    csv_path = tmp_path / "results.csv"
    real_is_symlink = Path.is_symlink

    def fake_is_symlink(path):
        if Path(path) == csv_path:
            return True
        return real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    with pytest.raises(RuntimeError, match="symlinked results.csv"):
        BenchmarkRunner._stream_append_row(csv_path, {"case_id": "cantilever_beam_5d"})


def test_skip_existing_rejects_rows_from_other_method_pool(tmp_path):
    """Resume must fail fast if stale results contain a method outside the pool."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        _resume_row(),
        _resume_row(method="botorch_sequential"),
    ]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="unexpected methods"):
        runner.run()


def test_skip_existing_rejects_rows_from_other_case_grid(tmp_path):
    """Resume must fail fast if stale results contain a case outside the run grid."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        _resume_row(),
        _resume_row(case_id="pressure_vessel_4d"),
    ]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="unexpected cases"):
        runner.run()


def test_skip_existing_rejects_rows_from_other_budget_or_repeat(tmp_path):
    """Resume must reject stale rows outside the declared budget/repeat grid."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        _resume_row(budget_mult=10, repeat=3, n_train=50),
    ]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=2,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="unexpected budgets=.*unexpected repeats"):
        runner.run()


def test_skip_existing_rejects_non_integer_budget_or_repeat(tmp_path):
    """Resume must not truncate malformed grid IDs such as 5.5 or 0.5."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        _resume_row(budget_mult="5.5", repeat="0.5"),
    ]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="non-integer budget_mult.*non-integer repeat"):
        runner.run()


@pytest.mark.parametrize("field", ["case_id", "method", "budget_mult", "repeat"])
def test_skip_existing_rejects_missing_protocol_key_values(tmp_path, field):
    """Resume must fail before preserving rows with blank grid identifiers."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    row = _resume_row()
    row[field] = None
    pd.DataFrame([row]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match=f"missing protocol key values.*{field}=1"):
        runner.run()


def test_skip_existing_rejects_duplicate_cells(tmp_path):
    """Resume must not hide duplicate cells behind a set() de-duplication."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        _resume_row(),
        _resume_row(),
    ]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="duplicate cells"):
        runner.run()


@pytest.mark.parametrize(
    "override, match",
    [
        ({"status": "rerun"}, "invalid status values"),
        ({"failure_category": "rerun"}, "invalid failure_category values"),
        ({"failure_category": ""}, "missing failure_category values"),
        (
            {
                "status": "timeout",
                "error": "timeout after 600s",
                "failure_category": "exception",
            },
            "failure_category inconsistent with status/error",
        ),
        ({"nrmse": float("nan")}, "invalid ok nrmse rows"),
        (
            {"status": "failed", "error": "RuntimeError: boom", "nrmse": 0.1},
            "timeout/failed rows with non-empty nrmse",
        ),
        ({"rmse": -1.0}, "invalid ok rmse rows"),
        (
            {"status": "timeout", "error": "timeout after 600s", "rmse": 0.1},
            "timeout/failed rows with non-empty rmse",
        ),
        ({"mae": float("inf")}, "invalid ok mae rows"),
        ({"r2": "not-a-number"}, "invalid ok r2 rows"),
        ({"t_total_s": -1.0}, "invalid t_total_s rows"),
        ({"error": "stale error on ok row"}, "ok rows with non-empty error"),
        ({"status": "timeout", "error": ""}, "timeout/failed rows with empty error"),
        ({"status": "failed", "error": ""}, "timeout/failed rows with empty error"),
    ],
)
def test_skip_existing_rejects_corrupt_completed_rows(tmp_path, override, match):
    """Resume must not skip corrupt rows that downstream paper gates reject."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([_resume_row(**override)]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match=match):
        runner.run()


@pytest.mark.parametrize(
    "override, match",
    [
        ({"worker_host": ""}, "empty worker_host"),
        ({"worker_pid": 0}, "invalid worker_pid"),
        ({"cell_started_utc": "not-a-time"}, "invalid cell_started_utc"),
        (
            {
                "cell_started_utc": "2026-06-01T00:00:02+00:00",
                "cell_finished_utc": "2026-06-01T00:00:01+00:00",
            },
            "cell_finished_utc before cell_started_utc",
        ),
        ({"t_total_s": 100.0}, "t_total_s inconsistent with cell runtime timestamps"),
        ({"rss_gb_start": -1.0}, "invalid rss_gb_start"),
        ({"rss_gb_peak": 1.0}, "rss_gb_peak below start/end"),
    ],
)
def test_skip_existing_rejects_corrupt_runtime_trace_rows(tmp_path, override, match):
    """Resume must not skip rows with stale or impossible runtime provenance."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([_resume_row(**override)]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match=match):
        runner.run()


def test_skip_existing_rejects_boolean_worker_pid(tmp_path):
    """Resume must not accept True as worker_pid=1 after CSV type inference."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([_resume_row(worker_pid=True)]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="invalid worker_pid"):
        runner.run()


def test_skip_existing_rejects_protocol_metadata_mismatches(tmp_path):
    """Resume must fail before skipping rows from a different seed/test protocol."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        _resume_row(test_seed=123, seed=9999, n_test=30),
    ]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="metadata/protocol mismatches"):
        runner.run()


def test_skip_existing_rejects_tampered_protocol_digest(tmp_path):
    """Resume must reject rows whose protocol digest no longer matches."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        _resume_row(protocol_digest="0" * 64),
    ]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="protocol_digest mismatch"):
        runner.run()


def test_skip_existing_rejects_stale_source_fingerprints(tmp_path):
    """Resume must fail before skipping rows generated by older case/method code."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    pd.DataFrame([
        _resume_row(
            case_source_sha256="0" * 64,
            method_source_sha256="1" * 64,
        ),
    ]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="metadata/protocol mismatches"):
        runner.run()


def test_skip_existing_accepts_absolute_source_paths_with_matching_hashes(tmp_path):
    """Resume should not reject a portable path-format difference by itself."""
    from benchmark.core import derive_seed
    from benchmark.runtime_contracts import protocol_digest_for_row

    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")
    pd = pytest.importorskip("pandas")
    row = _resume_row()
    row["case_source_path"] = (ROOT / row["case_source_path"]).as_posix()
    row["method_source_path"] = (ROOT / row["method_source_path"]).as_posix()
    row["method_family"] = m.family
    row["seed"] = derive_seed(
        row["case_id"],
        int(row["repeat"]),
        strategy="case_repeat",
        seed_start=1000,
    )
    row["protocol_digest"] = protocol_digest_for_row(row)
    pd.DataFrame([row]).to_csv(tmp_path / "results.csv", index=False)

    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        output_dir=tmp_path,
        skip_existing=True,
        verbose=False,
    )
    (tmp_path / "run_config.json").write_text(
        json.dumps(runner._run_config_payload(total_cells=1), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = runner.run()

    assert len(result.df) == 0
    restored = pd.read_csv(tmp_path / "results.csv")
    assert len(restored) == 1


# ─── Stall-resilience helpers ────────────────────────────────────────────
def test_call_with_soft_timeout_returns_reason_on_overrun():
    from benchmark.core import _call_with_soft_timeout

    def slow():
        time.sleep(2.0)
        return "should not see this"

    result, reason = _call_with_soft_timeout(slow, timeout_s=0.1)
    assert result is None
    assert reason and "timeout" in reason


def test_call_with_soft_timeout_passes_through_when_fast():
    from benchmark.core import _call_with_soft_timeout

    def quick(x, *, y):
        return x + y

    result, reason = _call_with_soft_timeout(quick, 1, y=2, timeout_s=5.0)
    assert reason is None
    assert result == 3


def test_call_with_soft_timeout_no_cap_passes_through():
    from benchmark.core import _call_with_soft_timeout

    def quick():
        return 42

    result, reason = _call_with_soft_timeout(quick, timeout_s=None)
    assert reason is None
    assert result == 42


def test_call_with_soft_timeout_captures_exception():
    from benchmark.core import _call_with_soft_timeout

    def boom():
        raise ValueError("nope")

    result, reason = _call_with_soft_timeout(boom, timeout_s=1.0)
    assert result is None
    assert "ValueError" in reason and "nope" in reason


def test_iter_with_watchdog_aborts_on_stalled_generator():
    """A generator that stops yielding must surface as RuntimeError."""
    from benchmark.core import _iter_with_watchdog

    def stalling_gen():
        yield 1
        yield 2
        time.sleep(2.0)        # silence > watchdog
        yield 3

    with pytest.raises(RuntimeError, match="stalled"):
        # Force list comprehension to drain the generator.
        list(_iter_with_watchdog(stalling_gen(), watchdog_s=0.3))


def test_iter_with_watchdog_passthrough_when_disabled():
    from benchmark.core import _iter_with_watchdog

    def gen():
        yield from range(5)

    assert list(_iter_with_watchdog(gen(), watchdog_s=None)) == [0, 1, 2, 3, 4]


# ─── Reference-set timeout ───────────────────────────────────────────────
def test_ref_timeout_skips_case_when_evaluate_hangs(tmp_path, monkeypatch):
    """A hung reference-set evaluate() must skip the case, not freeze the runner."""
    m = StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")

    # Monkey-patch the runner's reference-set helper so the first
    # case "hangs" (sleeps past the cap) and the second runs normally.
    real_build = BenchmarkRunner._build_reference_set
    seen = {"calls": 0}

    def slow_then_fast(self, case):
        seen["calls"] += 1
        if seen["calls"] == 1:
            time.sleep(3.0)
        return real_build(self, case)

    monkeypatch.setattr(
        BenchmarkRunner, "_build_reference_set", slow_then_fast
    )

    # ref_timeout_s=1.5 s is comfortably above the cold-start evaluate
    # time of the second analytical case and well below the
    # monkey-patched 3.0 s sleep.  Keep the second case deliberately
    # simple; aero cases may pay optional-backend cold-start costs when
    # this test runs in isolation.
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d", "pressure_vessel_4d"],
        methods=[m],
        budgets=[5],
        n_repeats=1,
        n_test=20,
        n_jobs=1,
        ref_timeout_s=1.5,
        heartbeat_s=0,           # silence heartbeat in tests
        output_dir=tmp_path,
        verbose=False,
    )
    result = runner.run()
    # First case skipped → only the second case produces a row.
    assert len(result.df) == 1
    assert result.df.iloc[0]["case_id"] == "pressure_vessel_4d"


# ─── Heartbeat ──────────────────────────────────────────────────────────
def test_heartbeat_emits_at_least_one_message(capsys):
    from benchmark.core import _Heartbeat

    state = {"i": 0}

    def msg():
        state["i"] += 1
        return f"tick {state['i']}"

    with _Heartbeat(msg, interval_s=0.05, stream_name="probe"):
        time.sleep(0.2)

    captured = capsys.readouterr().out
    # At least one heartbeat line; format `[probe] tick N`.
    assert "[probe] tick" in captured


def test_heartbeat_disabled_when_interval_zero(capsys):
    from benchmark.core import _Heartbeat

    def msg():
        return "should not appear"

    with _Heartbeat(msg, interval_s=0.0):
        time.sleep(0.1)

    captured = capsys.readouterr().out
    assert "should not appear" not in captured


# ─── Wall-clock budget ───────────────────────────────────────────────────
def test_max_wall_clock_stops_dispatch_early(tmp_path):
    """Once the budget is exhausted the runner must not start new cases."""
    slow = _ConstantMethod("slow", sleep_s=0.3, backend="loky")
    runner = BenchmarkRunner(
        # Two cases, multiple repeats — without a cap this would do 6 rows
        # at 0.3 s each = ~1.8 s; with a 0.4 s cap we expect ≤ 3 rows.
        case_ids=["cantilever_beam_5d", "airfoil_stall_2d"],
        methods=[slow],
        budgets=[5],
        n_repeats=3,
        n_test=20,
        n_jobs=1,
        max_wall_clock_s=0.4,
        heartbeat_s=0,
        output_dir=tmp_path,
        verbose=False,
    )
    t0 = time.time()
    result = runner.run()
    elapsed = time.time() - t0
    # Sanity-checks: we honoured the budget (with grace), and we did
    # *some* work but stopped before finishing.
    assert elapsed < 2.0
    assert 0 < len(result.df) < 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
