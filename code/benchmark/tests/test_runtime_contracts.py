from __future__ import annotations

import pytest

from benchmark.runtime_contracts import (
    RESULT_SCHEMA_VERSION,
    classify_failure_category,
    csv_header_contract_errors,
    failure_category_is_consistent,
    invalid_protocol_digest_mask,
    protocol_digest_for_row,
    strict_integer_series,
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("MemoryError: unable to allocate array", "oom"),
        ("torch.cuda.OutOfMemoryError: CUDA out of memory.", "oom"),
        ("CUDA out of memory while fitting GP", "oom"),
        ("RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED", "oom"),
        ("cudnn error: CUDNN_STATUS_ALLOC_FAILED while allocating workspace", "oom"),
        ("terminate called after throwing std::bad_alloc", "oom"),
        ("slurmstepd: error: Detected 1 oom-kill event", "oom"),
        ("slurmstepd: error: Detected 2 oom_kill events in StepId=123.batch", "oom"),
        ("JobState=OUT_OF_MEMORY Reason=OutOfMemory", "oom"),
        ("slurmstepd: error: Exceeded step memory limit at some point", "oom"),
        ("slurmstepd: error: JOB 42 CANCELLED AT 2026-06-07T12:00:00 DUE TO MEMORY LIMIT", "oom"),
        ("memory cgroup out of memory: Killed process 123456 (python)", "oom"),
        ("Container status: OOMKilled", "oom"),
        ("Pod terminated with reason OOMKilled", "oom"),
        ("worker was killed with exit code 137", "oom"),
        ("joblib worker failed: returncode=137", "oom"),
        ("loky worker stopped with exitcode 137", "oom"),
        ("subprocess return code -9 after memory pressure", "oom"),
        ("subprocess returncode: -9 after cgroup memory pressure", "oom"),
        ("PBS: job killed: vmem 33554432kb exceeded limit", "oom"),
        ("TERM_MEMLIMIT: job killed after reaching LSF memory limit", "oom"),
        ("SGE: h_vmem limit exceeded for python worker", "oom"),
        ("memory monitor: MaxRSS above allocation; cancelling worker", "oom"),
        ("CANCELLED AT 2026-06-07T12:00:00 DUE TO TIME LIMIT", "timeout"),
        ("JobState=TIMEOUT Reason=TimeLimit", "timeout"),
        ("PBS: walltime 04:00:00 exceeded", "timeout"),
        ("TERM_RUNLIMIT: job killed after reaching LSF run time limit", "timeout"),
        ("SGE: h_rt limit exceeded for job 12345", "timeout"),
        ("subprocess failed with returncode=124 after timeout expired", "timeout"),
        ("command timed out after 45 minutes", "timeout"),
        ("job reached wall clock limit on queue normal", "timeout"),
        ("RuntimeError: failed to allocate 8589934592 bytes on CPU", "oom"),
        ("java.lang.OutOfMemoryError: Java heap space", "oom"),
        ("allocator failed while allocating memory for tensor", "oom"),
        ("TerminatedWorkerError: worker process was killed", "worker_crash"),
        ("process failed with SIGKILL", "worker_crash"),
        ("segmentation fault (core dumped)", "worker_crash"),
        ("Bus error (core dumped)", "worker_crash"),
        ("Illegal instruction (core dumped)", "worker_crash"),
        ("process killed by signal 11 (core dumped)", "worker_crash"),
        ("worker died after signal 15 during shutdown", "worker_crash"),
        ("slurmstepd: error: *** JOB 12345 ON c001 CANCELLED AT 2026-06-07T12:00:00 DUE TO NODE FAILURE ***", "worker_crash"),
        ("JobState=NODE_FAIL Reason=NodeDown", "worker_crash"),
        ("subprocess exited with exit status 139", "worker_crash"),
        ("subprocess exited with exit status 134", "worker_crash"),
        ("subprocess return code -7", "worker_crash"),
        ("subprocess return code -11", "worker_crash"),
        ("job step failed with ExitCode=0:9", "worker_crash"),
        ("container runtime reported signal: killed", "worker_crash"),
        ("process terminated by signal 15", "worker_crash"),
        ("worker subprocess returncode=-11", "worker_crash"),
        ("JobState=PREEMPTED Reason=Preempted", "scheduler"),
        ("JobState=CANCELLED Reason=None", "scheduler"),
        ("JobState=PENDING Reason=ReqNodeNotAvail", "scheduler"),
        ("JobState=PENDING Reason=Resources", "scheduler"),
        ("JobState=PENDING Reason=JobLaunchFailure", "scheduler"),
        ("JobState=PENDING Reason=DependencyNeverSatisfied", "scheduler"),
        ("Reason=Dependency never satisfied for afterok dependency", "scheduler"),
        ("JobState=LAUNCH_FAILED Reason=None", "scheduler"),
        ("JobState=BOOT_FAIL Reason=None", "scheduler"),
        ("sbatch: error: Batch job submission failed: Invalid account or partition", "scheduler"),
        ("slurmctld: job launch failure on allocated node", "scheduler"),
        ("JobState=HELD Reason=QOSGrpCpuLimit", "scheduler"),
        ("slurmstepd: error: *** JOB 9 CANCELLED AT 2026-06-07T12:00:00 DUE TO PREEMPTION ***", "scheduler"),
        ("Pod was Evicted due to node pressure", "scheduler"),
        ("Node had condition: DiskPressure", "scheduler"),
        ("slurmstepd: error: JOB 12345 ON c001 CANCELLED BY 1001", "scheduler"),
        ("PBS: job canceled by user before completion", "scheduler"),
        ("qdel removed job 12345 from queue", "scheduler"),
        ("OSError: [Errno 28] No space left on device", "storage"),
        ("OSError: ENOSPC while replacing results.csv", "storage"),
        ("OSError: EDQUOT while writing artifact", "storage"),
        ("Disk quota exceeded while writing results.csv", "storage"),
        ("Permission denied: '/scratch/full_benchmark/results.csv'", "storage"),
        ("Read-only file system during atomic rename", "storage"),
        ("Input/output error while flushing CSV", "storage"),
        ("failed to fsync streamed results.csv", "storage"),
        ("ephemeral-storage usage exceeds the total limit of containers", "storage"),
        ("Node condition met: nodefs.available below threshold", "storage"),
        ("OSError: [Errno 116] Stale file handle", "storage"),
        ("OSError: ESTALE while reading scratch artifact", "storage"),
        ("OSError: [Errno 107] Transport endpoint is not connected", "storage"),
        ("NFS server not responding while writing results.csv", "storage"),
        ("LustreError: OST eviction during checkpoint write", "storage"),
        ("OSError: [Errno 24] Too many open files", "storage"),
        ("remote I/O error while moving result shard", "storage"),
        ("file system is full on /scratch", "storage"),
        ("sqlite3.OperationalError: database or disk is full", "storage"),
        ("disk full while flushing logs", "storage"),
        ("ModuleNotFoundError: No module named 'neuralfoil'", "dependency"),
        ("FloatingPointError: non-finite loss encountered", "numerical"),
        ("job timed out after 7200 seconds", "timeout"),
    ],
)
def test_classify_failure_category_covers_server_failure_texts(message, expected):
    assert classify_failure_category("failed", message) == expected
    assert failure_category_is_consistent("failed", message, expected)


def test_status_timeout_overrides_error_text():
    assert classify_failure_category("timeout", "worker was killed") == "timeout"
    assert failure_category_is_consistent("timeout", "worker was killed", "timeout")


def test_ok_rows_must_use_ok_failure_category():
    assert classify_failure_category("ok", "") == "ok"
    assert failure_category_is_consistent("ok", "", "ok")
    assert not failure_category_is_consistent("ok", "", "exception")


@pytest.mark.parametrize(
    "text",
    [
        "case_id,method,budget_mult,repeat,status\ncase_a,method_a,5,0,ok,extra\n",
        "case_id,method,budget_mult,repeat,status\ncase_a,method_a,5,0\n",
    ],
)
def test_csv_contract_rejects_rows_with_wrong_field_count(tmp_path, text):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(text, encoding="utf-8")

    errors = csv_header_contract_errors(csv_path)

    assert any("different number of fields than the header" in err for err in errors)


def test_strict_integer_series_rejects_boolean_masquerades():
    import pandas as pd

    parsed, valid = strict_integer_series(
        pd.Series([True, False, "True", "False", "1", "1.0", "5.5", ""])
    )

    assert valid.tolist() == [False, False, False, False, True, True, False, False]
    assert parsed.iloc[4] == 1
    assert parsed.iloc[5] == 1


def _protocol_row(**overrides):
    row = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "case_id": "cantilever_beam_5d",
        "case_source_path": "doe_testbed/cases/structures/cantilever_beam_5d.py",
        "case_source_sha256": "a" * 64,
        "domain": "structures",
        "dim": 5,
        "n_output": 3,
        "method": "lhs+linear",
        "method_family": "static",
        "method_source_path": "benchmark/core.py;benchmark/samplers.py",
        "method_source_sha256": "b" * 64,
        "budget_mult": 5,
        "n_train": 25,
        "n_test": 1000,
        "repeat": 0,
        "seed": 12345,
        "test_seed": 0,
        "seed_strategy": "case_repeat",
        "seed_start": 1000,
        "status": "ok",
        "t_total_s": 1.0,
        "worker_host": "server-a",
    }
    row.update(overrides)
    return row


def test_protocol_digest_tracks_protocol_not_runtime_trace():
    base = _protocol_row()
    digest = protocol_digest_for_row(base)

    changed_runtime = dict(base)
    changed_runtime.update({
        "status": "timeout",
        "t_total_s": 600.0,
        "worker_host": "server-b",
    })
    changed_protocol = dict(base)
    changed_protocol["budget_mult"] = 10

    assert protocol_digest_for_row(changed_runtime) == digest
    assert protocol_digest_for_row(changed_protocol) != digest


def test_invalid_protocol_digest_mask_flags_tampered_rows():
    import pandas as pd

    good = _protocol_row()
    good["protocol_digest"] = protocol_digest_for_row(good)
    bad = dict(good)
    bad["n_test"] = 999

    mask = invalid_protocol_digest_mask(pd.DataFrame([good, bad]))

    assert mask.tolist() == [False, True]
