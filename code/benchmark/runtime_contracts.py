"""Shared runtime-row integrity checks for benchmark result archives."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


RUNTIME_DURATION_ABS_TOLERANCE_S = 2.0
RUNTIME_DURATION_REL_TOLERANCE = 0.25
ALLOWED_FAILURE_CATEGORIES = {
    "ok",
    "timeout",
    "oom",
    "dependency",
    "backend",
    "numerical",
    "worker_crash",
    "scheduler",
    "storage",
    "exception",
}

RESULT_SCHEMA_VERSION = 2
PROTOCOL_DIGEST_FIELDS = (
    "result_schema_version",
    "case_id",
    "case_source_path",
    "case_source_sha256",
    "domain",
    "dim",
    "n_output",
    "method",
    "method_family",
    "method_source_path",
    "method_source_sha256",
    "budget_mult",
    "n_train",
    "n_test",
    "repeat",
    "seed",
    "test_seed",
    "seed_strategy",
    "seed_start",
)
PROTOCOL_DIGEST_COLUMNS = set(PROTOCOL_DIGEST_FIELDS) | {"protocol_digest"}

_PROTOCOL_INTEGER_FIELDS = {
    "result_schema_version",
    "dim",
    "n_output",
    "budget_mult",
    "n_train",
    "n_test",
    "repeat",
    "seed",
    "test_seed",
    "seed_start",
}

_FAILURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "timeout",
        re.compile(
            r"\b(timeout|timed\s*out|time\s*limit|deadline|job\s*state\s*[:=]\s*"
            r"timeout|cancelled\s+at\b.*\bdue\s+to\s+time\s+limit|due\s+to\s+"
            r"time\s+limit|walltime\b.*\b(?:exceeded|reached)|"
            r"wall\s*clock\s+limit|wallclock\s+limit|"
            r"term_runlimit|run\s*time\s+limit|h_rt\s+limit\s+exceeded|"
            r"command\s+timed\s+out|timeout\s+expired|"
            r"(?:exit\s*(?:code|status)?|return\s*code|returncode)\s*[:=]?\s*124)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oom",
        re.compile(
            r"\b(memoryerror|outofmemoryerror|out\s+of\s+memory|"
            r"cuda\s+out\s+of\s+memory|cublas_status_alloc_failed|"
            r"cudnn_status_alloc_failed|std::bad_alloc|bad_alloc|oom|"
            r"oomkilled|taskoom|"
            r"cannot\s+allocate\s+memory|killed\s+process|slurmstepd:.*oom|"
            r"java\.lang\.outofmemoryerror|"
            r"(?:allocation|allocating)\s+.*\b(?:bytes|memory)\b.*\bfailed|"
            r"failed\s+to\s+allocate\s+.*\b(?:bytes|memory)\b|"
            r"allocator\s+failed.*\bmemory\b|"
            r"oom\s+killer|oom[-_\s]kill|out_of_memory|job\s*state\s*[:=]\s*"
            r"out_of_memory|exceeded\s+(?:step\s+)?memory\s+limit|memory\s+"
            r"cgroup\s+out\s+of\s+memory|due\s+to\s+memory\s+limit|"
            r"term_memlimit|(?:vmem|mem|memory)\s+.*exceeded\s+limit|"
            r"h_vmem\s+limit\s+exceeded|job\s+killed:.*(?:vmem|mem)|"
            r"maxrss\s+.*(?:exceeded|over|above)|"
            r"exit\s*(?:code|status|code)?\s*[:=]?\s*137|"
            r"exitcode\s*[:=]?\s*137|"
            r"return\s*code\s*[:=]?\s*(?:137|-9)|"
            r"returncode\s*[:=]?\s*(?:137|-9))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dependency",
        re.compile(
            r"\b(modulenotfounderror|importerror|no\s+module\s+named|cannot\s+import|"
            r"missing\s+(?:optional\s+)?dependenc|package\s+not\s+installed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "backend",
        re.compile(
            r"\b(neuralfoil|aerosandbox|cantera|xfoil|backend\s+(?:error|failure)|"
            r"solver\s+backend)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "numerical",
        re.compile(
            r"\b(floatingpointerror|linalgerror|singular|non[-\s]?finite|nan|inf|"
            r"overflow|underflow|ill[-\s]?conditioned|svd\s+did\s+not\s+converge|"
            r"invalid\s+value\s+encountered|convergence\s+(?:failed|error))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "worker_crash",
        re.compile(
            r"\b(terminatedworkererror|brokenprocesspool|worker\s+(?:stopped|crashed|"
            r"terminated|exited|killed)|process\s+(?:crashed|terminated|killed)|"
            r"segmentation\s+fault|bus\s+error|illegal\s+instruction|sigsegv|"
            r"sigbus|sigill|sigabrt|sigkill|signal\s+(?:9|11|15)|"
            r"killed\s+by\s+signal\s+(?:9|11|15)|core\s+dumped|"
            r"node[_\s-]?fail|"
            r"node\s+failure|node\s+not\s+responding|exit\s+(?:code|status)\s+"
            r"(?:134|135|136|139)|exitcode\s*[:=]\s*0:(?:4|6|7|9|11|15)|"
            r"return\s*code\s*[:=]?\s*-(?:4|6|7|11|15)|"
            r"returncode\s*[:=]?\s*-(?:4|6|7|11|15)|"
            r"signal\s*[:=]\s*(?:killed|terminated)|"
            r"terminated\s+by\s+signal\s+(?:9|15))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "scheduler",
        re.compile(
            r"\b(job\s*state\s*[:=]\s*(?:cancelled|canceled|preempted|"
            r"requeued|suspended|stopped|deadline|boot_fail|launch_failed|"
            r"reconfig_fail)|reason\s*[:=]\s*(?:preempted|"
            r"dependencyneversatisfied|dependency\s+never\s+satisfied|"
            r"assocgrpwalllimit|qosmaxwalllimit|qosgrpwalllimit|"
            r"assocgrpcpulimit|qosmaxcpulimit|qosgrpcpulimit|"
            r"reqnodenotavail|resources|priority|joblaunchfailure)|"
            r"slurmstepd:.*(?:preempted|preemption|cancelled|canceled)|"
            r"\b(?:evicted|diskpressure|memorypressure)\b|"
            r"\b(?:boot_fail|launch_failed|reconfig_fail|deadline)\b|"
            r"batch\s+job\s+submission\s+failed|job\s+launch\s+failure|"
            r"job\s+(?:held|hold)|"
            r"\b(?:scancel|qdel)\b|pbs:.*(?:job\s+cancelled|job\s+canceled)|"
            r"(?:job|step)\s+(?:cancelled|canceled|preempted|preemption|requeued)\b|"
            r"due\s+to\s+preemption)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "storage",
        re.compile(
            r"\b(no\s+space\s+left\s+on\s+device|disk\s+quota\s+exceeded|"
            r"quota\s+exceeded|filesystem\s+quota|read-only\s+file\s+system|"
            r"read\s+only\s+file\s+system|permission\s+denied|oserror:\s*\[errno\s+28\]|"
            r"errno\s+28|errno\s+122|errno\s+116|errno\s+24|enospc|edquot|"
            r"estale|emfile|input/output\s+error|i/o\s+error|"
            r"stale\s+file\s+handle|transport\s+endpoint\s+is\s+not\s+connected|"
            r"remote\s+i/o\s+error|nfs\s+server\s+not\s+responding|"
            r"lustre.*(?:error|ost|mdt|evict)|file\s+system\s+is\s+full|"
            r"database\s+or\s+disk\s+is\s+full|disk\s+full|"
            r"too\s+many\s+open\s+files|file\s+too\s+large|broken\s+pipe|"
            r"ephemeral-storage|nodefs\.available|imagefs\.available|containerfs\.available|"
            r"failed\s+to\s+(?:write|flush|fsync|rename|replace|move)|"
            r"could\s+not\s+(?:write|flush|fsync|rename|replace|move))\b",
            re.IGNORECASE,
        ),
    ),
)


def _canonical_protocol_value(field: str, value: Any) -> int | str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if field in _PROTOCOL_INTEGER_FIELDS:
        if isinstance(value, (bool, np.bool_)):
            return str(value)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value).strip()
        if np.isfinite(numeric) and numeric == np.floor(numeric):
            return int(numeric)
        return str(value).strip()
    return str(value).strip()


def protocol_digest_payload(row: Mapping[str, Any]) -> dict[str, int | str]:
    """Return the canonical row-level protocol payload used for hashing."""
    return {
        field: _canonical_protocol_value(field, row.get(field))
        for field in PROTOCOL_DIGEST_FIELDS
    }


def protocol_digest_for_row(row: Mapping[str, Any]) -> str:
    """Stable SHA-256 digest for the non-transient protocol fields of a row."""
    payload = protocol_digest_payload(row)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def invalid_protocol_digest_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose stored protocol digest does not match their protocol payload."""
    if not PROTOCOL_DIGEST_COLUMNS.issubset(df.columns):
        return pd.Series(True, index=df.index)
    stored = df["protocol_digest"].fillna("").astype(str).str.strip()
    expected = df.apply(lambda row: protocol_digest_for_row(row.to_dict()), axis=1)
    return stored.ne(expected)


def csv_header_contract_errors(
    path: Path,
    *,
    artifact_name: str = "results.csv",
) -> list[str]:
    """Return structural CSV errors that pandas can otherwise mask.

    ``pandas.read_csv`` auto-mangles duplicate headers such as ``status,status``
    into ``status,status.1``.  That is convenient interactively, but unsafe for
    a resumable benchmark archive because a corrupt server CSV can then look
    merely like it has an extra column.  It can also silently reinterpret a
    one-row file with too many fields as if the first field were an index, or
    pad a crash-truncated row with NaNs.  Read the raw CSV first and reject
    ambiguous names or ragged rows before any row-level contract checks run.
    """
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            row_width_errors: list[str] = []
            expected_width = len(header)
            for line_number, row in enumerate(reader, start=2):
                if len(row) != expected_width:
                    row_width_errors.append(
                        f"line {line_number}: expected {expected_width} field(s), "
                        f"observed {len(row)}"
                    )
                    if len(row_width_errors) >= 10:
                        break
    except StopIteration:
        return [f"{artifact_name} is empty: {path}"]
    except Exception as exc:
        return [f"could not parse {artifact_name} header: {type(exc).__name__}: {exc}"]

    normalized = [str(name).strip() for name in header]
    errors: list[str] = []
    blank_positions = [idx + 1 for idx, name in enumerate(normalized) if not name]
    if blank_positions:
        errors.append(
            f"{artifact_name} has blank header column name(s) at 1-based "
            f"position(s): {blank_positions}"
        )

    seen: set[str] = set()
    duplicate_names: list[str] = []
    for name in normalized:
        if not name:
            continue
        if name in seen and name not in duplicate_names:
            duplicate_names.append(name)
        seen.add(name)
    if duplicate_names:
        errors.append(
            f"{artifact_name} has duplicate header column name(s): "
            f"{sorted(duplicate_names)}"
        )
    if row_width_errors:
        suffix = ""
        if len(row_width_errors) >= 10:
            suffix = "; additional ragged rows may exist"
        errors.append(
            f"{artifact_name} has row(s) with a different number of fields than "
            f"the header: {row_width_errors}{suffix}"
        )
    return errors


def classify_failure_category(status: object, error: object = "") -> str:
    """Map a result-row status/error pair to the canonical failure taxonomy."""
    status_text = str(status or "").strip().lower()
    error_text = str(error or "").strip()
    if status_text == "ok":
        return "ok"
    if status_text == "timeout":
        return "timeout"
    if status_text != "failed":
        return "exception"
    for category, pattern in _FAILURE_PATTERNS:
        if pattern.search(error_text):
            return category
    return "exception"


def failure_category_is_consistent(
    status: object,
    error: object,
    failure_category: object,
) -> bool:
    """Return True when ``failure_category`` matches the status/error contract."""
    category = str(failure_category or "").strip()
    if category not in ALLOWED_FAILURE_CATEGORIES:
        return False
    return category == classify_failure_category(status, error)


def runtime_duration_tolerance_s(t_total_s: float) -> float:
    """Tolerance for comparing wall-time traces with ``t_total_s``.

    Cell timestamps are recorded at whole-second resolution while ``t_total_s``
    comes from ``time.perf_counter()``.  The tolerance is therefore intentionally
    broad enough for timestamp rounding and scheduler jitter, but narrow enough
    to catch hand-edited or corrupt rows.
    """
    return max(
        RUNTIME_DURATION_ABS_TOLERANCE_S,
        RUNTIME_DURATION_REL_TOLERANCE * max(float(t_total_s), 1.0),
    )


def runtime_duration_is_consistent(
    t_total_s: float | None,
    started: datetime | None,
    finished: datetime | None,
) -> bool:
    if t_total_s is None or started is None or finished is None:
        return False
    try:
        t_value = float(t_total_s)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(t_value) or t_value < 0:
        return False
    elapsed = (finished - started).total_seconds()
    if not np.isfinite(elapsed) or elapsed < 0:
        return False
    return abs(elapsed - t_value) <= runtime_duration_tolerance_s(t_value)


def invalid_runtime_duration_mask(
    t_total_s: pd.Series,
    started: pd.Series,
    finished: pd.Series,
) -> pd.Series:
    """Rows where valid timestamps disagree materially with ``t_total_s``."""
    duration = pd.to_numeric(t_total_s, errors="coerce")
    elapsed = (finished - started).dt.total_seconds()
    valid_components = (
        duration.notna()
        & np.isfinite(duration)
        & duration.ge(0)
        & started.notna()
        & finished.notna()
        & elapsed.notna()
        & np.isfinite(elapsed)
        & elapsed.ge(0)
    )
    tolerance = duration.map(
        lambda value: (
            runtime_duration_tolerance_s(float(value))
            if pd.notna(value) and np.isfinite(value) and float(value) >= 0
            else np.nan
        )
    )
    return valid_components & (elapsed.sub(duration).abs() > tolerance)


def invalid_runtime_duration_entry(
    row_number: int,
    row: dict[str, Any],
    *,
    t_total_s: float,
    started: datetime,
    finished: datetime,
) -> dict[str, Any]:
    elapsed = (finished - started).total_seconds()
    return {
        "row_number": int(row_number),
        "case_id": row.get("case_id"),
        "method": row.get("method"),
        "budget_mult": row.get("budget_mult"),
        "repeat": row.get("repeat"),
        "field": "t_total_s",
        "value": row.get("t_total_s"),
        "elapsed_s": round(float(elapsed), 6),
        "abs_delta_s": round(abs(float(elapsed) - float(t_total_s)), 6),
        "tolerance_s": round(runtime_duration_tolerance_s(float(t_total_s)), 6),
    }


def strict_integer_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse integer-valued CSV fields while rejecting boolean masquerades.

    ``pandas.to_numeric`` treats boolean arrays as 0/1.  That is convenient for
    analysis notebooks, but unsafe for protocol fields such as ``repeat``,
    ``budget_mult``, ``worker_pid``, and seeds.  A manually edited or corrupted
    CSV containing ``True`` must not be accepted as repeat 1 or PID 1.
    """
    raw = series
    text = raw.fillna("").astype(str).str.strip()
    bool_like = (
        raw.map(lambda value: isinstance(value, (bool, np.bool_)))
        | text.str.lower().isin({"true", "false"})
    )
    numeric = pd.to_numeric(raw, errors="coerce")
    is_finite = numeric.notna() & np.isfinite(numeric)
    is_integer = is_finite & numeric.eq(np.floor(numeric)) & ~bool_like
    parsed = pd.Series(pd.NA, index=series.index, dtype="Int64")
    parsed.loc[is_integer] = numeric.loc[is_integer].astype(int).to_numpy()
    return parsed, is_integer
