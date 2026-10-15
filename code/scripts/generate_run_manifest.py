"""
scripts/generate_run_manifest.py
=================================
Emit ``RUN_MANIFEST.json``: an immutable, machine-readable snapshot of a
DOE-Benchmark run.  This is the §0.5 step 5 / GAP-13 artefact and the
single source of truth referenced by:

* Stage 10 reproducibility package
  (``docs/working-notes/DOE-Benchmark实验设计方案.md``)
* Zenodo / paper supplementary archives
* ``scripts/check_submission_ready.py`` audit pass

The manifest captures three layers of provenance:

1. **Code** — git commit, branch, dirty status, package versions for the
   numerically-relevant libraries (numpy/scipy/sklearn/torch/...).
2. **Machine** — platform, CPU count, RAM, BLAS thread-cap env vars.
3. **Experiment** — the frozen pre-registration knobs (pool, budgets,
   n_repeats, n_test, seed strategy, case list) plus a derived summary
   of the actual ``results.csv`` (n_rows, status counts, observed
   methods/cases/budgets) and a SHA-256 manifest of every file in
   ``--results-dir`` so reviewers can verify nothing was edited
   post-submission.

The script is **read-only** with respect to the experiment tree; it
writes a single JSON file (default: ``<results-dir>/RUN_MANIFEST.json``)
and never mutates ``results.csv`` or its siblings.

Usage
-----
    python scripts/generate_run_manifest.py \
        --results-dir benchmark/results/full_benchmark \
        --pool nature --budgets 5 10 20 50 \
        --n-repeats 30 --n-test 1000 \
        --seed-strategy case_repeat --seed-start 1000 \
        --output RUN_MANIFEST.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIP_FREEZE_ARTIFACT_NAME = "python_environment_pip_freeze.txt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.runtime_contracts import (  # noqa: E402
    ALLOWED_FAILURE_CATEGORIES,
    classify_failure_category,
    csv_header_contract_errors,
    invalid_runtime_duration_entry,
    runtime_duration_is_consistent,
)
from benchmark.core import _torch_reproducibility_snapshot  # noqa: E402

PACKAGE_NAMES = [
    "doe-testbed",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "torch",
    "botorch",
    "gpytorch",
]

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
]

TEMP_STORAGE_ENV_VARS = [
    "TMPDIR",
    "TMP",
    "TEMP",
    "TEMPDIR",
    "SLURM_TMPDIR",
    "SCRATCH",
    "LOCAL_SCRATCH",
    "JOBSCRATCH",
]

SCHEDULER_ENV_VARS = [
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_GPUS",
    "SLURM_SUBMIT_DIR",
    "PBS_JOBID",
    "PBS_JOBNAME",
    "PBS_QUEUE",
    "PBS_NODEFILE",
    "LSB_JOBID",
    "LSB_QUEUE",
    "LSB_HOSTS",
    "SGE_JOB_ID",
    "JOB_ID",
    "NSLOTS",
]

CONTAINER_ENV_VARS = [
    "APPTAINER_CONTAINER",
    "SINGULARITY_CONTAINER",
    "CONTAINER_IMAGE",
    "DOCKER_IMAGE",
    "KUBERNETES_SERVICE_HOST",
]

DERIVED_DIRTY_PREFIXES = (
    "benchmark/results/",
    "paper/figdata/",
    "paper/figures/",
)
DERIVED_DIRTY_FILES = {
    "docs/CASE_VALIDATION_SUMMARY.md",
    "docs/case_validation_note_sources.csv",
    "paper/figure_manifest.json",
    "paper/numbers.json",
    "paper/numbers.tex",
}


@dataclass(frozen=True)
class CaseResolution:
    case_ids: list[str]
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate RUN_MANIFEST.json for a DOE-Benchmark experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=ROOT / "benchmark" / "results" / "full_benchmark")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--pool", default="nature")
    p.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--n-repeats", type=int, default=30)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--seed-strategy", default="case_repeat")
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--cases", nargs="+", default=["ALL"])
    p.add_argument("--command", action="append", default=[])
    p.add_argument("--command-file", type=Path, action="append", default=[])
    p.add_argument(
        "--require-complete-archive",
        action="store_true",
        help=(
            "Fail before writing RUN_MANIFEST.json unless the archive is a "
            "clean, complete paper-run snapshot."
        ),
    )
    return p.parse_args(argv)


def _run_git(args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def _porcelain_path(line: str) -> str:
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1].strip()
    return path.strip('"').replace("\\", "/")


def _is_allowed_derived_dirty(line: str) -> bool:
    path = _porcelain_path(line)
    return path in DERIVED_DIRTY_FILES or any(
        path.startswith(prefix) for prefix in DERIVED_DIRTY_PREFIXES
    )


def _git_info() -> dict[str, Any]:
    status = _run_git(["status", "--porcelain"])
    status_lines = status.splitlines() if status else []
    derived_dirty_entries = [
        line for line in status_lines if _is_allowed_derived_dirty(line)
    ]
    source_dirty_entries = [
        line for line in status_lines if not _is_allowed_derived_dirty(line)
    ]
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
        "source_dirty": bool(source_dirty_entries),
        "source_dirty_entries": source_dirty_entries,
        "derived_dirty_entries": derived_dirty_entries,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _write_text_atomic(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_parent = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp.", dir=str(path.parent)))
    tmp_output = tmp_parent / path.name
    try:
        tmp_output.write_text(text, encoding="utf-8")
        shutil.move(str(tmp_output), str(path))
    finally:
        if tmp_parent.exists():
            shutil.rmtree(tmp_parent)


def _python_environment_info(output_path: Path | None = None) -> dict[str, Any]:
    """Record and optionally archive the active Python environment lock."""
    command = [sys.executable, "-m", "pip", "freeze", "--all"]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "command": shlex.join(command),
            "error": f"{type(exc).__name__}: {exc}",
        }

    stdout = completed.stdout.replace("\r\n", "\n")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    out: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok" if completed.returncode == 0 else "error",
        "command": shlex.join(command),
        "returncode": int(completed.returncode),
        "package_count": len(lines),
        "pip_freeze_sha256": digest,
    }
    if completed.returncode == 0 and output_path is not None:
        normalized_text = "\n".join(lines) + "\n"
        _write_text_atomic(output_path, normalized_text)
        out["pip_freeze_path"] = output_path.name
    if completed.returncode != 0:
        out["stderr"] = completed.stderr[-2000:]
    return out


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 3)
    except Exception:
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return round(kb / 1024**2, 3)
    return None


def _cgroup_memory_limit_candidates(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_self_cgroup: Path = Path("/proc/self/cgroup"),
) -> list[tuple[Path, str]]:
    """Return likely current-process cgroup memory-limit files."""
    candidates: list[tuple[Path, str]] = []

    def add(path: Path, source: str) -> None:
        key = path.as_posix()
        if not any(existing.as_posix() == key for existing, _ in candidates):
            candidates.append((path, source))

    try:
        lines = proc_self_cgroup.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        lines = []

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _hierarchy, controllers, rel = parts
        rel_parts = [part for part in rel.strip().split("/") if part]
        rel_paths = [Path(*rel_parts[:idx]) if idx else Path(".") for idx in range(len(rel_parts), -1, -1)]
        if controllers == "":
            for rel_path in rel_paths:
                add(cgroup_root / rel_path / "memory.max", "cgroup v2 memory.max")
        elif "memory" in {item.strip() for item in controllers.split(",")}:
            for rel_path in rel_paths:
                add(
                    cgroup_root / "memory" / rel_path / "memory.limit_in_bytes",
                    "cgroup v1 memory.limit_in_bytes",
                )
                add(
                    cgroup_root / rel_path / "memory.limit_in_bytes",
                    "cgroup v1 memory.limit_in_bytes",
                )

    add(cgroup_root / "memory.max", "cgroup v2 memory.max")
    add(cgroup_root / "memory" / "memory.limit_in_bytes", "cgroup v1 memory.limit_in_bytes")
    return candidates


def _cgroup_memory_limit_gb(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_self_cgroup: Path = Path("/proc/self/cgroup"),
) -> dict[str, Any]:
    """Capture Linux cgroup memory limits for scheduler/container provenance."""
    unlimited_seen: dict[str, Any] | None = None
    non_integer_seen: dict[str, Any] | None = None
    finite_limits: list[dict[str, Any]] = []
    for path, source in _cgroup_memory_limit_candidates(cgroup_root, proc_self_cgroup):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw:
            continue
        if raw.lower() == "max":
            unlimited_seen = {
                "schema_version": 1,
                "detected": True,
                "source": source,
                "path": path.as_posix(),
                "raw": raw,
                "limit_gb": None,
                "unlimited": True,
            }
            continue
        try:
            value = int(raw)
        except ValueError:
            non_integer_seen = {
                "schema_version": 1,
                "detected": True,
                "source": source,
                "path": path.as_posix(),
                "raw": raw,
                "limit_gb": None,
                "unlimited": None,
                "error": "non-integer cgroup memory limit",
            }
            continue
        unlimited = value <= 0 or value >= 2**60
        if unlimited:
            unlimited_seen = {
                "schema_version": 1,
                "detected": True,
                "source": source,
                "path": path.as_posix(),
                "raw": raw,
                "limit_gb": None,
                "unlimited": True,
            }
        else:
            finite_limits.append({
                "schema_version": 1,
                "detected": True,
                "source": source,
                "path": path.as_posix(),
                "raw": raw,
                "limit_gb": round(value / 1024**3, 3),
                "unlimited": False,
            })
    if finite_limits:
        return min(finite_limits, key=lambda item: float(item["limit_gb"]))
    if non_integer_seen is not None:
        return non_integer_seen
    if unlimited_seen is not None:
        return unlimited_seen
    return {
        "schema_version": 1,
        "detected": False,
        "source": None,
        "path": None,
        "raw": None,
        "limit_gb": None,
        "unlimited": None,
    }


def _effective_ram_gb(total_ram_gb: float | None, cgroup: dict[str, Any]) -> tuple[float | None, str]:
    cgroup_limit = cgroup.get("limit_gb") if isinstance(cgroup, dict) else None
    if isinstance(cgroup_limit, (int, float)) and cgroup_limit > 0:
        if total_ram_gb is None or float(cgroup_limit) < float(total_ram_gb):
            return round(float(cgroup_limit), 3), str(cgroup.get("source") or "cgroup")
    if total_ram_gb is not None:
        return round(float(total_ram_gb), 3), "physical RAM"
    return None, "unknown"


def _resource_limit_info() -> dict[str, Any]:
    """Record process ulimit values that can affect long benchmark runs."""
    try:
        import resource
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "limits": {},
        }

    names = [
        "RLIMIT_AS",
        "RLIMIT_DATA",
        "RLIMIT_NOFILE",
        "RLIMIT_NPROC",
        "RLIMIT_STACK",
        "RLIMIT_CPU",
    ]
    limits: dict[str, dict[str, int | str | None]] = {}
    for name in names:
        value = getattr(resource, name, None)
        if value is None:
            limits[name] = {"soft": None, "hard": None, "status": "not_supported"}
            continue
        try:
            soft, hard = resource.getrlimit(value)
        except Exception as exc:
            limits[name] = {
                "soft": None,
                "hard": None,
                "status": f"error: {type(exc).__name__}: {exc}",
            }
            continue

        def _format_limit(item: int) -> int | str:
            return "unlimited" if item == resource.RLIM_INFINITY else int(item)

        limits[name] = {
            "soft": _format_limit(soft),
            "hard": _format_limit(hard),
            "status": "ok",
        }
    return {
        "schema_version": 1,
        "status": "ok",
        "limits": limits,
    }


def _temp_storage_info() -> dict[str, Any]:
    """Record temporary/scratch storage context for crash and ENOSPC forensics."""
    env = {key: os.environ.get(key) for key in TEMP_STORAGE_ENV_VARS}
    try:
        resolved = Path(tempfile.gettempdir()).resolve()
        resolved_path = resolved.as_posix()
    except Exception as exc:
        return {
            "schema_version": 1,
            "resolved_path": None,
            "exists": False,
            "writable": False,
            "free_gb": None,
            "env": env,
            "error": f"{type(exc).__name__}: {exc}",
        }

    def probe_writable(directory: Path) -> bool:
        probe_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(directory),
                prefix=".doe_benchmark_manifest_write_probe_",
                suffix=".tmp",
                delete=False,
            ) as handle:
                probe_path = Path(handle.name)
                handle.write("probe\n")
                handle.flush()
                os.fsync(handle.fileno())
            return True
        except OSError:
            return False
        finally:
            if probe_path is not None:
                try:
                    probe_path.unlink(missing_ok=True)
                except OSError:
                    pass

    exists = resolved.exists()
    writable = bool(exists and probe_writable(resolved))
    free_gb: float | None = None
    if exists:
        try:
            free_gb = round(shutil.disk_usage(resolved).free / 1024**3, 3)
        except Exception:
            free_gb = None
    return {
        "schema_version": 1,
        "resolved_path": resolved_path,
        "exists": bool(exists),
        "writable": bool(writable),
        "free_gb": free_gb,
        "env": env,
    }


def _machine_info() -> dict[str, Any]:
    physical_ram_gb = _ram_gb()
    cgroup = _cgroup_memory_limit_gb()
    effective_ram_gb, effective_ram_source = _effective_ram_gb(physical_ram_gb, cgroup)
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "ram_gb": physical_ram_gb,
        "effective_ram_gb": effective_ram_gb,
        "effective_ram_source": effective_ram_source,
        "cgroup_memory": cgroup,
        "resource_limits": _resource_limit_info(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "temp_storage": _temp_storage_info(),
    }


def _torch_reproducibility_errors(
    record: object,
    *,
    context: str = "torch_reproducibility",
) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"{context} section is missing or malformed"]
    if record.get("schema_version") != 1:
        errors.append(f"{context}.schema_version must be 1")
    status = str(record.get("status", "")).strip()
    if status not in {"ok", "missing"}:
        errors.append(f"{context}.status must be ok/missing")
    policy = record.get("policy")
    if not isinstance(policy, dict):
        errors.append(f"{context}.policy is missing or malformed")
    else:
        expected = {
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
        for key, value in expected.items():
            if policy.get(key) != value:
                errors.append(f"{context}.policy.{key} must be {value!r}")
        if not isinstance(policy.get("cublas_workspace_config"), str) or not policy.get(
            "cublas_workspace_config", ""
        ).strip():
            errors.append(f"{context}.policy.cublas_workspace_config is missing")
    if status == "ok":
        state = record.get("state")
        if not isinstance(state, dict):
            errors.append(f"{context}.state is missing or malformed")
        else:
            if state.get("deterministic_algorithms") is not True:
                errors.append(f"{context}.state.deterministic_algorithms must be true")
            if state.get("cudnn_benchmark") is True:
                errors.append(f"{context}.state.cudnn_benchmark must not be true")
            if state.get("cuda_matmul_allow_tf32") is True:
                errors.append(f"{context}.state.cuda_matmul_allow_tf32 must not be true")
            if state.get("cudnn_allow_tf32") is True:
                errors.append(f"{context}.state.cudnn_allow_tf32 must not be true")
    return errors


def _scheduler_info() -> dict[str, Any]:
    """Record common HPC scheduler environment variables for auditability."""
    env = {key: os.environ.get(key) for key in SCHEDULER_ENV_VARS}
    detected = [key for key, value in env.items() if value not in (None, "")]
    if any(key.startswith("SLURM_") for key in detected):
        name = "slurm"
    elif any(key.startswith("PBS_") for key in detected):
        name = "pbs"
    elif any(key.startswith("LSB_") for key in detected):
        name = "lsf"
    elif any(key in {"SGE_JOB_ID", "JOB_ID", "NSLOTS"} for key in detected):
        name = "sge"
    else:
        name = "none"
    return {
        "schema_version": 1,
        "name": name,
        "detected": bool(detected),
        "env": env,
    }


def _run_command_for_json(command: list[str], *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"status": "not_found", "command": shlex.join(command)}
    except Exception as exc:
        return {
            "status": "error",
            "command": shlex.join(command),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "command": shlex.join(command),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip()[-4000:],
        "stderr": completed.stderr.strip()[-2000:],
    }


def _accelerator_info() -> dict[str, Any]:
    """Capture GPU/CUDA availability without requiring a GPU on every host."""
    info: dict[str, Any] = {
        "schema_version": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _run_command_for_json(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            timeout=10.0,
        ),
        "torch_cuda": {"status": "not_checked"},
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        devices: list[dict[str, Any]] = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 3),
                    "capability": list(props.major_minor)
                    if hasattr(props, "major_minor")
                    else [props.major, props.minor],
                })
        info["torch_cuda"] = {
            "status": "ok",
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": cuda_available,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "device_count": len(devices),
            "devices": devices,
        }
    except Exception as exc:
        info["torch_cuda"] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return info


def _container_info() -> dict[str, Any]:
    env = {key: os.environ.get(key) for key in CONTAINER_ENV_VARS}
    marker_files = {
        "/.dockerenv": Path("/.dockerenv").exists(),
        "/run/.containerenv": Path("/run/.containerenv").exists(),
    }
    detected = any(value not in (None, "") for value in env.values()) or any(marker_files.values())
    return {
        "schema_version": 1,
        "detected": bool(detected),
        "env": env,
        "marker_files": marker_files,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_manifest(root: Path, exclude: Path | None = None) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    exclude_resolved = exclude.resolve() if exclude is not None else None
    files = []
    candidates = sorted(
        (p for p in root.rglob("*") if p.is_file() or p.is_symlink()),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    for path in candidates:
        if exclude_resolved is not None and path.resolve() == exclude_resolved:
            continue
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            try:
                target = os.readlink(path)
            except OSError as exc:
                target = f"<unreadable: {type(exc).__name__}: {exc}>"
            files.append({
                "path": rel,
                "size_bytes": None,
                "mtime_utc": None,
                "sha256": None,
                "is_symlink": True,
                "symlink_target": target,
            })
            continue
        stat = path.stat()
        files.append({
            "path": rel,
            "size_bytes": int(stat.st_size),
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": _sha256(path),
            "is_symlink": False,
        })
    return files


def _run_window_from_files(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer the archived run window from artifact modification times."""
    mtimes = [
        item.get("mtime_utc")
        for item in files
        if isinstance(item, dict) and isinstance(item.get("mtime_utc"), str)
    ]
    if not mtimes:
        return {
            "source": "artifact_mtime_utc",
            "started_at_utc": None,
            "finished_at_utc": None,
            "n_artifacts": 0,
        }
    return {
        "source": "artifact_mtime_utc",
        "scope": (
            "batch that produced the archived artefacts; this is not necessarily "
            "the span of results.csv, which see dataset_window"
        ),
        "started_at_utc": min(mtimes),
        "finished_at_utc": max(mtimes),
        "n_artifacts": len(mtimes),
    }


#: A gap this long between consecutive cell start times marks a separate sweep
#: batch.  The archived grid was produced in more than one batch (the
#: high-dimensional "slow" cases were swept separately), so ``run_window``
#: alone does not describe the span of the row-level data.
DATASET_BATCH_GAP_HOURS = 6.0


def _dataset_window_from_results(results_dir: Path) -> dict[str, Any]:
    """Span and batch split of the row-level results actually archived.

    ``run_window`` is derived from artifact modification times and therefore
    records only the batch that produced the archived artefacts.  Readers who
    reconcile the manifest against ``results.csv`` need the true span of the
    data, so this section reports the batch boundaries derived from the cell
    timestamps themselves.
    """
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        return {"status": "missing", "source": "results.csv cell timestamps"}
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - pandas is a hard dependency
        return {"status": "unavailable", "error": f"pandas import failed: {exc}"}
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["case_id", "cell_started_utc", "cell_finished_utc"],
            low_memory=False,
        )
    except Exception as exc:
        return {"status": "unreadable", "error": str(exc)}

    started = pd.to_datetime(df["cell_started_utc"], utc=True, errors="coerce")
    finished = pd.to_datetime(df["cell_finished_utc"], utc=True, errors="coerce")
    if not started.notna().any():
        return {"status": "no_timestamps", "source": "results.csv cell timestamps"}

    order = started[started.notna()].sort_values()
    stamps = list(order)
    gap = pd.Timedelta(hours=DATASET_BATCH_GAP_HOURS)
    bounds = [(0, 1)]
    for i in range(1, len(stamps)):
        if stamps[i] - stamps[i - 1] > gap:
            bounds.append((i, i + 1))
        else:
            bounds[-1] = (bounds[-1][0], i + 1)

    batches: list[dict[str, Any]] = []
    for start_idx, end_idx in bounds:
        labels = order.index[start_idx:end_idx]
        segment_finish = finished.loc[labels].max()
        batches.append({
            "started_at_utc": stamps[start_idx].isoformat(),
            "finished_at_utc": (
                segment_finish.isoformat() if pd.notna(segment_finish) else None
            ),
            "n_rows": int(len(labels)),
            "n_cases": int(df.loc[labels, "case_id"].nunique()),
        })

    overall_finish = finished.max()
    return {
        "status": "ok",
        "source": "results.csv cell_started_utc and cell_finished_utc",
        "batch_gap_hours": DATASET_BATCH_GAP_HOURS,
        "n_rows": int(len(df)),
        "n_cases": int(df["case_id"].nunique()),
        "started_at_utc": stamps[0].isoformat(),
        "finished_at_utc": (
            overall_finish.isoformat() if pd.notna(overall_finish) else None
        ),
        "n_batches": len(batches),
        "batches": batches,
    }


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _results_summary(results_csv: Path) -> dict[str, Any]:
    if not results_csv.exists():
        return {"exists": False}
    csv_contract_errors = csv_header_contract_errors(results_csv)
    n_rows = 0
    case_ids: set[str] = set()
    methods: set[str] = set()
    budgets: set[int] = set()
    repeats: set[int] = set()
    invalid_budgets: list[dict[str, Any]] = []
    invalid_repeats: list[dict[str, Any]] = []
    missing_protocol_keys: list[dict[str, Any]] = []
    statuses: dict[str, int] = {}
    failure_categories: dict[str, int] = {}
    invalid_statuses: list[dict[str, Any]] = []
    missing_failure_categories: list[dict[str, Any]] = []
    invalid_failure_categories: list[dict[str, Any]] = []
    failure_category_status_mismatches: list[dict[str, Any]] = []
    invalid_ok_nrmse: list[dict[str, Any]] = []
    invalid_t_total_s: list[dict[str, Any]] = []
    invalid_runtime_timestamps: list[dict[str, Any]] = []
    invalid_runtime_time_order: list[dict[str, Any]] = []
    invalid_runtime_duration: list[dict[str, Any]] = []
    invalid_rss_values: list[dict[str, Any]] = []
    invalid_rss_peak_order: list[dict[str, Any]] = []
    missing_failure_errors: list[dict[str, Any]] = []
    ok_rows_with_errors: list[dict[str, Any]] = []
    rss_peak_values: list[float] = []
    rss_peak_by_method: dict[str, list[float]] = {}
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        timestamp_fields = ("cell_started_utc", "cell_finished_utc")
        missing_timestamp_columns = [
            field for field in timestamp_fields if field not in columns
        ]
        timestamp_columns_present = not missing_timestamp_columns
        rss_fields = ("rss_gb_start", "rss_gb_end", "rss_gb_peak")
        missing_rss_columns = [field for field in rss_fields if field not in columns]
        rss_columns_present = not missing_rss_columns
        for row in reader:
            n_rows += 1
            for field in ("case_id", "method", "budget_mult", "repeat"):
                if not str(row.get(field) or "").strip():
                    missing_protocol_keys.append(_row_contract_entry(n_rows, row, field))
            if str(row.get("case_id") or "").strip():
                case_ids.add(str(row["case_id"]).strip())
            if str(row.get("method") or "").strip():
                methods.add(str(row["method"]).strip())
            if str(row.get("budget_mult") or "").strip():
                parsed = _parse_int_field(row["budget_mult"])
                if parsed is None:
                    invalid_budgets.append(_invalid_numeric_entry(n_rows, row, "budget_mult"))
                else:
                    budgets.add(parsed)
            if str(row.get("repeat") or "").strip():
                parsed = _parse_int_field(row["repeat"])
                if parsed is None:
                    invalid_repeats.append(_invalid_numeric_entry(n_rows, row, "repeat"))
                else:
                    repeats.add(parsed)
            status = row.get("status") or "unknown"
            statuses[status] = statuses.get(status, 0) + 1
            if status not in {"ok", "timeout", "failed"}:
                invalid_statuses.append(_row_contract_entry(n_rows, row, "status"))
            error_text = str(row.get("error") or "").strip()
            failure_category = str(row.get("failure_category") or "").strip()
            failure_categories[failure_category or "unknown"] = (
                failure_categories.get(failure_category or "unknown", 0) + 1
            )
            if not failure_category:
                missing_failure_categories.append(
                    _row_contract_entry(n_rows, row, "failure_category")
                )
            elif failure_category not in ALLOWED_FAILURE_CATEGORIES:
                invalid_failure_categories.append(
                    _row_contract_entry(n_rows, row, "failure_category")
                )
            elif failure_category != classify_failure_category(status, error_text):
                failure_category_status_mismatches.append(
                    _row_contract_entry(n_rows, row, "failure_category")
                    | {"expected": classify_failure_category(status, error_text)}
                )
            if status == "ok":
                nrmse = _parse_float_field(row.get("nrmse"))
                if nrmse is None or nrmse < 0:
                    invalid_ok_nrmse.append(_row_contract_entry(n_rows, row, "nrmse"))
                if error_text:
                    ok_rows_with_errors.append(_row_contract_entry(n_rows, row, "error"))
            elif status in {"timeout", "failed"} and not error_text:
                missing_failure_errors.append(_row_contract_entry(n_rows, row, "error"))
            t_total_s = _parse_float_field(row.get("t_total_s"))
            if t_total_s is None or t_total_s < 0:
                invalid_t_total_s.append(_row_contract_entry(n_rows, row, "t_total_s"))
            if timestamp_columns_present:
                started = _parse_iso_datetime(row.get("cell_started_utc"))
                finished = _parse_iso_datetime(row.get("cell_finished_utc"))
                if started is None:
                    invalid_runtime_timestamps.append(
                        _row_contract_entry(n_rows, row, "cell_started_utc")
                    )
                if finished is None:
                    invalid_runtime_timestamps.append(
                        _row_contract_entry(n_rows, row, "cell_finished_utc")
                    )
                if started is not None and finished is not None and finished < started:
                    invalid_runtime_time_order.append(
                        _row_contract_entry(n_rows, row, "cell_finished_utc")
                    )
                if (
                    t_total_s is not None
                    and started is not None
                    and finished is not None
                    and finished >= started
                    and not runtime_duration_is_consistent(t_total_s, started, finished)
                ):
                    invalid_runtime_duration.append(
                        invalid_runtime_duration_entry(
                            n_rows,
                            row,
                            t_total_s=t_total_s,
                            started=started,
                            finished=finished,
                        )
                    )
            if rss_columns_present:
                rss_values: dict[str, float] = {}
                for field in rss_fields:
                    parsed = _parse_float_field(row.get(field))
                    if parsed is None or parsed < 0:
                        invalid_rss_values.append(_row_contract_entry(n_rows, row, field))
                    else:
                        rss_values[field] = parsed
                peak = rss_values.get("rss_gb_peak")
                if peak is not None:
                    rss_peak_values.append(peak)
                    method = str(row.get("method") or "").strip()
                    if method:
                        rss_peak_by_method.setdefault(method, []).append(peak)
                if set(rss_fields) <= set(rss_values) and (
                    rss_values["rss_gb_peak"] < rss_values["rss_gb_start"]
                    or rss_values["rss_gb_peak"] < rss_values["rss_gb_end"]
                ):
                    invalid_rss_peak_order.append(
                        _row_contract_entry(n_rows, row, "rss_gb_peak")
                    )
    return {
        "exists": True,
        "path": results_csv.as_posix(),
        "csv_contract_errors": csv_contract_errors[:30],
        "n_csv_contract_errors": len(csv_contract_errors),
        "columns": columns,
        "n_rows": n_rows,
        "n_cases_observed": len(case_ids),
        "case_ids_observed": sorted(case_ids),
        "n_methods_observed": len(methods),
        "methods_observed": sorted(methods),
        "budgets_observed": sorted(budgets),
        "repeats_observed": sorted(repeats),
        "invalid_budget_mult_values": invalid_budgets[:30],
        "n_invalid_budget_mult_values": len(invalid_budgets),
        "invalid_repeat_values": invalid_repeats[:30],
        "n_invalid_repeat_values": len(invalid_repeats),
        "missing_protocol_key_values": missing_protocol_keys[:30],
        "n_missing_protocol_key_values": len(missing_protocol_keys),
        "status_counts": dict(sorted(statuses.items())),
        "invalid_status_values": invalid_statuses[:30],
        "n_invalid_status_values": len(invalid_statuses),
        "failure_category_counts": dict(sorted(failure_categories.items())),
        "missing_failure_category_values": missing_failure_categories[:30],
        "n_missing_failure_category_values": len(missing_failure_categories),
        "invalid_failure_category_values": invalid_failure_categories[:30],
        "n_invalid_failure_category_values": len(invalid_failure_categories),
        "failure_category_status_mismatch_values": failure_category_status_mismatches[:30],
        "n_failure_category_status_mismatch_values": len(failure_category_status_mismatches),
        "invalid_ok_nrmse_values": invalid_ok_nrmse[:30],
        "n_invalid_ok_nrmse_values": len(invalid_ok_nrmse),
        "invalid_t_total_s_values": invalid_t_total_s[:30],
        "n_invalid_t_total_s_values": len(invalid_t_total_s),
        "invalid_runtime_timestamp_values": invalid_runtime_timestamps[:30],
        "n_invalid_runtime_timestamp_values": len(invalid_runtime_timestamps),
        "invalid_runtime_time_order_values": invalid_runtime_time_order[:30],
        "n_invalid_runtime_time_order_values": len(invalid_runtime_time_order),
        "invalid_runtime_duration_values": invalid_runtime_duration[:30],
        "n_invalid_runtime_duration_values": len(invalid_runtime_duration),
        "runtime_timestamp_summary": _runtime_timestamp_summary(
            n_rows=n_rows,
            timestamp_columns_present=timestamp_columns_present,
            missing_timestamp_columns=missing_timestamp_columns,
            invalid_runtime_timestamps=invalid_runtime_timestamps,
            invalid_runtime_time_order=invalid_runtime_time_order,
            invalid_runtime_duration=invalid_runtime_duration,
        ),
        "invalid_rss_gb_values": invalid_rss_values[:30],
        "n_invalid_rss_gb_values": len(invalid_rss_values),
        "invalid_rss_gb_peak_order_values": invalid_rss_peak_order[:30],
        "n_invalid_rss_gb_peak_order_values": len(invalid_rss_peak_order),
        "runtime_resource_summary": _runtime_resource_summary(
            n_rows=n_rows,
            rss_columns_present=rss_columns_present,
            missing_rss_columns=missing_rss_columns,
            rss_peak_values=rss_peak_values,
            rss_peak_by_method=rss_peak_by_method,
        ),
        "missing_failure_error_values": missing_failure_errors[:30],
        "n_missing_failure_error_values": len(missing_failure_errors),
        "ok_rows_with_error_values": ok_rows_with_errors[:30],
        "n_ok_rows_with_error_values": len(ok_rows_with_errors),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _runtime_resource_summary(
    *,
    n_rows: int,
    rss_columns_present: bool,
    missing_rss_columns: list[str],
    rss_peak_values: list[float],
    rss_peak_by_method: dict[str, list[float]],
) -> dict[str, Any]:
    return {
        "rss_trace_columns_present": bool(rss_columns_present),
        "missing_rss_trace_columns": sorted(missing_rss_columns),
        "n_rows": int(n_rows),
        "n_rows_with_valid_rss_gb_peak": len(rss_peak_values),
        "rss_gb_peak_min": _round_optional(min(rss_peak_values) if rss_peak_values else None),
        "rss_gb_peak_p50": _round_optional(_percentile(rss_peak_values, 0.50)),
        "rss_gb_peak_p95": _round_optional(_percentile(rss_peak_values, 0.95)),
        "rss_gb_peak_max": _round_optional(max(rss_peak_values) if rss_peak_values else None),
        "rss_gb_peak_max_by_method": {
            method: _round_optional(max(values))
            for method, values in sorted(rss_peak_by_method.items())
            if values
        },
    }


def _runtime_timestamp_summary(
    *,
    n_rows: int,
    timestamp_columns_present: bool,
    missing_timestamp_columns: list[str],
    invalid_runtime_timestamps: list[dict[str, Any]],
    invalid_runtime_time_order: list[dict[str, Any]],
    invalid_runtime_duration: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "timestamp_trace_columns_present": bool(timestamp_columns_present),
        "missing_timestamp_trace_columns": sorted(missing_timestamp_columns),
        "n_rows": int(n_rows),
        "n_invalid_runtime_timestamp_values": len(invalid_runtime_timestamps),
        "n_invalid_runtime_time_order_values": len(invalid_runtime_time_order),
        "n_invalid_runtime_duration_values": len(invalid_runtime_duration),
    }


def _parse_int_field(value: str) -> int | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not parsed.is_integer():
        return None
    return int(parsed)


def _parse_float_field(value: str | None) -> float | None:
    try:
        parsed = float(value) if value not in (None, "") else float("nan")
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _invalid_numeric_entry(row_number: int, row: dict[str, str], field: str) -> dict[str, Any]:
    return {
        "row": row_number,
        "field": field,
        "value": row.get(field),
        "case_id": row.get("case_id"),
        "method": row.get("method"),
        "budget_mult": row.get("budget_mult"),
        "repeat": row.get("repeat"),
    }


def _row_contract_entry(row_number: int, row: dict[str, str], field: str) -> dict[str, Any]:
    return {
        "row": row_number,
        "field": field,
        "value": row.get(field),
        "case_id": row.get("case_id"),
        "method": row.get("method"),
        "budget_mult": row.get("budget_mult"),
        "repeat": row.get("repeat"),
        "status": row.get("status"),
    }


def _summary_nonnegative_int(summary: dict[str, Any], key: str, errors: list[str]) -> int:
    raw = summary.get(key, 0)
    if raw in (None, ""):
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"results_summary.{key} must be a non-negative integer")
        return 0
    if value < 0:
        errors.append(f"results_summary.{key} must be a non-negative integer")
        return 0
    return value


def _int_list(value: Any, label: str, errors: list[str]) -> list[int]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        errors.append(f"{label} must be a list of integers")
        return []
    out: list[int] = []
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a list of integers")
            return []
        out.append(parsed)
    return sorted(out)


def _optional_nonnegative_int(value: Any, label: str, errors: list[str]) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a non-negative integer")
        return None
    if parsed < 0:
        errors.append(f"{label} must be a non-negative integer")
        return None
    return parsed


def _strict_observed_string_set_errors(
    *,
    summary: dict[str, Any],
    declared: set[str],
    declared_count: int | None,
    observed_key: str,
    count_key: str,
    label: str,
    errors: list[str],
) -> set[str]:
    raw = summary.get(observed_key)
    if not isinstance(raw, list):
        errors.append(f"results_summary.{observed_key} is missing or malformed")
        observed: list[str] = []
    else:
        observed = [str(item) for item in raw]
    observed_set = set(observed)
    duplicates = sorted({item for item in observed if observed.count(item) > 1})
    if duplicates:
        errors.append(f"results_summary.{observed_key} contains duplicates: {duplicates}")
    observed_count_raw = summary.get(count_key)
    observed_count = _optional_nonnegative_int(
        observed_count_raw,
        f"results_summary.{count_key}",
        errors,
    )
    if declared_count is not None and observed_count_raw in (None, ""):
        errors.append(f"results_summary.{count_key} is missing or malformed")
    if observed_count is not None:
        if observed_count != len(observed_set):
            errors.append(
                f"results_summary.{count_key} does not match the unique {label} observed set size"
            )
        if declared_count is not None and observed_count != declared_count:
            errors.append(
                f"{label} count drift: declared={declared_count}, observed={observed_count}"
            )
    if declared and declared != {"ALL"}:
        missing = sorted(declared - observed_set)
        extra = sorted(observed_set - declared)
        if missing or extra:
            errors.append(f"{label} set drift: missing={missing}, extra={extra}")
    return observed_set


def _resolve_cases(requested: list[str]) -> CaseResolution:
    if requested != ["ALL"]:
        return CaseResolution(list(requested), None)
    try:
        import doe_testbed as db

        return CaseResolution(sorted(c.case_id for c in db.list_cases()), None)
    except Exception as exc:
        return CaseResolution(["ALL"], f"{type(exc).__name__}: {exc}")


def _resolve_methods(pool: str) -> list[str]:
    try:
        from benchmark.scripts.run_full_benchmark import POOLS

        return [m.name for m in POOLS[pool]()]
    except Exception:
        return []


def _method_source_manifest(pool: str) -> dict[str, Any]:
    """Record composite source fingerprints for methods in the declared pool."""
    entries: list[dict[str, Any]] = []
    try:
        from benchmark.core import method_source_components, method_source_fingerprint
        from benchmark.scripts.run_full_benchmark import POOLS

        methods = POOLS[pool]()
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "methods": entries,
        }

    for method in methods:
        try:
            fingerprint = method_source_fingerprint(method)
            entries.append({
                "method": method.name,
                "family": method.family,
                "status": "ok",
                "class_name": type(method).__name__,
                "module": type(method).__module__,
                "components": method_source_components(method),
                **fingerprint,
            })
        except Exception as exc:
            entries.append({
                "method": getattr(method, "name", "<unknown>"),
                "status": "unresolved",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return {"schema_version": 1, "status": "ok", "methods": entries}


def _portable_path(path: Path) -> str:
    try:
        resolved = path.resolve()
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _command_file_status(args: argparse.Namespace) -> dict[str, list[str]]:
    requested = [_portable_path(path) for path in args.command_file]
    missing = [_portable_path(path) for path in args.command_file if not path.exists()]
    parse_errors = _command_file_parse_errors(args)
    return {"requested": requested, "missing": missing, "parse_errors": parse_errors}


def _command_file_parse_errors(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    for path in args.command_file:
        if not path.exists():
            continue
        display = _portable_path(path)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if "\t" not in line:
                errors.append(f"{display}: line {line_number} is not label<TAB>command")
                continue
            label, command = line.split("\t", 1)
            if not label.strip() or not command.strip():
                errors.append(f"{display}: line {line_number} has blank label or command")
    return errors


def _read_command_entries(args: argparse.Namespace) -> list[dict[str, str]]:
    entries = [
        {"label": "manual_command", "command": command}
        for command in args.command
        if isinstance(command, str) and command.strip()
    ]
    for path in args.command_file:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if "\t" not in line:
                continue
            label, command = line.split("\t", 1)
            label = label.strip()
            command = command.strip()
            if label and command:
                entries.append({"label": label, "command": command})
    return entries


def _read_commands(args: argparse.Namespace) -> list[str]:
    commands = [
        entry["command"]
        for entry in _read_command_entries(args)
        if entry.get("command")
    ]
    return commands


def _dry_run_commands(commands: Any) -> list[str]:
    if not isinstance(commands, list):
        return []
    return [
        str(command)
        for command in commands
        if isinstance(command, str) and "--dry-run" in command.split()
    ]


def _nonportable_command_file_requests(items: list[Any]) -> list[str]:
    """Return command-history paths that cannot be restored repo-relatively."""
    bad: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        normalized = text.replace("\\", "/")
        posix = PurePosixPath(normalized)
        windows = PureWindowsPath(text)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
            bad.append(item)
    return bad


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _has_flag_value(tokens: list[str], flag: str, value: str) -> bool:
    return any(
        token == f"{flag}={value}"
        or (token == flag and idx + 1 < len(tokens) and tokens[idx + 1] == value)
        for idx, token in enumerate(tokens)
    )


def _formal_pipeline_invocation_error(command: str) -> str | None:
    tokens = _split_command(command)
    normalized = command.replace("\\", "/")
    uses_python_orchestrator = "scripts/run_pipeline.py" in normalized
    uses_shell_wrapper = "scripts/run_pipeline.sh" in normalized
    if not uses_python_orchestrator and not uses_shell_wrapper:
        return (
            "pipeline_invocation must use the formal full-pipeline entrypoint "
            "`scripts/run_pipeline.py` or `scripts/run_pipeline.sh`"
        )
    if "--dry-run" in tokens:
        return "pipeline_invocation must not be a dry-run command"
    if uses_python_orchestrator:
        if not _has_flag_value(tokens, "--profile", "full"):
            return "pipeline_invocation must use `--profile full`"
        if "--confirm-full-run" not in tokens:
            return "pipeline_invocation must include `--confirm-full-run`"
    elif _has_flag_value(tokens, "--profile", "smoke"):
        return "pipeline_invocation must not use the smoke profile"
    return None


def _case_source_manifest(case_ids: list[str]) -> dict[str, Any]:
    """Record source-file hashes for registered simulation cases.

    These hashes make a result manifest sensitive to case response-surface
    changes even when the git commit alone is not enough to prove provenance
    (for example, dirty worktrees or restored artifacts).
    """
    entries: list[dict[str, Any]] = []
    try:
        import doe_testbed as db
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "cases": entries,
        }

    for case_id in case_ids:
        try:
            case = db.get_case(case_id)
            cls = type(case)
            module = importlib.import_module(cls.__module__)
            source = Path(str(module.__file__)).resolve()
            try:
                rel = source.relative_to(ROOT).as_posix()
            except ValueError:
                rel = source.as_posix()
            entries.append({
                "case_id": case_id,
                "status": "ok",
                "class_name": cls.__name__,
                "module": cls.__module__,
                "source_path": rel,
                "sha256": _sha256(source),
            })
        except Exception as exc:
            entries.append({
                "case_id": case_id,
                "status": "unresolved",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return {"schema_version": 1, "status": "ok", "cases": entries}


def _strict_archive_errors(manifest: dict[str, Any]) -> list[str]:
    """Return submission-blocking manifest defects before writing the archive."""
    errors: list[str] = []
    git = manifest.get("git", {})
    if git.get("source_dirty"):
        errors.append(
            "source/config worktree entries are dirty; commit/stash them or capture a non-strict draft manifest"
        )

    machine = manifest.get("machine")
    if not isinstance(machine, dict):
        errors.append("machine section is missing or malformed")
    else:
        if not isinstance(machine.get("node"), str) or not machine.get("node", "").strip():
            errors.append("machine.node is missing or malformed")
        if not isinstance(machine.get("platform"), str) or not machine.get("platform", "").strip():
            errors.append("machine.platform is missing or malformed")
        if not isinstance(machine.get("cpu_count"), int) or machine.get("cpu_count") <= 0:
            errors.append("machine.cpu_count must be a positive integer")
        if machine.get("ram_gb") in (None, ""):
            errors.append("machine.ram_gb is missing")
        effective_ram = machine.get("effective_ram_gb")
        if effective_ram is not None:
            try:
                if float(effective_ram) <= 0:
                    errors.append("machine.effective_ram_gb must be positive when present")
            except (TypeError, ValueError):
                errors.append("machine.effective_ram_gb must be numeric when present")
        if not isinstance(machine.get("effective_ram_source"), str) or not machine.get("effective_ram_source", "").strip():
            errors.append("machine.effective_ram_source is missing or malformed")
        cgroup = machine.get("cgroup_memory")
        if not isinstance(cgroup, dict):
            errors.append("machine.cgroup_memory is missing or malformed")
        else:
            if cgroup.get("schema_version") != 1:
                errors.append("machine.cgroup_memory.schema_version must be 1")
            if not isinstance(cgroup.get("detected"), bool):
                errors.append("machine.cgroup_memory.detected must be boolean")
            if cgroup.get("limit_gb") is not None:
                try:
                    if float(cgroup.get("limit_gb")) <= 0:
                        errors.append("machine.cgroup_memory.limit_gb must be positive when present")
                except (TypeError, ValueError):
                    errors.append("machine.cgroup_memory.limit_gb must be numeric when present")
            if cgroup.get("unlimited") not in (True, False, None):
                errors.append("machine.cgroup_memory.unlimited must be boolean or null")
        resource_limits = machine.get("resource_limits")
        if not isinstance(resource_limits, dict):
            errors.append("machine.resource_limits is missing or malformed")
        else:
            if resource_limits.get("schema_version") != 1:
                errors.append("machine.resource_limits.schema_version must be 1")
            if not isinstance(resource_limits.get("status"), str) or not resource_limits.get("status", "").strip():
                errors.append("machine.resource_limits.status is missing or malformed")
            if not isinstance(resource_limits.get("limits"), dict):
                errors.append("machine.resource_limits.limits is missing or malformed")
        temp_storage = machine.get("temp_storage")
        if not isinstance(temp_storage, dict):
            errors.append("machine.temp_storage is missing or malformed")
        else:
            if temp_storage.get("schema_version") != 1:
                errors.append("machine.temp_storage.schema_version must be 1")
            resolved_path = temp_storage.get("resolved_path")
            if not isinstance(resolved_path, str) or not resolved_path.strip():
                errors.append("machine.temp_storage.resolved_path is missing or malformed")
            if not isinstance(temp_storage.get("exists"), bool):
                errors.append("machine.temp_storage.exists must be boolean")
            if not isinstance(temp_storage.get("writable"), bool):
                errors.append("machine.temp_storage.writable must be boolean")
            free_gb = temp_storage.get("free_gb")
            if free_gb is not None:
                try:
                    if float(free_gb) < 0:
                        errors.append("machine.temp_storage.free_gb must be non-negative when present")
                except (TypeError, ValueError):
                    errors.append("machine.temp_storage.free_gb must be numeric when present")
            if not isinstance(temp_storage.get("env"), dict):
                errors.append("machine.temp_storage.env is missing or malformed")

    python_environment = manifest.get("python_environment")
    if not isinstance(python_environment, dict):
        errors.append("python_environment section is missing or malformed")
    else:
        if python_environment.get("status") != "ok":
            errors.append(
                "python_environment.status must be 'ok'; "
                f"got {python_environment.get('status')!r}"
            )
        if not isinstance(python_environment.get("command"), str) or not python_environment.get("command", "").strip():
            errors.append("python_environment.command is missing or malformed")
        sha = python_environment.get("pip_freeze_sha256")
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            errors.append("python_environment.pip_freeze_sha256 is missing or invalid")
        package_count = python_environment.get("package_count")
        if not isinstance(package_count, int) or package_count <= 0:
            errors.append("python_environment.package_count must be a positive integer")
        pip_freeze_path = python_environment.get("pip_freeze_path")
        if not isinstance(pip_freeze_path, str) or not pip_freeze_path.strip():
            errors.append("python_environment.pip_freeze_path is missing or malformed")

    scheduler = manifest.get("scheduler")
    if not isinstance(scheduler, dict):
        errors.append("scheduler section is missing or malformed")
    else:
        if scheduler.get("schema_version") != 1:
            errors.append("scheduler.schema_version must be 1")
        if not isinstance(scheduler.get("name"), str) or not scheduler.get("name", "").strip():
            errors.append("scheduler.name is missing or malformed")
        if not isinstance(scheduler.get("detected"), bool):
            errors.append("scheduler.detected must be boolean")
        if not isinstance(scheduler.get("env"), dict):
            errors.append("scheduler.env is missing or malformed")

    accelerator = manifest.get("accelerator")
    if not isinstance(accelerator, dict):
        errors.append("accelerator section is missing or malformed")
    else:
        if accelerator.get("schema_version") != 1:
            errors.append("accelerator.schema_version must be 1")
        nvidia = accelerator.get("nvidia_smi")
        if not isinstance(nvidia, dict):
            errors.append("accelerator.nvidia_smi is missing or malformed")
        torch_cuda = accelerator.get("torch_cuda")
        if not isinstance(torch_cuda, dict):
            errors.append("accelerator.torch_cuda is missing or malformed")
        elif torch_cuda.get("status") == "ok":
            if not isinstance(torch_cuda.get("cuda_available"), bool):
                errors.append("accelerator.torch_cuda.cuda_available must be boolean")
            if not isinstance(torch_cuda.get("device_count"), int) or torch_cuda.get("device_count") < 0:
                errors.append("accelerator.torch_cuda.device_count must be a non-negative integer")

    errors.extend(_torch_reproducibility_errors(manifest.get("torch_reproducibility")))

    container = manifest.get("container")
    if not isinstance(container, dict):
        errors.append("container section is missing or malformed")
    else:
        if container.get("schema_version") != 1:
            errors.append("container.schema_version must be 1")
        if not isinstance(container.get("detected"), bool):
            errors.append("container.detected must be boolean")
        if not isinstance(container.get("env"), dict):
            errors.append("container.env is missing or malformed")
        if not isinstance(container.get("marker_files"), dict):
            errors.append("container.marker_files is missing or malformed")

    run_window = manifest.get("run_window")
    if not isinstance(run_window, dict):
        errors.append("run_window section is missing or malformed")
    else:
        started_at = _parse_iso_datetime(run_window.get("started_at_utc"))
        finished_at = _parse_iso_datetime(run_window.get("finished_at_utc"))
        if started_at is None:
            errors.append("run_window.started_at_utc is missing or not ISO-8601")
        if finished_at is None:
            errors.append("run_window.finished_at_utc is missing or not ISO-8601")
        if started_at is not None and finished_at is not None and finished_at < started_at:
            errors.append("run_window.finished_at_utc is earlier than started_at_utc")
        n_artifacts = run_window.get("n_artifacts")
        if not isinstance(n_artifacts, int) or n_artifacts <= 0:
            errors.append("run_window.n_artifacts must be a positive integer")

    experiment = manifest.get("experiment", {})
    if "case_resolution_error" not in experiment:
        errors.append("experiment.case_resolution_error is missing or malformed")
    elif experiment.get("case_resolution_error") not in (None, ""):
        errors.append(
            "experiment.case_resolution_error is non-empty: "
            f"{experiment.get('case_resolution_error')}"
        )
    commands = experiment.get("commands")
    if not commands:
        errors.append("experiment.commands is empty; pass --command or --command-file")
    requested_command_files = experiment.get("command_files_requested")
    if not isinstance(requested_command_files, list):
        errors.append("experiment.command_files_requested is missing or malformed")
    elif not requested_command_files:
        errors.append("experiment.command_files_requested is empty; pass --command-file")
    else:
        bad_requested = [
            item for item in requested_command_files
            if not isinstance(item, str) or not item.strip()
        ]
        if bad_requested:
            errors.append("experiment.command_files_requested contains blank or non-string entries")
        nonportable_requested = _nonportable_command_file_requests(requested_command_files)
        if nonportable_requested:
            errors.append(
                "experiment.command_files_requested contains non-portable paths; "
                f"use repo-relative paths without parent traversal: {nonportable_requested}"
            )
        dry_run_requested = [
            item for item in requested_command_files
            if isinstance(item, str) and ".dry_run" in item
        ]
        if dry_run_requested:
            errors.append(
                f"experiment.command_files_requested contains dry-run command history files: {dry_run_requested}"
            )
    missing_command_files = experiment.get("command_files_missing")
    if not isinstance(missing_command_files, list):
        errors.append("experiment.command_files_missing is missing or malformed")
    elif missing_command_files:
        errors.append(f"experiment.command_files_missing is non-empty: {missing_command_files}")
    command_file_parse_errors = experiment.get("command_file_parse_errors")
    if not isinstance(command_file_parse_errors, list):
        errors.append("experiment.command_file_parse_errors is missing or malformed")
    elif command_file_parse_errors:
        errors.append(
            f"experiment.command_file_parse_errors is non-empty: {command_file_parse_errors}"
        )
    dry_run_commands = _dry_run_commands(commands)
    if dry_run_commands:
        errors.append(
            "experiment.commands contains --dry-run entries; strict archive requires commands that actually ran"
        )
    command_entries = experiment.get("command_entries")
    if not isinstance(command_entries, list) or not command_entries:
        errors.append("experiment.command_entries is missing or malformed")
    else:
        bad_entries = [
            item for item in command_entries
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("label"), str)
                or not item.get("label", "").strip()
                or not isinstance(item.get("command"), str)
                or not item.get("command", "").strip()
            )
        ]
        if bad_entries:
            errors.append("experiment.command_entries contains malformed entries")
        entry_commands = [
            item.get("command")
            for item in command_entries
            if isinstance(item, dict) and isinstance(item.get("command"), str)
        ]
        if isinstance(commands, list) and entry_commands != commands:
            errors.append("experiment.command_entries commands do not match experiment.commands")
        pipeline_invocations = [
            item for item in command_entries
            if (
                isinstance(item, dict)
                and item.get("label") == "pipeline_invocation"
                and isinstance(item.get("command"), str)
                and item.get("command", "").strip()
            )
        ]
        if not pipeline_invocations:
            errors.append("experiment.command_entries does not include pipeline_invocation")
        elif len(pipeline_invocations) > 1:
            errors.append("experiment.command_entries contains multiple pipeline_invocation entries")
        else:
            invocation_error = _formal_pipeline_invocation_error(
                str(pipeline_invocations[0].get("command", ""))
            )
            if invocation_error:
                errors.append(f"experiment.command_entries {invocation_error}")

    summary = manifest.get("results_summary", {})
    if not summary.get("exists"):
        errors.append("results.csv is missing from results-dir")
    else:
        observed_rows = _optional_nonnegative_int(
            summary.get("n_rows"),
            "results_summary.n_rows",
            errors,
        )
        row_contract_fields = [
            "n_csv_contract_errors",
            "n_invalid_budget_mult_values",
            "n_invalid_repeat_values",
            "n_missing_protocol_key_values",
            "n_invalid_status_values",
            "n_missing_failure_category_values",
            "n_invalid_failure_category_values",
            "n_failure_category_status_mismatch_values",
            "n_invalid_ok_nrmse_values",
            "n_invalid_t_total_s_values",
            "n_invalid_runtime_timestamp_values",
            "n_invalid_runtime_time_order_values",
            "n_invalid_runtime_duration_values",
            "n_invalid_rss_gb_values",
            "n_invalid_rss_gb_peak_order_values",
            "n_missing_failure_error_values",
            "n_ok_rows_with_error_values",
        ]
        for field in row_contract_fields:
            if _summary_nonnegative_int(summary, field, errors) > 0:
                errors.append(f"results_summary.{field} is non-zero")
        failure_category_counts = summary.get("failure_category_counts")
        if not isinstance(failure_category_counts, dict):
            errors.append("results_summary.failure_category_counts is missing or malformed")
        else:
            failure_category_total = 0
            failure_category_total_valid = True
            unknown_failure_categories: list[str] = []
            for key, value in failure_category_counts.items():
                if str(key) not in ALLOWED_FAILURE_CATEGORIES:
                    unknown_failure_categories.append(str(key))
                parsed = _optional_nonnegative_int(
                    value,
                    f"results_summary.failure_category_counts.{key}",
                    errors,
                )
                if parsed is None:
                    failure_category_total_valid = False
                else:
                    failure_category_total += parsed
            if unknown_failure_categories:
                errors.append(
                    "results_summary.failure_category_counts contains unknown "
                    f"categories: {sorted(unknown_failure_categories)}"
                )
            if (
                failure_category_total_valid
                and observed_rows is not None
                and failure_category_total != observed_rows
            ):
                errors.append(
                    "results_summary.failure_category_counts total "
                    f"{failure_category_total} does not match n_rows {observed_rows}"
                )

        runtime_timestamp = summary.get("runtime_timestamp_summary")
        if not isinstance(runtime_timestamp, dict):
            errors.append("results_summary.runtime_timestamp_summary is missing or malformed")
        else:
            if runtime_timestamp.get("timestamp_trace_columns_present") is not True:
                errors.append(
                    "results_summary.runtime_timestamp_summary reports missing "
                    "timestamp trace columns: "
                    f"{runtime_timestamp.get('missing_timestamp_trace_columns')}"
                )
            valid_rows = _optional_nonnegative_int(
                runtime_timestamp.get("n_rows"),
                "results_summary.runtime_timestamp_summary.n_rows",
                errors,
            )
            if observed_rows is not None and valid_rows != observed_rows:
                errors.append(
                    "results_summary.runtime_timestamp_summary row-count drift: "
                    f"expected={observed_rows}, observed={valid_rows}"
                )

        runtime_resource = summary.get("runtime_resource_summary")
        if not isinstance(runtime_resource, dict):
            errors.append("results_summary.runtime_resource_summary is missing or malformed")
        else:
            if runtime_resource.get("rss_trace_columns_present") is not True:
                errors.append(
                    "results_summary.runtime_resource_summary reports missing "
                    f"RSS trace columns: {runtime_resource.get('missing_rss_trace_columns')}"
                )
            valid_peak_rows = _optional_nonnegative_int(
                runtime_resource.get("n_rows_with_valid_rss_gb_peak"),
                "results_summary.runtime_resource_summary.n_rows_with_valid_rss_gb_peak",
                errors,
            )
            if observed_rows is not None and valid_peak_rows != observed_rows:
                errors.append(
                    "results_summary.runtime_resource_summary valid peak-RSS row "
                    f"count drift: expected={observed_rows}, observed={valid_peak_rows}"
                )
            peak_max = runtime_resource.get("rss_gb_peak_max")
            try:
                peak_max_float = float(peak_max)
            except (TypeError, ValueError):
                peak_max_float = -1.0
            if peak_max_float < 0:
                errors.append(
                    "results_summary.runtime_resource_summary.rss_gb_peak_max "
                    "must be a finite non-negative number"
                )

        declared_cases = {str(item) for item in (experiment.get("case_ids_declared") or [])}
        declared_case_count = (
            len(declared_cases)
            if declared_cases and declared_cases != {"ALL"}
            else None
        )
        observed_cases = _strict_observed_string_set_errors(
            summary=summary,
            declared=declared_cases,
            declared_count=declared_case_count,
            observed_key="case_ids_observed",
            count_key="n_cases_observed",
            label="case",
            errors=errors,
        )

        declared_methods = {str(item) for item in (experiment.get("methods_declared") or [])}
        observed_methods = _strict_observed_string_set_errors(
            summary=summary,
            declared=declared_methods,
            declared_count=len(declared_methods) if declared_methods else None,
            observed_key="methods_observed",
            count_key="n_methods_observed",
            label="method",
            errors=errors,
        )

        budgets = _int_list(experiment.get("budgets") or [], "experiment.budgets", errors)
        observed_budgets = _int_list(
            summary.get("budgets_observed") or [],
            "results_summary.budgets_observed",
            errors,
        )
        if budgets and observed_budgets != budgets:
            errors.append(
                f"budget set drift: expected={budgets}, observed={observed_budgets}"
            )

        repeats = _int_list(
            summary.get("repeats_observed") or [],
            "results_summary.repeats_observed",
            errors,
        )
        n_repeats = _optional_nonnegative_int(
            experiment.get("n_repeats"),
            "experiment.n_repeats",
            errors,
        )
        if n_repeats is not None:
            expected_repeats = list(range(n_repeats))
            if repeats != expected_repeats:
                errors.append(
                    f"repeat set drift: expected={expected_repeats}, observed={repeats}"
                )

        n_cases = declared_case_count or len(observed_cases)
        n_methods = len(declared_methods) if declared_methods else len(observed_methods)
        if n_cases and n_methods and budgets and n_repeats is not None:
            expected_rows = n_cases * n_methods * len(budgets) * n_repeats
            if observed_rows is not None and observed_rows != expected_rows:
                errors.append(
                    f"row-count drift: expected={expected_rows}, observed={observed_rows}"
                )

    artifact_entries = manifest.get("artifacts", {}).get("files", [])
    if not isinstance(artifact_entries, list):
        errors.append("artifacts.files is missing or malformed")
        artifact_entries = []
    artifact_paths_list = [
        str(item.get("path", "")).replace("\\", "/")
        for item in artifact_entries
        if isinstance(item, dict) and item.get("path")
    ]
    symlink_artifacts = sorted(
        str(item.get("path", "")).replace("\\", "/")
        for item in artifact_entries
        if isinstance(item, dict) and item.get("is_symlink") is True
    )
    if symlink_artifacts:
        errors.append(
            "artifacts.files contains symlink entries; rebuild the archive with "
            f"real files/directories only: {symlink_artifacts[:20]}"
        )
    artifact_paths = set(artifact_paths_list)
    duplicate_artifact_paths = sorted(
        path for path in artifact_paths if artifact_paths_list.count(path) > 1
    )
    if duplicate_artifact_paths:
        errors.append(f"artifacts.files contains duplicate paths: {duplicate_artifact_paths}")
    if "results.csv" not in artifact_paths:
        errors.append("artifacts.files does not include results.csv")
    if "sweep_integrity_report.md" not in artifact_paths:
        errors.append("artifacts.files does not include sweep_integrity_report.md")
    pip_freeze_path = manifest.get("python_environment", {}).get("pip_freeze_path")
    if isinstance(pip_freeze_path, str) and pip_freeze_path.strip():
        normalized_pip_freeze_path = pip_freeze_path.replace("\\", "/")
        if normalized_pip_freeze_path not in artifact_paths:
            errors.append(
                "artifacts.files does not include the archived pip freeze artifact: "
                f"{normalized_pip_freeze_path}"
            )

    case_sources = manifest.get("case_sources", {})
    case_entries = case_sources.get("cases") if isinstance(case_sources, dict) else None
    if case_sources.get("status") != "ok" or not isinstance(case_entries, list) or not case_entries:
        errors.append("case_sources fingerprints are unavailable")
    elif any(item.get("status") != "ok" or not item.get("sha256") for item in case_entries):
        errors.append("case_sources contains unresolved entries")

    method_sources = manifest.get("method_sources", {})
    method_entries = method_sources.get("methods") if isinstance(method_sources, dict) else None
    if method_sources.get("status") != "ok" or not isinstance(method_entries, list) or not method_entries:
        errors.append("method_sources fingerprints are unavailable")
    elif any(item.get("status") != "ok" or not item.get("method_source_sha256") for item in method_entries):
        errors.append("method_sources contains unresolved entries")

    return errors


def write_manifest_atomic(manifest: dict[str, Any], output: Path) -> None:
    """Write RUN_MANIFEST.json without exposing partial JSON to readers."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=str(output.parent)))
    tmp_output = tmp_parent / output.name
    try:
        tmp_output.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        shutil.move(str(tmp_output), str(output))
    finally:
        if tmp_parent.exists():
            shutil.rmtree(tmp_parent)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    canonical_output = (results_dir / "RUN_MANIFEST.json").resolve()
    output = args.output.resolve() if args.output else canonical_output
    output.parent.mkdir(parents=True, exist_ok=True)
    # A preview written elsewhere must be read-only with respect to the formal
    # archive.  Only canonical manifest generation refreshes its sibling lock.
    pip_freeze_output = (
        results_dir / PIP_FREEZE_ARTIFACT_NAME
        if output == canonical_output
        else None
    )

    case_resolution = _resolve_cases(args.cases)
    cases = case_resolution.case_ids
    methods = _resolve_methods(args.pool)
    command_file_status = _command_file_status(args)
    command_entries = _read_command_entries(args)
    python_environment = _python_environment_info(pip_freeze_output)
    artifact_files = _file_manifest(results_dir, exclude=output)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_script": Path(__file__).relative_to(ROOT).as_posix(),
        "manifest_command": shlex.join([sys.executable, str(Path(__file__).relative_to(ROOT)), *(argv or sys.argv[1:])]),
        "git": _git_info(),
        "machine": _machine_info(),
        "torch_reproducibility": _torch_reproducibility_snapshot(
            apply_policy=True
        ),
        "scheduler": _scheduler_info(),
        "accelerator": _accelerator_info(),
        "container": _container_info(),
        "python_environment": python_environment,
        "run_window": _run_window_from_files(artifact_files),
        "dataset_window": _dataset_window_from_results(results_dir),
        "packages": _package_versions(),
        "experiment": {
            "pool": args.pool,
            "n_methods_declared": len(methods) if methods else None,
            "methods_declared": methods,
            "budgets": args.budgets,
            "n_repeats": args.n_repeats,
            "n_test": args.n_test,
            "seed_strategy": args.seed_strategy,
            "seed_start": args.seed_start,
            "seed_derivation": "blake2b(f'{seed_start}|{case_id}|{repeat}', digest_size=4) & 0x7fffffff",
            "cases_requested": args.cases,
            "case_resolution_error": case_resolution.error,
            "n_cases_declared": len(cases) if cases != ["ALL"] else None,
            "case_ids_declared": cases,
            "command_files_requested": command_file_status["requested"],
            "command_files_missing": command_file_status["missing"],
            "command_file_parse_errors": command_file_status["parse_errors"],
            "command_entries": command_entries,
            "commands": [entry["command"] for entry in command_entries],
        },
        "case_sources": _case_source_manifest(cases),
        "method_sources": _method_source_manifest(args.pool),
        "results_summary": _results_summary(results_dir / "results.csv"),
        "artifacts": {
            "results_dir": results_dir.as_posix(),
            "files": artifact_files,
        },
    }
    if args.require_complete_archive:
        errors = _strict_archive_errors(manifest)
        if errors:
            for err in errors:
                print(f"[ERROR] {err}", file=sys.stderr)
            return 3
    write_manifest_atomic(manifest, output)
    print(f"Wrote RUN_MANIFEST: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
