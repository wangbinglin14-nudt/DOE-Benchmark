"""Run the no-sweep checks required before launching the server full run.

This script is intended for the Linux server/operator environment.  It checks
case-validation provenance, targeted physics regressions, the formal pipeline
Stage 1 gates, and the pre-full-sweep publication launch-preflight roll-up, but
it never enters Stage 2 or calls the large
``run_full_benchmark.py`` sweep.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.project_layout import MANUSCRIPTS_DIR  # noqa: E402
from scripts.server_operator_contracts import server_operator_contract_nodeids  # noqa: E402

CASE_VALIDATION_STATUS = ROOT / "docs" / "case_validation_status.csv"
CASE_WEB_VALIDATION_AUDIT = ROOT / "docs" / "CASE_WEB_VALIDATION_AUDIT.md"
REQUIRED_FULL_DRY_RUN_STAGES = (
    "validate",
    "case_readiness",
    "dependency_preflight",
    "sweep",
    "sweep_audit",
    "report",
    "d1",
    "d1_ci",
    "d2",
    "d3",
    "figdata",
    "figures",
    "test_protocol",
    "numbers",
    "si",
    "case_validation_note_sources",
    "case_validation_summary",
    "manifest",
    "submission_ready",
)
CASE_WEB_DECISION_MARKERS = (
    "keep",
    "fix",
    "fixed",
    "corrected",
    "rerun",
    "resweep",
    "no code change",
    "no response-surface change",
)
CASE_WEB_PLAUSIBILITY_NONVALIDATION_MARKERS = (
    "not validation data",
    "not external validation",
    "not strong external",
    "not calibration evidence",
    "not calibrated",
    "boundary evidence",
    "boundary/context evidence",
    "scope/context",
    "context anchors only",
    "context anchors",
    "keep p2 plausibility-only",
    "remains p2 plausibility-only",
)
P2_SOURCE_SNAPSHOT_REQUIREMENTS = (
    {
        "label": "aerodynamics",
        "case_ids": (
            "airfoil_stall_2d",
            "airfoil_transonic_5d",
            "rotor_bemt_6d",
            "supersonic_intake_5d",
            "waverider_4d",
            "wing_vlm_6d",
        ),
        "markers": (
            "Additional `airfoil_stall_2d` 2026-06-08 single-section boundary tightening:",
            "Additional `airfoil_transonic_5d` 2026-06-08 single-section boundary tightening:",
            "Additional `rotor_bemt_6d` 2026-06-08 production-model boundary tightening:",
            "Additional `supersonic_intake_5d` 2026-06-08 inlet-performance boundary re-check:",
            "Additional `waverider_4d` 2026-06-08 NASA hypersonic-flight boundary tightening:",
            "Additional `wing_vlm_6d` 2026-06-08 official-source boundary tightening:",
        ),
    },
    {
        "label": "geomechanics",
        "case_ids": (
            "dam_seepage_6d",
            "groundwater_contamination_6d",
            "oil_reservoir_9d",
            "soil_consolidation_5d",
        ),
        "markers": (
            "Additional `dam_seepage_6d` 2026-06-08 public-scope boundary tightening:",
            "Additional `groundwater_contamination_6d` 2026-06-08 advanced-package boundary re-check:",
            "Additional `groundwater_contamination_6d` 2026-06-08 USGS boundary-setting tightening",
            "Additional `oil_reservoir_9d` 2026-06-08 official-source boundary tightening:",
            "Additional `oil_reservoir_9d` 2026-06-08 USGS boundary-setting tightening:",
            "Additional `soil_consolidation_5d` 2026-06-08 official-source boundary tightening:",
        ),
    },
    {
        "label": "multidisciplinary",
        "case_ids": (
            "aircraft_conceptual_8d",
            "pressure_vessel_4d",
            "solar_pv_yield_6d",
            "wind_turbine_power_6d",
        ),
        "markers": (
            "Additional `aircraft_conceptual_8d` 2026-06-08 official-source boundary tightening:",
            "Additional `pressure_vessel_4d` 2026-06-08 ASME scope tightening:",
            "Additional `solar_pv_yield_6d` 2026-06-08 official-source boundary tightening:",
            "Additional `wind_turbine_power_6d` 2026-06-08 official-source boundary tightening:",
        ),
    },
    {
        "label": "analytical-control",
        "case_ids": (
            "multilayer_wall_4d",
            "multilayer_wall_8d",
            "multilayer_wall_16d",
            "pressure_vessel_4d",
            "soil_consolidation_5d",
        ),
        "markers": (
            "## 2026-06-08 Analytical-Control Source Verification Snapshot",
            "MIT thermal-resistance-circuit notes",
            "NPTEL composite-wall conduction material",
            "NASA Passive Thermal Control Engineering Guidebook",
            "ASME BPVC Section VIII",
            "Itasca FLAC3D one-dimensional consolidation example",
            "Evidence tier remains P2 analytic-control",
        ),
    },
)
REQUIRED_SHELL_SYNTAX_FILES = (
    "scripts/run_pipeline.sh",
    "scripts/check_progress_remote.sh",
    "scripts/create_server_return_package.sh",
    "scripts/launch_sweep_remote.sh",
    "scripts/launch_sota_robustness.sh",
    "scripts/launch_mfbo.sh",
    "scripts/launch_d1_pairings.sh",
    "scripts/setup_remote.sh",
    "scripts/_remote_lib.sh",
)
DEFAULT_RESULTS_CSV = ROOT / "benchmark" / "results" / "full_benchmark" / "results.csv"
DEFAULT_MIN_SERVER_RAM_GB = 64.0
DEFAULT_MIN_SERVER_DISK_FREE_GB = 25.0
DEFAULT_CHECK_TIMEOUT_S = 1800.0
TEMP_STORAGE_ENV_VARS = (
    "TMPDIR",
    "TMP",
    "TEMP",
    "TEMPDIR",
    "SLURM_TMPDIR",
    "SCRATCH",
    "LOCAL_SCRATCH",
    "JOBSCRATCH",
)
SERVER_RESOURCE_CHECK_PATHS = (
    ROOT,
    ROOT / "benchmark" / "results",
    ROOT / "logs",
    MANUSCRIPTS_DIR,
)


@dataclass(frozen=True)
class LaunchCheck:
    label: str
    command: list[str]


def _script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def command_to_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def _is_run_pipeline_command(command: Sequence[str]) -> bool:
    return any(str(part).replace("\\", "/").endswith("scripts/run_pipeline.py") for part in command)


def _is_full_pipeline_dry_run_command(command: Sequence[str]) -> bool:
    command = [str(part) for part in command]
    if not _is_run_pipeline_command(command) or "--dry-run" not in command:
        return False
    if "--profile" in command:
        idx = command.index("--profile")
        return command[idx + 1: idx + 2] == ["full"]
    return True


def validate_no_sweep_commands(checks: Sequence[LaunchCheck]) -> list[str]:
    """Return safety errors if the launch preflight command graph can run Stage 2."""
    errors: list[str] = []
    for check in checks:
        command = [str(part) for part in check.command]
        normalized = [part.replace("\\", "/") for part in command]
        text = command_to_text(command)
        if any(part.endswith("benchmark/scripts/run_full_benchmark.py") for part in normalized):
            errors.append(
                f"{check.label}: preflight must not call run_full_benchmark.py ({text})"
            )
        if "--confirm-large-run" in command:
            errors.append(
                f"{check.label}: preflight must not include --confirm-large-run ({text})"
            )
        if _is_run_pipeline_command(command) and "--confirm-full-run" in command:
            safe_pipeline = "--dry-run" in command or (
                "--to-stage" in command
                and command[command.index("--to-stage") + 1: command.index("--to-stage") + 2]
                == ["1"]
            )
            if not safe_pipeline:
                errors.append(
                    f"{check.label}: confirmed full pipeline command is not limited "
                    f"to Stage 1 or dry-run ({text})"
                )
    return errors


def validate_pipeline_dry_run_stage_graph(checks: Sequence[LaunchCheck]) -> list[str]:
    """Ensure the launch dry-run still covers every formal full-pipeline stage."""
    if not any(_is_full_pipeline_dry_run_command(check.command) for check in checks):
        return []
    try:
        spec = importlib.util.spec_from_file_location(
            "server_launch_ready_run_pipeline",
            ROOT / "scripts" / "run_pipeline.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load scripts/run_pipeline.py")
        run_pipeline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = run_pipeline
        spec.loader.exec_module(run_pipeline)

        args = run_pipeline.parse_args([
            "--profile",
            "full",
            "--confirm-full-run",
            "--dry-run",
        ])
        cfg = run_pipeline.resolve_config(args)
        stages = run_pipeline.select_stages(
            run_pipeline.build_stages(cfg),
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            only=args.only,
            skip=args.skip,
            include_smoke=bool(args.with_smoke or args.profile == "smoke"),
        )
    except Exception as exc:
        return [f"could not validate full pipeline dry-run stage graph: {type(exc).__name__}: {exc}"]

    observed = [stage.name for stage in stages]
    observed_set = set(observed)
    missing = [
        stage
        for stage in REQUIRED_FULL_DRY_RUN_STAGES
        if stage not in observed_set
    ]
    errors: list[str] = []
    if missing:
        errors.append(
            "full pipeline dry-run is missing required stage(s): "
            f"{missing}"
        )
    order = [
        stage
        for stage in REQUIRED_FULL_DRY_RUN_STAGES
        if stage in observed_set
    ]
    out_of_order = [
        (left, right)
        for left, right in zip(order, order[1:])
        if observed.index(left) > observed.index(right)
    ]
    if out_of_order:
        errors.append(
            "full pipeline dry-run stage order drifted for adjacent required "
            f"stage(s): {out_of_order}"
        )
    return errors


def validate_shell_script_syntax(
    *,
    bash_executable: str | None = None,
) -> tuple[list[str], list[str]]:
    """Check server-facing shell scripts for LF endings and shell syntax.

    The review workstation may be Windows without WSL, so missing bash is a
    local skip there, but CRLF line endings are still a launch risk and are
    checked on every platform.  On Linux/macOS, missing bash is a launch
    blocker because ``run_pipeline.sh`` is the formal server entry point.
    """
    errors: list[str] = []
    messages: list[str] = []
    for rel_path in REQUIRED_SHELL_SYNTAX_FILES:
        path = ROOT / rel_path
        if not path.is_file():
            errors.append(f"required shell script missing: {rel_path}")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            errors.append(
                f"required shell script could not be read: {rel_path} "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        if b"\r\n" in data or b"\r" in data:
            errors.append(
                f"{rel_path}: CRLF/CR line endings detected; convert server "
                "shell wrappers to LF before launching on Linux"
            )
        else:
            messages.append(f"{rel_path}: LF line endings PASS")

    bash = bash_executable or shutil.which("bash")
    if bash is None:
        if os.name == "nt":
            messages.append(
                "bash syntax check skipped on this Windows workstation; "
                "shell syntax will be checked on Linux server"
            )
            return errors, messages
        errors.append(
            "bash executable not found; install bash before running the server launch wrappers"
        )
        return errors, messages

    if os.name == "nt" and bash_executable is None:
        try:
            probe = subprocess.run(
                [bash, "--version"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            messages.append(
                "bash syntax check skipped on this Windows workstation; "
                f"found bash at {bash!r} but it is not usable "
                f"({type(exc).__name__}: {exc}); shell syntax will be checked "
                "on Linux server"
            )
            return errors, messages
        if probe.returncode != 0:
            stderr = " ".join(str(probe.stderr).split())
            detail = f" exit={probe.returncode}"
            if stderr:
                detail += f": {stderr}"
            messages.append(
                "bash syntax check skipped on this Windows workstation; "
                f"found bash at {bash!r} but it is not usable ({detail}); "
                "shell syntax will be checked on Linux server"
            )
            return errors, messages

    for rel_path in REQUIRED_SHELL_SYNTAX_FILES:
        path = ROOT / rel_path
        if not path.is_file():
            continue
        try:
            completed = subprocess.run(
                [bash, "-n", str(path)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            errors.append(
                f"{rel_path}: could not run bash -n with {bash!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if completed.returncode != 0:
            stderr = " ".join(completed.stderr.split())
            errors.append(
                f"{rel_path}: bash -n failed with exit={completed.returncode}: {stderr}"
            )
        else:
            messages.append(f"{rel_path}: bash -n PASS")
    return errors, messages


def _env_positive_float(name: str, default: float) -> tuple[float | None, str | None]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default), None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"{name} must be a positive finite number; got {raw!r}"
    if not math.isfinite(value) or value <= 0:
        return None, f"{name} must be a positive finite number; got {raw!r}"
    return value, None


def _detect_total_ram_gb() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().total) / 1024**3
    except Exception:
        pass
    meminfo = Path("/proc/meminfo")
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024**2
    except Exception:
        return None
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


def _detect_cgroup_memory_limit_gb(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_self_cgroup: Path = Path("/proc/self/cgroup"),
) -> tuple[float | None, str | None]:
    """Return an effective cgroup memory limit when Linux exposes one.

    ``psutil`` and ``/proc/meminfo`` often report node/container-visible
    physical memory, while Slurm/systemd/container cgroups can enforce a much
    lower job memory cap.  Treat "max" and huge kernel sentinel values as
    unlimited.
    """
    finite_limits: list[tuple[float, str]] = []
    for path, source in _cgroup_memory_limit_candidates(cgroup_root, proc_self_cgroup):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw.lower() == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= 2**60:
            continue
        finite_limits.append((float(value) / 1024**3, source))
    if not finite_limits:
        return None, None
    return min(finite_limits, key=lambda item: item[0])


def _detect_effective_ram_gb() -> tuple[float | None, str, float | None, float | None]:
    """Return RAM available to the sweep after cgroup limits are considered."""
    total_gb = _detect_total_ram_gb()
    cgroup_gb, cgroup_source = _detect_cgroup_memory_limit_gb()
    if total_gb is None:
        if cgroup_gb is None:
            return None, "unknown", None, None
        return cgroup_gb, str(cgroup_source), None, cgroup_gb
    if cgroup_gb is not None and cgroup_gb < total_gb:
        return cgroup_gb, str(cgroup_source), total_gb, cgroup_gb
    return total_gb, "physical RAM", total_gb, cgroup_gb


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _probe_directory_writable(path: Path) -> str | None:
    """Return an error string when ``path`` cannot actually accept writes.

    ``os.access(..., os.W_OK)`` is optimistic on some shared filesystems and
    quota-managed scratch mounts.  For the launch preflight we want the stronger
    guarantee that the current job can create, flush, and remove a small probe
    file in the directory that Stage 2 will rely on.
    """
    directory = path if path.is_dir() else path.parent
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(directory),
            prefix=".doe_benchmark_write_probe_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            probe_path = Path(handle.name)
            handle.write("probe\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        return (
            "temporary/scratch path is not writable for the current job: "
            f"{directory} ({type(exc).__name__}: {exc})"
        )
    finally:
        if probe_path is not None:
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                pass
    return None


def _temp_storage_candidates() -> list[Path]:
    """Return unique current temp/scratch paths worth preflight disk checks."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path_like: str | Path | None) -> None:
        if path_like in (None, ""):
            return
        try:
            path = Path(str(path_like)).expanduser()
        except Exception:
            return
        key = path.as_posix()
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    add(tempfile.gettempdir())
    for name in TEMP_STORAGE_ENV_VARS:
        add(os.environ.get(name))
    return candidates


def validate_server_resource_capacity(
    *,
    min_ram_gb: float | None = None,
    min_disk_free_gb: float | None = None,
    paths: Sequence[Path] = SERVER_RESOURCE_CHECK_PATHS,
) -> tuple[list[str], list[str]]:
    """Check coarse server capacity before the 81,600-cell sweep is launched."""
    if os.name == "nt":
        return [], [
            "server resource-capacity check skipped on this Windows workstation; "
            "RAM/disk capacity will be checked on Linux server"
        ]

    errors: list[str] = []
    messages: list[str] = []
    if min_ram_gb is None:
        min_ram_gb, error = _env_positive_float(
            "DOE_BENCHMARK_MIN_SERVER_RAM_GB",
            DEFAULT_MIN_SERVER_RAM_GB,
        )
        if error:
            errors.append(error)
            min_ram_gb = DEFAULT_MIN_SERVER_RAM_GB
    if min_disk_free_gb is None:
        min_disk_free_gb, error = _env_positive_float(
            "DOE_BENCHMARK_MIN_SERVER_DISK_FREE_GB",
            DEFAULT_MIN_SERVER_DISK_FREE_GB,
        )
        if error:
            errors.append(error)
            min_disk_free_gb = DEFAULT_MIN_SERVER_DISK_FREE_GB

    assert min_ram_gb is not None
    assert min_disk_free_gb is not None
    ram_gb, ram_source, total_ram_gb, cgroup_ram_gb = _detect_effective_ram_gb()
    if ram_gb is None:
        errors.append(
            "could not determine effective RAM; install psutil, expose "
            "/proc/meminfo, or expose Linux cgroup memory limits before "
            "launching the full server sweep"
        )
    elif ram_gb < min_ram_gb:
        if total_ram_gb is not None and cgroup_ram_gb is not None and cgroup_ram_gb < total_ram_gb:
            detail = f" ({ram_source}; physical RAM {total_ram_gb:.1f} GB)"
        else:
            detail = f" ({ram_source})"
        errors.append(
            f"effective server RAM {ram_gb:.1f} GB{detail} is below the launch minimum "
            f"{min_ram_gb:.1f} GB for the formal full sweep"
        )
    else:
        if total_ram_gb is not None and cgroup_ram_gb is not None and cgroup_ram_gb < total_ram_gb:
            detail = f" ({ram_source}; physical RAM {total_ram_gb:.1f} GB)"
        else:
            detail = f" ({ram_source})"
        messages.append(
            f"effective server RAM {ram_gb:.1f} GB{detail} >= required {min_ram_gb:.1f} GB"
        )

    checked_devices: set[tuple[int, int]] = set()
    temp_paths = _temp_storage_candidates()
    all_paths = tuple(paths) + tuple(temp_paths)
    for raw_path in all_paths:
        path = _nearest_existing_path(Path(raw_path))
        try:
            stat = path.stat()
            device_key = (stat.st_dev, stat.st_ino if path.is_file() else 0)
        except OSError:
            device_key = (hash(str(path.resolve() if path.exists() else path)), 0)
        if device_key in checked_devices:
            continue
        checked_devices.add(device_key)
        try:
            usage = shutil.disk_usage(path)
        except Exception as exc:
            errors.append(
                f"could not determine free disk space for {path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        free_gb = float(usage.free) / 1024**3
        if free_gb < min_disk_free_gb:
            errors.append(
                f"free disk space at {path} is {free_gb:.1f} GB, below the "
                f"launch minimum {min_disk_free_gb:.1f} GB"
            )
        else:
            messages.append(
                f"free disk space at {path}: {free_gb:.1f} GB >= "
                f"required {min_disk_free_gb:.1f} GB"
            )
    for temp_path in temp_paths:
        nearest = _nearest_existing_path(Path(temp_path))
        if not nearest.exists():
            errors.append(
                f"temporary/scratch path does not exist and cannot be checked: {temp_path}"
            )
            continue
        write_error = _probe_directory_writable(nearest)
        if write_error is not None:
            errors.append(write_error)
    return errors, messages


def _p2_case_evidence_tiers(
    status_csv: Path = CASE_VALIDATION_STATUS,
) -> tuple[dict[str, str], list[str]]:
    if not status_csv.exists():
        return {}, [f"case-validation ledger missing: {status_csv}"]
    try:
        from scripts.generate_case_validation_summary import evidence_tier

        with status_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        return {}, [
            f"could not read case-validation ledger {status_csv}: "
            f"{type(exc).__name__}: {exc}"
        ]
    return {
        str(row.get("case_id", "")).strip(): evidence_tier(row)
        for row in rows
        if str(row.get("risk_level", "")).strip() == "P2"
        and str(row.get("case_id", "")).strip()
    }, []


def _p2_audit_tier_error(case_id: str, tier: str, matches: Sequence[str]) -> str | None:
    joined = " ".join(matches).lower()
    has_p2_marker = "evidence tier remains p2" in joined
    has_analytic_marker = (
        "analytic-control" in joined
        or "analytical-control" in joined
        or "analytical control" in joined
        or "analytic control" in joined
    )
    if not has_p2_marker:
        return f"{case_id}: missing `Evidence tier remains P2` marker"
    if tier in {"targeted_analytical_control", "external_note_analytic_control"}:
        if not has_analytic_marker:
            return (
                f"{case_id}: ledger tier {tier} requires an explicit "
                "analytic/analytical-control marker"
            )
    elif tier == "targeted_plausibility_only" and has_analytic_marker:
        return (
            f"{case_id}: ledger tier targeted_plausibility_only must not be "
            "described as analytic/analytical-control in the web audit"
        )
    elif tier == "targeted_plausibility_only":
        has_plausibility_marker = "plausibility-only" in joined or "plausibility only" in joined
        has_nonvalidation_marker = any(
            marker in joined for marker in CASE_WEB_PLAUSIBILITY_NONVALIDATION_MARKERS
        )
        if not has_plausibility_marker or not has_nonvalidation_marker:
            return (
                f"{case_id}: ledger tier targeted_plausibility_only requires "
                "explicit plausibility-only wording and a non-validation "
                "boundary marker such as `not validation data`, "
                "`boundary/context evidence`, or `not external validation`"
            )
    return None


def validate_case_web_validation_audit_coverage(
    *,
    status_csv: Path = CASE_VALIDATION_STATUS,
    audit_md: Path = CASE_WEB_VALIDATION_AUDIT,
) -> list[str]:
    """Ensure every P2 case has an independent web/literature audit row."""
    p2_tiers, errors = _p2_case_evidence_tiers(status_csv)
    if errors:
        return errors
    if not audit_md.exists():
        return [f"case web-validation audit missing: {audit_md}"]
    try:
        lines = audit_md.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [
            f"could not read case web-validation audit {audit_md}: "
            f"{type(exc).__name__}: {exc}"
        ]

    table_lines = [line for line in lines if line.lstrip().startswith("|")]
    missing: list[str] = []
    tier_errors: list[str] = []
    weak_decision: list[str] = []
    for case_id, tier in sorted(p2_tiers.items()):
        needle = f"`{case_id}`"
        matches = [line for line in table_lines if needle in line]
        if not matches:
            missing.append(case_id)
            continue
        tier_error = _p2_audit_tier_error(case_id, tier, matches)
        if tier_error:
            tier_errors.append(tier_error)
        if not any(
            marker in line.lower()
            for line in matches
            for marker in CASE_WEB_DECISION_MARKERS
        ):
            weak_decision.append(case_id)

    out: list[str] = []
    if missing:
        out.append(
            "CASE_WEB_VALIDATION_AUDIT.md is missing independent web/literature "
            f"coverage for P2 case(s): {missing}"
        )
    if tier_errors:
        out.append(
            "CASE_WEB_VALIDATION_AUDIT.md has P2 case row(s) whose evidence-tier "
            f"markers do not match docs/case_validation_status.csv: {tier_errors}"
        )
    if weak_decision:
        out.append(
            "CASE_WEB_VALIDATION_AUDIT.md has P2 case row(s) without an explicit "
            f"keep/fix/resweep judgment: {weak_decision}"
        )
    return out


def validate_p2_source_verification_snapshots(
    *,
    status_csv: Path = CASE_VALIDATION_STATUS,
    audit_md: Path = CASE_WEB_VALIDATION_AUDIT,
) -> list[str]:
    """Require official-source snapshots for high-risk P2 case families.

    The row-by-row P2 audit coverage above prevents unlabelled plausibility
    cases.  This companion check protects the broader reviewer-facing source
    snapshots for domains where a compact surrogate can easily be mistaken for
    a calibrated simulator or code-compliant engineering workflow.
    """
    if not status_csv.exists():
        return [f"case-validation ledger missing: {status_csv}"]
    if not audit_md.exists():
        return [f"case web-validation audit missing: {audit_md}"]
    try:
        with status_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        audit_text = audit_md.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [
            "could not read P2 source-verification inputs: "
            f"{type(exc).__name__}: {exc}"
        ]
    normalized_audit_text = " ".join(audit_text.split())

    p2_cases = {
        str(row.get("case_id", "")).strip()
        for row in rows
        if str(row.get("risk_level", "")).strip() == "P2"
    }
    errors: list[str] = []
    for requirement in P2_SOURCE_SNAPSHOT_REQUIREMENTS:
        relevant = sorted(p2_cases.intersection(requirement["case_ids"]))
        if not relevant:
            continue
        missing = [
            marker
            for marker in requirement["markers"]
            if " ".join(marker.split()) not in normalized_audit_text
        ]
        if missing:
            errors.append(
                "CASE_WEB_VALIDATION_AUDIT.md is missing "
                f"{requirement['label']} P2 source-verification marker(s) "
                f"for {relevant}: {missing}"
            )
    return errors


def validate_existing_results_resume_health(
    results_csv: Path = DEFAULT_RESULTS_CSV,
    *,
    require_clean_missing_results: bool = False,
) -> tuple[list[str], list[str]]:
    """Fail early if an existing full-sweep CSV is unsafe to resume from.

    Missing ``results.csv`` is fine before a new server launch.  A non-empty
    partial CSV may be old if the server job stopped and is about to be
    resumed, so staleness is left as a warning.  Corrupt schemas, duplicate
    cells, symlinks, bad status/error contracts, source-hash drift, and stale
    header-only files remain launch blockers through the runtime health gate.
    """
    try:
        from scripts.check_sweep_runtime_health import check_runtime_health
    except Exception as exc:
        return [
            "could not import runtime health gate before launch: "
            f"{type(exc).__name__}: {exc}"
        ], []

    has_resume_csv = results_csv.exists() or results_csv.is_symlink()
    rc, errors, warnings = check_runtime_health(
        results_csv,
        run_config_json=(
            results_csv.with_name("run_config.json") if has_resume_csv else None
        ),
        require_run_config=has_resume_csv,
        allow_missing=True,
        fail_on_stale=False,
    )
    if not results_csv.exists():
        stale_derived = [
            sibling.name
            for sibling in (
                results_csv.with_name("run_config.json"),
                results_csv.with_name("python_environment_pip_freeze.txt"),
                results_csv.with_name("RUN_MANIFEST.json"),
                results_csv.with_name("sweep_integrity_report.md"),
                results_csv.with_name("runtime_health_report.txt"),
                results_csv.with_name("report"),
                results_csv.with_name("analysis"),
            )
            if sibling.exists()
        ]
        if stale_derived:
            message = (
                "results.csv is missing but stale derived full-sweep artifact(s) "
                f"are present next to it: {stale_derived}. Start the server run "
                "from a clean output directory, or move these artifacts aside so "
                "operators cannot mistake old reports/manifests for the new "
                "headline archive."
            )
            if require_clean_missing_results:
                errors = [
                    "clean full-sweep output directory required before first "
                    f"server launch: {message}"
                ]
                return errors, warnings
            warnings.append(message)
    if rc == 0:
        return [], warnings
    return errors or ["existing results.csv failed runtime health"], warnings


def _resume_health_warning_level(message: str) -> str:
    """Classify operator-facing resume-health messages."""
    if message.startswith("results.csv is missing:"):
        return "INFO"
    return "WARN"


def build_launch_checks(
    *,
    python: str = sys.executable,
    include_case_physics: bool = True,
    include_pipeline_dry_run: bool = True,
    include_publication_preflight: bool = True,
) -> list[LaunchCheck]:
    checks = [
        LaunchCheck(
            "case-source-fingerprints",
            [python, _script("update_case_validation_sources.py")],
        ),
        LaunchCheck(
            "case-note-fingerprints",
            [python, _script("update_case_validation_note_sources.py")],
        ),
        LaunchCheck(
            "case-validation-ledger",
            [python, "-m", "pytest", "tests/test_case_validation_status.py", "-q"],
        ),
        LaunchCheck(
            "nature-pool-cardinality",
            [
                python,
                _script("check_optional_dependencies.py"),
                "--pool",
                "nature",
                "--require-botorch",
                "--expected-methods",
                "20",
                "--require-formal-case-backends",
                "--require-runtime-monitoring",
            ],
        ),
        LaunchCheck(
            "runtime-row-contracts",
            [
                python,
                "-m",
                "pytest",
                "benchmark/tests/test_runtime_contracts.py",
                "-q",
            ],
        ),
        LaunchCheck(
            "sweep-integrity-audit-contract",
            [
                python,
                "-m",
                "pytest",
                "benchmark/tests/test_sweep_integrity_audit.py",
                "-q",
            ],
        ),
        LaunchCheck(
            "server-operator-contracts",
            [
                python,
                "-m",
                "pytest",
                *server_operator_contract_nodeids(),
                "-q",
            ],
        ),
    ]
    if include_case_physics:
        checks.append(
            LaunchCheck(
                "case-physics-tests",
                [python, _script("run_case_physics_tests.py"), "-q"],
            )
        )
    checks.append(
        LaunchCheck(
            "pipeline-stage1-preflight",
            [
                python,
                _script("run_pipeline.py"),
                "--profile",
                "full",
                "--to-stage",
                "1",
                "--confirm-full-run",
            ],
        )
    )
    if include_pipeline_dry_run:
        checks.append(
            LaunchCheck(
                "pipeline-full-dry-run",
                [
                    python,
                    _script("run_pipeline.py"),
                    "--profile",
                    "full",
                    "--confirm-full-run",
                    "--dry-run",
                ],
            )
        )
    if include_publication_preflight:
        checks.append(
            LaunchCheck(
                "publication-launch-preflight",
                [
                    python,
                    _script("audit_publication_readiness.py"),
                    "--launch-preflight",
                    "--output",
                    str(ROOT / "docs" / "PUBLICATION_LAUNCH_PREFLIGHT.md"),
                ],
            )
        )
    return checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for each preflight command.",
    )
    parser.add_argument(
        "--skip-case-physics",
        action="store_true",
        help="Skip targeted case-physics tests if they were just run separately.",
    )
    parser.add_argument(
        "--skip-pipeline-dry-run",
        action="store_true",
        help="Skip the full-pipeline command-graph dry-run.",
    )
    parser.add_argument(
        "--skip-publication-preflight",
        action="store_true",
        help="Skip the pre-full-sweep publication-readiness roll-up.",
    )
    parser.add_argument(
        "--require-clean-sweep-dir",
        action="store_true",
        help=(
            "Fail if benchmark/results/full_benchmark has no results.csv but "
            "still contains derived full-sweep artifacts such as an old "
            "RUN_MANIFEST.json or report directory.  Use for the final "
            "server first-launch check after intentionally moving stale "
            "artifacts aside."
        ),
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run all checks even after an earlier command fails.",
    )
    parser.add_argument(
        "--check-timeout-s",
        type=float,
        default=None,
        help=(
            "Maximum seconds allowed for each executed subprocess check. "
            "Defaults to DOE_BENCHMARK_LAUNCH_CHECK_TIMEOUT_S or 1800."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned checks without executing them.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Alias for --dry-run; prints the planned no-sweep checks.",
    )
    return parser.parse_args(argv)


def run_checks(
    checks: Sequence[LaunchCheck],
    *,
    dry_run: bool = False,
    continue_on_failure: bool = False,
    require_clean_sweep_dir: bool = False,
    check_timeout_s: float = DEFAULT_CHECK_TIMEOUT_S,
) -> int:
    safety_errors = validate_no_sweep_commands(checks)
    safety_errors.extend(validate_pipeline_dry_run_stage_graph(checks))
    safety_errors.extend(validate_case_web_validation_audit_coverage())
    safety_errors.extend(validate_p2_source_verification_snapshots())
    shell_errors: list[str] = []
    shell_messages: list[str] = []
    resource_messages: list[str] = []
    resume_warnings: list[str] = []
    if not dry_run:
        shell_errors, shell_messages = validate_shell_script_syntax()
        safety_errors.extend(shell_errors)
        resource_errors, resource_messages = validate_server_resource_capacity()
        safety_errors.extend(resource_errors)
        resume_errors, resume_warnings = validate_existing_results_resume_health(
            require_clean_missing_results=require_clean_sweep_dir,
        )
        safety_errors.extend(
            f"existing results.csv is not safe to resume: {error}"
            for error in resume_errors
        )
    if safety_errors:
        for error in safety_errors:
            print(f"[safety] FAIL {error}")
        print("[result] FAIL - server launch preflight safety checks failed.")
        return 1
    for message in shell_messages:
        print(f"[shell-syntax] {message}")
    for message in resource_messages:
        print(f"[server-resource] {message}")
    for warning in resume_warnings:
        level = _resume_health_warning_level(warning)
        print(f"[resume-health] {level} {warning}")

    failures: list[tuple[str, int]] = []
    for check in checks:
        text = command_to_text(check.command)
        print(f"[{check.label}] {text}", flush=True)
        if dry_run:
            continue
        started = time.monotonic()
        try:
            completed = subprocess.run(
                check.command,
                cwd=ROOT,
                timeout=check_timeout_s,
            )
        except subprocess.TimeoutExpired:
            elapsed_s = time.monotonic() - started
            failures.append((check.label, 124))
            print(
                f"[{check.label}] TIMEOUT after {elapsed_s:.1f}s "
                f"(limit={check_timeout_s:.1f}s)",
                flush=True,
            )
            if not continue_on_failure:
                break
            continue
        elapsed_s = time.monotonic() - started
        if completed.returncode != 0:
            failures.append((check.label, completed.returncode))
            print(
                f"[{check.label}] FAIL exit={completed.returncode} "
                f"elapsed={elapsed_s:.1f}s",
                flush=True,
            )
            if not continue_on_failure:
                break
        else:
            print(f"[{check.label}] PASS elapsed={elapsed_s:.1f}s", flush=True)

    if dry_run:
        print("[result] DRY-RUN - no checks executed and no full sweep launched.")
        return 0
    if failures:
        details = ", ".join(f"{label}:{code}" for label, code in failures)
        print(f"[result] FAIL - server launch preflight failed ({details}).")
        return 1
    print("[result] PASS - ready to launch the server full pipeline.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_timeout_s is None:
        check_timeout_s, timeout_error = _env_positive_float(
            "DOE_BENCHMARK_LAUNCH_CHECK_TIMEOUT_S",
            DEFAULT_CHECK_TIMEOUT_S,
        )
        if timeout_error is not None or check_timeout_s is None:
            print(f"[safety] FAIL {timeout_error}")
            print("[result] FAIL - server launch preflight safety checks failed.")
            return 1
    else:
        check_timeout_s = float(args.check_timeout_s)
        if not math.isfinite(check_timeout_s) or check_timeout_s <= 0:
            print(
                "[safety] FAIL --check-timeout-s must be a positive finite "
                f"number; got {args.check_timeout_s!r}"
            )
            print("[result] FAIL - server launch preflight safety checks failed.")
            return 1
    checks = build_launch_checks(
        python=args.python,
        include_case_physics=not args.skip_case_physics,
        include_pipeline_dry_run=not args.skip_pipeline_dry_run,
        include_publication_preflight=not args.skip_publication_preflight,
    )
    return run_checks(
        checks,
        dry_run=bool(args.dry_run or args.list),
        continue_on_failure=bool(args.continue_on_failure),
        require_clean_sweep_dir=bool(args.require_clean_sweep_dir),
        check_timeout_s=check_timeout_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
