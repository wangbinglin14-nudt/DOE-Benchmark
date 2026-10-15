#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DOE-Benchmark — remote sweep health + progress monitor (Linux/macOS).
#
# Wraps the cross-platform Python progress checker with a quick view of:
#   - the sweep PID (if alive)
#   - the last 30 lines of the Stage-2 run.log or latest formal pipeline log
#   - per-Phase completion percentage
#   - resource usage (CPU, RSS, IO)
#   - optional runtime health warnings via MEM_CAP_GB / DISK_WARNING_GB
#   - optional stale-writer failure mode via STALE_AFTER_MIN / FAIL_ON_STALE
#   - optional current-environment drift check via VALIDATE_CURRENT_ENV
#
# Usage:
#   bash scripts/check_progress_remote.sh             # one-shot snapshot
#   bash scripts/check_progress_remote.sh -w          # watch mode (refresh 30 s)
#   bash scripts/check_progress_remote.sh -t 5        # tail last 5 log lines
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WATCH=0
TAIL_N=30
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--watch)   WATCH=1; shift ;;
        -t|--tail)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ || "$2" -lt 1 ]]; then
                echo "ERROR: --tail requires a positive integer line count." >&2
                exit 2
            fi
            TAIL_N="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '1,18p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
resolve_python() {
    if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
        echo "${PYTHON}"
        return
    fi
    local venv_py="${VENV_DIR}/bin/python"
    if [[ -x "${venv_py}" ]]; then
        echo "${venv_py}"
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    echo "ERROR: cannot find a Python interpreter (set \$PYTHON or activate a venv)" >&2
    exit 127
}
PYTHON_CMD="$(resolve_python)"

OUT_DIR="${REPO_ROOT}/benchmark/results/full_benchmark"
LOG_OUT="${OUT_DIR}/run.log"
PID_FILE="${OUT_DIR}/sweep.pid"
LOGS_DIR="${REPO_ROOT}/logs"

latest_pipeline_log() {
    local latest=""
    latest="$(ls -t "${LOGS_DIR}"/pipeline_*.log 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest}" && -f "${latest}" ]]; then
        echo "${latest}"
    fi
}

find_active_sweep_processes() {
    if ! command -v pgrep >/dev/null 2>&1; then
        return 0
    fi
    pgrep -af "scripts/run_pipeline.py|benchmark/scripts/run_full_benchmark.py|scripts/run_pipeline.sh" 2>/dev/null \
        | grep -v "check_progress_remote.sh" || true
}

snapshot() {
    local HEALTH_RC=0
    local PROCESS_DEAD=0
    local SWEEP_COMPLETE=0
    local DEAD_INCOMPLETE_RC=0
    clear 2>/dev/null || true
    printf '\n========================================================================\n'
    printf ' DOE-Benchmark sweep snapshot — %s\n' "$(date -Iseconds)"
    printf '========================================================================\n\n'
    printf '  Python monitor : %s\n' "${PYTHON_CMD}"
    printf '\n'

    # ── Process state ─────────────────────────────────────────────────
    if [[ -f "${PID_FILE}" ]]; then
        PID="$(tr -d '[:space:]' < "${PID_FILE}")"
        if [[ ! "${PID}" =~ ^[0-9]+$ ]]; then
            printf '  Sweep process : unknown (malformed sweep.pid: %s)\n' "${PID_FILE}"
        elif kill -0 "${PID}" 2>/dev/null; then
            STATE="ALIVE"
            CPU_RSS=$(ps -p "${PID}" -o pid,pcpu,pmem,rss,etime,comm 2>/dev/null | tail -n +2 || echo "")
            printf '  Sweep process : %s (PID %s)\n' "${STATE}" "${PID}"
            printf '  ps           : %s\n' "${CPU_RSS}"
        else
            PROCESS_DEAD=1
            printf '  Sweep process : DEAD (PID %s no longer running)\n' "${PID}"
        fi
    else
        PROCESS_MATCHES="$(find_active_sweep_processes)"
        if [[ -n "${PROCESS_MATCHES}" ]]; then
            printf '  Sweep process : active process(es) found without sweep.pid\n'
            printf '%s\n' "${PROCESS_MATCHES}" | head -n 5 | sed 's/^/  pgrep       : /'
            FIRST_PID="$(printf '%s\n' "${PROCESS_MATCHES}" | head -n 1 | awk '{print $1}')"
            if [[ "${FIRST_PID}" =~ ^[0-9]+$ ]]; then
                CPU_RSS=$(ps -p "${FIRST_PID}" -o pid,pcpu,pmem,rss,etime,comm 2>/dev/null | tail -n +2 || echo "")
                printf '  ps(first)   : %s\n' "${CPU_RSS}"
            fi
        else
            printf '  Sweep process : (no sweep.pid found and no active formal pipeline process matched - run run_pipeline.sh, or launch_sweep_remote.sh for a low-level Stage 2 restart)\n'
        fi
    fi
    printf '\n'

    # ── Python progress check (reuses the cross-platform script) ─────
    if [[ -f scripts/_check_sweep_progress.py ]]; then
        "${PYTHON_CMD}" scripts/_check_sweep_progress.py || true
    fi
    printf '\n'

    # ── Runtime health gate (read-only; does not launch/resume) ──────
    if [[ -f scripts/check_sweep_runtime_health.py ]]; then
        printf '──── runtime health gate ────\n'
        HEALTH_ARGS=(--allow-missing)
        RUN_CONFIG="benchmark/results/full_benchmark/run_config.json"
        if [[ -f "${RUN_CONFIG}" ]]; then
            HEALTH_ARGS+=(--run-config-json "${RUN_CONFIG}" --require-run-config)
        fi
        if [[ -n "${VALIDATE_CURRENT_ENV:-}" ]]; then
            HEALTH_ARGS+=(--validate-current-environment)
        fi
        if [[ -n "${MEM_CAP_GB:-}" ]]; then
            HEALTH_ARGS+=(--mem-cap-gb "${MEM_CAP_GB}")
        fi
        if [[ -n "${DISK_WARNING_GB:-}" ]]; then
            HEALTH_ARGS+=(--disk-warning-gb "${DISK_WARNING_GB}")
        fi
        if [[ -n "${STALE_AFTER_MIN:-}" ]]; then
            HEALTH_ARGS+=(--stale-after-min "${STALE_AFTER_MIN}")
        fi
        if [[ -n "${FAIL_ON_STALE:-}" ]]; then
            HEALTH_ARGS+=(--fail-on-stale)
        fi
        set +e
        "${PYTHON_CMD}" scripts/check_sweep_runtime_health.py "${HEALTH_ARGS[@]}"
        HEALTH_RC=$?
        set -e
        if [[ "${HEALTH_RC}" -ne 0 ]]; then
            printf '[monitor] runtime health gate failed with exit=%s; inspect results.csv before resuming or trusting progress.\n' "${HEALTH_RC}"
        fi
    fi
    printf '\n'

    # ── Dead-process completion gate ─────────────────────────────────
    if [[ "${PROCESS_DEAD}" -eq 1 ]]; then
        printf '──── dead-process completion gate ────\n'
        set +e
        "${PYTHON_CMD}" - <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._check_sweep_progress import PREREG_TARGET, summarize_progress

csv_path = Path("benchmark/results/full_benchmark/results.csv")
if not csv_path.exists():
    print(f"completion check: results.csv missing at {csv_path}")
    raise SystemExit(1)

try:
    df = pd.read_csv(csv_path)
except Exception as exc:
    print(f"completion check: could not parse results.csv: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

summary = summarize_progress(df)
unique_cells = int(summary["protocol_unique_cells"])
target_cells = int(summary["target_cells"])
print(f"completion check: {unique_cells:,} / {target_cells:,} unique protocol cells")
raise SystemExit(0 if unique_cells >= PREREG_TARGET else 1)
PY
        SWEEP_COMPLETE=$?
        set -e
        if [[ "${SWEEP_COMPLETE}" -ne 0 ]]; then
            DEAD_INCOMPLETE_RC=3
            printf '[monitor] sweep.pid is dead and headline sweep is incomplete; investigate logs before resuming.\n'
        fi
    fi
    printf '\n'

    # ── Tail of run log ──────────────────────────────────────────────
    if [[ -f "${LOG_OUT}" ]]; then
        printf '──── tail of run.log (last %s lines) ────\n' "${TAIL_N}"
        tail -n "${TAIL_N}" "${LOG_OUT}"
    else
        PIPELINE_LOG="$(latest_pipeline_log)"
        if [[ -n "${PIPELINE_LOG}" ]]; then
            printf '──── tail of latest pipeline log: %s (last %s lines) ────\n' "${PIPELINE_LOG}" "${TAIL_N}"
            tail -n "${TAIL_N}" "${PIPELINE_LOG}"
        else
            printf '(no run.log or logs/pipeline_*.log yet — sweep has not started)\n'
        fi
    fi
    printf '\n'

    # ── Disk usage of results dir ────────────────────────────────────
    if [[ -d "${OUT_DIR}" ]]; then
        printf '──── disk usage ────\n'
        du -sh "${OUT_DIR}" 2>/dev/null || true
        if [[ -f "${OUT_DIR}/results.csv" ]]; then
            ROWS=$(wc -l < "${OUT_DIR}/results.csv")
            printf '  results.csv rows : %s (incl. header)\n' "${ROWS}"
        fi
    fi
    printf '\n'
    if [[ "${DEAD_INCOMPLETE_RC}" -ne 0 ]]; then
        return "${DEAD_INCOMPLETE_RC}"
    fi
    return "${HEALTH_RC}"
}

if [[ "${WATCH}" -eq 1 ]]; then
    while true; do
        snapshot || true
        printf 'Refreshing in 30 s — Ctrl+C to exit\n'
        sleep 30
    done
else
    set +e
    snapshot
    RC=$?
    set -e
    exit "${RC}"
fi
