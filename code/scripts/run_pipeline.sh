#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/run_pipeline.sh
# =========================
# One-click DOE-Benchmark pipeline launcher (Linux / macOS / WSL).
#
# Thin wrapper around scripts/run_pipeline.py — locates a Python
# interpreter, pins BLAS thread caps, then forwards every remaining
# argument to the Python orchestrator.  All real logic lives in the
# Python file.  The Windows .ps1 wrapper is intentionally smoke/preflight-only
# for local workstations; this bash wrapper is the formal server entry point.
#
# Usage
# -----
#   bash scripts/run_pipeline.sh                       # confirmed full sweep
#   bash scripts/run_pipeline.sh --profile smoke       # ~5 min sanity
#   bash scripts/run_pipeline.sh --from-stage 3        # skip the sweep
#   bash scripts/run_pipeline.sh --dry-run             # print, don't run
#   bash scripts/run_pipeline.sh --with-sota --with-mfbo
#                                                      # add Stage 7
#
# Persistence
# -----------
# This wrapper runs in the foreground.  For the long publication sweep
# wrap it in tmux/screen/nohup yourself:
#
#   tmux new -s doe 'bash scripts/run_pipeline.sh'
#   nohup bash scripts/run_pipeline.sh > logs/pipeline.out 2>&1 &
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# ── Locate Python ────────────────────────────────────────────────────────
# Order: $PYTHON, repo-local .venv, python3 on PATH, python on PATH.
resolve_python() {
    if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
        echo "${PYTHON}"
        return
    fi
    local venv_py="${REPO_ROOT}/.venv/bin/python"
    if [[ -x "${venv_py}" ]]; then
        echo "${venv_py}"
        return
    fi
    if command -v python3 &> /dev/null; then
        command -v python3
        return
    fi
    if command -v python &> /dev/null; then
        command -v python
        return
    fi
    echo "ERROR: cannot find a Python interpreter (set \$PYTHON or activate a venv)" >&2
    exit 127
}
PYTHON="$(resolve_python)"

# ── Pin BLAS / OpenMP threads BEFORE Python imports anything ────────────
# Joblib parallelises at the process level; if MKL / OpenBLAS / OMP also
# multi-thread we get nproc × nproc threads and the kernel scheduler
# thrashes.  Force them to 1 so inherited cluster/module settings cannot
# silently over-subscribe the formal run; joblib's --n-jobs is the only knob.
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
BLIS_NUM_THREADS=1
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
       NUMEXPR_NUM_THREADS BLIS_NUM_THREADS

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# ── Resolve and validate the forwarded orchestrator arguments ───────────
args=("$@")
has_profile=0
profile_value=""
has_confirm=0
for ((i = 0; i < ${#args[@]}; i++)); do
    arg="${args[$i]}"
    if [[ "${arg}" == "--profile" ]]; then
        has_profile=1
        profile_value="${args[$((i + 1))]:-}"
    elif [[ "${arg}" == --profile=* ]]; then
        has_profile=1
        profile_value="${arg#--profile=}"
    fi
    [[ "${arg}" == "--confirm-full-run" ]] && has_confirm=1
done
if [[ "${has_profile}" -eq 0 ]]; then
    args=("--profile" "full" "${args[@]}")
    profile_value="full"
fi
if [[ "${profile_value}" == "full" && "${has_confirm}" -eq 0 ]]; then
    args=("--confirm-full-run" "${args[@]}")
fi

has_cli_flag() {
    local flag="$1"
    local arg
    for arg in "${args[@]}"; do
        if [[ "${arg}" == "${flag}" || "${arg}" == "${flag}="* ]]; then
            return 0
        fi
    done
    return 1
}

require_positive_int_value() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer; got '${value:-<unset>}'." >&2
        exit 2
    fi
}

require_positive_number_value() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] \
        || ! awk -v v="${value}" 'BEGIN { exit !(v > 0) }'; then
        echo "ERROR: ${name} must be a positive number; got '${value:-<unset>}'." >&2
        exit 2
    fi
}

inject_env_int_arg() {
    local env_name="$1"
    local flag="$2"
    local value="${!env_name:-}"
    [[ -z "${value}" ]] && return 0
    if has_cli_flag "${flag}"; then
        return 0
    fi
    require_positive_int_value "${env_name}" "${value}"
    args+=("${flag}" "${value}")
}

inject_env_number_arg() {
    local env_name="$1"
    local flag="$2"
    local value="${!env_name:-}"
    [[ -z "${value}" ]] && return 0
    if has_cli_flag "${flag}"; then
        return 0
    fi
    require_positive_number_value "${env_name}" "${value}"
    args+=("${flag}" "${value}")
}

# Server operators can tune the formal pipeline without editing command
# lines.  Explicit CLI flags still win, and all injected values are recorded
# by the Python orchestrator in command history and run_config.json.
inject_env_int_arg N_JOBS --n-jobs
inject_env_int_arg BO_N_JOBS --bo-n-jobs
inject_env_number_arg MEM_CAP_GB --mem-cap-gb
inject_env_number_arg TIMEOUT_S --timeout-s
inject_env_number_arg BO_TIMEOUT_S --bo-timeout-s
inject_env_number_arg REF_TIMEOUT_S --ref-timeout-s
inject_env_number_arg HEARTBEAT_S --heartbeat-s
inject_env_number_arg MAX_WALL_CLOCK_H --max-wall-clock-h

# ── Guard the formal server entry point with the launch preflight ───────
should_run_launch_preflight() {
    [[ "${profile_value}" == "full" ]] || return 1
    has_cli_flag --dry-run && return 1
    has_cli_flag --list && return 1
    has_cli_flag --help && return 1
    has_cli_flag -h && return 1
    [[ "${DOE_BENCHMARK_SKIP_LAUNCH_PREFLIGHT:-}" == "1" ]] && return 1
    return 0
}

if should_run_launch_preflight; then
    PREFLIGHT_ARGS=()
    # Fresh full-profile launches should fail closed if an old Stage-2
    # run_config/report/manifest is sitting next to a missing results.csv.
    # A valid partial results.csv is still treated as a resume candidate by
    # check_server_launch_ready.py; this only blocks stale first-launch
    # derived artifacts.  Set DOE_BENCHMARK_ALLOW_STALE_DERIVED=1 only after a
    # human has intentionally chosen to keep those files in place.
    if [[ "${DOE_BENCHMARK_ALLOW_STALE_DERIVED:-}" != "1" ]] \
        || [[ "${DOE_BENCHMARK_REQUIRE_CLEAN_SWEEP_DIR:-}" == "1" ]]; then
        PREFLIGHT_ARGS+=(--require-clean-sweep-dir)
    fi
    printf '\n[launch-preflight] python scripts/check_server_launch_ready.py %s\n' "${PREFLIGHT_ARGS[*]:-}"
    "${PYTHON}" "${SCRIPT_DIR}/check_server_launch_ready.py" "${PREFLIGHT_ARGS[@]}"
fi

# ── Banner so the operator can confirm before the long run ──────────────
printf -v FINAL_ARGS '%q ' "${args[@]}"
cat <<BANNER

==============================================================================
 DOE-Benchmark one-click launcher (Linux / bash)
==============================================================================
  repo root : ${REPO_ROOT}
  python    : ${PYTHON}
  args      : ${FINAL_ARGS:-<none>}
  threads   : OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS}
==============================================================================

BANNER

# ── Forward every argument to the Python orchestrator ───────────────────
exec "${PYTHON}" "${SCRIPT_DIR}/run_pipeline.py" "${args[@]}"
