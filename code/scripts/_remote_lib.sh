#!/usr/bin/env bash
# Shared helpers sourced by launch_sweep_remote.sh, launch_sota_robustness.sh,
# launch_mfbo.sh.  Provides core/RAM auto-detection, BLAS thread pinning,
# and the persistence dispatch (nohup / tmux / fg).

# ── Auto-detect host capacity ────────────────────────────────────────────
detect_cores() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif [[ -r /proc/cpuinfo ]]; then
        grep -c '^processor' /proc/cpuinfo
    else
        echo 8
    fi
}

detect_ram_gb() {
    local cgroup_limit_gb
    cgroup_limit_gb="$(detect_cgroup_memory_limit_gb || true)"
    local physical_gb=""
    if [[ -r /proc/meminfo ]]; then
        local kb
        kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
        physical_gb="$((kb / 1024 / 1024))"
    fi
    if [[ -n "${cgroup_limit_gb}" ]]; then
        if [[ -z "${physical_gb}" || "${cgroup_limit_gb}" -lt "${physical_gb}" ]]; then
            echo "${cgroup_limit_gb}"
            return
        fi
    fi
    if [[ -n "${physical_gb}" ]]; then
        echo "${physical_gb}"
        return
    fi
    echo 16
}

emit_cgroup_memory_limit_candidates() {
    local root="${CGROUP_ROOT:-/sys/fs/cgroup}"
    local proc_cgroup="${CGROUP_PROC_SELF:-/proc/self/cgroup}"
    local -A seen=()

    add_candidate() {
        local path="$1"
        if [[ -z "${seen[${path}]:-}" ]]; then
            seen["${path}"]=1
            echo "${path}"
        fi
    }

    if [[ -r "${proc_cgroup}" ]]; then
        local hierarchy controllers rel
        while IFS=: read -r hierarchy controllers rel; do
            [[ -z "${rel:-}" ]] && continue
            rel="${rel#/}"
            rel="${rel%/}"
            local current="${rel}"
            while true; do
                local suffix=""
                [[ -n "${current}" ]] && suffix="/${current}"
                if [[ -z "${controllers}" ]]; then
                    add_candidate "${root}${suffix}/memory.max"
                elif [[ ",${controllers}," == *",memory,"* ]]; then
                    add_candidate "${root}/memory${suffix}/memory.limit_in_bytes"
                    add_candidate "${root}${suffix}/memory.limit_in_bytes"
                fi
                [[ -z "${current}" ]] && break
                if [[ "${current}" == */* ]]; then
                    current="${current%/*}"
                else
                    current=""
                fi
            done
        done < "${proc_cgroup}"
    fi

    add_candidate "${root}/memory.max"
    add_candidate "${root}/memory/memory.limit_in_bytes"
}

detect_cgroup_memory_limit_gb() {
    local best_bytes=""
    local path raw
    while IFS= read -r path; do
        [[ -r "${path}" ]] || continue
        raw="$(tr -d '[:space:]' < "${path}")"
        [[ -z "${raw}" || "${raw}" == "max" ]] && continue
        [[ "${raw}" =~ ^[0-9]+$ ]] || continue
        if (( raw > 0 && raw < 1152921504606846976 )); then
            if [[ -z "${best_bytes}" || "${raw}" -lt "${best_bytes}" ]]; then
                best_bytes="${raw}"
            fi
        fi
    done < <(emit_cgroup_memory_limit_candidates)
    if [[ -n "${best_bytes}" ]]; then
        local gb=$((best_bytes / 1024 / 1024 / 1024))
        [[ "${gb}" -lt 1 ]] && gb=1
        echo "${gb}"
    fi
}

# Pin BLAS threads to 1 so they don't oversubscribe the joblib pool.
pin_blas_threads() {
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
}

require_positive_int_env() {
    local name="$1"
    local value="${!name:-}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer; got '${value:-<unset>}'." >&2
        exit 2
    fi
}

require_positive_number_env() {
    local name="$1"
    local value="${!name:-}"
    if [[ ! "${value}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] \
        || ! awk -v v="${value}" 'BEGIN { exit !(v > 0) }'; then
        echo "ERROR: ${name} must be a positive number; got '${value:-<unset>}'." >&2
        exit 2
    fi
}

validate_remote_launch_tunables() {
    require_positive_int_env N_JOBS
    require_positive_int_env BO_N_JOBS
    require_positive_number_env MEM_CAP_GB
    require_positive_int_env WALL_H
    require_positive_int_env TIMEOUT_S
    if [[ -n "${BO_TIMEOUT_S:-}" ]]; then
        require_positive_int_env BO_TIMEOUT_S
    fi
    require_positive_int_env REF_TIMEOUT_S
    require_positive_int_env HEARTBEAT_S
    require_positive_int_env N_REPEATS
}

validate_persistence_mode() {
    local mode="${1:-}"
    case "${mode}" in
        nohup|tmux|fg)
            ;;
        *)
            echo "ERROR: persistence mode must be one of: nohup, tmux, fg; got '${mode:-<unset>}'." >&2
            exit 2
            ;;
    esac
}

refuse_live_pid_file() {
    local pid_file="$1"
    if [[ ! -f "${pid_file}" ]]; then
        return 0
    fi
    local old_pid
    old_pid="$(tr -d '[:space:]' < "${pid_file}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
        echo "ERROR: existing live process from ${pid_file} (PID ${old_pid}); refusing to start a second writer." >&2
        echo "  Stop it first, or remove the stale pid file only after confirming the process is gone." >&2
        exit 1
    fi
    echo "WARNING: stale or malformed pid file will be replaced: ${pid_file}" >&2
}

# Compute autotuned defaults for (N_JOBS, BO_N_JOBS, MEM_CAP_GB) and
# write them into the global env, **only if the env var is unset**
# (so explicit overrides via N_JOBS=... still win).
auto_tune_defaults() {
    local cores ram_gb ram_reserve ram_for_sweep
    cores="$(detect_cores)"
    ram_gb="$(detect_ram_gb)"
    ram_reserve=$(( ram_gb / 10 ))
    ram_for_sweep=$(( ram_gb - ram_reserve ))

    local n_jobs_auto bo_n_jobs_auto
    n_jobs_auto=$(( cores * 3 / 4 ))
    [[ "${n_jobs_auto}" -lt 4 ]] && n_jobs_auto=4

    bo_n_jobs_auto=$(( cores / 4 ))
    [[ "${bo_n_jobs_auto}" -lt 1 ]] && bo_n_jobs_auto=1
    local ram_limited=$(( ram_for_sweep * 2 / 3 ))
    [[ "${bo_n_jobs_auto}" -gt "${ram_limited}" ]] && bo_n_jobs_auto="${ram_limited}"
    [[ "${bo_n_jobs_auto}" -lt 1 ]] && bo_n_jobs_auto=1

    export CORES_DETECTED="${cores}"
    export RAM_DETECTED_GB="${ram_gb}"
    export RAM_FOR_SWEEP_GB="${ram_for_sweep}"
    export N_JOBS_AUTO="${n_jobs_auto}"
    export BO_N_JOBS_AUTO="${bo_n_jobs_auto}"

    # Apply only if unset
    : "${N_JOBS:=${n_jobs_auto}}"
    : "${BO_N_JOBS:=${bo_n_jobs_auto}}"
    : "${MEM_CAP_GB:=${ram_for_sweep}}"
    export N_JOBS BO_N_JOBS MEM_CAP_GB
}

# Run a python command under one of the supported persistence modes.
#   $1: persistence mode (nohup | tmux | fg)
#   $2: tmux session name
#   $3: log file (stdout)
#   $4: error log file (stderr)
#   $5: sweep PID file (only used for nohup mode)
#   $6+ : the actual command and its arguments (passed verbatim to exec)
dispatch_persistent_run() {
    local mode="$1"; shift
    local session="$1"; shift
    local log_out="$1"; shift
    local log_err="$1"; shift
    local pid_file="$1"; shift
    # remaining "$@" is the python command
    validate_persistence_mode "${mode}"
    refuse_live_pid_file "${pid_file}"

    case "${mode}" in
        fg)
            exec "$@"
            ;;
        tmux)
            if ! command -v tmux >/dev/null 2>&1; then
                echo "ERROR: tmux not installed; fall back to nohup." >&2
                exit 1
            fi
            if tmux has-session -t "${session}" 2>/dev/null; then
                echo "ERROR: tmux session '${session}' already exists." >&2
                echo "  Attach: tmux attach -t ${session}" >&2
                exit 1
            fi
            local cmd_str log_out_q
            cmd_str="$(printf '%q ' "$@")"
            log_out_q="$(printf '%q' "${log_out}")"
            tmux new-session -d -s "${session}" -c "${PWD}" \
                "exec bash -lc 'set -o pipefail; ${cmd_str} 2>&1 | tee -a ${log_out_q}; run_rc=\${PIPESTATUS[0]}; echo; echo \"[finished, exit=\${run_rc}]\"; exec bash'"
            echo "Started tmux session: ${session}"
            echo "  Attach: tmux attach -t ${session}"
            echo "  Tail  : tail -f ${log_out}"
            ;;
        nohup)
            nohup "$@" >>"${log_out}" 2>>"${log_err}" &
            local sp=$!
            disown "${sp}" 2>/dev/null || true
            sleep 2
            if ! kill -0 "${sp}" 2>/dev/null; then
                echo "ERROR: process died within 2s.  Tail of ${log_err}:" >&2
                tail -40 "${log_err}" >&2 || true
                exit 1
            fi
            echo "${sp}" > "${pid_file}"
            echo "Started in background (nohup)."
            echo "  PID   : ${sp}  -> ${pid_file}"
            echo "  Tail  : tail -f ${log_out}"
            echo "  Stop  : kill \$(cat ${pid_file})"
            ;;
    esac
}
