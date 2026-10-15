#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DOE-Benchmark — single-phase combined sweep on a high-spec Linux server.
#
# This launcher targets ≥ 32-core / ≥ 64 GB Linux hosts where BoTorch
# threading-backend can run safely with N_JOBS ≥ 8 (no OOM risk).
# It auto-detects the host's core count and total RAM (`nproc` + `/proc/meminfo`)
# and picks defaults that roughly saturate both:
#
#                 cores   N_JOBS  BO_N_JOBS  MEM_CAP_GB
#     32-core/64 GB host    16      4         56
#     48-core/96 GB host    32      6         84
#     64-core/200 GB host   48     12        160
#
# It runs the FULL `nature_pool` (20 methods) on all 34 cases in a single
# sweep — Linux's fork() avoids the BoTorch-loky segfault that forced
# the Windows two-phase split.
#
# Persistence: by default the run is detached via `nohup` so that
# closing the SSH session does not kill the sweep.  For interactive
# attachment use `tmux` (recommended; see PERSISTENCE_MODE below).
#
# Usage:
#   bash scripts/launch_sweep_remote.sh                    # nohup (default)
#   bash scripts/launch_sweep_remote.sh tmux               # tmux session
#   N_JOBS=24 WALL_H=48 bash scripts/launch_sweep_remote.sh
#
# Re-run anytime with --skip-existing semantics — already-completed cells
# in results.csv are skipped automatically.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

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

CORES_DETECTED="$(detect_cores)"
RAM_DETECTED_GB="$(detect_ram_gb)"

# Tier the defaults by detected capacity.  Reserve ~10 % of RAM for OS +
# CSV streaming + reference-set tensors; the rest goes into MEM_CAP_GB.
RAM_RESERVE_GB=$(( RAM_DETECTED_GB / 10 ))
RAM_FOR_SWEEP_GB=$(( RAM_DETECTED_GB - RAM_RESERVE_GB ))

# Light-method (loky) parallelism: ~75 % of cores, leaving headroom for
# BoTorch threading and the heartbeat / dispatcher threads.
N_JOBS_AUTO=$(( CORES_DETECTED * 3 / 4 ))
[[ "${N_JOBS_AUTO}" -lt 4 ]] && N_JOBS_AUTO=4

# Heavy-method (SAASBO / threading) cap.  Each instance peaks at ~1.5 GB
# and benefits from BLAS-parallel ops, so we go up to cores/4 but ensure
# the worst-case total stays under MEM_CAP_GB / 2.  On a 200 GB host
# this gives 12 simultaneous SAASBO instances; on 64 GB it gives 4.
BO_N_JOBS_AUTO=$(( CORES_DETECTED / 4 ))
[[ "${BO_N_JOBS_AUTO}" -lt 1 ]] && BO_N_JOBS_AUTO=1
# Cap by available RAM (≤ MEM_CAP_GB / 2 / 1.5 GB-per-SAASBO)
RAM_LIMITED_BO=$(( RAM_FOR_SWEEP_GB * 2 / 3 ))   # 2/3 of usable RAM
[[ "${BO_N_JOBS_AUTO}" -gt "${RAM_LIMITED_BO}" ]] && BO_N_JOBS_AUTO="${RAM_LIMITED_BO}"
[[ "${BO_N_JOBS_AUTO}" -lt 1 ]] && BO_N_JOBS_AUTO=1

# ── Tunables (env override) ─────────────────────────────────────────────
N_JOBS="${N_JOBS:-${N_JOBS_AUTO}}"
BO_N_JOBS="${BO_N_JOBS:-${BO_N_JOBS_AUTO}}"
MEM_CAP_GB="${MEM_CAP_GB:-${RAM_FOR_SWEEP_GB}}"
WALL_H="${WALL_H:-72}"                 # wall-clock cap in hours
TIMEOUT_S="${TIMEOUT_S:-600}"          # per-cell soft timeout for sklearn (10 min)
BO_TIMEOUT_S="${BO_TIMEOUT_S:-1800}"   # per-cell soft timeout for BoTorch (30 min)
REF_TIMEOUT_S="${REF_TIMEOUT_S:-600}"  # reference-set construction timeout (10 min)
HEARTBEAT_S="${HEARTBEAT_S:-60}"       # heartbeat interval (1 min)
N_REPEATS="${N_REPEATS:-30}"           # number of repeats per cell
PERSISTENCE_MODE="${1:-nohup}"         # "nohup" | "tmux" | "fg"

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
    require_positive_int_env BO_TIMEOUT_S
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

validate_remote_launch_tunables
validate_persistence_mode "${PERSISTENCE_MODE}"

# ── Resolve paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
if [[ -d "${VENV_DIR}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi
PYTHON="$(command -v python)"
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: no python in PATH.  Did you run scripts/setup_remote.sh first?" >&2
    exit 1
fi

OUT_DIR="${REPO_ROOT}/benchmark/results/full_benchmark"
mkdir -p "${OUT_DIR}"

# ── 34-case full pre-registration set (every d in doe_testbed) ──────────
# Per paper/preregistration.md §3: ALL 34 cases of doe_testbed are
# required for the headline 30-repeat sweep.  Deferring high-dim
# cases violates the freeze; do NOT trim this list without an
# amendment in paper/preregistration_amendments.md.
read -r -d '' CASES_ALL <<'EOF' || true
tuned_mass_damper_3d
airfoil_stall_2d fin_heat_sink_4d multilayer_wall_4d
pressure_vessel_4d rocket_isp_4d waverider_4d
airfoil_transonic_5d cantilever_beam_5d casting_solidification_5d
hetero_conduction_2d laminar_flame_speed_5d membrane_reactor_5d
plate_with_hole_5d soil_consolidation_5d supersonic_intake_5d
supersonic_nozzle_5d thin_plate_buckling_5d variable_beam_5d
dam_seepage_6d fixed_bed_reactor_6d groundwater_contamination_6d
rotor_bemt_6d solar_pv_yield_6d wind_turbine_power_6d wing_vlm_6d
battery_thermal_7d
aircraft_conceptual_8d multilayer_wall_8d
oil_reservoir_9d
ten_bar_truss_10d variable_beam_10d
multilayer_wall_16d variable_beam_20d
EOF
CASES_FLAT="$(echo "${CASES_ALL}" | tr -s '\n ' ' ' | sed 's/^ //;s/ $//')"
N_CASES=$(echo "${CASES_FLAT}" | wc -w | tr -d ' ')

# ── Pre-flight: pre-registration compliance check ────────────────────────
# Refuses to launch if the run config drifts from paper/preregistration.md.
# Override with ALLOW_DEVIATION=1 (logged to preregistration_deviations.txt).
ALLOW_DEV_FLAG=()
if [[ "${ALLOW_DEVIATION:-0}" == "1" ]]; then
    ALLOW_DEV_FLAG=(--allow-deviation --output-dir "${OUT_DIR}")
fi
if ! "${PYTHON}" scripts/validate_run_config.py \
        --pool nature \
        --budgets 5 10 20 50 \
        --n-repeats "${N_REPEATS}" \
        --n-test 1000 \
        --seed-strategy case_repeat \
        --seed-start 1000 \
        --cases ${CASES_FLAT} \
        "${ALLOW_DEV_FLAG[@]}"; then
    echo "" >&2
    echo "ERROR: pre-flight validation failed.  Set ALLOW_DEVIATION=1 to override." >&2
    exit 1
fi
echo

# ── Pre-flight: optional dependency compatibility check ─────────────────
# The publication archive claims the complete 20-method nature pool.
if ! "${PYTHON}" scripts/check_optional_dependencies.py \
        --pool nature \
        --require-botorch \
        --expected-methods 20 \
        --require-formal-case-backends \
        --require-runtime-monitoring; then
    echo "" >&2
    echo "ERROR: optional dependency pre-flight failed. Fix BoTorch/GPyTorch/Torch, formal case backends, and psutil runtime monitoring before the headline sweep." >&2
    exit 1
fi
echo

# ── Pre-flight: all-case smoke/readiness check ──────────────────────────
# This low-level launcher is mostly for Stage-2 resume.  Keep the same
# case-readiness gate as scripts/run_pipeline.py so a direct resume cannot
# bypass broken-case, NaN/Inf, determinism, or expected-range checks.
if ! "${PYTHON}" scripts/audit_case_experiment_readiness.py \
        --output docs/case_experiment_readiness.csv \
        --fail-on-warning; then
    echo "" >&2
    echo "ERROR: case experiment-readiness pre-flight failed. Fix the case audit before launching or resuming Stage 2." >&2
    exit 1
fi
echo

# ── Pre-flight: existing results.csv safe-resume contract ────────────────
# Missing results.csv is normal before the first launch.  If a partial archive
# exists, fail closed on corrupt schemas, duplicate/protocol-drift rows,
# source-hash drift, invalid status/error contracts, or bad timings before
# --skip-existing can trust any streamed cell.
if ! "${PYTHON}" scripts/check_sweep_runtime_health.py \
        "${OUT_DIR}/results.csv" \
        --run-config-json "${OUT_DIR}/run_config.json" \
        --require-run-config \
        --validate-current-environment \
        --allow-missing; then
    echo "" >&2
    echo "ERROR: existing results.csv is not safe to resume. Inspect, repair, or move it aside before launching Stage 2." >&2
    exit 1
fi
echo

LOG_OUT="${OUT_DIR}/run.log"
LOG_ERR="${OUT_DIR}/run.err"

# ── BLAS thread caps (critical for N_JOBS parallelism without oversub) ───
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Build the python command once
PY_CMD=(
    "${PYTHON}" "benchmark/scripts/run_full_benchmark.py"
    --cases ${CASES_FLAT}
    --pool nature
    --budgets 5 10 20 50
    --n-repeats "${N_REPEATS}"
    --n-test 1000
    --seed-strategy case_repeat
    --seed-start 1000
    --n-jobs "${N_JOBS}"
    --bo-n-jobs "${BO_N_JOBS}"
    --mem-cap-gb "${MEM_CAP_GB}"
    --skip-existing
    --timeout-per-cell "${TIMEOUT_S}"
    --bo-timeout-per-cell "${BO_TIMEOUT_S}"
    --ref-timeout-s "${REF_TIMEOUT_S}"
    --heartbeat-s "${HEARTBEAT_S}"
    --max-wall-clock-h "${WALL_H}"
    --output-dir "${OUT_DIR}"
    --confirm-large-run
    --require-botorch
    --require-formal-case-backends
    --require-runtime-monitoring
)

# ── Banner ────────────────────────────────────────────────────────────────
print_banner() {
    cat <<BANNER

================================================================
 DOE-Benchmark single-phase sweep (Linux high-spec)
================================================================
  python          : ${PYTHON}
  host detected   : ${CORES_DETECTED} cores / ${RAM_DETECTED_GB} GB RAM
                    (auto: N_JOBS=${N_JOBS_AUTO}, BO_N_JOBS=${BO_N_JOBS_AUTO}, MEM_CAP=${RAM_FOR_SWEEP_GB} GB)
  cases           : ${N_CASES} (full pre-registration set, d ∈ [2, 20])
  pool            : nature  (20 methods: 15 static + 3 sequential + 2 BoTorch)
  budgets         : 5 10 20 50  (× d)
  n_repeats       : ${N_REPEATS}
  n_jobs (light)  : ${N_JOBS}  (sklearn static methods, full speed)
  n_jobs (medium) : $(( N_JOBS / 2 > 0 ? N_JOBS / 2 : 1 ))  (sequential + qEI, half speed)
  n_jobs (heavy)  : ${BO_N_JOBS}  (BoTorch SAASBO, memory-bounded)
  mem_cap_gb      : ${MEM_CAP_GB} GB (soft RSS guard; exceeds → serial + gc)
  per-cell cap    : ${TIMEOUT_S}s sklearn / ${BO_TIMEOUT_S}s BoTorch
  ref cap         : ${REF_TIMEOUT_S}s
  heartbeat       : ${HEARTBEAT_S}s
  wall-clock cap  : ${WALL_H}h
  total cells     : $((N_CASES * 20 * 4 * N_REPEATS))
  out dir         : ${OUT_DIR}
  expected ETA    : ~30–50 h on 32-core/64 GB; ~16–28 h on 64-core/200 GB
  persistence     : ${PERSISTENCE_MODE}
================================================================

BANNER
}

# ── Dispatch ──────────────────────────────────────────────────────────────
refuse_live_pid_file "${OUT_DIR}/sweep.pid"
case "${PERSISTENCE_MODE}" in
    fg)
        print_banner
        echo "Running in foreground.  Ctrl+C to abort cleanly."
        exec "${PY_CMD[@]}"
        ;;

    tmux)
        if ! command -v tmux >/dev/null 2>&1; then
            echo "ERROR: tmux not installed.  Install via apt/yum/dnf or fall back to nohup mode." >&2
            exit 1
        fi
        SESSION="doe-sweep"
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            echo "ERROR: tmux session '${SESSION}' already exists." >&2
            echo "  Attach with    : tmux attach -t ${SESSION}" >&2
            echo "  Or kill first  : tmux kill-session -t ${SESSION}" >&2
            exit 1
        fi
        # Build a single command string for tmux to execute under bash -lc.
        # We append `; bash` so the session stays open after the run finishes
        # and the operator can inspect the final state.
        CMD_STR="$(printf '%q ' "${PY_CMD[@]}")"
        LOG_OUT_Q="$(printf '%q' "${LOG_OUT}")"
        tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}" \
            "exec bash -lc 'set -o pipefail; ${CMD_STR} 2>&1 | tee -a ${LOG_OUT_Q}; sweep_rc=\${PIPESTATUS[0]}; echo; echo \"[sweep finished, exit=\${sweep_rc}]\"; exec bash'"
        print_banner
        echo "Sweep started in tmux session: ${SESSION}"
        echo
        echo "Attach with:"
        echo "  tmux attach -t ${SESSION}"
        echo "Detach (without killing the run): Ctrl+b then d"
        echo
        echo "Tail logs without attaching:"
        echo "  tail -f ${LOG_OUT}"
        echo
        echo "Check progress:"
        echo "  bash scripts/check_progress_remote.sh"
        echo "  MEM_CAP_GB=${MEM_CAP_GB} DISK_WARNING_GB=${DISK_WARNING_GB:-25} bash scripts/check_progress_remote.sh -w"
        echo "  VALIDATE_CURRENT_ENV=1 MEM_CAP_GB=${MEM_CAP_GB} DISK_WARNING_GB=${DISK_WARNING_GB:-25} bash scripts/check_progress_remote.sh -w"
        echo "  MEM_CAP_GB=${MEM_CAP_GB} DISK_WARNING_GB=${DISK_WARNING_GB:-25} STALE_AFTER_MIN=${STALE_AFTER_MIN:-45} FAIL_ON_STALE=1 bash scripts/check_progress_remote.sh -w"
        ;;

    nohup)
        print_banner
        nohup "${PY_CMD[@]}" >>"${LOG_OUT}" 2>>"${LOG_ERR}" &
        SWEEP_PID=$!
        # Detach from this shell's job table so closing SSH won't SIGHUP it
        disown "${SWEEP_PID}" 2>/dev/null || true
        sleep 2
        if ! kill -0 "${SWEEP_PID}" 2>/dev/null; then
            echo "ERROR: sweep died within 2 s of launch.  Tail of run.err:" >&2
            tail -40 "${LOG_ERR}" >&2 || true
            exit 1
        fi
        echo "Sweep started in background (nohup)."
        echo "  PID            : ${SWEEP_PID}"
        echo "  PID file       : ${OUT_DIR}/sweep.pid"
        echo "${SWEEP_PID}" > "${OUT_DIR}/sweep.pid"
        echo
        echo "Monitor:"
        echo "  tail -f ${LOG_OUT}"
        echo "  bash scripts/check_progress_remote.sh"
        echo "  MEM_CAP_GB=${MEM_CAP_GB} DISK_WARNING_GB=${DISK_WARNING_GB:-25} bash scripts/check_progress_remote.sh -w"
        echo "  VALIDATE_CURRENT_ENV=1 MEM_CAP_GB=${MEM_CAP_GB} DISK_WARNING_GB=${DISK_WARNING_GB:-25} bash scripts/check_progress_remote.sh -w"
        echo "  MEM_CAP_GB=${MEM_CAP_GB} DISK_WARNING_GB=${DISK_WARNING_GB:-25} STALE_AFTER_MIN=${STALE_AFTER_MIN:-45} FAIL_ON_STALE=1 bash scripts/check_progress_remote.sh -w"
        echo
        echo "Stop cleanly:"
        echo "  kill ${SWEEP_PID}"
        echo "  # or: kill \$(cat ${OUT_DIR}/sweep.pid)"
        ;;
esac
