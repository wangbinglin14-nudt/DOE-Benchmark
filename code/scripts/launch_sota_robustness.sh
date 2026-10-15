#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch the **post-registration SOTA robustness sweep** on a remote Linux host.
#
# Pool      : sota   (= nature_pool + TURBO + D-optimal = 22 methods)
# Cases     : all 34 (same as headline)
# Budgets   : 5 10 20 50  (× d)
# Repeats   : 30
# Output    : benchmark/results/sota_robustness/
#
# This experiment is documented in paper/preregistration_amendments.md (A1).
# It does NOT replace the headline `nature_pool()` 30-repeat sweep —
# the paper still reports D1-D3 from the latter.
#
# Usage:
#   bash scripts/launch_sota_robustness.sh                # nohup (default)
#   bash scripts/launch_sota_robustness.sh tmux           # tmux session
#   N_JOBS=24 bash scripts/launch_sota_robustness.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Source the shared helpers
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_remote_lib.sh"

cd "${REPO_ROOT}"

# Activate venv if present
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

echo "[preflight] validating BoTorch/Torch compatibility for pool=sota"
"${PYTHON}" "${REPO_ROOT}/scripts/check_optional_dependencies.py" --pool sota

OUT_DIR="${REPO_ROOT}/benchmark/results/sota_robustness"
HEADLINE_DIR="${HEADLINE_DIR:-${REPO_ROOT}/benchmark/results/full_benchmark}"
CANONICAL_HEADLINE_DIR="${REPO_ROOT}/benchmark/results/full_benchmark"
MF_SUPPLEMENTARY_DIR="${REPO_ROOT}/benchmark/results/mf_supplementary"

canonical_path() {
    "${PYTHON}" - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
}

OUT_DIR_REAL="$(canonical_path "${OUT_DIR}")"
HEADLINE_DIR_REAL="$(canonical_path "${HEADLINE_DIR}")"
CANONICAL_HEADLINE_DIR_REAL="$(canonical_path "${CANONICAL_HEADLINE_DIR}")"
MF_SUPPLEMENTARY_DIR_REAL="$(canonical_path "${MF_SUPPLEMENTARY_DIR}")"
if [[ "${OUT_DIR_REAL}" == "${CANONICAL_HEADLINE_DIR_REAL}" ]]; then
    echo "ERROR: SOTA robustness OUT_DIR must not be benchmark/results/full_benchmark." >&2
    echo "Keep amendment A1 results isolated under benchmark/results/sota_robustness." >&2
    exit 1
fi
if [[ "${HEADLINE_DIR_REAL}" == "${OUT_DIR_REAL}" ]]; then
    echo "ERROR: HEADLINE_DIR must not point to the SOTA robustness output directory." >&2
    exit 1
fi
if [[ "${HEADLINE_DIR_REAL}" == "${MF_SUPPLEMENTARY_DIR_REAL}" ]]; then
    echo "ERROR: HEADLINE_DIR must not point to benchmark/results/mf_supplementary." >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"
LOG_OUT="${OUT_DIR}/run.log"
LOG_ERR="${OUT_DIR}/run.err"
PID_FILE="${OUT_DIR}/sweep.pid"

# ── Smart-reuse of the headline sweep ────────────────────────────────────
# A1 (sota pool) is a strict superset of nature pool: 22 = 20 + {TURBO,
# D-optimal}.  If the headline sweep has already produced a results.csv,
# we seed sota_robustness/ with it so --skip-existing reuses all 81 600
# nature-pool rows; only the 8 160 new (TURBO, D-optimal) cells need
# fresh computation, cutting Phase 2 wall-clock from ~24 h to ~2 h on a
# 64-core host.  Set REUSE_HEADLINE=0 to disable (forces full re-compute).
REUSE_HEADLINE="${REUSE_HEADLINE:-1}"
if [[ "${REUSE_HEADLINE}" == "1" ]]; then
    src_csv="${HEADLINE_DIR}/results.csv"
    dst_csv="${OUT_DIR}/results.csv"
    if [[ -f "${src_csv}" && ! -f "${dst_csv}" ]]; then
        echo "[reuse-headline]  copying ${src_csv} -> ${dst_csv}"
        cp -- "${src_csv}" "${dst_csv}"
        n_seed=$(awk 'END{print NR-1}' "${dst_csv}")
        echo "[reuse-headline]  seeded with ${n_seed} rows from headline sweep"
    elif [[ -f "${dst_csv}" ]]; then
        echo "[reuse-headline]  ${dst_csv} already exists — skipping seed copy"
    else
        echo "[reuse-headline]  WARNING: ${src_csv} not found"
        echo "[reuse-headline]  Phase 2 will run all 89 760 cells from scratch."
        echo "[reuse-headline]  Run scripts/launch_sweep_remote.sh first, or set REUSE_HEADLINE=0 to silence this."
    fi
fi

auto_tune_defaults
pin_blas_threads

WALL_H="${WALL_H:-72}"
TIMEOUT_S="${TIMEOUT_S:-600}"
BO_TIMEOUT_S="${BO_TIMEOUT_S:-1800}"
REF_TIMEOUT_S="${REF_TIMEOUT_S:-600}"
HEARTBEAT_S="${HEARTBEAT_S:-60}"
N_REPEATS="${N_REPEATS:-30}"
PERSISTENCE_MODE="${1:-nohup}"
validate_remote_launch_tunables
validate_persistence_mode "${PERSISTENCE_MODE}"

# 34-case full pre-reg set (same as headline)
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
"${PYTHON}" - "${CASES_FLAT}" <<'PY'
import sys

import doe_testbed as db

declared = sorted(sys.argv[1].split())
registered = sorted(case.case_id for case in db.list_cases())
if declared != registered:
    missing = sorted(set(registered) - set(declared))
    extra = sorted(set(declared) - set(registered))
    raise SystemExit(
        "SOTA launcher CASES_ALL does not match doe_testbed registry; "
        f"missing={missing}, extra={extra}"
    )
PY

# Build command
PY_CMD=(
    "${PYTHON}" "benchmark/scripts/run_full_benchmark.py"
    --cases ${CASES_FLAT}
    --pool sota
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
)

# Banner
cat <<BANNER

================================================================
 DOE-Benchmark POST-REGISTRATION SOTA ROBUSTNESS sweep
================================================================
  python          : ${PYTHON}
  host detected   : ${CORES_DETECTED} cores / ${RAM_DETECTED_GB} GB RAM
                    (auto: N_JOBS=${N_JOBS_AUTO}, BO_N_JOBS=${BO_N_JOBS_AUTO})
  cases           : ${N_CASES} (same 34-case set as headline)
  pool            : sota  (22 methods: nature + TURBO + D-optimal)
  amendment       : paper/preregistration_amendments.md §A1
  budgets         : 5 10 20 50  (× d)
  n_repeats       : ${N_REPEATS}
  n_test          : 1000
  seed strategy   : case_repeat, seed_start=1000
  n_jobs          : ${N_JOBS} light / $((N_JOBS/2)) medium / ${BO_N_JOBS} heavy
  mem_cap_gb      : ${MEM_CAP_GB} GB
  per-cell cap    : ${TIMEOUT_S}s sklearn / ${BO_TIMEOUT_S}s BoTorch
  total cells     : $((N_CASES * 22 * 4 * N_REPEATS))  (of which up to 81 600 reused)
  reuse headline  : REUSE_HEADLINE=${REUSE_HEADLINE}  (HEADLINE_DIR=${HEADLINE_DIR})
  out dir         : ${OUT_DIR}
  expected ETA    : ~2 h with reuse / ~18-32 h without  (64-core/200 GB)
  persistence     : ${PERSISTENCE_MODE}
================================================================

BANNER

dispatch_persistent_run \
    "${PERSISTENCE_MODE}" \
    "doe-sota" \
    "${LOG_OUT}" "${LOG_ERR}" "${PID_FILE}" \
    "${PY_CMD[@]}"
