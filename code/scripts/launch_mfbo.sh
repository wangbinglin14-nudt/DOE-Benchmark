#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch the **multi-fidelity BO supplementary sweep** on a remote Linux host.
#
# Pool      : mf      (1 method: MF-Cokriging)
# Cases     : 5  (the supports_multifidelity=True cases)
# Budgets   : 5 10 20 50  (× d, in low-fidelity units)
# Repeats   : 30
# Output    : benchmark/results/mf_supplementary/
#
# Documented in paper/preregistration_amendments.md (A2).
#
# Usage:
#   bash scripts/launch_mfbo.sh                # nohup (default)
#   bash scripts/launch_mfbo.sh tmux           # tmux session
#   N_JOBS=12 bash scripts/launch_mfbo.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_remote_lib.sh"

cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
if [[ -d "${VENV_DIR}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi
PYTHON="$(command -v python)"
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: no python in PATH." >&2
    exit 1
fi

echo "[preflight] validating BoTorch/Torch compatibility for pool=mf"
"${PYTHON}" "${REPO_ROOT}/scripts/check_optional_dependencies.py" --pool mf

OUT_DIR="${REPO_ROOT}/benchmark/results/mf_supplementary"
CANONICAL_HEADLINE_DIR="${REPO_ROOT}/benchmark/results/full_benchmark"
SOTA_ROBUSTNESS_DIR="${REPO_ROOT}/benchmark/results/sota_robustness"

canonical_path() {
    "${PYTHON}" - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
}

OUT_DIR_REAL="$(canonical_path "${OUT_DIR}")"
CANONICAL_HEADLINE_DIR_REAL="$(canonical_path "${CANONICAL_HEADLINE_DIR}")"
SOTA_ROBUSTNESS_DIR_REAL="$(canonical_path "${SOTA_ROBUSTNESS_DIR}")"
if [[ "${OUT_DIR_REAL}" == "${CANONICAL_HEADLINE_DIR_REAL}" ]]; then
    echo "ERROR: MF supplementary OUT_DIR must not be benchmark/results/full_benchmark." >&2
    echo "Keep amendment A2 results isolated under benchmark/results/mf_supplementary." >&2
    exit 1
fi
if [[ "${OUT_DIR_REAL}" == "${SOTA_ROBUSTNESS_DIR_REAL}" ]]; then
    echo "ERROR: MF supplementary OUT_DIR must not point to benchmark/results/sota_robustness." >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"
LOG_OUT="${OUT_DIR}/run.log"
LOG_ERR="${OUT_DIR}/run.err"
PID_FILE="${OUT_DIR}/sweep.pid"

USER_SET_N_JOBS="${N_JOBS+x}"
auto_tune_defaults
pin_blas_threads

# MF runs are mostly single-method; cap parallelism at the case count
# (5) to avoid wasting cores on idle workers, unless the operator explicitly
# provided N_JOBS in the environment.
if [[ -z "${USER_SET_N_JOBS}" ]]; then
    N_JOBS=5
    export N_JOBS
fi

WALL_H="${WALL_H:-24}"     # MF sweep is much smaller than headline
TIMEOUT_S="${TIMEOUT_S:-1800}"  # MF cells are slower (high-fidelity FEM)
REF_TIMEOUT_S="${REF_TIMEOUT_S:-1800}"
HEARTBEAT_S="${HEARTBEAT_S:-60}"
N_REPEATS="${N_REPEATS:-30}"
PERSISTENCE_MODE="${1:-nohup}"
validate_remote_launch_tunables
validate_persistence_mode "${PERSISTENCE_MODE}"
MF_CASES_FLAT="$("${PYTHON}" - <<'PY'
import doe_testbed as db

cases = sorted(
    case.case_id
    for case in db.list_cases()
    if getattr(case, "supports_multifidelity", False)
)
if not cases:
    raise SystemExit("no supports_multifidelity=True cases found")
print(" ".join(cases))
PY
)"
MF_N_CASES=$(echo "${MF_CASES_FLAT}" | wc -w | tr -d ' ')
if [[ "${MF_N_CASES}" -ne 5 ]]; then
    echo "ERROR: expected 5 multi-fidelity cases, found ${MF_N_CASES}: ${MF_CASES_FLAT}" >&2
    echo "Update paper/preregistration_amendments.md and launcher tests before running A2." >&2
    exit 1
fi

PY_CMD=(
    "${PYTHON}" "benchmark/scripts/run_full_benchmark.py"
    --pool mf
    --cases ${MF_CASES_FLAT}
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
    --ref-timeout-s "${REF_TIMEOUT_S}"
    --heartbeat-s "${HEARTBEAT_S}"
    --max-wall-clock-h "${WALL_H}"
    --output-dir "${OUT_DIR}"
)

cat <<BANNER

================================================================
 DOE-Benchmark MULTI-FIDELITY SUPPLEMENTARY sweep
================================================================
  python          : ${PYTHON}
  host detected   : ${CORES_DETECTED} cores / ${RAM_DETECTED_GB} GB RAM
  cases           : ${MF_N_CASES} multi-fidelity cases (${MF_CASES_FLAT})
  pool            : mf  (1 method: MF-Cokriging)
  amendment       : paper/preregistration_amendments.md §A2
  budgets         : 5 10 20 50  (× d, in low-fidelity cost units)
  n_repeats       : ${N_REPEATS}
  n_test          : 1000
  seed strategy   : case_repeat, seed_start=1000
  n_jobs          : ${N_JOBS}
  mem_cap_gb      : ${MEM_CAP_GB} GB
  per-cell cap    : ${TIMEOUT_S}s
  total cells     : $((MF_N_CASES * 1 * 4 * N_REPEATS))
  out dir         : ${OUT_DIR}
  expected ETA    : ~6–18 h depending on case fidelity ratios
  persistence     : ${PERSISTENCE_MODE}
================================================================

BANNER

dispatch_persistent_run \
    "${PERSISTENCE_MODE}" \
    "doe-mf" \
    "${LOG_OUT}" "${LOG_ERR}" "${PID_FILE}" \
    "${PY_CMD[@]}"
