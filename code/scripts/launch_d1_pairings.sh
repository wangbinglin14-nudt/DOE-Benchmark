#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch the **D1 pairing-mode robustness experiment** on a remote Linux host.
#
# Runs run_ranking_reversal.py with --pairing-mode all, which executes the
# analytical phase once and the physics phase three times (one per pairing
# mode: dimension / topology / random).
#
# Documented in paper/preregistration_amendments.md (A3).
#
# Output: benchmark/results/ranking_reversal/
#         ├── analytical_results.csv         (shared)
#         ├── pairing_dimension/
#         ├── pairing_topology/
#         └── pairing_random/
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

echo "[preflight] validating BoTorch/Torch compatibility for pool=nature"
"${PYTHON}" "${REPO_ROOT}/scripts/check_optional_dependencies.py" \
    --pool nature --require-botorch --expected-methods 20

OUT_DIR="${REPO_ROOT}/benchmark/results/ranking_reversal"
CANONICAL_HEADLINE_DIR="${REPO_ROOT}/benchmark/results/full_benchmark"
SOTA_ROBUSTNESS_DIR="${REPO_ROOT}/benchmark/results/sota_robustness"
MF_SUPPLEMENTARY_DIR="${REPO_ROOT}/benchmark/results/mf_supplementary"

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
MF_SUPPLEMENTARY_DIR_REAL="$(canonical_path "${MF_SUPPLEMENTARY_DIR}")"
if [[ "${OUT_DIR_REAL}" == "${CANONICAL_HEADLINE_DIR_REAL}" ]]; then
    echo "ERROR: D1 pairing output must not be benchmark/results/full_benchmark." >&2
    echo "Keep formal D1 analysis isolated under benchmark/results/ranking_reversal." >&2
    exit 1
fi
if [[ "${OUT_DIR_REAL}" == "${SOTA_ROBUSTNESS_DIR_REAL}" ]]; then
    echo "ERROR: D1 pairing output must not point to benchmark/results/sota_robustness." >&2
    exit 1
fi
if [[ "${OUT_DIR_REAL}" == "${MF_SUPPLEMENTARY_DIR_REAL}" ]]; then
    echo "ERROR: D1 pairing output must not point to benchmark/results/mf_supplementary." >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"
LOG_OUT="${OUT_DIR}/run_pairings.log"
LOG_ERR="${OUT_DIR}/run_pairings.err"
PID_FILE="${OUT_DIR}/pairings.pid"

auto_tune_defaults
pin_blas_threads

N_REPEATS="${N_REPEATS:-30}"
TIMEOUT_S="${TIMEOUT_S:-600}"
BO_TIMEOUT_S="${BO_TIMEOUT_S:-1800}"
REF_TIMEOUT_S="${REF_TIMEOUT_S:-600}"
WALL_H="${WALL_H:-24}"          # required by shared launcher validation
HEARTBEAT_S="${HEARTBEAT_S:-60}" # required by shared launcher validation
PERSISTENCE_MODE="${1:-nohup}"
validate_remote_launch_tunables
validate_persistence_mode "${PERSISTENCE_MODE}"

PY_CMD=(
    "${PYTHON}" -m benchmark.scripts.run_ranking_reversal
    --pool nature
    --pairing-mode all
    --n-repeats "${N_REPEATS}"
    --budgets 10 20
    --n-test 1000
    --seed-strategy case_repeat
    --seed-start 1000
    --n-jobs "${N_JOBS}"
    --bo-n-jobs "${BO_N_JOBS}"
    --mem-cap-gb "${MEM_CAP_GB}"
    --skip-existing
    --timeout-s "${TIMEOUT_S}"
    --bo-timeout-s "${BO_TIMEOUT_S}"
    --ref-timeout-s "${REF_TIMEOUT_S}"
    --output-dir "${OUT_DIR}"
)

cat <<BANNER

================================================================
 DOE-Benchmark D1 PAIRING-MODE ROBUSTNESS experiment
================================================================
  python          : ${PYTHON}
  host detected   : ${CORES_DETECTED} cores / ${RAM_DETECTED_GB} GB RAM
                    (auto: N_JOBS=${N_JOBS_AUTO}, BO_N_JOBS=${BO_N_JOBS_AUTO})
  amendment       : paper/preregistration_amendments.md §A3
  pairings        : dimension, topology, random
  pool            : nature (20 methods)
  n_repeats       : ${N_REPEATS}
  budgets         : 10 20  (× d)
  n_test          : 1000
  seed strategy   : case_repeat, seed_start=1000
  n_jobs          : ${N_JOBS} light / ${BO_N_JOBS} heavy
  mem_cap_gb      : ${MEM_CAP_GB} GB
  per-cell cap    : ${TIMEOUT_S}s sklearn / ${BO_TIMEOUT_S}s BoTorch
  out dir         : ${OUT_DIR}
  expected ETA    : ~30-60 min on 32+core hosts (parallel + skip-existing)
  persistence     : ${PERSISTENCE_MODE}
================================================================

BANNER

dispatch_persistent_run \
    "${PERSISTENCE_MODE}" \
    "doe-d1pairings" \
    "${LOG_OUT}" "${LOG_ERR}" "${PID_FILE}" \
    "${PY_CMD[@]}"
