#!/usr/bin/env bash
# Create a server-return tarball after the formal full pipeline has completed.
#
# This script is intentionally read-only with respect to benchmark outputs.  It
# delegates to the cross-platform Python helper so the exact packaging contract
# can also be exercised by local CI on Windows review workstations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

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
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "ERROR: cannot find a Python interpreter (set \$PYTHON or activate .venv)" >&2
  exit 127
}
PYTHON_CMD="$(resolve_python)"

OUT_DIR="${REPO_ROOT}/benchmark/results/full_benchmark"
HEALTH_REPORT="${OUT_DIR}/runtime_health_report.txt"
mkdir -p "${OUT_DIR}"
TMP_HEALTH="$(mktemp "${OUT_DIR}/.runtime_health_report.XXXXXX.txt")"
cleanup_tmp_health() {
  if [[ -n "${TMP_HEALTH:-}" && -f "${TMP_HEALTH}" ]]; then
    rm -f "${TMP_HEALTH}"
  fi
}
trap cleanup_tmp_health EXIT HUP INT TERM
if "${PYTHON_CMD}" scripts/check_sweep_runtime_health.py \
      "${OUT_DIR}/results.csv" \
      --run-config-json "${OUT_DIR}/run_config.json" \
      --validate-current-environment \
      --require-run-config >"${TMP_HEALTH}" 2>&1; then
  mv "${TMP_HEALTH}" "${HEALTH_REPORT}"
  TMP_HEALTH=""
  cat "${HEALTH_REPORT}"
else
  rc=$?
  cat "${TMP_HEALTH}" >&2 || true
  echo "ERROR: runtime health gate failed; not creating the server return package." >&2
  exit "${rc}"
fi

if [[ $# -gt 0 ]]; then
  exec "${PYTHON_CMD}" scripts/create_server_return_package.py "$@"
fi
exec "${PYTHON_CMD}" scripts/create_server_return_package.py
