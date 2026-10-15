#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DOE-Benchmark — remote server setup
#
# Run this ONCE on the remote server after extracting the tarball.
# Creates a clean Python venv, installs all benchmark dependencies, and
# runs a smoke test (1 case × 2 reps × default pool, ~30 s) to verify.
#
# Usage:
#   bash scripts/setup_remote.sh                    # auto-detect Python
#   bash scripts/setup_remote.sh /usr/bin/python3.11
#   PYTHON=/opt/conda/bin/python bash scripts/setup_remote.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve Python interpreter ───────────────────────────────────────────
PYTHON="${PYTHON:-${1:-}}"
if [[ -z "${PYTHON}" ]]; then
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            PYTHON="$(command -v "${candidate}")"
            break
        fi
    done
fi
if [[ -z "${PYTHON}" ]] || ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "ERROR: no Python 3.10+ found.  Install one or pass the path:" >&2
    echo "  bash scripts/setup_remote.sh /path/to/python3.11" >&2
    exit 1
fi

PY_VERSION=$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "==> Using Python: ${PYTHON}  (version ${PY_VERSION})"

# Resolve repo root (this script lives in scripts/ at repo root)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"
echo "==> Repo root: ${REPO_ROOT}"

# ── Create / reuse virtualenv ────────────────────────────────────────────
VENV_DIR="${REPO_ROOT}/.venv"
if [[ -d "${VENV_DIR}" ]]; then
    echo "==> Reusing existing venv at ${VENV_DIR}"
else
    echo "==> Creating venv at ${VENV_DIR}"
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "==> Active venv: $(which python)  ($(python --version))"

# ── Upgrade pip and install deps ─────────────────────────────────────────
echo "==> Upgrading pip / setuptools / wheel"
pip install -q --upgrade pip setuptools wheel

if [[ ! -f requirements.lock ]]; then
    echo "ERROR: requirements.lock missing.  Did the tar transfer succeed?" >&2
    exit 1
fi

echo "==> Installing benchmark dependencies (this may take 5–10 minutes the first time)"
pip install -r requirements.lock

echo "==> Installing doe-testbed in editable mode"
pip install -e .

# ── System info dump (saved for the run record) ──────────────────────────
INFO_OUT="${REPO_ROOT}/benchmark/results/full_benchmark/_remote_setup_info.txt"
mkdir -p "$(dirname "${INFO_OUT}")"
{
    echo "==== setup_remote.sh ran at $(date -Iseconds) ===="
    echo
    echo "---- system ----"
    uname -a
    echo
    if [[ -r /etc/os-release ]]; then
        cat /etc/os-release | head -10
    fi
    echo
    echo "---- cpu ----"
    lscpu 2>/dev/null | head -20 || echo "(lscpu not available)"
    echo
    echo "---- memory ----"
    free -h 2>/dev/null || echo "(free not available)"
    echo
    echo "---- disk ----"
    df -h "${REPO_ROOT}" || true
    echo
    echo "---- python ----"
    python --version
    pip --version
    echo
    echo "---- key package versions ----"
    pip show numpy scipy pandas scikit-learn joblib torch botorch gpytorch 2>/dev/null \
        | grep -E '^(Name|Version):' || true
} > "${INFO_OUT}"
echo "==> System info written to ${INFO_OUT}"

# ── Smoke test ────────────────────────────────────────────────────────────
echo
echo "==> Running smoke test (1 case × default pool × 2 reps × 1 budget, ~30 s)"
SMOKE_OUT="${REPO_ROOT}/benchmark/results/_remote_smoke"
rm -rf "${SMOKE_OUT}"
mkdir -p "${SMOKE_OUT}"
python benchmark/scripts/run_full_benchmark.py \
    --cases tuned_mass_damper_3d \
    --pool default \
    --budgets 5 \
    --n-repeats 2 \
    --n-jobs 2 \
    --max-wall-clock-h 0.05 \
    --output-dir "${SMOKE_OUT}" \
    > "${SMOKE_OUT}/smoke.log" 2>&1 || {
    echo "ERROR: smoke test failed.  See ${SMOKE_OUT}/smoke.log" >&2
    tail -40 "${SMOKE_OUT}/smoke.log" >&2 || true
    exit 1
}

SMOKE_ROWS=$(python -c "import pandas as pd; print(len(pd.read_csv('${SMOKE_OUT}/results.csv')))" 2>/dev/null || echo 0)
echo "==> Smoke test produced ${SMOKE_ROWS} rows."
if [[ "${SMOKE_ROWS}" -lt 5 ]]; then
    echo "WARNING: smoke test produced fewer rows than expected (${SMOKE_ROWS} < 5)." >&2
    echo "         Check ${SMOKE_OUT}/smoke.log for warnings." >&2
fi

# ── Cleanup smoke artefacts ──────────────────────────────────────────────
rm -rf "${SMOKE_OUT}"

# ── Final report ─────────────────────────────────────────────────────────
echo
echo "================================================================"
echo " Remote setup complete"
echo "================================================================"
echo "  venv         : ${VENV_DIR}"
echo "  Python       : $(which python)"
echo "  pip freeze   : pip freeze | head -20"
echo
echo "Activate the venv in any new shell with:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Run the no-sweep server readiness gate:"
echo "  python scripts/check_server_launch_ready.py"
echo
echo "Launch the formal full archive pipeline with:"
echo "  bash scripts/run_pipeline.sh"
echo
echo "For a low-level Stage 2 restart only:"
echo "  bash scripts/launch_sweep_remote.sh tmux"
echo
