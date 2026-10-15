# Project structure and execution entry points

`C:\wbl\DOE` is the current canonical project root. Run commands from this
directory unless a command explicitly states otherwise. The older
`C:\DOE-benchmark` tree and `C:\DOE-benchmark_external_archive` are retained
for historical provenance only; do not merge their generated results into the
canonical archive.

Current entry: START_HERE_zh.md（历史位置：`99_archive/process_documents/root/START_HERE_zh.md`）.

The dated delivery is under `submission/communications-engineering-2026-09-16/`.
This review archives transient files under `_archive_20260916/` with a hash inventory.

## Directory map

| Path | Role | Edit policy |
|---|---|---|
| `doe_testbed/` | Physics-grounded case library: base classes, metadata, case registry, 34 engineering cases. | Source code. Keep independent from `benchmark/`. |
| `benchmark/` | DOE benchmark engine: method pools, runner, metrics, analysis, Discovery D1/D2/D3 scripts, benchmark tests. | Source code. Writes generated outputs under `benchmark/results/`. |
| `scripts/` | Top-level orchestration, validation, figure-data, figure, manifest, and paper-number utilities. | Source code. Invoke from repo root with `python scripts/<name>.py`. |
| `tests/` | Case-library and metadata tests. | Source code. Run with `python -m pytest tests -q`. |
| `docs/` | Publication-track project documentation: reproducibility, project status, gap analysis, SAP, FAQ, glossary, plus `PROJECT_MEMORY.md` (cross-model handoff), `PROJECT_FILE_CATALOG.md` (per-file map) and `PAPER_CLAIM_TRACEABILITY.md` (result→code→data matrix). | Documentation. Keep relative links valid. |
| `paper/` | Pre-registration documents and submission templates. Pipeline outputs such as `figdata/`, `figures/`, and `numbers.*` are generated here. | Protocol/templates plus generated publication artifacts. |
| `paper-reviews/` | Current Communications Engineering review package plus dated historical review archive. | Keep current review under `communications-engineering/`; historical rounds are immutable. |
| `案例正确性校验/` | Per-case DOCX validation sources and public-source registers. | Evidence library; preserve paths referenced by the case ledger and checks. |
| `web/` | Optional web dashboard/backend source. | Optional application source. Regenerate frontend dependencies locally. |
| `examples/` | Small usage examples. | Source examples. |
| `tools/` | Audit helpers and local engineering checks. | Utility source. |
| `_archive_20260909/` | Current housekeeping archive for root caches, build residue and Sobol diagnostics. | Do not use as active inputs; consult its manifest for restoration. |

## Root files

| File | Purpose |
|---|---|
| `README.md` / `README_zh.md` | User-facing overview, install, quick start, benchmark commands, case list. |
| `ARCHITECTURE.md` | Layered architecture and data-flow invariants. |
| `PROJECT_STRUCTURE.md` | Current folder map and command entry points. |
| `docs/working-notes/DOE-Benchmark实验设计方案.md` | Full execution specification for Stage 0-11 (Chinese). |
| `docs/working-notes/DOE-Benchmark实验结果分析.md` | Working analysis notes for the benchmark results (Chinese). |
| `docs/working-notes/运行指南.md` | Operator quick-run guide (Chinese). |
| `docs/review-inputs/问题说明.pdf` | User-provided issue statement used as manuscript-review input. |
| `pyproject.toml` | Package metadata, optional dependencies, pytest/ruff/format configuration. |
| `requirements.in` | High-level dependency intent for the full benchmark pipeline. |
| `requirements.lock` | Fully pinned dependency snapshot for reproducible runs. |
| `Snakefile` | Snakemake reproduction workflow. |
| `Dockerfile.benchmark` | Reproducible container build for benchmark runs. |
| `mkdocs.yml` | Documentation-site configuration for the current `docs/` tree. |
| `ROOT_REORG_NOTE.md` | Record of the root-folder promotion and external archive location. |

## Main command entry points

### Install

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.lock
python -m pip install -e .
```

For development without strict pins:

```bash
python -m pip install -e ".[dev]"
```

### Smoke and validation

```bash
python scripts/check_doc_links.py
python scripts/validate_run_config.py --pool nature --budgets 5 10 20 50 --n-repeats 30 --n-test 1000 --seed-strategy case_repeat --seed-start 1000 --cases ALL
python -m pytest tests benchmark/tests -q
```

### One-click pipeline

```bash
python scripts/run_pipeline.py --dry-run
python scripts/run_pipeline.py --profile smoke
python scripts/run_pipeline.py
```

Windows wrapper:

```powershell
pwsh -File scripts/run_pipeline.ps1 -Profile smoke
```

Linux/macOS wrapper:

```bash
bash scripts/run_pipeline.sh
```

### Low-level benchmark sweep

```bash
python benchmark/scripts/run_full_benchmark.py \
    --pool nature \
    --budgets 5 10 20 50 \
    --n-repeats 30 \
    --n-test 1000 \
    --seed-strategy case_repeat \
    --seed-start 1000 \
    --output-dir benchmark/results/full_benchmark
```

### Stage outputs

| Stage | Primary command | Primary output |
|---:|---|---|
| 1 | `python scripts/validate_run_config.py ...` | Compliance report on stdout. |
| 2 | `python benchmark/scripts/run_full_benchmark.py ...` | `benchmark/results/full_benchmark/results.csv` |
| 3 | `python benchmark/scripts/generate_report.py benchmark/results/full_benchmark` | `benchmark/results/full_benchmark/report/REPORT.md` |
| 4 | `python benchmark/scripts/run_ranking_reversal.py ...` | `benchmark/results/ranking_reversal/correlation.json` |
| 5 | `python benchmark/scripts/run_sample_complexity.py ...` | `benchmark/results/sample_complexity_0.20/rule_summary.csv` |
| 6 | `python benchmark/scripts/run_reliability_recovery.py ...` | `benchmark/results/reliability_recovery/REPORT.md` |
| 8 | `python scripts/prepare_figdata.py ...` | `paper/figdata/*.csv` |
| 9 | `python scripts/plots_nature.py --figdata paper/figdata --out-dir paper/figures` | `paper/figures/{pdf,png}/fig*.{pdf,png}` |
| 10 | `python scripts/generate_run_manifest.py ...` | `RUN_MANIFEST.json` |

## Generated and archived content

- Generated benchmark outputs live under `benchmark/results/`.
- Generated paper artifacts live under `paper/figdata/`, `paper/figures/`, and `paper/numbers.*`.
- Pipeline runtime logs, when produced, are transient and should be moved to
  `_archive_20260909/root-build/` after the run.
- Regenerable caches such as `__pycache__`, `.pytest_cache`, `.ruff_cache`, `web/frontend/node_modules`, and `web/frontend/.next` should not be committed.
- Root-level transient files and Sobol diagnostic snapshots are retained under
  `_archive_20260909/`; its `README.md` and inventory identify the original
  paths and SHA-256 hashes.
- Dated `_archive_20260621/`, `_archive_20260728/`,
  `_archive_20260831_d1_precanonical/`, and `_archive_20260908/` directories
  are immutable historical snapshots.
