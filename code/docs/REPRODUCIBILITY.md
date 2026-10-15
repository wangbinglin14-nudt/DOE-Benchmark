# Reproducibility — Stage 0 → 10 walkthrough

> Submission archive update (2026-09-17): start with [RESEARCH_REUSE.md](RESEARCH_REUSE.md)
> for the supplied local archive and [CASE_EVIDENCE_INDEX.md](CASE_EVIDENCE_INDEX.md)
> for current case limitations. The clone URL and host instructions below are
> historical operating notes, not proof of a public deposit. Exact raw-sweep
> replay across dependency versions is not guaranteed by deterministic seeds.

> **Audience**: anyone who wants to reproduce the headline DOE-Benchmark
> v0.4.0 results (Discoveries 1, 2, 3) from a fresh clone.
> **Required time**: ~5 min (smoke), ~30-50 h (full publication sweep on
> Linux ≥ 32 cores). Windows workstations are for smoke, unit, physics, and
> dry-run preflight checks only.
> **Required disk**: ~15 GB during the sweep, ~3 GB for archived results.
> **Required RAM**: ≥ 16 GB minimum, ≥ 64 GB recommended for SAASBO @ 50·d.

This document is the main reproducibility walkthrough. Local workstations
support code edits, unit tests, static audits and smoke checks; the full
publication sweep requires the Linux server resources described below.
Every step is gated and idempotent; restarting from any failed stage is safe.

---

## 0. Pre-requisites

```bash
# Python 3.10–3.13 (3.13 tested in CI)
python --version

# Clone and enter the project root
git clone https://github.com/wangbinglin14-nudt/DOE-Benchmark.git
cd DOE-Benchmark/code

# Editable install with everything needed for the sweep
pip install -e .[dev,full]

# Sanity check
python -c "import doe_testbed as db; print(len(db.list_cases()), 'cases')"
# expected: 34 cases

python -c "from benchmark.methods import nature_pool; print(len(nature_pool()), 'methods')"
# expected for the formal publication run: 20
```

> **Note**.  If you only need to *re-render* figures from already-archived
> results (e.g. running the same data on a new machine just to verify
> figures), you can skip Stages 0-2 and jump straight to Stage 8 with
> `python scripts/run_pipeline.py --profile full --from-stage 8 --confirm-full-run`.

---

## 1. Local checks versus server runs

On the local review workstation, use the smoke and audit commands below.  Do
not launch the full publication sweep locally unless you are intentionally
acting as the server operator.

```bash
# Smoke pipeline (~5 min on a laptop, no GPU):
# Stage 0 writes _smoke_gate; the audited 10-method sweep and report write _smoke.
python scripts/run_pipeline.py --profile smoke

# Minimal Stage-0 runner gate only:
python scripts/run_pipeline.py --profile smoke --only smoke

# Server/operator mode only: full run with an isolated smoke preflight.
python scripts/run_pipeline.py --profile full --with-smoke --confirm-full-run
```

The smoke profile recomputes its tiny 40-cell `_smoke/results.csv` by default
instead of resuming from an older local CSV, so smoke evidence stays tied to
the current code and method fingerprints.  The readiness roll-up re-runs
`audit_sweep_integrity.py` against that CSV using the current smoke profile's
fixed 1-case, 10-method, 2-budget, 2-repeat protocol, so a self-consistent
subset is not enough.  The publication-scale `full` profile remains resumable
by default.

The full run belongs on the server:

```bash
# Linux server only; resource requirements are listed above
bash scripts/run_pipeline.sh
```

Before the server launches a long sweep, run the all-case experiment-readiness
audit on the current code:

```bash
python scripts/audit_case_experiment_readiness.py \
    --output docs/case_experiment_readiness.csv \
    --fail-on-warning
```

This writes a per-case CSV covering batch evaluate shape, finite outputs,
repeat-evaluate determinism on the same design, deterministic physical envelope
probes against each output `expected_range`, failure-threshold metadata, and
smoke timing warnings.  It is intentionally lighter than the full runner smoke
gate: it checks all registered cases with a small lower/upper/centre/random
design plus boundary envelope probes.

The orchestrator verifies expected outputs between stages and writes a
UTC-stamped log to
`logs/pipeline_<YYYYMMDD_HHMMSSZ>.log`.  Restart with the same command
on failure — every stage is resumable via `--skip-existing` when the partial
`results.csv` was produced by the same case/method/budget/repeat grid, seed
and test-set protocol, and current case/method source fingerprints.
The smoke profile runs the isolated Stage 0 path; the formal full-profile
server entry point runs Stage 1-10 by default and includes Stage 0 only when
`--with-smoke` is passed.

The remainder of this document expands each stage in case you want to
run them individually.

---

## 2. Stage 0 — Smoke gate (~5 min)

Sanity-check the runner / surrogate / metric stack on one case before
burning 30+ hours on the publication sweep.

```bash
python benchmark/scripts/run_full_benchmark.py \
    --cases cantilever_beam_5d \
    --pool default \
    --budgets 5 10 \
    --n-repeats 2 \
    --n-test 100 \
    --seed-strategy case_repeat --seed-start 1000 \
    --output-dir benchmark/results/_smoke \
    --quiet
```

**Go condition**: `_smoke/results.csv` contains the current smoke profile's 1
case × 10 methods × 2 budgets × 2 repeats = 40 rows, all `status="ok"`, no NaN
in `nrmse`, `_smoke/sweep_integrity_report.md` reports PASS, the readiness
roll-up's fresh smoke audit also passes, and Stage 3 can regenerate
`_smoke/report/REPORT.md` plus figure PNGs.

`run_pipeline.py --profile smoke` keeps the Stage 0 gate isolated in
`benchmark/results/_smoke_gate` and writes the audited smoke sweep to
`benchmark/results/_smoke`; the audited smoke sweep is recomputed by default
rather than resumed.
`run_pipeline.py --profile full --with-smoke --confirm-full-run`
also keeps Stage 0 out of `benchmark/results/full_benchmark`.  Use
`--smoke-dir <path>` only when you explicitly want a different Stage 0
gate-output location.

For smoke-profile runs, downstream D1/D2/D3, figdata, figures, and paper-number
artifacts are also scoped under the smoke sweep directory by default.  This
keeps exploratory smoke outputs from overwriting `paper/` or the publication
analysis directories.

---

## 3. Stage 1 — Pre-registration and case-readiness gates

```bash
python scripts/validate_run_config.py \
    --pool nature \
    --budgets 5 10 20 50 \
    --n-repeats 30 \
    --n-test 1000 \
    --seed-strategy case_repeat \
    --seed-start 1000 \
    --cases ALL
```

**Go condition**: exit 0 with `Result: COMPLIANT`.  On any deviation the
script exits 1 unless `--allow-deviation` is given (which logs to
`benchmark/results/full_benchmark/preregistration_deviations.txt`).

The one-click pipeline also runs the all-case experiment-readiness audit during
Stage 1:

```bash
python scripts/audit_case_experiment_readiness.py \
    --output docs/case_experiment_readiness.csv \
    --fail-on-warning
```

**Go condition**: the CSV case IDs exactly match the current 34-case registry,
every row has `status="ok"`, and
`n_nan = n_inf = input_mutation_n_changed = determinism_n_mismatch =
n_expected_range_violations = 0` for the formal full-pipeline preflight.  The
same CSV must also carry the
classification metadata used by D1/D2/D3 (`response_topology`, `pde_type`,
`governing_equation`, `n_topology_tags`, `n_physics_references`,
failure-boundary, and multi-fidelity fields).  The publication-readiness roll-up
requires those fields to be present and non-empty, with at least one topology
tag/reference per case, so an old runtime-only readiness ledger cannot mask
case-descriptor drift.  Standalone diagnostic runs may omit
`--fail-on-warning` to inspect `timing_warning` rows, but the one-click full
pipeline promotes timing warnings to failures so metadata timing drift is
reviewed before launching the full sweep.  Any `error`, `nonfinite`,
`input_mutation`, `nondeterministic`, metadata warning, expected-range drift, or
unreviewed timing warning should be fixed before launch.

For the formal publication run, Stage 1b also executes
`python scripts/check_optional_dependencies.py --pool nature --require-botorch --expected-methods 20 --require-formal-case-backends --require-runtime-monitoring`.
Missing or incompatible BoTorch / GPyTorch / Torch, or any resolved
`nature_pool()` cardinality other than 20, is therefore a launch blocker for the
headline sweep.  Standalone diagnostics can still inspect the 18-method
non-BoTorch fallback only when they do not pass the formal publication flags.

---

## 4. Stage 2 — Main 81 600-cell sweep

| Host class | Command | Wall clock (P50) | Memory cap |
|---|---|---|---|
| Linux ≥ 32 c / ≥ 64 GB | `bash scripts/run_pipeline.sh` | ≈ 21 h for Stage 2 plus downstream gates | default 56 GB; override explicitly with `MEM_CAP_GB=...` |
| Linux 16-32 c | `bash scripts/run_pipeline.sh` | ≈ 30-50 h for Stage 2 plus downstream gates | choose an explicit `MEM_CAP_GB` below the job limit |
| Windows workstation | no formal full sweep; use smoke/preflight only | n/a | n/a |

For long server sessions, wrap the formal pipeline in `tmux`, `screen`, or
`nohup`, for example `tmux new -s doe 'bash scripts/run_pipeline.sh'`.
For larger or smaller allocations, tune the formal wrapper with positive
environment variables such as
`N_JOBS=24 BO_N_JOBS=4 MEM_CAP_GB=96 bash scripts/run_pipeline.sh`; explicit
CLI flags still take precedence and the resolved runtime controls are recorded
in command history and `run_config.json`.  BLAS/OpenMP thread variables are
not tunables in the formal wrapper: inherited `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
`BLIS_NUM_THREADS` values are overridden to `1` before Python starts.
`scripts/launch_sweep_remote.sh` remains available for a reviewed low-level
Stage 2 restart and auto-tunes its concurrency from effective RAM/cgroup
limits, but it is not the preferred way to produce the complete paper archive
because it does not run the Stage 2b-10 gates by itself.

The runner streams `results.csv` row by row and is fully resumable; an
OOM kill or wall-clock timeout is recovered by re-running the same
formal pipeline command, or by using the low-level Stage 2 launcher when the
operator intentionally wants to resume only the sweep.
Each streamed append is flushed and fsynced, and schema-evolution rewrites use
a fsynced temporary file followed by atomic replacement.  Resume still treats
parse errors, truncated rows, and corrupt schemas as blocking failures.

Resume is intentionally strict.  With `--skip-existing`, the existing
`results.csv` must be a subset of the same declared case/method/budget/repeat
grid and must retain the current row schema, case/method source fingerprints,
test protocol, and integer grid IDs.  Malformed values such as
`budget_mult = 5.5` or `repeat = 0.5` are rejected before any cell is skipped,
so a partial CSV cannot silently contaminate a resumed paper run.  The same
pre-skip gate rejects corrupt completed rows: unknown/missing status values,
missing or inconsistent `failure_category` values, `ok` rows with non-finite
or negative `nrmse`, non-finite or negative `t_total_s`, timeout/failed rows
with populated `nrmse`/error metrics, timeout/failed rows without an error
reason, and `ok` rows that still carry an error message.

The runtime health monitor applies the same fail-closed posture while the
server run is active.  `--allow-missing` only tolerates the startup period
before rows exist; once `results.csv` has data rows, missing resume-critical
columns such as source fingerprints, case metadata, `method_family`, `n_train`,
`n_test`, seeds, timing, `status`, `failure_category`, error, or `nrmse` are
health failures rather than warnings.  Timeout/failed rows with populated
`nrmse`, `rmse`, `mae`, `maxae`, `q95ae`, or `r2` are also health failures;
the raw archive records the failure reason, while preregistered downstream
analyses apply any failure-denominator penalties.

**Go condition** (auto-checked by Stage 3 and the orchestrator):

```python
import pandas as pd
df = pd.read_csv('benchmark/results/full_benchmark/results.csv')
assert df['case_id'].nunique() == 34          # all cases
assert df['method'].nunique() == 20           # nature_pool
assert sorted(df['budget_mult'].unique()) == [5, 10, 20, 50]
assert df.groupby(['case_id','method','budget_mult'])['repeat'].nunique().min() >= 28
assert (df['status']=='ok').mean() >= 0.95
assert df['nrmse'].isna().sum() == 0          # for status==ok rows
```

If any of these fail the orchestrator refuses to launch Stage 3 and
emits an actionable error.

The Stage 2 `sweep_audit` writes
`benchmark/results/full_benchmark/sweep_integrity_report.md` via a temporary
sibling directory and atomic replacement.  If report replacement fails, the
previous complete report is preserved rather than leaving a partial audit
artifact for Stage 3-10.

---

## 5. Stage 3 — Headline report (~5-10 min)

```bash
python benchmark/scripts/generate_report.py \
    benchmark/results/full_benchmark
```

Writes `report/REPORT.md` and `report/figs/*.png` under the sweep directory.  Per
`docs/working-notes/DOE-Benchmark实验设计方案.md` §Stage 3, the report **does not** emit any
CSV — every downstream CSV (stratified ranks, Pareto, etc.) is recomputed
on demand by Stage 8 `prepare_figdata.py` from the same `results.csv`.

---

## 6. Stage 4 — D1 ranking-reversal + bootstrap CI

```bash
# 4a. analytical + physical phases × three pairings
python benchmark/scripts/run_ranking_reversal.py \
    --pool nature --pairing-mode all \
    --n-repeats 30 --budgets 10 20 \
    --n-test 1000 --seed-strategy case_repeat --seed-start 1000 \
    --output-dir benchmark/results/ranking_reversal

# 4b. augment correlation.json with bootstrap 95% CI
python -m benchmark.scripts.recompute_spearman_ci \
    --rr-dir benchmark/results/ranking_reversal --n-boot 10000
```

Outputs:

```
benchmark/results/ranking_reversal/
├── analytical_results.csv          (shared across pairings)
├── pairing_dimension/
│   ├── physical_results.csv
│   └── rank_comparison.csv         (with pairing_mode column)
├── pairing_topology/
├── pairing_random/
└── correlation.json                (6 records: 3 pairings × 2 budgets;
                                     each carries spearman_rho_ci_lo/hi
                                     and kendall_tau_ci_lo/hi)
```

> **D1 falsification window**: see SAP §2.1.  If ρ at 10·d does not lie
> below 0.70 under both `dimension` and `topology`, the headline must be
> rewritten as falsified per `paper/preregistration_decisions.md`.

---

## 7. Stage 5 — D2 sample-complexity

```bash
python benchmark/scripts/run_sample_complexity.py \
    benchmark/results/full_benchmark/results.csv \
    --metric nrmse --threshold 0.20 --rule-budget 10 \
    --output-dir benchmark/results/sample_complexity_0.20
```

**Required columns** in `rule_summary.csv` (GAP-11):
`topology, methods_total, methods_pass_at_10d, frac_pass_at_10d, median_n_star, max_n_star, n_methods_never_pass, n_cases, case_ids, exploratory, n_failed_cells_per_topology`.

The CLI fails before creating the output directory if `results.csv` is missing,
`--threshold <= 0`, `--rule-budget <= 0`, or `--metric` is blank.

> Topologies with `n_cases < 3` are flagged `exploratory=True` and **not**
> counted in the headline numerator/denominator; they are reported
> separately in SI Table E.5.

---

## 8. Stage 6 — D3 reliability-recovery

```bash
python benchmark/scripts/run_reliability_recovery.py \
    benchmark/results/full_benchmark/results.csv \
    --budget-mult 10 --output-aggregate max --cluster-by case_id \
    --output-dir benchmark/results/reliability_recovery
```

Cluster-aware bootstrap is the **primary** CI estimator per pre-registration §1.3.
The i.i.d. variant is computed in parallel for sensitivity (Stage 8 figdata picks both up).

The output `reliability_per_method.csv` carries `p_holm` already
Holm-corrected; downstream scripts read this column directly without
recomputing.

The CLI fails before creating the output directory if `results.csv` is missing
or `--budget-mult` is non-positive.

---

## 9. Stage 7 — Post-registration robustness (optional, Linux only)

```bash
# A1. SOTA pool (TURBO + D-optimal); P50 ≈ 2 h with REUSE_HEADLINE=1
bash scripts/launch_sota_robustness.sh tmux

# A2. Multi-fidelity supplement; P50 ≈ 12 h
bash scripts/launch_mfbo.sh tmux
```

These do **not** alter D1/D2/D3 headline numbers.  They produce
Extended Data Table E.1 / Fig E.2 only.

---

## 10. Stage 8 — Figure data package

```bash
python scripts/prepare_figdata.py \
    --results-csv benchmark/results/full_benchmark/results.csv \
    --d1-dir benchmark/results/ranking_reversal \
    --d2-dir benchmark/results/sample_complexity_0.20 \
    --d3-dir benchmark/results/reliability_recovery \
    --out-dir paper/figdata
```

Emits one CSV per main-text figure:

| Figure | CSV(s) |
|---:|---|
| Fig 1 | `fig1_case_catalog.csv`, `fig1_method_catalog.csv` |
| Fig 2 | `fig2_d1_spearman.csv`, `fig2_d1_ranks.csv` |
| Fig 3 | `fig3_d2_learning.csv`, `fig3_d2_rule.csv` |
| Fig 4 | `fig4_d3_bias.csv` |
| Fig 5 | `fig5_mechanism.csv` |
| Fig 6 | `fig6_pareto.csv` |

Each CSV carries `_source` and `_generated_at` columns so reviewers can
trace any number to a raw input.

Before writing any figdata, `prepare_figdata.py` now performs a local
`results.csv` contract check: required columns must be present, `status` must be
one of `ok`, `timeout`, or `failed`, at least one `ok` row must exist, every
`ok` row must have finite non-negative `nrmse` and blank `error`, every
`timeout` or `failed` row must carry a non-empty error reason and leave
`nrmse`, `rmse`, `mae`, `maxae`, `q95ae`, and `r2` blank, every
`failure_category` must match the canonical status/error taxonomy, all rows
must have a finite non-negative timing value, and `ok` rows must have positive
timing.  `budget_mult` and `repeat` must be finite integer grid IDs, and
duplicate `(case_id, method, budget_mult, repeat)` rows are rejected before any
figure CSV is written.  Grid completeness remains the responsibility of the
Stage 2 `sweep_audit`, but Stage 8 will fail fast if someone tries to build
paper artifacts from a malformed or hand-edited CSV.

---

## 11. Stage 9 — Nature-format figures

```bash
python scripts/plots_nature.py \
    --figdata paper/figdata \
    --out-dir paper/figures
```

Outputs:

```
paper/figures/
├── pdf/fig{1..6}.pdf                    (300 dpi, embedded fonts, ≤ 183 mm width)
├── png/fig{1..6}.png                    (300 dpi rasterisation)
└── ../figure_manifest.json              (per-figure SHA-256 of pdf + png + every figdata input)
```

`figure_manifest.json` is the audit artefact: it pins each PDF to the
SHA-256 of every CSV that produced it.  The upstream
`paper/figdata/figdata_manifest.json` is checked separately by the submission
gate and must hash-pin `results.csv`, D1 `correlation.json`, every required D1
`pairing_*/rank_comparison.csv`, and the D2/D3 source CSVs used to build the
figdata layer.

When all six main figures are requested (the default Stage 9 path),
`plots_nature.py` renders into a temporary sibling directory and replaces
`paper/figures` plus `paper/figure_manifest.json` only after every PDF/PNG is
present.  Explicit partial runs such as `--fig 1` remain direct writes for
interactive re-rendering and print a warning.

The figure script uses Wong's CVD-safe categorical palette and matplotlib's
`cividis_r` sequential map for heatmaps.  The palette contract is pinned by
`benchmark/tests/test_paper_pipeline.py::TestPlotsNatureColorblindPalette`,
which rejects unsafe categorical/diverging maps such as `tab10`, `tab20`,
`RdYlBu`, and `coolwarm`, and also rejects legacy hard-coded blues outside the
Wong-derived constants.  The final `check_submission_ready.py` gate repeats the
same palette contract through `check_figure_palette_contract()`, so palette
drift is caught even if the pytest suite is skipped during a manual freeze.

---

## 12. Stage 10 — Reproducibility package

```bash
# 10a. Paper-ready macros from results.csv + D1/D2/D3 outputs
python scripts/fill_paper_numbers.py \
    --results-csv benchmark/results/full_benchmark/results.csv \
    --rr-dir      benchmark/results/ranking_reversal \
    --d2-dir      benchmark/results/sample_complexity_0.20 \
    --d3-dir      benchmark/results/reliability_recovery \
    --figdata-dir paper/figdata \
    --test-protocol-dir benchmark/results/full_benchmark/analysis \
    --output-tex  paper/numbers.tex \
    --output-json paper/numbers.json \
    --require-results-csv \
    --require-results-contract \
    --require-submission-complete

# 10a.5. Baseline supplementary tables and multiplicity-policy SI note
python scripts/generate_si.py \
    --output-dir paper

# 10a.6. Validation-note/reference fingerprints
python scripts/update_case_validation_note_sources.py \
    --write \
    --output docs/case_validation_note_sources.csv

# 10a.7. Case-validation audit summary from the ledger + review folders
python scripts/generate_case_validation_summary.py \
    --note-sources-csv docs/case_validation_note_sources.csv \
    --output docs/CASE_VALIDATION_SUMMARY.md \
    --strict

# 10b. Immutable run manifest (git commit, env, scheduler/GPU/container
#      provenance, pip-freeze hash, sha256 of every artefact)
# The one-click pipeline writes logs/pipeline_command_history.txt automatically.
# Dry-runs write logs/pipeline_command_history.dry_run.txt instead, so an
# inspection run cannot overwrite the command history frozen into a real manifest.
# The history includes a `pipeline_invocation` row plus every resolved stage
# command, so resumed/downstream-only freezes remain distinguishable.
# Strict manifests record `experiment.command_files_requested` as portable
# repo-relative paths, require `experiment.command_files_missing = []`, and
# reject `.dry_run` history files.
# They also keep labeled `experiment.command_entries`; one entry must be the
# formal full-pipeline `pipeline_invocation` line from the command-history
# file, not a low-level `run_full_benchmark.py` command.  Existing command
# files must use `label<TAB>command` lines; malformed lines are recorded in
# `experiment.command_file_parse_errors` and rejected by strict archive mode.
# The publication-readiness and final submission gates also reject non-empty
# `command_file_parse_errors`, so a malformed history cannot reach paper sign-off.
# For `--cases ALL`, the manifest must resolve the current case registry into
# concrete `case_ids_declared` and keep `experiment.case_resolution_error = null`.
# If running Stage 10b manually, pass the same command-history file (or another
# reviewed command file) so strict archive mode can prove the reproduction path.
python scripts/generate_run_manifest.py \
    --results-dir benchmark/results/full_benchmark \
    --pool nature --budgets 5 10 20 50 \
    --n-repeats 30 --n-test 1000 \
    --seed-strategy case_repeat --seed-start 1000 \
    --command "python scripts/run_pipeline.py --profile full --confirm-full-run" \
    --command-file logs/pipeline_command_history.txt \
    --require-complete-archive \
    --output benchmark/results/full_benchmark/RUN_MANIFEST.json
```

With the formal Stage 10 flags, `fill_paper_numbers.py` fails before writing if
`results.csv` is missing, if the row-level result contract used by
`prepare_figdata.py` fails, or if any submission-critical macro would still be
`N/A`.  When it does write, `paper/numbers.tex` and `paper/numbers.json` are
first materialized in a temporary directory and then replaced as a pair, so a
partial write cannot leave the manuscript macros out of sync with the JSON
audit file.  The submission and return-package gates also reject duplicate
`\newcommand` definitions in `paper/numbers.tex`, because a manually merged
macro file can otherwise contain both stale and fresh definitions even when
the final parsed value happens to match `paper/numbers.json`.
The JSON file also carries a `_provenance` block with path/size/SHA-256 entries
for the restored `results.csv`, D1 `correlation.json`, D2 `rule_summary.csv`,
the D3 bias source used for paper numbers, and `analysis/test_set_sufficiency.csv`;
the final submission gate rejects missing or hash-drifted number inputs.

Stage 10 first runs `benchmark/scripts/validate_test_protocol.py` against the
completed sweep archive, materializing `analysis/test_set_sufficiency.csv` and
the cross-metric / robustness diagnostics that support the manuscript's
test-set-noise macros and Extended Data discussion.

The final submission gate also checks the decision-level D1/D2/D3 analysis
artifacts before the manuscript can be marked submission-ready:
`ranking_reversal/correlation.json` must include all three pairing modes with
Spearman/Kendall bootstrap CIs, each `pairing_*/rank_comparison.csv` must carry
the primary complete-case rank table, `sample_complexity_0.20/rule_summary.csv`
and `learning_curves.csv` must carry the primary D2 denominator policy, and the
D3 cluster/i.i.d. reliability tables plus `reliability_ci_sensitivity.csv` must
exist with the expected schema.  The same gate requires the test-set MC-noise
CSV (`analysis/test_set_sufficiency.csv`) with `n_test=1000`, so supporting
Extended Data evidence cannot be omitted while headline macros pass.

`generate_si.py` uses the same package discipline for the ten generated SI
artifacts (`si_hyperparameters.*`, `si_sobol.*`, `si_cases.*`,
`si_multiple_testing.*`, and `si_d2_policy.*`): it builds them in a temporary
directory and replaces the submitted `.tex`/`.md` files only after the full set
is present and non-empty.  The draft and final submission gates also re-check
that the `si_cases.*` evidence-tier rows match `docs/case_validation_status.csv`,
so a stale SI table cannot silently contradict the case-validation ledger.

`generate_case_validation_summary.py --strict` fails before writing if a ledger
row points at a missing validation-note folder, a note folder lacks a real
`.docx` or reference attachment, a targeted physics test path is missing, or a
real `案例正确性校验/**/*.docx` folder is not referenced by the ledger.  It also
checks `docs/case_validation_note_sources.csv`, which fingerprints every real
validation note/reference attachment by path, byte size, and SHA-256; refresh it
with `python scripts/update_case_validation_note_sources.py --write` after a
deliberate review-material edit.  The summary is also written through a
temporary file so a strict failure preserves the previous audit document.  The
final submission gate reuses these strict evidence-chain checks before
accepting `docs/CASE_VALIDATION_SUMMARY.md`.

`generate_run_manifest.py --require-complete-archive` validates the archive
contract before writing and then materializes `RUN_MANIFEST.json` through a
temporary file.  A strict-check, serialization, or filesystem failure therefore
preserves the previous manifest rather than exposing partial JSON.  Because
the manifest freezes `results_summary` row-contract counters, non-zero invalid
grid values, missing `case_id`/`method`/`budget_mult`/`repeat` protocol keys,
invalid statuses, bad successful-row metrics, missing failure reasons, or
invalid timings block strict archive generation before any JSON is replaced.
It also records `python_environment.command`,
`python_environment.package_count`, and
`python_environment.pip_freeze_sha256` from `python -m pip freeze --all`.
These fields describe the environment in which the manifest is generated.  In
the canonical returned archive, that capture occurred during post-run
packaging, so `run_config.json` remains the authority for the formal startup
Python and core-package versions, thread controls and seed policy.  The
post-run dependency listing must not be described as a complete lock of the
formal Linux run host.
It also records `scheduler`, `accelerator`, and `container` sections: common
Slurm/PBS/LSF/SGE allocation variables, `nvidia-smi` output when available,
Torch CUDA availability/device metadata, `CUDA_VISIBLE_DEVICES`, and
Apptainer/Singularity/Docker/container marker variables.  These sections are
present even when no scheduler, GPU, or container is detected, so absence is
itself provenance.  Strict archive and submission gates reject a formal
manifest that omits or malforms these sections.
Stage 8-10 rewrites tracked derived paper/readiness files during a formal
freeze, `docs/CASE_VALIDATION_SUMMARY.md`,
`docs/case_validation_note_sources.csv`, `paper/numbers.*`,
`paper/figure_manifest.json`, `paper/figdata/`, and `paper/figures/` are treated
as derived dirty outputs; source/config edits remain blockers.

The manifest is what you upload alongside the paper PDF to Zenodo /
the journal's data repository.

---

## 13. Verification commands

Before the full sweep has finished, the manuscript package should pass the
draft gate and fail the submission gate.  The draft gate proves the Nature-family
paper skeleton is complete enough to keep writing while preserving
`RESULT_PENDING` markers for unfinished claims:

```bash
python scripts/check_paper_draft_ready.py paper/
```

For a one-file handoff of the current case, sweep, draft, and submission-gate
state, run:

```bash
python scripts/audit_publication_readiness.py
```

After the server archive is restored, first validate the returned package:

```bash
python scripts/check_server_return_package.py
python scripts/audit_publication_readiness.py
```

If `logs/pipeline_command_history.txt` is staged outside the default `logs/`
directory, pass the same directory to both commands with `--logs-dir`.

Before the server archive is restored, this roll-up is expected to return
non-zero with `Full sweep results = BLOCKED`; use it as a handoff report rather
than a local full-sweep trigger.  The Markdown roll-up is written through a
temporary sibling directory and atomic replacement, so a filesystem failure
preserves the previous complete readiness report.  After the server archive is
restored, `Full sweep results` becomes `PASS` only when the full-row
`results.csv` exists and `sweep_integrity_report.md` contains the explicit PASS
verdict, and a non-empty `RUN_MANIFEST.json` is present for the archive.
The roll-up also parses the manifest and requires the core `experiment`,
`results_summary`, and `artifacts.files` sections before calling the full sweep
ready for downstream paper stages.  The manifest must say
`results_summary.exists = true`, its `results_summary.n_rows` must match the
local `results.csv` row count, and `artifacts.files` must include `results.csv`.
The manifest `results_summary` observed case/method/budget/repeat sets must
close against the declared protocol, with observed case/method counts matching
their unique observed sets.
When the `results.csv` artifact entry carries `size_bytes` or `sha256`, those
values must also match the local file.  The roll-up also re-runs
`scripts/audit_sweep_integrity.py` against the restored local `results.csv`
using the preregistered protocol, so an archived or hand-edited PASS report is
not sufficient by itself.  `artifacts.files` must also include
`sweep_integrity_report.md`, contain only object entries with non-empty paths,
and not duplicate paths; recorded `sweep_integrity_report.md` `size_bytes` or
`sha256` must match the local file.  `experiment.commands` must include a
non-empty command history and `manifest_command` must record the manifest
generation command so the archive remains reproducible.  The manifest must also
record non-empty portable `experiment.command_files_requested` entries, an empty
`experiment.command_files_missing` list, and no `.dry_run` command-history
files.  Its labeled `experiment.command_entries` must include
`pipeline_invocation`.  It must also contain well-formed `scheduler`,
`accelerator`, and `container` provenance sections so returned server results
can be audited against the actual scheduler allocation, CUDA/GPU state, and
container state used during the run.  Once the full sweep passes, the roll-up also includes a
`Server return package` gate that reuses `check_server_return_package.py` to
verify returned command history, root metadata (`.zenodo.json`,
`CITATION.cff`, `codemeta.json`, `croissant-metadata.json`, `ro-crate-metadata.json`), D1/D2/D3 artifacts, figdata,
figures, and paper numbers before final manuscript sign-off.  A manifest generated
from dirty source/config entries is not accepted by the roll-up; legacy dirty
manifests without the `source_dirty` split are also blocked.  `git.commit` must
be present to bind the archive to a code revision.  The same roll-up checks the
headline preregistered protocol recorded in `experiment`: `pool = nature`,
`budgets = [5, 10, 20, 50]`, `n_repeats = 30`, `n_test = 1000`,
`seed_strategy = case_repeat`, `seed_start = 1000`, and 34 declared cases whose
`case_ids_declared` exactly match the current registered case library.
`n_methods_declared` must be 20 and `methods_declared` must exactly match the
current registered `nature` method pool.  The manifest `results_summary`
observed case/method/budget/repeat sets must be present as lists and close
against those declarations, with no duplicate observed case or method IDs and
with observed counts matching the unique observed sets,
`status_counts` must be present as an object, use only `ok`, `timeout`, or
`failed` statuses, carry non-negative integer counts, and sum to `n_rows`;
`failure_category_counts` must be present as an object, use only the canonical
categories `ok`, `timeout`, `oom`, `dependency`, `backend`, `numerical`,
`worker_crash`, `scheduler`, `storage`, and `exception`, and sum to `n_rows`;
invalid numeric `budget_mult` or `repeat` values are blocking.
`case_sources` and `method_sources` fingerprints must be present and match the
current case library and method registry source hashes.

After Stage 10 finishes, run these audits before considering the
reproduction complete:

```bash
# All paper macros are emitted and non-default
python scripts/audit_macros.py

# Final submission-ready check. This should fail before the final results
# archive and signed preregistration_decisions.md exist.
# When signing that ledger, fill Date signed (UTC+08:00) as an ISO-style
# timestamp with an explicit +08:00 offset, e.g. 2026-06-03T10:00:00+08:00.
python scripts/check_submission_ready.py paper/ \
    --case-experiment-readiness docs/case_experiment_readiness.csv \
    --d1-dir benchmark/results/ranking_reversal \
    --d2-dir benchmark/results/sample_complexity_0.20 \
    --d3-dir benchmark/results/reliability_recovery \
    --test-protocol-dir benchmark/results/full_benchmark/analysis

# Tests still pass with the new data
python -m pytest benchmark/tests -x -q

# Pipeline orchestrator dry-run (sanity)
python scripts/run_pipeline.py --dry-run
```

---

## 14. Determinism guarantees

The reproduction is deterministic up to the following components:

| Layer | Determinism |
|---|---|
| Sampler seed → design points | exact |
| Surrogate fit on identical training set | exact (sklearn) |
| GP / BoTorch with `torch.use_deterministic_algorithms(True)` | exact (CPU) |
| Bootstrap CI | exact given numpy seed |
| Friedman χ², Wilcoxon, Spearman ρ | exact (scipy) |
| Wall-clock timings | non-deterministic (excluded from contract) |

Any departure from bit-exactness on `nrmse`, `pf_pred_*`, `f_bal_acc_*`,
`spearman_rho`, `signed_bias_mean`, `p_holm` is a **regression** and
must be fixed before merging.

`benchmark/results/determinism_audit/` archives a small reference run
that the CI replays on every commit.

---

## 15. Known reproduction caveats

| Caveat | Effect | Mitigation |
|---|---|---|
| BoTorch optional dep absent | diagnostic `nature_pool()` checks may degrade to 18 methods | the formal `--profile full` pipeline passes `--require-botorch`, so the 20-method publication run fails fast instead of silently degrading |
| Formal case backend absent | cases such as `airfoil_stall_2d` could otherwise use a different diagnostic backend or fail mid-sweep | the formal pipeline passes `--require-formal-case-backends`, and `airfoil_stall_2d.evaluate` refuses analytical fallback |
| Runtime monitoring absent | RSS traces and memory soft caps would be missing, weakening OOM diagnosis and server-return evidence | the formal pipeline passes `--require-runtime-monitoring`, which currently requires `psutil` before dispatch |
| MKL / OpenBLAS thread oversubscription | up to 30 % slowdown, no numerical impact | `run_pipeline.sh` and the Python orchestrator override BLAS/OpenMP caps to 1 before spawning; the workstation `.ps1` wrapper pins the same caps for smoke/preflight use |
| Cantera missing on Windows | `laminar_flame_speed_5d` falls back to surrogate | declared in case metadata; not affecting D1/D2/D3 headline |
| SAASBO at 50·d on 16 GB | OOM risk | use `--bo-n-jobs 1`; or run on Linux ≥ 64 GB |
| `topology` tag drift between case-library versions | rule_summary.csv changes | locked by the per-case metadata checks in `tests/test_cases.py` |

---

## 16. Cross-references

- [`STATISTICAL_ANALYSIS_PLAN.md`](STATISTICAL_ANALYSIS_PLAN.md) — pre-specified hypotheses, decision rules, sensitivities.
- `../paper/preregistration.md` — frozen v0.4.0-prereg protocol.
- `../paper/preregistration_amendments.md` — A1/A2/A3 post-freeze amendments.
